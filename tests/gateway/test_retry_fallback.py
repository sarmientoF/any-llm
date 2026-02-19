"""Unit tests for retry with fallback logic."""

from unittest.mock import AsyncMock, patch

import pytest

from any_llm.gateway.circuit_breaker import CircuitBreaker
from any_llm.gateway.config import GatewayConfig, RetryConfig
from any_llm.gateway.retry import call_with_retry_and_fallback
from any_llm.types.completion import ChatCompletion, CompletionUsage


def _make_response(model: str = "test-model") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1234567890,
        model=model,
        choices=[],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _make_config(**overrides) -> GatewayConfig:
    defaults = {
        "database_url": "postgresql://localhost/test",
        "master_key": "test",
        "auto_migrate": False,
        "retry": RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05),
        "fallbacks": {},
        "providers": {"openai": {"api_key": "test"}, "anthropic": {"api_key": "test"}},
    }
    defaults.update(overrides)
    return GatewayConfig(**defaults)


@pytest.mark.asyncio
async def test_success_on_first_try():
    mock_response = _make_response()
    config = _make_config()

    with patch("any_llm.gateway.retry.acompletion", new_callable=AsyncMock, return_value=mock_response):
        response, model_used = await call_with_retry_and_fallback(
            completion_kwargs={"messages": []},
            model="openai/gpt-4o",
            config=config,
        )
    assert response == mock_response
    assert model_used == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_retry_then_succeed():
    mock_response = _make_response()
    call_count = 0

    async def flaky_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("transient error")
        return mock_response

    config = _make_config()

    with patch("any_llm.gateway.retry.acompletion", side_effect=flaky_acompletion):
        response, model_used = await call_with_retry_and_fallback(
            completion_kwargs={"messages": []},
            model="openai/gpt-4o",
            config=config,
        )
    assert response == mock_response
    assert call_count == 2


@pytest.mark.asyncio
async def test_fallback_after_retries_exhausted():
    mock_response = _make_response("claude-3-sonnet")

    async def fail_openai(**kwargs):
        if "openai" in kwargs.get("model", ""):
            raise RuntimeError("openai down")
        return mock_response

    config = _make_config(fallbacks={"openai/gpt-4o": ["anthropic/claude-3-sonnet"]})

    with patch("any_llm.gateway.retry.acompletion", side_effect=fail_openai):
        response, model_used = await call_with_retry_and_fallback(
            completion_kwargs={"messages": []},
            model="openai/gpt-4o",
            config=config,
        )
    assert model_used == "anthropic/claude-3-sonnet"


@pytest.mark.asyncio
async def test_all_models_fail():
    config = _make_config(fallbacks={"openai/gpt-4o": ["anthropic/claude-3-sonnet"]})

    with patch("any_llm.gateway.retry.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("all down")):
        with pytest.raises(RuntimeError, match="all down"):
            await call_with_retry_and_fallback(
                completion_kwargs={"messages": []},
                model="openai/gpt-4o",
                config=config,
            )


@pytest.mark.asyncio
async def test_circuit_breaker_skips_open_provider():
    mock_response = _make_response()
    cb = CircuitBreaker(failure_threshold=1, cooldown_sec=9999)
    cb.record_failure("openai")  # opens circuit

    config = _make_config(fallbacks={"openai/gpt-4o": ["anthropic/claude-3-sonnet"]})

    with patch("any_llm.gateway.retry.acompletion", new_callable=AsyncMock, return_value=mock_response):
        response, model_used = await call_with_retry_and_fallback(
            completion_kwargs={"messages": []},
            model="openai/gpt-4o",
            config=config,
            circuit_breaker=cb,
        )
    assert model_used == "anthropic/claude-3-sonnet"
