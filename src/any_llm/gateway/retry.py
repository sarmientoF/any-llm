"""Retry with exponential backoff and model fallback."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from any_llm import AnyLLM, acompletion
from any_llm.gateway.log_config import logger

if TYPE_CHECKING:
    from any_llm.gateway.circuit_breaker import CircuitBreaker
    from any_llm.gateway.config import GatewayConfig
    from any_llm.types.completion import ChatCompletion


async def call_with_retry_and_fallback(
    completion_kwargs: dict[str, Any],
    model: str,
    config: GatewayConfig,
    circuit_breaker: CircuitBreaker | None = None,
) -> tuple[ChatCompletion, str]:
    """Try primary model with retries, then fall through to fallbacks.

    Returns:
        (response, model_used) — the model string that actually served the request.
    """
    retry_cfg = config.retry
    models_to_try = [model, *config.fallbacks.get(model, [])]

    last_exc: Exception | None = None
    for candidate in models_to_try:
        provider_enum, bare_model = AnyLLM.split_model_provider(candidate)
        provider_str = provider_enum.value

        # Skip if circuit breaker says provider is down
        if circuit_breaker and not circuit_breaker.is_available(provider_str):
            logger.warning(f"Circuit breaker open for {provider_str}, skipping {candidate}")
            continue

        for attempt in range(1, retry_cfg.max_retries + 1):
            try:
                kwargs = {**completion_kwargs, "model": candidate}
                response: ChatCompletion = await acompletion(**kwargs)  # type: ignore[assignment]
            except Exception as e:
                last_exc = e
                if circuit_breaker:
                    circuit_breaker.record_failure(provider_str)
                if attempt < retry_cfg.max_retries:
                    delay = min(retry_cfg.base_delay * (2 ** (attempt - 1)), retry_cfg.max_delay)
                    logger.warning(f"Retry {attempt}/{retry_cfg.max_retries} for {candidate}: {e}")
                    await asyncio.sleep(delay)
            else:
                if circuit_breaker:
                    circuit_breaker.record_success(provider_str)
                return response, candidate

        logger.warning(f"All retries exhausted for {candidate}")

    raise last_exc or RuntimeError("No models available")
