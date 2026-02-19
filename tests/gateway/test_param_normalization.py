"""Unit tests for table-driven parameter normalization."""

from any_llm.gateway.param_rules import PROVIDER_PARAM_RULES, normalize_params


def test_openai_o1_max_tokens_renamed():
    kwargs = {"max_tokens": 100, "temperature": 0.7}
    normalize_params(kwargs, "o1-mini", "openai")
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == 100
    assert "temperature" not in kwargs


def test_openai_o3_drops_temperature():
    kwargs = {"temperature": 0.5}
    normalize_params(kwargs, "o3-mini", "openai")
    assert "temperature" not in kwargs


def test_openai_gpt5_max_tokens_renamed():
    kwargs = {"max_tokens": 200}
    normalize_params(kwargs, "gpt-5-turbo", "openai")
    assert kwargs["max_completion_tokens"] == 200
    assert "max_tokens" not in kwargs


def test_openai_gpt4_unchanged():
    kwargs = {"max_tokens": 100, "temperature": 0.7}
    normalize_params(kwargs, "gpt-4o", "openai")
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.7


def test_openai_no_double_rename():
    kwargs = {"max_tokens": 100, "max_completion_tokens": 50}
    normalize_params(kwargs, "o1", "openai")
    assert kwargs["max_tokens"] == 100
    assert kwargs["max_completion_tokens"] == 50


def test_anthropic_default_max_tokens():
    kwargs = {"temperature": 0.5}
    normalize_params(kwargs, "claude-3-sonnet", "anthropic")
    assert kwargs["max_tokens"] == 4096
    assert kwargs["temperature"] == 0.5


def test_anthropic_drops_unsupported():
    kwargs = {"logprobs": True, "top_logprobs": 5, "logit_bias": {}, "n": 2, "seed": 42, "temperature": 0.5}
    normalize_params(kwargs, "claude-3-sonnet", "anthropic")
    assert "logprobs" not in kwargs
    assert "top_logprobs" not in kwargs
    assert "logit_bias" not in kwargs
    assert "n" not in kwargs
    assert "seed" not in kwargs
    assert kwargs["temperature"] == 0.5


def test_anthropic_preserves_explicit_max_tokens():
    kwargs = {"max_tokens": 1000}
    normalize_params(kwargs, "claude-3-sonnet", "anthropic")
    assert kwargs["max_tokens"] == 1000


def test_gemini_drops_unsupported():
    kwargs = {"logprobs": True, "top_logprobs": 5, "logit_bias": {}, "n": 2, "seed": 42}
    normalize_params(kwargs, "gemini-2.5-flash", "gemini")
    for param in ("logprobs", "top_logprobs", "logit_bias", "n", "seed"):
        assert param not in kwargs


def test_deepseek_drops_unsupported():
    kwargs = {"logprobs": True, "top_logprobs": 3, "logit_bias": {"1": 0.5}, "temperature": 0.7}
    normalize_params(kwargs, "deepseek-chat", "deepseek")
    assert "logprobs" not in kwargs
    assert "top_logprobs" not in kwargs
    assert "logit_bias" not in kwargs
    assert kwargs["temperature"] == 0.7


def test_groq_drops_unsupported():
    kwargs = {"logprobs": True, "logit_bias": {}, "n": 3, "temperature": 0.5}
    normalize_params(kwargs, "llama-3.3-70b", "groq")
    assert "logprobs" not in kwargs
    assert "logit_bias" not in kwargs
    assert "n" not in kwargs
    assert kwargs["temperature"] == 0.5


def test_unknown_provider_passthrough():
    kwargs = {"max_tokens": 100, "temperature": 0.7, "logprobs": True}
    normalize_params(kwargs, "some-model", "unknown_provider")
    assert kwargs == {"max_tokens": 100, "temperature": 0.7, "logprobs": True}


def test_model_with_slash_in_name():
    kwargs = {"max_tokens": 100}
    normalize_params(kwargs, "some/o1-preview", "openai")
    assert kwargs["max_completion_tokens"] == 100


def test_provider_rules_keys():
    assert "openai" in PROVIDER_PARAM_RULES
    assert "anthropic" in PROVIDER_PARAM_RULES
    assert "gemini" in PROVIDER_PARAM_RULES
    assert "deepseek" in PROVIDER_PARAM_RULES
    assert "groq" in PROVIDER_PARAM_RULES
