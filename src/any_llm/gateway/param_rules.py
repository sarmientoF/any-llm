"""Table-driven parameter normalization for LLM providers.

Each provider has quirks in what parameters it accepts. This module
centralizes those rules so the gateway can fix params before forwarding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderRules:
    """Normalization rules for a single provider."""

    default_max_tokens: int | None = None
    unsupported_params: frozenset[str] = frozenset()
    param_renames: dict[str, str] = field(default_factory=dict)
    max_completion_tokens_models: tuple[str, ...] = ()
    fixed_temperature_models: tuple[str, ...] = ()


_OPENAI_REASONING = ("o1", "o3", "o4", "gpt-5")

PROVIDER_PARAM_RULES: dict[str, ProviderRules] = {
    "openai": ProviderRules(
        max_completion_tokens_models=_OPENAI_REASONING,
        fixed_temperature_models=_OPENAI_REASONING,
    ),
    "anthropic": ProviderRules(
        default_max_tokens=4096,
        unsupported_params=frozenset({"logprobs", "top_logprobs", "logit_bias", "n", "seed"}),
    ),
    "gemini": ProviderRules(
        unsupported_params=frozenset({"logprobs", "top_logprobs", "logit_bias", "n", "seed"}),
    ),
    "deepseek": ProviderRules(
        unsupported_params=frozenset({"logprobs", "top_logprobs", "logit_bias"}),
    ),
    "groq": ProviderRules(
        unsupported_params=frozenset({"logprobs", "logit_bias", "n"}),
    ),
}


def normalize_params(kwargs: dict[str, Any], model: str, provider: str) -> None:
    """Normalize completion kwargs in-place based on provider rules.

    Args:
        kwargs: Mutable dict of completion parameters.
        model: Bare model name (after splitting from provider prefix).
        provider: Provider string (e.g. "openai", "anthropic").

    """
    rules = PROVIDER_PARAM_RULES.get(provider)
    if rules is None:
        return

    bare = model.rsplit("/", maxsplit=1)[-1].lower() if "/" in model else model.lower()

    # Drop unsupported params
    for param in rules.unsupported_params:
        kwargs.pop(param, None)

    # Apply param renames
    for old_name, new_name in rules.param_renames.items():
        if old_name in kwargs and new_name not in kwargs:
            kwargs[new_name] = kwargs.pop(old_name)

    # max_tokens → max_completion_tokens for specific models
    if (
        rules.max_completion_tokens_models
        and "max_tokens" in kwargs
        and "max_completion_tokens" not in kwargs
        and any(bare.startswith(p) for p in rules.max_completion_tokens_models)
    ):
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    # Drop temperature for fixed-temperature models
    if rules.fixed_temperature_models and "temperature" in kwargs:
        if any(bare.startswith(p) for p in rules.fixed_temperature_models):
            kwargs.pop("temperature")

    # Set default max_tokens if provider requires it and none specified
    if rules.default_max_tokens and "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
        kwargs["max_tokens"] = rules.default_max_tokens
