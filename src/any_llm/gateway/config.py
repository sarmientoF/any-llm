import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

API_KEY_HEADER = "X-AnyLLM-Key"


class PricingConfig(BaseModel):
    """Model pricing configuration."""

    input_price_per_million: float
    output_price_per_million: float


class RetryConfig(BaseModel):
    """Retry configuration for provider calls."""

    max_retries: int = 1
    base_delay: float = 0.5
    max_delay: float = 10.0


class RateLimitConfig(BaseModel):
    """Per-provider rate limit configuration."""

    rpm: int = 60
    tpm: int = 100_000


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    cooldown_sec: float = 60.0


class CacheConfig(BaseModel):
    """Response cache configuration."""

    enabled: bool = False
    max_size: int = 1000
    default_ttl: float = 300.0


class GatewayConfig(BaseSettings):
    """Gateway configuration with support for YAML files and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/any_llm_gateway",
        description="Database connection URL for PostgreSQL",
    )
    auto_migrate: bool = Field(
        default=True,
        description="Automatically run database migrations on startup",
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")  # noqa: S104
    port: int = Field(default=8000, description="Port to bind the server to")
    master_key: str | None = Field(default=None, description="Master key for protecting management endpoints")
    providers: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Pre-configured provider credentials"
    )
    pricing: dict[str, PricingConfig] = Field(
        default_factory=dict,
        description="Pre-configured model USD pricing (model_key -> {input_price_per_million, output_price_per_million})",
    )
    model_aliases: dict[str, str] = Field(default_factory=dict, description="Model alias → provider/model mapping")
    fallbacks: dict[str, list[str]] = Field(default_factory=dict, description="Model → fallback models list")
    retry: RetryConfig = Field(default_factory=RetryConfig)
    rate_limits: dict[str, RateLimitConfig] = Field(default_factory=dict, description="Per-provider rate limits")
    default_rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def load_config(config_path: str | None = None) -> GatewayConfig:
    """Load configuration from file and environment variables.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        GatewayConfig instance with merged configuration

    """
    config_dict: dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict = _resolve_env_vars(yaml_config)

    return GatewayConfig(**config_dict)


def _resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve environment variable references in config.

    Supports ${VAR_NAME} syntax in string values.
    """
    if isinstance(config, dict):
        return {key: _resolve_env_vars(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    if isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    return config
