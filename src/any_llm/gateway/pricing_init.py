"""Pricing initialization from configuration."""

from sqlalchemy.orm import Session

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import UnsupportedProviderError
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import ModelPricing
from any_llm.gateway.log_config import logger


def initialize_pricing_from_config(config: GatewayConfig, db: Session) -> None:
    """Initialize model pricing from configuration file.

    Loads pricing from config.pricing and stores it in the database.
    Database pricing takes precedence - if pricing exists in DB, it is not overwritten.

    Args:
        config: Gateway configuration containing pricing definitions
        db: Database session

    Raises:
        ValueError: If pricing is defined for a model from an unconfigured provider

    """
    if not config.pricing:
        logger.debug("No pricing configuration found in config file")
        return

    logger.info(f"Loading pricing configuration for {len(config.pricing)} model(s)")

    # Validate providers upfront
    for model_key in config.pricing:
        try:
            provider, _ = AnyLLM.split_model_provider(model_key)
            provider_name = provider.value
        except (ValueError, UnsupportedProviderError):
            # Non-LLM provider (e.g., stt)
            provider_name = model_key.split("/", 1)[0] if "/" in model_key else model_key.split(":", 1)[0]

        if provider_name not in config.providers:
            msg = (
                f"Cannot set pricing for model '{model_key}': "
                f"provider '{provider_name}' is not configured in the providers section"
            )
            raise ValueError(msg)

    # Batch check: single SELECT for all configured model keys
    all_keys = list(config.pricing.keys())
    existing_keys = {
        row[0]
        for row in db.query(ModelPricing.model_key)
        .filter(ModelPricing.model_key.in_(all_keys))
        .all()
    }

    # Steady state: all pricing already seeded — skip entirely
    missing_keys = set(all_keys) - existing_keys
    if not missing_keys:
        logger.info(f"Pricing initialization complete (all {len(all_keys)} models already exist)")
        return

    for model_key in missing_keys:
        pricing_config = config.pricing[model_key]
        db.add(ModelPricing(
            model_key=model_key,
            input_price_per_million=pricing_config.input_price_per_million,
            output_price_per_million=pricing_config.output_price_per_million,
        ))
        logger.info(
            f"Added pricing for '{model_key}': "
            f"input=${pricing_config.input_price_per_million}/M, "
            f"output=${pricing_config.output_price_per_million}/M"
        )

    db.commit()
    logger.info(f"Pricing initialization complete ({len(missing_keys)} added, {len(existing_keys)} existing)")
