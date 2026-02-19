"""GET /v1/models — list configured aliases and available providers."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.config import GatewayConfig

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any]:
    """List model aliases and configured providers."""
    models = []
    for alias, target in config.model_aliases.items():
        models.append({"id": alias, "target": target, "type": "alias"})
    for provider_name in config.providers:
        models.append({"id": provider_name, "type": "provider"})
    return {"object": "list", "data": models}
