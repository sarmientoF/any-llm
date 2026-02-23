"""Web search proxy route — routes through gateway for usage tracking and budget enforcement."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.budget import validate_user_budget
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, ModelPricing, UsageLog, User, get_db
from any_llm.gateway.log_config import logger

from .._auth_helpers import verify_with_subscription_token

router = APIRouter(tags=["search"])

PROVIDER = "search"
MODEL = "searxng"
ENDPOINT = "/res/v1/web/search"


def _resolve_search_backend(config: GatewayConfig) -> str:
    """Resolve search backend URL from config providers.

    Returns base_url (without trailing slash).
    """
    if "search" in config.providers:
        base = config.providers["search"].get("api_base", "")
        if base:
            return base.rstrip("/")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Search provider not configured",
    )


@router.get(ENDPOINT)
async def web_search(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict:
    """Proxy web search with auth, budget check, and usage tracking."""
    api_key, is_master_key = await verify_with_subscription_token(request, db, config)

    user_id: str | None = None
    if not is_master_key and api_key is not None:
        user_id = str(api_key.user_id) if api_key.user_id else None

    if user_id:
        await validate_user_budget(db, user_id)

    backend_url = _resolve_search_backend(config)
    log_id = str(uuid.uuid4())

    # Forward query params to backend
    query_string = request.url.query or ""
    url = f"{backend_url}{ENDPOINT}"
    if query_string:
        url = f"{url}?{query_string}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        _write_log(db, log_id, api_key, user_id, error=str(exc))
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text
        ) from exc
    except Exception as exc:
        _write_log(db, log_id, api_key, user_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Cost tracking — flat per-request
    cost: float | None = None
    model_key = f"{PROVIDER}/{MODEL}"
    pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()
    if pricing:
        cost = pricing.input_price_per_million / 1_000_000  # cost per single request
        if user_id:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                user.spend = float(user.spend) + cost

    _write_log(db, log_id, api_key, user_id, cost=cost)
    return data


def _write_log(
    db: Session,
    log_id: str,
    api_key: APIKey | None,
    user_id: str | None,
    error: str | None = None,
    cost: float | None = None,
) -> None:
    entry = UsageLog(
        id=log_id,
        api_key_id=api_key.id if api_key else None,
        user_id=user_id,
        timestamp=datetime.now(UTC).replace(tzinfo=None),
        model=MODEL,
        provider=PROVIDER,
        endpoint=ENDPOINT,
        status="success" if error is None else "error",
        error_message=error,
        prompt_tokens=1,  # request count (not tokens — search is flat-rate)
        cost=cost,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log search usage: {e}")
        db.rollback()
