"""Shared auth helpers for non-standard header extraction."""

import secrets

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from any_llm.gateway.auth.dependencies import _verify_and_update_api_key
from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.db import APIKey

SUBSCRIPTION_TOKEN_HEADER = "X-Subscription-Token"


async def verify_with_subscription_token(
    request: Request,
    db: Session,
    config: GatewayConfig,
) -> tuple[APIKey | None, bool]:
    """Verify auth from X-Subscription-Token, X-AnyLLM-Key, or Authorization header.

    Returns (APIKey | None, is_master_key).
    """
    # Try X-Subscription-Token first (Brave-compat / nanobot search)
    token = request.headers.get(SUBSCRIPTION_TOKEN_HEADER)
    if not token:
        # Fall back to standard headers
        auth_header = request.headers.get(API_KEY_HEADER) or request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication header",
            )
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid header format. Expected 'Bearer <token>'",
            )
        token = auth_header[7:]

    # Check master key
    if config.master_key and secrets.compare_digest(token, config.master_key):
        return None, True

    # Verify as API key
    api_key = _verify_and_update_api_key(db, token)
    return api_key, False
