"""Audio transcription proxy route — multi-backend routing with cost tracking."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_api_key_or_master_key
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.budget import validate_user_budget
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, ModelPricing, UsageLog, User, get_db
from any_llm.gateway.log_config import logger

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

router = APIRouter(prefix="/v1/audio", tags=["audio"])


def _resolve_stt_backend(model: str, config: GatewayConfig) -> tuple[str, dict[str, str], str, str]:
    """Resolve STT backend from model name.

    Returns (url, headers, provider, model_name).
    """
    if model.startswith("groq/"):
        model_name = model[len("groq/") :]
        groq_key = ""
        if "groq" in config.providers:
            groq_key = config.providers["groq"].get("api_key", "")
        if not groq_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Groq provider not configured",
            )
        return (
            GROQ_TRANSCRIPTION_URL,
            {"Authorization": f"Bearer {groq_key}"},
            "groq",
            model_name,
        )

    # Self-hosted whisper (unprefixed or stt/ prefix)
    if model.startswith("stt/"):
        model_name = model[len("stt/") :]
    else:
        model_name = model

    stt_base = ""
    if "stt" in config.providers:
        stt_base = config.providers["stt"].get("api_base", "")
    if not stt_base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="STT provider not configured",
        )
    stt_base = stt_base.rstrip("/")
    return (
        f"{stt_base}/v1/audio/transcriptions",
        {},
        "stt",
        model_name,
    )


@router.post("/transcriptions")
async def audio_transcriptions(
    request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict:
    """Proxy audio transcription with model-based routing and cost tracking."""
    api_key, is_master_key = auth_result

    user_id: str | None = None
    if not is_master_key and api_key is not None:
        user_id = str(api_key.user_id) if api_key.user_id else None

    # Budget check
    if user_id:
        await validate_user_budget(db, user_id)

    # Parse model from multipart form body
    body = await request.body()
    content_type = request.headers.get("content-type", "")
    model = _extract_model_from_multipart(body, content_type)

    url, extra_headers, provider, model_name = _resolve_stt_backend(model, config)
    log_id = str(uuid.uuid4())

    headers = {"Content-Type": content_type, **extra_headers}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, content=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        _write_log(db, log_id, api_key, user_id, provider, model_name, error=str(exc))
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except Exception as exc:
        _write_log(db, log_id, api_key, user_id, provider, model_name, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Cost tracking
    duration_ms = data.get("duration_ms")
    cost: float | None = None
    if duration_ms is not None:
        model_key = f"{provider}/{model_name}"
        pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()
        if not pricing:
            # Try with stt/ prefix
            pricing = db.query(ModelPricing).filter(ModelPricing.model_key == f"stt/{model_name}").first()
        if pricing:
            cost = (duration_ms / 1_000_000) * pricing.input_price_per_million
            if user_id:
                user = db.query(User).filter(User.user_id == user_id).first()
                if user:
                    user.spend = float(user.spend) + cost

    _write_log(
        db,
        log_id,
        api_key,
        user_id,
        provider,
        model_name,
        duration_ms=duration_ms,
        cost=cost,
    )
    return data


def _extract_model_from_multipart(body: bytes, content_type: str) -> str:
    """Extract model field from multipart form data. Returns default if not found."""
    default = "whisper-local"
    if "multipart/form-data" not in content_type:
        return default
    try:
        # Parse boundary from content-type
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[len("boundary=") :]
                break
        else:
            return default

        # Search for model field in multipart body
        boundary_bytes = f"--{boundary}".encode()
        parts = body.split(boundary_bytes)
        for p in parts:
            if b'name="model"' in p:
                # Value is after double CRLF
                idx = p.find(b"\r\n\r\n")
                if idx != -1:
                    val = p[idx + 4 :].strip().rstrip(b"\r\n-").decode().strip()
                    if val:
                        return val
    except Exception:
        logger.debug("Failed to parse model from multipart body")
    return default


def _write_log(
    db: Session,
    log_id: str,
    api_key: APIKey | None,
    user_id: str | None,
    provider: str,
    model: str,
    error: str | None = None,
    duration_ms: int | None = None,
    cost: float | None = None,
) -> None:
    entry = UsageLog(
        id=log_id,
        api_key_id=api_key.id if api_key else None,
        user_id=user_id,
        timestamp=datetime.now(UTC).replace(tzinfo=None),
        model=model,
        provider=provider,
        endpoint="/v1/audio/transcriptions",
        status="success" if error is None else "error",
        error_message=error,
        prompt_tokens=duration_ms or 0,
        cost=cost,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log STT usage: {e}")
        db.rollback()
