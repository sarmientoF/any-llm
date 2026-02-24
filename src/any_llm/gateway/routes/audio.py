"""Audio transcription proxy route — multi-backend routing with cost tracking."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_api_key_or_master_key
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.budget import validate_user_budget
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, ModelPricing, UsageLog, User, get_db
from any_llm.gateway.log_config import logger

router = APIRouter(prefix="/v1/audio", tags=["audio"])


# ---------------------------------------------------------------------------
# Parsed request — multipart is parsed once at route level
# ---------------------------------------------------------------------------


@dataclass
class STTRequest:
    """Structured STT request parsed from incoming multipart/raw body."""

    file_bytes: bytes
    file_name: str = "audio.wav"
    file_content_type: str = "application/octet-stream"
    model: str = "whisper-local"
    response_format: str = "json"
    language: str | None = None
    extra_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class STTProviderResult:
    """Normalized result from an STT provider."""

    raw_response: dict
    duration_ms: int | None


# ---------------------------------------------------------------------------
# STT Handler abstraction
# ---------------------------------------------------------------------------


class STTHandler(ABC):
    """Base class for STT provider handlers."""

    @abstractmethod
    def get_url(self, model: str, config_base: str | None) -> str: ...

    @abstractmethod
    def build_request(
        self, req: STTRequest, model_name: str, api_key: str
    ) -> tuple[bytes | None, dict | None, dict[str, str]]:
        """Return (content, files_dict, headers) for upstream request.

        Exactly one of content or files_dict should be set:
        - content (bytes) → raw binary body (httpx content=...)
        - files_dict (dict) → multipart (httpx files=...)
        """
        ...

    @abstractmethod
    def parse_response(self, data: dict) -> STTProviderResult: ...


class OpenAICompatHandler(STTHandler):
    """Handles OpenAI, Groq, Fireworks — multipart /audio/transcriptions."""

    def __init__(self, default_base: str) -> None:
        self.default_base = default_base

    def get_url(self, model: str, config_base: str | None) -> str:
        base = (config_base or self.default_base).rstrip("/")
        return f"{base}/audio/transcriptions"

    def build_request(
        self, req: STTRequest, model_name: str, api_key: str
    ) -> tuple[bytes | None, dict | None, dict[str, str]]:
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {
            "file": (req.file_name, req.file_bytes, req.file_content_type),
            "model": (None, model_name),
            "response_format": (None, req.response_format),
        }
        if req.language:
            files["language"] = (None, req.language)
        for k, v in req.extra_fields.items():
            files[k] = (None, v)
        return None, files, headers

    def parse_response(self, data: dict) -> STTProviderResult:
        duration_ms: int | None = None
        if "duration_ms" in data:
            duration_ms = data["duration_ms"]
        elif "duration" in data:
            try:
                duration_ms = int(float(data["duration"]) * 1000)
            except (TypeError, ValueError):
                pass
        # Groq x_groq.usage.audio_seconds
        if duration_ms is None:
            x_groq = data.get("x_groq", {})
            if isinstance(x_groq, dict):
                usage = x_groq.get("usage", {})
                if isinstance(usage, dict) and "audio_seconds" in usage:
                    try:
                        duration_ms = int(float(usage["audio_seconds"]) * 1000)
                    except (TypeError, ValueError):
                        pass
        return STTProviderResult(raw_response=data, duration_ms=duration_ms)


class DeepgramHandler(STTHandler):
    """Deepgram — binary body to /v1/listen."""

    def __init__(self, default_base: str = "https://api.deepgram.com/v1") -> None:
        self.default_base = default_base

    def get_url(self, model: str, config_base: str | None) -> str:
        base = (config_base or self.default_base).rstrip("/")
        return f"{base}/listen?{urlencode({'model': model})}"

    def build_request(
        self, req: STTRequest, model_name: str, api_key: str
    ) -> tuple[bytes | None, dict | None, dict[str, str]]:
        ct = req.file_content_type
        if ct == "application/octet-stream":
            ct = "audio/wav"
        headers = {"Authorization": f"Token {api_key}", "Content-Type": ct}
        return req.file_bytes, None, headers

    def parse_response(self, data: dict) -> STTProviderResult:
        text = ""
        duration_ms: int | None = None
        results = data.get("results", {})
        if isinstance(results, dict):
            channels = results.get("channels", [])
            if channels and isinstance(channels[0], dict):
                alts = channels[0].get("alternatives", [])
                if alts and isinstance(alts[0], dict):
                    text = alts[0].get("transcript", "")
        # Duration from metadata (top-level, independent of results)
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict) and "duration" in metadata:
            try:
                duration_ms = int(float(metadata["duration"]) * 1000)
            except (TypeError, ValueError):
                pass
        normalized = {"text": text}
        return STTProviderResult(raw_response=normalized, duration_ms=duration_ms)


class SelfHostedHandler(STTHandler):
    """Self-hosted whisper — multipart, no auth."""

    def get_url(self, model: str, config_base: str | None) -> str:
        if not config_base:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="STT provider not configured",
            )
        base = config_base.rstrip("/")
        return f"{base}/v1/audio/transcriptions"

    def build_request(
        self, req: STTRequest, model_name: str, api_key: str
    ) -> tuple[bytes | None, dict | None, dict[str, str]]:
        files = {
            "file": (req.file_name, req.file_bytes, req.file_content_type),
            "model": (None, model_name),
            "response_format": (None, req.response_format),
        }
        if req.language:
            files["language"] = (None, req.language)
        return None, files, {}

    def parse_response(self, data: dict) -> STTProviderResult:
        duration_ms: int | None = None
        if "duration_ms" in data:
            duration_ms = data["duration_ms"]
        elif "duration" in data:
            try:
                duration_ms = int(float(data["duration"]) * 1000)
            except (TypeError, ValueError):
                pass
        return STTProviderResult(raw_response=data, duration_ms=duration_ms)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

STT_PROVIDERS: dict[str, STTHandler] = {
    "stt": SelfHostedHandler(),
    "openai": OpenAICompatHandler("https://api.openai.com/v1"),
    "groq": OpenAICompatHandler("https://api.groq.com/openai/v1"),
    "fireworks": OpenAICompatHandler("https://api.fireworks.ai/inference/v1"),
    "deepgram": DeepgramHandler(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Known multipart field names used by OpenAI STT API
_KNOWN_STT_FIELDS = {"file", "model", "response_format", "language"}


def parse_stt_request(body: bytes, content_type: str) -> STTRequest:
    """Parse incoming request body into a structured STTRequest."""
    if "multipart/form-data" not in content_type:
        return STTRequest(file_bytes=body)

    boundary = _extract_boundary(content_type)
    if not boundary:
        return STTRequest(file_bytes=body)

    boundary_bytes = f"--{boundary}".encode()
    parts = body.split(boundary_bytes)

    file_bytes = b""
    file_name = "audio.wav"
    file_ct = "application/octet-stream"
    model = "whisper-local"
    response_format = "json"
    language: str | None = None
    extra: dict[str, str] = {}

    for p in parts:
        header_end = p.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        header_section = p[:header_end].decode(errors="replace")
        value_bytes = p[header_end + 4 :]
        # Strip trailing \r\n
        if value_bytes.endswith(b"\r\n"):
            value_bytes = value_bytes[:-2]
        # Strip trailing closing boundary --
        if value_bytes.endswith(b"--"):
            value_bytes = value_bytes[:-2]
        if value_bytes.endswith(b"\r\n"):
            value_bytes = value_bytes[:-2]

        if 'name="file"' in header_section:
            file_bytes = value_bytes
            # Extract filename from Content-Disposition line only
            for line in header_section.split("\r\n"):
                if "content-disposition" in line.lower():
                    for token in line.split(";"):
                        token = token.strip()
                        if token.startswith("filename="):
                            file_name = token[len("filename=") :].strip('"')
                elif line.strip().lower().startswith("content-type:"):
                    file_ct = line.split(":", 1)[1].strip()
        elif 'name="model"' in header_section:
            val = value_bytes.decode(errors="replace").strip()
            if val:
                model = val
        elif 'name="response_format"' in header_section:
            val = value_bytes.decode(errors="replace").strip()
            if val:
                response_format = val
        elif 'name="language"' in header_section:
            val = value_bytes.decode(errors="replace").strip()
            if val:
                language = val
        else:
            # Capture any extra fields (e.g. temperature, prompt)
            for token in header_section.split(";"):
                token = token.strip()
                if token.startswith('name="'):
                    field_name = token[len('name="') :].rstrip('"')
                    if field_name not in _KNOWN_STT_FIELDS:
                        extra[field_name] = value_bytes.decode(errors="replace").strip()

    return STTRequest(
        file_bytes=file_bytes,
        file_name=file_name,
        file_content_type=file_ct,
        model=model,
        response_format=response_format,
        language=language,
        extra_fields=extra,
    )


def _extract_boundary(content_type: str) -> str:
    """Extract boundary from Content-Type header."""
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            return part[len("boundary=") :].strip('"')
    return ""


def _resolve_stt_backend(
    model: str, config: GatewayConfig
) -> tuple[STTHandler, str, str, str, str]:
    """Resolve STT backend from model name.

    Routing order:
    1. "stt/" prefix → self-hosted whisper
    2. "provider/model" where provider is in registry or config → external API
    3. Unprefixed model → fall back to self-hosted STT

    Returns (handler, url, api_key, provider, model_name).
    """
    # 1. Self-hosted STT (explicit stt/ prefix)
    if model.startswith("stt/"):
        model_name = model[len("stt/") :]
        handler = STT_PROVIDERS["stt"]
        stt_base = config.providers.get("stt", {}).get("api_base", "")
        url = handler.get_url(model_name, stt_base or None)
        return handler, url, "", "stt", model_name

    # 2. Try provider/model split
    if "/" in model:
        provider, model_name = model.split("/", 1)
        provider_cfg = config.providers.get(provider, {})

        # Known provider in registry
        if provider in STT_PROVIDERS:
            handler = STT_PROVIDERS[provider]
            api_key = provider_cfg.get("api_key", "")
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"{provider} provider not configured (missing api_key)",
                )
            config_base = provider_cfg.get("api_base")
            url = handler.get_url(model_name, config_base)
            return handler, url, api_key, provider, model_name

        # Unknown provider but has config with api_base + api_key → assume OpenAI-compat
        if provider_cfg.get("api_base") and provider_cfg.get("api_key"):
            handler = OpenAICompatHandler(provider_cfg["api_base"])
            url = handler.get_url(model_name, provider_cfg["api_base"])
            return handler, url, provider_cfg["api_key"], provider, model_name

    # 3. Unprefixed or unresolved provider → self-hosted fallback
    handler = STT_PROVIDERS["stt"]
    fallback_name = model.split("/", 1)[-1] if "/" in model else model
    stt_base = config.providers.get("stt", {}).get("api_base", "")
    url = handler.get_url(fallback_name, stt_base or None)
    return handler, url, "", "stt", fallback_name


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


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

    if user_id:
        await validate_user_budget(db, user_id)

    # Parse multipart once into structured request
    body = await request.body()
    content_type = request.headers.get("content-type", "")
    stt_req = parse_stt_request(body, content_type)

    handler, url, api_key_str, provider, model_name = _resolve_stt_backend(stt_req.model, config)
    log_id = str(uuid.uuid4())

    # Force verbose_json upstream to get duration metadata for cost tracking.
    # We'll downgrade the response back to the client's requested format.
    client_format = stt_req.response_format
    if client_format == "json":
        stt_req.response_format = "verbose_json"

    # Handler builds the upstream request from structured fields
    content, files, headers = handler.build_request(stt_req, model_name, api_key_str)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if files is not None:
                resp = await client.post(url, files=files, headers=headers)
            else:
                resp = await client.post(url, content=content, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        _write_log(db, log_id, api_key, user_id, provider, model_name, error=str(exc))
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text
        ) from exc
    except Exception as exc:
        _write_log(db, log_id, api_key, user_id, provider, model_name, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Parse response via handler (normalizes format + extracts duration)
    result = handler.parse_response(data)
    duration_ms = result.duration_ms

    cost: float | None = None
    if duration_ms is not None:
        model_key = f"{provider}/{model_name}"
        pricing = (
            db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()
        )
        if not pricing:
            pricing = (
                db.query(ModelPricing)
                .filter(ModelPricing.model_key == f"stt/{model_name}")
                .first()
            )
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

    # Downgrade response if we upgraded format for cost tracking
    response = result.raw_response
    if client_format == "json" and isinstance(response, dict):
        response = {"text": response.get("text", "")}
    return response


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
        prompt_tokens=duration_ms or 0,  # STT: stores audio duration_ms, not token count
        cost=cost,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log STT usage: {e}")
        db.rollback()
