"""Tests for STT provider routing, handlers, and request parsing."""

import pytest
from fastapi import HTTPException

from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.routes.audio import (
    STT_PROVIDERS,
    DeepgramHandler,
    OpenAICompatHandler,
    SelfHostedHandler,
    STTRequest,
    _resolve_stt_backend,
    parse_stt_request,
)


def _make_config(**providers: dict) -> GatewayConfig:
    return GatewayConfig(
        database_url="postgresql://x:x@localhost/test",
        master_key="test",
        providers=providers,
    )


def _build_multipart_with_file(
    file_bytes: bytes,
    file_name: str = "audio.wav",
    file_ct: str = "audio/wav",
    model: str = "whisper-1",
    response_format: str = "json",
    boundary: str = "abc123",
    extra_fields: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    """Build a multipart body with a file field and form fields."""
    parts = []
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
        f"Content-Type: {file_ct}\r\n"
        f"\r\n"
    )
    body_parts = [parts[0].encode(), file_bytes, b"\r\n"]
    for name, value in [("model", model), ("response_format", response_format)]:
        part = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"\r\n"
            f"{value}\r\n"
        )
        body_parts.append(part.encode())
    if extra_fields:
        for name, value in extra_fields.items():
            part = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n"
            )
            body_parts.append(part.encode())
    body_parts.append(f"--{boundary}--\r\n".encode())
    ct = f"multipart/form-data; boundary={boundary}"
    return b"".join(body_parts), ct


# ---------------------------------------------------------------------------
# parse_stt_request
# ---------------------------------------------------------------------------


def test_parse_stt_request_multipart():
    body, ct = _build_multipart_with_file(b"AUDIODATA", model="groq/whisper-large-v3-turbo")
    req = parse_stt_request(body, ct)
    assert req.file_bytes == b"AUDIODATA"
    assert req.file_name == "audio.wav"
    assert req.file_content_type == "audio/wav"
    assert req.model == "groq/whisper-large-v3-turbo"
    assert req.response_format == "json"


def test_parse_stt_request_extracts_language():
    body, ct = _build_multipart_with_file(
        b"AUDIO", model="openai/whisper-1", extra_fields={"language": "en"}
    )
    req = parse_stt_request(body, ct)
    assert req.language == "en"


def test_parse_stt_request_extra_fields():
    body, ct = _build_multipart_with_file(
        b"AUDIO", model="openai/whisper-1", extra_fields={"temperature": "0.2"}
    )
    req = parse_stt_request(body, ct)
    assert req.extra_fields == {"temperature": "0.2"}


def test_parse_stt_request_non_multipart():
    req = parse_stt_request(b"raw-audio-bytes", "audio/wav")
    assert req.file_bytes == b"raw-audio-bytes"
    assert req.model == "whisper-local"


def test_parse_stt_request_missing_model():
    body, ct = _build_multipart_with_file(b"AUDIO")
    # Model is "whisper-1" from the builder
    req = parse_stt_request(body, ct)
    assert req.model == "whisper-1"


def test_parse_stt_request_no_model_field():
    boundary = "testbound"
    ct = f"multipart/form-data; boundary={boundary}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        f"Content-Type: audio/wav\r\n"
        f"\r\n"
        f"DATA\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = parse_stt_request(body, ct)
    assert req.model == "whisper-local"
    assert req.file_bytes == b"DATA"


def test_parse_stt_request_quoted_boundary():
    ct = 'multipart/form-data; boundary="----testboundary"'
    body = (
        b"------testboundary\r\n"
        b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        b"Content-Type: audio/wav\r\n"
        b"\r\n"
        b"WAVDATA\r\n"
        b"------testboundary\r\n"
        b'Content-Disposition: form-data; name="model"\r\n'
        b"\r\n"
        b"openai/whisper-1\r\n"
        b"------testboundary--\r\n"
    )
    req = parse_stt_request(body, ct)
    assert req.file_bytes == b"WAVDATA"
    assert req.model == "openai/whisper-1"


# ---------------------------------------------------------------------------
# _resolve_stt_backend
# ---------------------------------------------------------------------------


def test_stt_prefix_routes_to_self_hosted():
    config = _make_config(stt={"api_base": "http://stt-host"})
    handler, url, api_key, provider, model = _resolve_stt_backend(
        "stt/whisper-large-v3-turbo", config
    )
    assert isinstance(handler, SelfHostedHandler)
    assert url == "http://stt-host/v1/audio/transcriptions"
    assert api_key == ""
    assert provider == "stt"
    assert model == "whisper-large-v3-turbo"


def test_unprefixed_model_falls_back_to_self_hosted():
    config = _make_config(stt={"api_base": "http://stt-host"})
    handler, url, _, provider, model = _resolve_stt_backend("whisper-local", config)
    assert isinstance(handler, SelfHostedHandler)
    assert provider == "stt"
    assert model == "whisper-local"
    assert url == "http://stt-host/v1/audio/transcriptions"


def test_groq_provider_routing():
    config = _make_config(groq={"api_key": "gsk_test"})
    handler, url, api_key, provider, model = _resolve_stt_backend(
        "groq/whisper-large-v3-turbo", config
    )
    assert isinstance(handler, OpenAICompatHandler)
    assert url == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert api_key == "gsk_test"
    assert provider == "groq"
    assert model == "whisper-large-v3-turbo"


def test_openai_provider_routing():
    config = _make_config(openai={"api_key": "sk-test"})
    handler, url, api_key, provider, model = _resolve_stt_backend("openai/whisper-1", config)
    assert isinstance(handler, OpenAICompatHandler)
    assert url == "https://api.openai.com/v1/audio/transcriptions"
    assert api_key == "sk-test"
    assert provider == "openai"
    assert model == "whisper-1"


def test_fireworks_provider_routing():
    config = _make_config(fireworks={"api_key": "fw-test"})
    handler, url, api_key, provider, model = _resolve_stt_backend(
        "fireworks/whisper-v3-turbo", config
    )
    assert isinstance(handler, OpenAICompatHandler)
    assert url == "https://api.fireworks.ai/inference/v1/audio/transcriptions"
    assert api_key == "fw-test"
    assert provider == "fireworks"
    assert model == "whisper-v3-turbo"


def test_deepgram_provider_routing():
    config = _make_config(deepgram={"api_key": "dg-test"})
    handler, url, api_key, provider, model = _resolve_stt_backend("deepgram/nova-3", config)
    assert isinstance(handler, DeepgramHandler)
    assert url == "https://api.deepgram.com/v1/listen?model=nova-3"
    assert api_key == "dg-test"
    assert provider == "deepgram"
    assert model == "nova-3"


def test_deepgram_config_api_base_override():
    config = _make_config(
        deepgram={"api_key": "dg-test", "api_base": "https://custom.deepgram.example/v1"}
    )
    _, url, _, _, _ = _resolve_stt_backend("deepgram/nova-3", config)
    assert url == "https://custom.deepgram.example/v1/listen?model=nova-3"


def test_config_api_base_overrides_default():
    config = _make_config(groq={"api_key": "gsk_test", "api_base": "https://custom.groq.example"})
    _, url, _, _, _ = _resolve_stt_backend("groq/whisper-large-v3", config)
    assert url == "https://custom.groq.example/audio/transcriptions"


def test_custom_provider_with_config():
    config = _make_config(
        acme={"api_key": "acme-key", "api_base": "https://acme.example/v1"}
    )
    handler, url, api_key, provider, model = _resolve_stt_backend("acme/custom-model", config)
    assert isinstance(handler, OpenAICompatHandler)
    assert url == "https://acme.example/v1/audio/transcriptions"
    assert api_key == "acme-key"
    assert provider == "acme"
    assert model == "custom-model"


def test_known_provider_missing_api_key_raises():
    config = _make_config(openai={})
    with pytest.raises(HTTPException) as exc_info:
        _resolve_stt_backend("openai/whisper-1", config)
    assert exc_info.value.status_code == 503
    assert "openai" in exc_info.value.detail


def test_unknown_provider_without_config_falls_back_to_stt():
    config = _make_config(stt={"api_base": "http://stt-host"})
    _, _, _, provider, model = _resolve_stt_backend("unknown/some-model", config)
    assert provider == "stt"
    assert model == "some-model"


def test_unknown_provider_no_stt_raises():
    """Unknown provider with no config AND no stt → 503."""
    config = _make_config()
    with pytest.raises(HTTPException) as exc_info:
        _resolve_stt_backend("unknown/some-model", config)
    assert exc_info.value.status_code == 503


def test_unprefixed_no_stt_raises():
    config = _make_config()
    with pytest.raises(HTTPException) as exc_info:
        _resolve_stt_backend("whisper-local", config)
    assert exc_info.value.status_code == 503


def test_stt_base_trailing_slash_stripped():
    config = _make_config(stt={"api_base": "http://stt-host/"})
    _, url, _, _, _ = _resolve_stt_backend("stt/model", config)
    assert url == "http://stt-host/v1/audio/transcriptions"


def test_all_known_providers_have_handlers():
    """Sanity check: all known providers are in the registry."""
    for provider in ["openai", "groq", "fireworks", "deepgram"]:
        assert provider in STT_PROVIDERS


# ---------------------------------------------------------------------------
# OpenAICompatHandler
# ---------------------------------------------------------------------------


def test_openai_handler_get_url_default():
    h = OpenAICompatHandler("https://api.openai.com/v1")
    assert h.get_url("whisper-1", None) == "https://api.openai.com/v1/audio/transcriptions"


def test_openai_handler_get_url_override():
    h = OpenAICompatHandler("https://api.openai.com/v1")
    assert h.get_url("whisper-1", "https://custom/v1") == "https://custom/v1/audio/transcriptions"


def test_openai_handler_build_request_multipart():
    h = OpenAICompatHandler("https://api.openai.com/v1")
    req = STTRequest(
        file_bytes=b"AUDIO",
        file_name="test.wav",
        file_content_type="audio/wav",
        model="openai/whisper-1",
        response_format="verbose_json",
        language="en",
    )
    content, files, headers = h.build_request(req, "whisper-1", "sk-test")
    assert content is None
    assert files is not None
    assert files["file"] == ("test.wav", b"AUDIO", "audio/wav")
    assert files["model"] == (None, "whisper-1")  # Provider prefix stripped
    assert files["response_format"] == (None, "verbose_json")
    assert files["language"] == (None, "en")
    assert headers["Authorization"] == "Bearer sk-test"


def test_openai_handler_build_request_extra_fields():
    h = OpenAICompatHandler("https://api.openai.com/v1")
    req = STTRequest(
        file_bytes=b"AUDIO",
        extra_fields={"temperature": "0.2"},
    )
    _, files, _ = h.build_request(req, "whisper-1", "sk-test")
    assert files["temperature"] == (None, "0.2")


def test_openai_handler_parse_duration_seconds():
    h = OpenAICompatHandler("https://api.openai.com/v1")
    result = h.parse_response({"text": "hello", "duration": 3.5})
    assert result.duration_ms == 3500
    assert result.raw_response == {"text": "hello", "duration": 3.5}


def test_openai_handler_parse_duration_ms():
    h = OpenAICompatHandler("https://api.openai.com/v1")
    result = h.parse_response({"text": "hello", "duration_ms": 5000})
    assert result.duration_ms == 5000


def test_openai_handler_parse_groq_duration():
    h = OpenAICompatHandler("https://api.openai.com/v1")
    data = {"text": "hello", "x_groq": {"usage": {"audio_seconds": 10.2}}}
    result = h.parse_response(data)
    assert result.duration_ms == 10200


def test_openai_handler_parse_no_duration():
    h = OpenAICompatHandler("https://api.openai.com/v1")
    result = h.parse_response({"text": "hello"})
    assert result.duration_ms is None


# ---------------------------------------------------------------------------
# DeepgramHandler
# ---------------------------------------------------------------------------


def test_deepgram_handler_get_url():
    h = DeepgramHandler()
    assert h.get_url("nova-3", None) == "https://api.deepgram.com/v1/listen?model=nova-3"


def test_deepgram_handler_get_url_override():
    h = DeepgramHandler()
    url = h.get_url("nova-3", "https://custom.dg/v1")
    assert url == "https://custom.dg/v1/listen?model=nova-3"


def test_deepgram_handler_build_request_binary():
    h = DeepgramHandler()
    req = STTRequest(file_bytes=b"raw-audio", file_content_type="audio/wav")
    content, files, headers = h.build_request(req, "nova-3", "dg-key")
    assert content == b"raw-audio"
    assert files is None
    assert headers["Authorization"] == "Token dg-key"
    assert headers["Content-Type"] == "audio/wav"


def test_deepgram_handler_octet_stream_fallback():
    h = DeepgramHandler()
    req = STTRequest(file_bytes=b"raw", file_content_type="application/octet-stream")
    _, _, headers = h.build_request(req, "nova-3", "dg-key")
    assert headers["Content-Type"] == "audio/wav"


def test_deepgram_handler_parse_response():
    h = DeepgramHandler()
    data = {
        "metadata": {"duration": 5.5, "model_info": {}},
        "results": {
            "channels": [
                {"alternatives": [{"transcript": "hello world", "confidence": 0.99}]}
            ]
        },
    }
    result = h.parse_response(data)
    assert result.raw_response == {"text": "hello world"}
    assert result.duration_ms == 5500


def test_deepgram_handler_parse_empty_response():
    h = DeepgramHandler()
    result = h.parse_response({})
    assert result.raw_response == {"text": ""}
    assert result.duration_ms is None


# ---------------------------------------------------------------------------
# SelfHostedHandler
# ---------------------------------------------------------------------------


def test_self_hosted_handler_get_url():
    h = SelfHostedHandler()
    assert h.get_url("model", "http://stt") == "http://stt/v1/audio/transcriptions"


def test_self_hosted_handler_no_base_raises():
    h = SelfHostedHandler()
    with pytest.raises(HTTPException) as exc_info:
        h.get_url("model", None)
    assert exc_info.value.status_code == 503


def test_self_hosted_handler_build_request_multipart():
    h = SelfHostedHandler()
    req = STTRequest(
        file_bytes=b"AUDIO",
        file_name="test.wav",
        file_content_type="audio/wav",
        model="stt/whisper-large-v3-turbo",
    )
    content, files, headers = h.build_request(req, "whisper-large-v3-turbo", "")
    assert content is None
    assert files is not None
    assert files["file"] == ("test.wav", b"AUDIO", "audio/wav")
    assert files["model"] == (None, "whisper-large-v3-turbo")
    assert "Authorization" not in headers


def test_self_hosted_handler_parse_duration_ms():
    h = SelfHostedHandler()
    result = h.parse_response({"text": "hi", "duration_ms": 3000})
    assert result.duration_ms == 3000


def test_self_hosted_handler_parse_duration_seconds():
    h = SelfHostedHandler()
    result = h.parse_response({"text": "hi", "duration": 2.5})
    assert result.duration_ms == 2500
