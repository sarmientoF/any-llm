"""Unit tests for model alias resolution and /v1/models endpoint."""

from __future__ import annotations

from unittest.mock import patch

from any_llm.gateway.config import GatewayConfig


def _make_config(**overrides) -> GatewayConfig:
    defaults = {
        "database_url": "postgresql://localhost/test",
        "master_key": "test",
        "auto_migrate": False,
        "model_aliases": {
            "gpt4": "openai/gpt-4o",
            "claude": "anthropic/claude-3-sonnet-20240229",
            "fast": "groq/llama-3.3-70b",
        },
        "providers": {"openai": {"api_key": "test"}, "anthropic": {"api_key": "test"}},
    }
    defaults.update(overrides)
    return GatewayConfig(**defaults)


def test_alias_resolution():
    config = _make_config()
    model = "gpt4"
    resolved = config.model_aliases.get(model, model)
    assert resolved == "openai/gpt-4o"


def test_alias_passthrough_unknown():
    config = _make_config()
    model = "openai/gpt-4o"
    resolved = config.model_aliases.get(model, model)
    assert resolved == "openai/gpt-4o"


def test_alias_config_empty_by_default():
    config = GatewayConfig(
        database_url="postgresql://localhost/test",
        master_key="test",
        auto_migrate=False,
    )
    assert config.model_aliases == {}


def test_models_endpoint():
    """Test /v1/models returns aliases and providers."""
    from fastapi.testclient import TestClient

    from any_llm.gateway.routes.models import router

    # Minimal FastAPI app with just the models router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    config = _make_config()

    # Patch get_config to return our test config
    with patch("any_llm.gateway.routes.models.get_config", return_value=config):
        # Override the dependency
        from any_llm.gateway.auth.dependencies import get_config

        app.dependency_overrides[get_config] = lambda: config

        with TestClient(app) as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"

            aliases = [m for m in data["data"] if m["type"] == "alias"]
            assert len(aliases) == 3
            alias_ids = {a["id"] for a in aliases}
            assert alias_ids == {"gpt4", "claude", "fast"}

            providers = [m for m in data["data"] if m["type"] == "provider"]
            provider_ids = {p["id"] for p in providers}
            assert "openai" in provider_ids
            assert "anthropic" in provider_ids


def test_models_alias_targets():
    """Test alias → target mappings are correct."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from any_llm.gateway.auth.dependencies import get_config
    from any_llm.gateway.routes.models import router

    app = FastAPI()
    app.include_router(router)
    config = _make_config()
    app.dependency_overrides[get_config] = lambda: config

    with TestClient(app) as client:
        data = client.get("/v1/models").json()
        alias_map = {m["id"]: m["target"] for m in data["data"] if m["type"] == "alias"}
        assert alias_map["gpt4"] == "openai/gpt-4o"
        assert alias_map["claude"] == "anthropic/claude-3-sonnet-20240229"
        assert alias_map["fast"] == "groq/llama-3.3-70b"
