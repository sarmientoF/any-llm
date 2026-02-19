"""Unit tests for in-memory response cache."""

import time

from any_llm.gateway.cache import ResponseCache, _cache_key
from any_llm.types.completion import ChatCompletion, CompletionUsage


def _make_response(model: str = "test") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1234567890,
        model=model,
        choices=[],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def test_cache_put_and_get():
    cache = ResponseCache(max_size=10, default_ttl=60.0)
    kwargs = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    response = _make_response()
    cache.put(kwargs, response)
    assert cache.get(kwargs) == response


def test_cache_miss():
    cache = ResponseCache()
    kwargs = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    assert cache.get(kwargs) is None


def test_cache_ttl_expiry():
    cache = ResponseCache(default_ttl=0.01)
    kwargs = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    cache.put(kwargs, _make_response())
    time.sleep(0.02)
    assert cache.get(kwargs) is None


def test_cache_lru_eviction():
    cache = ResponseCache(max_size=2, default_ttl=60.0)
    kwargs1 = {"model": "m1", "messages": []}
    kwargs2 = {"model": "m2", "messages": []}
    kwargs3 = {"model": "m3", "messages": []}

    cache.put(kwargs1, _make_response("m1"))
    cache.put(kwargs2, _make_response("m2"))
    cache.put(kwargs3, _make_response("m3"))

    # m1 should be evicted
    assert cache.get(kwargs1) is None
    assert cache.get(kwargs2) is not None
    assert cache.get(kwargs3) is not None


def test_cache_clear():
    cache = ResponseCache()
    kwargs = {"model": "m1", "messages": []}
    cache.put(kwargs, _make_response())
    assert cache.size == 1
    cache.clear()
    assert cache.size == 0


def test_cache_size():
    cache = ResponseCache()
    assert cache.size == 0
    cache.put({"model": "m1", "messages": []}, _make_response())
    assert cache.size == 1


def test_cache_key_deterministic():
    kwargs = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.7}
    assert _cache_key(kwargs) == _cache_key(kwargs)


def test_cache_key_differs_by_model():
    k1 = {"model": "openai/gpt-4o", "messages": []}
    k2 = {"model": "anthropic/claude-3", "messages": []}
    assert _cache_key(k1) != _cache_key(k2)


def test_cache_key_differs_by_messages():
    k1 = {"model": "m", "messages": [{"role": "user", "content": "a"}]}
    k2 = {"model": "m", "messages": [{"role": "user", "content": "b"}]}
    assert _cache_key(k1) != _cache_key(k2)
