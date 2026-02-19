"""In-memory LRU response cache with TTL."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion


def _cache_key(kwargs: dict[str, Any]) -> str:
    """Hash (model + messages + key params) to produce a cache key."""
    key_parts = {
        "model": kwargs.get("model"),
        "messages": kwargs.get("messages"),
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
        "max_completion_tokens": kwargs.get("max_completion_tokens"),
        "top_p": kwargs.get("top_p"),
        "tools": kwargs.get("tools"),
        "tool_choice": kwargs.get("tool_choice"),
        "response_format": kwargs.get("response_format"),
    }
    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache:
    """Thread-safe LRU cache with TTL for non-streaming responses."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[ChatCompletion, float]] = OrderedDict()
        self._lock = Lock()

    def get(self, kwargs: dict[str, Any]) -> ChatCompletion | None:
        key = _cache_key(kwargs)
        now = time.monotonic()
        with self._lock:
            if key in self._cache:
                response, expires_at = self._cache[key]
                if now < expires_at:
                    self._cache.move_to_end(key)
                    return response
                del self._cache[key]
        return None

    def put(self, kwargs: dict[str, Any], response: ChatCompletion, ttl: float | None = None) -> None:
        key = _cache_key(kwargs)
        expires_at = time.monotonic() + (ttl or self.default_ttl)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (response, expires_at)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)
