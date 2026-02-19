"""Per-provider circuit breaker."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class _ProviderState:
    consecutive_failures: int = 0
    is_open: bool = False
    opened_at: float = 0.0


@dataclass
class CircuitBreaker:
    """Track consecutive failures per provider, open circuit after threshold."""

    failure_threshold: int = 5
    cooldown_sec: float = 60.0
    _states: dict[str, _ProviderState] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def _get_state(self, provider: str) -> _ProviderState:
        if provider not in self._states:
            self._states[provider] = _ProviderState()
        return self._states[provider]

    def record_success(self, provider: str) -> None:
        with self._lock:
            state = self._get_state(provider)
            state.consecutive_failures = 0
            state.is_open = False

    def record_failure(self, provider: str) -> None:
        with self._lock:
            state = self._get_state(provider)
            state.consecutive_failures += 1
            if state.consecutive_failures >= self.failure_threshold:
                state.is_open = True
                state.opened_at = time.monotonic()

    def is_available(self, provider: str) -> bool:
        with self._lock:
            state = self._get_state(provider)
            if not state.is_open:
                return True
            # Check if cooldown has elapsed (half-open)
            if time.monotonic() - state.opened_at >= self.cooldown_sec:
                return True
            return False

    def get_status(self) -> dict[str, dict[str, object]]:
        with self._lock:
            result: dict[str, dict[str, object]] = {}
            for provider, state in self._states.items():
                now = time.monotonic()
                is_open = state.is_open and (now - state.opened_at < self.cooldown_sec)
                result[provider] = {
                    "status": "open" if is_open else "closed",
                    "consecutive_failures": state.consecutive_failures,
                }
            return result
