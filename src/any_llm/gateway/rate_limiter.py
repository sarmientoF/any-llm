"""In-memory sliding-window rate limiter."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock


class SlidingWindowCounter:
    """Sliding window counter using a deque of timestamps."""

    def __init__(self, limit: int, window_sec: float = 60.0) -> None:
        self.limit = limit
        self.window_sec = window_sec
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def allow(self) -> tuple[bool, float]:
        """Check if a request is allowed.

        Returns:
            (allowed, retry_after_sec). retry_after is 0 if allowed.
        """
        now = time.monotonic()
        cutoff = now - self.window_sec

        with self._lock:
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) < self.limit:
                self._timestamps.append(now)
                return True, 0.0

            retry_after = self._timestamps[0] + self.window_sec - now
            return False, max(retry_after, 0.1)

    def peek(self) -> tuple[bool, float]:
        """Check headroom without recording a new timestamp."""
        now = time.monotonic()
        cutoff = now - self.window_sec
        with self._lock:
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) < self.limit:
                return True, 0.0
            retry_after = self._timestamps[0] + self.window_sec - now
            return False, max(retry_after, 0.1)

    def record_tokens(self, count: int) -> None:
        """Record token usage (for TPM tracking)."""
        now = time.monotonic()
        with self._lock:
            for _ in range(count):
                self._timestamps.append(now)


class RateLimiter:
    """Per-provider rate limiter with RPM and TPM counters."""

    def __init__(
        self,
        rate_limits: dict[str, dict[str, int]],
        default_rpm: int = 60,
        default_tpm: int = 100_000,
    ) -> None:
        self._rpm_counters: dict[str, SlidingWindowCounter] = {}
        self._tpm_counters: dict[str, SlidingWindowCounter] = {}
        self._rate_limits = rate_limits
        self._default_rpm = default_rpm
        self._default_tpm = default_tpm

    def _get_rpm(self, provider: str) -> SlidingWindowCounter:
        if provider not in self._rpm_counters:
            limits = self._rate_limits.get(provider, {})
            rpm = limits.get("rpm", self._default_rpm)
            self._rpm_counters[provider] = SlidingWindowCounter(rpm)
        return self._rpm_counters[provider]

    def _get_tpm(self, provider: str) -> SlidingWindowCounter:
        if provider not in self._tpm_counters:
            limits = self._rate_limits.get(provider, {})
            tpm = limits.get("tpm", self._default_tpm)
            self._tpm_counters[provider] = SlidingWindowCounter(tpm)
        return self._tpm_counters[provider]

    def check_rpm(self, provider: str) -> tuple[bool, float]:
        """Check if request is within RPM limit."""
        return self._get_rpm(provider).allow()

    def record_tokens(self, provider: str, count: int) -> None:
        """Record token usage against TPM limit."""
        self._get_tpm(provider).record_tokens(count)

    def check_tpm(self, provider: str) -> tuple[bool, float]:
        """Check current TPM headroom without recording a new request."""
        counter = self._get_tpm(provider)
        return counter.peek()
