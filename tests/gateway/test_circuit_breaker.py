"""Unit tests for circuit breaker."""

import time
from unittest.mock import patch

from any_llm.gateway.circuit_breaker import CircuitBreaker


def test_closed_by_default():
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.is_available("openai")


def test_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, cooldown_sec=9999)
    for _ in range(3):
        cb.record_failure("openai")
    assert not cb.is_available("openai")


def test_stays_closed_below_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure("openai")
    cb.record_failure("openai")
    assert cb.is_available("openai")


def test_success_resets_counter():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure("openai")
    cb.record_failure("openai")
    cb.record_success("openai")
    cb.record_failure("openai")
    cb.record_failure("openai")
    assert cb.is_available("openai")


def test_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=1, cooldown_sec=0.05)
    cb.record_failure("openai")
    assert not cb.is_available("openai")
    time.sleep(0.06)
    assert cb.is_available("openai")


def test_independent_providers():
    cb = CircuitBreaker(failure_threshold=2, cooldown_sec=9999)
    cb.record_failure("openai")
    cb.record_failure("openai")
    assert not cb.is_available("openai")
    assert cb.is_available("anthropic")


def test_get_status():
    cb = CircuitBreaker(failure_threshold=2, cooldown_sec=9999)
    cb.record_failure("openai")
    cb.record_failure("openai")
    cb.record_success("anthropic")

    status = cb.get_status()
    assert status["openai"]["status"] == "open"
    assert status["openai"]["consecutive_failures"] == 2
    assert status["anthropic"]["status"] == "closed"
    assert status["anthropic"]["consecutive_failures"] == 0


def test_get_status_empty():
    cb = CircuitBreaker()
    assert cb.get_status() == {}
