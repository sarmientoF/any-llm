"""Unit tests for in-memory rate limiter."""

from any_llm.gateway.rate_limiter import RateLimiter, SlidingWindowCounter


def test_sliding_window_allows_within_limit():
    counter = SlidingWindowCounter(limit=3, window_sec=60.0)
    for _ in range(3):
        allowed, _ = counter.allow()
        assert allowed


def test_sliding_window_blocks_over_limit():
    counter = SlidingWindowCounter(limit=2, window_sec=60.0)
    counter.allow()
    counter.allow()
    allowed, retry_after = counter.allow()
    assert not allowed
    assert retry_after > 0


def test_rate_limiter_rpm_default():
    limiter = RateLimiter(rate_limits={}, default_rpm=3)
    for _ in range(3):
        allowed, _ = limiter.check_rpm("openai")
        assert allowed
    allowed, retry_after = limiter.check_rpm("openai")
    assert not allowed
    assert retry_after > 0


def test_rate_limiter_per_provider_config():
    limiter = RateLimiter(
        rate_limits={"openai": {"rpm": 2, "tpm": 100}},
        default_rpm=100,
    )
    limiter.check_rpm("openai")
    limiter.check_rpm("openai")
    allowed, _ = limiter.check_rpm("openai")
    assert not allowed

    # Different provider uses default
    for _ in range(100):
        allowed, _ = limiter.check_rpm("anthropic")
        assert allowed


def test_rate_limiter_independent_providers():
    limiter = RateLimiter(rate_limits={}, default_rpm=2)
    limiter.check_rpm("openai")
    limiter.check_rpm("openai")
    # openai exhausted
    allowed, _ = limiter.check_rpm("openai")
    assert not allowed
    # anthropic still has quota
    allowed, _ = limiter.check_rpm("anthropic")
    assert allowed


def test_record_tokens():
    counter = SlidingWindowCounter(limit=10, window_sec=60.0)
    counter.record_tokens(5)
    for _ in range(5):
        allowed, _ = counter.allow()
        assert allowed
    allowed, _ = counter.allow()
    assert not allowed
