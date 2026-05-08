"""Tests for _cost.py: compute_cost() for vLLM (self-hosted, always None).

Since vLLM is self-hosted, compute_cost() must always return None —
cost is indeterminate, not $0.00.

Covers:
  (a) Any model always returns None
  (b) Extra kwargs are accepted and ignored
  (c) None != Decimal('0') — indeterminate is distinct from free
"""

from decimal import Decimal

from amplifier_module_provider_vllm._cost import compute_cost


# ---------------------------------------------------------------------------
# (a) Any model always returns None
# ---------------------------------------------------------------------------
def test_always_returns_none():
    assert compute_cost("any-model", prompt_tokens=1_000_000) is None


# ---------------------------------------------------------------------------
# (b) Extra kwargs are accepted without raising
# ---------------------------------------------------------------------------
def test_accepts_any_kwargs():
    result = compute_cost(
        "some-model",
        prompt_tokens=500,
        completion_tokens=100,
        total_tokens=600,
    )
    assert result is None


# ---------------------------------------------------------------------------
# (c) None != Decimal('0') — indeterminate is distinct from free
# ---------------------------------------------------------------------------
def test_none_distinct_from_zero():
    result = compute_cost("some-model", prompt_tokens=0)
    assert result is None
    assert result != Decimal("0")
