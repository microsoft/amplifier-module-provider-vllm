"""Tests for vLLM RetryConfig alignment with Rust field names (task-11).

Verifies that:
- RetryConfig uses `initial_delay` (not `min_delay`)
- RetryConfig uses `jitter` as bool (not float)
- No jitter bool/float compat code remains
"""

import inspect

from amplifier_module_provider_vllm import VLLMProvider


def test_retry_config_uses_initial_delay():
    """RetryConfig should be constructed with initial_delay, not min_delay."""
    provider = VLLMProvider(
        base_url="http://localhost:8000/v1",
        config={"min_retry_delay": 2.5},
    )
    # initial_delay should match the config value
    assert provider._retry_config.initial_delay == 2.5


def test_retry_config_jitter_is_bool_true_by_default():
    """RetryConfig jitter should be a bool (True by default), not a float."""
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config={})
    # Default retry_jitter should be True -> jitter stored as 0.2 internally
    # With Rust RetryConfig, jitter=True produces 0.2 internally
    assert provider._retry_config.jitter == 0.2  # True -> 0.2 internal


def test_retry_config_jitter_false():
    """RetryConfig jitter=False should disable jitter."""
    provider = VLLMProvider(
        base_url="http://localhost:8000/v1",
        config={"retry_jitter": False},
    )
    assert provider._retry_config.jitter == 0.0  # False -> 0.0


def test_retry_config_jitter_true():
    """RetryConfig jitter=True should enable jitter."""
    provider = VLLMProvider(
        base_url="http://localhost:8000/v1",
        config={"retry_jitter": True},
    )
    assert provider._retry_config.jitter == 0.2  # True -> 0.2


def test_no_jitter_compat_code_in_init():
    """The __init__ method should NOT contain jitter bool/float compat code.

    Specifically, there should be no:
    - jitter_val variable
    - isinstance(jitter_val, bool) check
    """
    source = inspect.getsource(VLLMProvider.__init__)
    assert "jitter_val" not in source, "jitter_val compat variable should be removed"
    for line in source.splitlines():
        if "isinstance" in line:
            assert "jitter" not in line, "isinstance check for jitter should be removed"


def test_no_min_delay_in_retry_config_construction():
    """RetryConfig construction should use initial_delay=, not min_delay=."""
    source = inspect.getsource(VLLMProvider.__init__)
    # Should NOT have min_delay= in the RetryConfig call
    assert "min_delay=" not in source, (
        "RetryConfig should use initial_delay= not min_delay="
    )


def test_no_deprecation_warnings_from_retry_config(recwarn):
    """Creating VLLMProvider should not trigger deprecation warnings from RetryConfig."""
    VLLMProvider(base_url="http://localhost:8000/v1", config={})
    deprecation_warnings = [
        w for w in recwarn if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 0, (
        f"Got deprecation warnings: {[str(w.message) for w in deprecation_warnings]}"
    )
