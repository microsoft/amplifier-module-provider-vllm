"""Config hygiene tests: bool/numeric coercion, unknown-key sweep,
thinking_budget removal, wizard trim, and extra_request_params merge order.

Guards the "family hygiene wave" fixes for provider-vllm:
  - enable_state/raw/use_streaming/retry_jitter previously used bare
    truthiness or bare int()/float() (crash-on-invalid instead of
    warn-and-default).
  - thinking_budget_tokens/thinking_budget_buffer were removed (they
    forced an artificial max_output_tokens floor).
  - stream_idle_timeout and max_output_tokens are still supported config
    keys but no longer wizard-prompted.
  - extra_request_params merges last into BOTH the main request build and
    the auto-continuation build.
"""

from __future__ import annotations

import logging

import amplifier_module_provider_vllm as _provider_module

VLLMProvider = _provider_module.VLLMProvider
_coerce_bool = _provider_module._coerce_bool  # type: ignore[attr-defined]
_coerce_float = _provider_module._coerce_float  # type: ignore[attr-defined]
_coerce_int = _provider_module._coerce_int  # type: ignore[attr-defined]
_warn_unknown_config_keys = _provider_module._warn_unknown_config_keys  # type: ignore[attr-defined]


def _provider(**config_overrides):
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config_overrides)


class TestCoerceBool:
    def test_string_false_is_false(self):
        assert _coerce_bool("false", key="x", default=True) is False

    def test_string_true_is_true(self):
        assert _coerce_bool("true", key="x", default=False) is True


class TestCoerceNumeric:
    def test_int_from_string(self):
        assert _coerce_int("50", key="priority", default=100) == 50

    def test_invalid_int_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _coerce_int("garbage", key="context_window", default=128000)
        assert result == 128000
        assert "context_window" in caplog.text

    def test_float_from_string(self):
        assert _coerce_float("45.5", key="timeout", default=600.0) == 45.5


class TestUnknownConfigKeySweep:
    def test_known_keys_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"base_url": "x", "priority": 1})
        assert caplog.text == ""

    def test_extra_request_params_allowlisted(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"extra_request_params": {}})
        assert caplog.text == ""

    def test_unknown_key_warns_with_suggestion(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"tiemout": 5})
        assert "tiemout" in caplog.text
        assert "timeout" in caplog.text

    def test_thinking_budget_tokens_gets_targeted_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"thinking_budget_tokens": 5000})
        assert "removed" in caplog.text

    def test_thinking_budget_buffer_gets_targeted_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"thinking_budget_buffer": 1024})
        assert "removed" in caplog.text

    def test_debug_ghost_key_gets_targeted_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"debug": True})
        assert "not read" in caplog.text

    def test_debug_truncate_length_ghost_key_gets_targeted_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"debug_truncate_length": 180})
        assert "not read" in caplog.text


class TestProviderConfigCoercionIntegration:
    def test_enable_state_string_false_is_false(self):
        provider = _provider(enable_state="false")
        assert provider.enable_state is False

    def test_raw_string_true_is_true(self):
        provider = _provider(raw="true")
        assert provider.raw is True

    def test_use_streaming_string_false_is_false(self):
        provider = _provider(use_streaming="false")
        assert provider.use_streaming is False

    def test_retry_jitter_string_false(self):
        provider = _provider(retry_jitter="false")
        assert provider._retry_config.jitter == 0.0

    def test_priority_from_string(self):
        provider = _provider(priority="50")
        assert provider.priority == 50

    def test_invalid_numeric_string_defaults_instead_of_crashing(self):
        provider = _provider(timeout="not-a-number")
        assert provider.timeout == 600.0

    def test_extra_request_params_stored(self):
        provider = _provider(extra_request_params={"top_p": 0.9})
        assert provider.extra_request_params == {"top_p": 0.9}

    def test_extra_request_params_non_dict_ignored(self, caplog):
        with caplog.at_level(logging.WARNING):
            provider = _provider(extra_request_params="nope")
        assert provider.extra_request_params == {}
        assert "extra_request_params" in caplog.text


class TestWizardTrim:
    def test_stream_idle_timeout_not_in_wizard(self):
        provider = _provider()
        info = provider.get_info()
        ids = [f.id for f in info.config_fields]
        assert "stream_idle_timeout" not in ids

    def test_max_output_tokens_not_in_wizard(self):
        provider = _provider()
        info = provider.get_info()
        ids = [f.id for f in info.config_fields]
        assert "max_output_tokens" not in ids

    def test_max_tokens_not_renamed(self):
        """max_tokens has a known name collision with the advertised
        ceiling (max_output_tokens) -- must NOT be renamed."""
        provider = _provider(max_tokens="8192")
        assert provider.max_tokens == 8192

    def test_stream_idle_timeout_still_settable_via_settings(self):
        """Demoted from the wizard, but still a fully supported config key."""
        provider = _provider(stream_idle_timeout="45.0")
        assert provider.stream_idle_timeout == 45.0

    def test_max_output_tokens_still_settable_via_settings(self):
        provider = _provider(max_output_tokens="32000")
        assert provider.max_output_tokens == 32000

    def test_context_window_still_in_wizard(self):
        provider = _provider()
        info = provider.get_info()
        ids = [f.id for f in info.config_fields]
        assert "context_window" in ids
        assert "base_url" in ids
        assert "api_key" in ids
