"""Tests for configurable context_window / max_output_tokens limits.

vLLM does not expose context length via /v1/models, so the provider
advertises defaults to downstream context managers (which derive their
token budget from get_info().defaults or get_model_info()). These limits
must be configurable per provider instance so deployments can match the
real limits of their endpoint instead of being capped by hardcoded values.

Regression context: the hardcoded defaults previously disagreed
(ProviderInfo.defaults said max_output_tokens=128000 while list_models()
said 32768), and the 128000 value capped the effective input budget at
~59k tokens on endpoints that handle 120k+ prompts.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from amplifier_module_provider_vllm import VLLMProvider

BASE_URL = "http://localhost:8000/v1"


def _mock_client(model_ids: list[str] | None = None) -> MagicMock:
    """Build a mock AsyncOpenAI client whose models.list() is awaitable."""
    client = MagicMock()
    data = [SimpleNamespace(id=model_id) for model_id in (model_ids or ["test-model"])]
    client.models.list = AsyncMock(return_value=SimpleNamespace(data=data))
    return client


class TestDefaults:
    def test_get_info_defaults_are_reconciled(self):
        """Defaults: context_window=128000, max_output_tokens=32768.

        Asserts the reconciliation of the previously inconsistent hardcoded
        values (128000 in ProviderInfo.defaults vs 32768 in ModelInfo):
        the saner ModelInfo value wins everywhere.
        """
        provider = VLLMProvider(base_url=BASE_URL, config={})
        defaults = provider.get_info().defaults

        assert defaults["context_window"] == 128000
        assert defaults["max_output_tokens"] == 32768

    def test_provider_attributes_default(self):
        provider = VLLMProvider(base_url=BASE_URL, config={})

        assert provider.context_window == 128000
        assert provider.max_output_tokens == 32768


class TestConfigOverride:
    def test_get_info_defaults_reflect_config(self):
        provider = VLLMProvider(
            base_url=BASE_URL,
            config={"context_window": 200000, "max_output_tokens": 65536},
        )
        defaults = provider.get_info().defaults

        assert defaults["context_window"] == 200000
        assert defaults["max_output_tokens"] == 65536

    @pytest.mark.asyncio
    async def test_list_models_reflects_config(self):
        provider = VLLMProvider(
            client=_mock_client(),
            config={"context_window": 200000, "max_output_tokens": 65536},
        )
        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].context_window == 200000
        assert models[0].max_output_tokens == 65536

    def test_string_values_coerced_to_int(self):
        """Config values arriving as strings (e.g. from env/YAML) coerce to int."""
        provider = VLLMProvider(
            base_url=BASE_URL,
            config={"context_window": "200000", "max_output_tokens": "65536"},
        )

        assert provider.context_window == 200000
        assert provider.max_output_tokens == 65536
        assert provider.get_info().defaults["context_window"] == 200000
        assert provider.get_info().defaults["max_output_tokens"] == 65536

    def test_max_tokens_request_cap_untouched_by_overrides(self):
        """max_output_tokens (advertised model max) is distinct from
        max_tokens (per-request completion cap) — overriding one must not
        affect the other."""
        provider = VLLMProvider(
            base_url=BASE_URL,
            config={"max_output_tokens": 65536},
        )

        assert provider.max_output_tokens == 65536
        assert provider.max_tokens != 65536


class TestEnvFallback:
    def test_env_vars_used_when_config_absent(self, monkeypatch):
        monkeypatch.setenv("VLLM_CONTEXT_WINDOW", "150000")
        monkeypatch.setenv("VLLM_MAX_OUTPUT_TOKENS", "40000")
        provider = VLLMProvider(base_url=BASE_URL, config={})

        assert provider.context_window == 150000
        assert provider.max_output_tokens == 40000

    def test_config_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_CONTEXT_WINDOW", "150000")
        provider = VLLMProvider(base_url=BASE_URL, config={"context_window": 200000})

        assert provider.context_window == 200000


class TestConfigFields:
    def test_config_fields_include_context_limits(self):
        provider = VLLMProvider(base_url=BASE_URL, config={})
        fields = {field.id: field for field in provider.get_info().config_fields}

        assert "context_window" in fields
        assert "max_output_tokens" in fields
        assert fields["context_window"].env_var == "VLLM_CONTEXT_WINDOW"
        assert fields["max_output_tokens"].env_var == "VLLM_MAX_OUTPUT_TOKENS"
        assert fields["context_window"].required is False
        assert fields["max_output_tokens"].required is False
        assert fields["context_window"].default == "128000"
        assert fields["max_output_tokens"].default == "32768"


class TestGetModelInfo:
    """context managers (e.g. context-simple) prefer provider.get_model_info()
    over get_info().defaults when present — it must report the configured limits."""

    def test_returns_configured_limits(self):
        provider = VLLMProvider(
            base_url=BASE_URL,
            config={"context_window": 200000, "max_output_tokens": 65536},
        )
        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.context_window == 200000
        assert model_info.max_output_tokens == 65536

    def test_returns_defaults_when_unconfigured(self):
        provider = VLLMProvider(base_url=BASE_URL, config={})
        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.context_window == 128000
        assert model_info.max_output_tokens == 32768
