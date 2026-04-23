"""Tests for VLLMProvider.list_models() context_window fallback logic.

Validates three getattr branches for max_model_len on the model object
returned by client.models.list():

  a. attribute present with real value  → value wins over fallback
  b. attribute absent entirely          → fallback 128000 fires
  c. attribute present but None         → fallback 128000 fires (validates `or 128000` fix)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_vllm import VLLMProvider


def _make_provider_with_models(models_data: list) -> VLLMProvider:
    """Create VLLMProvider with client.models.list() mocked to return models_data."""
    mock_client = MagicMock()
    mock_client.models.list = AsyncMock(return_value=SimpleNamespace(data=models_data))
    return VLLMProvider(client=mock_client, config={})


class TestListModelsContextWindow:
    """Tests for max_model_len → context_window getattr fallback branches."""

    @pytest.mark.asyncio
    async def test_max_model_len_present_with_value(self):
        """Model WITH max_model_len=8192 → context_window == 8192 (actual value wins)."""
        model = SimpleNamespace(id="test-model", max_model_len=8192)
        provider = _make_provider_with_models([model])

        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].context_window == 8192

    @pytest.mark.asyncio
    async def test_max_model_len_absent_uses_fallback(self):
        """Model WITHOUT max_model_len attribute → context_window == 128000 (fallback fires)."""
        # SimpleNamespace with only .id — no max_model_len attribute at all
        model = SimpleNamespace(id="test-model")
        provider = _make_provider_with_models([model])

        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].context_window == 128000

    @pytest.mark.asyncio
    async def test_max_model_len_present_but_none_uses_fallback(self):
        """Model WITH max_model_len=None → context_window == 128000 (validates `or 128000` fix).

        Without the `or 128000` fix, getattr returns None (the attribute exists, so
        the default never fires), and None propagates into ModelInfo.context_window
        causing a validation error.  This test would fail against the pre-fix code.
        """
        model = SimpleNamespace(id="test-model", max_model_len=None)
        provider = _make_provider_with_models([model])

        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].context_window == 128000
