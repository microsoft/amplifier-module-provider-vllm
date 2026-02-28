"""Tests for cost tier metadata in vLLM provider.

Validates:
1. list_models() adds cost_per_input/output_token = 0.0 (self-hosted = free)
2. list_models() adds metadata={"cost_tier": "free"} to all models
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_vllm import VLLMProvider


class TestVLLMCostTier:
    """Verify all vLLM models get free cost tier and zero cost."""

    @pytest.fixture
    def provider(self):
        return VLLMProvider(base_url="http://localhost:8000/v1")

    def _make_mock_model(self, model_id: str):
        m = MagicMock()
        m.id = model_id
        return m

    @pytest.mark.asyncio
    async def test_model_has_free_cost_tier(self, provider):
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_model("Qwen/Qwen3-Coder-480B-A35B")]

        provider._client = MagicMock()
        provider._client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()

        assert len(models) == 1
        model = models[0]
        assert model.metadata == {"cost_tier": "free"}

    @pytest.mark.asyncio
    async def test_model_has_zero_cost(self, provider):
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_model("meta-llama/Llama-3.3-70B")]

        provider._client = MagicMock()
        provider._client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()

        assert len(models) == 1
        model = models[0]
        assert model.cost_per_input_token == 0.0
        assert model.cost_per_output_token == 0.0
