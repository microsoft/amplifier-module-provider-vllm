"""Behavioral tests for vllm provider.

Inherits authoritative tests from amplifier-core.
"""

import pytest

from amplifier_core.validation.behavioral import ProviderBehaviorTests


class TestVllmProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for vllm.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """Override to mark this inherited test 'live'.

        ProviderBehaviorTests.test_list_models_returns_list calls
        provider_module.list_models(), which makes a real network call
        to the configured vLLM/OpenAI-compatible endpoint -- it cannot
        pass in CI without a reachable server. Deselected in CI via
        `-m "not live"`; run locally against a real vLLM server to
        validate.
        """
        await super().test_list_models_returns_list(provider_module)
