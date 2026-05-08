"""Tests for VLLMProvider.close() and mount() cleanup bug fix."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_vllm import VLLMProvider, mount


class TestVLLMProviderClose:
    """Tests for the async close() method on VLLMProvider."""

    @pytest.mark.asyncio
    async def test_close_calls_client_close_when_initialized(self):
        """close() should call _client.close() and nil the reference."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1")
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._client = mock_client

        await provider.close()

        mock_client.close.assert_awaited_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_is_safe_when_client_is_none(self):
        """close() should not crash when _client is None."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1")
        assert provider._client is None

        await provider.close()  # Should not raise

        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_can_be_called_twice(self):
        """close() called twice should only close the client once."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1")
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._client = mock_client

        await provider.close()
        await provider.close()

        mock_client.close.assert_awaited_once()
        assert provider._client is None


class TestMountCleanupBugFix:
    """Tests that mount() cleanup does not trigger lazy client initialization."""

    @pytest.mark.asyncio
    async def test_mount_cleanup_does_not_trigger_lazy_init(self):
        """Calling the mount cleanup should not create a client via the .client property."""

        class FakeHooks:
            def register(self, event, handler):
                pass

            async def emit(self, event, data):
                pass

        class FakeCoordinator:
            mounted_provider = None
            hooks = FakeHooks()

            async def mount(self, slot, provider, name=None):
                self.mounted_provider = provider

            def register_contributor(self, channel, name, callback):
                pass

        coordinator = FakeCoordinator()

        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://localhost:8000/v1"}):
            cleanup_ref = await mount(coordinator)

        provider = coordinator.mounted_provider
        assert provider is not None, "mount() should have set mounted_provider"
        assert provider._client is None, (
            "Client should not be initialized before cleanup"
        )

        # Calling cleanup should not trigger lazy client init or raise
        await cleanup_ref()

        assert provider._client is None, (
            "Bug: cleanup triggered lazy client initialization via .client property"
        )
