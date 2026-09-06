"""Tests for VLLMProvider.close() and mount() cleanup bug fix."""

import asyncio
import logging
import os
import time
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

    @pytest.mark.asyncio
    async def test_close_is_bounded_when_client_close_never_returns(self, caplog):
        """A client whose close() never returns must not hang cleanup.

        Regression guard: close() previously awaited ``self._client.close()``
        with no ceiling, and mount()'s cleanup() awaits close() directly --
        so a wedged httpx transport (a self-hosted vLLM server behind a
        flaky gateway is exactly this case) hung session cleanup for the
        whole process.
        """
        provider = VLLMProvider(
            base_url="http://localhost:8000/v1", config={"close_timeout": 0.05}
        )
        assert provider.close_timeout == 0.05

        release = asyncio.Event()

        class _UnclosableClient:
            async def close(self):
                # Never returns until the test explicitly releases it.
                await release.wait()

        provider._client = _UnclosableClient()  # type: ignore[assignment]

        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            await provider.close()  # must not raise, must not hang
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, f"close() took {elapsed:.2f}s; expected ~0.05s"
        assert "did not complete within" in caplog.text
        assert "abandoning client" in caplog.text
        assert "vllm" in caplog.text
        # Client reference dropped so the lazy-init property can rebuild.
        assert provider._client is None

        # Let the abandoned close task finish so the loop shuts down clean.
        release.set()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_close_normal_client_logs_no_warning(self, caplog):
        """A well-behaved client closes once, quietly, and is released."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1")
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._client = mock_client

        with caplog.at_level(logging.WARNING):
            await provider.close()

        mock_client.close.assert_awaited_once()
        assert caplog.text == ""
        assert provider._client is None

    def test_close_timeout_defaults_to_five_seconds(self):
        """Unconfigured providers get the 5.0s default ceiling."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1")
        assert provider.close_timeout == 5.0

    def test_close_timeout_coerces_string_and_falls_back_on_garbage(self, caplog):
        """settings.yaml strings coerce; garbage warns and uses the default."""
        provider = VLLMProvider(
            base_url="http://localhost:8000/v1", config={"close_timeout": "2.5"}
        )
        assert provider.close_timeout == 2.5

        with caplog.at_level(logging.WARNING):
            provider = VLLMProvider(
                base_url="http://localhost:8000/v1",
                config={"close_timeout": "not-a-number"},
            )
        assert provider.close_timeout == 5.0
        assert "close_timeout" in caplog.text


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
