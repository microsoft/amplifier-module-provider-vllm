"""Tests for _accumulate hook and register_contributor in mount().

Since vLLM is self-hosted, cost is always indeterminate (None).
Verifies that mount() registers:
  - an `llm:response` hook (_accumulate) that ignores None cost events
  - a lazy contributor callback on session.cost channel under name 'provider-vllm'
  - the contributor always returns None (cost is never accumulated)
"""

import pytest

from amplifier_module_provider_vllm import mount


# ---------------------------------------------------------------------------
# Mock coordinator fixture
# ---------------------------------------------------------------------------


class _MockHooks:
    def __init__(self):
        self._handlers: dict = {}

    def register(self, event: str, handler) -> None:
        self._handlers[event] = handler

    async def emit(self, event: str, data: dict) -> None:
        if event in self._handlers:
            await self._handlers[event](event, data)


class _MockCoordinator:
    def __init__(self):
        self.hooks = _MockHooks()
        self.registered_hooks = self.hooks._handlers  # shared reference
        self.registered_contributors: dict = {}

    async def mount(self, *args, **kwargs) -> None:
        pass

    def register_contributor(self, channel: str, name: str, callback) -> None:
        self.registered_contributors[(channel, name)] = callback


@pytest.fixture
def mock_coordinator():
    return _MockCoordinator()


# ---------------------------------------------------------------------------
# test_contributor_registered_at_mount
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_registered_at_mount(mock_coordinator):
    """mount() must register a contributor on ('session.cost', 'provider-vllm')."""
    await mount(mock_coordinator, config={})
    assert ("session.cost", "provider-vllm") in mock_coordinator.registered_contributors


# ---------------------------------------------------------------------------
# test_contributor_returns_none_before_any_calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_returns_none_before_any_calls(mock_coordinator):
    """Contributor callback returns None when no llm:response events have fired."""
    await mount(mock_coordinator, config={})
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-vllm")
    ]
    assert callback() is None


# ---------------------------------------------------------------------------
# test_contributor_always_returns_none_after_events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_always_returns_none_after_events(mock_coordinator):
    """_accumulate ignores None cost events; contributor stays None (self-hosted = indeterminate)."""
    await mount(mock_coordinator, config={})

    accumulate = mock_coordinator.registered_hooks["llm:response"]
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-vllm")
    ]

    # vLLM always emits cost_usd=None
    await accumulate("llm:response", {"provider": "vllm", "usage": {"cost_usd": None}})
    await accumulate("llm:response", {"provider": "vllm", "usage": {"cost_usd": None}})

    assert callback() is None, (
        "Contributor should always return None for self-hosted vLLM"
    )


# ---------------------------------------------------------------------------
# test_accumulate_hook_registered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accumulate_hook_registered(mock_coordinator):
    """mount() registers an llm:response hook for the accumulator."""
    await mount(mock_coordinator, config={})
    assert "llm:response" in mock_coordinator.registered_hooks
