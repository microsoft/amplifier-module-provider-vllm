"""Structural validation tests for vllm provider.

Inherits authoritative tests from amplifier-core.
"""

from amplifier_core.validation.structural import ProviderStructuralTests


class TestVLLMProviderStructural(ProviderStructuralTests):
    """Run standard provider structural tests for vllm.

    All tests from ProviderStructuralTests run automatically.
    Add module-specific structural tests below if needed.
    """
