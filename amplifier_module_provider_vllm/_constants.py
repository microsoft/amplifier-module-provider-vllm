"""Constants for vLLM provider.

This module defines constants used across the vLLM provider implementation,
following the principle of single source of truth.
"""

# Metadata keys for vLLM Responses API state
# These keys are namespaced with "vllm:" to prevent collisions with other providers
METADATA_RESPONSE_ID = "vllm:response_id"
METADATA_STATUS = "vllm:status"
METADATA_INCOMPLETE_REASON = "vllm:incomplete_reason"
METADATA_REASONING_ITEMS = "vllm:reasoning_items"
METADATA_CONTINUATION_COUNT = "vllm:continuation_count"

# Default configuration values
DEFAULT_MODEL = "openai/gpt-oss-20b"  # Default model for vLLM
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_SUMMARY = "detailed"
DEFAULT_DEBUG_TRUNCATE_LENGTH = 180
DEFAULT_TIMEOUT = 600.0  # 10 minutes
DEFAULT_TRUNCATION = "auto"  # Automatic context management

# Maximum number of continuation attempts for incomplete responses
# This prevents infinite loops while being generous enough for legitimate large responses
MAX_CONTINUATION_ATTEMPTS = 5
