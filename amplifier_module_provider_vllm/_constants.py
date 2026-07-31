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

# Inter-chunk idle timeout for streaming responses (seconds).
#
# Bounds the wait for EVERY chunk on an established stream — including the
# FIRST one (a hang before the first chunk is the same failure mode).
# Remote vLLM endpoints behind hosted-GPU HTTPS proxies (RunPod et al.)
# routinely drop quiet connections without FIN, which previously left a
# stream hanging silently forever (observed: ~8.7 hours mid-turn).
#
# Why 300s and not something aggressive like 30s: this endpoint class
# legitimately has long time-to-first-token during prefill of 60-90k-token
# prompts (minutes, not seconds). The default must never false-positive on
# a healthy long prefill, while still guaranteeing a hung stream surfaces
# as a retryable error instead of hanging forever.
DEFAULT_STREAM_IDLE_TIMEOUT = 300.0  # 5 minutes

# Fallback model limits for downstream context managers.
# vLLM's /v1/models model cards expose the real context length
# (max_model_len), so context_window is normally discovered per model at
# runtime (see VLLMProvider._resolve_limits() in __init__.py). These
# defaults apply when the server does not report a limit -- e.g. behind a
# proxy that strips the field -- and are overridden by an explicit
# context_window / max_output_tokens config key or the
# VLLM_CONTEXT_WINDOW / VLLM_MAX_OUTPUT_TOKENS env vars.
# Note: DEFAULT_MAX_OUTPUT_TOKENS is the advertised model maximum (used for
# token budgeting), NOT the per-request completion cap (DEFAULT_MAX_TOKENS).
DEFAULT_CONTEXT_WINDOW = 128000
DEFAULT_MAX_OUTPUT_TOKENS = 32768

# Maximum number of continuation attempts for incomplete responses
# This prevents infinite loops while being generous enough for legitimate large responses
MAX_CONTINUATION_ATTEMPTS = 5
