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

# BEHAVIOR CHANGE (see README "Truncation" section): this default was "auto"
# prior to this change. With truncation="auto", vLLM's Responses API silently
# drops the oldest input content when a prompt exceeds the model's context
# window -- HTTP 200, no error, no warning, and the model answers from
# whatever survived. Verified live against a direct vLLM endpoint (glm-5.2,
# real max_model_len=131072): a ~150k-token prompt with truncation="auto"
# returned HTTP 200 with usage.input_tokens=131056 (~19k tokens silently
# discarded), while the identical prompt with truncation="disabled" returned
# a clear HTTP 400 naming the exact limit.
#
# OpenAI's own Responses API defaults truncation to "disabled" for the same
# reason: a caller that cannot detect data loss cannot recover from it, and a
# loud, immediate 400 is strictly more useful than a model silently reasoning
# over a truncated prompt. Operators who want the old auto-truncating
# behavior (e.g. because their context manager doesn't yet cap prompt size)
# can opt back in by setting `truncation: "auto"` explicitly in provider
# config -- see _maybe_warn_truncated_input() below, which logs a warning
# when "auto" truncation is actually observed to have kicked in, so the
# opt-in is never silent either.
#
# Sent EXPLICITLY rather than omitted, which is a deliberate divergence from
# provider-openai (which leaves its DEFAULT_TRUNCATION as None so the field is
# dropped from the request). On vLLM, omitting is not equivalent to sending
# "disabled". vLLM declares the field as
# `truncation: Literal["auto", "disabled"] | None = "disabled"` and every
# consumption site tests `!= "disabled"` -- so an omitted field parses to the
# "disabled" default and fails loud, but an explicit JSON `null` is NOT
# "disabled" and takes the truncating branch. Any layer between us and the
# server that serializes unset optionals as `null` rather than omitting them
# would therefore silently re-enable the exact data loss this default exists
# to prevent. The literal string is the only value that is unambiguously safe
# on every serialization path, and unlike on OpenAI there is no prompt-cache
# prefix cost to sending it, since vLLM's prefix caching keys on prompt tokens
# rather than request parameters.
DEFAULT_TRUNCATION = "disabled"

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
