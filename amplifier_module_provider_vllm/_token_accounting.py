"""Token accounting for vLLM GPT-OSS models using Harmony tokenizer.

vLLM returns zero usage values for GPT-OSS models when using the Responses API.
This module computes accurate input tokens and approximate output tokens using
the official Harmony tokenizer which matches GPT-OSS-20B's training format
(roles, channels, separators, etc.).

Following the "bricks and studs" philosophy, this is a self-contained module that
can be regenerated independently of the main provider code.

Key limitations:
- Output token count only reflects visible 'final' text, not hidden chain-of-thought
  channels (requires direct vLLM-Python call with raw token IDs, not possible via REST)
- This provides the best possible accuracy without modifying vLLM's API layer
- Requires vocab files: automatically downloads to ~/.amplifier/vocab/ on first use
"""

import logging
import os
from pathlib import Path
from typing import Any

from openai_harmony import Conversation
from openai_harmony import DeveloperContent
from openai_harmony import HarmonyEncodingName
from openai_harmony import Message
from openai_harmony import Role
from openai_harmony import SystemContent
from openai_harmony import load_harmony_encoding

logger = logging.getLogger(__name__)

# Global Harmony encoder (loaded once per process)
_HARMONY_ENCODING = None
_VOCAB_SETUP_ATTEMPTED = False


def _ensure_vocab_files() -> bool:
    """Ensure required vocab files exist, auto-download if needed.

    Downloads vocab files to ~/.amplifier/vocab/ on first use.
    Sets TIKTOKEN_ENCODINGS_BASE to point to this directory.

    Returns:
        True if vocab files are available, False otherwise
    """
    global _VOCAB_SETUP_ATTEMPTED

    # Only attempt once per process
    if _VOCAB_SETUP_ATTEMPTED:
        return os.environ.get("TIKTOKEN_ENCODINGS_BASE") is not None

    _VOCAB_SETUP_ATTEMPTED = True

    # Check if already configured
    if "TIKTOKEN_ENCODINGS_BASE" in os.environ:
        logger.debug("[TOKEN_ACCOUNTING] TIKTOKEN_ENCODINGS_BASE already set, skipping download")
        return True

    # Setup vocab directory in ~/.amplifier/vocab/
    vocab_dir = Path.home() / ".amplifier" / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    required_files = {
        "o200k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    }

    # Check if files already exist
    all_exist = all((vocab_dir / filename).exists() for filename in required_files)

    if not all_exist:
        logger.info("[TOKEN_ACCOUNTING] Downloading Harmony vocab files to ~/.amplifier/vocab/...")

        try:
            import urllib.request

            for filename, url in required_files.items():
                filepath = vocab_dir / filename
                if not filepath.exists():
                    logger.debug(f"[TOKEN_ACCOUNTING] Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                    logger.info(f"[TOKEN_ACCOUNTING] Downloaded {filename} ({filepath.stat().st_size} bytes)")

        except Exception as e:
            logger.warning(
                f"[TOKEN_ACCOUNTING] Failed to download vocab files: {e}. "
                "Token accounting will return zeros. "
                "See README for manual setup instructions."
            )
            return False

    # Set environment variable for this process
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(vocab_dir)
    logger.info(f"[TOKEN_ACCOUNTING] Using vocab files from {vocab_dir}")
    return True


def _get_harmony_encoding():
    """Get or load the Harmony encoder (lazy initialization).

    Returns:
        HarmonyEncoding if successful, None if vocab files unavailable
    """
    global _HARMONY_ENCODING

    if _HARMONY_ENCODING is None:
        # Ensure vocab files are available
        if not _ensure_vocab_files():
            logger.warning("[TOKEN_ACCOUNTING] Vocab files not available, token accounting disabled")
            return None

        try:
            _HARMONY_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            logger.info("[TOKEN_ACCOUNTING] Loaded Harmony GPT-OSS encoder")
        except Exception as e:
            logger.warning(
                f"[TOKEN_ACCOUNTING] Failed to load Harmony encoder: {e}. "
                "Token accounting disabled. Check vocab files in ~/.amplifier/vocab/"
            )
            return None

    return _HARMONY_ENCODING


def should_apply_token_accounting(model: str) -> bool:
    """Check if token accounting should be applied for this model.

    Args:
        model: Model name from params

    Returns:
        True if this is a GPT-OSS model needing token accounting
    """
    return "gpt-oss" in model.lower()


def build_harmony_conversation(params: dict[str, Any]) -> Conversation:
    """Build Harmony Conversation from vLLM Responses API params.

    Converts the params dict into a Harmony Conversation that matches
    what vLLM will see, allowing accurate token counting.

    Args:
        params: Request parameters for vLLM Responses API

    Returns:
        Harmony Conversation object
    """
    messages = []

    # 1. System (kept empty per spec)
    messages.append(Message.from_role_and_content(Role.SYSTEM, SystemContent.new()))

    # 2. Developer instructions (from system messages)
    instructions = params.get("instructions") or ""
    messages.append(
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(instructions),
        )
    )

    # 3. User inputs (from input array)
    for item in params.get("input", []):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            # Handle different message formats
            if item.get("role") == "user":
                # Extract content from user messages
                content = item.get("content", "")
                text = content if isinstance(content, str) else str(content)
            elif item.get("role") == "assistant":
                # Skip assistant messages in input counting
                # (they're part of context but not new input)
                continue
            else:
                # Other message types (tool results, etc.)
                text = str(item.get("content", "")) or str(item)
        else:
            # Fallback for unknown formats
            text = str(item)

        if text:
            messages.append(Message.from_role_and_content(Role.USER, text))

    return Conversation.from_messages(messages)


def compute_input_tokens(params: dict[str, Any]) -> int:
    """Compute accurate input token count using Harmony.

    Builds a Harmony conversation from params and generates prefill
    token IDs to count exactly what the model will see.

    Args:
        params: Request parameters for vLLM Responses API

    Returns:
        Number of input tokens (accurate), or 0 if encoder unavailable
    """
    try:
        encoding = _get_harmony_encoding()
        if encoding is None:
            return 0

        conversation = build_harmony_conversation(params)

        # Generate prefill token IDs (what the model sees as input)
        prefill_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

        input_tokens = len(prefill_ids)
        logger.debug(f"[TOKEN_ACCOUNTING] Computed input tokens: {input_tokens}")
        return input_tokens

    except Exception as e:
        logger.warning(f"[TOKEN_ACCOUNTING] Failed to compute input tokens: {e}", exc_info=True)
        return 0


def extract_final_text(response: Any) -> str:
    """Extract final assistant text from vLLM response.

    Robustly extracts the visible output text from the response object.
    Handles multiple response formats with fallback chain.

    Note: This only captures visible 'final' text, not hidden analysis/commentary
    channels (limitation of REST API).

    Args:
        response: OpenAI API response object

    Returns:
        Extracted text or empty string if extraction fails
    """
    # Fast path: Check for output_text attribute
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Generic scan: Walk response.output for message blocks
    try:
        if hasattr(response, "output"):
            for item in response.output or []:
                # Handle SDK objects
                if hasattr(item, "type"):
                    if item.type == "message":
                        msg_content = getattr(item, "content", [])
                        if isinstance(msg_content, list):
                            parts = []
                            for content_item in msg_content:
                                if hasattr(content_item, "type") and content_item.type in (
                                    "text",
                                    "output_text",
                                ):
                                    text = getattr(content_item, "text", None) or getattr(
                                        content_item, "output_text", None
                                    )
                                    if text:
                                        parts.append(text)
                            if parts:
                                return "\n".join(parts)

                # Handle dict format
                elif isinstance(item, dict) and item.get("type") == "message":
                    msg_content = item.get("content", [])
                    if isinstance(msg_content, list):
                        parts = []
                        for content_item in msg_content:
                            if content_item.get("type") in ("text", "output_text"):
                                text = content_item.get("text") or content_item.get("output_text")
                                if text:
                                    parts.append(text)
                        if parts:
                            return "\n".join(parts)
    except Exception as e:
        logger.debug(f"[TOKEN_ACCOUNTING] Text extraction error: {e}")

    # Fallback: Empty string
    return ""


def compute_output_tokens(text: str) -> int:
    """Compute approximate output token count using Harmony.

    Tokenizes the final assistant text to count output tokens.
    This is approximate because hidden reasoning/analysis channels
    are not accessible via REST API.

    Args:
        text: Extracted assistant text

    Returns:
        Number of output tokens (approximate), or 0 if encoder unavailable
    """
    if not text:
        return 0

    try:
        encoding = _get_harmony_encoding()
        if encoding is None:
            return 0

        completion_ids = encoding.encode(text)
        output_tokens = len(completion_ids)
        logger.debug(f"[TOKEN_ACCOUNTING] Computed output tokens: {output_tokens}")
        return output_tokens

    except Exception as e:
        logger.warning(f"[TOKEN_ACCOUNTING] Failed to compute output tokens: {e}", exc_info=True)
        return 0


def inject_usage(response: Any, input_tokens: int, output_tokens: int) -> Any:
    """Inject computed usage values into response object.

    Overrides the broken vLLM usage values with our computed counts.
    Uses proper OpenAI SDK ResponseUsage objects for Pydantic compatibility.

    Args:
        response: OpenAI API response object
        input_tokens: Computed input token count
        output_tokens: Computed output token count

    Returns:
        Response object with updated usage
    """
    from openai.types.responses.response_usage import InputTokensDetails
    from openai.types.responses.response_usage import OutputTokensDetails
    from openai.types.responses.response_usage import ResponseUsage

    total_tokens = input_tokens + output_tokens

    # Create proper Pydantic models (not dicts)
    usage = ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=total_tokens,
    )

    # Approach 1: Direct mutation (if supported)
    try:
        response.usage = usage
        logger.info(
            f"[TOKEN_ACCOUNTING] Injected usage: input={input_tokens}, output={output_tokens}, total={total_tokens}"
        )
        return response
    except Exception:
        pass

    # Approach 2: Reconstruction (if immutable)
    try:
        data = response.model_dump()
        data["usage"] = usage
        response = type(response)(**data)
        logger.info(
            f"[TOKEN_ACCOUNTING] Reconstructed response with usage: input={input_tokens}, "
            f"output={output_tokens}, total={total_tokens}"
        )
        return response
    except Exception as e:
        logger.warning(
            f"[TOKEN_ACCOUNTING] Failed to inject usage (both approaches): {e}. "
            f"Returning response with original (zero) usage values."
        )
        return response


def apply_token_accounting(params: dict[str, Any], response: Any) -> Any:
    """Apply token accounting to vLLM response.

    Main entry point: computes input/output tokens and injects them
    into the response object before returning.

    Args:
        params: Request parameters used to call vLLM
        response: Response object from vLLM

    Returns:
        Response object with corrected usage values
    """
    logger.debug("[TOKEN_ACCOUNTING] Applying token accounting for GPT-OSS model")

    # Compute input tokens (accurate)
    input_tokens = compute_input_tokens(params)

    # Extract final text and compute output tokens (approximate)
    final_text = extract_final_text(response)
    output_tokens = compute_output_tokens(final_text)

    # Inject usage into response
    return inject_usage(response, input_tokens, output_tokens)
