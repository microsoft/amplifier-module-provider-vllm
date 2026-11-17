"""Small Core-image/Responses adapter. No image decoding or server discovery."""

import base64
import binascii
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

IMAGE_TYPES = ("image", "input_image", "image_url")
UNAVAILABLE = "[Image unavailable: invalid or unsupported image source.]"


def _mapping(block: Any) -> Any:
    return block.model_dump() if hasattr(block, "model_dump") else block


def nonuser_image_content(content: Any, role: str) -> Any:
    """Replace unsupported non-user image blocks, never their source history.

    Native image replay is qualified only for user/tool input. Keep a text
    marker at each image position without inspecting or echoing its source.
    Other blocks remain untouched for the existing role/reasoning serializer.
    """
    if not isinstance(content, list):
        return content
    result = []
    for block in content:
        value = _mapping(block)
        if isinstance(value, Mapping) and value.get("type") in IMAGE_TYPES:
            result.append({
                "type": "text",
                "text": f"[Image unavailable: native image replay is not supported "
                        f"by this provider in {role} content.]",
            })
        else:
            result.append(block)
    return result


def contains_image(value: Any) -> bool:
    """Recognize structured image blocks, never inspect/stringify pixel data."""
    value = _mapping(value)
    if isinstance(value, Mapping):
        return value.get("type") in IMAGE_TYPES or any(
            contains_image(item) for item in value.values()
            if isinstance(item, (Mapping, list))
        )
    return isinstance(value, list) and any(contains_image(item) for item in value)


def _valid_base64(data: Any) -> bool:
    if not isinstance(data, str) or not data:
        return False
    try:
        # Syntax check only: server/model owns format decoding and modality limits.
        return bool(base64.b64decode(data, validate=True))
    except (ValueError, binascii.Error):
        return False


def _image_url(source: Any) -> str | None:
    if not isinstance(source, Mapping):
        return None
    if source.get("type") == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if (isinstance(media_type, str)
                and re.fullmatch(r"image/[a-zA-Z0-9.+-]+", media_type)
                and _valid_base64(data)):
            return f"data:{media_type};base64,{data}"
    elif source.get("type") == "url":
        url = source.get("url")
        if not isinstance(url, str) or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url
        ):
            return None
        if url.startswith("data:"):
            header, separator, data = url.partition(",")
            if (separator and re.fullmatch(r"data:image/[a-zA-Z0-9.+-]+;base64", header)
                    and _valid_base64(data)):
                return url
            return None
        try:
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https") and parsed.hostname:
                return url
        except ValueError:
            pass
    return None


def user_content_parts(content: list) -> list[dict[str, Any]]:
    """Preserve text/image order, including explicit per-image failure markers."""
    parts = []
    for block in content:
        block = _mapping(block)
        if not isinstance(block, Mapping):
            continue
        kind = block.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append({"type": "input_text", "text": block.get("text", "")})
        elif kind == "image":
            url = _image_url(block.get("source"))
            parts.append(
                {"type": "input_image", "image_url": url, "detail": "auto"}
                if url else {"type": "input_text", "text": UNAVAILABLE}
            )
        elif kind in IMAGE_TYPES:
            # Only the Core ImageBlock.source contract is accepted as input.
            parts.append({"type": "input_text", "text": UNAVAILABLE})
        elif contains_image(block):
            # Not a Core content block: do not silently discard nested pixels.
            parts.append({"type": "input_text", "text": UNAVAILABLE})
    return parts


def tool_content_parts(content: Any, call_id: str) -> list[dict[str, Any]]:
    """Unwrap Core ToolResultBlock.output, keeping the original call identity."""
    blocks = content if isinstance(content, list) else [content]
    parts = []
    for block in blocks:
        block = _mapping(block)
        if isinstance(block, Mapping) and block.get("type") == "tool_result":
            if block.get("tool_call_id") != call_id:
                raise ValueError("Image tool result has a mismatched tool_call_id")
            parts.extend(tool_content_parts(block.get("output"), call_id))
        elif isinstance(block, str):
            parts.append({"type": "input_text", "text": block})
        else:
            parts.extend(user_content_parts([block]))
    return parts


def image_diagnostic_copy(value: Any) -> Any:
    """Copy diagnostics, replacing entire image blocks (including signed URLs).

    This is structured-media redaction, not general arbitrary-text redaction.
    Secret-key redaction is applied separately by the Core utility.
    """
    value = _mapping(value)
    if isinstance(value, Mapping):
        if value.get("type") in IMAGE_TYPES:
            return {"type": value["type"], "redacted": "[image omitted]"}
        return {key: image_diagnostic_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [image_diagnostic_copy(item) for item in value]
    return value


def image_error_message(error: Exception) -> str:
    """Keep error category/metadata without echoing server image validation input."""
    body = getattr(error, "body", None)
    if body is None:
        return str(error)
    return json.dumps(image_diagnostic_copy(body))