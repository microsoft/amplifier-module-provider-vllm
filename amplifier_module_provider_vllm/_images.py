"""Small Core-image/Responses adapter. No image decoding or server discovery."""

import base64
import binascii
import json
import re
from collections.abc import Mapping
from functools import cached_property
from typing import Any
from urllib.parse import urlparse

IMAGE_TYPES = ("image", "input_image", "image_url")
UNAVAILABLE = "[Image unavailable: invalid or unsupported image source.]"


def _mapping(block: Any) -> Any:
    return block.model_dump() if hasattr(block, "model_dump") else block


def has_direct_image(content: Any) -> bool:
    """Match only content blocks handled by nonuser_image_content.

    Image-like tags nested in text extras or tool arguments are business data,
    not a reason to change a message's role or instructions policy.
    """
    if not isinstance(content, list):
        return False
    return any(
        isinstance(value := _mapping(block), Mapping)
        and value.get("type") in IMAGE_TYPES
        for block in content
    )


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


def _submitted_image_urls(value):
    if isinstance(value, Mapping):
        if value.get("type") == "input_image":
            url = value.get("image_url")
            if isinstance(url, str) and url:
                yield url
                if url.startswith("data:"):
                    payload = url.partition(",")[2]
                    if payload:
                        yield payload
        else:
            for child in value.values():
                yield from _submitted_image_urls(child)
    elif isinstance(value, list):
        for item in value:
            yield from _submitted_image_urls(item)


class ImageDiagnostics:
    """Keep the exact submitted image echoes scoped to one Responses request."""

    def __init__(self, params):
        self.images = set(_submitted_image_urls(params))

    @cached_property
    def patterns(self):
        variants = set(self.images)
        for image in self.images:
            # Plain ASCII data URLs/base64 have identical JSON string contents.
            if image.isascii() and not any(ch in image for ch in ('"', '\\')) and image.isprintable():
                continue
            for ascii_only in (True, False):
                variants.add(json.dumps(image, ensure_ascii=ascii_only)[1:-1])
        # JSON also permits escaped forward slashes, including inside base64.
        variants |= {value.replace("/", r"\/") for value in variants}
        return sorted(variants, key=len, reverse=True)

    def copy(self, value: Any) -> Any:
        value = _mapping(value)
        if isinstance(value, Mapping):
            if value.get("type") in IMAGE_TYPES:
                return {"type": value["type"], "redacted": "[image omitted]"}
            return {key: self.copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.copy(item) for item in value]
        if isinstance(value, str):
            for pattern in self.patterns:
                value = value.replace(pattern, "[image omitted]")
        return value

    def clean_exception(self, error: Exception) -> None:
        if self.images:
            # SDK causes can hold the entire outgoing HTTP request.
            error.args = tuple(self.copy(arg) for arg in error.args)
            error.__cause__ = error.__context__ = None

    def error_message(self, error: Exception) -> str:
        body = getattr(error, "body", None)
        if body is None:
            return self.copy(str(error))
        # Remove any JSON-escaped URL in a field before serializing the body.
        return json.dumps(self.copy(body))


def image_diagnostic_copy(value: Any) -> Any:
    return ImageDiagnostics(value).copy(value)
