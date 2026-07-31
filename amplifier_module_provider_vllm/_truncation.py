"""Silent input truncation detection for vLLM's truncation="auto" mode.

vLLM's Responses API accepts a ``truncation`` parameter with two values:

- ``"disabled"`` (the amplifier default -- see _constants.py:DEFAULT_TRUNCATION):
  an oversized prompt returns a clear HTTP 400 naming the exact context limit.
- ``"auto"``: an oversized prompt is silently truncated server-side -- HTTP 200,
  no error, no warning, and the model answers from whatever survived.

Operators who explicitly opt back into ``truncation="auto"`` (e.g. because an
upstream context manager doesn't yet cap prompt size) still deserve
visibility when it actually kicks in. This module implements the pure
detection rule; the caller (``VLLMProvider._maybe_warn_truncated_input``)
resolves the model's context window (via the same per-model
``_resolve_limits()`` discovery/config precedence used everywhere else) and
performs the actual logging.

Following the "bricks and studs" philosophy, this is a self-contained,
side-effect-free module that can be regenerated independently of the main
provider code.

Detection rule and margin -- evidence-based, verified live against a direct
vLLM endpoint (glm-5.2, real max_model_len=131072)::

    sent ~150,000 input tokens, truncation="auto", max_output_tokens=16
        -> HTTP 200, usage.input_tokens = 131056

131056 is exactly 16 below the true 131072 ceiling -- the same 16 tokens the
request asked to reserve for output. This suggests the server fills the
context window minus its output reservation: it truncates input to
``(context_window - requested_max_output_tokens)``. A margin tied to the
request's own ``max_output_tokens`` is therefore more principled than a
fixed magic-number margin: it tracks whatever completion budget the current
request actually asked for (which can range from a handful of tokens to
tens of thousands), rather than guessing a single constant that would be
wrong at both ends of that range.

NEGATIVE RESULT -- explicitly NOT used as a signal: ``response.status`` /
``incomplete_details``. In the same live probe, BOTH the oversized request
AND a 1,000-token control request returned ``status="incomplete"`` with
``incomplete_details.reason="max_output_tokens"`` (because
``max_output_tokens=16`` was tiny in both cases). That field reflects OUTPUT
truncation, not INPUT truncation, and fires on nearly every request at small
output budgets -- it cannot distinguish "the model ran out of tokens to
write" from "the server dropped part of the prompt". ``usage.input_tokens``
landing at/near the resolved context ceiling is the only signal that
isolates input truncation specifically.
"""

from typing import Any


def check_silent_input_truncation(
    *,
    truncation: Any,
    input_tokens: int | None,
    context_window: int | None,
    requested_max_output_tokens: int | None,
) -> str | None:
    """Return a warning message if truncation="auto" likely discarded input.

    Returns ``None`` (no warning warranted) when any of the following hold:

    - ``truncation`` is not exactly the string ``"auto"`` -- covers ``None``,
      ``"disabled"``, and any other/garbage value. This check only applies
      to the one mode that can silently drop data; ``"disabled"`` fails
      loud (HTTP 400) instead, so there is nothing to detect after the fact.
    - ``input_tokens`` is ``None`` -- usage wasn't available, nothing to compare.
    - ``context_window`` is ``None`` or non-positive -- nothing to compare against.

    The margin is ``requested_max_output_tokens`` (falling back to 0 when
    absent or non-positive) -- see the module docstring for the live-probe
    evidence justifying this choice over a fixed constant.

    Args:
        truncation: The effective truncation value sent on this request
            (e.g. the resolved ``params["truncation"]``).
        input_tokens: Reported ``usage.input_tokens`` for this response, or
            ``None`` if usage was unavailable.
        context_window: The resolved context window (tokens) for the model
            that served this request, or ``None``/non-positive if unknown.
        requested_max_output_tokens: The ``max_output_tokens`` actually sent
            on this request (the per-request completion cap), or ``None``.

    Returns:
        A human-readable warning message (caller decides how/whether to log
        it), or ``None`` if no warning is warranted.
    """
    if truncation != "auto":
        return None
    if input_tokens is None:
        return None
    if not context_window or context_window <= 0:
        return None

    margin = (
        requested_max_output_tokens
        if requested_max_output_tokens and requested_max_output_tokens > 0
        else 0
    )
    threshold = context_window - margin

    if input_tokens < threshold:
        return None

    return (
        f"input_tokens={input_tokens} is at/near the resolved context window "
        f"({context_window} tokens, {margin}-token output reservation). "
        f'truncation="auto" may have silently discarded part of the prompt '
        f"server-side -- the request still returned HTTP 200 with no error. "
        f'Set truncation="disabled" (the default) to fail loudly instead.'
    )
