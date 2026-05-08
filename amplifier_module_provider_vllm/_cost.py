"""vllm cost computation — self-hosted, always None."""

from decimal import Decimal

_RATES: dict[str, dict[str, Decimal]] = {}  # self-hosted — no rate data


def compute_cost(model: str, **kwargs) -> Decimal | None:
    """Self-hosted provider — cost is always indeterminate (not $0.00)."""
    return None
