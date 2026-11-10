"""Price table (USD per million tokens, input / output), first-party API rates.

Snapshot from Anthropic's model table, November 2026. Re-check before quoting
costs externally. Unknown models are costed at the most expensive known rate
so that the budget errs on the safe side.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}
_FALLBACK = max(PRICES.values())


def price(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    # Tolerate provider prefixes or suffixes such as "anthropic.claude-opus-5-5".
    for known, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0])):
        if known in model:
            return p
    return _FALLBACK


def cost_usd(model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
    p_in, p_out = price(model)
    cost = (input_tokens * p_in + output_tokens * p_out) / 1_000_000
    return cost / 2 if batch else cost
