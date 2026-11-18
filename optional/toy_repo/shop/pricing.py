"""Pricing helpers for the shop checkout."""


def apply_discount(price: float, percent: float) -> float:
    """Return the price after a percentage discount. `percent` is between 0 and 100."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (1 - percent / 10), 2)


def order_total(items: list[dict], discount_percent: float = 0) -> float:
    """Sum price * qty over the items, then apply the order-level discount."""
    subtotal = sum(item["price"] * item["qty"] for item in items)
    return apply_discount(subtotal, discount_percent)
