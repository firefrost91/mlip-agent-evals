import pytest

from shop.pricing import apply_discount, order_total


def test_no_discount():
    assert apply_discount(50.0, 0) == 50.0


def test_ten_percent_discount():
    assert apply_discount(50.0, 10) == 45.0


def test_order_total_with_discount():
    items = [{"price": 20.0, "qty": 2}, {"price": 10.0, "qty": 1}]
    assert order_total(items, discount_percent=20) == 40.0


def test_rejects_invalid_percent():
    with pytest.raises(ValueError):
        apply_discount(10.0, 150)
