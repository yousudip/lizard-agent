"""Numeric and temporal constraints — parsed and checked in plain Python.

Jev is never asked "is 1099 less than 2000". Arithmetic is not a judgement
call, and a model that answers it is a model that can get it wrong. Jev
decides what a thing *is*; this module decides whether it *qualifies*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PRICE_LIMIT = re.compile(
    r"(?:under|below|less than|within|upto|up to|max(?:imum)?|cheaper than)\s*"
    r"(?:rs\.?|inr|₹|\$)?\s*([\d,]+(?:\.\d+)?)\s*(k)?",
    re.I,
)
_DAY_LIMIT = re.compile(r"within\s+(\d+)\s*(?:day|days)", re.I)
_MONEY = re.compile(r"(?:₹|rs\.?\s*|inr\s*)([\d,]+(?:\.\d{1,2})?)", re.I)


@dataclass
class Constraints:
    max_price: float | None = None
    max_days: int | None = None

    def describe(self) -> str:
        bits = []
        if self.max_price is not None:
            bits.append(f"price at most {self.max_price:.0f}")
        if self.max_days is not None:
            bits.append(f"delivery within {self.max_days} days")
        return "; ".join(bits)

    def __bool__(self) -> bool:
        return self.max_price is not None or self.max_days is not None


def parse(task: str) -> Constraints:
    c = Constraints()
    if m := _PRICE_LIMIT.search(task):
        # "within 2 days" must not be read as a price of 2
        if not _DAY_LIMIT.search(m.group(0)):
            val = float(m.group(1).replace(",", ""))
            if m.group(2):          # "2k"
                val *= 1000
            c.max_price = val
    if m := _DAY_LIMIT.search(task):
        c.max_days = int(m.group(1))
    return c


def prices_in(text: str) -> list[float]:
    """Every currency amount in a blob of text, largest-first duplicates kept."""
    out = []
    for m in _MONEY.finditer(text):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


def selling_price(values: list[float]) -> float | None:
    """Pick the actual selling price out of every number on the page.

    Taking the minimum looks obvious and is wrong: pages are littered with
    coupon amounts and cashback lines ("Save ₹10"), and a first version of
    this happily reported a ₹1,499 product as costing ₹10. Taking the
    maximum is equally wrong - struck-through MRP and 24-month EMI totals
    are larger than the price.

    So: discard values an order of magnitude below the typical figure on
    the page, then take the smallest of what survives.
    """
    if not values:
        return None
    vals = sorted(values)
    mid = vals[len(vals) // 2]
    plausible = [v for v in vals if v >= mid * 0.1]
    return plausible[0] if plausible else vals[-1]


def check(text: str, c: Constraints,
          price: float | None = None) -> tuple[bool | None, str]:
    """Verify a page against the constraints.

    `price` may be supplied by a DOM-level extractor, which is far more
    reliable than scraping numbers out of rendered text. Returns
    (passed, explanation); `None` means undecidable - reported as such,
    never guessed at.
    """
    if c.max_price is None:
        return None, "no price constraint to check"

    if price is None:
        price = selling_price(prices_in(text))
    if price is None:
        return None, "no price found on the page"

    ok = price <= c.max_price
    return ok, f"₹{price:,.0f} is {'within' if ok else 'over'} the ₹{c.max_price:,.0f} limit"
