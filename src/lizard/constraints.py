"""Numeric and temporal constraints — parsed and checked in plain Python.

Jev is never asked "is 1099 less than 2000". Arithmetic is not a judgement
call, and a model that answers it is a model that can get it wrong. Jev
decides what a thing *is*; this module decides whether it *qualifies*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

_PRICE_LIMIT = re.compile(
    r"(?:under|below|less than|within|upto|up to|max(?:imum)?|cheaper than)\s*"
    r"(?:rs\.?|inr|₹|\$)?\s*([\d,]+(?:\.\d+)?)\s*(k)?",
    re.I,
)
_DAY_LIMIT = re.compile(
    r"(?:within|under|in|less than|max(?:imum)?)\s+(\d+)\s*(?:day|days)", re.I)
# Indian PIN codes are six digits and never start with 0.
_PIN = re.compile(r"\b([1-9]\d{5})\b")
_MONEY = re.compile(r"(?:₹|rs\.?\s*|inr\s*)([\d,]+(?:\.\d{1,2})?)", re.I)


@dataclass
class Constraints:
    max_price: float | None = None
    max_days: int | None = None
    pincode: str | None = None

    def describe(self) -> str:
        bits = []
        if self.max_price is not None:
            bits.append(f"price at most {self.max_price:.0f}")
        if self.max_days is not None:
            bits.append(f"delivery within {self.max_days} days")
        if self.pincode:
            bits.append(f"deliverable to {self.pincode}")
        return "; ".join(bits)

    def __bool__(self) -> bool:
        return any((self.max_price is not None, self.max_days is not None,
                    self.pincode))


def parse(task: str) -> Constraints:
    """Read the machine-checkable constraints out of a task."""
    c = Constraints()

    # A duration clause looks exactly like a price clause ("under 2 days"
    # vs "under 2000"), so skip any match whose text carries a time unit.
    price = next(
        (m for m in _PRICE_LIMIT.finditer(task)
         if not re.search(r"\d\s*(?:day|days|hour|hours|week|weeks)",
                          m.group(0), re.I)),
        None,
    )
    if price:
        val = float(price.group(1).replace(",", ""))
        if price.group(2):                      # "2k"
            val *= 1000
        c.max_price = val

    if m := _DAY_LIMIT.search(task):
        c.max_days = int(m.group(1))

    # A PIN must not be mistaken for a price, so ignore a number already
    # claimed by the price clause.
    for m in _PIN.finditer(task):
        if c.max_price is None or float(m.group(1)) != c.max_price:
            c.pincode = m.group(1)
            break

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


# "FREE delivery Sunday, 20 September" / "Get it by Tomorrow"
_DELIVERY = re.compile(
    r"(?:delivery|delivered|get it)\s+(?:by\s+)?"
    r"(tomorrow|today|[A-Z][a-z]+day|[A-Z][a-z]{2},?\s*\d{1,2}\s*[A-Z][a-z]+"
    r"|\d{1,2}\s*[A-Z][a-z]+)",
    re.I,
)


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_DAYNUM = re.compile(r"(\d{1,2})\s*([A-Za-z]{3,})")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday",
             "friday", "saturday", "sunday"]


def days_until(when: str, today: date | None = None) -> int | None:
    """How many days away is a delivery estimate like "Sun, 20 Sept"?

    Date arithmetic is not a judgement call, so it is done here rather than
    asked of a model. Returns None when the string holds no readable date -
    reported as unchecked rather than guessed at.
    """
    today = today or date.today()
    w = when.strip().lower()
    if w == "today":
        return 0
    if w == "tomorrow":
        return 1

    if m := _DAYNUM.search(w):
        day = int(m.group(1))
        mon = _MONTHS.get(m.group(2)[:3])
        if not mon or not 1 <= day <= 31:
            return None
        year = today.year
        try:
            target = date(year, mon, day)
        except ValueError:
            return None
        # A date that has already passed refers to next year.
        if (target - today).days < -180:
            try:
                target = date(year + 1, mon, day)
            except ValueError:
                return None
        return (target - today).days

    # A bare weekday ("delivery Sunday") means the next one, and today
    # itself would have been written as "today".
    for i, name in enumerate(_WEEKDAYS):
        if re.fullmatch(rf"{name}|{name[:3]}\.?", w):
            ahead = (i - today.weekday()) % 7
            return ahead or 7

    return None


def check(text: str, c: Constraints,
          price: float | None = None) -> tuple[bool | None, str]:
    """Verify a page against every constraint the task carried.

    Checking only the convenient one is how an agent reports success it
    did not achieve: an early version verified the price and said PASS on
    a run where the delivery PIN had never actually been applied. Each
    constraint is now reported separately, and one that cannot be decided
    from the page says so rather than being quietly dropped.

    Returns (passed, explanation). `None` means undecidable - never
    guessed at. A single failure fails the whole check.
    """
    if not c:
        return None, "no constraints to check"

    parts: list[str] = []
    verdicts: list[bool | None] = []

    if c.max_price is not None:
        p = price if price is not None else selling_price(prices_in(text))
        if p is None:
            verdicts.append(None)
            parts.append("price: not found on page")
        else:
            ok = p <= c.max_price
            verdicts.append(ok)
            parts.append(f"₹{p:,.0f} {'≤' if ok else '>'} ₹{c.max_price:,.0f}")

    if c.pincode:
        seen = c.pincode in text
        verdicts.append(seen)
        parts.append(f"PIN {c.pincode} "
                     f"{'applied' if seen else 'NOT applied to this page'}")

    if c.max_days is not None:
        m = _DELIVERY.search(text)
        if not m:
            verdicts.append(None)
            parts.append("delivery date: not found on page")
        else:
            when = m.group(1).lower()
            # Only same/next-day wording is decidable without a calendar;
            # a named date is reported rather than guessed at.
            days = days_until(m.group(1))
            if days is None:
                verdicts.append(None)
                parts.append(f"delivery '{m.group(1)}' (could not read a date)")
            else:
                ok = days <= c.max_days
                verdicts.append(ok)
                parts.append(f"delivery in {days} day{'s' * (days != 1)} "
                             f"({m.group(1)}) {'≤' if ok else '>'} "
                             f"{c.max_days}")

    if any(v is False for v in verdicts):
        return False, "; ".join(parts)
    if any(v is None for v in verdicts):
        return None, "; ".join(parts)
    return True, "; ".join(parts)
