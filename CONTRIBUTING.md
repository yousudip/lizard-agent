# Contributing

## Setup

```bash
uv sync
uv run playwright install chromium
cp .env.example .env     # add your TypeSafe key
```

## The one rule

**Jev decides, code acts.**

Every value Jev returns is an integer or an enum member — never text. The
action space is closed by construction: it cannot name an element that isn't
on the page, and it cannot emit a selector at all.

Anything numeric, temporal or mechanical stays in Python: loop detection,
step budgets, price and date comparisons, the safety denylist. Arithmetic is
not a judgement call, and a model asked to do it is a model that can get it
wrong. Jev decides what a thing *is*; Python decides whether it *qualifies*.

If you find yourself asking Jev whether 749 is less than 2000, stop.

## Benchmarks must stay fair

Both arms of the race share the perception layer, the 255-element cap, the
executor, every deterministic guard, and the warmup. The only thing that may
differ is the object answering `decide(state, elements)`.

If you change something that affects one arm, it must affect both.

Report what you measure, including when it is unflattering. The network
distance to the two APIs is not equal — see the README — so latency
comparisons carry both the raw number and the round-trip-adjusted one.

## Adding a site

Sites fail here for one reason: accessible names. Jev sees `role + name`.
`link "MIT license"` works; `<div onclick>` with no label does not. Check
what the page actually exposes before blaming the model:

```bash
uv run python scripts/probe_perception.py
```
