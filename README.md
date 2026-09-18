# 🦎 Lizard

**No cortex required.**

A browser agent driven by deterministic code and [Jev](https://typesafe.ai)
(TypeSafe's System One model) — with **zero LLM calls** in the loop.

Jev decides, code acts. Every decision Jev returns is an integer or an enum,
never text, so the action space is closed by construction: it cannot invent an
element that isn't on the page or emit a malformed selector.

```bash
uv sync && uv run playwright install chromium
cp .env.example .env          # add your TypeSafe key
uv run python scripts/run.py https://www.amazon.in \
  "find a wireless headphone under Rs 2000 delivered within 2 days"
```

```
1. type    searchbox "Search Amazon.in"                conf=0.90
2. scroll                                              conf=0.50
3. click   "Apply the filter Get It in 2 Days"         conf=0.73
4. click   "Zebronics Thunder (2026 Upgrade)"          conf=0.63
5. done                                                → new tab followed

verified: PASS — ₹749 is within the ₹2,000 limit
```

Nobody told it Amazon has a delivery filter. It read the page, found the
control that satisfied "within 2 days", and used it.

## Status

- [x] Perception layer — page → prunable, typed action space
- [x] Jev client — pooled, typed answers
- [x] Decision loop — 7-question fan-out, deterministic guards
- [x] Deterministic constraints — prices and dates never touch the model
- [x] Trace logger — JSONL, one line per decision
- [x] Overlay / HUD — decisions painted onto the page, captured on video
- [x] Compute vs network split — measured, not estimated
- [x] Side-by-side race runner

## Runs

```bash
uv run python scripts/run.py https://github.com/langchain-ai/langchain \
  "find the contributing guidelines document for this project"

uv run python scripts/run.py https://www.amazon.in \
  "find a wireless headphone under Rs 2000 delivered within 2 days"

# with the on-page HUD, recorded to video, holding 900ms per decision
uv run python scripts/run.py https://www.amazon.in "<task>" \
  --video videos --dwell 900

# race it against an LLM agent (needs ANTHROPIC_API_KEY)
uv run python scripts/race.py https://github.com/langchain-ai/langchain \
  "find the contributing guidelines document" --headed --video videos
```

## Fairness of the race

The two arms share the perception layer, the 255-element cap, the executor,
every deterministic guard, and the warmup. The only thing that differs is
the object that answers `decide(state, elements)` — `JevBrain` or
`ClaudeBrain`. Anything else would make the comparison meaningless.

Both are asked for the same seven typed signals; the LLM arm gets them via
structured outputs so the comparison is decision-for-decision.

Run with `--sequential` for clean timing (the two arms otherwise share
bandwidth) and without it for the video.

| task | steps | Jev time | wall | cost | outcome |
|---|---|---|---|---|---|
| GitHub → contributing guide | 3 | 1.6s | 3.3s | $0.00075 | crossed to docs.langchain.com |
| Amazon → headphone under ₹2000, 2-day | 5 | 2.4s | 8.4s | $0.0021 | ₹749, delivery filter applied |

Zero LLM calls in either.

## Perception benchmarks

Measured on live pages, 1440x900 viewport:

| site | elements | after dedupe | kept | named | tokens | $/call |
|---|---|---|---|---|---|---|
| GitHub repo | 200 | 182 | 182 | 100% | 2186 | $0.000092 |
| Python docs | 250 | 131 | 131 | 100% | 1833 | $0.000077 |
| Amazon search | 395 | 358 | 255 | 100% | 4628 | $0.000194 |
| HN front page | 230 | 183 | 183 | 83% | 2232 | $0.000094 |

A 30-step task costs roughly **$0.0034**.

```bash
uv run python scripts/probe_perception.py
```

## Where the time actually goes

The API's proxy returns `x-envoy-upstream-service-time`, so the split is
measured rather than inferred. One Amazon run, 5 steps, from Bengaluru:

| | ms | share |
|---|---|---|
| Jev compute | 576 | **4%** |
| Network (India ↔ US) | 1,708 | 13% |
| Page loads | 10,914 | **83%** |
| wall | 13,198 | |

Median compute per decision: **118 ms**.

The model is 4% of the run. Making it faster would change almost nothing —
the bottleneck left the model and became the internet. That is the whole
argument for a System One model in an agent loop.

## Measured latency (from India, Sept 2026)

TypeSafe claims 70–500ms. What I actually measured, and where it goes:

| | ms |
|---|---|
| DNS | 3 |
| TCP connect (1 RTT) | 251 |
| TLS handshake (+1 RTT) | 254 |
| request + compute + response | ~350 |
| **fresh connection per call** | **1066** |
| **pooled connection** | **349** |

One network round-trip to the US is ~250ms, so **model compute is ~100ms** —
consistent with TypeSafe's claim. The rest is the Indian Ocean.

**Reusing the TLS connection is a 3x win** (1066ms → 349ms) and is a single
line of client config. It is the difference between the model being faster
than a page load and slower than one.

### Extra questions are genuinely free

| questions | ms |
|---|---|
| 1 | 368 |
| 5 | 369 |
| 15 | 356 |

They evaluate in parallel, so there is no reason to ask one thing at a time.
Fan out everything you might need in a single call.

## Race results

`google/gemini-2.5-flash-lite` — deliberately the hardest baseline. The
objection to a System One model is "why not just use a small fast LLM?",
so the comparison worth publishing is against the fastest cheap generalist,
not a frontier model that was never going to win a latency race.

| task | arm | steps | per decision | cost | outcome |
|---|---|---|---|---|---|
| GitHub → contributing guide | jev | 3 | 481 ms | $0.00070 | done |
| | llm | 3 | 1005 ms | $0.00062 | done |
| Amazon → headphone under ₹2000 | jev | 5 | 529 ms | $0.00174 | **done** |
| | llm | 5 | 1072 ms | $0.00211 | **stuck** |

**2.0–2.1x faster per decision as measured from Bengaluru.**

### The networks are not equidistant

| host | TCP RTT from Bengaluru |
|---|---|
| openrouter.ai | **14 ms** (CDN edge in India) |
| api.typesafe.ai | **250–271 ms** (US origin) |

Jev pays a quarter-second geographic tax per call that the baseline does
not. Subtracting one round trip from each side: **4.1–4.2x faster**
(258 ms vs 1058 ms per decision). Both numbers belong in any honest
write-up — a user in India really does pay that 250 ms.

### Cost is a wash — say so

**0.8–1.2x.** Against Flash Lite, Jev is somewhere between slightly cheaper
and slightly more expensive. The 400x figure is real against frontier
models and irrelevant here. The story against Flash Lite is latency and
reliability, not price.

### Calibration is the more interesting result

From the Amazon traces, the baseline stuck in a scroll loop:

```
llm   2. scroll  conf=0.80
      3. scroll  conf=0.80
      4. scroll  conf=1.00     looping: 0.0
      5. stuck                 <- our deterministic guard, not its own judgement
```

It reported `looping: 0.0` **while visibly looping**, at confidence 1.00.

Jev on the same task: 0.91 on the obvious search box, 0.56 when unsure,
0.17 on an ambiguous final step — and it finished. The confidence tracked
reality instead of decorating it. For a system that gates actions on
confidence, that difference matters more than the milliseconds.
