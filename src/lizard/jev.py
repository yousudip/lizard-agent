"""Jev client.

Two measured facts shape this module:

1. Connection reuse is worth 3x. A fresh TLS handshake to api.typesafe.ai
   costs ~500ms from India; the call itself is ~350ms. Opening a new
   connection per decision took a measured 1066ms -> 349ms once pooled.
   So the client is long-lived and shared for the whole run.

2. Extra questions are free. Measured from India: 1 question 368ms,
   5 questions 369ms, 15 questions 356ms. They evaluate in parallel, so
   there is no reason to ask one thing at a time. Fan out aggressively.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import JEV_ENDPOINT, JEV_MODEL, api_key


# ---------------------------------------------------------------- questions

def noul(instructions: str) -> dict:
    """Yes/no. Returns a probability the statement is true."""
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: dict[str, Any]) -> dict:
    """Pick one option. `criteria` maps option name -> description (or None).
    Max 255 options."""
    if len(criteria) > 255:
        raise ValueError(f"choice supports at most 255 options, got {len(criteria)}")
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list[Any]) -> dict:
    """Rate on an ordered scale. `criteria` is 2-10 level descriptions,
    lowest first."""
    if not 2 <= len(criteria) <= 10:
        raise ValueError(f"score needs 2-10 levels, got {len(criteria)}")
    return {"type": "score", "instructions": instructions, "criteria": criteria}


# ---------------------------------------------------------------- responses

@dataclass
class Answer:
    """One typed answer. Never a string the caller has to parse."""
    kind: str
    value: Any                       # bool-ish float | chosen option | score
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<{self.kind} {self.value!r} conf={self.confidence:.2f}>"


@dataclass
class Verdict:
    answers: dict[str, Answer]
    latency_ms: float            # wall clock, client side
    compute_ms: float            # server-side, from x-envoy-upstream-service-time
    input_tokens: int
    output_tokens: int

    @property
    def network_ms(self) -> float:
        """Everything that wasn't the model: round-trip, TLS record
        handling, proxy hops. From India this dominates."""
        return max(self.latency_ms - self.compute_ms, 0.0)

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1e6 * 0.042      # output is free


def _parse(name: str, raw: dict) -> Answer:
    kind = raw["type"]
    if kind == "noul":
        # noul returns a bare probability and no separate confidence;
        # distance from 0.5 is how decided it is.
        p = raw["noul"]
        return Answer(kind, p, abs(p - 0.5) * 2)
    if kind == "choice":
        return Answer(kind, raw["choice"], raw.get("confidence", 0.0),
                      raw.get("probabilities", {}))
    if kind == "score":
        return Answer(kind, raw["score"], raw.get("confidence", 0.0),
                      raw.get("probabilities", {}))
    raise ValueError(f"unknown answer type {kind!r} for question {name!r}")


# ---------------------------------------------------------------- client

class Jev:
    """Long-lived, connection-pooled client. Use one per run."""

    def __init__(self, model: str = JEV_MODEL, timeout: float = 30.0):
        self.model = model
        self._c = httpx.Client(
            headers={"Authorization": f"Bearer {api_key()}",
                     "Content-Type": "application/json"},
            timeout=timeout,
            # keep the TLS connection hot: this is the 3x
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300),
        )
        self.calls = 0
        self.total_ms = 0.0
        self.total_compute_ms = 0.0
        self.total_cost = 0.0

    def ask(self, state: str, questions: dict[str, dict]) -> Verdict:
        t0 = time.perf_counter()
        r = self._c.post(JEV_ENDPOINT, json={
            "state": state, "model": self.model, "questions": questions,
        })
        ms = (time.perf_counter() - t0) * 1000

        if r.status_code != 200:
            raise RuntimeError(f"Jev HTTP {r.status_code}: {r.text[:500]}")

        body = r.json()
        usage = body.get("usage", {})
        # Their Envoy proxy reports server-side processing time, so the
        # compute/network split is measured rather than guessed at.
        try:
            compute = float(r.headers.get("x-envoy-upstream-service-time", 0))
        except (TypeError, ValueError):
            compute = 0.0

        v = Verdict(
            answers={k: _parse(k, a) for k, a in body["answers"].items()},
            latency_ms=ms,
            compute_ms=compute,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        self.calls += 1
        self.total_ms += ms
        self.total_compute_ms += compute
        self.total_cost += v.cost_usd
        return v

    def warm(self) -> None:
        """Pay the TLS handshake before the clock starts."""
        try:
            self.ask("warmup", {"_": noul("Is this a warmup?")})
            self.calls -= 1
        except Exception:
            pass

    def close(self) -> None:
        self._c.close()

    def __enter__(self) -> "Jev":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
