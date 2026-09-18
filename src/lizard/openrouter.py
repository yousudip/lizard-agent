"""LLM baseline via OpenRouter.

Gemini Flash Lite is the *hardest* baseline to beat, which is why it is the
default. The obvious objection to a System One model is "why not just use a
small fast LLM?" - so the comparison worth publishing is against the
fastest, cheapest generalist available, not against a frontier model that
was never going to win a latency race.

Fairness: this client is pooled exactly like the Jev client, gets the same
warmup, sees the same state string, and is asked for the same seven typed
signals via structured outputs. Only the decision-maker differs.
"""

from __future__ import annotations

import json
import time

import httpx

from .brain import Decision
from .config import OPENROUTER_KEY, OPENROUTER_MODEL

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Mirrors the Jev question set one-for-one.
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string",
                   "enum": ["click", "type", "scroll", "back", "done"]},
        "target": {"type": ["integer", "null"],
                   "description": "id of the element to act on, or null"},
        "confidence": {"type": "number", "description": "0-1 in the target"},
        "relevant": {"type": "number"},
        "complete": {"type": "number"},
        "answer_here": {"type": "number"},
        "looping": {"type": "number"},
        "risky": {"type": "number",
                  "description": "0-1 that this is irreversible: buying, "
                                 "paying, deleting, signing out"},
    },
    "required": ["action", "target", "confidence", "relevant", "complete",
                 "answer_here", "looping", "risky"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are the decision layer of a browser agent. You are given the task, "
    "the current page, and a numbered list of the interactive elements you "
    "may act on. Choose the next action and, when clicking or typing, the id "
    "of the element to act on. All probabilities are 0-1. "
    "Respond only with the structured result."
)


class OpenRouterBrain:
    name = "llm"

    def __init__(self, model: str = OPENROUTER_MODEL, timeout: float = 60.0):
        if not OPENROUTER_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
        self.model = model
        self._c = httpx.Client(
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
                # OpenRouter asks for these; they also make the run
                # identifiable in the dashboard.
                "HTTP-Referer": "https://github.com/lizard-agent",
                "X-Title": "lizard",
            },
            timeout=timeout,
            # same pooling the Jev arm gets - an unpooled baseline would be
            # a rigged race
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300),
        )
        self.calls = 0
        self.total_ms = 0.0
        self.total_cost = 0.0

    def decide(self, state: str, elements=None) -> Decision:
        t0 = time.perf_counter()
        r = self._c.post(ENDPOINT, json={
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": state},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "strict": True,
                                "schema": SCHEMA},
            },
            "max_tokens": 700,
            # OpenRouter returns the real dollar cost, so there is no
            # pricing table here to drift out of date.
            "usage": {"include": True},
        })
        ms = (time.perf_counter() - t0) * 1000

        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text[:400]}")

        body = r.json()
        if "choices" not in body:
            raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(body)[:400]}")

        d = json.loads(body["choices"][0]["message"]["content"])
        usage = body.get("usage", {}) or {}
        cost = float(usage.get("cost", 0.0))

        self.calls += 1
        self.total_ms += ms
        self.total_cost += cost

        tgt = d.get("target")
        return Decision(
            action=d["action"],
            target=int(tgt) if tgt is not None else None,
            confidence=float(d.get("confidence", 0.0)),
            signals={k: round(float(d.get(k, 0.0)), 3) for k in
                     ("relevant", "complete", "answer_here", "looping", "risky")},
            latency_ms=ms, cost_usd=cost,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def warm(self) -> None:
        """Pay TLS and cold-start before the clock starts, exactly as the
        Jev arm does."""
        try:
            self.decide("TASK: warmup\nINTERACTIVE ELEMENTS (0 shown of 0 found):")
            self.calls = 0
            self.total_ms = 0.0
            self.total_cost = 0.0
        except Exception:
            pass

    def close(self) -> None:
        self._c.close()
