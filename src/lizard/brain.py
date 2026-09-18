"""The decision-maker interface.

Both brains return the same Decision, so the agent loop, the perception
layer, the executor and every deterministic guard are literally the same
code in both arms of the race. Only this object differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .jev import Jev, choice, noul, score


@dataclass
class Decision:
    action: str
    target: int | None
    confidence: float
    signals: dict
    latency_ms: float
    cost_usd: float
    compute_ms: float = 0.0        # server-side, when the API reports it
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def network_ms(self) -> float:
        return max(self.latency_ms - self.compute_ms, 0.0)


ACTIONS = {
    "click": "Click one of the listed elements to make progress",
    "type": "Type the search terms into a text input",
    "scroll": "Scroll down to reveal more of the page",
    "back": "This page was a dead end; return to the previous one",
    "done": "The task is complete; the answer is on this page",
}


class JevBrain:
    """Every question in one call - the fan-out is free."""

    name = "jev"

    def __init__(self, jev: Jev | None = None):
        self.jev = jev or Jev()

    def decide(self, state: str, elements) -> Decision:
        qs = {
            "action": choice("The agent is working on the task. "
                             "What should it do next?", ACTIONS),
            "relevant": noul("Is this page relevant to the task?"),
            "complete": noul("Has the task been fully accomplished?"),
            "answer_here": noul("Does the visible page text contain the answer?"),
            "looping": noul("Is the agent repeating actions without progress?"),
            "progress": score("How far along is the task?", [
                "No progress; on an unrelated or starting page",
                "Partway; on the right site but not the right page",
                "The answer is visible on this page",
            ]),
        }
        if elements:
            qs["target"] = choice(
                "Which element best advances the task if clicked or typed into?",
                {str(e.id): e.render() for e in elements},
            )
            qs["risky"] = noul(
                "Would interacting with the chosen element do something "
                "irreversible, such as buying, paying, deleting or signing out?"
            )

        v = self.jev.ask(state, qs)
        tgt = int(v["target"].value) if "target" in v.answers else None
        return Decision(
            action=v["action"].value,
            target=tgt,
            confidence=v["target"].confidence if "target" in v.answers else 0.0,
            signals={k: round(float(a.value), 3) for k, a in v.answers.items()
                     if k in ("relevant", "complete", "answer_here",
                              "looping", "progress", "risky")},
            latency_ms=v.latency_ms, cost_usd=v.cost_usd,
            compute_ms=v.compute_ms,
            input_tokens=v.input_tokens, output_tokens=v.output_tokens,
        )

    def warm(self) -> None:
        self.jev.warm()

    @property
    def total_cost(self) -> float:
        return self.jev.total_cost

    def close(self) -> None:
        self.jev.close()
