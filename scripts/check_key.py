"""Verify the Jev key works, and measure real latency.

Sends one call containing all three primitives, so it doubles as a check
that the request/response shapes are what we think they are.
"""

import sys, time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lizard.config import JEV_ENDPOINT, JEV_MODEL, api_key  # noqa: E402

STATE = (
    "TASK: find a wireless headphone under Rs 2000\n"
    'ELEMENT: link "boAt Rockerz 371, 40mm Drivers, 50hrs Battery - Rs 1,099"'
)

QUESTIONS = {
    "is_headphone": {
        "type": "noul",
        "instructions": "Is this element a listing for wireless headphones?",
    },
    "action": {
        "type": "choice",
        "instructions": "What should the agent do next?",
        # choice criteria: a dict of option -> description (or null)
        "criteria": {
            "click": "Click an element on the page",
            "type": "Type text into an input",
            "scroll": "Scroll to reveal more of the page",
            "back": "Go back to the previous page",
            "done": "The task is complete",
        },
    },
    "relevance": {
        "type": "score",
        "instructions": "How well does this listing match the task?",
        # score criteria: an ordered list of levels, 2-10 entries
        "criteria": [
            "Unrelated to the task",
            "Related but does not satisfy the constraints",
            "Matches the task well",
        ],
    },
}


def main() -> int:
    try:
        key = api_key()
    except RuntimeError as e:
        print(f"✗ {e}")
        return 1

    print(f"endpoint : {JEV_ENDPOINT}\nmodel    : {JEV_MODEL}\nkey      : {key[:6]}…{key[-4:]}\n")

    timings = []
    for i in range(3):
        t0 = time.perf_counter()
        try:
            r = httpx.post(
                JEV_ENDPOINT,
                headers={"Authorization": f"Bearer {key}"},
                json={"state": STATE, "model": JEV_MODEL, "questions": QUESTIONS},
                timeout=30,
            )
        except Exception as e:
            print(f"✗ request failed: {type(e).__name__}: {e}")
            return 1
        ms = (time.perf_counter() - t0) * 1000
        timings.append(ms)

        if r.status_code != 200:
            print(f"✗ HTTP {r.status_code}\n{r.text[:600]}")
            return 1
        if i == 0:
            import json
            print("response:")
            print(json.dumps(r.json(), indent=2)[:900])
            print()
        print(f"  call {i+1}: {ms:6.1f} ms")

    print(f"\n✓ working — median {sorted(timings)[1]:.0f} ms for 3 parallel questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
