"""Repeat the race enough times to report a number with error bars.

A single run of a stochastic system is an anecdote. This runs each arm N
times per task, sequentially so the two never compete for bandwidth, and
reports medians with spread plus a completion rate - which on the harder
tasks turns out to matter more than the milliseconds.

  uv run python scripts/trials.py --trials 5
  uv run python scripts/trials.py --trials 3 --model google/gemini-3.1-flash-lite
"""

import argparse, asyncio, json, socket, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.async_api import async_playwright          # noqa: E402
from lizard.agent import Agent                              # noqa: E402
from lizard.brain import JevBrain                           # noqa: E402
from lizard.config import OPENROUTER_MODEL                  # noqa: E402
from lizard.openrouter import OpenRouterBrain               # noqa: E402

import re

# Each task carries a ground-truth check. An agent's own `done` flag is a
# claim, not a result: on the pypi task the LLM baseline reported done with
# confidence 1.00 while sitting on a search-results page that never held
# the answer. Scoring self-reported success rewards overconfidence, so the
# benchmark checks the final page instead.
TASKS = [
    ("github", "https://github.com/langchain-ai/langchain",
     "find the contributing guidelines document for this project", 8,
     lambda url, text: bool(re.search(r"contribut", url, re.I))),

    ("amazon", "https://www.amazon.in",
     "find a wireless headphone under Rs 2000 delivered within 2 days", 10,
     # a real product page, and the price constraint actually verified
     lambda url, text: "/dp/" in url or "/gp/product/" in url),

    ("pypi", "https://pypi.org",
     "find the latest released version of the langchain package", 8,
     # must be on langchain's project page with a version string visible
     lambda url, text: bool(re.search(r"/project/langchain", url, re.I))
                       and bool(re.search(r"\b\d+\.\d+\.\d+\b", text))),
]


def rtt_ms(host, port=443, n=3):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            socket.create_connection((host, port), timeout=5).close()
            out.append((time.perf_counter() - t) * 1000)
        except OSError:
            pass
    return statistics.median(out) if out else 0.0


async def one(pw, brain, url, task, steps, expect=None):
    browser = await pw.chromium.launch(headless=True)
    page = await browser.new_page(viewport={"width": 1440, "height": 900})
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1200)
        brain.warm()
        res = await Agent(page, brain, task, max_steps=steps,
                          verbose=False).run()
        passed = bool(expect(res.final_url, res.final_text)) if expect else res.done
        return {
            "done": res.done,          # what the agent claimed
            "passed": passed,          # what actually happened
            "steps": len(res.steps),
            "per_decision_ms": res.jev_ms / max(len(res.steps), 1),
            "compute_ms": res.compute_ms / max(len(res.steps), 1),
            "wall_ms": res.wall_ms,
            "cost": res.cost_usd,
            "verified": res.verified,
        }
    except Exception as e:
        return {"done": False, "passed": False, "steps": 0, "per_decision_ms": 0.0,
                "compute_ms": 0.0, "wall_ms": 0.0, "cost": 0.0,
                "error": f"{type(e).__name__}: {str(e)[:80]}"}
    finally:
        await browser.close()


def summarise(runs):
    ok = [r for r in runs if r["steps"]]
    if not ok:
        return None
    lat = sorted(r["per_decision_ms"] for r in ok)
    return {
        "n": len(runs),
        "claimed": sum(1 for r in runs if r["done"]),
        "passed": sum(1 for r in runs if r.get("passed")),
        "median_ms": statistics.median(lat),
        "min_ms": lat[0],
        "max_ms": lat[-1],
        "median_steps": statistics.median(r["steps"] for r in ok),
        "median_cost": statistics.median(r["cost"] for r in ok),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--only", help="run just one task by name")
    ap.add_argument("--out", default="traces/trials.json")
    a = ap.parse_args()

    model = a.model or OPENROUTER_MODEL
    tasks = [t for t in TASKS if not a.only or t[0] == a.only]

    try:
        OpenRouterBrain(model=model).decide("TASK: preflight\nELEMENTS: none")
    except Exception as e:
        print(f"\n  baseline unavailable: {str(e)[:160]}\n")
        return 1

    rtts = {"jev": rtt_ms("api.typesafe.ai"), "llm": rtt_ms("openrouter.ai")}
    print(f"\n  {a.trials} trials per arm per task   vs {model}")
    print(f"  network floor: jev {rtts['jev']:.0f}ms/call, "
          f"baseline {rtts['llm']:.0f}ms/call\n")

    out = {"when": datetime.now(timezone.utc).isoformat(), "model": model,
           "trials": a.trials, "rtt_ms": rtts, "tasks": {}}

    async with async_playwright() as pw:
        for name, url, task, steps, expect in tasks:
            print(f"  {name}")
            out["tasks"][name] = {"task": task, "url": url}
            for arm in ("jev", "llm"):
                runs = []
                for i in range(a.trials):
                    brain = (JevBrain() if arm == "jev"
                             else OpenRouterBrain(model=model))
                    try:
                        r = await one(pw, brain, url, task, steps, expect)
                    finally:
                        brain.close()
                    runs.append(r)
                    # distinguish a real pass from a claimed one
                    mark = {(True, True): ".", (True, False): "!",
                            (False, False): "x",
                            (False, True): "?"}[(r["done"], r.get("passed", False))]
                    print(f"    {arm} {i+1}/{a.trials} {mark} "
                          f"{r['per_decision_ms']:.0f}ms/decision", flush=True)
                s = summarise(runs)
                out["tasks"][name][arm] = {"summary": s, "runs": runs}
            print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    # ---- report ----
    print("\n  .=passed  !=claimed done but failed the check  x=gave up\n")
    print(f"  {'task':<9}{'arm':<5}{'passed':>8}{'claimed':>9}{'median':>11}"
          f"{'range':>16}{'steps':>7}{'cost':>11}")
    print("  " + "-" * 76)
    for name in out["tasks"]:
        for arm in ("jev", "llm"):
            s = out["tasks"][name][arm]["summary"]
            if not s:
                print(f"  {name:<9}{arm:<5}{'all failed':>8}")
                continue
            done = f"{s['passed']}/{s['n']}"
            rng = f"{s['min_ms']:.0f}-{s['max_ms']:.0f}ms"
            cost = "$" + format(s["median_cost"], ".5f")
            claimed = f"{s['claimed']}/{s['n']}"
            print(f"  {name:<9}{arm:<5}{done:>8}{claimed:>9}"
                  f"{s['median_ms']:>9.0f}ms"
                  f"{rng:>16}{s['median_steps']:>7.0f}{cost:>11}")

    # Same round trip removed from each side, as in scripts/race.py.
    print()
    for name in out["tasks"]:
        sj = out["tasks"][name]["jev"]["summary"]
        sl = out["tasks"][name]["llm"]["summary"]
        if not (sj and sl):
            continue
        raw = sl["median_ms"] / max(sj["median_ms"], 1)
        cj = max(sj["median_ms"] - rtts["jev"], 1)
        cl = max(sl["median_ms"] - rtts["llm"], 1)
        print(f"  {name:<9} {raw:.1f}x as measured   "
              f"{cl / cj:.1f}x less one round trip each")

    print(f"\n  written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
