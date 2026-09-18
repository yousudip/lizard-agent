"""Run the demo reel: a curated set of tasks, overlay on, recorded.

  uv run python scripts/demo.py                # all of them, to video/
  uv run python scripts/demo.py --only amazon  # just one
  uv run python scripts/demo.py --headed       # watch it live
  uv run python scripts/demo.py --dwell 1400   # hold each decision longer

Each clip lands in videos/<name>/. The overlay is real DOM, so Playwright's
own capture records it - no screen recorder needed.
"""

import argparse, asyncio, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.async_api import async_playwright          # noqa: E402
from lizard.agent import Agent                              # noqa: E402
from lizard.brain import JevBrain                           # noqa: E402

# Ordered for a reel: open on the one everybody understands, close on the
# one that shows the agent declining to do something.
DEMOS = [
    ("amazon", "https://www.amazon.in",
     "find a wireless headphone under Rs 2000 delivered within 2 days", 10,
     "Constraint satisfaction. Nobody told it Amazon has a delivery filter."),

    ("amazon-pin", "https://www.amazon.in",
     "wireless headphones under Rs 2000 which can get delivered to "
     "562125 under 2 days", 12,
     "Four constraints and two fields: price, PIN, delivery window, query."),

    ("pydocs", "https://docs.python.org/3/",
     "find the documentation page for the asyncio gather function", 8,
     "Precision: lands on the exact anchor and quotes the signature."),

    ("asyncio", "https://docs.python.org/3/",
     "find which exception asyncio.wait_for raises when it times out", 8,
     "Four pages deep, and the answer is a sentence quoted off the page."),

    ("github", "https://github.com/langchain-ai/langchain",
     "find the contributing guidelines document for this project", 8,
     "Cross-domain: github.com to docs.langchain.com, unprompted."),

    ("wikipedia", "https://en.wikipedia.org/wiki/Main_Page",
     "find the article about the Kolkata Metro", 6,
     "Two steps. Search terms chosen by fanned-out yes/no questions."),

    ("captcha", "https://pypi.org",
     "find the latest released version of the langchain package", 6,
     "Detects a bot challenge and stops rather than trying to get past it."),
]


async def run_one(pw, name, url, task, steps, note, a):
    print(f"\n  {'─' * 64}\n  {name}  —  {note}\n  task: {task}\n")
    browser = await pw.chromium.launch(headless=not a.headed)
    kw = {"viewport": {"width": 1440, "height": 900}}
    if not a.no_video:
        kw["record_video_dir"] = f"videos/{name}"
        kw["record_video_size"] = {"width": 1440, "height": 900}
    ctx = await browser.new_context(**kw)
    page = await ctx.new_page()

    res = None
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1800)
        brain = JevBrain()
        try:
            brain.warm()                 # pay TLS before the clock starts
            agent = Agent(page, brain, task, max_steps=steps,
                          overlay=True, dwell_ms=a.dwell)
            res = await agent.run()
        finally:
            brain.close()

        res.save(f"traces/demo-{name}.jsonl")
        print(f"\n  {'done' if res.done else 'stopped'}: {res.reason}")
        print(f"  url    : {res.final_url[:76]}")
        if res.answer:
            print(f"  ANSWER : {res.answer[:88]}")
            print(f"           confidence {res.answer_confidence:.2f}, "
                  f"present {res.answer_present:.2f}")
        if res.constraints:
            mark = {True: "PASS", False: "FAIL", None: "unknown"}[res.verified]
            print(f"  check  : {mark} — {res.verdict}")
        print(f"  {len(res.steps)} steps · {res.compute_ms:.0f} ms thinking · "
              f"{res.network_ms:.0f} ms network · "
              f"{res.wall_ms - res.jev_ms:.0f} ms page loads · "
              f"${res.cost_usd:.5f}")
        if res.wall_ms:
            print(f"  the model was {100 * res.compute_ms / res.wall_ms:.0f}% "
                  f"of elapsed time")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:110]}")
    finally:
        if not a.no_video:
            await page.wait_for_timeout(1400)
            await ctx.close()
        await browser.close()
    return name, res


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single demo by name")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dwell", type=int, default=1100,
                    help="ms to hold each decision on screen, for recording")
    a = ap.parse_args()

    demos = [d for d in DEMOS if not a.only or d[0] == a.only]
    if not demos:
        print(f"  no demo named {a.only!r}; have: "
              f"{', '.join(d[0] for d in DEMOS)}")
        return 1

    rows = []
    async with async_playwright() as pw:
        for name, url, task, steps, note in demos:
            rows.append(await run_one(pw, name, url, task, steps, note, a))

    print(f"\n  {'═' * 64}\n  reel")
    for name, res in rows:
        if res is None:
            print(f"    {name:<10} failed to run")
            continue
        tail = f"{res.answer[:44]}" if res.answer else res.reason[:44]
        print(f"    {name:<10} {len(res.steps)} steps  "
              f"{res.compute_ms:>5.0f}ms think  ${res.cost_usd:.5f}  {tail}")
    if not a.no_video:
        print(f"\n  clips in videos/<name>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
