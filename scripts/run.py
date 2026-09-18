"""Run Lizard on a task.

  uv run python scripts/run.py <url> "<task>" [--headed] [--steps N]
"""

import argparse, asyncio, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.async_api import async_playwright          # noqa: E402
from lizard.agent import Agent                              # noqa: E402
from lizard.config import CHROME_CDP_URL                    # noqa: E402
from lizard.brain import JevBrain                           # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("task")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--overlay", action="store_true", help="draw the on-page HUD")
    ap.add_argument("--dwell", type=int, default=0,
                    help="ms to hold on each decision, for recording")
    ap.add_argument("--video", metavar="DIR", help="record video to DIR")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--trace", default="traces/last.jsonl")
    a = ap.parse_args()

    print(f"\n  task: {a.task}\n  from: {a.url}\n")

    async with async_playwright() as pw:
        if CHROME_CDP_URL:
            browser = await pw.chromium.connect_over_cdp(CHROME_CDP_URL)
            ctx = browser.contexts[0]
            page = await ctx.new_page()
        else:
            browser = await pw.chromium.launch(headless=not a.headed)
            kw = {"viewport": {"width": 1440, "height": 900}}
            if a.video:
                kw["record_video_dir"] = a.video
                kw["record_video_size"] = {"width": 1440, "height": 900}
            ctx = await browser.new_context(**kw)
            page = await ctx.new_page()

        await page.goto(a.url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1500)

        brain = JevBrain()
        try:
            brain.warm()                     # pay TLS before the clock starts
            agent = Agent(page, brain, a.task, max_steps=a.steps,
                          overlay=a.overlay or bool(a.video), dwell_ms=a.dwell)
            res = await agent.run()
        finally:
            brain.close()

        p = res.save(a.trace)
        print(f"\n  {'done' if res.done else 'stopped'}: {res.reason}")
        print(f"  final url  : {res.final_url[:78]}")
        if res.answer:
            print(f"\n  ANSWER     : {res.answer[:100]}")
            print(f"  confidence : {res.answer_confidence:.2f}  "
                  f"(answer present on page: {res.answer_present:.2f})\n")
        print(f"  steps      : {len(res.steps)}")
        print(f"  jev total  : {res.jev_ms:.0f} ms "
              f"({100*res.jev_ms/max(res.wall_ms,1):.0f}% of wall)")
        print(f"    compute  : {res.compute_ms:.0f} ms  <- the model")
        print(f"    network  : {res.network_ms:.0f} ms  <- the Indian Ocean")
        print(f"  page loads : {res.wall_ms - res.jev_ms:.0f} ms")
        print(f"  wall       : {res.wall_ms:.0f} ms")
        print(f"  cost       : ${res.cost_usd:.6f}")
        if res.constraints:
            mark = {True: "PASS", False: "FAIL", None: "unknown"}[res.verified]
            print(f"  constraints: {res.constraints}")
            print(f"  verified   : {mark} — {res.verdict}")
        print(f"  trace      : {p}")
        if a.video:
            await page.wait_for_timeout(1200)
            await ctx.close()
            print(f"  video      : {a.video}/")
        await browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
