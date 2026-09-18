"""Race the Jev agent against an LLM agent on the same task.

The two arms share everything except the decision-maker: same perception
layer, same 255-element cap, same executor, same deterministic guards.
Anything else would make the result meaningless.

  uv run python scripts/race.py <url> "<task>" [--headed] [--sequential]
"""

import argparse, asyncio, socket, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.async_api import async_playwright          # noqa: E402
from lizard.agent import Agent                              # noqa: E402
from lizard.brain import JevBrain                           # noqa: E402


async def arm(pw, label, brain, url, task, steps, headed, x, video):
    browser = await pw.chromium.launch(
        headless=not headed,
        args=[f"--window-position={x},0", "--window-size=960,1040"],
    )
    kw = {"viewport": {"width": 940, "height": 940}}
    if video:
        kw["record_video_dir"] = f"{video}/{label}"
        kw["record_video_size"] = {"width": 940, "height": 940}
    ctx = await browser.new_context(**kw)
    page = await ctx.new_page()
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(1200)

    brain.warm()
    agent = Agent(page, brain, task, max_steps=steps, overlay=True, verbose=False)
    res = await agent.run()
    res.save(f"traces/race-{label}.jsonl")

    if video:
        await page.wait_for_timeout(900)
        await ctx.close()
    await browser.close()
    return label, res


def rtt_ms(host: str, port: int = 443, n: int = 3) -> float:
    """One TCP round trip to a host - the floor on any request to it.

    This matters more than it looks. The two services are not equidistant:
    OpenRouter fronts its API with a CDN that has an Indian PoP, while
    TypeSafe serves from the US. Comparing raw latencies from Bengaluru
    therefore charges Jev a geographic tax the baseline never pays, and
    understates it by roughly a quarter of a second per call.
    """
    out = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            socket.create_connection((host, port), timeout=5).close()
            out.append((time.perf_counter() - t) * 1000)
        except OSError:
            pass
    return statistics.median(out) if out else 0.0


def report(rows, rtt_jev: float, rtt_llm: float):
    print(f"\n  {'':<10}{'steps':>7}{'wall':>10}{'thinking':>11}"
          f"{'per step':>11}{'cost':>12}{'outcome':>10}")
    print("  " + "-" * 71)
    for label, r in rows:
        per = r.jev_ms / max(len(r.steps), 1)
        print(f"  {label:<10}{len(r.steps):>7}{r.wall_ms/1000:>9.1f}s"
              f"{r.jev_ms/1000:>10.1f}s{per:>10.0f}ms"
              f"{'$' + format(r.cost_usd, '.5f'):>12}"
              f"{'done' if r.done else 'stopped':>10}")

    if len(rows) != 2:
        return
    (_, a), (_, b) = rows                          # a = jev, b = llm
    na, nb = len(a.steps) or 1, len(b.steps) or 1
    pa, pb = a.jev_ms / na, b.jev_ms / nb           # per-decision wall

    print()
    print(f"  as measured from here")
    print(f"    thinking : {pb / max(pa, 1):.1f}x faster per decision")
    if a.cost_usd > 0:
        print(f"    cost     : {b.cost_usd / max(a.cost_usd, 1e-9):.1f}x cheaper")
    print(f"    wall     : {b.wall_ms / max(a.wall_ms, 1):.1f}x faster "
          f"(page loads dilute it - that is the point)")

    # Strip one round trip from each side. Not a correction to be hidden:
    # a user in India really does pay Jev's 250ms, so both numbers belong
    # in any honest write-up.
    if rtt_jev and rtt_llm:
        ca, cb = max(pa - rtt_jev, 1.0), max(pb - rtt_llm, 1.0)
        print()
        print(f"  network floor: jev {rtt_jev:.0f}ms/call, "
              f"baseline {rtt_llm:.0f}ms/call "
              f"({'not ' if abs(rtt_jev-rtt_llm) > 40 else ''}comparable)")
        print(f"  less one round trip each")
        print(f"    thinking : {cb / ca:.1f}x faster  "
              f"({ca:.0f}ms vs {cb:.0f}ms per decision)")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("task")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--model", default=None,
                    help="OpenRouter model id "
                         "(default: google/gemini-2.5-flash-lite)")
    ap.add_argument("--video", metavar="DIR")
    ap.add_argument("--sequential", action="store_true",
                    help="run one after the other: slower, but the two arms "
                         "do not compete for bandwidth")
    a = ap.parse_args()

    # Preflight the baseline: the SDK only fails at call time, and a stack
    # trace halfway through a race is a poor way to learn the key is missing.
    # Preflight the baseline: a stack trace halfway through a race is a
    # poor way to learn the key is missing.
    from lizard.openrouter import OpenRouterBrain
    from lizard.config import OPENROUTER_MODEL
    model = a.model or OPENROUTER_MODEL
    try:
        llm = OpenRouterBrain(model=model)
        llm.decide("TASK: preflight\nINTERACTIVE ELEMENTS (0 shown of 0 found):")
        llm.calls, llm.total_ms, llm.total_cost = 0, 0.0, 0.0
    except Exception as e:
        msg = str(e)
        print("\n  LLM baseline unavailable — the race needs both arms.\n")
        if "OPENROUTER_API_KEY" in msg or "401" in msg or "auth" in msg.lower():
            print("  Add your OpenRouter key to .env:")
            print("      OPENROUTER_API_KEY=sk-or-...\n")
            print("  The Jev arm alone still runs:")
            print(f'      uv run python scripts/run.py "{a.url}" "{a.task}"\n')
        else:
            print(f"  {type(e).__name__}: {msg[:300]}\n")
        return 1

    print(f"\n  task : {a.task}\n  from : {a.url}\n  vs   : {model}\n")

    async with async_playwright() as pw:
        jobs = [
            arm(pw, "jev", JevBrain(), a.url, a.task, a.steps, a.headed, 0, a.video),
            arm(pw, "llm", llm, a.url, a.task, a.steps, a.headed, 960, a.video),
        ]
        if a.sequential:
            rows = [await jobs[0], await jobs[1]]
        else:
            rows = list(await asyncio.gather(*jobs))

    rj, rl = rtt_ms("api.typesafe.ai"), rtt_ms("openrouter.ai")
    report(rows, rj, rl)
    print(f"\n  traces: traces/race-jev.jsonl, traces/race-llm.jsonl")
    if a.video:
        print(f"  video : {a.video}/jev/, {a.video}/llm/")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
