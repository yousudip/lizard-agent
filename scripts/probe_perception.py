import asyncio, sys
sys.path.insert(0, "src")
from playwright.async_api import async_playwright
from lizard.perceive import perceive, render_state

TARGETS = [
    ("GitHub repo",   "https://github.com/langchain-ai/langchain", "find the license of this repo"),
    ("Python docs",   "https://docs.python.org/3/library/asyncio-task.html", "find asyncio gather documentation"),
    ("Amazon search", "https://www.amazon.in/s?k=wireless+headphones", "wireless headphone under 2000 delivered in 2 days"),
    ("HN front page", "https://news.ycombinator.com", "find the top story about AI"),
]
async def main():
    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        pg = await b.new_page(viewport={"width":1440,"height":900})
        print(f"{'site':16} {'found':>6} {'dedup':>6} {'kept':>5} {'named':>6} {'tokens':>7} {'$/call':>9}")
        print("-"*62)
        tot = 0
        for label, url, task in TARGETS:
            try:
                await pg.goto(url, wait_until="domcontentloaded", timeout=25000)
                await pg.wait_for_timeout(2200)
                p = await perceive(pg, task)
                st = render_state(p, task, [])
                named = 100*sum(1 for e in p.elements if e.name)//max(len(p.elements),1)
                tok = len(st)//4; tot += tok
                print(f"{label:16} {p.total_found:>6} {p.after_dedupe:>6} {len(p.elements):>5} {named:>5}% {tok:>7} ${tok/1e6*0.042:>8.6f}")
            except Exception as ex:
                print(f"{label:16} FAILED {type(ex).__name__}: {str(ex)[:40]}")
        print("-"*62)
        print(f"{'30-step task':16} {'':>6} {'':>6} {'':>5} {'':>6} {30*tot//4:>7} ${30*(tot/4)/1e6*0.042:>8.6f}")
        await b.close()
asyncio.run(main())
