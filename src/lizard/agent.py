"""The decision loop.

Jev decides, code acts. Every value Jev returns here is an integer or an
enum member, never text, so the action space is closed: it cannot name an
element that isn't on the page or emit a selector at all.

Everything numeric, temporal or mechanical stays in Python - loop
detection, step budgets, the safety denylist, scrolling. Jev is asked only
the things that genuinely require reading the page.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .constraints import Constraints, check, parse, prices_in
from .brain import Decision, JevBrain
from .overlay import draw
from .perceive import perceive, render_state

# Controls that do something irreversible or spend money. Jev is also asked
# whether an action looks risky, but this regex is the backstop: a model
# judgement should never be the only thing standing between an agent and a
# purchase.
DENYLIST = re.compile(
    r"\b(buy now|place (your )?order|pay now|proceed to (pay|checkout)|"
    r"confirm (order|purchase|payment)|add payment|"
    r"delete (account|item|order)|deactivate|close account|"
    r"unsubscribe|sign out|log out)\b",
    re.I,
)

# Not dangerous, just counterproductive: controls that undo work the agent
# has already done. The first version of DENYLIST matched a bare "remove"
# and killed a run over "Remove the filter Get It in 2 Days" - which is not
# a destructive action, only a wrong one. Wrong choices deserve exclusion
# and another go; irreversible ones deserve a full stop.
AVOID = re.compile(
    r"\b(remove the filter|clear (all|filters?)|reset|undo|"
    r"remove from (cart|list))\b",
    re.I,
)

STOPWORDS = {
    "find", "a", "an", "the", "which", "will", "be", "to", "me", "my", "for",
    "in", "on", "of", "and", "or", "with", "under", "over", "within", "that",
    "get", "show", "search", "look", "up", "is", "are", "what", "whats",
}


def search_terms(task: str) -> str:
    """Build a query from the user's own words - selection, not generation.

    Nothing here is invented: every token in the output appeared in the
    task. Constraint clauses (prices, durations) are dropped because they
    belong to the deterministic filters, not the search box.
    """
    cleaned = re.sub(r"\b(under|below|above|within)\s+\S+\s*\d*\b", " ", task, flags=re.I)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    words = [w for w in cleaned.split() if w.lower() not in STOPWORDS and not w.isdigit()]
    return " ".join(words[:8])


@dataclass
class Step:
    n: int
    url: str
    action: str
    action_conf: float
    target_id: int | None
    target_name: str
    target_conf: float
    latency_ms: float
    compute_ms: float
    network_ms: float
    cost_usd: float
    elements_shown: int
    elements_found: int
    signals: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class Result:
    task: str
    steps: list[Step]
    done: bool
    reason: str
    final_url: str
    final_text: str
    wall_ms: float
    constraints: str = ""
    verified: bool | None = None
    verdict: str = ""

    @property
    def jev_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def compute_ms(self) -> float:
        """Time the model actually spent thinking."""
        return sum(s.compute_ms for s in self.steps)

    @property
    def network_ms(self) -> float:
        return sum(s.network_ms for s in self.steps)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for s in self.steps:
                f.write(json.dumps(asdict(s)) + "\n")
        return p


class Agent:
    def __init__(self, page, brain, task: str, max_steps: int = 20,
                 min_confidence: float = 0.25, verbose: bool = True,
                 overlay: bool = False, dwell_ms: int = 0):
        self.page, self.brain, self.task = page, brain, task
        self.max_steps, self.min_confidence = max_steps, min_confidence
        self.verbose = verbose
        self.overlay = overlay
        self.dwell_ms = dwell_ms      # pause after painting, for recording
        self.constraints: Constraints = parse(task)
        self.history: list[str] = []
        self.steps: list[Step] = []
        self._seen: set[tuple] = set()
        self._scrolls = 0          # consecutive; scrolling is the fallback,
                                   # so it needs its own escape hatch
        self._dead: set[str] = set()   # elements that failed to act; a click
                                       # that times out must not be offered
                                       # again or the agent thrashes on it

    async def run(self) -> Result:
        t0 = time.perf_counter()
        reason = "step budget exhausted"
        done = False

        for n in range(1, self.max_steps + 1):
            p = await perceive(self.page, self.task)
            # Drop elements that have already failed, and controls that
            # would undo work. Both arms see the same filtered list.
            p.elements = [e for e in p.elements
                          if e.render() not in self._dead
                          and not AVOID.search(e.name)] or p.elements
            state = render_state(p, self.task, self.history[-6:])
            if self.constraints:
                state += f"\n\nCONSTRAINTS (already parsed): {self.constraints.describe()}" 
            d: Decision = self.brain.decide(state, p.elements)

            action = d.action
            a_conf = d.confidence
            by_id = {e.id: e for e in p.elements}
            tgt = by_id.get(d.target) if d.target is not None else None
            t_conf = d.confidence

            step = Step(
                n=n, url=p.url, action=action, action_conf=a_conf,
                target_id=tgt.id if tgt else None,
                target_name=tgt.render() if tgt else "",
                target_conf=t_conf,
                latency_ms=d.latency_ms, compute_ms=d.compute_ms,
                network_ms=d.network_ms, cost_usd=d.cost_usd,
                elements_shown=len(p.elements), elements_found=p.total_found,
                signals=d.signals,
            )

            if self.overlay:
                await self._paint(p, d, action, tgt, t_conf, step)

            # ---- deterministic guards, before anything touches the page ----
            if tgt and DENYLIST.search(tgt.name):
                step.action, step.note = "refused", f"denylist matched: {tgt.name[:40]}"
                if self.overlay:
                    await self._paint(p, d, "refused", tgt, t_conf, step, denied=True)
                self._record(step); reason = step.note; break

            if tgt and d.signals.get("risky", 0) > 0.5:
                step.action = "refused"
                step.note = f"flagged risky ({d.signals['risky']:.2f})"
                if self.overlay:
                    await self._paint(p, d, "refused", tgt, t_conf, step, denied=True)
                self._record(step); reason = step.note; break

            if d.signals.get("looping", 0) > 0.7:
                step.note = "loop suspected; scrolling instead"
                action = "scroll"

            if action in ("click", "type") and t_conf < self.min_confidence:
                # Low confidence falls back to a heuristic, not to an LLM.
                step.note = f"low target confidence {t_conf:.2f}; scrolling instead"
                action = "scroll"

            key = (p.url, action, tgt.id if tgt else None)
            if key in self._seen and action != "scroll":
                step.note = "repeat of an earlier action; scrolling instead"
                action = "scroll"
            self._seen.add(key)

            if action == "scroll":
                self._scrolls += 1
                if self._scrolls > 2 and p.scroll_y >= p.scroll_max - 50:
                    step.action, step.note = "stuck", "bottom of page, no progress"
                    self._record(step); reason = step.note; break
                if self._scrolls > 3:
                    step.action, step.note = "stuck", "scrolled repeatedly without progress"
                    self._record(step); reason = step.note; break
            else:
                self._scrolls = 0

            if action == "done":
                step.action = "done"
                self._record(step)
                done, reason = True, "Jev reported the task complete"
                break

            # ---- act ----
            try:
                await self._execute(action, tgt, p)
            except Exception as e:
                step.note = f"{type(e).__name__}: {str(e)[:60]}"
                if tgt:
                    # Never offer this element again - thrashing on a dead
                    # control is how a run burns its whole budget.
                    self._dead.add(tgt.render())
            step.action = action
            self._record(step)
            await self.page.wait_for_timeout(600)

        text = await self.page.evaluate("(document.querySelector('main,[role=main],article')||document.body).innerText")
        text = text or ""
        ok, verdict = check(text[:6000], self.constraints,
                            price=await self._dom_price())
        return Result(
            task=self.task, steps=self.steps, done=done, reason=reason,
            final_url=self.page.url, final_text=text[:2000],
            wall_ms=(time.perf_counter() - t0) * 1000,
            constraints=self.constraints.describe(), verified=ok, verdict=verdict,
        )

    async def _follow_new_tab(self) -> bool:
        """Search results often carry target=_blank, so a successful click
        opens a tab the agent would otherwise never look at. Follow it."""
        ctx = self.page.context
        for other in ctx.pages:
            if other is not self.page and not other.is_closed():
                await other.bring_to_front()
                try:
                    await other.wait_for_load_state("domcontentloaded", timeout=8000)
                except Exception:
                    pass
                self.page = other
                self.history.append("followed link into a new tab")
                return True
        return False

    async def _paint(self, p, d, action, tgt, t_conf, step, denied=False) -> None:
        await draw(
            self.page, task=self.task, step=step.n, action=action, conf=t_conf,
            pick=tgt.id if tgt else None,
            candidates=[e.id for e in p.elements],
            shown=len(p.elements), found=p.total_found,
            compute=d.compute_ms, network=d.network_ms,
            signals=step.signals, cost=self.brain.total_cost, denied=denied,
        )
        if self.dwell_ms:
            await self.page.wait_for_timeout(self.dwell_ms)

    async def _dom_price(self) -> float | None:
        """Read the price out of the markup rather than the rendered text.

        Sites label their prices - schema.org metadata, or a site-specific
        class - and those labels are far more trustworthy than guessing
        which number on the page is the one that matters.
        """
        try:
            raw = await self.page.evaluate("""(() => {
              const SEL = [
                'meta[property="product:price:amount"]',
                '[itemprop=price]',
                '.a-price .a-offscreen',              // Amazon
                '[data-testid*=price]', '[class*=selling-price]',
              ];
              for (const sel of SEL) {
                for (const el of document.querySelectorAll(sel)) {
                  const v = el.content || el.getAttribute('content')
                            || el.textContent || '';
                  if (v && /\d/.test(v)) return v.trim();
                }
              }
              return null;
            })()""")
        except Exception:
            return None
        if not raw:
            return None
        vals = prices_in(raw) or prices_in("₹" + raw)
        return vals[0] if vals else None

    async def _execute(self, action: str, tgt, p) -> None:
        if action == "click" and tgt:
            before = self.page.url
            await self.page.click(tgt.selector, timeout=6000)
            self.history.append(f'clicked [{tgt.id}] {tgt.render()}')
            await self.page.wait_for_timeout(900)
            if self.page.url == before:
                await self._follow_new_tab()
        elif action == "type" and tgt:
            q = search_terms(self.task)
            await self.page.fill(tgt.selector, q, timeout=6000)
            await self.page.press(tgt.selector, "Enter")
            self.history.append(f'typed "{q}" into [{tgt.id}]')
        elif action == "back":
            await self.page.go_back(timeout=8000)
            self.history.append("went back")
        else:
            await self.page.mouse.wheel(0, 900)
            self.history.append("scrolled down")

    def _record(self, s: Step) -> None:
        self.steps.append(s)
        if self.verbose:
            tail = f"  {s.note}" if s.note else ""
            tgt = f" [{s.target_id}] {s.target_name[:38]}" if s.target_id is not None else ""
            print(f"  {s.n:>2}. {s.action:<8}{tgt:<46} "
                  f"conf={s.target_conf:.2f} "
                  f"{s.compute_ms:>4.0f}ms+{s.network_ms:>4.0f}net{tail}")
