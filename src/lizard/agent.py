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
from .answer import extract
from .brain import Decision, JevBrain
from .overlay import draw
from .preconditions import apply as apply_preconditions
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

# Bot challenges. Hitting one is a stop condition, not an obstacle to work
# around: the agent reports it and ends the run. Detecting it explicitly
# also keeps a CAPTCHA from looking like a model failure - without this the
# agent thrashes against a page it can never get past, and the trace makes
# it look as though it simply could not decide what to do.
CAPTCHA = re.compile(
    r"enter the characters|unusual traffic|verify you are|are you a robot|"
    r"security check|captcha|prove you.{0,10}human|"
    r"to discuss automated access",
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


def candidate_terms(task: str) -> list[str]:
    """Words from the task that could plausibly belong in a search box.

    Constraint clauses (prices, durations) are dropped here because they
    belong to the deterministic filters, not the query.
    """
    cleaned = re.sub(r"\b(under|below|above|within)\s+\S+\s*\d*\b", " ", task, flags=re.I)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return [w for w in cleaned.split()
            if w.lower() not in STOPWORDS and not w.isdigit()][:12]


def search_terms(task: str, keep: set[str] | None = None) -> str:
    """Build a query from the user's own words - selection, not generation.

    Nothing here is invented: every token appeared in the task. `keep`,
    when supplied, is the subset Jev judged to be naming the thing rather
    than describing it - see JevBrain.pick_terms.
    """
    words = candidate_terms(task)
    if keep:
        words = [w for w in words if w.lower() in keep] or words
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
    answer: str = ""
    answer_confidence: float = 0.0
    answer_present: float = 0.0
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
                 overlay: bool = False, dwell_ms: int = 0,
                 done_confidence: float = 0.6, scroll_budget: int = 3,
                 allow_search: bool = True):
        self.page, self.brain, self.task = page, brain, task
        self.max_steps, self.min_confidence = max_steps, min_confidence
        self.verbose = verbose
        self.overlay = overlay
        self.dwell_ms = dwell_ms      # pause after painting, for recording
        self.done_confidence = done_confidence
        # How long to tolerate scrolling before calling a run stuck.
        # Exploration tasks legitimately wander; lookups should not.
        self.scroll_budget = scroll_budget
        # A rule you can enforce should not be left to the model. Told in
        # prose to reach a page "by clicking links only", the agent typed
        # the destination into the search box on step 2 - which is a
        # perfectly sensible way to reach a page and exactly what it was
        # asked not to do. Removing the box removes the question.
        self.allow_search = allow_search
        self._done_rejected = 0
        self.constraints: Constraints = parse(task)
        self._terms: str | None = None      # resolved lazily, once
        self.history: list[str] = []
        self.steps: list[Step] = []
        self._seen: set[tuple] = set()
        self._scrolls = 0          # consecutive; scrolling is the fallback,
                                   # so it needs its own escape hatch
        self._clicked: set[str] = set()      # controls already actioned
        self._blocked: dict[str, int] = {}   # times an action was suppressed
        self._dead: set[str] = set()   # elements that failed to act; a click
                                       # that times out must not be offered
                                       # again or the agent thrashes on it

    async def run(self) -> Result:
        t0 = self._t0 = time.perf_counter()

        # Mechanical, known-in-advance setup happens before the loop, so
        # the agent's steps are spent on judgement rather than on
        # rediscovering a fixed modal sequence.
        if self.constraints:
            for note in await apply_preconditions(self.page, self.constraints):
                self.history.append(note)
                if self.verbose:
                    print(f"   ·  {note}")
        reason = "step budget exhausted"
        done = False

        for n in range(1, self.max_steps + 1):
            p = await self._perceive()
            # Drop elements that have already failed, and controls that
            # would undo work. Both arms see the same filtered list.
            # A control that has already been clicked does not get offered
            # again. Applying a filter does not remove its link from the
            # sidebar, so without this the agent re-clicks "Get It in 2
            # Days" it has already applied, twice, and spends the rest of
            # the run oscillating between two delivery filters.
            p.elements = [e for e in p.elements
                          if e.render() not in self._dead
                          and e.render() not in self._clicked
                          and not AVOID.search(e.name)] or p.elements

            if not self.allow_search:
                p.elements = [e for e in p.elements
                              if not (e.typable
                                      or "search" in e.role.lower()
                                      or "search" in e.name.lower())
                              ] or p.elements
            # A bot challenge ends the run. We do not attempt to solve it.
            if CAPTCHA.search(p.text[:1500]) or any(
                    CAPTCHA.search(e.name) for e in p.elements[:60]):
                self._record(Step(
                    n=n, url=p.url, action="captcha", action_conf=0.0,
                    target_id=None, target_name="", target_conf=0.0,
                    latency_ms=0.0, compute_ms=0.0, network_ms=0.0,
                    cost_usd=0.0, elements_shown=len(p.elements),
                    elements_found=p.total_found,
                    note="bot challenge detected; stopping"))
                reason = "hit a bot challenge (not bypassed)"
                break

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

            # Key on what the element *is*, not its index. data-lz ids are
            # reassigned on every perception pass, so an id-keyed memory
            # never matches and the agent will happily click the same
            # button twice - which is how a run gets past "Go", applies the
            # PIN, then clicks "Go" again and loses the thread.
            key = (p.url, action, tgt.render() if tgt else None)
            if key in self._seen and action != "scroll":
                step.note = "repeat of an earlier action; scrolling instead"
                action = "scroll"
                # Suppression alone is not enough: the element stays in the
                # action space, gets chosen again next step, is suppressed
                # again, and the run scrolls to its death. The blacklist
                # used to fill only on a raised exception, which never
                # happens for an action that was stopped before it ran.
                if tgt:
                    self._blocked[tgt.render()] = \
                        self._blocked.get(tgt.render(), 0) + 1
                    if self._blocked[tgt.render()] >= 2:
                        self._dead.add(tgt.render())
                        step.note = ("chosen repeatedly without progress; "
                                     "removing it from the action space")
            self._seen.add(key)

            if action == "scroll":
                self._scrolls += 1
                if (self._scrolls > self.scroll_budget - 1
                        and p.scroll_y >= p.scroll_max - 50):
                    step.action, step.note = "stuck", "bottom of page, no progress"
                    self._record(step); reason = step.note; break
                if self._scrolls > self.scroll_budget:
                    step.action, step.note = "stuck", "scrolled repeatedly without progress"
                    self._record(step); reason = step.note; break
            else:
                self._scrolls = 0

            # The answer being visibly present is itself a stopping
            # condition. Without this the agent lands on the right anchor,
            # reports answer_here at 0.86, and then scrolls past it until
            # the loop calls the run stuck - a navigation success recorded
            # as a failure.
            if (action != "done"
                    and d.signals.get("answer_here", 0) >= 0.8
                    and d.signals.get("progress", 0) >= 1.6
                    and n > 1):
                step.action = "done"
                step.note = (f"answer visible on this page "
                             f"(answer_here={d.signals['answer_here']:.2f})")
                self._record(step)
                done, reason = True, "answer located on the page"
                break

            if action == "done":
                # "done" is a claim, and a hesitant one deserves scepticism.
                # Left ungated, the agent stops on a search-results page
                # that merely *contains* plausible answers - `complete` sat
                # at 0.52 there, which is the model saying it is unsure,
                # not that it has finished. Push it to go one level deeper.
                unsure = d.signals.get("complete", 1.0) < self.done_confidence
                if unsure and self._done_rejected < 2 and n < self.max_steps:
                    self._done_rejected += 1
                    step.note = (f"done claimed at complete="
                                 f"{d.signals.get('complete', 0):.2f}; "
                                 f"looking closer")
                    action = "click" if tgt else "scroll"
                else:
                    step.action = "done"
                    self._record(step)
                    done, reason = True, "task reported complete"
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
            # Let modals close and re-renders settle before looking again.
            await self.page.wait_for_timeout(900)
            try:
                await self.page.wait_for_load_state("networkidle", timeout=3500)
            except Exception:
                pass

        text = await self.page.evaluate("(document.querySelector('main,[role=main],article')||document.body).innerText")
        text = text or ""
        ok, verdict = check(text[:6000], self.constraints,
                            price=await self._dom_price())

        # Locate the answer rather than compose it.
        ans = None
        if getattr(self.brain, "jev", None) is not None:
            try:
                ans = await extract(self.page, self.brain.jev, self.task)
            except Exception:
                ans = None
        return Result(
            task=self.task, steps=self.steps, done=done, reason=reason,
            final_url=self.page.url, final_text=text[:2000],
            wall_ms=(time.perf_counter() - t0) * 1000,
            constraints=self.constraints.describe(), verified=ok, verdict=verdict,
            answer=ans.text if ans else "",
            answer_confidence=ans.confidence if ans else 0.0,
            answer_present=ans.present if ans else 0.0,
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
            cum_compute=sum(x.compute_ms for x in self.steps) + d.compute_ms,
            wall=(time.perf_counter() - self._t0) * 1000,
        )
        if self.dwell_ms:
            await self.page.wait_for_timeout(self.dwell_ms)

    async def _perceive(self):
        """Read the page, tolerating a navigation landing mid-read.

        A click can trigger a redirect that tears down the execution
        context while the accessibility snapshot is still running. That is
        normal browsing, not an error, so wait for the new page and look
        again.
        """
        for attempt in range(3):
            try:
                return await perceive(self.page, self.task)
            except Exception as e:
                if "context was destroyed" not in str(e) and attempt == 2:
                    raise
                try:
                    await self.page.wait_for_load_state(
                        "domcontentloaded", timeout=8000)
                except Exception:
                    pass
                await self.page.wait_for_timeout(700)
        return await perceive(self.page, self.task)

    def _value_for(self, tgt) -> str:
        """Decide what to type into this particular field.

        Always typing the search query was fine while every task had one
        box to fill. A task like "headphones under Rs 2000 delivered to
        562125" has two: the PIN belongs in the location field and the
        query belongs in the search box, and putting either in the other
        gets nowhere.
        """
        options = {"query": f'the search terms: "{self._search_query()}"'}
        if self.constraints.pincode:
            options["pincode"] = (f"the postal/PIN code "
                                  f"{self.constraints.pincode}")
        if self.constraints.max_price is not None:
            options["max_price"] = (f"the price ceiling "
                                    f"{self.constraints.max_price:.0f}")
        if len(options) == 1 or not hasattr(self.brain, "pick_value"):
            return self._search_query()

        try:
            pick = self.brain.pick_value(self.task, tgt.render(), options)
        except Exception:
            return self._search_query()

        if pick == "pincode":
            return self.constraints.pincode or self._search_query()
        if pick == "max_price":
            return f"{self.constraints.max_price:.0f}"
        return self._search_query()

    def _search_query(self) -> str:
        """The query to type. Computed once, then reused."""
        if self._terms is not None:
            return self._terms
        words = candidate_terms(self.task)
        keep = None
        if len(words) > 2 and hasattr(self.brain, "pick_terms"):
            try:
                keep = self.brain.pick_terms(self.task, words)
            except Exception:
                keep = None
        self._terms = search_terms(self.task, keep)
        return self._terms

    async def _dom_price(self) -> float | None:
        """Read the price out of the markup rather than the rendered text.

        Sites label their prices - schema.org metadata, or a site-specific
        class - and those labels are far more trustworthy than guessing
        which number on the page is the one that matters.
        """
        try:
            raw = await self.page.evaluate(r"""(() => {
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
            # Navigate in place. Search results routinely carry
            # target="_blank"; letting them open a tab means the agent has
            # to chase it, and Playwright records each page to its own
            # video file, so one run arrives as two half-clips.
            try:
                await self.page.eval_on_selector(
                    tgt.selector, "el => el.removeAttribute('target')")
            except Exception:
                pass
            try:
                await self.page.click(tgt.selector, timeout=5000)
            except Exception:
                # Some controls are visible to the accessibility tree but
                # not to a real mouse click - covered by an overlay, or
                # zero-sized with a clickable parent. Dispatching the event
                # directly rescues most of them, and it is the difference
                # between a filter being applied and a wasted step.
                await self.page.dispatch_event(tgt.selector, "click",
                                               timeout=3000)
            self.history.append(f'clicked [{tgt.id}] {tgt.render()}')
            self._clicked.add(tgt.render())
            await self.page.wait_for_timeout(900)
            if self.page.url == before:
                await self._follow_new_tab()
        elif action == "type" and tgt:
            q = self._value_for(tgt)
            await self.page.fill(tgt.selector, q, timeout=6000)
            # Enter submits a search box. It does not submit Amazon's
            # location modal, which wants its Apply button clicked - and
            # pressing Enter there silently leaves the PIN unapplied while
            # looking like it worked. Only auto-submit true search fields;
            # otherwise leave the next move to the agent, which can see
            # the Apply button in its action space.
            is_search = ("search" in tgt.role.lower()
                         or "search" in tgt.name.lower())
            if is_search:
                await self.page.press(tgt.selector, "Enter")
            self.history.append(
                f'typed "{q}" into [{tgt.id}] {tgt.render()}'
                + ("" if is_search else " - now find the button that applies it")
            )
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
