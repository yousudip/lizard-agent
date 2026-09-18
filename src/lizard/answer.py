"""Extractive answering.

Jev cannot write a sentence, so the answer is not composed - it is
*located*. Code collects the candidate text blocks on the page, Jev picks
the one holding the answer, and code returns that block verbatim.

The result is a quotation from the page rather than a paraphrase of it,
which means it cannot be subtly wrong in the way a generated summary can.
For "what version is langchain on" or "what does this cost", the answer
really is a span someone could point at.
"""

from __future__ import annotations

from dataclasses import dataclass

from .jev import choice, noul

# Blocks likely to hold a fact, in rough order of how often they do.
_COLLECT_JS = r"""
(() => {
  const SEL = 'h1,h2,h3,p,li,td,th,dd,dt,[class*=price],[class*=version],'
            + '[class*=release],[data-testid],figcaption,strong';
  const seen = new Set();
  const out = [];
  for (const el of document.querySelectorAll(SEL)) {
    if (out.length >= 200) break;
    const r = el.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) continue;
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none') continue;
    // leaf-ish blocks only: a wrapper repeats its children's text
    if (el.querySelectorAll(SEL).length > 0) continue;
    const t = (el.innerText || '').replace(/\s+/g, ' ').trim();
    if (t.length < 2 || t.length > 220) continue;
    if (seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  return out;
})()
"""


@dataclass
class Answer:
    text: str
    confidence: float
    present: float        # Jev's probability the answer is on this page at all

    def __str__(self) -> str:
        return self.text


async def extract(page, jev, task: str) -> Answer | None:
    """Find the span on the current page that answers the task."""
    try:
        blocks: list[str] = await page.evaluate(_COLLECT_JS)
    except Exception:
        return None
    if not blocks:
        return None

    blocks = blocks[:255]                     # Jev's choice ceiling
    v = jev.ask(
        f"TASK: {task}\n\nThe agent has finished navigating. Below are the "
        f"text blocks on the page it ended on.\n\n"
        + "\n".join(f"[{i}] {b}" for i, b in enumerate(blocks)),
        {
            "block": choice(
                "Which block contains the answer to the task?",
                {str(i): b for i, b in enumerate(blocks)},
            ),
            "present": noul("Does this page actually contain the answer "
                            "to the task at all?"),
        },
    )

    idx = int(v["block"].value)
    if not 0 <= idx < len(blocks):
        return None
    return Answer(text=blocks[idx],
                  confidence=v["block"].confidence,
                  present=v["present"].value)
