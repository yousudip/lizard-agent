"""Turn a live page into something Jev can read.

Jev has no eyes. It reads a string. So the job here is to compress a page
down to two things: a list of things that can be clicked, and enough text
to judge whether this page is the right place to be.

Every element keeps a stable integer id, written back onto the DOM as
data-lz. Jev returns an integer; we look up the selector. Jev never sees
or emits a selector, which is what keeps the action space closed.
"""

import re
from dataclasses import dataclass

MAX_ELEMENTS = 255  # Jev's ceiling on a single choice

# The JS runs in page context. It tags, measures and names every candidate,
# then hands back plain data.
_COLLECT_JS = r"""
(() => {
  const SEL = [
    'a[href]', 'button', 'input:not([type=hidden])', 'select', 'textarea',
    '[role=button]', '[role=link]', '[role=tab]', '[role=menuitem]',
    '[role=checkbox]', '[role=radio]', '[role=option]', '[role=searchbox]',
    '[role=combobox]', '[onclick]', 'summary', '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  const vis = (el, r) => {
    if (r.width < 4 || r.height < 4) return false;
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none') return false;
    if (parseFloat(s.opacity) < 0.05) return false;
    return true;
  };

  // A rough accessible-name computation: the same sources a screen reader
  // consults, in roughly the same priority order.
  const name = (el) => {
    const pick = (v) => (v || '').replace(/\s+/g, ' ').trim();
    let n = pick(el.getAttribute('aria-label'));
    if (n) return n;
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const t = lb.split(/\s+/).map(id => {
        const e = document.getElementById(id);
        return e ? pick(e.innerText) : '';
      }).filter(Boolean).join(' ');
      if (t) return t;
    }
    n = pick(el.innerText);
    if (n) return n;
    for (const a of ['title', 'placeholder', 'value', 'alt', 'name']) {
      n = pick(el.getAttribute(a));
      if (n) return n;
    }
    const img = el.querySelector('img[alt]');
    if (img) { n = pick(img.getAttribute('alt')); if (n) return n; }
    return '';
  };

  const role = (el) => {
    const r = el.getAttribute('role');
    if (r) return r;
    const t = el.tagName.toLowerCase();
    if (t === 'a') return 'link';
    if (t === 'input') return (el.type || 'text') === 'submit' ? 'button' : `input:${el.type||'text'}`;
    return t;
  };

  // Which landmark is this element in? Header/nav/footer elements are
  // site chrome and almost never the answer to a task.
  const zone = (el) => {
    if (el.closest('header,nav,footer,[role=banner],[role=navigation],[role=contentinfo]'))
      return 'chrome';
    if (el.closest('main,[role=main],[role=search],#search,[data-component-type="s-search-result"]'))
      return 'main';
    return 'body';
  };

  // An open dialog owns the interaction: while one is up, the controls
  // behind it are unreachable, and wandering off to them mid-flow is how
  // a multi-step form (set a location, confirm it) gets abandoned halfway.
  const dialog = [...document.querySelectorAll(
      '[role=dialog],[aria-modal=true],.a-popover:not([aria-hidden=true]),'
      + '[class*=modal]:not([aria-hidden=true])')]
    .find(d => {
      const r = d.getBoundingClientRect();
      const s = getComputedStyle(d);
      return r.width > 120 && r.height > 60
             && s.visibility !== 'hidden' && s.display !== 'none';
    }) || null;

  const vh = innerHeight, vw = innerWidth;
  const out = [];
  let i = 0;
  for (const el of document.querySelectorAll(SEL)) {
    const r = el.getBoundingClientRect();
    if (!vis(el, r)) continue;
    el.setAttribute('data-lz', String(i));
    out.push({
      id: i++,
      role: role(el),
      zone: zone(el),
      name: name(el).slice(0, 120),
      // in the visible viewport right now?
      inView: r.top < vh && r.bottom > 0 && r.left < vw && r.right > 0,
      y: Math.round(r.top + scrollY),
      typable: ['input', 'textarea'].includes(el.tagName.toLowerCase())
               || el.isContentEditable,
      inDialog: dialog ? dialog.contains(el) : false,
      disabled: !!el.disabled,
    });
  }

  const main = document.querySelector('main, [role=main], article') || document.body;
  return {
    url: location.href,
    title: document.title,
    text: (main.innerText || '').replace(/\n{3,}/g, '\n\n').trim(),
    hasDialog: !!dialog,
    scrollY: Math.round(scrollY),
    scrollMax: Math.round(document.body.scrollHeight - innerHeight),
    elements: out,
  };
})()
"""


@dataclass
class Element:
    id: int
    role: str
    name: str
    in_view: bool
    zone: str
    in_dialog: bool
    y: int
    typable: bool
    disabled: bool

    @property
    def selector(self) -> str:
        return f'[data-lz="{self.id}"]'

    def render(self) -> str:
        return f'{self.role} "{self.name}"' if self.name else f"{self.role} (unlabeled)"


@dataclass
class Percept:
    url: str
    title: str
    text: str
    scroll_y: int
    scroll_max: int
    elements: list[Element]   # already pruned
    total_found: int          # before pruning
    after_dedupe: int         # after collapsing duplicate targets
    has_dialog: bool = False  # an open modal owns the interaction


# Accelerator hints, skip links and other things that are only ever noise.
_JUNK = re.compile(
    r"shift, option|^skip to|^main content|show/hide shortcuts|^back to top",
    re.I,
)


def _relevance(el: Element, terms: list[str]) -> float:
    """Rank for pruning. Naive truncation drops the element you needed,
    so order by usefulness before cutting.

    Two lessons are baked into these weights, both learned the hard way:

    1. A large viewport bonus quietly deleted every search result below
       the fold - i.e. the entire point of a results page.
    2. Penalising header/nav/footer wholesale deleted Amazon's search box
       and GitHub's own "MIT license" link. Site chrome and the answer to
       the task are not disjoint sets. Target the junk, not the zone.
    """
    score = 0.0

    if _JUNK.search(el.name):
        score -= 200         # decisive: these are never the answer
    if el.zone == "chrome":
        score -= 15          # mild nudge, not a death sentence
    elif el.zone == "main":
        score += 40

    if el.in_view:
        score += 15
    if el.name:
        score += 30
    else:
        score -= 40
    if el.disabled:
        score -= 50

    low = el.name.lower()
    score += 25 * sum(1 for t in terms if t and t in low)

    # Long descriptive labels tend to be real content (product titles,
    # article headlines) rather than navigation.
    if len(el.name) > 40:
        score += 25

    if el.role in ("link", "button") or el.role.startswith("input"):
        score += 10
    score -= el.y / 5000
    return score


def _protected(el: Element, terms: list[str]) -> bool:
    """Never prune these, whatever the ranking says.

    Text inputs are the agent's only way to ask a question, and anything
    literally naming a task term is the likeliest answer on the page.
    """
    if el.typable and not el.disabled:
        return True
    low = el.name.lower()
    return bool(terms) and any(t in low for t in terms)


# Controls the executor has no way to operate. Offering them is worse than
# useless: the agent picks the price slider, the click does nothing, and the
# run burns its budget rediscovering that. Range and file inputs need drag
# and file-picker gestures this agent does not perform.
_UNUSABLE = ("input:range", "input:file", "input:color", "input:date")


def _dedupe(els: list[Element]) -> list[Element]:
    """Cards often expose the same target three times (image, title, price).
    Keep the best-labelled one per destination and free up the slots."""
    seen: dict[str, Element] = {}
    for el in els:
        key = (el.role, re.sub(r"[^a-z0-9]+", "", el.name.lower())[:60])
        if not key[1]:
            seen[f"__unnamed_{el.id}"] = el      # keep unnamed ones distinct
            continue
        prev = seen.get(key)
        if prev is None or len(el.name) > len(prev.name):
            seen[key] = el
    return list(seen.values())


async def perceive(page, task: str = "", limit: int = MAX_ELEMENTS) -> Percept:
    raw = await page.evaluate(_COLLECT_JS)

    els = [
        Element(
            id=e["id"], role=e["role"], name=e["name"], in_view=e["inView"],
            zone=e["zone"], in_dialog=e["inDialog"], y=e["y"],
            typable=e["typable"], disabled=e["disabled"],
        )
        for e in raw["elements"]
    ]
    total = len(els)

    terms = [w for w in task.lower().split() if len(w) > 3]
    els = [e for e in els if e.role not in _UNUSABLE]
    deduped = _dedupe(els)

    # While a dialog is open, it is the only thing the agent may touch.
    if raw.get("hasDialog"):
        inside = [e for e in deduped if e.in_dialog]
        if inside:
            deduped = inside

    keep = [e for e in deduped if _protected(e, terms) and not _JUNK.search(e.name)]
    keep = keep[:limit]
    taken = {e.id for e in keep}
    rest = sorted(
        (e for e in deduped if e.id not in taken),
        key=lambda e: -_relevance(e, terms),
    )
    kept = keep + rest[: limit - len(keep)]
    kept.sort(key=lambda e: e.id)        # restore document order for readability

    return Percept(
        url=raw["url"], title=raw["title"], text=raw["text"],
        scroll_y=raw["scrollY"], scroll_max=raw["scrollMax"],
        elements=kept, total_found=total, after_dedupe=len(deduped),
        has_dialog=bool(raw.get("hasDialog")),
    )


def render_state(p: Percept, task: str, history: list[str]) -> str:
    """The string Jev actually sees."""
    lines = [
        f"TASK: {task}",
        f"URL: {p.url}",
        f"TITLE: {p.title}",
        f"SCROLL: {p.scroll_y} of {p.scroll_max}",
        "",
        "PAGE TEXT:",
        p.text[:3000],
        "",
        (f"A DIALOG IS OPEN. Only its controls are listed; finish or close "
         f"it before anything else."
         if p.has_dialog else
         f"INTERACTIVE ELEMENTS ({len(p.elements)} shown of {p.total_found} found):"),
    ]
    lines += [f"  [{e.id}] {e.render()}" for e in p.elements]
    if history:
        lines += ["", "HISTORY:"] + [f"  {i}. {h}" for i, h in enumerate(history, 1)]
    return "\n".join(lines)
