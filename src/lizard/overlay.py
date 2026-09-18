"""On-page instrumentation.

Without this the demo is a browser clicking quickly, which looks like a
hardcoded script - the opposite of the point. The overlay makes the
decision visible: what was considered, what was chosen, how sure it was,
and where the milliseconds actually went.

It is injected as real DOM, so Playwright's own video capture records it.
"""

from __future__ import annotations

_CSS = """
#lz-hud{position:fixed;top:0;right:0;z-index:2147483647;width:300px;
 font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
 background:rgba(12,14,18,.94);color:#e6edf3;padding:12px 14px;
 border-bottom-left-radius:10px;box-shadow:0 6px 28px rgba(0,0,0,.45);
 pointer-events:none;backdrop-filter:blur(6px)}
#lz-hud .lz-t{font-size:14px;font-weight:700;letter-spacing:.3px;margin-bottom:8px;color:#7ee787}
#lz-hud .lz-task{color:#8b949e;font-size:11px;margin-bottom:9px;line-height:1.35}
#lz-hud .lz-r{display:flex;justify-content:space-between;gap:8px;margin:3px 0}
#lz-hud .lz-k{color:#8b949e}
#lz-hud .lz-v{color:#e6edf3;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#lz-hud .lz-act{color:#79c0ff;font-weight:700}
#lz-hud .lz-sep{height:1px;background:#30363d;margin:9px 0}
#lz-hud .lz-bar{height:5px;background:#21262d;border-radius:3px;overflow:hidden;display:flex;margin-top:3px}
#lz-hud .lz-c{background:#7ee787}
#lz-hud .lz-n{background:#484f58}
#lz-hud .lz-sig{display:flex;justify-content:space-between;font-size:11px;margin:2px 0}
.lz-cand{outline:1.5px dashed rgba(121,192,255,.40)!important;outline-offset:1px!important}
.lz-pick{outline:3px solid #7ee787!important;outline-offset:2px!important;
 box-shadow:0 0 0 6px rgba(126,231,135,.22)!important}
.lz-deny{outline:3px solid #f85149!important;outline-offset:2px!important;
 box-shadow:0 0 0 6px rgba(248,81,73,.22)!important}
#lz-tag{position:absolute;z-index:2147483646;background:#7ee787;color:#0d1117;
 font:700 11px/1 ui-monospace,Menlo,monospace;padding:4px 7px;border-radius:4px;
 pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.4)}
#lz-tag.deny{background:#f85149;color:#fff}
"""

_JS = r"""
(d) => {
  if (!document.getElementById('lz-style')) {
    const st = document.createElement('style');
    st.id = 'lz-style'; st.textContent = d.css;
    document.head.appendChild(st);
  }
  document.querySelectorAll('.lz-cand,.lz-pick,.lz-deny')
    .forEach(e => e.classList.remove('lz-cand','lz-pick','lz-deny'));
  const old = document.getElementById('lz-tag'); if (old) old.remove();

  // faint boxes on everything Jev was allowed to choose from
  (d.candidates || []).forEach(id => {
    const el = document.querySelector(`[data-lz="${id}"]`);
    if (el) el.classList.add('lz-cand');
  });

  // the pick
  if (d.pick !== null && d.pick !== undefined) {
    const el = document.querySelector(`[data-lz="${d.pick}"]`);
    if (el) {
      el.classList.add(d.denied ? 'lz-deny' : 'lz-pick');
      const r = el.getBoundingClientRect();
      const tag = document.createElement('div');
      tag.id = 'lz-tag';
      if (d.denied) tag.className = 'deny';
      tag.textContent = d.denied
        ? `REFUSED  ${d.action}`
        : `${d.action}  ${(d.conf*100).toFixed(0)}%`;
      tag.style.left = (r.left + scrollX) + 'px';
      tag.style.top  = Math.max(r.top + scrollY - 24, 2) + 'px';
      document.body.appendChild(tag);
    }
  }

  let hud = document.getElementById('lz-hud');
  if (!hud) { hud = document.createElement('div'); hud.id = 'lz-hud'; document.body.appendChild(hud); }

  const pct = d.total ? Math.max(4, Math.round(100 * d.compute / d.total)) : 0;
  const row = (k, v, cls) => `<div class="lz-r"><span class="lz-k">${k}</span><span class="lz-v ${cls||''}">${v}</span></div>`;
  const sig = (k, v) => {
    const g = Math.round(255 - v*130), b = Math.round(120 + v*60);
    return `<div class="lz-sig"><span class="lz-k">${k}</span>`
         + `<span style="color:rgb(${g},${Math.round(150+v*80)},${b})">${v.toFixed(2)}</span></div>`;
  };

  hud.innerHTML =
    `<div class="lz-t">🦎 lizard · no LLM</div>`
  + `<div class="lz-task">${d.task}</div>`
  + row('step', d.step)
  + row('action', d.action, 'lz-act')
  + row('confidence', (d.conf*100).toFixed(0) + '%')
  + row('choices', d.shown + ' of ' + d.found)
  + `<div class="lz-sep"></div>`
  + row('compute', d.compute.toFixed(0) + ' ms')
  + row('network', d.network.toFixed(0) + ' ms')
  + `<div class="lz-bar"><div class="lz-c" style="width:${pct}%"></div>`
  + `<div class="lz-n" style="width:${100-pct}%"></div></div>`
  + `<div class="lz-sep"></div>`
  + Object.entries(d.signals || {}).map(([k, v]) => sig(k, v)).join('')
  + `<div class="lz-sep"></div>`
  + row('cost so far', '$' + d.cost.toFixed(6));
}
"""


async def draw(page, *, task: str, step: int, action: str, conf: float,
               pick: int | None, candidates: list[int], shown: int, found: int,
               compute: float, network: float, signals: dict, cost: float,
               denied: bool = False) -> None:
    """Paint the current decision onto the page. Never raises - a failed
    overlay must not take the run down with it."""
    try:
        await page.evaluate(_JS, {
            "css": _CSS, "task": task, "step": step, "action": action,
            "conf": conf, "pick": pick, "candidates": candidates[:120],
            "shown": shown, "found": found, "compute": compute,
            "network": network, "total": compute + network,
            "signals": signals, "cost": cost, "denied": denied,
        })
    except Exception:
        pass
