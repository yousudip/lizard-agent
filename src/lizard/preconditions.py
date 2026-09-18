"""Site-specific setup steps, run before the agent starts.

This file is the exception to the rest of the project, and it is deliberately
quarantined here.

Setting a known delivery PIN is not a judgement. The value came verbatim from
the task, the sequence never varies, and there is nothing for a model to
decide. Asking the agent to discover a multi-step modal - open the location
popover, type, apply, dismiss the confirmation - spends steps and fails often,
because a confirmation sheet appearing mid-flow is exactly the deep,
order-dependent interaction a System One model is weakest at.

So: mechanical and known in advance goes here, in plain Playwright. Judgement
stays with Jev.

The cost is honesty about generality. These selectors work on one site and
will break when it changes. Nothing else in the project knows what site it is
looking at, and it should stay that way - if you find yourself adding a third
site here, the feature probably belongs in the agent loop instead.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse


async def _settle(page, cap_ms: int = 1200) -> None:
    """Wait for the modal to stop changing rather than for a fixed time."""
    try:
        await page.wait_for_function(
            """(cap) => new Promise(resolve => {
                let t;
                const done = () => { obs.disconnect(); resolve(true); };
                const obs = new MutationObserver(() => {
                    clearTimeout(t); t = setTimeout(done, 250);
                });
                obs.observe(document.body, {childList: true, subtree: true});
                t = setTimeout(done, 250);
                setTimeout(done, cap);
            })""",
            arg=cap_ms, timeout=cap_ms + 600,
        )
    except Exception:
        pass


async def _amazon_set_pincode(page, pin: str) -> bool:
    """Open the location popover, apply a PIN, dismiss the confirmation."""
    try:
        await page.click("#nav-global-location-popover-link", timeout=8000)
        await _settle(page)
        await page.fill("#GLUXZipUpdateInput", pin, timeout=8000)
        await page.click("#GLUXZipUpdate input, #GLUXZipUpdate-announce",
                         timeout=8000)
        await _settle(page, 2000)
    except Exception:
        return False

    # The confirmation sheet has carried several different markups, and
    # often does not appear at all. Clicking each candidate with a timeout
    # meant paying that timeout three times over for a sheet that was not
    # there - six seconds of waiting, and the single largest delay before
    # the agent took its first step. Ask the DOM instead; a miss costs
    # nothing.
    # One wait for any of them, not three waits in series. Clicking each
    # candidate with its own timeout cost six seconds whenever the sheet
    # was absent; querying instantly was fast but missed a sheet that
    # arrived a moment later. A single combined wait handles both.
    CONFIRM = ("button[name=glowDoneButton], "
               ".a-popover-footer input[type=submit], "
               "#GLUXConfirmClose")
    try:
        el = await page.wait_for_selector(CONFIRM, timeout=1800, state="visible")
        if el:
            await el.click(timeout=1500)
    except Exception:
        pass                          # no sheet appeared; that is normal

    # Amazon reloads after applying a location, so the PIN is not on the
    # page the instant the click returns. Poll for it rather than sleeping
    # a fixed amount: this finishes as soon as the reload lands, and only
    # spends the full budget when something has genuinely gone wrong.
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        try:
            if pin in await page.evaluate("document.body.innerText"):
                return True
        except Exception:
            pass                      # mid-navigation; try again shortly
        await page.wait_for_timeout(250)
    return False


async def apply(page, constraints, task: str = "",
                show: bool = False) -> list[str]:
    """Run whatever setup this site and task need. Returns what was done."""
    done: list[str] = []
    host = (urlparse(page.url).hostname or "").lower()

    if constraints.pincode and "amazon." in host:
        if show:
            from .overlay import notice
            await notice(page, task,
                         f"setting delivery location to {constraints.pincode}")
        if await _amazon_set_pincode(page, constraints.pincode):
            done.append(f"set delivery PIN to {constraints.pincode}")
        else:
            done.append(f"could not set delivery PIN {constraints.pincode}")

    return done
