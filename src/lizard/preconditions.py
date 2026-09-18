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

from urllib.parse import urlparse


async def _amazon_set_pincode(page, pin: str) -> bool:
    """Open the location popover, apply a PIN, dismiss the confirmation."""
    try:
        await page.click("#nav-global-location-popover-link", timeout=8000)
        await page.wait_for_timeout(1500)
        await page.fill("#GLUXZipUpdateInput", pin, timeout=8000)
        await page.wait_for_timeout(400)
        await page.click("#GLUXZipUpdate input, #GLUXZipUpdate-announce",
                         timeout=8000)
        await page.wait_for_timeout(2500)
    except Exception:
        return False

    # The confirmation sheet has carried several different markups; try the
    # known ones and treat its absence as success.
    for sel in ("button[name=glowDoneButton]",
                ".a-popover-footer input[type=submit]",
                "#GLUXConfirmClose"):
        try:
            await page.click(sel, timeout=2000)
            break
        except Exception:
            continue

    await page.wait_for_timeout(2000)
    try:
        return pin in await page.evaluate("document.body.innerText")
    except Exception:
        return False


async def apply(page, constraints) -> list[str]:
    """Run whatever setup this site and task need. Returns what was done."""
    done: list[str] = []
    host = (urlparse(page.url).hostname or "").lower()

    if constraints.pincode and "amazon." in host:
        if await _amazon_set_pincode(page, constraints.pincode):
            done.append(f"set delivery PIN to {constraints.pincode}")
        else:
            done.append(f"could not set delivery PIN {constraints.pincode}")

    return done
