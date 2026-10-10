"""Post-run hard gate: verify HBM is actually freed before the next cell.

Inference engines routinely leave orphaned workers holding tens of GB of
HBM after their parent is killed; without this gate the next cell fails
with a phantom OOM and pollutes the whole sweep.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any, Callable, Iterable

# probe() -> {device: used_mb} | None
Probe = Callable[[], dict[int, int] | None]
Sleep = Callable[[float], None]

DEFAULT_BUDGET_MB = 500
DEFAULT_TIMEOUT_S = 240
DEFAULT_POLL_S = 5.0


def wait_hbm_freed(
    probe: Probe | None,
    baseline: dict[int, int] | None,
    *,
    budget_mb: int = DEFAULT_BUDGET_MB,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    extra_kill: Callable[[], None] | None = None,
    log: Callable[[str], None] = lambda _msg: None,
    sleep: Sleep = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Wait until per-device HBM usage is back near the pre-run baseline.

    Returns True when freed (or when no probe exists — degraded mode waits
    a short grace period and trusts the process-group kill).  On timeout,
    ``extra_kill`` is invoked once (straggler sweep by PID) and polling
    continues until the deadline; returning False means the node is still
    dirty and the caller must flag the cell.
    """
    if probe is None:
        log("hbm probe unavailable; degraded mode — fixed grace wait")
        sleep(min(10.0, timeout_s))
        return True

    baseline = baseline or {}
    deadline = monotonic() + timeout_s
    killed_stragglers = False
    while monotonic() < deadline:
        usage = probe()
        if usage is not None:
            dirty = {
                dev: used
                for dev, used in usage.items()
                if used > baseline.get(dev, 0) + budget_mb
            }
            if not dirty:
                return True
            log(f"hbm still above baseline: {dirty}")
        if not killed_stragglers and extra_kill is not None:
            log("invoking straggler kill and re-probing")
            extra_kill()
            killed_stragglers = True
        sleep(poll_s)
    return False


STRAGGLER_PATTERNS = (
    "sglang::",  # engine workers: sglang::scheduler / detokenizer / ...
    # server main; the char class keeps this pattern from matching any
    # plain-text occurrence (the pkill self-match trap)
    "sglang[.]launch_server",
)


def kill_stragglers(
    patterns: Iterable[str] = STRAGGLER_PATTERNS,
    *,
    run: Callable[..., Any] = subprocess.run,
    log: Callable[[str], None] = lambda _msg: None,
) -> None:
    """Best-effort SIGKILL sweep for engine processes that survived the
    process-group kill (orphaned workers keep holding HBM).  Silently
    no-ops where pkill is unavailable (e.g. Windows dev hosts).
    """
    for pattern in patterns:
        try:
            proc = run(
                ["pkill", "-9", "-f", pattern],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            log(f"pkill -9 -f {pattern!r} killed straggler(s)")
