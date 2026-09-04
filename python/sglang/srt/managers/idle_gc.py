import logging

logger = logging.getLogger(__name__)

# Only freeze after the process has been continuously idle this long,
# so sub-second gaps in a live request stream never trigger it.
MIN_IDLE_DURATION_S = 5.0
# Do not freeze more often than this. gc.freeze() itself costs
# milliseconds; the interval only bounds how much recently-created
# state an ill-timed freeze can pin.
MIN_FREEZE_INTERVAL_S = 60.0


class IdleGCFreezeGate:
    """Decide when to broadcast gc.freeze() across the server processes.

    The long-lived heap is live by design (import-graph objects), so
    freezing at sustained-idle moments — not collecting — is what keeps
    full collections from re-scanning it during serving. Pure decision
    logic: the owner observes busy/idle, calls note(), and performs the
    freeze broadcast when it returns True.
    """

    def __init__(
        self,
        min_idle_duration_s: float = MIN_IDLE_DURATION_S,
        min_freeze_interval_s: float = MIN_FREEZE_INTERVAL_S,
    ):
        self.min_idle_duration_s = min_idle_duration_s
        self.min_freeze_interval_s = min_freeze_interval_s
        self.idle_since = None
        self.seen_work = False
        self.last_freeze_time = -float("inf")

    def note(self, busy: bool, now: float) -> bool:
        """Record the current busy/idle state; True means freeze now."""
        if busy:
            self.idle_since = None
            self.seen_work = True
            return False
        if self.idle_since is None:
            self.idle_since = now
        if not self.seen_work:
            return False
        if now - self.idle_since < self.min_idle_duration_s:
            return False
        if now - self.last_freeze_time < self.min_freeze_interval_s:
            return False
        return True

    def mark_frozen(self, now: float) -> None:
        self.last_freeze_time = now
        self.seen_work = False
