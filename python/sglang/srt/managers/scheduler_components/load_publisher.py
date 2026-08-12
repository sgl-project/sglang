from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.load_inquirer import (
        SchedulerLoadInquirer,
    )

logger = logging.getLogger(__name__)

# The router-facing socket publishes no faster than this. Sits between the two
# rates that matter: subscribers treat entries older than a few seconds as
# stale, and the busy path produces snapshots on the order of once per second,
# so 10 Hz starves neither.
ROUTER_MIN_PUBLISH_PERIOD_S = 0.1

# After the first report, re-warn about a failing sink at most this often.
FAIL_WARN_PERIOD_S = 60.0


class SchedulerLoadPublisher:
    """Ships one scheduler's load snapshot to its sinks, on each sink's own schedule.

    Two sinks, two cadences, and the difference is the reason this owns them
    rather than iterating a uniform list:

    * the **internal** writer (SHM, or ZMQ PUSH across nodes) feeds ``/v1/loads``
      and DP dispatch. Its cadence is iteration-based -- every Nth call, and
      immediately when forced -- which is what its consumers want: they are
      reading local state and care about seeing a transition promptly.
    * the **router-facing** writer broadcasts to out-of-process subscribers. An
      iteration count stops being a rate limit exactly when iterations stop being
      work: an idle scheduler spins its loop freely and forces a publish on every
      pass, which would put an unchanged gauge on the wire at loop rate. The cost
      of that is not local -- every subscribed router decodes and stores each
      copy -- so this sink is bounded in seconds instead.

    What both share is the snapshot itself, which is the one thing worth
    coupling: `SchedulerLoadInquirer.get_loads` walks the running batch, the
    waiting queue and four disaggregation queues, so it runs at most once per
    call however many sinks end up being due.
    """

    def __init__(
        self,
        inquirer: SchedulerLoadInquirer,
        internal_writer=None,
        router_writer=None,
        internal_interval: int = 1,
        router_min_period_s: float = ROUTER_MIN_PUBLISH_PERIOD_S,
    ):
        self.inquirer = inquirer
        self.internal_writer = internal_writer
        self.router_writer = router_writer
        self.internal_interval = max(1, internal_interval)
        self.router_min_period_s = router_min_period_s
        self._internal_counter = 0
        self._last_router_send = float("-inf")
        # Per failing sink: consecutive count, and when it was last reported.
        self._failures: dict = {}

    @property
    def enabled(self) -> bool:
        return self.internal_writer is not None or self.router_writer is not None

    def publish(self, force: bool = False) -> None:
        """Publish to whichever sinks are due, collecting the snapshot at most once.

        Best-effort throughout: the router-facing socket failing must not stop
        the internal writer that ``/v1/loads`` and DP dispatch depend on, and
        neither may take down the scheduler loop.
        """
        if not self.enabled:
            return

        due = []
        if self.internal_writer is not None:
            self._internal_counter += 1
            if force or self._internal_counter >= self.internal_interval:
                self._internal_counter = 0
                due.append(self.internal_writer)

        now = time.monotonic()
        router_due = (
            self.router_writer is not None
            and now - self._last_router_send >= self.router_min_period_s
        )
        if router_due:
            due.append(self.router_writer)

        if not due:
            return

        try:
            snapshot = self.inquirer.get_loads()
        except Exception as e:
            # Nothing reached a sink, so no cadence is spent: a failing
            # collector must not also halve the router's effective rate.
            self._warn_throttled(None, e)
            return
        self._failures.pop(None, None)
        if router_due:
            self._last_router_send = now

        for writer in due:
            try:
                writer.write(snapshot)
            except Exception as e:
                self._warn_throttled(writer, e)
            else:
                self._failures.pop(writer, None)

    def _warn_throttled(self, writer, exc: Exception) -> None:
        """Report a failing sink (``writer=None`` for collection) on the first
        occurrence, then at most once per warn period.

        The bound is wall-clock rather than a count because a permanently broken
        sink fails once per publish, and publishes are driven by the scheduler
        loop -- so re-warning every Nth failure would still emit at a rate
        proportional to that loop, which is the flood it is meant to stop.
        The count is still worth reporting; it just cannot be what gates.
        """
        count, last_warned = self._failures.get(writer, (0, float("-inf")))
        count += 1
        now = time.monotonic()
        should_warn = count == 1 or now - last_warned >= FAIL_WARN_PERIOD_S
        self._failures[writer] = (count, now if should_warn else last_warned)
        if should_warn:
            logger.warning(
                "load snapshot %s failed (%s consecutive): %s",
                "collection" if writer is None else type(writer).__name__,
                count,
                exc,
            )
