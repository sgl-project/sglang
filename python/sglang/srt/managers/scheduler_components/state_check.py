"""The vocabulary of a scheduler state check.

A leaf module: it imports nothing heavy, so a component can import it to
register its own checks without a cycle through the scheduler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# A leak warning is logged on its first occurrence and then once per this many.
_WARN_LOG_INTERVAL = 1000


class CheckPhase(Enum):
    """Where in the scheduler loop a state check runs."""

    # A step is launched.
    LAUNCH = "launch"
    # A step's result is processed.
    RESULT = "result"
    # The end of an event-loop iteration that ran a batch.
    BUSY = "busy"
    # A fully idle iteration.
    IDLE = "idle"
    # After the cache is flushed.
    FLUSH = "flush"


@dataclass(frozen=True, slots=True)
class StateCheck:
    phase: CheckPhase
    # Evaluated on every run.
    when: Callable[[], bool]
    # A LAUNCH or RESULT check takes the step's batch: `fn(batch)`, else `fn()`.
    fn: Callable[..., None]


class StateCheckError(RuntimeError):
    """A state check found the scheduler's state corrupt, or leaking in strict mode."""


class Severity(Enum):
    # Resources are lost but the state stays consistent: raises or only warns,
    # as SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE says.
    LEAK = "leak"
    # The state is inconsistent: always raises.
    CORRUPTION = "corruption"


class FailureReporter:
    """Reports failed state checks; the single exit for every check.

    Each owner (the scheduler's checker, a component's checks) holds its own
    reporter, so the warning throttle is per owner, not process-global.
    """

    def __init__(self) -> None:
        self._warning_count = 0

    def report(self, severity: Severity, message: str) -> None:
        """A leak found while SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE is off
        is only counted and logged at a low rate."""
        if (
            severity is Severity.CORRUPTION
            or envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get()
        ):
            raise StateCheckError(message)
        if self._warning_count % _WARN_LOG_INTERVAL == 0:
            logger.warning(message)
        self._warning_count += 1
