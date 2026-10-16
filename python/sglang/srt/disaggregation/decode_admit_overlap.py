"""Where PD decode does its admission host work relative to the forward launch.

In the overlap decode loop the host blocks once per iteration until the in-flight
forward reaches its WAR fence (the MLP sync read-back). Host work that sits behind an
earlier blocking copy runs after that release and delays the next forward launch.
These switches move admission work ahead of the block; every switch defaults off, and
off keeps the stock path:

- ``defer_prealloc``: preallocate new requests right after the forward launch. Their
  page-index read-back then no longer waits for the fence; the transfer metadata
  goes out one host iteration later.
- ``nonblocking_admit``: seed admitted (prebuilt) requests on the forward stream from
  pinned, non-blocking copies, instead of making the schedule stream wait for the whole
  in-flight forward and copying with blocking H2Ds.

A control file can switch them between iterations of one server.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

DEFER_PREALLOC = "defer_prealloc"
NONBLOCKING_ADMIT = "nonblocking_admit"
_KNOWN = frozenset({DEFER_PREALLOC, NONBLOCKING_ADMIT})

# Seconds between reads of the A/B control file. A daemon thread does the reads so a
# slow (e.g. network filesystem) read never stalls the scheduler loop.
_CTL_REFRESH_S = 0.2


class AdmitOverlap:
    """Per-scheduler switch state; exists only when a switch or the control file is set."""

    def __init__(
        self,
        *,
        static_flags: frozenset[str],
        ctl_path: Optional[str],
        start_poller: bool = True,
    ):
        self.ctl_path = ctl_path
        self.flags = frozenset() if ctl_path else static_flags
        self._ctl_flags: frozenset[str] = frozenset()
        self._warned: set[str] = set()
        self.round_ct = {name: 0 for name in sorted(_KNOWN)}
        if ctl_path and start_poller:
            threading.Thread(
                target=self._poll_ctl_forever,
                name="pd-admit-overlap-ctl",
                daemon=True,
            ).start()
        logger.info(
            "PD admission overlap: static=%s ab_ctl=%s",
            sorted(static_flags),
            ctl_path,
        )

    @classmethod
    def from_env(cls, *, prealloc_queue) -> Optional[AdmitOverlap]:
        static_flags = frozenset(
            name
            for name, on in (
                (DEFER_PREALLOC, envs.SGLANG_Q35MF_PDDEC_DEFER_PREALLOC.get()),
                (NONBLOCKING_ADMIT, envs.SGLANG_Q35MF_PDDEC_NONBLOCKING_ADMIT.get()),
            )
            if on
        )
        ctl_path = envs.SGLANG_Q35MF_PDDEC_ADMIT_AB_CTL.get() or None
        # Ranks read the control file independently, so one may move its prealloc poll
        # collective across the forward launch an iteration before another: safe only
        # when that collective's group has a single rank.
        if ctl_path and prealloc_queue.gloo_group.size() != 1:
            logger.warning(
                "PD admission overlap: ignoring the A/B control file; per-iteration "
                "switching needs a single-rank prealloc poll group"
            )
            ctl_path = None
        if not static_flags and not ctl_path:
            return None
        return cls(static_flags=static_flags, ctl_path=ctl_path)

    def _read_ctl(self) -> frozenset[str]:
        try:
            with open(self.ctl_path) as f:
                words = frozenset(f.read().split())
        except OSError:
            return frozenset()
        for word in words - _KNOWN:
            if word not in self._warned:
                self._warned.add(word)
                logger.warning("PD admission overlap: unknown switch %r", word)
        return words & _KNOWN

    def _poll_ctl_once(self) -> None:
        self._ctl_flags = self._read_ctl()

    def _poll_ctl_forever(self) -> None:
        while True:
            self._poll_ctl_once()
            time.sleep(_CTL_REFRESH_S)

    def begin_iteration(self) -> frozenset[str]:
        """Switches for this loop iteration; they stay fixed until the next call."""
        if self.ctl_path:
            self.flags = self._ctl_flags
        for name in self.flags:
            self.round_ct[name] += 1
        return self.flags
