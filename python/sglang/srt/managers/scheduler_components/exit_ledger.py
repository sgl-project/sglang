"""Retire-once ledger, a CI-only check that every request exits exactly once.

A request exits when its resources go back (`RequestRetirer.release` and
`abort_retracted`, the decode-offload release), and its final output is sent
once (the output stream's finished branch, `Scheduler.send_abort_output`).
Retract, chunk steps and beam member rows are not exits: only a request's
own terminal events count, and a beam leader's exit is its group's.

The state is on the `Req`, so it dies with it. A second exit or send raises at
the second event. A request that left every owner list with no exit is
reported by `LeakDetector.check`, which the invariant checker runs at BUSY and
IDLE. Both are gated on `SGLANG_ENABLE_RUNNING_REQS_CHECK` (on in CI), the flag
of the request-list checks this one completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Set

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def record_exit(req: Req, via: str) -> None:
    """Record that `req`'s resources went back through `via`."""
    if not envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.get():
        return
    if req.exit_via is not None:
        raise RuntimeError(
            f"Request {req.rid} exited twice: through {req.exit_via}, then {via}"
        )
    req.exit_via = via


def record_send(req: Req, via: str) -> None:
    """Record that `req`'s final output was sent through `via`. The stream
    asserts `finished_output` only against itself; this also catches an abort
    output followed by a stream of the same finished request."""
    if not envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.get():
        return
    if req.send_via is not None:
        raise RuntimeError(
            f"Request {req.rid} sent its final output twice: through "
            f"{req.send_via}, then {via}"
        )
    req.send_via = via


class LeakDetector:
    """The requests seen in an owner list that have not exited. Bounded by the
    live requests: a request leaves at the first check after its exit, or at the
    report that it was lost."""

    def __init__(self) -> None:
        self.unexited: Set[Req] = set()

    def check(
        self,
        lists: Iterable[Iterable[Req]],
        is_offloading: Callable[[Req], bool],
    ) -> None:
        """Raise if a request seen in a list is in none now, has not exited, has
        no step in flight, and is not held by the decode offload."""
        listed: Set[Req] = set()
        for reqs in lists:
            listed.update(reqs)
        self.unexited |= listed
        self.unexited = {req for req in self.unexited if req.exit_via is None}
        lost = {
            req
            for req in self.unexited - listed
            if req.inflight_steps <= 0 and not is_offloading(req)
        }
        self.unexited -= lost
        if lost:
            raise RuntimeError(
                "Requests left every list without exiting: "
                f"{sorted(req.rid for req in lost)}"
            )
