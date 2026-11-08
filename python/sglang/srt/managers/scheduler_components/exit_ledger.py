"""CI-only check that a request exits and sends its final output at most once,
and that a request the scheduler took sends it at least once.

An exit gives the request's resources back (`RequestRetirer.release`,
`abort_retracted`); a send is its final output (the streams' finished branches,
`Scheduler.send_abort_output`). Retract, chunk steps and beam member rows are
not exits. Gated on `SGLANG_ENABLE_RUNNING_REQS_CHECK`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class UnsentOutputs:
    """The requests a scheduler took that have not sent their final output.
    `record_send` drops one; the invariant checker reports, at idle, any left
    that no request list holds."""

    def __init__(self) -> None:
        # Used as an insertion-ordered set.
        self.reqs: Dict[Req, None] = {}

    def record_take(self, req: Req) -> None:
        """Record that the scheduler took `req`, which now owes a final output."""
        if not envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.get():
            return
        self.reqs[req] = None
        req.taken_by = self


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
    """Record that `req`'s final output was sent through `via`."""
    if not envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.get():
        return
    if req.send_via is not None:
        raise RuntimeError(
            f"Request {req.rid} sent its final output twice: through "
            f"{req.send_via}, then {via}"
        )
    req.send_via = via
    if req.taken_by is not None:
        del req.taken_by.reqs[req]
