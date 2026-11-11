"""CI-only check that a request exits and sends its final output at most once.
The invariant checker checks at idle that a request the scheduler kept and no
list holds did both.

An exit gives the request's resources back (`RequestRetirer.release`,
`abort_retracted`); a send is its final output (the streams' finished branches,
`Scheduler.send_abort_output`). Retract, chunk steps and beam member rows are
not exits. Gated on `SGLANG_ENABLE_RUNNING_REQS_CHECK`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    """Record that `req`'s final output was sent through `via`."""
    if not envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.get():
        return
    if req.send_via is not None:
        raise RuntimeError(
            f"Request {req.rid} sent its final output twice: through "
            f"{req.send_via}, then {via}"
        )
    req.send_via = via
