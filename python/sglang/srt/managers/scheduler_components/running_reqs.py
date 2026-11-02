from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from sglang.srt.managers.scheduler_components.request_list import RequestList

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RunningReqs(RequestList):
    """The running requests, in join order (the same on every rank).

    A request runs from the merge of its launched prefill into a running batch
    until it finishes; the steps still computing for it are tracked apart. It
    leaves when it finishes, is retracted or preempted, or a pause retracts it.
    """

    def check(self, held: Iterable[Req], chunked_req: Optional[Req]) -> None:
        """Raise unless the running requests are exactly the unfinished held ones,
        apart from the chunked request."""
        expected = {
            req for req in held if not req.finished() and req is not chunked_req
        }
        running = self._reqs.keys()
        if expected == running:
            return
        not_running = sorted(req.rid for req in expected - running)
        not_held = sorted(req.rid for req in running - expected)
        raise RuntimeError(
            "Running requests disagree with the batches: "
            f"held by a batch but not running: {not_running}; "
            f"running but held by no batch: {not_held}"
        )
