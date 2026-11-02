from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT

if TYPE_CHECKING:
    from sglang.srt.managers.deferred_output import ResponseCompletion
    from sglang.srt.managers.io_struct import AbortReq, FinishReasonDict
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )

logger = logging.getLogger(__name__)


class DeferredResponses:
    """The finished responses held until their completion is done; see
    deferred_output. Only the output rank holds them."""

    def __init__(
        self,
        *,
        disaggregation_mode: DisaggregationMode,
        output_streamer: SchedulerOutputStreamer,
        send_abort: Callable[[Req, Optional[FinishReasonDict]], None],
    ):
        # Fixed for the process: a PD role switch never leaves or enters PD.
        self.disaggregation_mode = disaggregation_mode
        self.output_streamer = output_streamer
        self.send_abort = send_abort
        # The held responses by rid, each with its completion.
        self.held: Dict[str, Tuple[Req, ResponseCompletion]] = {}

    def held_reqs(self) -> List[Req]:
        return [req for req, _ in self.held.values()]

    def defer(self, req: Req, completion: ResponseCompletion) -> None:
        """Hold a finished response until completion; see deferred_output."""
        if self.disaggregation_mode != DisaggregationMode.NULL:
            raise ValueError("deferred responses do not support PD disaggregation")
        if not req.finished() or req.finished_output:
            raise ValueError("only finished, unstreamed responses can be deferred")
        if req.rid in self.held:
            raise ValueError(f"response {req.rid!r} is already deferred")
        self.held[req.rid] = (req, completion)
        req.defer_output = True
        self.output_streamer.defer_outputs = True

    def stream_released(self) -> None:
        if not self.held:
            return
        ready = []
        for rid, (req, completion) in list(self.held.items()):
            if not completion.done():
                continue
            del self.held[rid]
            req.defer_output = False
            try:
                completion.result()
            except Exception as exc:
                logger.exception("Deferred response failed for %s", rid)
                self._abort_response(
                    req, FINISH_ABORT(str(exc), status_code=500).to_json()
                )
            else:
                ready.append(req)
        if ready:
            self.output_streamer.stream_output(
                ready, any(req.return_logprob for req in ready)
            )

    def has_pending(self) -> bool:
        return bool(self.held)

    def _abort_response(
        self, req: Req, finished_reason: Optional[FinishReasonDict]
    ) -> None:
        req.defer_output = False
        req.finished_output = True
        self.send_abort(req, finished_reason)

    def abort(self, recv_req: AbortReq) -> None:
        for rid, (req, completion) in list(self.held.items()):
            if recv_req.abort_all or rid.startswith(recv_req.rid):
                del self.held[rid]
                try:
                    completion.cancel()
                except Exception:
                    logger.exception("Failed to cancel deferred work for %s", rid)
                self._abort_response(req, recv_req.finished_reason)
