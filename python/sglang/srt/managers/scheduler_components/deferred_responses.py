from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    ReqList,
)

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
        self.disaggregation_mode = disaggregation_mode
        self.output_streamer = output_streamer
        self.send_abort = send_abort
        # (req, completion) per held response.
        self.held = IndexedReqList(
            ListKind.DEFERRED, abort=self._abort, req_of=lambda x: x[0]
        )

    def request_lists(self) -> Dict[ListKind, ReqList]:
        return {ListKind.DEFERRED: self.held}

    def defer(self, req: Req, completion: ResponseCompletion) -> None:
        if self.disaggregation_mode != DisaggregationMode.NULL:
            raise ValueError("deferred responses do not support PD disaggregation")
        if not req.finished() or req.finished_output:
            raise ValueError("only finished, unstreamed responses can be deferred")
        if any(held.rid == req.rid for held in self.held):
            raise ValueError(f"response {req.rid!r} is already deferred")
        self.held.join([(req, completion)])
        req.defer_output = True
        self.output_streamer.defer_outputs = True

    def stream_released(self) -> None:
        if not self.held:
            return
        ready = []
        for req, completion in list(self.held.items()):
            if not completion.done():
                continue
            self.held.leave([req])
            req.defer_output = False
            try:
                completion.result()
            except Exception as exc:
                logger.exception("Deferred response failed for %s", req.rid)
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

    def _abort(self, recv_req: AbortReq) -> None:
        for req, completion in list(self.held.items()):
            if recv_req.matches(req.rid):
                self.held.leave([req])
                try:
                    completion.cancel()
                except Exception:
                    logger.exception("Failed to cancel deferred work for %s", req.rid)
                self._abort_response(req, recv_req.finished_reason)
