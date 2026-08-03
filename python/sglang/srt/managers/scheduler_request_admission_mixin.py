from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class QueueAdmissionResult(Enum):
    ADMIT = auto()
    REJECT = auto()


class RejectedRequestCleanup(Enum):
    NOT_HANDLED = auto()
    RESOURCES_RELEASED = auto()


class SchedulerRequestAdmissionMixin:
    """Extension points for model-specific request admission."""

    def _admit_request_to_queue(
        self, req: Req, is_retracted: bool = False
    ) -> QueueAdmissionResult:
        return QueueAdmissionResult.ADMIT

    def _begin_request_prefill_admission(self, req: Req) -> Any:
        return None

    def _finish_request_prefill_admission(
        self, req: Req, context: Any, *, admitted: bool
    ) -> None:
        return None

    def _cleanup_rejected_queued_request(self, req: Req) -> RejectedRequestCleanup:
        return RejectedRequestCleanup.NOT_HANDLED

    def build_custom_decode_admission(
        self, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        return None

    def _merge_custom_decode_admission(
        self, running_batch: ScheduleBatch
    ) -> tuple[ScheduleBatch, bool]:
        custom_decode_batch = self.build_custom_decode_admission(running_batch)
        if custom_decode_batch is not None:
            if running_batch.is_empty():
                running_batch = custom_decode_batch
            else:
                if running_batch.multimodal_inputs is None:
                    running_batch.multimodal_inputs = [None] * len(running_batch.reqs)
                running_batch.merge_batch(custom_decode_batch)

        defer_prefill = any(
            req.custom_decode_needs_prefill_completion for req in running_batch.reqs
        )
        return running_batch, defer_prefill
