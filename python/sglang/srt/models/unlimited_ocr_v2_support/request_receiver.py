"""bounded host preparation for Unlimited-OCR v2 requests"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.mm_utils import has_shm_features
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.models.unlimited_ocr_v2_support.async_shm import AsyncShmMaterializer


@dataclass(kw_only=True, slots=True, frozen=True)
class UocrRequestReceiver(SchedulerRequestReceiver):
    async_shm_materializer: Optional[AsyncShmMaterializer]
    submit_crop_prefetch: Callable[[list], None]

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        materializer = self.async_shm_materializer
        if materializer is None:
            return super(UocrRequestReceiver, self).recv_limit_reached(num_recv_reqs)
        return num_recv_reqs >= materializer.max_recv_per_poll

    def _recv_requests(self, local_reqs=None):
        if self.async_shm_materializer is not None:
            return self._recv_requests_async(local_reqs)
        values = super(UocrRequestReceiver, self)._recv_requests(local_reqs)
        self._submit_crop_prefetch(values)
        return values

    def has_pending(self) -> bool:
        return (
            self.async_shm_materializer is not None
            and self.async_shm_materializer.has_pending()
        )

    def wait_for_progress(self) -> None:
        if self.async_shm_materializer is not None:
            self.async_shm_materializer.wait_for_progress()

    def close(self) -> None:
        if self.async_shm_materializer is not None:
            self.async_shm_materializer.shutdown()

    def _recv_requests_async(self, local_reqs: Optional[List[AbortReq]]) -> List:
        """Return materialized values in arrival order; new ones materialize off-thread."""
        materializer = self.async_shm_materializer
        ready = materializer.pop_ready()

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if (
            self.recv_skipper is not None
            and not self.recv_skipper.handle(self.get_last_batch())
        ) or not materializer.reserve_receive(extra_items=len(local_reqs or ())):
            self._submit_crop_prefetch(ready)
            return ready

        try:
            recv_reqs = self._recv_and_broadcast(local_reqs)
            materializer.submit(
                recv_reqs,
                finalize=self._finalize_shm_features,
                has_shm=has_shm_features,
            )
            ready = ready + materializer.pop_ready()
        except BaseException:
            materializer.cancel_receive()
            raise

        self._submit_crop_prefetch(ready)
        return ready

    def _submit_crop_prefetch(self, values: List) -> None:
        if self.submit_crop_prefetch is not None and values:
            self.submit_crop_prefetch(values)
