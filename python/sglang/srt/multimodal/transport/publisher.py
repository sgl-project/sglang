"""Async publication shared by guarded CUDA IPC and VMM transports."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import torch

from sglang.srt.multimodal.transport.admission import FeaturePublicationAdmission

logger = logging.getLogger(__name__)


class BoundedFeaturePublisher:
    """Reserve a whole request before submitting any GPU copy.

    The pool supplies atomic reservation, publication and cancellation methods.
    Requests waiting for capacity own no slices and occupy no executor thread.
    """

    def __init__(self, pool, timeout_ms: int, *, cpu_tensors: bool):
        self.pool = pool
        self.cpu_tensors = cpu_tensors
        self.admission = FeaturePublicationAdmission(
            pool.publication_capacity, timeout_ms
        )
        pool.set_capacity_callback(self.admission.notify)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mm-feature-publisher"
        )

    async def prepare(self, mm_inputs_batch):
        fields = []
        for mm_inputs in mm_inputs_batch:
            if mm_inputs is None:
                continue
            for item in mm_inputs.mm_items:
                for name, tensor in (
                    ("feature", item.feature),
                    ("precomputed_embeddings", item.precomputed_embeddings),
                ):
                    if isinstance(tensor, (tuple, list)):
                        raise TypeError(
                            "Guarded GPU transport requires a tensor per feature field"
                        )
                    if (
                        isinstance(tensor, torch.Tensor)
                        and tensor.numel() > 0
                        and (self.cpu_tensors or tensor.is_cuda)
                    ):
                        fields.append((item, name, tensor))
        if not fields:
            return []

        tensors = [tensor for _, _, tensor in fields]
        sizes = [tensor.numel() * tensor.element_size() for tensor in tensors]
        required = sum(self.pool.reservation_size(size) for size in sizes)
        reservations, ticket = await self.admission.acquire(
            required, lambda: self.pool.try_reserve_batch(sizes)
        )
        loop = asyncio.get_running_loop()
        try:
            # source streams are thread-local; capture readiness on the caller
            events = []
            for device in {tensor.device for tensor in tensors if tensor.is_cuda}:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                events.append((device, event))
            future = self.executor.submit(self._publish, tensors, reservations, events)
        except BaseException:
            self.pool.release_reserved_batch(reservations)
            self.admission.finish(ticket)
            raise

        # the native task owns input tensors and accounting after submission
        try:
            proxies = await asyncio.shield(asyncio.wrap_future(future))
        except asyncio.CancelledError:

            def cancel_completed(done):
                try:
                    self._cancel_completed(done)
                finally:
                    loop.call_soon_threadsafe(self.admission.finish, ticket)

            future.add_done_callback(cancel_completed)
            raise
        except BaseException:
            self.admission.finish(ticket)
            raise
        for (item, name, _), proxy in zip(fields, proxies, strict=True):
            setattr(item, name, proxy)
        self.admission.finish(ticket)
        return list({id(item): item for item, _, _ in fields}.values())

    def _publish(self, tensors, reservations, events):
        try:
            for device, event in events:
                torch.cuda.current_stream(device).wait_event(event)
        except BaseException:
            self.pool.release_reserved_batch(reservations)
            raise
        return self.pool.publish_reserved_batch(tensors, reservations)

    def _cancel_completed(self, future):
        if future.exception() is None:
            errors = []
            for proxy in future.result():
                try:
                    self.pool.cancel_proxy(proxy)
                except Exception as error:
                    errors.append(error)
            if errors:
                logger.error("Failed to release abandoned GPU publication: %s", errors)

    def shutdown(self):
        self.admission.close()
        # queued native tasks already own reservations and must run their cleanup
        self.executor.shutdown(wait=True)
