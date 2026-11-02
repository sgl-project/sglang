"""Async publication shared by guarded CUDA IPC and VMM transports."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch

from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.transport.admission import FeaturePublicationAdmission
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)

logger = logging.getLogger(__name__)


class BoundedFeaturePublisher:
    """Reserve a whole request before submitting any GPU copy.

    The pool supplies atomic reservation, publication and cancellation methods.
    Requests waiting for capacity own no slices and occupy no executor thread.
    """

    def __init__(
        self, pool, timeout_ms: int, *, cpu_tensors: bool, pack_features: bool = False
    ):
        self.pool = pool
        self.cpu_tensors = cpu_tensors
        self.pack_features = pack_features
        self.copy_stream = None
        self.admission = FeaturePublicationAdmission(
            pool.publication_capacity, timeout_ms
        )
        pool.set_capacity_callback(self.admission.notify)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mm-feature-publisher"
        )

    async def prepare(self, mm_inputs_batch):
        groups = []
        for mm_inputs in mm_inputs_batch:
            if mm_inputs is None:
                continue
            packed = []
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
                        entry = (item, name, tensor)
                        if (
                            self.pack_features
                            and name == "feature"
                            and item.modality in (Modality.IMAGE, Modality.VIDEO)
                            and not item.model_specific_data.get(
                                DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY, False
                            )
                        ):
                            packed.append(entry)
                        else:
                            groups.append([entry])
            # only tensors with the same request/consumer lifetime may be packed
            if packed:
                groups.append(packed)
        if not groups:
            return []

        fields = [entry for group in groups for entry in group]
        tensors = [tensor for _, _, tensor in fields]
        tensor_groups = [[tensor for _, _, tensor in group] for group in groups]
        sizes = [self.pool.publication_data_size(group) for group in tensor_groups]
        required = sum(self.pool.reservation_size(size) for size in sizes)
        reservations, ticket = await self.admission.acquire(
            required, lambda: self.pool.try_reserve_batch(sizes)
        )
        try:
            # source streams are thread-local; capture readiness on the caller
            events = []
            for device in {tensor.device for tensor in tensors if tensor.is_cuda}:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                events.append((device, event))
            future = self.executor.submit(
                self._publish, tensor_groups, reservations, events
            )
        except BaseException:
            self.pool.release_reserved_batch(reservations)
            self.admission.finish(ticket)
            raise

        # notify once, without a shield/wrap_future chain delaying dispatch;
        # cancelling this event wait cannot cancel the native task
        loop = asyncio.get_running_loop()
        completed = asyncio.Event()

        def notify_completion(done):
            try:
                loop.call_soon_threadsafe(completed.set)
            except RuntimeError:
                if not loop.is_closed():
                    raise

        future.add_done_callback(notify_completion)
        try:
            await completed.wait()
            proxies = future.result()
        except asyncio.CancelledError:

            def cancel_completed(done):
                try:
                    self._cancel_completed(done)
                finally:
                    self.admission.finish_from_thread(ticket)

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
            if events and self.copy_stream is None:
                self.copy_stream = torch.cuda.Stream(
                    device=self.pool.publication_device_id
                )
        except BaseException:
            self.pool.release_reserved_batch(reservations)
            raise
        context = (
            torch.cuda.stream(self.copy_stream)
            if self.copy_stream is not None
            else nullcontext()
        )
        with context:
            try:
                # a copy completion must not wait for unrelated preprocessing
                # subsequently enqueued on the tokenizer's default stream
                for _, event in events:
                    self.copy_stream.wait_event(event)
            except BaseException:
                self.pool.release_reserved_batch(reservations)
                raise
            return self.pool.publish_reserved_batch(tensors, reservations)

    def _cancel_completed(self, future):
        error = future.exception()
        if error is not None:
            logger.error("Abandoned GPU publication failed: %s", error)
        else:
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
