"""CUDA IPC multimodal feature transport regression tests.

This covers the production path where a tokenizer worker places a feature in
the bounded pool and the scheduler process opens the shared CUDA allocation.
CPU-only policy tests intentionally cannot exercise this cross-process handle.
"""

import gc
import multiprocessing as mp
import queue
import time
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    _pool_handle_cache_clear,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=37, stage="base-b", runner_config="1-gpu-large")


def _produce_pooled_tensor(proxy_queue, consumer_done, result_queue):
    """Create a tokenizer-worker-like CUDA IPC pool in a spawned producer."""
    pool = source = proxy = None
    try:
        torch.cuda.set_device(0)
        pool = MmItemMemoryPool(
            memory_size=1 << 20,
            recycle_interval=0.01,
            base_gpu_id=0,
            consumer_count=1,
        )
        source = torch.arange(35, dtype=torch.float32, device="cuda").reshape(5, 7)
        expected = torch.arange(35, dtype=torch.float32).reshape(5, 7).tolist()
        proxy = pool.wrap_tensor(
            source,
            use_pool_handle_cache=True,
        )
        if proxy is None:
            raise RuntimeError("test tensor did not fit in the CUDA IPC pool")
        # Intentionally do not synchronize the producer. The consumer stream
        # wait must order its copy after the producer-ready write.
        proxy_queue.put((proxy, expected))
        if not consumer_done.wait(timeout=60):
            raise TimeoutError("consumer did not release the CUDA IPC tensor")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pool.active_lease_count == 0:
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("pool did not observe the stream-ordered consumer ack")
    except Exception as exc:  # pragma: no cover - returned to the parent
        result_queue.put(("error", repr(exc)))
        return
    finally:
        del proxy, source
        if pool is not None:
            pool.shutdown()
            del pool
        gc.collect()
        torch.cuda.ipc_collect()
    result_queue.put(("ok", None))


class TestCudaIpcTransport(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")

    def test_pooled_tensor_reconstructs_in_spawned_process(self):
        """Consumer releases the pool mapping before the producer tears down."""
        ctx = mp.get_context("spawn")
        proxy_queue = ctx.Queue()
        producer_results = ctx.Queue()
        consumer_done = ctx.Event()
        producer = ctx.Process(
            target=_produce_pooled_tensor,
            args=(proxy_queue, consumer_done, producer_results),
        )
        producer.start()
        proxy = reconstructed = None
        producer_result = None
        try:
            try:
                proxy, expected = proxy_queue.get(timeout=60)
            except queue.Empty:
                producer_result = producer_results.get(timeout=5)
                _status, payload = producer_result
                self.fail(
                    f"CUDA IPC producer failed before sending its proxy: {payload}"
                )

            reconstructed = proxy.reconstruct_on_target_device(0)
            torch.cuda.synchronize()
            self.assertEqual(reconstructed.cpu().tolist(), expected)
        finally:
            # The scheduler retains this cache for its lifetime. The test's
            # consumer exits quickly, so it must close the mapping before the
            # producer destroys the shared allocation.
            del reconstructed, proxy
            _pool_handle_cache_clear()
            gc.collect()
            torch.cuda.ipc_collect()
            consumer_done.set()
            producer.join(timeout=60)
            try:
                if producer_result is None:
                    producer_result = producer_results.get(timeout=5)
                status, payload = producer_result
                self.assertEqual(status, "ok", payload)
            finally:
                if producer.is_alive():
                    producer.terminate()
                    producer.join(timeout=10)
            self.assertEqual(producer.exitcode, 0)

    def test_failed_reconstruction_releases_pooled_tensor(self):
        ctx = mp.get_context("spawn")
        proxy_queue = ctx.Queue()
        producer_results = ctx.Queue()
        consumer_done = ctx.Event()
        producer = ctx.Process(
            target=_produce_pooled_tensor,
            args=(proxy_queue, consumer_done, producer_results),
        )
        producer.start()
        proxy = None
        producer_result = None
        original_empty = torch.empty
        try:
            try:
                proxy, _expected = proxy_queue.get(timeout=60)
            except queue.Empty:
                producer_result = producer_results.get(timeout=5)
                _status, payload = producer_result
                self.fail(
                    f"CUDA IPC producer failed before sending its proxy: {payload}"
                )

            output_shape = proxy.proxy_state["ipc_extra"]["recons_shape"]

            def fail_destination_allocation(size, *args, **kwargs):
                if isinstance(size, (tuple, torch.Size)) and tuple(size) == tuple(
                    output_shape
                ):
                    raise RuntimeError("forced reconstruction failure")
                return original_empty(size, *args, **kwargs)

            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=1,
                pad_value=1,
                feature=proxy,
            )
            output = MultimodalProcessorOutput(input_ids=[1], mm_items=[item])
            with (
                patch(
                    "sglang.srt.multimodal.transport.cuda_ipc.torch.empty",
                    side_effect=fail_destination_allocation,
                ),
                self.assertRaisesRegex(RuntimeError, "forced reconstruction failure"),
            ):
                MultimodalInputs.from_processor_output(output)

            torch.cuda.synchronize()
            self.assertTrue(proxy._consumer_acknowledged)
        finally:
            del proxy
            _pool_handle_cache_clear()
            gc.collect()
            torch.cuda.ipc_collect()
            consumer_done.set()
            producer.join(timeout=60)
            try:
                if producer_result is None:
                    producer_result = producer_results.get(timeout=5)
                status, payload = producer_result
                self.assertEqual(status, "ok", payload)
            finally:
                if producer.is_alive():
                    producer.terminate()
                    producer.join(timeout=10)
            self.assertEqual(producer.exitcode, 0)

    def test_uncached_mapping_waits_before_proxy_release(self):
        proxy = object.__new__(CudaIpcTensorTransportProxy)
        proxy.proxy_state = {"ipc_extra": {"use_pool_handle_cache": False}}
        proxy._pool_storage = None
        stream = Mock()

        with patch(
            "sglang.srt.multimodal.transport.cuda_ipc.torch.cuda.current_stream",
            return_value=stream,
        ):
            proxy._retain_storage_until_stream_completes(object(), 0)

        stream.synchronize.assert_called_once_with()
        self.assertIsNone(proxy._pool_storage)

    def test_failed_item_batch_releases_undispatched_pool_slice(self):
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        pool = MmItemMemoryPool(
            memory_size=1 << 20,
            recycle_interval=0.01,
            base_gpu_id=0,
            consumer_count=4,
        )
        with patch.object(BaseMultimodalProcessor, "__abstractmethods__", set()):
            processor = BaseMultimodalProcessor.__new__(BaseMultimodalProcessor)
        processor.use_cuda_ipc = True
        processor.use_ipc_pool_handle_cache = True
        processor.cudaipc_mmfeature_pool = pool
        features = [
            torch.ones(16, device="cuda"),
            torch.empty(0, device="cuda"),
        ]
        items = [
            MultimodalDataItem(modality=Modality.IMAGE, feature=feature)
            for feature in features
        ]

        try:
            with self.assertRaisesRegex(ValueError, "empty tensor"):
                processor._prepare_mm_items_for_transport(items)

            deadline = time.monotonic() + 5
            while pool.active_lease_count and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(pool.active_lease_count, 0)
            self.assertIs(items[0].feature, features[0])
            self.assertIs(items[1].feature, features[1])
        finally:
            pool.shutdown()

    def test_rejected_request_releases_unconsumed_pool_slice(self):
        ctx = mp.get_context("spawn")
        proxy_queue = ctx.Queue()
        producer_results = ctx.Queue()
        consumer_done = ctx.Event()
        producer = ctx.Process(
            target=_produce_pooled_tensor,
            args=(proxy_queue, consumer_done, producer_results),
        )
        producer.start()
        proxy = None
        producer_result = None
        try:
            proxy, _ = proxy_queue.get(timeout=60)
            item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)
            mm_inputs = MultimodalInputs(mm_items=[item])

            mm_inputs.release_features()
            torch.cuda.synchronize()

            self.assertIsNone(item.feature)
        finally:
            del proxy
            _pool_handle_cache_clear()
            gc.collect()
            torch.cuda.ipc_collect()
            consumer_done.set()
            producer.join(timeout=60)
            try:
                producer_result = producer_results.get(timeout=5)
                status, payload = producer_result
                self.assertEqual(status, "ok", payload)
            finally:
                if producer.is_alive():
                    producer.terminate()
                    producer.join(timeout=10)
            self.assertEqual(producer.exitcode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
