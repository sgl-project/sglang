"""CUDA IPC multimodal feature transport regression tests.

This covers the production path where a tokenizer worker places a feature in
the bounded pool and the scheduler process opens the shared CUDA allocation.
CPU-only policy tests intentionally cannot exercise this cross-process handle.
"""

import gc
import multiprocessing as mp
import queue
import unittest

import torch

from sglang.srt.utils.cuda_ipc_transport_utils import (
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    _pool_handle_cache_clear,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


def _produce_pooled_tensor(proxy_queue, consumer_done, result_queue):
    """Create a tokenizer-worker-like CUDA IPC pool in a spawned producer."""
    pool = source = pool_slice = proxy = None
    try:
        torch.cuda.set_device(0)
        pool = MmItemMemoryPool(
            memory_size=1 << 20,
            recycle_interval=60,
            base_gpu_id=0,
        )
        source = torch.arange(35, dtype=torch.float32, device="cuda").reshape(5, 7)
        expected = source.cpu().tolist()
        sync_meta, pool_slice, byte_offset = pool.return_a_slice_tensor_with_flag(
            source
        )
        if pool_slice is None:
            raise RuntimeError("test tensor did not fit in the CUDA IPC pool")
        pool_slice.copy_(source.view(torch.int8).view(-1), non_blocking=True)
        torch.cuda.synchronize()
        proxy = CudaIpcTensorTransportProxy(
            data=pool_slice,
            info_data=source,
            sync_buffer_meta=sync_meta,
            pool_ipc_handle=pool._pool_ipc_handle,
            pool_byte_offset=byte_offset,
            pool_device_index=pool._pool_device_index,
        )
        proxy_queue.put((proxy, expected))
        if not consumer_done.wait(timeout=60):
            raise TimeoutError("consumer did not release the CUDA IPC tensor")
    except Exception as exc:  # pragma: no cover - returned to the parent
        result_queue.put(("error", repr(exc)))
        return
    finally:
        del proxy, pool_slice, source
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
