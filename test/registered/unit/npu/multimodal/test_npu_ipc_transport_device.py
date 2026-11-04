"""Real-device coverage for the npu_ipc multimodal feature transport.

Verifies the two NPU-specific assumptions behind the shared stream-ordered
pool (``sglang/srt.multimodal.transport.memory_pool``):
aclrtValueWait/aclrtValueWrite work on the consumer's IPC-opened mapping —
so the ready/ack words need no interprocess event — including when the
consumer sits on a different NPU than the pool.

Each test spawns a producer that wraps a feature into the bounded pool and a
consumer process that opens it, waits the ready word, reads the slice, and
acknowledges the lease.
"""

import gc
import multiprocessing as mp
import queue
import time
import unittest

import torch

from sglang.srt.multimodal.transport.cuda_ipc import MmItemMemoryPool
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

# Registered on a two-NPU suite so the cross-device case actually runs.
register_npu_ci(est_time=30, suite="base-b-test-2-npu-a3")

POOL_BYTES = 1 << 20
# One consumer slot keeps the acknowledgement independent of the parallel
# state, which a bare spawned process has not initialized.
CONSUMER_SLOTS = 1


def _produce_pooled_tensor(proxy_queue, consumer_done, result_queue):
    """Create a tokenizer-worker-like NPU IPC pool in a spawned producer."""
    pool = source = proxy = None
    try:
        torch.npu.set_device(0)
        pool = MmItemMemoryPool(
            memory_size=POOL_BYTES,
            recycle_interval=0.01,
            base_gpu_id=0,
            consumer_count=CONSUMER_SLOTS,
        )
        source = torch.arange(35, dtype=torch.float32, device="npu").reshape(5, 7)
        expected = torch.arange(35, dtype=torch.float32).reshape(5, 7).tolist()
        proxy = pool.wrap_tensor(source, use_pool_handle_cache=True)
        if proxy is None:
            raise RuntimeError("test tensor did not fit in the NPU IPC pool")
        proxy_queue.put((proxy, expected))
        if not consumer_done.wait(timeout=60):
            raise TimeoutError("consumer did not acknowledge the NPU IPC tensor")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if pool.active_lease_count == 0:
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("pool did not observe the consumer acknowledgement")
    except Exception as exc:  # pragma: no cover - returned to the parent
        result_queue.put(("error", repr(exc)))
        return
    finally:
        del proxy, source
        if pool is not None:
            pool.shutdown()
            del pool
        gc.collect()
    result_queue.put(("ok", None))


def _consume_pooled_tensor(proxy_queue, result_queue, rebuild_device=0):
    """Open the pool and read the slice in a spawned scheduler-like consumer."""
    try:
        torch.npu.set_device(rebuild_device)
        try:
            proxy, expected = proxy_queue.get(timeout=60)
        except queue.Empty:
            raise TimeoutError("consumer never received the NPU IPC proxy")
        # This reconstruct value_waits the ready word the producer wrote in
        # another process, on a mapping opened from the producer's handle.
        reconstructed = proxy.reconstruct_on_target_device(rebuild_device)
        torch.npu.synchronize()
        if reconstructed.cpu().tolist() != expected:
            raise AssertionError("the consumer read back the wrong feature")
    except Exception as exc:  # pragma: no cover - returned to the parent
        result_queue.put(("error", repr(exc)))
        return
    # The process-wide pool mapping dies with this process; the transport
    # has no manual cache teardown because the cache lives exactly as long
    # as the process holding it.
    result_queue.put(("ok", None))


def _run_producer_consumer_test(test, consumer_device=0):
    ctx = mp.get_context("spawn")
    proxy_queue = ctx.Queue()
    producer_results = ctx.Queue()
    consumer_results = ctx.Queue()
    consumer_done = ctx.Event()
    producer = ctx.Process(
        target=_produce_pooled_tensor,
        args=(proxy_queue, consumer_done, producer_results),
    )
    consumer = ctx.Process(
        target=_consume_pooled_tensor,
        args=(proxy_queue, consumer_results, consumer_device),
    )
    producer.start()
    consumer.start()
    try:
        try:
            consumer_result = consumer_results.get(timeout=120)
        except queue.Empty:
            test.fail("NPU IPC consumer did not report in time")
        consumer.join(timeout=10)
        if consumer_result[0] != "ok":
            test.fail(f"NPU IPC consumer failed: {consumer_result[1]}")
        test.assertEqual(consumer.exitcode, 0)

        # The consumer's process-wide pool mapping died with its process,
        # so the producer may now destroy the shared allocation.
        consumer_done.set()
        try:
            producer_result = producer_results.get(timeout=120)
        except queue.Empty:
            test.fail("NPU IPC producer did not finish after the consumer exited")
        test.assertEqual(producer_result[0], "ok", producer_result[1])
        producer.join(timeout=10)
        test.assertEqual(producer.exitcode, 0)
    finally:
        # Whatever failed, never strand the producer on its wait for the
        # consumer, and leave no child process behind.
        consumer_done.set()
        for process in (consumer, producer):
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.is_available(), "requires an NPU device"
)
class TestNpuIpcTransportOnDevice(CustomTestCase):
    def test_pooled_tensor_reconstructs_in_a_spawned_process(self):
        """The consumer process exits before the producer destroys the pool."""
        _run_producer_consumer_test(self, consumer_device=0)

    @unittest.skipUnless(
        hasattr(torch, "npu") and torch.npu.device_count() >= 2,
        "requires two NPUs",
    )
    def test_pooled_tensor_reconstructs_across_devices(self):
        """The consumer waits and acknowledges through a cross-device mapping.

        TP deployments put scheduler ranks on other devices than the pool's
        base NPU; the control-word wait/write must work over that peer view
        of the imported pool, not only on the pool's home device.
        """
        _run_producer_consumer_test(self, consumer_device=1)


if __name__ == "__main__":
    unittest.main()
