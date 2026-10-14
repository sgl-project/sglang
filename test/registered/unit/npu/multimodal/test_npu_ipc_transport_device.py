"""Real-device coverage for the NPU IPC multimodal feature transport.

The bookkeeping tests in ``test/registered/unit/multimodal`` run on CPU because
the lease/acknowledgement state machine is device independent. This file covers
what only an Ascend device can: a tokenizer-worker-like producer placing a
feature in the bounded NPU pool, and a scheduler-like consumer in another
process opening that pool, reading the slice, and acknowledging the lease.
"""

import gc
import multiprocessing as mp
import queue
import time
import unittest

import torch

from sglang.srt.multimodal.transport.npu_ipc import MmItemMemoryPool
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=30, suite="stage-a-unit-test-npu")

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
            base_npu_id=0,
            tp_size=CONSUMER_SLOTS,
        )
        source = torch.arange(35, dtype=torch.float32, device="npu").reshape(5, 7)
        expected = torch.arange(35, dtype=torch.float32).reshape(5, 7).tolist()
        proxy = pool.wrap_tensor(source, use_pool_handle_cache=True)
        if proxy is None:
            raise RuntimeError("test tensor did not fit in the NPU IPC pool")
        # The pool owner hashes a feature before it dispatches the proxy. That
        # read belongs to this process and must not travel to the consumer.
        if proxy.reconstruct_on_target_device(0).cpu().tolist() != expected:
            raise RuntimeError("the pool owner read back the wrong feature")
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


def _consume_pooled_tensor(proxy_queue, result_queue):
    """Open the pool and read the slice in a spawned scheduler-like consumer."""
    try:
        torch.npu.set_device(0)
        try:
            proxy, expected = proxy_queue.get(timeout=60)
        except queue.Empty:
            raise TimeoutError("consumer never received the NPU IPC proxy")
        # The owner's hashing read was dropped by the IPC hop, so this
        # process owes the pool an acknowledgement for the lease.
        if proxy.reconstruct_tensor is not None:
            raise AssertionError("the IPC hop leaked the owner's hashing read")
        reconstructed = proxy.reconstruct_on_target_device(0)
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


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.is_available(), "requires an Ascend NPU"
)
class TestNpuIpcTransportOnDevice(CustomTestCase):
    def test_pooled_tensor_reconstructs_in_a_spawned_process(self):
        """The consumer process exits before the producer destroys the pool."""
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
            args=(proxy_queue, consumer_results),
        )
        producer.start()
        consumer.start()
        try:
            try:
                consumer_result = consumer_results.get(timeout=120)
            except queue.Empty:
                self.fail("NPU IPC consumer did not report in time")
            consumer.join(timeout=10)
            if consumer_result[0] != "ok":
                self.fail(f"NPU IPC consumer failed: {consumer_result[1]}")
            self.assertEqual(consumer.exitcode, 0)

            # The consumer's process-wide pool mapping died with its process,
            # so the producer may now destroy the shared allocation.
            consumer_done.set()
            try:
                producer_result = producer_results.get(timeout=120)
            except queue.Empty:
                self.fail("NPU IPC producer did not finish after the consumer exited")
            self.assertEqual(producer_result[0], "ok", producer_result[1])
            producer.join(timeout=10)
            self.assertEqual(producer.exitcode, 0)
        finally:
            # Whatever failed, never strand the producer on its wait for the
            # consumer, and leave no child process behind.
            consumer_done.set()
            for process in (consumer, producer):
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
