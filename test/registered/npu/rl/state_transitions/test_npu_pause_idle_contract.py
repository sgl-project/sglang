"""NPU: ``pause()`` must leave the device idle, so the unmap cannot land under it.

``TorchMemorySaverAdapter.pause()`` hands the region to the memory saver, whose
``aclrtUnmapMem`` does not wait for the device. The busy work below is queued
*outside* the paused region on purpose: what this case asserts is the contract
("pause returns only once the device is idle"), and a missing drain then shows up
as a stream that is still busy rather than as a silent corruption.

If the drain is missing the device usually reports an aicore exception (507015)
while the work is being torn down; that error is surfaced as context on the
assertion message instead of replacing it, so a failing run still reads as
"pause() returned with work still queued".
"""

import time
import unittest

import torch

from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="extra-b-test-1-npu-a3")

TAG = "pause_idle"
MATMUL = 4096
# A 4096^2 fp16 matmul drains in ~0.4 ms on Atlas A2/A3, so this queues far more
# than the few ms an unmap needs and the queue is still busy when pause() runs.
ITERATIONS = 256


class TestPauseLeavesIdleDevice(CustomTestCase):
    def test_pause_waits_for_queued_work(self):
        adapter = TorchMemorySaverAdapter.create(enable=True)
        if not adapter.enabled:
            self.skipTest("torch_memory_saver is not enabled; nothing would unmap")

        device_module = torch.get_device_module()
        stream = device_module.current_stream()
        device_module.synchronize()

        left = torch.randn(MATMUL, MATMUL, dtype=torch.float16, device="npu")
        right = torch.randn(MATMUL, MATMUL, dtype=torch.float16, device="npu")
        try:
            with adapter.region(TAG, enable_cpu_backup=False):
                for _ in range(ITERATIONS):
                    left = left @ right
                self.assertFalse(
                    stream.query(),
                    "the queued work drained before pause(); raise ITERATIONS",
                )
                start = time.time()
                adapter.pause(TAG)
                elapsed_ms = (time.time() - start) * 1000
                idle = stream.query()
                print(f"\n[pause_idle] pause() {elapsed_ms:.1f} ms, idle={idle}", flush=True)
                if not idle:
                    drain_error = None
                    try:
                        device_module.synchronize()
                    except RuntimeError as exc:
                        drain_error = exc
                    self.fail(
                        f"pause() returned after {elapsed_ms:.1f} ms with work still "
                        "queued -- the unmap raced the in-flight kernels instead of "
                        f"draining the device (drain raised: {drain_error})"
                    )
                adapter.resume(TAG)
        finally:
            try:
                device_module.synchronize()
            except RuntimeError:
                pass


if __name__ == "__main__":
    unittest.main()
