"""Manual smoke: pausing a memory-saver region must drain the device first.

``TorchMemorySaverAdapter.pause()`` hands the region to the memory saver, whose
``aclrtUnmapMem`` does not wait for the device. This queues a batch of matmuls,
never waits for them, and then pauses: the drain inside ``pause()`` lets them
finish before the pages are unmapped, while without it the in-flight kernels read
released pages and the device raises an aicore exception (507015).

Kept out of the registered suites on purpose: the CPU release-order test already
guards the ordering deterministically, and a failing run here leaves the device
in an error state that would take the rest of the suite with it. Run it by hand:

    ASCEND_RT_VISIBLE_DEVICES=0 python test/manual/ascend/test_npu_device_synchronize.py
"""

import unittest

import torch

from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.test.test_utils import CustomTestCase

TAG = "pause_race"
MATMUL = 4096
# A 4096^2 fp16 matmul drains in ~0.4 ms on Atlas A2/A3, so this queues far more
# work than the few ms an unmap needs -- the queue is still busy when pause()
# runs, which is what the test asserts below.
ITERATIONS = 256


class TestPauseDrainsDevice(CustomTestCase):
    def test_pause_waits_for_queued_work(self):
        adapter = TorchMemorySaverAdapter.create(enable=True)
        if not adapter.enabled:
            self.skipTest("torch_memory_saver is not enabled; nothing would unmap")

        device_module = torch.get_device_module()
        stream = device_module.current_stream()
        device_module.synchronize()
        try:
            with adapter.region(TAG, enable_cpu_backup=False):
                a = torch.randn(MATMUL, MATMUL, dtype=torch.float16, device="npu")
                b = torch.randn(MATMUL, MATMUL, dtype=torch.float16, device="npu")
                for _ in range(ITERATIONS):
                    a = a @ b
                self.assertFalse(
                    stream.query(),
                    "the queued work drained before pause(); raise ITERATIONS",
                )
                adapter.pause(TAG)
                self.assertTrue(
                    stream.query(),
                    "pause() returned with work still queued -- the unmap raced "
                    "the in-flight kernels instead of draining the device first",
                )
                adapter.resume(TAG)
        finally:
            # A failing case leaves the device in an error state; swallowing the
            # trailing synchronize keeps the assertion above as the reported cause.
            try:
                device_module.synchronize()
            except RuntimeError:
                pass


if __name__ == "__main__":
    unittest.main()
