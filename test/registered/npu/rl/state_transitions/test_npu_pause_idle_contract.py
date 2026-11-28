"""NPU: ``pause()`` must leave the device idle, so the unmap cannot land under it.

``TorchMemorySaverAdapter.pause()`` hands the region to the memory saver, whose
``aclrtUnmapMem`` does not wait for the device. This case asserts the contract
behind that: once ``pause()`` returns, the device is idle.

The work is queued before the region is entered, so nothing is allocated inside
it — not the matmul outputs, and not the scratch an aclnn op may take from the
current pool. The region therefore wraps only ``pause()``/``resume()``, and a
missing drain leaves the stream busy instead of faulting on unmapped pages.
"""

import time
import unittest

import torch

from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=120, suite="nightly-test-1-npu-a3", nightly=True)

TAG = "pause_idle"
MATMUL = 4096
# A 4096^2 fp16 matmul drains in ~0.4 ms on Atlas A2/A3, so this queues far more
# than the few ms an unmap needs and the queue is still busy when pause() runs.
ITERATIONS = 256


def _saver_available():
    # create(enable=True) raises when the package is missing, so availability has
    # to be decided before the adapter is built.
    try:
        import torch_memory_saver  # noqa: F401
    except ImportError:
        return False
    return True


_SAVER_OK = _saver_available()


@unittest.skipUnless(_SAVER_OK, "requires torch_memory_saver")
class TestPauseLeavesIdleDevice(CustomTestCase):
    def test_pause_waits_for_queued_work(self):
        adapter = TorchMemorySaverAdapter.create(enable=True)
        if not adapter.enabled:
            self.skipTest("torch_memory_saver is enabled but not active")

        device_module = torch.get_device_module()
        stream = device_module.current_stream()
        device_module.synchronize()

        left = torch.randn(MATMUL, MATMUL, dtype=torch.float16, device="npu")
        right = torch.randn(MATMUL, MATMUL, dtype=torch.float16, device="npu")
        out = torch.empty(MATMUL, MATMUL, dtype=torch.float16, device="npu")
        for _ in range(ITERATIONS):
            torch.matmul(left, right, out=out)

        try:
            with adapter.region(TAG, enable_cpu_backup=False):
                self.assertFalse(
                    stream.query(),
                    "the queued work drained before pause(); raise ITERATIONS",
                )
                start = time.time()
                adapter.pause(TAG)
                elapsed_ms = (time.time() - start) * 1000
                idle = stream.query()
                print(
                    f"\n[pause_idle] pause() {elapsed_ms:.1f} ms, idle={idle}",
                    flush=True,
                )
                self.assertTrue(
                    idle,
                    f"pause() returned after {elapsed_ms:.1f} ms with work still "
                    "queued -- it did not drain the device before handing the "
                    "region to the memory saver",
                )
                adapter.resume(TAG)
        finally:
            # A hardware error here must not replace the assertion above.
            try:
                device_module.synchronize()
            except RuntimeError:
                pass


if __name__ == "__main__":
    unittest.main()
