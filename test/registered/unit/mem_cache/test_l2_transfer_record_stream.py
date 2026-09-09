"""Index tensors on any accelerator must be kept alive across an L2 copy.

Regression guard: index tensors handed to an async host<->device copy were only
pinned to the transfer stream when the tensor was CUDA, so on every other
accelerator they could be freed and reused while the copy was still in flight.
"""

import unittest

import torch

from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _IndexTensorStub:
    """Stands in for an index tensor resident on ``device_type``.

    A real tensor cannot express the case under test: xpu/npu/musa tensors are
    unallocatable on CI, and on CUDA the two predicates agree for every tensor
    (cuda -> both true, pinned cpu -> both false), so only a non-CUDA
    accelerator separates ``is_cuda`` from ``device.type != "cpu"``.
    """

    def __init__(self, device_type: str):
        self.device = torch.device(device_type)
        self.is_cuda = device_type == "cuda"
        self.recorded = []

    def record_stream(self, stream):
        self.recorded.append(stream)


def _transfer(host_indices, device_indices) -> L2Transfer:
    return L2Transfer(
        host_pool=None,
        device_pool=None,
        host_indices=host_indices,
        device_indices=device_indices,
    )


class TestL2TransferRecordStream(CustomTestCase):
    def test_non_cuda_accelerator_indices_are_kept_alive(self):
        host = _IndexTensorStub("cpu")
        device = _IndexTensorStub("xpu")
        stream = object()

        L2TransferEngine._record_stream([_transfer(host, device)], stream)

        self.assertEqual(
            device.recorded,
            [stream],
            "xpu index tensor was not pinned to the transfer stream; it can be "
            "freed and reused while the copy is in flight",
        )
        self.assertEqual(
            host.recorded, [], "record_stream is meaningless for a host tensor"
        )

    def test_cuda_indices_are_kept_alive(self):
        host = _IndexTensorStub("cpu")
        device = _IndexTensorStub("cuda")
        stream = object()

        L2TransferEngine._record_stream([_transfer(host, device)], stream)

        self.assertEqual(device.recorded, [stream])
        self.assertEqual(host.recorded, [])


if __name__ == "__main__":
    unittest.main()
