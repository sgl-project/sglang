from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


class TestFutureTensor(CustomTestCase):
    def test_cuda_stage_then_wait_returns_host_copy(self) -> None:
        """Verify staged CUDA tensors are copied back on wait."""
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        default_stream = torch.cuda.current_stream(device)
        self.assertNotEqual(alt_stream.stream_id, default_stream.stream_id)

        src_first = torch.tensor([41], dtype=torch.int32, device=device)
        future_first = FutureTensor.device_to_host(src_device=src_first, stream=alt_stream)
        result_first = future_first.wait()
        self.assertEqual(int(result_first.item()), 41)

        src_second = torch.tensor([97], dtype=torch.int32, device=device)
        future_second = FutureTensor.device_to_host(src_device=src_second, stream=alt_stream)
        result_second = future_second.wait()
        self.assertEqual(int(result_second.item()), 97)

    def test_cuda_pinned_when_stream_is_provided(self) -> None:
        """Verify CUDA staging uses pinned host memory with a stream."""
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        src = torch.tensor([5], dtype=torch.int32, device=device)
        future = FutureTensor.device_to_host(src_device=src, stream=alt_stream)
        self.assertTrue(future._tensor.is_pinned())
        self.assertEqual(int(future.wait().item()), 5)

    def test_cuda_each_call_allocates_fresh_host(self) -> None:
        """Verify each CUDA staging call owns a fresh host buffer."""
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        src_a = torch.tensor([13], dtype=torch.int32, device=device)
        src_b = torch.tensor([29], dtype=torch.int32, device=device)
        future_a = FutureTensor.device_to_host(src_device=src_a, stream=alt_stream)
        future_b = FutureTensor.device_to_host(src_device=src_b, stream=alt_stream)
        self.assertNotEqual(future_a._tensor.data_ptr(), future_b._tensor.data_ptr())
        self.assertEqual(int(future_a.wait().item()), 13)
        self.assertEqual(int(future_b.wait().item()), 29)


if __name__ == "__main__":
    unittest.main()
