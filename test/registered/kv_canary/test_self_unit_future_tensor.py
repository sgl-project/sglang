from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


class TestFutureTensor(CustomTestCase):
    def test_cpu_fallback_returns_host_copy_immediately(self) -> None:
        src = torch.tensor([7], dtype=torch.int32)
        future = FutureTensor.create(src_device=src, stream=None)
        self.assertIsInstance(future, FutureTensor)
        result = future.wait()
        self.assertEqual(int(result.item()), 7)
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(tuple(result.shape), (1,))

    def test_cpu_fallback_does_not_allocate_pinned(self) -> None:
        src = torch.tensor([3], dtype=torch.int32)
        future = FutureTensor.create(src_device=src, stream=None)
        self.assertFalse(future._tensor.is_pinned())

    def test_cpu_fallback_each_call_allocates_fresh_host(self) -> None:
        src_a = torch.tensor([11], dtype=torch.int32)
        src_b = torch.tensor([22], dtype=torch.int32)
        future_a = FutureTensor.create(src_device=src_a, stream=None)
        future_b = FutureTensor.create(src_device=src_b, stream=None)
        self.assertNotEqual(future_a._tensor.data_ptr(), future_b._tensor.data_ptr())
        self.assertEqual(int(future_a.wait().item()), 11)
        self.assertEqual(int(future_b.wait().item()), 22)

    def test_cuda_stage_then_wait_returns_host_copy(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        default_stream = torch.cuda.current_stream(device)
        self.assertNotEqual(alt_stream.stream_id, default_stream.stream_id)

        src_first = torch.tensor([41], dtype=torch.int32, device=device)
        future_first = FutureTensor.create(src_device=src_first, stream=alt_stream)
        result_first = future_first.wait()
        self.assertEqual(int(result_first.item()), 41)

        src_second = torch.tensor([97], dtype=torch.int32, device=device)
        future_second = FutureTensor.create(src_device=src_second, stream=alt_stream)
        result_second = future_second.wait()
        self.assertEqual(int(result_second.item()), 97)

    def test_cuda_pinned_when_stream_is_provided(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        src = torch.tensor([5], dtype=torch.int32, device=device)
        future = FutureTensor.create(src_device=src, stream=alt_stream)
        self.assertTrue(future._tensor.is_pinned())
        self.assertEqual(int(future.wait().item()), 5)

    def test_cuda_each_call_allocates_fresh_host(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        src_a = torch.tensor([13], dtype=torch.int32, device=device)
        src_b = torch.tensor([29], dtype=torch.int32, device=device)
        future_a = FutureTensor.create(src_device=src_a, stream=alt_stream)
        future_b = FutureTensor.create(src_device=src_b, stream=alt_stream)
        self.assertNotEqual(future_a._tensor.data_ptr(), future_b._tensor.data_ptr())
        self.assertEqual(int(future_a.wait().item()), 13)
        self.assertEqual(int(future_b.wait().item()), 29)


if __name__ == "__main__":
    unittest.main()
