from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensors
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")


class TestFutureTensors(CustomTestCase):
    def test_cuda_stage_then_wait_returns_host_copy(self) -> None:
        """Verify staged CUDA tensors are copied back on wait."""
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        default_stream = torch.cuda.current_stream(device)
        self.assertNotEqual(alt_stream.stream_id, default_stream.stream_id)

        src_first = torch.tensor([41], dtype=torch.int32, device=device)
        future_first = FutureTensors.device_to_host(
            xs_device=src_first, stream=alt_stream
        )
        result_first = future_first.wait()
        self.assertEqual(int(result_first.item()), 41)

        src_second = torch.tensor([97], dtype=torch.int32, device=device)
        future_second = FutureTensors.device_to_host(
            xs_device=src_second, stream=alt_stream
        )
        result_second = future_second.wait()
        self.assertEqual(int(result_second.item()), 97)

    def test_cuda_pinned_when_stream_is_provided(self) -> None:
        """Verify CUDA staging uses pinned host memory with a stream."""
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        src = torch.tensor([5], dtype=torch.int32, device=device)
        future = FutureTensors.device_to_host(xs_device=src, stream=alt_stream)
        self.assertTrue(future._data.is_pinned())
        self.assertEqual(int(future.wait().item()), 5)

    def test_cuda_each_call_allocates_fresh_host(self) -> None:
        """Verify each CUDA staging call owns a fresh host buffer."""
        device = torch.device("cuda")
        alt_stream = torch.cuda.Stream(device=device)
        src_a = torch.tensor([13], dtype=torch.int32, device=device)
        src_b = torch.tensor([29], dtype=torch.int32, device=device)
        future_a = FutureTensors.device_to_host(xs_device=src_a, stream=alt_stream)
        future_b = FutureTensors.device_to_host(xs_device=src_b, stream=alt_stream)
        self.assertNotEqual(future_a._data.data_ptr(), future_b._data.data_ptr())
        self.assertEqual(int(future_a.wait().item()), 13)
        self.assertEqual(int(future_b.wait().item()), 29)

    def test_dict_of_all_tensors_roundtrip(self) -> None:
        """Verify a dict of multiple tensors round-trips entry-by-entry."""
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)
        src = {
            "x": torch.tensor([11, 22], dtype=torch.int64, device=device),
            "y": torch.tensor([99], dtype=torch.int32, device=device),
        }
        future = FutureTensors.device_to_host(xs_device=src, stream=stream)
        out = future.wait()
        self.assertIsInstance(out, dict)
        self.assertEqual(out["x"].tolist(), [11, 22])
        self.assertEqual(int(out["y"].item()), 99)
        self.assertTrue(out["x"].is_pinned())
        self.assertTrue(out["y"].is_pinned())

    def test_dict_mixes_tensor_and_passthrough(self) -> None:
        """Verify non-tensor dict entries ride through verbatim alongside staging."""
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)
        sentinel_obj = {"nested": [1, 2, 3]}
        src = {
            "step": 42,
            "label": "decode",
            "extra": sentinel_obj,
            "counter": torch.tensor([7], dtype=torch.int32, device=device),
        }
        future = FutureTensors.device_to_host(xs_device=src, stream=stream)
        out = future.wait()
        self.assertEqual(out["step"], 42)
        self.assertEqual(out["label"], "decode")
        # Identity (not deep-copy) — callers can rely on shared mutable references.
        self.assertIs(out["extra"], sentinel_obj)
        self.assertEqual(int(out["counter"].item()), 7)
        self.assertTrue(out["counter"].is_pinned())

    def test_dict_passthrough_preserves_tensor_value(self) -> None:
        """Verify tensors share device memory but non-tensor types are not staged."""
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)
        src_tensor = torch.tensor([3], dtype=torch.int32, device=device)
        src = {"step": 100, "buf": src_tensor}
        future = FutureTensors.device_to_host(xs_device=src, stream=stream)
        out = future.wait()
        # Tensor is staged to a fresh pinned-host buffer (different storage from src).
        self.assertNotEqual(out["buf"].data_ptr(), src_tensor.data_ptr())
        self.assertTrue(out["buf"].is_pinned())
        # Non-tensor passes through with no copy.
        self.assertEqual(out["step"], 100)
        self.assertIsInstance(out["step"], int)

    def test_dict_without_tensor_raises(self) -> None:
        """Verify a tensor-less dict raises (no device to anchor the d2h sync)."""
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)
        with self.assertRaises(ValueError):
            FutureTensors.device_to_host(
                xs_device={"step": 0, "label": "decode"}, stream=stream
            )

    def test_wait_called_twice_raises(self) -> None:
        """Verify wait() after the first drain raises (state cleared)."""
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)
        src = torch.tensor([3], dtype=torch.int32, device=device)
        future = FutureTensors.device_to_host(xs_device=src, stream=stream)
        self.assertEqual(int(future.wait().item()), 3)
        with self.assertRaises(RuntimeError):
            future.wait()

    def test_dict_anchor_picked_from_first_tensor(self) -> None:
        """Verify staging works when the first key is a non-tensor (anchor must scan)."""
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)
        src = {
            "step": 5,
            "buf": torch.tensor([17], dtype=torch.int32, device=device),
        }
        out = FutureTensors.device_to_host(xs_device=src, stream=stream).wait()
        self.assertEqual(out["step"], 5)
        self.assertEqual(int(out["buf"].item()), 17)


if __name__ == "__main__":
    unittest.main()
