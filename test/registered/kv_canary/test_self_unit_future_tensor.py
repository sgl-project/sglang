from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensors
from sglang.srt.utils import create_device_stream, current_device_stream
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=20, suite="extra-a-test-1-gpu-small-amd")
register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


class _FakeEvent:
    def __init__(self) -> None:
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1


class TestFutureTensors(CustomTestCase):
    def test_device_stage_then_wait_returns_host_copy(self) -> None:
        """Verify staged device tensors are copied back on wait."""
        alt_stream = create_device_stream(DEFAULT_DEVICE)
        default_stream = current_device_stream(DEFAULT_DEVICE)
        self.assertNotEqual(alt_stream.stream_id, default_stream.stream_id)

        src_first = torch.tensor([41], dtype=torch.int32, device=DEFAULT_DEVICE)
        future_first = FutureTensors.device_to_host(
            xs_device=src_first, d2h_stream=alt_stream
        )
        result_first = future_first.wait()
        self.assertEqual(int(result_first.item()), 41)

        src_second = torch.tensor([97], dtype=torch.int32, device=DEFAULT_DEVICE)
        future_second = FutureTensors.device_to_host(
            xs_device=src_second, d2h_stream=alt_stream
        )
        result_second = future_second.wait()
        self.assertEqual(int(result_second.item()), 97)

    def test_device_pinned_when_stream_is_provided(self) -> None:
        """Verify device staging uses pinned host memory with a stream."""
        alt_stream = create_device_stream(DEFAULT_DEVICE)
        src = torch.tensor([5], dtype=torch.int32, device=DEFAULT_DEVICE)
        future = FutureTensors.device_to_host(xs_device=src, d2h_stream=alt_stream)
        staged_tensors = [
            v for v in future._data.values() if isinstance(v, torch.Tensor)
        ]
        self.assertTrue(staged_tensors)
        self.assertTrue(all(t.is_pinned() for t in staged_tensors))
        self.assertEqual(int(future.wait().item()), 5)

    def test_device_each_call_allocates_fresh_host(self) -> None:
        """Verify each device staging call owns a fresh host buffer."""
        alt_stream = create_device_stream(DEFAULT_DEVICE)
        src_a = torch.tensor([13], dtype=torch.int32, device=DEFAULT_DEVICE)
        src_b = torch.tensor([29], dtype=torch.int32, device=DEFAULT_DEVICE)
        future_a = FutureTensors.device_to_host(xs_device=src_a, d2h_stream=alt_stream)
        future_b = FutureTensors.device_to_host(xs_device=src_b, d2h_stream=alt_stream)
        ptrs_a = {
            v.data_ptr() for v in future_a._data.values() if isinstance(v, torch.Tensor)
        }
        ptrs_b = {
            v.data_ptr() for v in future_b._data.values() if isinstance(v, torch.Tensor)
        }
        self.assertTrue(ptrs_a and ptrs_b)
        self.assertFalse(ptrs_a & ptrs_b)
        self.assertEqual(int(future_a.wait().item()), 13)
        self.assertEqual(int(future_b.wait().item()), 29)

    def test_dict_of_all_tensors_roundtrip(self) -> None:
        """Verify a dict of multiple tensors round-trips entry-by-entry."""
        stream = create_device_stream(DEFAULT_DEVICE)
        src = {
            "x": torch.tensor([11, 22], dtype=torch.int64, device=DEFAULT_DEVICE),
            "y": torch.tensor([99], dtype=torch.int32, device=DEFAULT_DEVICE),
        }
        future = FutureTensors.device_to_host(xs_device=src, d2h_stream=stream)
        out = future.wait()
        self.assertIsInstance(out, dict)
        self.assertEqual(out["x"].tolist(), [11, 22])
        self.assertEqual(int(out["y"].item()), 99)
        self.assertTrue(out["x"].is_pinned())
        self.assertTrue(out["y"].is_pinned())

    def test_dict_mixes_tensor_and_passthrough(self) -> None:
        """Verify non-tensor dict entries ride through verbatim alongside staging."""
        stream = create_device_stream(DEFAULT_DEVICE)
        sentinel_obj = {"nested": [1, 2, 3]}
        src = {
            "step": 42,
            "label": "decode",
            "extra": sentinel_obj,
            "counter": torch.tensor([7], dtype=torch.int32, device=DEFAULT_DEVICE),
        }
        future = FutureTensors.device_to_host(xs_device=src, d2h_stream=stream)
        out = future.wait()
        self.assertEqual(out["step"], 42)
        self.assertEqual(out["label"], "decode")
        # Identity (not deep-copy) — callers can rely on shared mutable references.
        self.assertIs(out["extra"], sentinel_obj)
        self.assertEqual(int(out["counter"].item()), 7)
        self.assertTrue(out["counter"].is_pinned())

    def test_dict_passthrough_preserves_tensor_value(self) -> None:
        """Verify tensors share device memory but non-tensor types are not staged."""
        stream = create_device_stream(DEFAULT_DEVICE)
        src_tensor = torch.tensor([3], dtype=torch.int32, device=DEFAULT_DEVICE)
        src = {"step": 100, "buf": src_tensor}
        future = FutureTensors.device_to_host(xs_device=src, d2h_stream=stream)
        out = future.wait()
        # Tensor is staged to a fresh pinned-host buffer (different storage from src).
        self.assertNotEqual(out["buf"].data_ptr(), src_tensor.data_ptr())
        self.assertTrue(out["buf"].is_pinned())
        # Non-tensor passes through with no copy.
        self.assertEqual(out["step"], 100)
        self.assertIsInstance(out["step"], int)

    def test_dict_without_tensor_raises(self) -> None:
        """Verify a tensor-less dict raises (no device to anchor the d2h sync)."""
        stream = create_device_stream(DEFAULT_DEVICE)
        with self.assertRaises(ValueError):
            FutureTensors.device_to_host(
                xs_device={"step": 0, "label": "decode"}, d2h_stream=stream
            )

    def test_wait_called_twice_raises(self) -> None:
        """Verify wait() after the first drain raises (state cleared)."""
        stream = create_device_stream(DEFAULT_DEVICE)
        src = torch.tensor([3], dtype=torch.int32, device=DEFAULT_DEVICE)
        future = FutureTensors.device_to_host(xs_device=src, d2h_stream=stream)
        self.assertEqual(int(future.wait().item()), 3)
        with self.assertRaises(RuntimeError):
            future.wait()

    def test_wait_clears_fields_and_rejects_second_wait(self) -> None:
        """Verify wait() syncs the event exactly once and clears internal state."""
        tensor = torch.tensor([1, 2, 3])
        event = _FakeEvent()
        future = FutureTensors(_data={"x": tensor}, _event=event)

        result = future.wait()
        self.assertIs(result["x"], tensor)
        self.assertEqual(event.synchronize_count, 1)
        self.assertIsNone(future._data)
        self.assertIsNone(future._event)

        with self.assertRaisesRegex(RuntimeError, "called more than once"):
            future.wait()
        # Failed wait must not re-trigger event.synchronize.
        self.assertEqual(event.synchronize_count, 1)

    def test_dict_anchor_picked_from_first_tensor(self) -> None:
        """Verify staging works when the first key is a non-tensor (anchor must scan)."""
        stream = create_device_stream(DEFAULT_DEVICE)
        src = {
            "step": 5,
            "buf": torch.tensor([17], dtype=torch.int32, device=DEFAULT_DEVICE),
        }
        out = FutureTensors.device_to_host(xs_device=src, d2h_stream=stream).wait()
        self.assertEqual(out["step"], 5)
        self.assertEqual(int(out["buf"].item()), 17)


if __name__ == "__main__":
    unittest.main()
