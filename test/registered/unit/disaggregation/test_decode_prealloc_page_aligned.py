import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation import decode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, page_size: int = 4) -> None:
        self.page_size = page_size
        self.device = torch.device("cpu")
        self.calls: list[dict[str, object]] = []

    def alloc_extend(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.arange(int(kwargs["extend_num_tokens"]), dtype=torch.int64)

    def alloc_logical_only(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.arange(int(kwargs["extend_num_tokens"]), dtype=torch.int64)

    def alloc_extend_swa_tail(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.arange(int(kwargs["extend_num_tokens"]), dtype=torch.int64)


class TestDecodePreallocPageAligned(unittest.TestCase):
    def test_decode_prealloc_rounds_physical_endpoint(self) -> None:
        """PD preallocation rounds physical capacity without changing fill_len."""
        allocator = _Allocator()
        req = SimpleNamespace(kv=None)

        with mock.patch.object(decode, "_is_npu", False):
            locations = decode.alloc_for_decode_prealloc(
                req=req,
                allocator=allocator,
                total_prefix_len=3,
                prefix_len=3,
                prefix_indices=torch.tensor([5, 6, 7], dtype=torch.int64),
                fill_len=5,
                delta_len=2,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        self.assertEqual(req.kv.kv_allocated_len, 8)
        self.assertEqual(locations.numel(), 5)
        self.assertEqual(allocator.calls[0]["seq_lens_cpu"].tolist(), [8])
        self.assertEqual(allocator.calls[0]["extend_num_tokens"], 5)

    def test_hisparse_prealloc_rounds_logical_capacity(self) -> None:
        """HiSparse preallocation publishes slots through the aligned endpoint."""
        allocator = _Allocator()
        req = SimpleNamespace(kv=None)

        with mock.patch.object(decode, "_is_npu", False):
            locations = decode.alloc_for_decode_prealloc_hisparse(
                req=req,
                allocator=allocator,
                fill_len=5,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        self.assertEqual(req.kv.kv_allocated_len, 8)
        self.assertEqual(locations.numel(), 8)
        self.assertEqual(allocator.calls[0]["seq_lens_cpu"].tolist(), [8])

    def test_decode_prealloc_aligns_swa_tail_around_real_endpoint(self) -> None:
        """SWA-tail preallocation separates padded capacity from the real endpoint."""
        allocator = _Allocator()
        req = SimpleNamespace(kv=None)

        with mock.patch.object(decode, "_is_npu", False):
            locations = decode.alloc_for_decode_prealloc(
                req=req,
                allocator=allocator,
                total_prefix_len=0,
                prefix_len=0,
                prefix_indices=torch.empty((0,), dtype=torch.int64),
                fill_len=6,
                delta_len=6,
                uses_swa_tail=True,
                swa_tail_len=2,
            )

        self.assertEqual(req.kv.kv_allocated_len, 8)
        self.assertEqual(locations.numel(), 8)
        self.assertEqual(allocator.calls[0]["seq_lens_cpu"].tolist(), [8])
        self.assertEqual(allocator.calls[0]["swa_tail_end"], 6)

    def test_npu_prealloc_keeps_continuation_length(self) -> None:
        """NPU preallocation retains its existing real-length continuation."""
        allocator = _Allocator()
        req = SimpleNamespace(kv=None)

        with mock.patch.object(decode, "_is_npu", True):
            locations = decode.alloc_for_decode_prealloc(
                req=req,
                allocator=allocator,
                total_prefix_len=3,
                prefix_len=3,
                prefix_indices=torch.tensor([5, 6, 7], dtype=torch.int64),
                fill_len=5,
                delta_len=2,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        self.assertEqual(req.kv.kv_allocated_len, 5)
        self.assertEqual(locations.numel(), 2)
        self.assertEqual(allocator.calls[0]["seq_lens_cpu"].tolist(), [5])


if __name__ == "__main__":
    unittest.main()
