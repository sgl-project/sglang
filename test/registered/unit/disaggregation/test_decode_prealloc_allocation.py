import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation import decode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_req():
    return SimpleNamespace(kv=None)


class TestDecodePreallocAllocation(unittest.TestCase):
    def test_hisparse_prealloc_uses_only_page_aligned_logical_allocation(self) -> None:
        """HiSparse PD preallocation passes only the aligned logical size."""
        allocator = SimpleNamespace(
            page_size=4,
            alloc_logical_only=Mock(
                return_value=torch.arange(4, 12, dtype=torch.int64)
            ),
        )
        req = _make_req()

        result = decode.alloc_for_decode_prealloc_hisparse(
            allocator,
            req=req,
            fill_len=6,
        )

        allocator.alloc_logical_only.assert_called_once_with(need_size=8)
        self.assertEqual(result.numel(), 8)
        self.assertEqual(req.kv.kv_allocated_len, 8)

    def test_non_npu_no_tail_prealloc_calls_direct_allocator(self) -> None:
        """Non-NPU page allocation avoids legacy lengths and last_loc."""
        allocator = SimpleNamespace(
            page_size=4,
            alloc=Mock(return_value=torch.arange(8, 12, dtype=torch.int64)),
        )
        req = _make_req()

        with patch.object(decode, "_is_npu", False):
            result = decode.alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=8,
                delta_len=4,
                prefix_len=4,
                total_prefix_len=4,
                prefix_indices=torch.arange(4, 8, dtype=torch.int64),
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        allocator.alloc.assert_called_once_with(4)
        self.assertEqual(result.numel(), 4)

    def test_non_npu_tail_prealloc_uses_reduced_direct_signature(self) -> None:
        """Non-NPU SWA-tail allocation passes only aligned counts and tail bounds."""
        allocator = SimpleNamespace(
            page_size=4,
            alloc_extend_swa_tail=Mock(
                return_value=torch.arange(4, 12, dtype=torch.int64)
            ),
        )
        req = _make_req()

        with patch.object(decode, "_is_npu", False):
            decode.alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=6,
                delta_len=6,
                prefix_len=0,
                total_prefix_len=0,
                prefix_indices=None,
                uses_swa_tail=True,
                swa_tail_len=6,
            )

        allocator.alloc_extend_swa_tail.assert_called_once_with(
            extend_num_tokens=8,
            swa_tail_len=6,
            swa_tail_end=6,
        )
        self.assertEqual(req.kv.swa_evicted_seqlen, 0)

    def test_npu_tail_prealloc_preserves_real_length_legacy_inputs(self) -> None:
        """NPU SWA-tail dispatch keeps real lengths and the prefix anchor."""
        allocator = SimpleNamespace(page_size=4, device="cpu")
        req = _make_req()
        prefix_indices = torch.arange(4, 8, dtype=torch.int64)

        with patch.object(decode, "_is_npu", True), patch(
            "sglang.srt.hardware_backend.npu.allocator_npu.alloc_extend_swa_tail_npu",
            return_value=torch.arange(8, 12, dtype=torch.int64),
        ) as npu_tail_allocator:
            decode.alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=8,
                delta_len=4,
                prefix_len=4,
                total_prefix_len=4,
                prefix_indices=prefix_indices,
                uses_swa_tail=True,
                swa_tail_len=4,
            )

        npu_tail_allocator.assert_called_once()
        call_kwargs = npu_tail_allocator.call_args.kwargs
        self.assertEqual(call_kwargs["extend_num_tokens"], 8)
        self.assertEqual(call_kwargs["swa_tail_len"], 4)
        self.assertEqual(call_kwargs["swa_tail_end"], 8)
        self.assertTrue(torch.equal(call_kwargs["last_loc"], prefix_indices[-1:]))


if __name__ == "__main__":
    unittest.main()
