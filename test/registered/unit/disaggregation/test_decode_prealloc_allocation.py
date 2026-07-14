import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation import decode
from sglang.srt.hardware_backend.npu import allocator_npu
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

    def test_npu_prealloc_routes_all_authority_to_explicit_entry(self) -> None:
        """NPU routing passes the complete preallocation contract to its entry."""
        allocator = SimpleNamespace(page_size=4, device="cpu")
        req = _make_req()

        with patch.object(decode, "_is_npu", True), patch.object(
            allocator_npu,
            "alloc_for_decode_prealloc_npu",
            return_value=torch.arange(8, 12, dtype=torch.int64),
        ) as npu_entry:
            decode.alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=8,
                delta_len=8,
                prefix_len=0,
                total_prefix_len=0,
                prefix_indices=None,
                uses_swa_tail=True,
                swa_tail_len=4,
            )

        npu_entry.assert_called_once_with(
            allocator,
            prefix_indices=None,
            fill_len=8,
            prefix_len=0,
            total_prefix_len=0,
            delta_len=8,
            uses_swa_tail=True,
            swa_tail_len=4,
            swa_tail_end=8,
            req=req,
        )

    def test_npu_tail_entry_owns_real_lengths_and_watermarks(self) -> None:
        """The NPU tail entry owns allocation lengths and request watermarks."""
        tail_locations = torch.arange(8, 16, dtype=torch.int64)
        allocator = SimpleNamespace(
            alloc_extend_swa_tail=Mock(return_value=tail_locations),
        )
        req = _make_req()

        result = allocator_npu.alloc_for_decode_prealloc_npu(
            allocator,
            prefix_indices=None,
            fill_len=8,
            prefix_len=0,
            total_prefix_len=0,
            delta_len=8,
            uses_swa_tail=True,
            swa_tail_len=6,
            swa_tail_end=8,
            req=req,
        )

        allocator.alloc_extend_swa_tail.assert_called_once_with(
            extend_num_tokens=8,
            swa_tail_len=6,
            swa_tail_end=8,
        )
        self.assertIs(result, tail_locations)
        self.assertEqual(req.kv.kv_allocated_len, 8)
        self.assertEqual(req.kv.swa_evicted_seqlen, 2)

    def test_npu_non_tail_entry_uses_prefix_endpoint_authority(self) -> None:
        """The NPU non-tail entry anchors allocation at the supplied prefix end."""
        allocated_locations = torch.arange(8, 12, dtype=torch.int64)
        allocator = SimpleNamespace(
            device="cpu",
            alloc_extend=Mock(return_value=allocated_locations),
        )
        req = _make_req()
        prefix_indices = torch.arange(4, 8, dtype=torch.int64)

        result = allocator_npu.alloc_for_decode_prealloc_npu(
            allocator,
            prefix_indices=prefix_indices,
            fill_len=8,
            prefix_len=4,
            total_prefix_len=4,
            delta_len=4,
            uses_swa_tail=False,
            swa_tail_len=0,
            swa_tail_end=8,
            req=req,
        )

        allocator.alloc_extend.assert_called_once()
        call_kwargs = allocator.alloc_extend.call_args.kwargs
        self.assertTrue(
            torch.equal(call_kwargs["prefix_lens"], torch.tensor([4]))
        )
        self.assertTrue(
            torch.equal(call_kwargs["prefix_lens_cpu"], torch.tensor([4]))
        )
        self.assertTrue(torch.equal(call_kwargs["seq_lens"], torch.tensor([8])))
        self.assertTrue(
            torch.equal(call_kwargs["seq_lens_cpu"], torch.tensor([8]))
        )
        self.assertTrue(torch.equal(call_kwargs["last_loc"], prefix_indices[-1:]))
        self.assertEqual(call_kwargs["extend_num_tokens"], 4)
        self.assertIs(result, allocated_locations)
        self.assertEqual(req.kv.kv_allocated_len, 8)

    def test_npu_non_tail_entry_uses_fresh_prefix_sentinel(self) -> None:
        """The NPU non-tail entry owns the fresh-prefix minus-one sentinel."""
        allocator = SimpleNamespace(
            device="cpu",
            alloc_extend=Mock(return_value=torch.arange(8, dtype=torch.int64)),
        )
        req = _make_req()

        allocator_npu.alloc_for_decode_prealloc_npu(
            allocator,
            prefix_indices=None,
            fill_len=8,
            prefix_len=0,
            total_prefix_len=0,
            delta_len=8,
            uses_swa_tail=False,
            swa_tail_len=0,
            swa_tail_end=8,
            req=req,
        )

        call_kwargs = allocator.alloc_extend.call_args.kwargs
        self.assertTrue(
            torch.equal(call_kwargs["last_loc"], torch.tensor([-1]))
        )
        self.assertEqual(call_kwargs["extend_num_tokens"], 8)
        self.assertEqual(req.kv.kv_allocated_len, 8)


if __name__ == "__main__":
    unittest.main()
