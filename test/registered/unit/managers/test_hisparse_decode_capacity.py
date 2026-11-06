"""CPU regressions for logical-only speculative decode capacity."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import schedule_batch as sb
from sglang.srt.mem_cache import common
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestHiSparseDecodeCapacity(unittest.TestCase):
    def allocator(self, logical=1024, physical=0):
        a = object.__new__(HiSparseTokenToKVPoolAllocator)
        a.page_size = 64
        a.configure_spec_scratch(4096, device_buffer_size=4096)
        a._size_full = 4096
        a._size_hisparse = 512
        a._kvcache = NS(cpu_copy_carries_mamba=True)
        a.logical_attn_allocator = Mock()
        a.logical_attn_allocator.available_size.return_value = logical
        a.hisparse_attn_allocator = Mock()
        a.hisparse_attn_allocator.available_size.return_value = physical
        return a

    def req(self, row=0):
        r = object.__new__(sb.Req)
        r.kv = sb.ReqKvInfo()
        r.kv.req_pool_idx = row
        r.kv.kv_committed_len = 64
        r.kv.kv_allocated_len = 64
        r.origin_input_ids = [1] * 64
        r.output_ids = [2]
        r.rid = f"audit-{row}"
        r.beam_group = None
        r.multimodal_inputs = None
        r.finished = Mock(return_value=False)
        r.reset_for_retract = Mock()
        return r

    def batch(self, a, count=2):
        b = object.__new__(sb.ScheduleBatch)
        b.reqs = [self.req(i) for i in range(count)]
        b.token_to_kv_pool_allocator = a
        b.req_to_token_pool = NS(
            req_to_token=torch.arange(count * 192).reshape(count, 192)
        )
        b.tree_cache = Mock()
        b.tree_cache.token_to_kv_pool_allocator = a
        b.hisparse_coordinator = Mock()
        b.spec_algorithm = NS(is_none=lambda: False)
        b._get_decode_retraction_order = lambda reqs: list(range(len(reqs)))
        b.filter_batch = lambda keep_indices: setattr(
            b, "reqs", [b.reqs[i] for i in keep_indices]
        )
        return b

    def contexts(self, backend="cpu_tensor", mode="decode"):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(
            patch.object(
                sb,
                "get_disagg",
                return_value=NS(
                    disaggregation_mode=mode,
                    disaggregation_decode_retraction_backup=backend,
                ),
            )
        )
        stack.enter_context(
            patch.object(sb, "get_alloc_reserve_per_decode", return_value=8)
        )
        stack.enter_context(
            patch.object(common, "dsv4_needs_rebootstrap", return_value=False)
        )
        stack.enter_context(
            patch.object(
                sb, "beam_retraction_order", side_effect=lambda indices, reqs: indices
            )
        )
        stack.enter_context(patch.object(sb, "release_kv_cache"))
        stack.enter_context(patch.object(sb, "evict_from_tree_cache"))
        stack.enter_context(
            patch.object(
                sb.NewTokenRatioTracker,
                "estimate_new_token_ratio_after_retract",
                return_value=0.5,
            )
        )
        return stack

    def test_physical_exhaustion_does_not_retract_spec_decode(self):
        a = self.allocator(logical=1024, physical=0)
        b = self.batch(a)
        with self.contexts():
            self.assertTrue(b.check_decode_mem())
        self.assertEqual(a.available_size(), 0)  # Admission remains physical-limited.
        a.logical_attn_allocator.evict_to_free_tokens.assert_called_once()

    def test_pending_transfer_workspace_does_not_block_running_batch(self):
        a = self.allocator(logical=1189888, physical=64)
        b = self.batch(a, count=8)
        with self.contexts():
            self.assertEqual(b.new_tokens_required_next_decode(), 512)
            self.assertTrue(b.check_decode_mem())

    def test_logical_shortfall_is_still_rejected(self):
        a = self.allocator(logical=63, physical=4096)
        with self.contexts():
            self.assertFalse(self.batch(a, count=1).check_decode_mem())

    def test_reclaimable_logical_pages_are_evicted_before_retraction(self):
        a = self.allocator(logical=0, physical=0)

        def evict(tree_cache, num_tokens):
            a.logical_attn_allocator.available_size.return_value = num_tokens

        a.logical_attn_allocator.evict_to_free_tokens.side_effect = evict
        with self.contexts():
            self.assertTrue(self.batch(a).check_decode_mem())

    def test_exact_logical_boundary(self):
        a = self.allocator(logical=128, physical=0)
        with self.contexts():
            self.assertTrue(self.batch(a).check_decode_mem())
        a.logical_attn_allocator.available_size.return_value = 127
        with self.contexts():
            self.assertFalse(self.batch(a).check_decode_mem())

    def test_non_spec_page_one_and_unconfigured_keep_physical_gate(self):
        for page, scratch, spec in [(1, 4096, True), (64, 0, True), (64, 4096, False)]:
            with self.subTest(page=page, scratch=scratch, spec=spec):
                a = self.allocator()
                a.page_size = page
                a._spec_scratch_capacity = scratch
                a.evict_to_free_tokens = Mock()
                self.assertFalse(
                    a.check_decode_capacity(
                        num_tokens=64,
                        tree_cache=Mock(),
                        spec_algorithm=NS(is_none=lambda: not spec),
                    )
                )

    def test_real_logical_allocation_with_no_free_physical_pages(self):
        a = self.allocator()
        a.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            1024, 64, torch.float32, "cpu", a._kvcache, False
        )
        a.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            512, 64, torch.float32, "cpu", a._kvcache, False
        )
        a.hisparse_attn_allocator.alloc(512)
        tree = Mock()
        tree.supports_prefix_sharing.return_value = False
        self.assertTrue(
            a.check_decode_capacity(
                num_tokens=64, tree_cache=tree, spec_algorithm=NS(is_none=lambda: False)
            )
        )
        self.assertIsNotNone(a.logical_attn_allocator.alloc(64))
        self.assertEqual(a.available_size(), 0)
        self.assertEqual(a.logical_attn_allocator.available_size(), 960)


if __name__ == "__main__":
    unittest.main()
