"""CPU regressions for HiSparse decode capacity and terminal PD retraction."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import schedule_batch as sb
from sglang.srt.mem_cache import common
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestHiSparseRetraction(unittest.TestCase):
    def allocator(self, logical=1024, physical=4096):
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
        r.input_embeds = None
        r.positional_embed_overrides = None
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

    def test_all_configured_backups_abort_and_complete_cleanup(self):
        for backend in ["cpu_tensor", "host_pool", "none"]:
            with self.subTest(backend=backend):
                a = self.allocator()
                b = self.batch(a)
                victim = b.reqs[-1]
                with (
                    self.contexts(backend=backend),
                    patch.object(
                        a, "get_cpu_copy", side_effect=NotImplementedError
                    ) as copy,
                    patch.object(sb, "release_kv_cache") as free,
                ):
                    resumed, _, aborted = b.retract_decode()
                self.assertEqual(len(resumed), 0)
                self.assertEqual(aborted, [victim])
                self.assertEqual(
                    victim.to_finish.status_code, HTTPStatus.SERVICE_UNAVAILABLE
                )
                copy.assert_not_called()
                b.hisparse_coordinator.retract_req.assert_called_once_with(victim)
                free.assert_called_once_with(victim, b.tree_cache, checkpoint=False)
                victim.reset_for_retract.assert_called_once()
                self.assertEqual(len(b.reqs), 1)

    def test_genuine_shortfall_aborts_until_remaining_batch_fits(self):
        a = self.allocator(logical=0, physical=4096)
        b = self.batch(a, count=3)

        def free(req, *args, **kwargs):
            a.logical_attn_allocator.available_size.return_value += 64

        with self.contexts(), patch.object(sb, "release_kv_cache", side_effect=free):
            resumed, _, aborted = b.retract_decode()
            self.assertTrue(b.check_decode_mem())
        self.assertEqual(len(resumed), 0)
        self.assertEqual(len(aborted), 2)
        self.assertEqual(len(b.reqs), 1)
        self.assertTrue(
            all(
                r.to_finish.status_code == HTTPStatus.SERVICE_UNAVAILABLE
                for r in aborted
            )
        )

    def test_last_request_shortfall_is_503_and_freed_once(self):
        b = self.batch(self.allocator(logical=0), count=1)
        victim = b.reqs[0]
        with self.contexts(), patch.object(sb, "release_kv_cache") as free:
            resumed, _, aborted = b.retract_decode()
        self.assertEqual(len(resumed), 0)
        self.assertEqual(aborted, [victim])
        self.assertEqual(victim.to_finish.status_code, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(b.reqs, [])
        free.assert_called_once()
        b.hisparse_coordinator.retract_req.assert_called_once_with(victim)

    def test_rotated_workspace_and_host_slots_are_freed_without_double_free(self):
        from sglang.srt.managers.hisparse_coordinator import (
            HiSparseCoordinator,
            HiSparseSpecSwapManager,
        )

        a = self.allocator()
        a.full_to_hisparse_device_index_mapping = torch.zeros(1088, dtype=torch.int64)
        a._kvcache._translate_loc_to_hisparse_device = lambda x: (
            a.full_to_hisparse_device_index_mapping[x]
        )
        a.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            1024, 64, torch.float32, "cpu", a._kvcache, False
        )
        a.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            512, 64, torch.float32, "cpu", a._kvcache, False
        )
        host = PagedTokenToKVPoolAllocator(
            1024, 64, torch.float32, "cpu", a._kvcache, False
        )
        for _ in range(10):
            logical = a.logical_attn_allocator.alloc(192)
            physical = a.hisparse_attn_allocator.alloc(192)
            host_locs = host.alloc(192)
            a.full_to_hisparse_device_index_mapping[logical] = physical
            req = self.req()
            req.hisparse_staging = False
            req.kv.kv_allocated_len = 192  # Includes uncommitted spec reserve.
            c = object.__new__(HiSparseCoordinator)
            c.decode_producer_stream = None
            c.wait_for_pending_backup = Mock()
            c.token_to_kv_pool_allocator = a
            c.req_to_token_pool = NS(req_to_token=logical.reshape(1, -1))
            c.mem_pool_device = NS(
                translate_loc_from_full_to_compressed=lambda x: x,
                full_to_hisparse_device_index_mapping=a.full_to_hisparse_device_index_mapping,
            )
            c.req_device_buffer_size = torch.tensor([128])
            # Different layers rotate different physical pages into scratch.
            c.req_device_buffer_token_locs = (
                torch.stack([physical[:128], physical[64:]]).reshape(2, 1, 128).clone()
            )
            c.req_device_buffer_tokens = torch.zeros((2, 1, 128), dtype=torch.int64)
            c.req_to_device_buffer = torch.zeros((1, 128), dtype=torch.int64)
            c.mem_pool_host = NS(
                allocated_host_indices=lambda *args: host_locs, free=host.free
            )
            c.req_to_host_pool = host_locs.reshape(1, -1).clone()
            c.req_to_host_pool_allocated_len = torch.tensor([192])
            c.lru_slots = torch.zeros((2, 1, 128), dtype=torch.int64)
            c._lru_init = torch.arange(128)
            c._skip_first_backup = [True]
            swap = object.__new__(HiSparseSpecSwapManager)
            swap.enabled = True
            swap._scratch_reqs = {0}
            swap.req_to_scratch = (
                torch.stack([physical[128:], physical[:64]]).reshape(2, 1, 64).clone()
            )
            c.spec_swap = swap
            c.retract_req(req)
            self.assertEqual(a.hisparse_attn_allocator.available_size(), 512)
            self.assertEqual(host.available_size(), 1024)
            self.assertTrue(
                torch.all(a.full_to_hisparse_device_index_mapping[logical] == 0)
            )
            self.assertEqual(swap._scratch_reqs, set())
            # Same allocator.free used by release_kv_cache; stale physical
            # mappings here would double-free pages returned above.
            a.free(logical)
            self.assertEqual(a.logical_attn_allocator.available_size(), 1024)
            self.assertEqual(a.hisparse_attn_allocator.available_size(), 512)

    def test_tokenizer_nonstream_503_and_stream_cleanup(self):
        import asyncio

        import fastapi

        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        for stream in (False, True):
            tm = object.__new__(TokenizerManager)
            state = NS(obj=NS(rid="retract"), lora_released=False)
            tm.rid_to_state = {"retract": state}
            tm.enable_lora = False
            out = {
                "meta_info": {
                    "finish_reason": sb.FINISH_ABORT(
                        "Retry later", status_code=HTTPStatus.SERVICE_UNAVAILABLE
                    ).to_json()
                }
            }
            if stream:
                self.assertIs(
                    asyncio.run(tm._handle_abort_finish_reason(out, state, stream)), out
                )
            else:
                with self.assertRaises(fastapi.HTTPException) as error:
                    asyncio.run(tm._handle_abort_finish_reason(out, state, stream))
                self.assertEqual(error.exception.status_code, 503)
            self.assertEqual(tm.rid_to_state, {})

    def test_normal_decode_still_backs_up_and_requeues(self):
        a = Mock(spec=BaseTokenToKVPoolAllocator)
        b = self.batch(a)
        b.hisparse_coordinator = None
        b.check_decode_mem = Mock(return_value=True)
        with (
            self.contexts(),
            patch.object(sb, "backup_kv_cache", return_value=True) as backup,
        ):
            resumed, _, aborted = b.retract_decode()
        self.assertEqual(len(resumed), 1)
        self.assertEqual(aborted, [])
        self.assertEqual(backup.call_args.args[-1], "cpu_tensor")

    def test_non_pd_retraction_can_recompute(self):
        b = self.batch(self.allocator())
        with self.contexts(mode="null"), patch.object(sb, "backup_kv_cache") as backup:
            self.assertTrue(b.release_req(0, 1))
        backup.assert_not_called()

    def test_explicit_no_offload_remains_rebootstrap_compatible(self):
        b = self.batch(self.allocator())
        with self.contexts(), patch.object(sb, "backup_kv_cache") as backup:
            self.assertTrue(b.release_req(0, 1, offload_kv=False))
        backup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
