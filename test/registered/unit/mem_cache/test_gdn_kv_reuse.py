"""Ownership and fallback checks for borrowing Full KV during GDN replay."""

import unittest
from array import array
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache import gdn_kv_reuse as reuse
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGdnKVReuse(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        values = {
            "attention_backends": lambda: ("flashinfer", "flashinfer"),
            "get_disagg": lambda: NS(disaggregation_mode="null"),
            "get_exec": lambda: NS(graph=NS(cuda_graph_backend_prefill="disabled")),
            "get_lora": lambda: NS(enable_lora=False),
            "get_memory": lambda: NS(
                enable_hierarchical_cache=False, enable_session_radix_cache=False
            ),
            "get_parallel": lambda: NS(
                tp_size=1, pp_size=1, attn_cp_size=1, attn_dp_size=1, dcp_enabled=False
            ),
            "get_schedule": lambda: NS(enable_mixed_chunk=False),
        }
        self.context.enter_context(patch.multiple(reuse, **values))
        self.context.enter_context(
            patch.object(
                reuse.envs.SGLANG_ENABLE_GDN_KV_REUSE, "get", return_value=True
            )
        )
        self.cache = Mock(spec=UnifiedRadixCache)
        self.cache.disable = False
        self.cache.supports_mamba.return_value = True
        self.cache.components = {0: None, 1: None}
        self.cache.enable_mamba_extra_buffer = True
        self.cache.page_size = 256
        self.cache.tree_core = Mock()
        self.cache.tree_core.match_full_device_prefix.return_value = (8192, 17, 12288)
        self.cache.tree_core.match_full_device_prefix.side_effect = lambda key: (
            (4096, 17, 4096)
            if key.cache_salt == "uncached"
            else self.cache.tree_core.match_full_device_prefix.return_value
        )
        self.indices = torch.arange(12288, dtype=torch.int64) + 256
        self.cache.path_device_indices.return_value = self.indices
        self.req = NS(
            prefix_len=4096,
            extend_end=8192,
            kv=ReqKvInfo(cache_protected_len=4096),
            full_kv_replay_node=None,
            skip_radix_cache_insert=False,
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            session_id=None,
            mamba_branching_seqlen=8192,
            full_untruncated_fill_ids=array("q", range(8193)),
            extra_key=None,
            cache_salt="test-namespace",
            rid="test",
            lora_id=None,
        )
        self.batch = NS(
            device="cpu",
            tree_cache=self.cache,
            reqs=[self.req],
            enable_overlap=False,
            model_config=NS(
                hf_text_config=NS(model_type="qwen3_5_text", attn_output_gate=True),
                quantization=None,
                dtype=torch.bfloat16,
            ),
            spec_algorithm=NS(is_none=lambda: True),
            full_kv_replay=False,
            full_kv_fresh_token_indices=None,
        )

    def assert_fallback(self):
        self.assertIsNone(reuse.try_borrow_full_kv(self.batch))
        self.assertEqual(self.req.kv.cache_protected_len, 4096)
        self.assertIsNone(self.req.full_kv_replay_node)
        self.assertFalse(self.batch.full_kv_replay)
        self.cache.tree_core.inc_full_pin.assert_not_called()

    def test_borrows_only_current_chunk_and_preserves_compute_prefix(self):
        out = reuse.try_borrow_full_kv(self.batch)[0]
        self.assertTrue(torch.equal(out, self.indices[4096:8192]))
        self.assertEqual(
            out.untyped_storage().data_ptr(), self.indices.untyped_storage().data_ptr()
        )
        self.assertEqual(self.req.prefix_len, 4096)
        self.assertEqual(self.req.kv.cache_protected_len, 8192)
        self.assertTrue(self.batch.full_kv_replay)
        self.cache.tree_core.inc_full_pin.assert_called_once_with(17)
        key = self.cache.tree_core.match_full_device_prefix.call_args.args[0]
        self.assertEqual(len(key), 8192)
        self.assertEqual(key.cache_salt, "test-namespace")
        reuse.release_full_kv_pin(self.cache, self.req)
        reuse.release_full_kv_pin(self.cache, self.req)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)

    def test_disabled_flag_falls_back(self):
        with patch.object(
            reuse.envs.SGLANG_ENABLE_GDN_KV_REUSE, "get", return_value=False
        ):
            self.assert_fallback()

    def test_release_without_checkpoint_cleans_up_and_unpins(self):
        reuse.try_borrow_full_kv(self.batch)
        component = Mock()
        self.cache._components_tuple = (component,)
        UnifiedRadixCache.on_release(self.cache, self.req, checkpointed=False)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)
        component.cleanup_after_caching_req.assert_called_once_with(
            self.req, is_finished=True
        )

    def test_release_after_checkpoint_still_unpins(self):
        reuse.try_borrow_full_kv(self.batch)
        UnifiedRadixCache.on_release(self.cache, self.req, checkpointed=True)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)

    def test_experimental_olmo_adapter_can_borrow(self):
        config = self.batch.model_config.hf_text_config
        config.model_type = "olmo_hybrid"
        config.attn_output_gate = False
        config.sglang_olmo_hybrid_adapter = True
        self.assertIsNotNone(reuse.try_borrow_full_kv(self.batch))
        reuse.release_full_kv_pin(self.cache, self.req)

    def test_unregistered_olmo_falls_back(self):
        self.batch.model_config.hf_text_config.model_type = "olmo_hybrid"
        self.assert_fallback()

    def test_partial_device_hit_falls_back(self):
        self.cache.tree_core.match_full_device_prefix.return_value = (7936, 17, 8192)
        self.assert_fallback()

    def test_mixed_new_tokens_fall_back(self):
        self.req.extend_end = 8448
        self.req.mamba_branching_seqlen = 8448
        self.assert_fallback()

    def test_eight_requests_share_cached_slots_with_independent_pins(self):
        self.batch.reqs = [deepcopy(self.req) for _ in range(8)]
        borrowed = reuse.try_borrow_full_kv(self.batch)
        self.assertEqual(len(borrowed), 8)
        for req, indices in zip(self.batch.reqs, borrowed):
            self.assertTrue(torch.equal(indices, self.indices[4096:8192]))
            self.assertEqual(req.prefix_len, 4096)
            self.assertEqual(req.kv.cache_protected_len, 8192)
            reuse.release_full_kv_pin(self.cache, req)
            reuse.release_full_kv_pin(self.cache, req)
        self.assertEqual(self.cache.tree_core.inc_full_pin.call_count, 8)
        self.assertEqual(self.cache.tree_core.dec_full_pin.call_count, 8)
        self.assertIsNone(self.batch.full_kv_fresh_token_indices)

    def test_mixed_batch_selects_only_fresh_tokens_in_request_order(self):
        fresh = deepcopy(self.req)
        fresh.extend_end = 4352
        fresh.mamba_branching_seqlen = 4352
        fresh.cache_salt = "uncached"
        other = deepcopy(self.req)
        self.batch.reqs = [fresh, self.req, other]
        borrowed = reuse.try_borrow_full_kv(self.batch)
        self.assertIsNone(borrowed[0])
        self.assertEqual(fresh.kv.cache_protected_len, 4096)
        self.assertIsNone(fresh.full_kv_replay_node)
        self.assertTrue(
            torch.equal(self.batch.full_kv_fresh_token_indices, torch.arange(256))
        )
        self.assertTrue(self.batch.full_kv_replay)
        self.assertEqual(self.cache.tree_core.inc_full_pin.call_count, 2)

    def test_failed_later_request_rolls_back_earlier_borrow(self):
        second = deepcopy(self.req)
        self.batch.reqs.append(second)
        self.cache.path_device_indices.side_effect = [
            self.indices,
            RuntimeError("failed collection"),
        ]
        with self.assertRaisesRegex(RuntimeError, "failed collection"):
            reuse.try_borrow_full_kv(self.batch)
        for req in self.batch.reqs:
            self.assertEqual(req.kv.cache_protected_len, 4096)
            self.assertIsNone(req.full_kv_replay_node)
        self.assertEqual(self.cache.tree_core.inc_full_pin.call_count, 2)
        self.assertEqual(self.cache.tree_core.dec_full_pin.call_count, 2)
        self.assertFalse(self.batch.full_kv_replay)

    def test_mixed_allocation_only_allocates_fresh_request_slots(self):
        fresh = deepcopy(self.req)
        fresh.extend_end = 4608
        fresh.cache_salt = "uncached"
        self.batch.reqs = [self.req, fresh]
        self.batch.seq_lens_cpu = torch.tensor([8192, 4608])
        self.batch.req_to_token_pool = NS(req_to_token=torch.empty((2, 8192)))
        borrowed = reuse.try_borrow_full_kv(self.batch)
        fresh_slots = torch.arange(20000, 20512)
        with patch.object(
            allocation, "alloc_paged_token_slots_extend", return_value=fresh_slots
        ) as allocate:
            loc = allocation._alloc_extend_loc_with_kv_reuse(
                self.batch,
                [True, False],
                [self.indices[:4096], self.indices[:4096]],
                torch.tensor([0, 1]),
                torch.tensor([4096, 4096]),
                torch.tensor([4096, 512]),
                torch.tensor([0, 1]),
                256,
                borrowed_kv=borrowed,
            )
        self.assertTrue(torch.equal(loc[:4096], self.indices[4096:8192]))
        self.assertTrue(torch.equal(loc[4096:], fresh_slots))
        self.assertEqual(allocate.call_args.kwargs["extend_num_tokens"], 512)
        self.assertEqual(
            allocate.call_args.kwargs["seq_lens_cpu"].tolist(), [4096, 4608]
        )
        self.assertEqual(self.req.kv.cache_protected_len, 8192)
        self.assertEqual(fresh.kv.cache_protected_len, 4096)

    def _check_allocation_failure(self, stage):
        fresh = deepcopy(self.req)
        fresh.extend_end = 4352
        fresh.cache_salt = "uncached"
        self.batch.reqs.append(fresh)
        self.batch.maybe_evict_swa = Mock()
        self.batch.is_dllm = lambda: False
        self.batch.hisparse_coordinator = None
        self.batch.token_to_kv_pool_allocator = NS(state=None)
        self.batch.prefix_lens = [4096, 4096]
        self.batch.extend_lens = [4096, 256]
        self.batch.seq_lens = self.batch.seq_lens_cpu = torch.tensor([8192, 4352])
        self.batch.req_to_token_pool = Mock()
        fresh_slots = torch.arange(20000, 20256)
        all_slots = torch.cat([self.indices[4096:8192], fresh_slots])
        storage = Mock()
        error = RuntimeError("allocation failed")
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(allocation, "kv_storage_of", return_value=storage)
            )
            stack.enter_context(
                patch.object(
                    allocation, "_prefix_kv_indices", return_value=self.indices[:4096]
                )
            )
            stack.enter_context(
                patch.object(allocation, "alloc_req_slots", return_value=[0, 1])
            )
            stack.enter_context(
                patch.object(allocation, "_alloc_page_size", return_value=256)
            )
            stack.enter_context(
                patch.object(allocation, "is_cp_sharded_padded", return_value=False)
            )
            allocate = stack.enter_context(
                patch.object(
                    allocation,
                    "_alloc_extend_loc_with_kv_reuse",
                    return_value=all_slots,
                )
            )
            write = stack.enter_context(patch.object(allocation, "write_cache_indices"))
            if stage == "slots":
                allocate.side_effect = error
            elif stage == "write":
                write.side_effect = error
            else:
                self.batch.req_to_token_pool.alloc_aux_to_lengths.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                allocation.alloc_for_extend(self.batch)
        if stage == "slots":
            storage.free_tokens.assert_not_called()
        else:
            storage.free_tokens.assert_called_once()
            self.assertTrue(
                torch.equal(storage.free_tokens.call_args.args[0], fresh_slots)
            )
        for req in self.batch.reqs:
            self.assertEqual(req.kv.cache_protected_len, 4096)
            self.assertIsNone(req.full_kv_replay_node)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)
        self.assertFalse(self.batch.full_kv_replay)
        self.assertIsNone(self.batch.full_kv_fresh_token_indices)

    def test_fresh_allocation_failure_releases_borrowed_pins(self):
        self._check_allocation_failure("slots")

    def test_request_table_failure_frees_only_fresh_slots(self):
        self._check_allocation_failure("write")

    def test_auxiliary_allocation_failure_frees_only_fresh_slots(self):
        self.batch.enable_overlap = True
        self._check_allocation_failure("aux")

    def test_unaligned_chunk_falls_back(self):
        self.req.extend_end = 8191
        self.assert_fallback()

    def test_unpaged_cache_does_not_require_state_chunk_alignment(self):
        self.cache.page_size = 1
        self.cache.enable_mamba_extra_buffer = False
        self.req.extend_end = 8191
        out = reuse.try_borrow_full_kv(self.batch)[0]
        self.assertTrue(torch.equal(out, self.indices[4096:8191]))
        self.assertEqual(self.req.kv.cache_protected_len, 8191)

    def test_quantized_model_falls_back(self):
        self.batch.model_config.quantization = "modelopt_fp4"
        self.assert_fallback()

    def test_overlap_mixed_batch_retains_pin_while_forward_is_pending(self):
        self.batch.enable_overlap = True
        fresh = deepcopy(self.req)
        fresh.extend_end = 4352
        fresh.cache_salt = "uncached"
        self.batch.reqs = [self.req, fresh]
        borrowed = reuse.try_borrow_full_kv(self.batch)
        self.assertIsNotNone(borrowed[0])
        self.assertIsNone(borrowed[1])
        self.assertTrue(
            torch.equal(
                self.batch.full_kv_fresh_token_indices, torch.arange(4096, 4352)
            )
        )
        # A later scheduling pass cannot borrow again while the earlier forward
        # still owns the temporary pin, nor release that pin on fallback.
        self.assertIsNone(reuse._try_borrow_request(self.cache, self.req))
        self.cache.tree_core.dec_full_pin.assert_not_called()
        reuse.release_full_kv_pin(self.cache, self.req)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)

    def test_overlap_snapshot_keeps_previous_fresh_indices_alive(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.managers.scheduler import Scheduler

        fresh = torch.arange(256)
        batch = ScheduleBatch(
            reqs=[],
            enable_overlap=True,
            full_kv_replay=True,
            full_kv_fresh_token_indices=fresh,
        )
        scheduler = NS(
            batch_record_ct=0, batch_record_buf=[None, None], uocr_runtime=None
        )
        Scheduler.record_batch_in_overlap(scheduler, batch)
        batch.full_kv_replay = False
        batch.full_kv_fresh_token_indices = None
        saved = scheduler.batch_record_buf[1][1]
        self.assertTrue(any(value is fresh for value in saved))

    def test_interior_gdn_checkpoint_borrows_entire_kv_chunk(self):
        self.req.full_untruncated_fill_ids = self.req.full_untruncated_fill_ids[:8192]
        self.req.mamba_branching_seqlen = 7936
        out = reuse.try_borrow_full_kv(self.batch)[0]
        self.assertTrue(torch.equal(out, self.indices[4096:8192]))
        self.assertEqual(self.req.kv.cache_protected_len, 8192)
        reuse.release_full_kv_pin(self.cache, self.req, covered_prefix_len=7936)
        self.cache.tree_core.dec_full_pin.assert_not_called()
        self.assertEqual(self.req.full_kv_replay_node, 17)
        reuse.release_full_kv_pin(self.cache, self.req, covered_prefix_len=8192)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)

    def test_interior_checkpoint_pin_is_released_on_teardown(self):
        reuse.try_borrow_full_kv(self.batch)
        reuse.release_full_kv_pin(self.cache, self.req, covered_prefix_len=7936)
        UnifiedRadixCache.on_release(self.cache, self.req, checkpointed=True)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)

    def test_zero_gdn_prefix_can_replay_cached_kv(self):
        self.req.prefix_len = 0
        self.req.extend_end = 4096
        self.req.kv.protect_prefix(0)
        out = reuse.try_borrow_full_kv(self.batch)[0]
        self.assertTrue(torch.equal(out, self.indices[:4096]))

    def test_fp16_and_tensor_parallel_can_borrow_rank_local_slots(self):
        self.batch.model_config.dtype = torch.float16
        with patch.object(
            reuse,
            "get_parallel",
            return_value=NS(
                tp_size=2, pp_size=1, attn_cp_size=1, attn_dp_size=1, dcp_enabled=False
            ),
        ):
            out = reuse.try_borrow_full_kv(self.batch)[0]
        self.assertTrue(torch.equal(out, self.indices[4096:8192]))

    def test_pipeline_parallel_can_borrow_stage_local_slots(self):
        with patch.object(
            reuse,
            "get_parallel",
            return_value=NS(
                tp_size=1,
                pp_size=2,
                attn_cp_size=1,
                attn_dp_size=1,
                dcp_enabled=False,
            ),
        ):
            self.assertIsNotNone(reuse.try_borrow_full_kv(self.batch))

    def test_dcp_layout_requires_index_remapping(self):
        with patch.object(
            reuse,
            "get_parallel",
            return_value=NS(
                tp_size=2,
                pp_size=1,
                attn_cp_size=1,
                attn_dp_size=1,
                dcp_enabled=True,
            ),
        ):
            self.assert_fallback()

    def test_skip_insert_still_allows_cached_kv_reads(self):
        self.req.skip_radix_cache_insert = True
        self.assertIsNotNone(reuse.try_borrow_full_kv(self.batch))
        self.cache._components_tuple = ()
        UnifiedRadixCache.on_release(self.cache, self.req, checkpointed=False)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)

    def test_lora_enabled_without_active_adapters_can_borrow(self):
        with patch.object(reuse, "get_lora", return_value=NS(enable_lora=True)):
            self.assertIsNotNone(reuse.try_borrow_full_kv(self.batch))

    def test_active_lora_in_any_request_falls_back_for_whole_batch(self):
        other = deepcopy(self.req)
        other.lora_id = "adapter"
        self.batch.reqs.append(other)
        with patch.object(reuse, "get_lora", return_value=NS(enable_lora=True)):
            self.assert_fallback()

    def test_failed_path_collection_releases_pin(self):
        self.cache.path_device_indices.side_effect = RuntimeError("failed collection")
        with self.assertRaisesRegex(RuntimeError, "failed collection"):
            reuse.try_borrow_full_kv(self.batch)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)
        self.assertEqual(self.req.kv.cache_protected_len, 4096)
        self.assertIsNone(self.req.full_kv_replay_node)


if __name__ == "__main__":
    unittest.main()
