import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    IncLockRefResult,
    InsertParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_draft_content_allocator import (
    DFlashDraftContentAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CheckpointBoundaryComponent(TreeComponent):
    """Test oracle for Mamba's boundary-only checkpoint semantics."""

    component_type = ComponentType.MAMBA

    def create_match_validator(self, match_device_only=False):
        return lambda node: node.component_data[self.component_type].value is not None

    def commit_insert_component_data(
        self, node, is_new_leaf, params, result, cache_actions
    ):
        if params.mamba_value is None:
            return
        cd = node.component_data[self.component_type]
        if cd.value is None:
            cd.value = params.mamba_value.clone()
            self.tree_core.lru_lists[self.component_type].insert_mru(node)
            self.tree_core.component_evictable_size_[self.component_type] += 1

    def redistribute_on_node_split(self, new_parent, child):
        # A Mamba checkpoint remains only at the original deeper boundary.
        new_parent.component_data[self.component_type].value = None

    def evict_component(self, node, device_frees, host_frees, target=EvictLayer.DEVICE):
        return 0, 0

    def acquire_component_lock(self, node, result: IncLockRefResult, lock_host=False):
        return result

    def release_component_lock(self, node, params, lock_host=False):
        return None

    def _evict_device_start(self, request_cnt):
        return None

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self):
        return None

    def _dec_session_coverage(self, session_id, leaf):
        return None

    def _advance_session_coverage(self, session_id, leaf, old_ancestor):
        return None

    def _recede_session_coverage(self, session_id, leaf, fallback):
        return None


class _FailingFinalizeComponent(_CheckpointBoundaryComponent):
    def finalize_match_result_in_cache(self, params, result):
        raise RuntimeError("injected Mamba finalizer failure")


class _FailingPrepareComponent(_CheckpointBoundaryComponent):
    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished):
        raise RuntimeError("injected Mamba prepare failure")


class _CachingCheckpointBoundaryComponent(_CheckpointBoundaryComponent):
    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished):
        insert_params.mamba_value = torch.tensor([1])
        return None


class _RejectingPlanRegistry(dict):
    def __setitem__(self, key, value):
        raise RuntimeError("injected DFlash plan registration failure")


class TestDFlashDraftComponent(CustomTestCase):
    def setUp(self):
        self.full_allocator = TokenToKVPoolAllocator(
            size=64,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.content_allocator = DFlashDraftContentAllocator(
            start=100, size=16, device="cpu"
        )
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4,
                    max_context_len=32,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=self.full_allocator,
                page_size=1,
                tree_components=(ComponentType.FULL, ComponentType.DRAFT),
                dflash_draft_content_allocator=self.content_allocator,
                dflash_draft_window_size=4,
            )
        )
        self.component = self.cache.components[ComponentType.DRAFT]

    @staticmethod
    def _key(token_ids):
        return RadixKey(array("q", token_ids))

    def _insert(self, token_ids, *, draft_start):
        full_rows = self.full_allocator.alloc(len(token_ids))
        draft_count = len(token_ids) - draft_start
        draft_rows = self.content_allocator.alloc(draft_count)
        self.assertIsNotNone(full_rows)
        self.assertIsNotNone(draft_rows)
        self.content_allocator.claim_lease(draft_rows)
        params = InsertParams(
            key=self._key(token_ids),
            value=full_rows,
            draft_value=draft_rows,
            draft_start_seqlen=draft_start,
            draft_processed=torch.zeros(draft_count, dtype=torch.bool),
        )
        result = self.cache.insert(params)
        self.component.cleanup_pending_rows(params)
        self.assertTrue(bool(torch.all(params.draft_processed)))
        return result, draft_rows.clone()

    def _match(self, token_ids, rid):
        return self.cache.match_prefix(
            MatchPrefixParams(
                key=self._key(token_ids),
                req=SimpleNamespace(rid=rid, session=None),
            )
        )

    def test_common_match_requires_one_contiguous_draft_window(self):
        _, draft_rows = self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        result = self._match([1, 2, 3, 4, 5, 6], "warm")
        self.assertEqual(result.device_indices.numel(), 6)
        plan = self.component.get_match_plan("warm")
        self.assertTrue(torch.equal(plan.source_rows, draft_rows))
        self.assertEqual(plan.matched_tokens, 4)
        self.assertTrue(self.cache.release_dflash_draft_match_pin("warm"))

        self.cache.reset()
        _, short_rows = self._insert([1, 2, 3], draft_start=0)
        result = self._match([1, 2, 3], "short-prefix")
        plan = self.component.get_match_plan("short-prefix")
        self.assertEqual(result.device_indices.numel(), 3)
        self.assertEqual(plan.matched_tokens, 3)
        self.assertTrue(torch.equal(plan.source_rows, short_rows))
        self.component.release_match_pin("short-prefix")

        self.cache.reset()
        self._insert([1, 2, 3, 4, 5, 6], draft_start=3)
        result = self._match([1, 2, 3, 4, 5, 6], "short")
        self.assertEqual(result.device_indices.numel(), 0)
        self.assertIsNone(self.component.get_match_plan("short"))

    def test_cap_history_preserves_an_earlier_long_prompt_branch_window(self):
        _, rows = self._insert(list(range(1, 11)), draft_start=0)

        result = self._match([1, 2, 3, 4, 5, 6, 99], "earlier-branch")
        self.assertEqual(result.device_indices.numel(), 6)
        plan = self.component.get_match_plan("earlier-branch")
        self.assertEqual(plan.matched_tokens, 4)
        torch.testing.assert_close(plan.source_rows, rows[2:6])
        self.component.release_match_pin("earlier-branch")

    def test_match_pin_survives_split_and_releases_exactly(self):
        self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        self._match([1, 2, 3, 4, 5, 6], "split")
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 4
        )

        self._insert([1, 2, 3, 4, 9, 10], draft_start=2)
        self.assertTrue(self.component.release_match_pin("split"))
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 0
        )
        self.assertEqual(
            self.cache.tree_core.component_evictable_size(ComponentType.DRAFT), 6
        )

    def test_match_pin_failure_is_atomic_before_and_after_locking(self):
        self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        draft_evictable = self.cache.tree_core.component_evictable_size(
            ComponentType.DRAFT
        )

        with patch.object(
            self.content_allocator,
            "assert_allocated",
            side_effect=RuntimeError("injected DFlash source validation failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "source validation failure"):
                self._match([1, 2, 3, 4, 5, 6], "validate-failure")

        self.assertIsNone(self.component.get_match_plan("validate-failure"))
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 0
        )
        self.assertEqual(
            self.cache.tree_core.component_evictable_size(ComponentType.DRAFT),
            draft_evictable,
        )

        original_registry = self.component._match_plans
        self.component._match_plans = _RejectingPlanRegistry()
        try:
            with self.assertRaisesRegex(RuntimeError, "plan registration failure"):
                self._match([1, 2, 3, 4, 5, 6], "register-failure")
        finally:
            self.component._match_plans = original_registry

        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 0
        )
        self.assertEqual(
            self.cache.tree_core.component_evictable_size(ComponentType.DRAFT),
            draft_evictable,
        )
        recovered = self._match([1, 2, 3, 4, 5, 6], "recovered")
        self.assertEqual(recovered.device_indices.numel(), 6)
        self.assertTrue(self.component.release_match_pin("recovered"))

    def test_restore_pin_requires_draft_validation(self):
        self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        with self.assertRaisesRegex(RuntimeError, "requires DRAFT coverage"):
            self.cache.match_prefix(
                MatchPrefixParams(
                    key=self._key([1, 2, 3, 4, 5, 6]),
                    req=SimpleNamespace(rid="invalid-flags", session=None),
                    require_dflash_draft=False,
                )
            )
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 0
        )

    def test_shared_pins_survive_two_splits_and_release_independently(self):
        self._insert([1, 2, 3, 4, 5, 6, 7, 8], draft_start=0)
        self._match([1, 2, 3, 4, 5, 6, 7, 8], "first-pin")
        self._match([1, 2, 3, 4, 5, 6, 7, 8], "second-pin")
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 8
        )

        self._insert([1, 2, 3, 4, 9, 10], draft_start=2)
        self._insert([1, 2, 11, 12], draft_start=0)
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 8
        )

        self.assertTrue(self.component.release_match_pin("first-pin"))
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 8
        )
        self.assertTrue(self.component.release_match_pin("second-pin"))
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 0
        )
        self.assertEqual(
            self.cache.tree_core.component_evictable_size(ComponentType.DRAFT), 12
        )

    def test_draft_eviction_never_deletes_full_tree_data(self):
        result, _ = self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        node = self.cache.tree_core.node_by_id(result.last_device_node)
        full_value = node.component_data[ComponentType.FULL].value.clone()

        self._match([1, 2, 3, 4, 5, 6], "pinned")
        blocked = self.cache.evict(EvictParams(draft_num_tokens=1))
        self.assertEqual(blocked.draft_num_tokens_evicted, 0)
        self.component.release_match_pin("pinned")

        evicted = self.cache.evict(EvictParams(draft_num_tokens=1))
        self.assertEqual(evicted.num_tokens_evicted, 0)
        self.assertEqual(evicted.draft_num_tokens_evicted, 4)
        self.assertTrue(
            torch.equal(node.component_data[ComponentType.FULL].value, full_value)
        )
        self.assertIsNone(node.component_data[ComponentType.DRAFT].value)

    def test_content_allocation_pressure_evicts_only_draft_rows(self):
        result, _ = self._insert(list(range(1, 17)), draft_start=0)
        node = self.cache.tree_core.node_by_id(result.last_device_node)
        full_value = node.component_data[ComponentType.FULL].value.clone()
        self.assertEqual(self.content_allocator.available_size(), 0)

        rows = self.cache.alloc_dflash_draft_content(4)

        self.assertIsNotNone(rows)
        self.assertEqual(len(rows), 4)
        self.assertEqual(self.content_allocator.available_size(), 12)
        self.assertTrue(
            torch.equal(node.component_data[ComponentType.FULL].value, full_value)
        )
        self.assertIsNone(node.component_data[ComponentType.DRAFT].value)
        self.cache.free_unstaged_dflash_draft_content(rows)
        self.assertEqual(self.content_allocator.available_size(), 16)

    def test_draft_eviction_preserves_full_and_mamba_components(self):
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4,
                    max_context_len=32,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=self.full_allocator,
                page_size=1,
                tree_components=(
                    ComponentType.FULL,
                    ComponentType.DRAFT,
                    ComponentType.MAMBA,
                ),
                component_registry_override={
                    ComponentType.MAMBA: _CheckpointBoundaryComponent
                },
                dflash_draft_content_allocator=self.content_allocator,
                dflash_draft_window_size=4,
            )
        )
        component = cache.components[ComponentType.DRAFT]
        draft_rows = self.content_allocator.alloc(4)
        self.content_allocator.claim_lease(draft_rows)
        params = InsertParams(
            key=self._key([1, 2, 3, 4, 5, 6]),
            value=self.full_allocator.alloc(6),
            mamba_value=torch.tensor([1]),
            draft_value=draft_rows,
            draft_start_seqlen=2,
            draft_processed=torch.zeros(4, dtype=torch.bool),
        )
        result = cache.insert(params)
        component.cleanup_pending_rows(params)
        node = cache.tree_core.node_by_id(result.last_device_node)

        evicted = cache.evict(EvictParams(draft_num_tokens=1))
        self.assertEqual(evicted.draft_num_tokens_evicted, 4)
        self.assertEqual(evicted.num_tokens_evicted, 0)
        self.assertIsNotNone(node.component_data[ComponentType.FULL].value)
        self.assertIsNotNone(node.component_data[ComponentType.MAMBA].value)
        self.assertIsNone(node.component_data[ComponentType.DRAFT].value)

    def test_publish_accepts_only_allocated_content_rows(self):
        with self.assertRaisesRegex(RuntimeError, "one-shot allocation lease"):
            self.component.stage_pending_publish(
                "request-or-scratch", 0, torch.tensor([99, 116])
            )
        with self.assertRaisesRegex(RuntimeError, "one-shot allocation lease"):
            self.component.stage_pending_publish("free-content", 0, torch.tensor([100]))

        rows = self.content_allocator.alloc(1)
        self.component.stage_pending_publish("canonical", 0, rows)
        with self.assertRaisesRegex(RuntimeError, "one-shot allocation lease"):
            self.component.stage_pending_publish("duplicate-owner", 0, rows)
        self.component.release_request_state("canonical")

        _, tree_rows = self._insert([1, 2, 3, 4], draft_start=0)
        with self.assertRaisesRegex(RuntimeError, "one-shot allocation lease"):
            self.component.stage_pending_publish("tree-owned", 0, tree_rows)
        self.cache.reset()
        self.assertEqual(self.content_allocator.available_size(), 16)

    def test_overlap_frees_duplicate_rows_and_keeps_branch_content(self):
        _, first_rows = self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        _, second_rows = self._insert([1, 2, 3, 4, 9, 10], draft_start=2)
        self.assertEqual(self.content_allocator.available_size(), 10)

        result = self._match([1, 2, 3, 4, 9, 10], "branch")
        plan = self.component.get_match_plan("branch")
        expected = torch.cat((first_rows[:2], second_rows[2:]))
        self.assertEqual(result.device_indices.numel(), 6)
        self.assertTrue(torch.equal(plan.source_rows, expected))
        self.component.release_match_pin("branch")

    def test_pending_abort_and_reset_return_all_content_rows(self):
        pending = self.content_allocator.alloc(4)
        params = InsertParams(
            draft_value=pending,
            draft_start_seqlen=8,
            draft_processed=torch.zeros(4, dtype=torch.bool),
        )
        self.component.cleanup_pending_rows(params)
        self.assertEqual(self.content_allocator.available_size(), 16)

        self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        self.assertEqual(self.content_allocator.available_size(), 12)
        self.cache.reset()
        self.assertEqual(self.content_allocator.available_size(), 16)

    def test_reset_clears_tree_pending_and_active_pin_together(self):
        self._insert([1, 2, 3, 4], draft_start=0)
        self._match([1, 2, 3, 4], "pinned")
        pending = self.cache.alloc_dflash_draft_content(4)
        self.cache.stage_dflash_draft_publish("pending", 4, pending)
        self.assertEqual(self.content_allocator.available_size(), 8)
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 4
        )

        self.cache.reset()

        self.assertEqual(self.content_allocator.available_size(), 16)
        self.assertIsNone(self.component.get_match_plan("pinned"))
        self.assertFalse(self.component.discard_pending_publish("pending"))
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 0
        )
        self.assertEqual(
            self.cache.tree_core.component_evictable_size(ComponentType.DRAFT), 0
        )

    def test_staged_abort_and_forced_miss_release_component_state(self):
        rows = self.cache.alloc_dflash_draft_content(4)
        self.cache.stage_dflash_draft_publish("abort", 8, rows)
        self.cache.release_aborted_request("abort")
        self.assertEqual(self.content_allocator.available_size(), 16)

        self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        result = self._match([1, 2, 3, 4, 5, 6], "forced-miss")
        self.assertIsNotNone(self.component.get_match_plan("forced-miss"))
        missed = zero_match_result(
            self.cache, result, extra_key=None, rid="forced-miss"
        )
        self.assertEqual(missed.device_indices.numel(), 0)
        self.assertIsNone(self.component.get_match_plan("forced-miss"))

    def test_finished_without_insert_returns_staged_publish(self):
        rows = self.cache.alloc_dflash_draft_content(4)
        self.cache.stage_dflash_draft_publish("skip-or-abort", 8, rows)
        self.assertEqual(self.content_allocator.available_size(), 12)

        self.component.cleanup_after_caching_req(
            SimpleNamespace(rid="skip-or-abort"),
            is_finished=True,
            insert_params=None,
        )

        self.assertEqual(self.content_allocator.available_size(), 16)
        self.assertFalse(self.component.discard_pending_publish("skip-or-abort"))

    def test_internal_maintenance_match_does_not_create_restore_pin(self):
        self._insert([1, 2, 3, 4, 5, 6], draft_start=2)
        result = self.cache.match_prefix(
            MatchPrefixParams(
                key=self._key([1, 2, 3, 4, 5, 6]),
                req=SimpleNamespace(rid="maintenance", session=None),
                pin_dflash_restore=False,
            )
        )

        self.assertEqual(result.device_indices.numel(), 6)
        self.assertIsNone(self.component.get_match_plan("maintenance"))
        self.assertEqual(
            self.cache.tree_core.component_protected_size_[ComponentType.DRAFT], 0
        )

    def test_partial_draft_maintenance_rematch_preserves_full_ownership(self):
        req_to_token_pool = ReqToTokenPool(
            size=4,
            max_context_len=32,
            device="cpu",
            enable_memory_saver=False,
        )
        content_allocator = DFlashDraftContentAllocator(start=100, size=3, device="cpu")
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=self.full_allocator,
                page_size=1,
                tree_components=(
                    ComponentType.FULL,
                    ComponentType.DRAFT,
                    ComponentType.MAMBA,
                ),
                component_registry_override={
                    ComponentType.MAMBA: _CachingCheckpointBoundaryComponent
                },
                dflash_draft_content_allocator=content_allocator,
                dflash_draft_window_size=4,
            )
        )

        prompt = array("q", [1, 2, 3, 4, 5, 6])
        req = Req(
            rid="partial-draft-owner",
            origin_input_text="",
            origin_input_ids=prompt,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=2),
        )
        req_to_token_pool.alloc([req])
        req.full_untruncated_fill_ids = prompt
        req.set_extend_range(0, len(prompt))
        req.last_node = cache.root_node_handle()
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        prompt_rows = self.full_allocator.alloc(len(prompt))
        req_to_token_pool.write((req.req_pool_idx, slice(0, len(prompt))), prompt_rows)
        req.kv_committed_len = len(prompt)

        draft_rows = cache.alloc_dflash_draft_content(3)
        cache.stage_dflash_draft_publish(req.rid, 3, draft_rows)
        cache.cache_unfinished_req(req)

        self.assertEqual(req.cache_protected_len, len(prompt))
        torch.testing.assert_close(req.prefix_indices, prompt_rows)
        self.assertEqual(cache.full_protected_size(), len(prompt))

        # Admission still requires a complete trailing draft window.  Only the
        # internal ownership rematch is allowed to ignore DRAFT coverage.
        admission = cache.match_prefix(
            MatchPrefixParams(
                key=self._key(prompt),
                req=SimpleNamespace(rid="partial-admission", session=None),
            )
        )
        self.assertEqual(admission.device_indices.numel(), 0)

        req.output_ids = array("q", [7, 8])
        req.full_untruncated_fill_ids = prompt + req.output_ids
        req.set_extend_range(len(prompt), len(req.full_untruncated_fill_ids))
        output_rows = self.full_allocator.alloc(len(req.output_ids))
        req_to_token_pool.write(
            (
                req.req_pool_idx,
                slice(len(prompt), len(req.full_untruncated_fill_ids)),
            ),
            output_rows,
        )
        req.kv_committed_len = len(req.full_untruncated_fill_ids)
        cache.cache_finished_req(
            req, is_insert=True, kv_len_to_handle=req.kv_committed_len
        )

        all_rows = torch.cat([prompt_rows, output_rows])
        self.assertEqual(self.full_allocator.available_size(), 64 - len(all_rows))
        self.assertEqual(cache.full_evictable_size(), len(all_rows))
        self.assertEqual(cache.full_protected_size(), 0)
        self.assertFalse(
            bool(torch.any(torch.isin(all_rows, self.full_allocator.free_pages)))
        )
        cache.sanity_check()

    def test_later_component_finalizer_failure_releases_restore_pin(self):
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4,
                    max_context_len=32,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=self.full_allocator,
                page_size=1,
                tree_components=(
                    ComponentType.FULL,
                    ComponentType.DRAFT,
                    ComponentType.MAMBA,
                ),
                component_registry_override={
                    ComponentType.MAMBA: _FailingFinalizeComponent
                },
                dflash_draft_content_allocator=self.content_allocator,
                dflash_draft_window_size=4,
            )
        )
        component = cache.components[ComponentType.DRAFT]
        draft_rows = self.content_allocator.alloc(4)
        self.content_allocator.claim_lease(draft_rows)
        params = InsertParams(
            key=self._key([1, 2, 3, 4, 5, 6]),
            value=self.full_allocator.alloc(6),
            mamba_value=torch.tensor([1]),
            draft_value=draft_rows,
            draft_start_seqlen=2,
            draft_processed=torch.zeros(4, dtype=torch.bool),
        )
        cache.insert(params)
        component.cleanup_pending_rows(params)

        with self.assertRaisesRegex(RuntimeError, "injected Mamba"):
            cache.match_prefix(
                MatchPrefixParams(
                    key=self._key([1, 2, 3, 4, 5, 6]),
                    req=SimpleNamespace(rid="finalizer-error", session=None),
                )
            )
        self.assertIsNone(component.get_match_plan("finalizer-error"))
        self.assertEqual(
            cache.tree_core.component_protected_size_[ComponentType.DRAFT], 0
        )

    def test_cache_prepare_failure_returns_finished_and_unfinished_rows(self):
        for is_finished in (False, True):
            content_allocator = DFlashDraftContentAllocator(
                start=100, size=8, device="cpu"
            )
            req_pool = ReqToTokenPool(
                size=2,
                max_context_len=32,
                device="cpu",
                enable_memory_saver=False,
            )
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=req_pool,
                    token_to_kv_pool_allocator=self.full_allocator,
                    page_size=1,
                    tree_components=(
                        ComponentType.FULL,
                        ComponentType.DRAFT,
                        ComponentType.MAMBA,
                    ),
                    component_registry_override={
                        ComponentType.MAMBA: _FailingPrepareComponent
                    },
                    dflash_draft_content_allocator=content_allocator,
                    dflash_draft_window_size=4,
                )
            )
            full_rows = self.full_allocator.alloc(4)
            req_pool.req_to_token[1, :4] = full_rows
            req = SimpleNamespace(
                rid=f"prepare-error-{is_finished}",
                session=None,
                req_pool_idx=1,
                cache_protected_len=0,
                priority=0,
                extra_key=None,
                cache_salt=None,
                origin_input_ids=[1, 2, 3, 4],
                output_ids=[],
                get_fill_ids=lambda: [1, 2, 3, 4],
            )
            rows = cache.alloc_dflash_draft_content(4)
            cache.stage_dflash_draft_publish(req.rid, 0, rows)

            with self.assertRaisesRegex(RuntimeError, "injected Mamba prepare"):
                if is_finished:
                    cache.cache_finished_req(req, is_insert=True, kv_len_to_handle=4)
                else:
                    cache.cache_unfinished_req(req)
            self.assertEqual(content_allocator.available_size(), 8)

    def test_full_mamba_draft_common_boundary_never_overreports(self):
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4,
                    max_context_len=32,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=self.full_allocator,
                page_size=1,
                tree_components=(
                    ComponentType.FULL,
                    ComponentType.DRAFT,
                    ComponentType.MAMBA,
                ),
                component_registry_override={
                    ComponentType.MAMBA: _CheckpointBoundaryComponent
                },
                dflash_draft_content_allocator=self.content_allocator,
                dflash_draft_window_size=4,
            )
        )
        component = cache.components[ComponentType.DRAFT]

        def insert(tokens, *, draft_start, has_mamba):
            draft_rows = self.content_allocator.alloc(len(tokens) - draft_start)
            self.content_allocator.claim_lease(draft_rows)
            params = InsertParams(
                key=self._key(tokens),
                value=self.full_allocator.alloc(len(tokens)),
                mamba_value=torch.tensor([1]) if has_mamba else None,
                draft_value=draft_rows,
                draft_start_seqlen=draft_start,
                draft_processed=torch.zeros(len(draft_rows), dtype=torch.bool),
            )
            cache.insert(params)
            component.cleanup_pending_rows(params)

        insert([1, 2, 3, 4, 5, 6], draft_start=2, has_mamba=True)
        hit = cache.match_prefix(
            MatchPrefixParams(
                key=self._key([1, 2, 3, 4, 5, 6]),
                req=SimpleNamespace(rid="all", session=None),
            )
        )
        self.assertEqual(hit.device_indices.numel(), 6)
        component.release_match_pin("all")

        insert([1, 2, 3, 4, 9, 10], draft_start=2, has_mamba=False)
        missing_mamba = cache.match_prefix(
            MatchPrefixParams(
                key=self._key([1, 2, 3, 4, 9, 10]),
                req=SimpleNamespace(rid="missing-mamba", session=None),
            )
        )
        self.assertEqual(missing_mamba.device_indices.numel(), 0)
        maintenance_missing_mamba = cache.match_prefix(
            MatchPrefixParams(
                key=self._key([1, 2, 3, 4, 9, 10]),
                req=SimpleNamespace(rid="maintenance-missing-mamba", session=None),
                pin_dflash_restore=False,
                require_dflash_draft=False,
            )
        )
        self.assertEqual(maintenance_missing_mamba.device_indices.numel(), 0)

        insert([7, 8, 9, 10, 11, 12], draft_start=3, has_mamba=True)
        missing_draft = cache.match_prefix(
            MatchPrefixParams(
                key=self._key([7, 8, 9, 10, 11, 12]),
                req=SimpleNamespace(rid="missing-draft", session=None),
            )
        )
        self.assertEqual(missing_draft.device_indices.numel(), 0)


if __name__ == "__main__":
    unittest.main()
