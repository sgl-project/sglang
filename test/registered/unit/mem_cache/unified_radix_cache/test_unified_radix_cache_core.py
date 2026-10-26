import unittest
from array import array
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Req,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
    TreeLock,
    zero_match_result,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import available_and_evictable_str, release_kv_cache
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeDeviceKV,
    RecoverSWAWithLockedFull,
    UpdatePendingBackupOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    EvictDeviceLeafResult,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    COMPONENT_REGISTRY,
    UnifiedLRUList,
    UnifiedRadixCache,
    UnifiedTreeNode,
    _OngoingWriteThrough,
)
from sglang.srt.runtime_context import (
    get_serving,
    publish,
    reset_context,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
    define_config_test_classes,
    requires,
    requires_full_swa_ps1_window4,
    requires_mamba,
    requires_paged,
    requires_ps1_window4,
    requires_swa,
    requires_swa_and_mamba,
    requires_swa_only,
)
from sglang.test.kits.unified_radix_cache.core import (
    DisabledUnifiedRadixCacheFixtures,
    FakeFullComponent,
    MambaCheckpointGridFixtures,
    MambaFinishedOvershootCheckpointFixtures,
    PagedFullComponent,
    SWADraftHistoryFixtures,
    SWAWindowUnderBigramKeyFixtures,
    UnifiedLRUListBoundedRefreshFixtures,
    UnifiedMambaLRUMatchRefreshFixtures,
    UnifiedRadixAllocationEvictionRealComponentsFixtures,
    UnifiedRadixCacheCoreFixtures,
    UnifiedRadixCacheDecoderSwaBoundedReplayHoldBackFixtures,
    UnifiedRadixCacheInt8MambaCheckpointFixtures,
    component_with_cache,
    session_radix_cache_test_values,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_children,
    node_device_lock_ref,
    node_device_value,
    node_key_length,
    node_parent,
)
from sglang.test.mem_cache_utils import finish_req
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=24, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=24, suite="stage-b-test-1-gpu-small-amd")


class TestUnifiedRadixComponentRegistryOverride(CustomTestCase):
    def test_component_registry_override_is_instance_local(self):
        params = CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=2,
                max_context_len=8,
                device="cpu",
                enable_memory_saver=False,
            ),
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=True,
            tree_components=(ComponentType.FULL,),
            component_registry_override={ComponentType.FULL: FakeFullComponent},
        )

        cache = UnifiedRadixCache(params=params)

        self.assertIsInstance(cache.components[ComponentType.FULL], FakeFullComponent)
        self.assertIsNot(COMPONENT_REGISTRY[ComponentType.FULL], FakeFullComponent)


class TestUnifiedRadixComponentLedgerUnits(CustomTestCase):
    def test_ledgers_count_reclaimable_tokens(self):
        """Ledgers count allocator units: every site reads reclaimable_value_len,
        not len(value), or the ledger drifts from sanity_check's recount."""
        cache, allocator, _ = build_fixture(
            CacheConfig(),
            component_registry_override={ComponentType.FULL: PagedFullComponent},
            tree_core_backend="python",
        )
        component = cache.components[ComponentType.FULL]

        def ledgers():
            cache.sanity_check()
            return cache.evictable_size(), cache.protected_size()

        tokens = array("q", range(1, 9))
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        cache.insert(InsertParams(key=RadixKey(tokens), value=value))
        self.assertEqual(ledgers(), (4, 0))

        leaf = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens))
        ).last_device_node
        lock_params = cache.inc_lock_ref(leaf).to_dec_params()
        self.assertEqual(ledgers(), (0, 4))

        # The figure changes in place, without a lock transition.
        component.rows_handed_back[leaf] = 1
        cache.tree_core.adjust_component_size(leaf, ComponentType.FULL, -1)
        self.assertEqual(ledgers(), (0, 3))

        cache.dec_lock_ref(leaf, lock_params)
        self.assertEqual(ledgers(), (3, 0))

        # The eviction walk is satisfied by what the node reclaims, not by
        # its value length.
        cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(ledgers(), (0, 0))
        self.assertEqual(
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey(tokens))
            ).device_prefix_len,
            0,
        )


class TestDisabledUnifiedRadixCache(DisabledUnifiedRadixCacheFixtures, CustomTestCase):
    def test_disabled_cache_skips_events_and_eviction_config(self):
        cache = UnifiedRadixCache(params=self._params(disable=True))
        cache.reset()
        self.assertEqual(cache.take_events(), [])

        with self.assertRaises(TypeError):
            UnifiedRadixCache(params=self._params(disable=False))


class TestUnifiedTreeNodeGetPrefixHashValues(CustomTestCase):
    def test_get_prefix_hash_values_not_shared_across_calls(self):
        """Cached prefix hash lists must not be mutated by callers."""

        def make_node():
            return UnifiedTreeNode(tree_components=(ComponentType.FULL,))

        root = make_node()
        n1 = make_node()
        n1.parent = root
        n1.hash_value = ["h1"]
        n2 = make_node()
        n2.parent = n1
        n2.hash_value = ["h2"]
        n3 = make_node()
        n3.parent = n2
        n3.hash_value = ["h3"]

        first = n3.get_prefix_hash_values(n2)
        self.assertEqual(first, ["h1", "h2"])

        # Mimic downstream storage code that extends `prefix_keys` in place.
        first += ["h3"]

        second = n3.get_prefix_hash_values(n2)
        self.assertEqual(second, ["h1", "h2"])
        self.assertIsNot(second, first)

        n4 = make_node()
        n4.parent = n3
        n4.hash_value = ["h4"]
        self.assertEqual(n4.get_prefix_hash_values(n3), ["h1", "h2", "h3"])


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestUnifiedRadixAllocationEvictionRealComponents(
    UnifiedRadixAllocationEvictionRealComponentsFixtures, CustomTestCase
):
    """Allocation targets are observed between real auxiliary-tree steps."""

    _SHORTFALL = 100

    def test_swa_write_back_host_full_preserves_full_and_continues(self):
        ct = ComponentType.SWA
        for session in session_radix_cache_test_values():
            for pinned in (False, True):
                with self.subTest(session=session, pinned=pinned):
                    # Rust independently of the fixtures' default backend;
                    # session caches fall back to the Python TreeCore.
                    with mock.patch(
                        "sglang.test.kits.unified_radix_cache.fixtures.TREE_CORE_TEST_BACKEND",
                        "rust",
                    ):
                        cache, first, second, leaf = self._build_internal_chain(
                            ct, session
                        )
                    cache.tree_core.is_write_back = True
                    cache.tree_core.has_swa_host_pool = True
                    # Both cores gate the internal SWA component backup on the
                    # HiCache attachment.
                    cache.tree_core.set_hicache_enabled()
                    cache.host_pool_group = mock.Mock()
                    cache.host_pool_group.get_pool.return_value = None
                    if pinned:
                        receipt = cache.inc_host_lock_ref(first).to_dec_params()
                    tracker = {ComponentType.FULL: 0, ct: 0}
                    # Only host allocation fails; tree walking and freeing are real.
                    with (
                        mock.patch.object(cache, "cache_controller", mock.Mock()),
                        mock.patch.object(
                            cache, "_execute_and_commit_kv_backup", return_value=0
                        ) as backup,
                    ):
                        cache._evict_components({ComponentType.FULL: 0, ct: 2}, tracker)
                    backup.assert_called_once()
                    self.assertEqual(tracker[ct], 2)
                    self.assertEqual(tracker[ComponentType.FULL], 0)
                    for node_id in (first, second, leaf):
                        self.assertIsNotNone(
                            node_device_value(cache, node_id, ComponentType.FULL)
                        )
                    if pinned:
                        # A host pin does not pin a device-only SWA value.
                        self.assertIsNone(node_device_value(cache, first, ct))
                        cache.dec_host_lock_ref(first, receipt)
                    cache.sanity_check()

    def test_allocation_target_stops_after_one_internal_tombstone(self):
        for component_type in (ComponentType.SWA, ComponentType.MAMBA):
            for enable_session_radix_cache in session_radix_cache_test_values():
                with self.subTest(
                    component_type=component_type,
                    enable_session_radix_cache=enable_session_radix_cache,
                ):
                    cache, first, second, leaf = self._build_internal_chain(
                        component_type, enable_session_radix_cache
                    )
                    first_size = len(node_device_value(cache, first, component_type))

                    result, drain_count = self._evict_for_alloc_after_first_drain(
                        cache, component_type
                    )

                    self.assertIsNone(node_device_value(cache, first, component_type))
                    self.assertIsNotNone(
                        node_device_value(cache, second, component_type)
                    )
                    self.assertIsNotNone(node_device_value(cache, leaf, component_type))
                    self.assertIsNotNone(
                        node_device_value(cache, leaf, ComponentType.FULL)
                    )
                    self.assertEqual(result.num_tokens_evicted, 0)
                    self.assertEqual(drain_count, 1)
                    evicted = (
                        result.swa_num_tokens_evicted
                        if component_type is ComponentType.SWA
                        else result.mamba_num_evicted
                    )
                    self.assertEqual(evicted, first_size)
                    cache.sanity_check()

    def test_explicit_evict_continues_across_internal_steps(self):
        for component_type in (ComponentType.SWA, ComponentType.MAMBA):
            for enable_session_radix_cache in session_radix_cache_test_values():
                with self.subTest(
                    component_type=component_type,
                    enable_session_radix_cache=enable_session_radix_cache,
                ):
                    cache, first, second, leaf = self._build_internal_chain(
                        component_type, enable_session_radix_cache
                    )
                    request_count = sum(
                        len(node_device_value(cache, node_id, component_type))
                        for node_id in (first, second)
                    )
                    params = (
                        EvictParams(swa_num_tokens=request_count)
                        if component_type is ComponentType.SWA
                        else EvictParams(mamba_num=request_count)
                    )

                    result = cache.evict(params)

                    self.assertIsNone(node_device_value(cache, first, component_type))
                    self.assertIsNone(node_device_value(cache, second, component_type))
                    self.assertIsNotNone(node_device_value(cache, leaf, component_type))
                    self.assertIsNotNone(
                        node_device_value(cache, leaf, ComponentType.FULL)
                    )
                    evicted = (
                        result.swa_num_tokens_evicted
                        if component_type is ComponentType.SWA
                        else result.mamba_num_evicted
                    )
                    self.assertEqual(evicted, request_count)
                    cache.sanity_check()


class UnifiedRadixCacheCoreSuite(UnifiedRadixCacheCoreFixtures):
    def test_insert_and_match_basic(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 2)
        seq_b = seq_a + self._make_seq(1000, 1)

        self._insert(cache, allocator, req_to_token_pool, seq_a)
        result = self._insert(cache, allocator, req_to_token_pool, seq_b)
        self.assertEqual(result.prefix_len, len(seq_a))

        m = match_prefix(cache, seq_b)
        self.assertEqual(result.last_device_node, m.last_device_node)
        self.assertEqual(m.device_prefix_len, len(seq_b))

        m = match_prefix(cache, seq_a + self._make_seq(9000, 1))
        self.assertEqual(m.device_prefix_len, len(seq_a))

        m = match_prefix(cache, self._make_seq(5000, 2))
        self.assertEqual(m.device_prefix_len, 0)

        cache.sanity_check()

    def test_cache_salt_and_extra_key_form_independent_namespaces(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)

        first = self._insert(
            cache,
            allocator,
            req_to_token_pool,
            seq,
            extra_key="bc",
            cache_salt="a",
        )
        second = self._insert(
            cache,
            allocator,
            req_to_token_pool,
            seq,
            extra_key="c",
            cache_salt="ab",
        )
        self.assertEqual(first.prefix_len, 0)
        self.assertEqual(second.prefix_len, 0)

        first_match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", seq), extra_key="bc", cache_salt="a")
            )
        )
        second_match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", seq), extra_key="c", cache_salt="ab")
            )
        )
        default_match = match_prefix(cache, seq)
        self.assertEqual(first_match.device_prefix_len, len(seq))
        self.assertEqual(second_match.device_prefix_len, len(seq))
        self.assertEqual(default_match.device_prefix_len, 0)
        self.assertNotEqual(first_match.last_device_node, second_match.last_device_node)
        cache.sanity_check()

    def test_shared_prefix_split(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, base)

        branch_a = base + self._make_seq(100, 2)
        branch_b = base + self._make_seq(200, 2)

        result_a = self._insert(cache, allocator, req_to_token_pool, branch_a)
        self.assertEqual(result_a.prefix_len, len(base))
        result_b = self._insert(cache, allocator, req_to_token_pool, branch_b)
        self.assertEqual(result_b.prefix_len, len(base))

        for seq in (branch_a, branch_b):
            m = match_prefix(cache, seq)
            self.assertEqual(m.device_prefix_len, len(seq))

        m = match_prefix(cache, base + self._make_seq(999, 1))
        self.assertEqual(m.device_prefix_len, len(base))
        cache.sanity_check()

    def test_evict_respects_lock_ref(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_a = self._make_seq(1, 2)
        seq_b = self._make_seq(500, 2)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_b)

        m = match_prefix(cache, seq_a)
        lock_result = cache.inc_lock_ref(m.last_device_node)

        result = cache.evict(EvictParams(num_tokens=len(seq_a) + len(seq_b)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_b))

        m = match_prefix(cache, seq_a)
        self.assertEqual(m.device_prefix_len, len(seq_a))

        cache.dec_lock_ref(
            m.last_device_node,
            lock_result.to_dec_params(),
        )
        result = cache.evict(EvictParams(num_tokens=len(seq_a)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_a))
        cache.sanity_check()

    def test_prev_prefix_len(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        initial_avail = allocator.available_size()

        seq_1p = self._make_seq(1, 1)  # 1 page
        seq_2p = self._make_seq(1, 2)  # 2 pages (extends seq_1p)
        seq_3p = self._make_seq(1, 3)  # 3 pages (extends seq_2p)

        self._insert(cache, allocator, req_to_token_pool, seq_1p)
        self.assertEqual(allocator.available_size(), initial_avail - len(seq_1p))

        key_2p = RadixKey(array("q", seq_2p))
        value_2p = self._alloc(allocator, len(seq_2p))
        params = InsertParams(
            key=key_2p,
            value=value_2p[: len(key_2p)],
            prev_prefix_len=0,
            track_adopted_ranges=True,
        )
        if self.cfg.has_mamba:
            req = self._make_req(req_to_token_pool)
            params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
        result = cache.insert(params)
        self.assertEqual(result.prefix_len, len(seq_1p))
        self.assertEqual(
            result.adopted_ranges[ComponentType.FULL],
            [(len(seq_1p), len(seq_2p))],
        )
        self.assertEqual(
            allocator.available_size(),
            initial_avail - len(seq_1p) - (len(seq_2p) - len(seq_1p)),
        )

        avail_before = allocator.available_size()
        key_3p = RadixKey(array("q", seq_3p))
        value_3p = self._alloc(allocator, len(seq_3p))
        params = InsertParams(
            key=key_3p,
            value=value_3p[: len(key_3p)],
            prev_prefix_len=len(seq_2p),
            track_adopted_ranges=True,
        )
        if self.cfg.has_mamba:
            req = self._make_req(req_to_token_pool)
            params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
        result = cache.insert(params)
        self.assertEqual(result.prefix_len, len(seq_2p))
        self.assertEqual(
            result.adopted_ranges[ComponentType.FULL],
            [(len(seq_2p), len(seq_3p))],
        )
        # alloc(3p), freed 0, stored 1p new -> net -3p.
        self.assertEqual(allocator.available_size(), avail_before - len(seq_3p))
        cache.sanity_check()

    def test_checkpoint(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size

        input_ids = self._make_seq(1, 3)
        output_ids = self._make_seq(2000, 1)
        req, _ = self._make_inflight_req(
            cache, allocator, req_to_token_pool, input_ids, output_ids
        )
        kv_len = req.extend_end
        if self.cfg.has_mamba:
            req.kv.mamba_last_track_seqlen = kv_len

        finish_req(cache, req, req.owned_kv_len())

        all_ids = input_ids + output_ids
        aligned_len = (len(all_ids) // ps) * ps
        m = match_prefix(cache, all_ids[:aligned_len])
        self.assertEqual(m.device_prefix_len, aligned_len)

        prompt_aligned_len = (len(input_ids) // ps) * ps
        if self.cfg.components == (ComponentType.FULL,):
            (prompt_node,) = node_children(cache, cache.root_node_handle())
            (output_node,) = node_children(cache, prompt_node)
            self.assertEqual(node_key_length(cache, prompt_node), prompt_aligned_len)
            self.assertEqual(
                node_key_length(cache, output_node),
                aligned_len - prompt_aligned_len,
            )

            result = cache.evict(
                EvictParams(num_tokens=aligned_len - prompt_aligned_len)
            )
            self.assertEqual(
                result.num_tokens_evicted,
                aligned_len - prompt_aligned_len,
            )
            prompt_only = match_prefix(cache, all_ids[:aligned_len])
            self.assertEqual(prompt_only.device_prefix_len, prompt_aligned_len)
        cache.sanity_check()

    def test_checkpoint_strips_thinking(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size

        prompt_ids = self._make_seq(1, 3)
        output_ids = self._make_seq(2000, 7)
        req, _ = self._make_inflight_req(
            cache, allocator, req_to_token_pool, prompt_ids, output_ids, allocated=True
        )
        kv_len = req.extend_end
        if self.cfg.has_mamba:
            req.kv.mamba_last_track_seqlen = kv_len
            if self.cfg.enable_mamba_extra_buffer:
                # The prompt-only key needs its own checkpoint, not the
                # latest state that has already consumed thinking + answer.
                req.kv.mamba_prev_track_seqlen = len(prompt_ids)
                prompt_slot = req.kv.mamba_ping_pong_track_buffer[
                    req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.kv.mamba_last_track_idx
                    )
                ].clone()
        req.reasoning_tokens = 1

        # owned_kv_len reads get_serving().strip_thinking_cache
        with get_serving().override(strip_thinking_cache=True):
            avail_before = allocator.available_size()
            finish_req(cache, req, req.owned_kv_len())
            start_p, end_p = req.owned_kv_len(), req.kv.kv_allocated_len
        if ps > 1:
            start_p = ((start_p + ps - 1) // ps) * ps
        if start_p < end_p:
            allocator.free(
                req_to_token_pool.req_to_token[req.kv.req_pool_idx][start_p:end_p]
            )

        prompt_aligned = (len(prompt_ids) // ps) * ps
        # Thinking+answer must not be reachable past the prompt.
        m = match_prefix(cache, prompt_ids + output_ids)
        self.assertEqual(m.device_prefix_len, prompt_aligned)
        if self.cfg.has_mamba and self.cfg.enable_mamba_extra_buffer:
            node_value = cache.tree_core.get_component_device_value(
                m.last_device_node, ComponentType.MAMBA
            )
            self.assertTrue(
                torch.equal(
                    node_value.reshape(-1),
                    prompt_slot.reshape(-1),
                )
            )
        # Only prompt-aligned pages remain owned by the tree.
        self.assertEqual(
            allocator.available_size(), avail_before + kv_len - prompt_aligned
        )
        cache.sanity_check()

    def test_release_without_insert(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        tokens = self._make_seq(1, 2)
        req, _ = self._make_inflight_req(
            cache, allocator, req_to_token_pool, tokens, allocated=True
        )
        kv_len = len(tokens)

        avail_before = allocator.available_size()
        release_kv_cache(req, cache, checkpoint=False)

        self.assertEqual(allocator.available_size(), avail_before + kv_len)
        m = match_prefix(cache, tokens)
        self.assertEqual(m.device_prefix_len, 0)
        cache.sanity_check()

    def test_checkpoint_mid_flight(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        tokens = self._make_seq(1, 3)
        req, _ = self._make_inflight_req(cache, allocator, req_to_token_pool, tokens)
        kv_len = len(tokens)
        if self.cfg.has_mamba:
            req.kv.mamba_last_track_seqlen = kv_len

        cache.checkpoint(req, up_to=req.extend_end)

        self.assertGreater(req.kv.cache_protected_len, 0)
        self.assertIsNotNone(req.last_node)
        self.assertFalse(req.lock.swa_released)

        cache.unlock(req.lock)
        cache.sanity_check()

    @requires_swa_only
    @requires_ps1_window4
    def test_swa_unfinished_req_repoints_whole_insert_below_window(self):
        # Even a live SWA tail shorter than the window must be protected for the whole
        # insert, else the request frees tree-owned SWA pages and re-inserts them at
        # finish.
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        full_avail = allocator.full_attn_allocator.available_size()
        swa_avail = allocator.swa_attn_allocator.available_size()

        tokens = self._make_seq(1, 8)
        # Live SWA is [6, 8): two tokens, below the window of four.
        evicted_len = 6
        req, kv_indices = self._make_inflight_req(
            cache, allocator, req_to_token_pool, tokens
        )
        # The window already slid past [0, 6): those SWA slots are freed.
        allocator.free_swa(kv_indices[:evicted_len])
        req.kv.set_evicted_seqlen(ComponentType.SWA, evicted_len)

        cache.checkpoint(req, up_to=req.extend_end)

        inserted = len(tokens) - (1 if self.cfg.is_eagle else 0)
        self.assertEqual(req.kv.cache_protected_len, inserted)
        torch.testing.assert_close(
            req_to_token_pool.req_to_token[req.kv.req_pool_idx, : len(tokens)],
            kv_indices,
            check_dtype=False,
        )
        # A fresh matcher still needs a full live window and gets nothing.
        m = match_prefix(cache, tokens)
        self.assertEqual(m.device_prefix_len, 0)

        finish_req(cache, req, len(tokens))
        req.kv.mark_kv_released()
        cache.sanity_check()
        # Everything the request held is now tree-owned exactly once.
        self.assertEqual(
            allocator.full_attn_allocator.available_size()
            + cache.full_evictable_size(),
            full_avail,
        )
        self.assertEqual(
            allocator.swa_attn_allocator.available_size() + cache.swa_evictable_size(),
            swa_avail,
        )

    @requires_swa_only
    @requires_ps1_window4
    def test_swa_unfinished_req_preserves_existing_eviction_boundary(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        tokens = self._make_seq(1, 8)
        evicted_len = 4
        req, _ = self._make_inflight_req(cache, allocator, req_to_token_pool, tokens)
        req.kv.set_evicted_seqlen(ComponentType.SWA, evicted_len)

        cache.checkpoint(req, up_to=req.extend_end)

        (first,) = node_children(cache, cache.root_node_handle())
        self.assertEqual(node_key_length(cache, first), evicted_len)
        self.assertIsNone(node_device_value(cache, first, ComponentType.SWA))
        (live,) = node_children(cache, first)
        self.assertEqual(node_key_length(cache, live), len(tokens) - evicted_len)
        self.assertIsNotNone(node_device_value(cache, live, ComponentType.SWA))

        m = match_prefix(cache, tokens)
        self.assertEqual(m.device_prefix_len, len(tokens))

        cache.unlock(req.lock)
        cache.sanity_check()

    def test_diagnostics(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._insert(cache, allocator, req_to_token_pool, self._make_seq(1, 2))

        diag = cache.available_and_evictable_str()
        self.assertIn("Available full tokens", diag)
        if self.cfg.has_mamba:
            self.assertIn("mamba", diag.lower())
        if self.cfg.has_swa:
            self.assertIn("swa", diag.lower())

        diag2 = available_and_evictable_str(cache)
        self.assertIn("Available full tokens", diag2)
        cache.pretty_print()
        cache.sanity_check()

    @requires_paged
    def test_paged_match_truncates_unaligned_key(self):
        ps = self.cfg.page_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)

        unaligned = seq + list(range(9000, 9000 + ps - 1))
        m = match_prefix(cache, unaligned)
        self.assertEqual(m.device_prefix_len, len(seq))

        m = match_prefix(cache, seq[: ps - 1])
        self.assertEqual(m.device_prefix_len, 0)

        cache.sanity_check()

    @requires_paged
    def test_paged_page_boundary_mismatch(self):
        ps = self.cfg.page_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        first_page = self._make_seq(1, 1)
        seq = self._make_seq(1, 2)
        # Insert first page so it retains component data after the split
        # triggered by the partial-page match below.
        self._insert(cache, allocator, req_to_token_pool, first_page)
        self._insert(cache, allocator, req_to_token_pool, seq)

        bad_page2 = seq[:ps] + [9999] * ps
        m = match_prefix(cache, bad_page2)
        self.assertEqual(m.device_prefix_len, ps)

        bad_page1 = [9999] + seq[1:]
        m = match_prefix(cache, bad_page1)
        self.assertEqual(m.device_prefix_len, 0)
        cache.sanity_check()

    @requires_paged
    @requires(
        lambda cfg: not cfg.has_swa, "SWA paged allocator accounts in pages, not tokens"
    )
    def test_paged_checkpoint_unaligned_tail_freed(self):
        ps = self.cfg.page_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        tail_extra = ps // 2
        input_ids = self._make_seq(1, 1) + list(range(8000, 8000 + tail_extra))
        req, _ = self._make_inflight_req(cache, allocator, req_to_token_pool, input_ids)
        kv_len = len(input_ids)
        if self.cfg.has_mamba:
            req.kv.mamba_last_track_seqlen = kv_len

        avail_before = allocator.available_size()
        finish_req(cache, req, req.owned_kv_len())

        self.assertEqual(allocator.available_size(), avail_before + tail_extra)
        aligned = input_ids[: (len(input_ids) // ps) * ps]
        m = match_prefix(cache, aligned)
        self.assertEqual(m.device_prefix_len, len(aligned))
        cache.sanity_check()

    @requires_mamba
    def test_mamba_evict_breaks_match(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_short = self._make_seq(1, 2)
        seq_long = seq_short + self._make_seq(500, 1)
        self._insert(cache, allocator, req_to_token_pool, seq_short)
        self._insert(cache, allocator, req_to_token_pool, seq_long)

        cache.evict(EvictParams(num_tokens=0, mamba_num=10))
        self.assertEqual(cache.mamba_evictable_size(), 0)

        m = match_prefix(cache, seq_long)
        self.assertEqual(m.device_prefix_len, 0)
        cache.sanity_check()

    @requires_mamba
    def test_mamba_evict_cascades_on_full_leaf(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)

        result = cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        cache.sanity_check()

    @requires_mamba
    def test_mamba_cow_on_match(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        mamba_pool = req_to_token_pool.mamba_pool

        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)

        req2 = self._make_req(req_to_token_pool)
        m = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq)), cow_mamba=True, req=req2)
        )
        self.assertEqual(m.device_prefix_len, len(seq))
        self.assertIsNotNone(req2.kv.mamba_pool_idx)

        src_value = node_device_value(cache, m.last_device_node, ComponentType.MAMBA)
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.conv[0][:, req2.kv.mamba_pool_idx]
                == mamba_pool.mamba_cache.conv[0][:, src_value]
            )
        )
        cache.sanity_check()

    @requires_swa_only
    @requires_ps1_window4
    def test_swa_unfinished_recovery_preserves_locked_full_value(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        tokens = self._make_seq(1, 4)
        self._insert(cache, allocator, req_to_token_pool, tokens)
        self._insert(
            cache, allocator, req_to_token_pool, tokens + self._make_seq(100, 1)
        )

        last_device_node = match_prefix(cache, tokens).last_device_node
        old_full_value = node_device_value(
            cache, last_device_node, ComponentType.FULL
        ).clone()
        result = cache.tree_core.evict_component(
            last_device_node, ComponentType.SWA, EvictLayer.DEVICE
        )
        cache._free_values(result.device_frees, result.host_frees)
        self.assertIsNone(node_device_value(cache, last_device_node, ComponentType.SWA))

        lock_result = cache.inc_lock_ref(last_device_node)
        req, _ = self._make_inflight_req(cache, allocator, req_to_token_pool, tokens)
        req.kv.set_evicted_seqlen(ComponentType.SWA, 0)

        full_available_before_insert = allocator.full_attn_allocator.available_size()

        cache.checkpoint(req, up_to=req.extend_end)

        self.assertEqual(
            allocator.full_attn_allocator.available_size(),
            full_available_before_insert + len(tokens),
        )
        self.assertTrue(
            torch.equal(
                node_device_value(cache, last_device_node, ComponentType.FULL),
                old_full_value,
            )
        )
        swa_value = node_device_value(cache, last_device_node, ComponentType.SWA)
        self.assertIsNotNone(swa_value)
        self.assertTrue(
            torch.equal(
                allocator.translate_loc_from_full_to_swa(old_full_value),
                swa_value,
            )
        )
        self.assertEqual(req.kv.cache_protected_len, len(tokens))

        cache.unlock(req.lock)
        cache.dec_lock_ref(last_device_node, lock_result.to_dec_params())
        cache.sanity_check()

    @requires_swa_only
    def test_swa_insert_keeps_full_leaf_when_entire_span_is_outside_window(self):
        # A leaf survives on its Full value alone: even when the whole span is
        # past the SWA window (swa_evicted_seqlen >= total), the Full leaf must
        # be materialized (and the Full KV kept) so the prefix stays cacheable.
        # Runs across all SWA configs, including page_size > sliding_window_size
        cache, allocator, _ = build_fixture(self.cfg)

        tokens = self._make_seq(1, 2)
        value = self._alloc(allocator, len(tokens))
        if value is None:
            self.skipTest("insufficient pool for this config")
        full_available_before = allocator.full_attn_allocator.available_size()

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value,
                prev_prefix_len=0,
                component_evicted_seqlens={ComponentType.SWA: len(tokens)},
            )
        )

        self.assertEqual(
            allocator.full_attn_allocator.available_size(), full_available_before
        )
        (node,) = node_children(cache, cache.root_node_handle())
        self.assertTrue(
            torch.equal(node_device_value(cache, node, ComponentType.FULL), value)
        )
        self.assertIsNone(node_device_value(cache, node, ComponentType.SWA))
        cache.sanity_check()

    @requires_swa_and_mamba
    def test_leaf_transition_swa_evict_spares_locked_full(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        seq_ab = seq_a + self._make_seq(7000, 2)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_ab)

        m = match_prefix(cache, seq_a)
        node_a = m.last_device_node
        self.assertGreater(len(node_children(cache, node_a)), 0, "A must be internal")

        # Early SWA release on an internal node also drops the lower-tier Mamba lock,
        # so later SWA eviction never meets a locked Mamba.
        lock_result = cache.inc_lock_ref(node_a)
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.MAMBA),
            1,
            "Mamba locked before release",
        )
        cache.release_swa(TreeLock(node_a, lock_result.to_dec_params()))
        self.assertEqual(node_device_lock_ref(cache, node_a, ComponentType.SWA), 0)
        self.assertEqual(
            node_device_lock_ref(cache, node_a, ComponentType.MAMBA),
            0,
            "dec_swa_lock_only drops the lower-tier Mamba lock",
        )
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.FULL), 1
        )
        self.assertTrue(
            cache.tree_core.is_node_in_device_lru(node_a, ComponentType.SWA)
        )

        # Evicting the child makes A a Full-locked leaf with its SWA still in the LRU.
        cache.evict(EvictParams(num_tokens=len(seq_ab)))

        self.assertEqual(len(node_children(cache, node_a)), 0, "A should now be a leaf")
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.FULL),
            1,
            "Full must stay locked",
        )
        self.assertTrue(
            cache.tree_core.is_node_in_device_lru(node_a, ComponentType.SWA),
            "A's unlocked SWA stays in the LRU (not tombstoned at transition)",
        )

        # SWA eviction skips A's Full lock and cascades the unlocked lower-tier Mamba.
        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=len(seq_a)))
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.FULL),
            1,
            "Full must remain locked",
        )
        self.assertIsNotNone(
            node_device_value(cache, node_a, ComponentType.FULL),
            "Full KV must survive the SWA cascade",
        )
        self.assertIsNone(
            node_device_value(cache, node_a, ComponentType.SWA),
            "A's SWA was freed by its own eviction",
        )
        cache.sanity_check()

        cache.dec_lock_ref(node_a, lock_result.to_dec_params(), skip_swa=True)
        cache.sanity_check()

    @requires_swa_and_mamba
    def test_mamba_opt_out_holder_cannot_release_another_holders_mamba_lock(self):
        """Holder A takes the mamba lock; holder B opts out (skip_lock_components=(ComponentType.MAMBA,))
        on the same node. B's early SWA release and final release must leave
        A's mamba lock intact."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq = self._make_seq(
            1, (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        )
        self._insert(cache, allocator, req_to_token_pool, seq)
        node = match_prefix(cache, seq).last_device_node
        self.assertIsNotNone(node_device_value(cache, node, ComponentType.MAMBA))

        lock_a = cache.inc_lock_ref(node)
        lock_b = cache.inc_lock_ref(node, skip_lock_components=(ComponentType.MAMBA,))
        self.assertNotIn(ComponentType.MAMBA, lock_a.skipped_lock_components)
        self.assertIn(ComponentType.MAMBA, lock_b.skipped_lock_components)
        self.assertEqual(
            node_device_lock_ref(cache, node, ComponentType.MAMBA),
            1,
            "only A holds mamba",
        )

        # B: early SWA release, then final release -- both replay B's receipt.
        cache.release_swa(TreeLock(node, lock_b.to_dec_params()))
        self.assertEqual(
            node_device_lock_ref(cache, node, ComponentType.MAMBA),
            1,
            "B's early release spares A",
        )
        cache.dec_lock_ref(node, lock_b.to_dec_params(), skip_swa=True)
        self.assertEqual(
            node_device_lock_ref(cache, node, ComponentType.MAMBA),
            1,
            "B's final release spares A",
        )

        cache.release_swa(TreeLock(node, lock_a.to_dec_params()))
        self.assertEqual(
            node_device_lock_ref(cache, node, ComponentType.MAMBA),
            0,
            "A's release drops its own lock",
        )
        cache.dec_lock_ref(node, lock_a.to_dec_params(), skip_swa=True)
        cache.sanity_check()

    @requires_swa_and_mamba
    def test_swa_early_release_drops_co_located_mamba_lock(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        node_a = match_prefix(cache, seq_a).last_device_node
        self.assertEqual(len(node_children(cache, node_a)), 0, "A must be a leaf")
        self.assertIsNotNone(
            node_device_value(cache, node_a, ComponentType.MAMBA),
            "A must hold a Mamba checkpoint",
        )

        lock_result = cache.inc_lock_ref(node_a)
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.SWA), 1, "SWA locked"
        )
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.MAMBA), 1, "Mamba locked"
        )
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.FULL), 1, "Full locked"
        )

        # Early SWA release (decode advanced past the window), via the public
        # path the scheduler calls. The leaf's SWA is tombstoned and the
        # co-located lower-tier Mamba lock must drop in the same release.
        cache.release_swa(TreeLock(node_a, lock_result.to_dec_params()))
        self.assertEqual(
            node_device_lock_ref(cache, node_a, ComponentType.SWA),
            0,
            "SWA early-released",
        )
        self.assertEqual(
            node_device_lock_ref(cache, node_a, ComponentType.MAMBA),
            0,
            "Mamba lock must drop on early SWA release",
        )
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.FULL),
            1,
            "Full stays locked",
        )

    @requires_swa_and_mamba
    def test_cascade_evict_asserts_on_locked_leaf_mamba(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        self._insert(cache, allocator, req_to_token_pool, seq_a)

        node_a = match_prefix(cache, seq_a).last_device_node
        self.assertEqual(len(node_children(cache, node_a)), 0, "A must be a leaf")
        self.assertIsNotNone(
            node_device_value(cache, node_a, ComponentType.MAMBA),
            "A must hold a Mamba checkpoint",
        )

        # Same stranded-Mamba-lock setup as above.
        cache.tree_core.set_component_device_lock_ref(node_a, ComponentType.MAMBA, 1)
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.MAMBA), 1, "Mamba locked"
        )
        self.assertEqual(
            node_device_lock_ref(cache, node_a, ComponentType.FULL), 0, "Full unlocked"
        )

        result = cache.tree_core.evict_component(
            node_a, ComponentType.SWA, EvictLayer.DEVICE
        )
        cache._free_values(result.device_frees, result.host_frees)
        with self.assertRaises(AssertionError):
            cache.tree_core.validate_cascade_evict(
                node_a, ComponentType.SWA, EvictLayer.DEVICE
            )

        cache.tree_core.set_component_device_lock_ref(node_a, ComponentType.MAMBA, 0)

    @requires_swa
    def test_swa_leaf_capped_to_window_on_insert(self):
        ps = self.cfg.page_size
        window = self.cfg.sliding_window_size
        tail_size = ((window + ps - 1) // ps) * ps
        tail_pages = tail_size // ps

        for case in ("long_splits", "short_keeps"):
            with self.subTest(case=case):
                cache, allocator, req_to_token_pool = build_fixture(self.cfg)
                num_pages = tail_pages + 2 if case == "long_splits" else tail_pages
                seq = self._make_seq(1, num_pages)
                self._insert(cache, allocator, req_to_token_pool, seq)
                cache.sanity_check()

                leaf = match_prefix(cache, seq).last_device_node
                swa_val = node_device_value(cache, leaf, ComponentType.SWA)
                self.assertIsNotNone(swa_val)

                if case == "long_splits":
                    self.assertEqual(len(swa_val), tail_size)
                    self.assertNotEqual(
                        node_parent(cache, leaf), cache.root_node_handle()
                    )
                else:
                    self.assertEqual(len(swa_val), len(seq))
                    self.assertEqual(node_parent(cache, leaf), cache.root_node_handle())

                lock_result = cache.inc_lock_ref(leaf)
                # SWA pins one window; full attention pins everything.
                self.assertEqual(cache.swa_protected_size(), len(swa_val))
                self.assertEqual(cache.full_protected_size(), len(seq))
                cache.sanity_check()
                cache.dec_lock_ref(
                    leaf,
                    lock_result.to_dec_params(),
                )
                cache.sanity_check()

    def test_swa_lru_walk_down_does_not_refresh_ancestors_during_insert(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 8)
        seq_ab = seq_a + self._make_seq(100, 8)
        seq_abc = seq_ab + self._make_seq(200, 8)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_ab)
        self._insert(cache, allocator, req_to_token_pool, seq_abc)

        seq_side = self._make_seq(900, 5)
        self._insert(cache, allocator, req_to_token_pool, seq_side)

        pre = self._swa_lru_order(cache)
        # Each 8-page segment is cap-split into [prefix, tail]; the tail leads
        # the pair in MRU order, so segment tails sit at even indices.
        self.assertEqual(len(pre), 8)
        side_node, c_node, b_node, a_node = pre[0], pre[2], pre[4], pre[6]

        seq_abcd = seq_abc + self._make_seq(300, 8)
        self._insert(cache, allocator, req_to_token_pool, seq_abcd)

        post = self._swa_lru_order(cache)
        # New segment E adds two nodes (prefix + tail).
        self.assertEqual(len(post), 10)
        # side branch must still appear BEFORE B and A in MRU->LRU order:
        # walk-down on the new segment must not refresh old ancestors.
        side_pos = post.index(side_node)
        self.assertLess(
            side_pos,
            post.index(b_node),
            f"side branch must remain ahead of B (no walk-down refresh); "
            f"post={post}, side={side_node}, B={b_node}",
        )
        self.assertLess(
            side_pos,
            post.index(a_node),
            f"side branch must remain ahead of A (no walk-down refresh); "
            f"post={post}, side={side_node}, A={a_node}",
        )
        cache.sanity_check()

    @requires(lambda cfg: cfg.has_swa or cfg.has_mamba, "requires an aux LRU")
    def test_partial_match_keeps_unmatched_suffix_lru_position(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        cold_tokens = self._make_seq(1, 2)
        hot_tokens = self._make_seq(100, 2)
        cold = self._insert(cache, allocator, req_to_token_pool, cold_tokens)
        hot = self._insert(cache, allocator, req_to_token_pool, hot_tokens)

        match_prefix(cache, cold_tokens[: self.cfg.page_size])

        for ct in self.cfg.components:
            if ct == ComponentType.FULL:
                continue
            order = cache.tree_core.get_component_device_lru_node_ids(ct)
            self.assertLess(
                order.index(hot.last_device_node), order.index(cold.last_device_node)
            )
        cache.sanity_check()

    def test_swa_lru_match_only_refreshes_window_cushion(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 8)
        seq_ab = seq_a + self._make_seq(100, 8)
        seq_abc = seq_ab + self._make_seq(200, 8)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_ab)
        self._insert(cache, allocator, req_to_token_pool, seq_abc)

        seq_side = self._make_seq(900, 5)
        self._insert(cache, allocator, req_to_token_pool, seq_side)

        pre = self._swa_lru_order(cache)
        # Same segment layout as above.
        self.assertEqual(len(pre), 8)
        side_node, c_node, b_node, a_node = pre[0], pre[2], pre[4], pre[6]

        m = match_prefix(cache, seq_abc)
        self.assertEqual(m.device_prefix_len, len(seq_abc))

        post = self._swa_lru_order(cache)
        # Matching seq_abc refreshes only the window cushion (C's capped nodes)
        # to the MRU side; out-of-cushion ancestors B and A keep their order.
        self.assertIn(c_node, post[:2], f"C must be refreshed to MRU; post={post}")
        self.assertLess(
            post.index(side_node),
            post.index(b_node),
            "Side branch must NOT be pushed below ancestors after deep match",
        )
        self.assertLess(post.index(b_node), post.index(a_node), "B must stay above A")
        self.assertGreaterEqual(
            post.index(a_node),
            len(post) - 2,
            "Oldest out-of-cushion ancestor A must stay at the LRU-tail end",
        )
        cache.sanity_check()

    def test_swa_lru_old_ancestors_evict_first_under_pressure(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 8)
        seq_ab = seq_a + self._make_seq(100, 8)
        seq_abc = seq_ab + self._make_seq(200, 8)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_ab)
        self._insert(cache, allocator, req_to_token_pool, seq_abc)

        seq_side = self._make_seq(900, 5)
        self._insert(cache, allocator, req_to_token_pool, seq_side)

        m = match_prefix(cache, seq_abc)
        self.assertEqual(m.device_prefix_len, len(seq_abc))

        m_side_before = match_prefix(cache, seq_side)
        self.assertEqual(m_side_before.device_prefix_len, len(seq_side))

        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=self.cfg.page_size))

        m_side_after = match_prefix(cache, seq_side)
        self.assertEqual(
            m_side_after.device_prefix_len,
            len(seq_side),
            "Side branch SWA must survive eviction; oldest ancestors (A) "
            "should be evicted first under bounded SWA LRU refresh.",
        )
        cache.sanity_check()

    @requires(
        lambda cfg: cfg.has_swa and not cfg.has_mamba,
        "requires SWA without Mamba (mamba alters effective_cache_len)",
    )
    @requires_ps1_window4
    def test_swa_eager_eviction_on_unfinished_req(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        pre_len = 20
        tokens = self._make_seq(1, pre_len)
        req, _ = self._make_inflight_req(cache, allocator, req_to_token_pool, tokens)

        swa_avail_before = allocator.swa_attn_allocator.available_size()

        with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
            cache.checkpoint(req, up_to=req.extend_end)

        cushion = max(self.cfg.sliding_window_size, self.cfg.page_size)
        expected_evicted = (pre_len - 1) - cushion
        self.assertEqual(
            req.kv.get_evicted_seqlen(ComponentType.SWA),
            expected_evicted,
            f"swa_evicted_seqlen should advance to (pre_len-1) - cushion = "
            f"{expected_evicted}, got {req.kv.get_evicted_seqlen(ComponentType.SWA)}",
        )

        swa_avail_after = allocator.swa_attn_allocator.available_size()
        self.assertGreaterEqual(
            swa_avail_after - swa_avail_before,
            expected_evicted,
            f"SWA pool should have freed at least {expected_evicted} slots; "
            f"before={swa_avail_before}, after={swa_avail_after}",
        )

        cache.unlock(req.lock)
        cache.sanity_check()

    def test_swa_lru_fresh_leaf_cap_rebuilds_both_nodes(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        # A single fresh in-window leaf longer than the cushion is cap-split into
        # an older prefix + a one-window tail; both are rebuilt via SWARebuild.
        seq = self._make_seq(1, 8)
        self._insert(cache, allocator, req_to_token_pool, seq)

        order = self._swa_lru_order(cache)
        self.assertEqual(len(order), 2)
        tail, prefix = order[0], order[1]

        # Both nodes received an SWA value at apply time.
        tail_swa = node_device_value(cache, tail, ComponentType.SWA)
        prefix_swa = node_device_value(cache, prefix, ComponentType.SWA)
        self.assertIsNotNone(tail_swa)
        self.assertIsNotNone(prefix_swa)

        # The in-window tail leaf is more-MRU (order[0]) than its older prefix
        # parent (order[1]); the tail is capped to one cushion.
        cushion = self.cfg.sliding_window_size + self.cfg.page_size
        self.assertEqual(len(node_children(cache, tail)), 0)
        self.assertEqual(prefix, node_parent(cache, tail))
        self.assertLess(node_key_length(cache, tail), cushion)
        self.assertEqual(
            node_key_length(cache, tail) + node_key_length(cache, prefix), len(seq)
        )

        # Rebuilt SWA spans the whole leaf, and evictable SWA size matches it.
        self.assertEqual(len(tail_swa) + len(prefix_swa), len(seq))
        self.assertEqual(
            cache.tree_core.component_evictable_size(ComponentType.SWA), len(seq)
        )

        cache.sanity_check()

    @requires_swa_only
    @requires_ps1_window4
    def test_swa_eager_eviction_noop_when_within_window(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        cushion = self.cfg.sliding_window_size + self.cfg.page_size  # = 5
        pre_len = cushion  # exactly at the boundary, nothing slid out
        tokens = self._make_seq(1, pre_len)
        req, _ = self._make_inflight_req(cache, allocator, req_to_token_pool, tokens)

        with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
            cache.checkpoint(req, up_to=req.extend_end)

        self.assertEqual(
            req.kv.get_evicted_seqlen(ComponentType.SWA),
            0,
            "Nothing should be evicted when prefill fits inside the cushion",
        )

        cache.unlock(req.lock)
        cache.sanity_check()

    def test_aux_evict_full_locked_leaf_tombstones_aux_only(self):
        aux_types = [
            ct
            for ct in (ComponentType.SWA, ComponentType.MAMBA)
            if ct in self.cfg.components
        ]
        if not aux_types:
            self.skipTest("requires an auxiliary component")
        if len(aux_types) > 1:
            self.skipTest("single-aux case keeps cascade expectations precise")
        aux = aux_types[0]

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 1)
        self._insert(cache, allocator, req_to_token_pool, seq)

        match = match_prefix(cache, seq)
        node = match.last_device_node
        self.assertEqual(len(node_children(cache, node)), 0)
        self.assertIsNotNone(node_device_value(cache, node, ComponentType.FULL))
        self.assertIsNotNone(node_device_value(cache, node, aux))

        # Reach "FULL locked, aux unlocked" as production does: Mamba via the
        # decode-hold opt-out, SWA via early window release.
        aux_len = len(node_device_value(cache, node, aux))
        if aux == ComponentType.MAMBA:
            lock_result = cache.inc_lock_ref(
                node, skip_lock_components=(ComponentType.MAMBA,)
            )
        else:
            lock_result = cache.inc_lock_ref(node)
            self.assertGreater(node_device_lock_ref(cache, node, aux), 0)
            cache.release_swa(TreeLock(node, lock_result.to_dec_params()))
            self.assertIsNotNone(node_device_value(cache, node, aux))
        self.assertGreater(node_device_lock_ref(cache, node, ComponentType.FULL), 0)
        self.assertEqual(node_device_lock_ref(cache, node, aux), 0)
        self.assertFalse(cache.tree_core.is_device_evictable_leaf(node))

        evict_params = EvictParams(num_tokens=0)
        if aux == ComponentType.SWA:
            evict_params.swa_num_tokens = aux_len
        else:
            evict_params.mamba_num = aux_len
        result = cache.evict(evict_params)

        self.assertEqual(result.num_tokens_evicted, 0)
        if aux == ComponentType.SWA:
            self.assertEqual(result.swa_num_tokens_evicted, aux_len)
        else:
            self.assertEqual(result.mamba_num_evicted, aux_len)
        self.assertIsNotNone(node_device_value(cache, node, ComponentType.FULL))
        self.assertIsNone(node_device_value(cache, node, aux))
        self.assertFalse(cache.tree_core.is_node_in_device_lru(node, aux))

        cache.dec_lock_ref(
            node,
            lock_result.to_dec_params(),
            skip_swa=(aux == ComponentType.SWA),
        )
        cache.sanity_check()

    def test_evict_cascade_parent_becomes_d_leaf(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 2)
        leaf = base + self._make_seq(500, 2)
        self._insert(cache, allocator, req_to_token_pool, base)
        self._insert(cache, allocator, req_to_token_pool, leaf)

        m_base = match_prefix(cache, base)
        lock_result = cache.inc_lock_ref(m_base.last_device_node)

        result = cache.evict(EvictParams(num_tokens=len(leaf)))
        cache.sanity_check()

        cache.dec_lock_ref(
            m_base.last_device_node,
            lock_result.to_dec_params(),
        )
        self.assertTrue(
            cache.tree_core.is_device_evictable_leaf(m_base.last_device_node)
        )
        cache.sanity_check()

    def test_evict_iterative_tombstone_cleanup(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size
        chain = self._make_seq(1, 6)
        self._insert(cache, allocator, req_to_token_pool, chain[: 2 * ps])
        self._insert(cache, allocator, req_to_token_pool, chain[: 4 * ps])
        self._insert(cache, allocator, req_to_token_pool, chain)

        initial_evictable = cache.full_evictable_size()
        self.assertGreater(initial_evictable, 0)

        result = cache.evict(EvictParams(num_tokens=initial_evictable * 2))
        self.assertGreaterEqual(result.num_tokens_evicted, initial_evictable)
        self.assertEqual(cache.full_evictable_size(), 0)
        if self.cfg.has_mamba:
            self.assertEqual(cache.mamba_evictable_size(), 0)
        if self.cfg.has_swa:
            self.assertEqual(cache.swa_evictable_size(), 0)
        self.assertEqual(len(node_children(cache, cache.root_node_handle())), 0)
        cache.sanity_check()

    def test_evict_respects_lru_order(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size
        seq_old = self._make_seq(1, 2)
        seq_new = self._make_seq(500, 2)

        self._insert(cache, allocator, req_to_token_pool, seq_old)
        self._insert(cache, allocator, req_to_token_pool, seq_new)

        match_prefix(cache, seq_new)

        cache.evict(EvictParams(num_tokens=len(seq_old)))

        m_old = match_prefix(cache, seq_old)
        m_new = match_prefix(cache, seq_new)
        self.assertEqual(m_old.device_prefix_len, 0)
        self.assertEqual(m_new.device_prefix_len, len(seq_new))
        cache.sanity_check()

    @requires(
        lambda cfg: cfg.components == (ComponentType.FULL,),
        "priority policy ordering is covered on Full-only configs",
    )
    def test_evict_respects_priority_policy(self):
        priority_cfg = replace(self.cfg, eviction_policy="priority")
        cache, allocator, req_to_token_pool = build_fixture(priority_cfg)
        seq_high = self._make_seq(1, 2)
        seq_low = self._make_seq(500, 2)

        self._insert(cache, allocator, req_to_token_pool, seq_high, priority=10)
        self._insert(cache, allocator, req_to_token_pool, seq_low, priority=0)

        cache.evict(EvictParams(num_tokens=len(seq_low)))

        m_high = match_prefix(cache, seq_high)
        m_low = match_prefix(cache, seq_low)
        self.assertEqual(m_high.device_prefix_len, len(seq_high))
        self.assertEqual(m_low.device_prefix_len, 0)
        cache.sanity_check()

    def test_evict_result_accounting_matches_actual(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seqs = [self._make_seq(i * 100, 2) for i in range(5)]
        for s in seqs:
            self._insert(cache, allocator, req_to_token_pool, s)

        before = cache.full_evictable_size()
        half = before // 2
        result = cache.evict(EvictParams(num_tokens=half))
        after = cache.full_evictable_size()
        self.assertGreaterEqual(result.num_tokens_evicted, half)
        self.assertEqual(result.num_tokens_evicted, before - after)

        remaining = cache.full_evictable_size()
        cache.evict(EvictParams(num_tokens=remaining * 2))
        self.assertEqual(cache.full_evictable_size(), 0)
        cache.sanity_check()

    @requires_swa
    def test_swa_evict_internal_tombstone(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 3)
        leaf = base + self._make_seq(500, 3)
        self._insert(cache, allocator, req_to_token_pool, base)
        self._insert(cache, allocator, req_to_token_pool, leaf)

        swa_before = cache.swa_evictable_size()
        result = cache.evict(EvictParams(num_tokens=0, swa_num_tokens=swa_before * 2))
        self.assertEqual(cache.swa_evictable_size(), 0)
        cache.sanity_check()

    @requires(
        lambda cfg: cfg.page_size == 1, "page_size=1 keeps the split boundary mid-node"
    )
    def test_full_kv_hit_length_counts_the_split_fragment(self):
        """A mid-node partial match splits the node; the matched fragment still
        counts toward full_kv_hit_length (independent of component validators)."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 9)))
        result = match_prefix(cache, [1, 2, 3, 4, 99])
        self.assertEqual(result.full_kv_hit_length, 4)

    def test_zero_match_result_carries_node_id_handles(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size
        min_tokens = 2 * ps
        if self.cfg.has_swa:
            min_tokens = max(min_tokens, self.cfg.sliding_window_size + ps)
        seq = self._make_seq(1, (min_tokens + ps - 1) // ps)
        self._insert(cache, allocator, req_to_token_pool, seq)
        result = match_prefix(cache, seq)

        zeroed = zero_match_result(cache, result)

        self.assertEqual(zeroed.device_prefix_len, 0)
        # zeroed results must carry the root's NodeId, not the raw node
        self.assertEqual(zeroed.best_match_node, cache.root_node_handle())
        # The env-gated force-miss path passes the request's extra key.
        salted = zero_match_result(cache, result, extra_key="salt")
        self.assertEqual(salted.best_match_node, cache.root_node_handle())
        self.assertEqual(zeroed.last_device_node, cache.root_node_handle())
        self.assertEqual(zeroed.last_host_node, cache.root_node_handle())
        # and the handles must work with the NodeId-based lock APIs
        lock = cache.inc_lock_ref(zeroed.best_match_node)
        cache.dec_lock_ref(zeroed.best_match_node, lock.to_dec_params())

    @requires_full_swa_ps1_window4
    def test_swa_branch_insert_releases_forward_overshoot(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        branching_seqlen = 8
        forward_len = 20
        tokens = self._make_seq(1, forward_len)
        req, kv_indices = self._make_inflight_req(
            cache, allocator, req_to_token_pool, tokens
        )
        req.swa_branching_seqlen = branching_seqlen

        with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
            cache.checkpoint(req, up_to=forward_len)

        # Only the branch point is cached; the forward past it is released.
        self.assertEqual(
            match_prefix(cache, tokens).device_prefix_len, branching_seqlen
        )
        expected_evicted = forward_len - 1 - self.cfg.sliding_window_size
        self.assertEqual(req.kv.get_evicted_seqlen(ComponentType.SWA), expected_evicted)

        # The cached branch keeps SWA for its last window; the forward overshoot
        # past it is released, and the request keeps its own live window.
        branch_freed = branching_seqlen - 1 - self.cfg.sliding_window_size
        mapping = allocator.full_to_swa_index_mapping
        self.assertEqual(
            torch.count_nonzero(mapping[kv_indices[:branch_freed]]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(
                mapping[kv_indices[branch_freed:branching_seqlen]]
            ).item(),
            branching_seqlen - branch_freed,
        )
        self.assertEqual(
            torch.count_nonzero(
                mapping[kv_indices[branching_seqlen:expected_evicted]]
            ).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(mapping[kv_indices[expected_evicted:]]).item(),
            forward_len - expected_evicted,
        )
        cache.unlock(req.lock)
        cache.sanity_check()


define_config_test_classes(UnifiedRadixCacheCoreSuite, globals())


class UnifiedLRUListBoundedRefreshTest(
    UnifiedLRUListBoundedRefreshFixtures, CustomTestCase
):
    components = (ComponentType.FULL, ComponentType.SWA)

    def test_bounded_refresh_skips_non_included(self):
        root, [a, b, c, d] = self._build_chain([2, 2, 2, 2])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        for n in (a, c, d):  # b excluded from LRU (simulated tombstone)
            lru.insert_mru(n)
        self.assertEqual(self._lru_order(lru), [d, c, a])

        included = {a, c, d}
        lru.reset_node_and_window_ancestors_mru(
            d, root, window_size=5, should_include=lambda n: n in included
        )
        # A is not visited because the walk stops at B.
        self.assertEqual(self._lru_order(lru), [d, c, a])

    def test_bounded_refresh_visits_only_until_window_filled(self):
        root, [a, b, c, d] = self._build_chain([3, 3, 3, 3])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        for n in (d, c, b, a):
            lru.insert_mru(n)
        self.assertEqual(self._lru_order(lru), [a, b, c, d])

        # D and C move to the head in that order; A and B keep their relative
        # positions.
        lru.reset_node_and_window_ancestors_mru(
            d, root, window_size=5, should_include=lambda _n: True
        )
        self.assertEqual(self._lru_order(lru), [d, c, a, b])

    def test_bounded_refresh_stops_at_root(self):
        root, [a, b] = self._build_chain([1, 1])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        for n in (a, b):
            lru.insert_mru(n)
        self.assertEqual(self._lru_order(lru), [b, a])

        # Big window — refresh walks A and B both, then hits root and stops.
        lru.reset_node_and_window_ancestors_mru(
            b, root, window_size=1000, should_include=lambda _n: True
        )
        self.assertEqual(self._lru_order(lru), [b, a])

    def test_bounded_refresh_window_zero_is_noop(self):
        root, [a, b, c] = self._build_chain([2, 2, 2])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        for n in (a, b, c):
            lru.insert_mru(n)
        before = self._lru_order(lru)
        lru.reset_node_and_window_ancestors_mru(
            c, root, window_size=0, should_include=lambda _n: True
        )
        self.assertEqual(self._lru_order(lru), before)


class TestUnifiedMambaLRUMatchRefresh(
    UnifiedMambaLRUMatchRefreshFixtures, CustomTestCase
):
    """A hit refreshes only best_match_node's mamba state, so pressure evicts whole
    cold sessions rather than intermediate states."""

    cfg = CacheConfig(page_size=1, components=(ComponentType.FULL, ComponentType.MAMBA))

    def test_match_refreshes_only_used_node(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        def insert(tokens):
            value = allocator.alloc(len(tokens))
            req = self._make_req(req_to_token_pool)
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)),
                    value=value[: len(tokens)],
                    mamba_value=req.kv.mamba_pool_idx.unsqueeze(0),
                )
            )

        def match_leaf(tokens):
            return match_prefix(cache, tokens).best_match_node

        # Two independent sessions, each a 2-node mamba chain:
        #   root -> a1 -> b1  and  root -> a2 -> b2
        insert([1, 2, 3])
        insert([1, 2, 3, 4, 5, 6])
        insert([7, 8, 9])
        insert([7, 8, 9, 10, 11, 12])

        b1 = match_leaf([1, 2, 3, 4, 5, 6])
        a1 = cache.tree_core.get_parent_node_id(b1)
        b2 = match_leaf([7, 8, 9, 10, 11, 12])
        a2 = cache.tree_core.get_parent_node_id(b2)
        # Session 2 matched last, so session 1's ancestor a1 is older than a2.
        order = self._mamba_lru_mru_to_lru(cache)
        self.assertGreater(order.index(a1), order.index(a2))

        # Re-access session 1: only its consumed leaf b1 moves to MRU; ancestor a1
        # must stay put -- whole-chain reset would bump a1 ahead of a2.
        self.assertEqual(match_leaf([1, 2, 3, 4, 5, 6]), b1)
        order = self._mamba_lru_mru_to_lru(cache)
        self.assertEqual(order[0], b1)
        self.assertGreater(order.index(a1), order.index(a2))


class TestMambaCheckpointGrid(MambaCheckpointGridFixtures, CustomTestCase):
    """A donated mamba checkpoint is only reusable at a depth the tree can name.

    ``tree_page_size`` simulates DCP: it widens the page the tree allocates on
    while ``page_size`` and the mamba chunk grid stay where they are, which is
    exactly the split that lets a checkpoint land between two node boundaries.
    """

    cfg = CacheConfig(
        page_size=32,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        kv_size=1024,
        max_context_len=1024,
    )

    def test_grid_follows_the_widened_tree_page(self):
        # lcm(chunk=64, tree page=96) is 192. Chunk-only alignment would
        # incorrectly report 256, which is not a radix-node boundary.
        self.assertEqual(
            self._branching_seqlen(tree_page_size=96, full_hit_length=288), 192
        )

    def test_grid_is_the_chunk_size_without_widening(self):
        self.assertEqual(
            self._branching_seqlen(tree_page_size=32, full_hit_length=160), 128
        )


class TestMambaFinishedOvershootCheckpoint(
    MambaFinishedOvershootCheckpointFixtures, CustomTestCase
):
    _rid = 0
    cfg = CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        kv_size=64,
        max_context_len=64,
    )

    def test_previous_checkpoint_or_no_donation(self):
        for previous_len, expected_len in ((8, 8), (None, 0), (12, 0)):
            with self.subTest(previous_len=previous_len):
                cache, allocator, pool = build_fixture(
                    self.cfg, mamba_cache_chunk_size=4
                )
                req, tokens = self._build_req(allocator, pool, previous_len)
                previous_slot = req.kv.mamba_ping_pong_track_buffer[
                    req.kv.mamba_next_track_idx
                ].clone()
                req.last_node = cache.root_node_handle()
                finish_req(cache, req, 11)
                match = match_prefix(cache, tokens)
                self.assertEqual(match.device_prefix_len, expected_len)
                if expected_len:
                    node_value = cache.tree_core.get_component_device_value(
                        match.last_device_node, ComponentType.MAMBA
                    )
                    self.assertTrue(
                        torch.equal(
                            node_value.reshape(-1),
                            previous_slot.reshape(-1),
                        )
                    )
                else:
                    self.assertIsNone(req.kv.mamba_pool_idx)
                cache.sanity_check()


class TestUnifiedRadixCacheInt8MambaCheckpoint(
    UnifiedRadixCacheInt8MambaCheckpointFixtures, CustomTestCase
):
    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        enable_int8_mamba_checkpoint=True,
        mamba_cache_size=8,
        kv_size=64,
        max_context_len=64,
    )

    def test_finished_req_stores_radix_mamba_state_in_int8_pool(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        ckpt_pool = req_to_token_pool.mamba_ckpt_pool
        self.assertIsNotNone(ckpt_pool)
        active_initial = req_to_token_pool.mamba_allocator.available_size()
        ckpt_initial = ckpt_pool.available_size()
        tokens = [1, 2, 3, 4]

        self._finished_req(cache, allocator, req_to_token_pool, tokens)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), active_initial
        )
        self.assertEqual(ckpt_pool.available_size(), ckpt_initial - 1)
        self.assertEqual(cache.mamba_evictable_size(), 1)

        self._finished_req(cache, allocator, req_to_token_pool, tokens)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), active_initial
        )
        self.assertEqual(ckpt_pool.available_size(), ckpt_initial - 1)
        self.assertEqual(cache.mamba_evictable_size(), 1)

        result = cache.evict(EvictParams(mamba_num=1))
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), active_initial
        )
        self.assertEqual(ckpt_pool.available_size(), ckpt_initial)
        self.assertEqual(cache.mamba_evictable_size(), 0)


def _component_with_cache(component_type, cache):
    component = object.__new__(COMPONENT_REGISTRY[component_type])
    component.cache = cache
    component.tree_core = cache.tree_core
    return component


class TestUnifiedRadixCacheActionRouting(CustomTestCase):
    def test_retraction_uses_engine_transfer_index_domains(self):
        cache = object.__new__(UnifiedRadixCache)
        cache.is_swa_enabled = True
        cache._sliding_window_size = 3
        cache.tree_core = mock.Mock(page_size=2)
        cache.page_size = 2
        cache.sidecar_pool_specs = ()
        cache.req_to_token_pool = mock.Mock()
        cache.req_to_token_pool.req_to_token = torch.tensor(
            [[1, 2, 3, 4, 5, 0]], dtype=torch.int64
        )
        cache.token_to_kv_pool_allocator = mock.Mock()
        cache.token_to_kv_pool_allocator.translate_swa_indices_for_transfer.side_effect = (
            lambda indices: indices + 100
        )
        req = mock.Mock(rid="req", seqlen=6)
        req.kv.req_pool_idx = 0

        full_indices, transfers = cache._retraction_device_transfers(req)

        self.assertTrue(torch.equal(full_indices, torch.tensor([1, 2, 3, 4, 5, 6])))
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].name, PoolName.SWA)
        self.assertTrue(
            torch.equal(
                transfers[0].device_indices,
                torch.tensor([103, 104, 105, 106]),
            )
        )
        cache.token_to_kv_pool_allocator.translate_swa_indices_for_transfer.assert_called_once()
        cache.token_to_kv_pool_allocator.translate_kv_indices_for_transfer.assert_not_called()
        cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa.assert_not_called()

    def test_backup_publish_node_ids_collects_component_nodes_once(self):
        comp_xfers = {
            ComponentType.SWA: [PoolTransfer(name=PoolName.SWA, nodes_to_load=[3, 4])],
            ComponentType.MAMBA: [
                PoolTransfer(name=PoolName.MAMBA, nodes_to_load=[4, 5])
            ],
        }

        self.assertEqual(
            UnifiedRadixCache._backup_publish_node_ids(7, comp_xfers),
            [3, 4, 5, 7],
        )

    def test_apply_component_action_swa_recover_on_full_locked(self):
        cache = mock.MagicMock()
        alloc = cache.token_to_kv_pool_allocator
        kept_full = torch.tensor([1, 2], dtype=torch.int64)
        incoming_full = torch.tensor([3, 4], dtype=torch.int64)
        swa_value = torch.tensor([7, 8], dtype=torch.int64)
        alloc.translate_swa_indices_for_transfer.return_value = swa_value
        component_with_cache(ComponentType.SWA, cache).apply_component_action(
            RecoverSWAWithLockedFull(
                node_id=5,
                kept_full=kept_full,
                incoming_full=incoming_full,
            ),
        )
        # keep the locked full, remap it onto the incoming full's SWA translation
        alloc.translate_swa_indices_for_transfer.assert_called_once_with(incoming_full)
        alloc.set_full_to_swa_mapping.assert_called_once_with(kept_full, swa_value)
        # the incoming full's stale mapping is cleared, then its slot freed (full-only)
        alloc.clear_full_to_swa_mapping.assert_called_once_with(incoming_full)
        alloc.free_full_segment.assert_called_once_with(incoming_full, start_pos=0)
        # Never by indexing the tensor: the unified composite has no
        # `full_to_swa_index_mapping` to index into.
        alloc.full_to_swa_index_mapping.__setitem__.assert_not_called()
        # not the inner allocator (skips the free-group defer) and not both halves
        alloc.full_attn_allocator.free.assert_not_called()
        alloc.free.assert_not_called()
        cache.tree_core.set_component_device_value.assert_called_once_with(
            5, ComponentType.SWA, swa_value
        )

    def test_eviction_free_failure_still_drains_actions_in_order(self):
        cache = object.__new__(UnifiedRadixCache)
        first = UpdatePendingBackupOnNodeSplit(
            ack_id=1, old_node_id=1, new_node_id=2, new_child_node_id=1
        )
        second = UpdatePendingBackupOnNodeSplit(
            ack_id=2, old_node_id=3, new_node_id=4, new_child_node_id=3
        )
        result = EvictDeviceLeafResult()
        result.cache_actions.extend([first, second])
        error = RuntimeError("allocator free failed")
        calls = []

        def fail_free(device_frees, host_frees):
            calls.append("free")
            raise error

        cache._free_values = mock.Mock(side_effect=fail_free)
        cache._apply_cache_action = mock.Mock(side_effect=calls.append)
        with self.assertRaises(RuntimeError) as raised:
            cache._apply_eviction_result(result)

        self.assertIs(raised.exception, error)
        cache._free_values.assert_called_once_with(
            result.device_frees, result.host_frees
        )
        self.assertEqual(calls, ["free", first, second])
        self.assertFalse(result.cache_actions)
        cache._apply_cache_actions(result.cache_actions)
        self.assertEqual(cache._apply_cache_action.call_count, 2)

    def test_match_completion_follows_actions_and_preserves_component_order(self):
        # Match completion reads the disaggregation role from the config bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.session = mock.Mock()
        cache.session.try_match_prefix.return_value = None
        cache.tree_core = mock.Mock()
        order = mock.Mock()
        cache.linker = order.linker
        cache.token_to_kv_pool_allocator = order.allocator
        cache.components = {
            ComponentType.SWA: order.swa,
            ComponentType.FULL: order.full,
            ComponentType.MAMBA: order.mamba,
        }

        def complete_match(params, result):
            return UnifiedRadixCache._finalize_match_result(cache, params, finalized)

        order.complete.side_effect = complete_match
        cache._finalize_match_result = order.complete
        indices = torch.tensor([4, 5])
        for names in (("swa", "full", "mamba"), ("mamba", "swa", "full")):
            cache._components_tuple = tuple(getattr(order, name) for name in names)
            for req in (SimpleNamespace(rid="first"), SimpleNamespace(rid="second")):
                with self.subTest(components=names, req=req.rid):
                    order.reset_mock()
                    params = MatchPrefixParams(
                        key=RadixKey(array("q", [1, 2])), req=req
                    )
                    split = UpdatePendingBackupOnNodeSplit(None, 1, 2, 1)
                    result = MatchResult(
                        indices.numel(),
                        1,
                        1,
                        1,
                        cache_actions=[split, FreeDeviceKV([indices])],
                        last_visited_node_id=1,
                        full_device_match_len=1,
                    )
                    finalized = result._replace(device_prefix_len=1)
                    cache.tree_core.match_prefix.return_value = result
                    for component in cache.components.values():
                        component.finalize_match_result_in_cache.return_value = (
                            finalized
                        )
                    order.linker.match.return_value = finalized

                    self.assertIs(cache.match_prefix(params), finalized)
                    expected = [
                        mock.call.full.apply_component_action(split),
                        mock.call.allocator.free_segment(indices, start_pos=0),
                        mock.call.complete(params, result),
                    ]
                    expected.extend(
                        getattr(mock.call, name).finalize_match_result_in_cache(
                            params, finalized
                        )
                        for name in names
                    )
                    expected.append(mock.call.linker.match(params.key, req, finalized))
                    self.assertEqual(order.mock_calls, expected)
                    received = order.complete.call_args.args
                    self.assertIs(received[0], params)
                    self.assertIs(received[0].req, req)
                    self.assertIs(received[1], result)
                    self.assertEqual(finalized.last_visited_node_id, 1)
                    self.assertEqual(finalized.full_device_match_len, 1)
                    self.assertEqual(result.cache_actions, [])
                    cache._apply_cache_actions(result.cache_actions)
                    self.assertEqual(order.mock_calls, expected)

    def test_chained_replace_write_through_requires_list_order(self):
        # A pending node split twice in one walk emits two chained Replaces:
        # the second one's old_node_id only enters the publish list when the
        # first is applied, so list order is a hard contract.
        def make_cache():
            cache = mock.MagicMock()
            cache.ongoing_write_through = {7: _OngoingWriteThrough(10, None, [5, 10])}
            cache._replace_pending_write_through_node = lambda *args: (
                UnifiedRadixCache._replace_pending_write_through_node(cache, *args)
            )
            cache.components = {ComponentType.FULL: mock.Mock()}
            return cache

        def apply(cache, action):
            UnifiedRadixCache._apply_cache_action(cache, action)

        first = UpdatePendingBackupOnNodeSplit(
            ack_id=7, old_node_id=10, new_node_id=11, new_child_node_id=10
        )
        second = UpdatePendingBackupOnNodeSplit(
            ack_id=7, old_node_id=11, new_node_id=12, new_child_node_id=11
        )

        cache = make_cache()
        apply(cache, first)
        apply(cache, second)
        self.assertEqual(
            cache.ongoing_write_through[7].publish_node_ids, [5, 12, 11, 10]
        )
        cache.linker.replace_pending_offload_node.assert_has_calls(
            [mock.call(7, 10, [11, 10]), mock.call(7, 11, [12, 11])]
        )
        receiver = cache.components[ComponentType.FULL].apply_component_action
        self.assertEqual(receiver.call_args_list, [mock.call(first), mock.call(second)])
        self.assertEqual(cache.linker.replace_pending_offload_node.call_count, 2)

        # Reversed order silently drops the second fragment, so emission order
        # must be preserved end to end.
        cache = make_cache()
        apply(cache, second)
        apply(cache, first)
        self.assertEqual(cache.ongoing_write_through[7].publish_node_ids, [5, 11, 10])

    def test_split_action_without_ack_still_reaches_full(self):
        cache = mock.MagicMock()
        component = _component_with_cache(ComponentType.FULL, cache)
        component.apply_component_action = mock.Mock(
            wraps=component.apply_component_action
        )
        cache.components = {ComponentType.FULL: component}
        action = UpdatePendingBackupOnNodeSplit(
            ack_id=None, old_node_id=10, new_node_id=11, new_child_node_id=10
        )
        UnifiedRadixCache._apply_cache_action(cache, action)
        component.apply_component_action.assert_called_once_with(action)
        cache._replace_pending_write_through_node.assert_not_called()
        cache.linker.replace_pending_offload_node.assert_not_called()


class TestSWADraftHistory(SWADraftHistoryFixtures, CustomTestCase):
    """A widened draft needs SWA history before the target window. A running
    checkpoint must lock that history, and an incomplete cached window must
    not be reused even when the target alone could read it.
    """

    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
        sliding_window_size=511,
        draft_swa_window=516,
        is_eagle=True,
        kv_size=2048,
        max_context_len=2048,
        num_layers=2,
        full_attention_layer_ids=(0,),
        head_num=1,
        head_dim=8,
        enable_mamba_extra_buffer=True,
    )

    def test_checkpoint_protects_widened_draft_history_under_eviction(self):
        cache, allocator, pool = build_fixture(self.cfg, mamba_cache_chunk_size=64)
        req = self._request(pool)
        req.last_node = cache.root_node_handle()
        kv = allocator.alloc(1024)
        pool.write((req.kv.req_pool_idx, slice(0, 1024)), kv)
        cache.checkpoint(req, up_to=1024)
        self.addCleanup(lambda: cache.unlock(req.lock))

        # Mamba disables bigram keys: this checkpoint is exactly at 1024.
        self.assertEqual(req.kv.cache_protected_len, 1024)
        # A three-depth draft with five front rows and three warmup rows has
        # its first real query at 1022. Its window needs key 511, which lies
        # before the target-only lock [513, 1024).
        needed = kv[1024 - self.cfg.draft_swa_window : 1024]
        mapping = allocator.full_to_swa_index_mapping
        before = mapping[needed].clone()
        self.assertTrue(bool((before > 0).all()))
        cache.evict(EvictParams(swa_num_tokens=1))
        torch.testing.assert_close(mapping[needed], before)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(req.origin_input_ids))
        )
        self.assertEqual(match.device_prefix_len, 1024)

        # Decode advances past the last recurrent checkpoint. Its complete
        # draft history must remain until a newer checkpoint replaces it.
        cache.evict_sliding_windows(req, 1040)
        torch.testing.assert_close(mapping[needed], before)
        cache.sanity_check()

    def test_prefix_match_rejects_target_only_swa_history(self):
        for retained, expected_match in ((511, 0), (516, 1024)):
            with self.subTest(retained=retained):
                cache, allocator, pool = build_fixture(
                    self.cfg, mamba_cache_chunk_size=64
                )
                req = self._request(pool)
                kv = allocator.alloc(1024)
                cutoff = 1024 - retained
                allocator.free_swa_segment(kv[:cutoff], start_pos=0)
                cache.insert(
                    InsertParams(
                        key=RadixKey(req.origin_input_ids),
                        value=kv,
                        mamba_value=req.kv.mamba_pool_idx.unsqueeze(0),
                        component_evicted_seqlens={ComponentType.SWA: cutoff},
                    )
                )
                result = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(req.origin_input_ids))
                )
                self.assertEqual(result.device_prefix_len, expected_match)
                cache.sanity_check()


class TestSWAWindowUnderBigramKey(SWAWindowUnderBigramKeyFixtures, CustomTestCase):
    """checkpoint must leave its leaf with a full live SWA window, else
    cache_protected_len stalls and the next insert frees tree-owned KV. EAGLE
    bigram keys hold one entry fewer than seq_len, so measure the frontier
    against the key."""

    cfg = CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=7,
        is_eagle=True,
        kv_size=256,
        max_context_len=64,
    )

    def test_match_after_insert_reaches_the_new_leaf(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        page_size = self.cfg.page_size
        # Page-aligned length is the shape that costs the bigram key a page.
        seq_len = 4 * page_size

        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([req])
        tokens = list(range(1, seq_len + 1))
        req.origin_input_ids = tokens
        req.output_ids = []
        req.full_untruncated_fill_ids = array("q", tokens)
        req.extend_end = len(req.full_untruncated_fill_ids)
        kv_indices = self._alloc_paged(allocator, seq_len)
        req_to_token_pool.write((req.kv.req_pool_idx, slice(0, seq_len)), kv_indices)
        req.kv.kv_committed_len = seq_len
        req.last_node = cache.root_node_handle()
        req.kv.cache_protected_len = 0
        req.lock = None
        req.extra_key = None

        with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
            cache.checkpoint(req, up_to=req.extend_end)

        boundary = (seq_len - 1) // page_size * page_size
        self.assertGreaterEqual(
            boundary - req.kv.get_evicted_seqlen(ComponentType.SWA),
            self.cfg.sliding_window_size,
            f"leaf ending at {boundary} keeps only "
            f"{boundary - req.kv.get_evicted_seqlen(ComponentType.SWA)} live SWA tokens against a "
            f"{self.cfg.sliding_window_size} window",
        )
        self.assertEqual(
            req.kv.cache_protected_len,
            boundary,
            "the match after the insert must reach the leaf the insert created",
        )

        cache.unlock(req.lock)
        cache.sanity_check()

    def test_finish_after_decode_eviction_frees_only_the_requests_pages(self):
        # Decode evicts at pre_len = seq_len - 1. When the last verify then commits
        # one token, the finished insert ends at page_floor(pre_len): the frontier
        # must sit a full window below it although the window is not page-aligned.
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        page_size = self.cfg.page_size
        full_total = allocator.full_available_size()
        swa_total = allocator.swa_available_size()
        prompt_len, owned = 4 * page_size, 6 * page_size
        tokens = list(range(1, owned + 1))

        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=array("q", tokens[:prompt_len]),
            sampling_params=SamplingParams(
                temperature=0, max_new_tokens=owned - prompt_len
            ),
        )
        req_to_token_pool.alloc([req])
        req.output_ids = []
        req.full_untruncated_fill_ids = array("q", tokens[:prompt_len])
        req.extend_end = prompt_len
        kv_indices = self._alloc_paged(allocator, owned)
        req_to_token_pool.write((req.kv.req_pool_idx, slice(0, owned)), kv_indices)
        req.kv.kv_committed_len = prompt_len
        req.kv.kv_allocated_len = owned
        req.last_node = cache.root_node_handle()
        req.kv.cache_protected_len = 0
        req.lock = None
        req.extra_key = None
        cache.checkpoint(req, up_to=prompt_len)

        req.output_ids = tokens[prompt_len:]
        req.kv.kv_committed_len = owned
        cache.evict_sliding_windows(req, owned - 1)
        insert_boundary = (owned - 1) // page_size * page_size
        self.assertGreaterEqual(
            insert_boundary - req.kv.get_evicted_seqlen(ComponentType.SWA),
            self.cfg.sliding_window_size,
        )
        req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        release_kv_cache(req, cache, checkpoint=True)

        # Each page is back in the pool or held by the tree, never both.
        self.assertEqual(
            allocator.full_available_size() + cache.full_evictable_size(), full_total
        )
        self.assertEqual(
            allocator.swa_available_size() + cache.swa_evictable_size(), swa_total
        )
        cache.sanity_check()


class TestUnifiedRadixCacheDecoderSwaBoundedReplayHoldBack(
    UnifiedRadixCacheDecoderSwaBoundedReplayHoldBackFixtures, CustomTestCase
):
    def test_match_stops_one_window_short_under_bounded_replay(self):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=128,
        )
        for bounded_replay, matched in ((False, 200), (True, 260 - 128)):
            with self.subTest(decoder_swa_bounded_replay=bounded_replay):
                cache, allocator, _ = build_fixture(
                    replace(cfg, decoder_swa_bounded_replay=bounded_replay)
                )
                cached = list(range(1, 201))
                value = allocator.alloc(len(cached))
                self.assertIsNotNone(value)
                cache.insert(
                    InsertParams(
                        key=RadixKey(array("q", cached)), value=value[: len(cached)]
                    )
                )
                # 200 of 260 cached: a plain hit would leave a 60-row extend
                prompt = cached + list(range(1001, 1061))
                self.assertEqual(
                    cache.swa_reprefill_tail_tokens(), 128 if bounded_replay else 0
                )
                self.assertEqual(self._matched_len(cache, prompt), matched)


if __name__ == "__main__":
    unittest.main()
