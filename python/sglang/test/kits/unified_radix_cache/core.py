from array import array
from unittest import mock

from sglang.srt.managers.schedule_batch import (
    Req,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_radix_cache import (
    COMPONENT_REGISTRY,
    UnifiedLRUList,
    UnifiedTreeNode,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_device_value,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.mem_cache_utils import finish_req


def session_radix_cache_test_values() -> tuple[bool, ...]:
    return False, True


class FakeFullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: True

    def redistribute_on_node_split(self, new_parent, child):
        return None

    def evict_component(
        self,
        node,
        device_frees,
        host_frees,
        cache_actions,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        return 0, 0

    def acquire_component_lock(self, node, result):
        return result

    def release_component_lock(self, node, params):
        return None

    def _evict_device_start(self, request_cnt) -> None:
        pass

    def _evict_device_next_node(self, tracker, device_frees, host_frees, cache_actions):
        return None

    def _evict_device_end(self) -> None:
        pass

    def _dec_session_coverage(self, session_id, leaf) -> None:
        pass

    def _advance_session_coverage(self, session_id, leaf, old_ancestor) -> None:
        pass

    def _recede_session_coverage(self, session_id, leaf, fallback) -> None:
        pass


class PagedFullComponent(FullComponent):
    """FULL component whose device values are not backed one-to-one by pool
    rows: a value lists two entries per row, and the component can hand rows
    of a node back to the allocator while the value stays on device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows_handed_back: dict[int, int] = {}

    def reclaimable_value_len(self, node):
        return self.value_len(node) // 2 - self.rows_handed_back.get(node.id, 0)


def component_with_cache(component_type, cache):
    """A registry component instance bound to a (mock) cache and its tree_core."""
    component = object.__new__(COMPONENT_REGISTRY[component_type])
    component.cache = cache
    component.tree_core = cache.tree_core
    return component


class DisabledUnifiedRadixCacheFixtures:
    def _params(self, disable):
        return CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=2,
                max_context_len=8,
                device="cpu",
                enable_memory_saver=False,
            ),
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=disable,
            tree_components=(ComponentType.FULL,),
            component_registry_override={ComponentType.FULL: FakeFullComponent},
            enable_kv_cache_events=True,
            eviction_policy="lru",
            eviction_policy_config={"not_an_lru_option": 1},
        )


class UnifiedRadixAllocationEvictionRealComponentsFixtures:
    def _insert(self, cache, allocator, req_to_token_pool, tokens) -> None:
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        params = InsertParams(
            key=RadixKey(array("q", tokens)),
            value=value[: len(tokens)],
        )
        if cache.supports_mamba():
            req = Req(
                rid=f"mamba-{len(tokens)}",
                origin_input_text="",
                origin_input_ids=array("q"),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req_to_token_pool.alloc([req])
            params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
        cache.insert(params)

    def _build_internal_chain(self, component_type, enable_session_radix_cache):
        cfg = (
            CacheConfig(
                components=(ComponentType.FULL, ComponentType.SWA),
                sliding_window_size=128,
            )
            if component_type is ComponentType.SWA
            else CacheConfig(
                components=(ComponentType.FULL, ComponentType.MAMBA),
                mamba_cache_size=8,
            )
        )
        cache, allocator, req_to_token_pool = build_fixture(
            cfg, enable_session_radix_cache=enable_session_radix_cache
        )
        for length in (2, 4, 6):
            self._insert(
                cache,
                allocator,
                req_to_token_pool,
                list(range(1, length + 1)),
            )

        lru_node_ids = cache.tree_core.get_component_device_lru_node_ids(component_type)
        first = lru_node_ids[-1]
        second = lru_node_ids[-2]
        leaf = lru_node_ids[-3]
        self.assertFalse(cache.tree_core.is_device_evictable_leaf(first))
        self.assertFalse(cache.tree_core.is_device_evictable_leaf(second))
        self.assertTrue(cache.tree_core.is_device_evictable_leaf(leaf))
        for node_id in (first, second, leaf):
            self.assertIsNotNone(node_device_value(cache, node_id, component_type))
            self.assertIsNotNone(node_device_value(cache, node_id, ComponentType.FULL))
        return cache, first, second, leaf

    def _evict_for_alloc_after_first_drain(self, cache, component_type):
        capacity = {"available": 0}
        auxiliary_drains = {"count": 0}
        real_available_size = cache._component_available_size
        real_free_values = cache._free_values

        def available_size(requested_type):
            if requested_type is component_type:
                return capacity["available"]
            return real_available_size(requested_type)

        def free_values(device_frees, host_frees):
            freed_auxiliary = bool(device_frees.get(component_type))
            real_free_values(device_frees, host_frees)
            if freed_auxiliary:
                auxiliary_drains["count"] += 1
                capacity["available"] = self._SHORTFALL

        params = (
            EvictParams(swa_num_tokens=self._SHORTFALL)
            if component_type is ComponentType.SWA
            else EvictParams(mamba_num=self._SHORTFALL)
        )
        with (
            mock.patch.object(
                cache, "_component_available_size", side_effect=available_size
            ),
            mock.patch.object(cache, "_free_values", side_effect=free_values),
        ):
            result = cache.evict_for_alloc(params)
        return result, auxiliary_drains["count"]


class UnifiedRadixCacheCoreFixtures(UnifiedRadixCacheFixtures):
    def _swa_lru_order(self, cache):
        return cache.tree_core.get_component_device_lru_node_ids(ComponentType.SWA)

    def _swa_pinning_cfg_supported(self) -> bool:
        if not self.cfg.has_swa or self.cfg.has_mamba:
            return False
        cushion = self.cfg.sliding_window_size + self.cfg.page_size
        pages_per_node = 8
        side_pages = 5
        if pages_per_node * self.cfg.page_size < cushion:
            return False
        chain_inserts_pages = 4 * pages_per_node + side_pages
        if self.cfg.kv_size < chain_inserts_pages * self.cfg.page_size:
            return False
        return True


class UnifiedLRUListBoundedRefreshFixtures:
    def _make_node(self, key_len: int) -> UnifiedTreeNode:
        n = UnifiedTreeNode(self.components)
        n.key = RadixKey(list(range(key_len)))
        return n

    def _build_chain(self, key_lens: list[int]) -> tuple:
        root = self._make_node(0)
        nodes = []
        parent = root
        for kl in key_lens:
            n = self._make_node(kl)
            n.parent = parent
            nodes.append(n)
            parent = n
        return root, nodes

    def _lru_order(self, lru: UnifiedLRUList) -> list:
        pt = lru._pt
        out = []
        cur = lru.head.lru_next[pt]
        while cur is not lru.tail:
            out.append(cur)
            cur = cur.lru_next[pt]
        return out


class UnifiedMambaLRUMatchRefreshFixtures:
    def _mamba_lru_mru_to_lru(self, cache):
        return cache.tree_core.get_component_device_lru_node_ids(ComponentType.MAMBA)

    def _make_req(self, req_to_token_pool):
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([req])
        return req


class MambaCheckpointGridFixtures:
    def _branching_seqlen(self, *, tree_page_size: int, full_hit_length: int):
        cache, allocator, req_to_token_pool = build_fixture(
            self.cfg,
            tree_page_size=tree_page_size,
            mamba_cache_chunk_size=64,
        )
        prefix = list(range(1, tree_page_size + 1))
        tokens = list(range(1, full_hit_length + 1))
        for sequence in (prefix, tokens):
            value = allocator.alloc(len(sequence))
            self.assertIsNotNone(value)
            req = Req(
                rid=f"checkpoint-grid-{len(sequence)}",
                origin_input_text="",
                origin_input_ids=array("q"),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req_to_token_pool.alloc([req])
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", sequence)),
                    value=value,
                    mamba_value=req.kv.mamba_pool_idx.unsqueeze(0),
                )
            )

        leaf = match_prefix(cache, tokens).last_device_node
        cache.tree_core.set_component_device_value_raw(leaf, ComponentType.MAMBA, None)
        result = match_prefix(cache, tokens)
        self.assertEqual(result.full_kv_hit_length, full_hit_length)
        self.assertEqual(result.device_prefix_len, tree_page_size)
        return result.mamba_branching_seqlen


class MambaFinishedOvershootCheckpointFixtures:
    def _build_req(self, allocator, pool, previous_track_seqlen):
        tokens = list(range(12))
        req = UnifiedRadixCacheFixtures._make_req(self, pool)
        req.origin_input_ids = array("q", tokens)
        req.output_ids = array("q")
        req.extra_key = None
        req.swa_uuid_for_lock = None
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = 0
        req.kv.mamba_last_track_seqlen = len(tokens)
        req.kv.mamba_prev_track_seqlen = previous_track_seqlen
        req.kv.mamba_last_track_idx = 0
        req.kv.mamba_next_track_idx = pool.get_mamba_ping_pong_other_idx(0)
        indices = allocator.alloc(len(tokens))
        self.assertIsNotNone(indices)
        pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), indices)
        return req, tokens


class UnifiedRadixCacheInt8MambaCheckpointFixtures:
    def _make_req(self, req_to_token_pool, tokens):
        req = Req(
            rid="int8-mamba",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([req])
        req.output_ids = array("q")
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = 0
        req.lock = None
        req.extra_key = None
        req.kv.mamba_last_track_seqlen = len(tokens)
        return req

    def _finished_req(self, cache, allocator, req_to_token_pool, tokens):
        req = self._make_req(req_to_token_pool, tokens)
        kv_indices = allocator.alloc(len(tokens))
        self.assertIsNotNone(kv_indices)
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, len(tokens))), kv_indices
        )
        req.last_node = cache.root_node_handle()

        finish_req(cache, req, req.owned_kv_len())


class SWADraftHistoryFixtures:
    def _request(self, pool):
        tokens = array("q", range(1, 1025))
        req = Req(
            rid="widened-draft",
            origin_input_text="",
            origin_input_ids=tokens,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
        )
        pool.alloc([req])
        req.output_ids = []
        req.full_untruncated_fill_ids = tokens
        req.extend_end = len(tokens)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = 0
        req.kv.mamba_last_track_seqlen = len(tokens)
        req.kv.mamba_last_track_idx = 0
        req.kv.mamba_next_track_idx = pool.get_mamba_ping_pong_other_idx(0)
        req.lock = None
        req.extra_key = None
        return req


class SWAWindowUnderBigramKeyFixtures:
    def _alloc_paged(self, allocator, need_size):
        ps = self.cfg.page_size
        aligned = ((need_size + ps - 1) // ps) * ps
        full_indices = allocator.full_attn_allocator.alloc(aligned)
        swa_indices = allocator.swa_attn_allocator.alloc(aligned)
        self.assertIsNotNone(full_indices)
        self.assertIsNotNone(swa_indices)
        allocator.full_to_swa_index_mapping[full_indices] = swa_indices
        return full_indices[:need_size]


class UnifiedRadixCacheDecoderSwaBoundedReplayHoldBackFixtures:
    def _matched_len(self, cache, tokens) -> int:
        req = Req(
            rid="decoder-swa-bounded-replay",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.init_next_round_input(cache)
        return req.prefix_len
