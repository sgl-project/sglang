from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import InsertResult, MatchResult
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FullComponent:
    component_type = ComponentType.FULL
    participates_in_linker = True

    def __init__(self):
        self.phases = []

    def build_external_linker_transfer(self, phase, node, keys):
        self.phases.append(phase)
        if phase == LinkerTransferPhase.OFFLOAD:
            return PoolTransfer(
                name=PoolName.KV,
                keys=["offload"],
                device_indices=torch.tensor([0, 1]),
            )
        return PoolTransfer(
            name=PoolName.KV,
            keys=list(keys),
            device_indices=(
                torch.arange(len(keys) * 2, dtype=torch.int64)
                if phase == LinkerTransferPhase.LOAD
                else None
            ),
        )

    def update_external_linker_load(
        self,
        phase,
        req,
        full_transfer,
        transfer,
        prefix_len,
        **kwargs,
    ):
        return transfer


class _ExcludedSWA:
    component_type = ComponentType.SWA
    participates_in_linker = False
    reused_swa_is_trustworthy = False
    sliding_window_size = 128

    def build_external_linker_transfer(self, *args, **kwargs):
        raise AssertionError("excluded SWA component reached linker transfer loop")


class _IncludedSWA:
    component_type = ComponentType.SWA
    participates_in_linker = True

    def build_external_linker_transfer(self, phase, node, keys):
        return PoolTransfer(
            name=PoolName.SWA,
            keys=list(keys),
            device_indices=torch.arange(20, 20 + len(keys) * 2, dtype=torch.int64),
        )

    def update_external_linker_load(
        self,
        phase,
        req,
        full_transfer,
        transfer,
        prefix_len,
        **kwargs,
    ):
        if phase == ExternalLinkerLoadPhase.PREPARE:
            req.kv = ReqKvInfo(
                kv_allocated_len=prefix_len,
                swa_evicted_seqlen=prefix_len - 2,
            )
        return transfer


def test_deepseek_v4_swa_addressing_mode_tracks_layout():
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = False
    assert pool.swa_is_index_addressed

    pool._unified_kv = True
    assert not pool.swa_is_index_addressed


def test_swa_trust_and_linker_participation_follow_addressing_mode():
    component = SWAComponent.__new__(SWAComponent)
    component.swa_is_index_addressed = False
    assert not component.participates_in_linker
    assert not component.reused_swa_is_trustworthy

    component.swa_is_index_addressed = True
    assert component.participates_in_linker
    assert component.reused_swa_is_trustworthy


def test_tree_component_participates_in_linker_by_default():
    component = FullComponent.__new__(FullComponent)
    assert component.participates_in_linker


def test_untrusted_swa_enables_reprefill_without_a_tier_condition():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    component = SWAComponent.__new__(SWAComponent)
    component.sliding_window_size = 128
    component.swa_is_index_addressed = False
    cache.components = {ComponentType.SWA: component}
    assert cache.swa_reprefill_tail_tokens() == 128

    component.swa_is_index_addressed = True
    assert cache.swa_reprefill_tail_tokens() == 0

    cache.components = {}
    assert cache.swa_reprefill_tail_tokens() == 0


def test_untrusted_request_relative_swa_tombstone_does_not_gate_full_match():
    component = SWAComponent.__new__(SWAComponent)
    component.sliding_window_size = 128
    component.swa_is_index_addressed = False
    node = SimpleNamespace(
        component_data={
            ComponentType.SWA: SimpleNamespace(value=None, host_value=None)
        },
        backuped=False,
        evicted=False,
    )

    assert component.create_match_validator(match_device_only=True)(node)

    component.swa_is_index_addressed = True
    assert not component.create_match_validator(match_device_only=True)(node)


def test_linker_filters_nonparticipating_swa_from_all_transfer_phases():
    full = _FullComponent()
    swa = _ExcludedSWA()
    cache = SimpleNamespace(
        _components_tuple=(full, swa),
        components={ComponentType.FULL: full, ComponentType.SWA: swa},
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=MagicMock(),
        ),
        write_through_threshold=0,
        page_size=2,
    )
    backend = MagicMock()
    backend.lookup.return_value = [2]
    backend.offload.return_value = True
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)

    assert wrapper._components == (full,)

    cache._all_reduce_attn_groups = lambda value, op: None
    cache.get_last_hash_value = lambda node: None
    result = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=0,
        last_host_node=0,
        best_match_node=0,
    )
    matched = wrapper.match(
        RadixKey(array("q", [1, 2, 3, 4])),
        SimpleNamespace(rid="match"),
        result,
    )
    assert matched.host_hit_length == 4

    node = SimpleNamespace(id=1, external_cache_stored=False)
    cache.resolve_node_handle = lambda node_id: node
    cache.inc_lock_ref = lambda node_id: SimpleNamespace(to_dec_params=lambda: object())
    cache.dec_lock_ref = MagicMock()
    wrapper._offload_node(node.id)

    assert full.phases == [LinkerTransferPhase.LOOKUP, LinkerTransferPhase.OFFLOAD]


def test_linker_load_marks_an_excluded_swa_prefix_as_tombstones():
    full = _FullComponent()
    swa = _ExcludedSWA()
    inserted = []
    full_indices = torch.arange(4, dtype=torch.int64)

    def insert(params):
        inserted.append(params)
        return InsertResult(
            prefix_len=4,
            total_len=4,
            last_device_node=0,
            adopted_ranges={ComponentType.FULL: [(0, 4)]},
        )

    cache = SimpleNamespace(
        page_size=2,
        components={ComponentType.FULL: full, ComponentType.SWA: swa},
        tree_core=SimpleNamespace(
            empty_match_result=SimpleNamespace(
                device_indices=torch.empty(0, dtype=torch.int64)
            ),
            collect_full_device_indices=lambda node, ancestor: full_indices,
        ),
        insert=insert,
        resolve_node_handle=lambda node_id: SimpleNamespace(id=0),
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper._components = (full,)
    wrapper.hit_markers = {
        "rid": ExternalCacheHitMarker(
            prefix_key=RadixKey(array("q", [1, 2, 3, 4])),
            tail_hashes=["a", "b"],
            device_hit_len=0,
        )
    }
    wrapper._queue_load = MagicMock()
    req = SimpleNamespace(
        rid="rid",
        kv=None,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        last_node=0,
        priority=0,
    )

    restored, last_node = wrapper.load_back(req)

    assert restored.tolist() == full_indices.tolist()
    assert last_node == 0
    assert req.kv == ReqKvInfo(kv_allocated_len=4, swa_evicted_seqlen=4)
    assert inserted[0].swa_evicted_seqlen == 4
    assert full.phases == [LinkerTransferPhase.LOAD]


def test_linker_load_does_not_move_an_existing_swa_boundary_backwards():
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    full = _FullComponent()
    swa = _ExcludedSWA()
    full_indices = torch.arange(4, dtype=torch.int64)
    inserted = []
    wrapper.cache = SimpleNamespace(
        page_size=2,
        components={ComponentType.FULL: full, ComponentType.SWA: swa},
        tree_core=SimpleNamespace(
            empty_match_result=SimpleNamespace(
                device_indices=torch.empty(0, dtype=torch.int64)
            ),
            collect_full_device_indices=lambda node, ancestor: full_indices,
        ),
        insert=lambda params: (
            inserted.append(params)
            or InsertResult(
                prefix_len=4,
                total_len=4,
                last_device_node=0,
                adopted_ranges={ComponentType.FULL: [(0, 4)]},
            )
        ),
        resolve_node_handle=lambda node_id: SimpleNamespace(id=0),
    )
    wrapper._components = (full,)
    wrapper.hit_markers = {
        "rid": ExternalCacheHitMarker(
            prefix_key=RadixKey(array("q", [1, 2, 3, 4])),
            tail_hashes=["a", "b"],
            device_hit_len=0,
        )
    }
    wrapper._queue_load = MagicMock()
    req = SimpleNamespace(
        rid="rid",
        kv=ReqKvInfo(kv_allocated_len=8, swa_evicted_seqlen=8),
        prefix_indices=torch.empty(0, dtype=torch.int64),
        last_node=0,
        priority=0,
    )

    wrapper.load_back(req)

    assert req.kv.swa_evicted_seqlen == 8
    assert inserted[0].swa_evicted_seqlen == 8


def test_linker_load_does_not_override_participating_swa_prepare_boundary():
    full = _FullComponent()
    swa = _IncludedSWA()
    full_indices = torch.arange(4, dtype=torch.int64)
    inserted = []
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(
        page_size=2,
        components={ComponentType.FULL: full, ComponentType.SWA: swa},
        tree_core=SimpleNamespace(
            empty_match_result=SimpleNamespace(
                device_indices=torch.empty(0, dtype=torch.int64)
            ),
            collect_full_device_indices=lambda node, ancestor: full_indices,
        ),
        insert=lambda params: (
            inserted.append(params)
            or InsertResult(
                prefix_len=4,
                total_len=4,
                last_device_node=0,
                adopted_ranges={
                    ComponentType.FULL: [(0, 4)],
                    ComponentType.SWA: [(0, 4)],
                },
            )
        ),
        resolve_node_handle=lambda node_id: SimpleNamespace(id=0),
    )
    wrapper._components = (full, swa)
    wrapper.hit_markers = {
        "rid": ExternalCacheHitMarker(
            prefix_key=RadixKey(array("q", [1, 2, 3, 4])),
            tail_hashes=["a", "b"],
            device_hit_len=0,
        )
    }
    wrapper._queue_load = MagicMock()
    req = SimpleNamespace(
        rid="rid",
        kv=None,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        last_node=0,
        priority=0,
    )

    wrapper.load_back(req)

    assert req.kv.swa_evicted_seqlen == 2
    assert inserted[0].swa_evicted_seqlen == 2
