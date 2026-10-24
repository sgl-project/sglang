"""Fail-soft degradation for Mooncake direct linker loads.

Covers the eviction-race window between the match-time L3 lookup and
load-back: when the linker's revalidation reports keys missing, the
wrapper must abort the load and degrade the request to a plain cache
miss (re-prefill) instead of committing device slots to an async
layer-wise session that would fail fatally.
"""

from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _RevalidatingFakeLinker(UnifiedCacheLinker):
    """Minimal backend whose revalidate_load verdict is scripted per test."""

    def __init__(self, revalidate_result: bool):
        self.layer_done_counter = object()
        self.revalidate_result = revalidate_result
        self.revalidated_transfers = None
        self.queued_loads = {}
        self.completed_loads = []
        self.completed_offloads = []
        self.queued_offloads = []
        self.reset_count = 0
        self.closed = False

    def lookup(self, rid, transfers):
        return []

    def load(self, rid, transfers):
        self.queued_loads[rid] = list(transfers)
        return True

    def revalidate_load(self, transfers):
        self.revalidated_transfers = list(transfers)
        return self.revalidate_result

    def start_layer_wise_loading(self):
        return 1

    def cancel_queued_load(self, rid):
        return self.queued_loads.pop(rid, None) is not None

    def num_completed_loads(self):
        return len(self.completed_loads)

    def pop_completed_load(self):
        return self.completed_loads.pop(0)

    def num_completed_offloads(self):
        return len(self.completed_offloads)

    def take_completed_offloads(self, finish_count):
        return [True] * min(finish_count, len(self.completed_offloads))

    def commit_completed_offloads(self, results):
        pass

    def offload(self, transfers):
        self.queued_offloads.append(list(transfers))
        return True

    def replace_pending_offload_node(self, ack_id, old_node_id, new_node_ids):
        pass

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class _LoadBackComponent:
    """Component handing out a fixed KV transfer and recording load phases."""

    def __init__(self):
        self.phases = []

    def build_external_linker_transfer(self, phase, node, keys):
        assert phase == LinkerTransferPhase.LOAD
        return PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(len(keys), dtype=torch.int64),
            keys=list(keys),
        )

    def update_external_linker_load(
        self,
        phase,
        req,
        full_transfer,
        transfer,
        prefix_len,
        *,
        insert_result=None,
        canonical_full=None,
    ):
        self.phases.append(phase)
        if phase == ExternalLinkerLoadPhase.ABORT:
            return None
        return transfer


def _cache_for_load_back(component, empty_indices):
    lock_params = object()
    cache = SimpleNamespace(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            empty_match_result=SimpleNamespace(device_indices=empty_indices),
        ),
        write_through_threshold=256,
        pp_size=1,
        pp_group=None,
        page_size=1,
        _components_tuple=(component,),
        insert=lambda params: SimpleNamespace(last_device_node=9, mamba_exist=False, adopted_ranges={}),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: lock_params),
        dec_lock_ref=lambda node_id, params: None,
        resolve_node_handle=lambda node_id: SimpleNamespace(id=node_id),
        req_to_token_pool=SimpleNamespace(mamba_allocator=SimpleNamespace(free=lambda indices: None)),
    )
    return cache, lock_params


def _hit_marker():
    return ExternalCacheHitMarker(
        prefix_key=RadixKey(array("q", range(4))),
        tail_hashes=["k0", "k1"],
        device_hit_len=2,
    )


def _load_back(wrapper, component):
    req = SimpleNamespace(
        rid="rid",
        last_node=5,
        prefix_indices=torch.zeros(2, dtype=torch.int64),
        kv=None,
        priority=0,
    )
    return wrapper.load_back(req), req, component


def test_revalidation_failure_degrades_to_cache_miss():
    empty_indices = torch.zeros(0, dtype=torch.int64)
    component = _LoadBackComponent()
    linker = _RevalidatingFakeLinker(revalidate_result=False)
    cache, lock_params = _cache_for_load_back(component, empty_indices)
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.hit_markers["rid"] = _hit_marker()

    (indices, last_node), _, component = _load_back(wrapper, component)

    # Degrade: the caller sees an empty cache miss on the same node.
    assert indices is empty_indices
    assert last_node == 5
    # The aborted load freed the component slots and never queued a load.
    assert component.phases == [ExternalLinkerLoadPhase.ABORT]
    assert linker.queued_loads == {}
    assert wrapper.hit_markers == {}
    # Revalidation ran before the abort, with the raw component transfers.
    assert linker.revalidated_transfers is not None
    assert linker.revalidated_transfers[0].name == PoolName.KV


def test_revalidation_success_proceeds_to_prepare_and_load():
    empty_indices = torch.zeros(0, dtype=torch.int64)
    component = _LoadBackComponent()
    linker = _RevalidatingFakeLinker(revalidate_result=True)
    cache, lock_params = _cache_for_load_back(component, empty_indices)
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.hit_markers["rid"] = _hit_marker()

    (indices, last_node), _, component = _load_back(wrapper, component)

    # The load proceeds: slots committed and the async load queued.
    assert component.phases == [
        ExternalLinkerLoadPhase.PREPARE,
        ExternalLinkerLoadPhase.COMMIT,
    ]
    assert "rid" in linker.queued_loads
    assert len(indices) == 2  # device prefix + loaded tail pages
    assert last_node == 9


def test_backend_without_revalidation_keeps_legacy_path():
    empty_indices = torch.zeros(0, dtype=torch.int64)
    component = _LoadBackComponent()
    linker = _RevalidatingFakeLinker(revalidate_result=True)
    # Simulate a backend that has not implemented the hook.
    del linker.revalidate_load
    cache, lock_params = _cache_for_load_back(component, empty_indices)
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.hit_markers["rid"] = _hit_marker()

    (indices, last_node), _, component = _load_back(wrapper, component)

    assert "rid" in linker.queued_loads
    assert component.phases == [
        ExternalLinkerLoadPhase.PREPARE,
        ExternalLinkerLoadPhase.COMMIT,
    ]
