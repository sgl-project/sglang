"""Tests for --hicache-serialize-load-back on UnifiedRadixCache.

Load-back must not clear device room for a whole chain at once: under write_back
that eviction cascades into a host write whose only non-destructive funding is
reclaimable host duplicates, and a chain that has not been loaded yet contains
none. Per-node steps make each node a duplicate at its own ack.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from collections import defaultdict
from unittest import mock

import torch
from unified_tree_core_inspector import UnifiedTreeCoreInspector

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import _TREE_CORE_REGISTRY
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

FULL = ComponentType.FULL
NODE_LEN = 4
CHAIN = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]


def _inspector_registry():
    """Build either backend as its inspector, which the chain setup reads through."""

    def python_factory(params, components):
        return UnifiedTreeCoreInspector(params, components)

    def rust_factory(params, _components):
        from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

        return RustUnifiedTreeCoreInspector(params)

    return mock.patch.dict(
        _TREE_CORE_REGISTRY, {"python": python_factory, "rust": rust_factory}
    )


def make_cache(serialize_load_back: bool, tree_core_backend: str):
    """FULL-only, page_size=1, CPU cache; hicache is stubbed by the tests."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=64,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=64,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="lru",
        tree_components=(FULL,),
        hicache_serialize_load_back=serialize_load_back,
        tree_core_backend=tree_core_backend,
    )
    with _inspector_registry():
        cache = UnifiedRadixCache(params)
    assert cache._tree_core_backend == tree_core_backend
    cache.serialize_load_back = serialize_load_back
    # Set by the hicache init branch, which a controller-less cache skips.
    cache.load_back_threshold = 10
    cache.tree_core.is_write_back = True
    return cache, allocator


class _FakeController:
    """Records each load and hands back freshly allocated device slots."""

    def __init__(self, allocator):
        self.allocator = allocator
        self.loads = []

    def load(self, *, host_indices, node_id, extra_pools=None):
        device_indices = self.allocator.alloc(len(host_indices))
        assert device_indices is not None
        self.loads.append(
            {
                "node_id": node_id,
                "host_indices": host_indices.tolist(),
                "extra_pools": extra_pools,
            }
        )
        return device_indices


class SerializedLoadBackTest(CustomTestCase):
    tree_core_backend = "python"

    def setUp(self):
        self.cache, self.allocator = make_cache(
            serialize_load_back=True, tree_core_backend=self.tree_core_backend
        )
        self.tree_core = self.cache.tree_core
        self.controller = _FakeController(self.allocator)
        self.cache.cache_controller = self.controller
        self.events = []
        self.cache.evict_for_alloc = self._record_evict
        self.cache._drain_serialized_load_back = self._fake_drain

    def _record_evict(self, params):
        self.events.append(("evict", params.num_tokens))
        raise AssertionError("no eviction expected: the pool has room")

    def _fake_drain(self):
        """Stand in for start_loading + loading_check on the queued step."""
        step_id = next(reversed(self.cache.ongoing_load_back))
        node, lock_params, host_lock_params = self.cache.ongoing_load_back.pop(step_id)
        self.cache.dec_lock_ref(node, lock_params)
        self.cache.dec_host_lock_ref(node, host_lock_params)
        self.tree_core.finish_load_back(node)
        self.events.append(("drain", step_id))

    def _build_evicted_chain(self) -> tuple[int, list[int]]:
        """Insert CHAIN as a path, then back up and demote every node, which is
        the host-only shape ``build_load_back_spec`` walks."""
        tokens = [t for seg in CHAIN for t in seg]
        device_indices = self.allocator.alloc(len(tokens))
        self.cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", tokens)), value=device_indices
            )
        )

        # Split into one node per CHAIN segment by matching each prefix.
        for i in range(1, len(CHAIN)):
            prefix = [t for seg in CHAIN[:i] for t in seg]
            self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(token_ids=array("q", prefix)))
            )

        node_id = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", tokens)))
        ).best_match_node

        chain = []
        nid = node_id
        while not self.tree_core.is_root(nid):
            chain.append(nid)
            nid = self.tree_core.get_parent_node_id(nid)
        chain.reverse()

        # Back up with the device values as the host image, then demote
        # leaf-first. Real eviction does this through the controller.
        for nid in chain:
            host_image = self.tree_core.get_component_device_value(nid, FULL).cpu()
            self.tree_core.commit_backup(nid, host_image, {})
            self.tree_core.mark_write_through_pending([nid], nid)
            self.tree_core.finish_write_through([nid], nid)
        for nid in reversed(chain):
            self.cache._demote(nid, defaultdict(int))
        return node_id, chain

    def test_split_partitions_the_chain_root_first(self):
        node_id, chain = self._build_evicted_chain()
        kv_xfer, _ = self.tree_core.build_load_back_spec(node_id)
        steps = self.tree_core.split_full_load_back_spec(kv_xfer)

        self.assertEqual([s.nodes_to_load for s in steps], [[nid] for nid in chain])
        # The anchor is last: _load_back_per_node relies on it to attach aux.
        self.assertEqual(steps[-1].nodes_to_load, [node_id])
        flat = [t for s in steps for t in s.host_indices.tolist()]
        self.assertEqual(flat, kv_xfer.host_indices.tolist())

    def test_split_of_a_resident_anchor_is_empty(self):
        """An anchor loaded only for aux state has no Full chain to split."""
        empty = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.empty((0,), dtype=torch.int64),
            nodes_to_load=[],
        )
        self.assertEqual(self.tree_core.split_full_load_back_spec(empty), [])

    def test_each_node_loads_and_drains_before_the_next(self):
        node_id, chain = self._build_evicted_chain()
        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(node_id)
        host_anchor = self.cache.inc_host_lock_ref(node_id).to_dec_params()
        ancestor = self.cache.inc_lock_ref(node_id).to_dec_params()

        loaded = self.cache._load_back_per_node(
            anchor_id=node_id,
            kv_xfer=kv_xfer,
            comp_xfers=comp_xfers,
            sidecar_xfers=[],
            ancestor_lock_params=ancestor,
            host_anchor_params=host_anchor,
        )

        self.assertTrue(loaded)
        self.assertEqual(len(self.controller.loads), len(chain))
        for load in self.controller.loads:
            self.assertEqual(len(load["host_indices"]), NODE_LEN)
        # One drain per load, and every load is followed by its own drain.
        self.assertEqual(self.events, [("drain", nid) for nid in chain])
        # Only the final step carries the anchor id, so aux commits once.
        self.assertEqual(self.controller.loads[-1]["node_id"], node_id)

    def test_each_acked_node_becomes_a_reclaimable_duplicate(self):
        """The point of the split: a loaded node funds the next node's
        write-back instead of the cascade destroying a sole host copy."""
        node_id, chain = self._build_evicted_chain()
        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(node_id)

        seen_at_load = []
        inner_load = self.controller.load

        def counting_load(**kwargs):
            seen_at_load.append(sum(map(self.tree_core.is_full_host_duplicate, chain)))
            return inner_load(**kwargs)

        self.controller.load = counting_load
        self.cache._load_back_per_node(
            anchor_id=node_id,
            kv_xfer=kv_xfer,
            comp_xfers=comp_xfers,
            sidecar_xfers=[],
            ancestor_lock_params=self.cache.inc_lock_ref(node_id).to_dec_params(),
            host_anchor_params=self.cache.inc_host_lock_ref(node_id).to_dec_params(),
        )

        # Nothing is reclaimable before the first load; each ack adds one.
        self.assertEqual(seen_at_load, list(range(len(chain))))
        for nid in chain:
            self.assertTrue(self.tree_core.is_full_host_duplicate(nid))
            self.assertTrue(self.tree_core.can_reclaim_full_host_duplicate(nid))

    def test_batched_load_back_has_no_duplicates_to_draw_on(self):
        """Control for the split: loading the chain as one transfer reaches its
        eviction with zero reclaimable duplicates, so a host pool under pressure
        can only fund the write-back by destroying a sole copy."""
        node_id, chain = self._build_evicted_chain()
        self.cache.serialize_load_back = False
        seen_at_load = []
        inner_load = self.controller.load

        def counting_load(**kwargs):
            seen_at_load.append(sum(map(self.tree_core.is_full_host_duplicate, chain)))
            return inner_load(**kwargs)

        self.controller.load = counting_load
        result = self.cache.inc_lock_ref(node_id)
        self.cache._load_back_transfers(
            node_id=node_id,
            mem_quota=None,
            req=None,
            result=result,
            ancestor_lock_params=result.to_dec_params(),
            host_anchor_params=self.cache.inc_host_lock_ref(node_id).to_dec_params(),
        )

        # One transfer for the whole chain, and nothing had become reclaimable.
        self.assertEqual(seen_at_load, [0])
        self.assertEqual(len(self.controller.loads), 1)
        self.assertEqual(
            len(self.controller.loads[0]["host_indices"]), NODE_LEN * len(chain)
        )

    def test_no_device_lock_survives_the_loop(self):
        node_id, chain = self._build_evicted_chain()
        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(node_id)
        self.cache._load_back_per_node(
            anchor_id=node_id,
            kv_xfer=kv_xfer,
            comp_xfers=comp_xfers,
            sidecar_xfers=[],
            ancestor_lock_params=self.cache.inc_lock_ref(node_id).to_dec_params(),
            host_anchor_params=self.cache.inc_host_lock_ref(node_id).to_dec_params(),
        )
        for nid in chain:
            self.assertEqual(self.tree_core.get_component_device_lock_ref(nid, FULL), 0)
            self.assertEqual(self.tree_core.get_component_host_lock_ref(nid, FULL), 0)
        self.assertEqual(self.cache.ongoing_load_back, {})


class SerializedLoadBackRustTest(SerializedLoadBackTest):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
