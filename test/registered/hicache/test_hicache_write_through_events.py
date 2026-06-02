import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.utils import compute_node_hash_values


class FakeDoneEvent:
    def query(self):
        return True

    def synchronize(self):
        pass


class FakeHostPool:
    def available_size(self):
        return 1_000_000


class FakeWriteThroughController:
    write_policy = "write_through"

    def __init__(self):
        self.ack_write_queue = []
        self.mem_pool_host = FakeHostPool()
        self.next_host_index = 1000

    def write(self, device_indices=None, *args, node_id, **kwargs):
        if device_indices is None:
            device_indices = args[0]
        host_indices = torch.arange(
            self.next_host_index,
            self.next_host_index + len(device_indices),
            dtype=torch.int64,
        )
        self.next_host_index += len(device_indices)
        self.ack_write_queue.append((None, FakeDoneEvent(), [node_id]))
        return host_indices


class FakeUnifiedFullComponent:
    component_type = ComponentType.FULL

    def redistribute_on_node_split(self, new_parent, child):
        new_parent_cd = new_parent.component_data[self.component_type]
        child_cd = child.component_data[self.component_type]
        split_len = len(new_parent.key)
        new_parent_cd.lock_ref = child_cd.lock_ref
        if child_cd.value is not None:
            new_parent_cd.value = child_cd.value[:split_len].clone()
            child_cd.value = child_cd.value[split_len:].clone()
        if child_cd.host_value is not None:
            new_parent_cd.host_value = child_cd.host_value[:split_len].clone()
            child_cd.host_value = child_cd.host_value[split_len:].clone()

    def build_hicache_transfers(self, node, phase, **kwargs):
        return None

    def commit_hicache_transfer(self, node, phase, transfers=(), **kwargs):
        if phase == CacheTransferPhase.BACKUP_HOST:
            node.component_data[self.component_type].host_value = (
                transfers[0].host_indices.clone()
            )


def _cpu_events(cache):
    return [
        event for event in cache.kv_event_queue if event.medium == StorageMedium.CPU
    ]


class HiCacheWriteThroughEventTest(unittest.TestCase):
    def test_hiradix_split_pending_write_through_publishes_cpu_prefix(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.disable = False
        cache.page_size = 2
        cache.is_eagle = False
        cache.enable_storage = False
        cache.enable_kv_cache_events = True
        cache.enable_shared_hicache = False
        cache.cache_controller = FakeWriteThroughController()
        cache.write_through_threshold = 1
        cache.ongoing_write_through = {}
        cache.kv_event_queue = []
        cache.evictable_size_ = 0
        cache.protected_size_ = 0
        cache.evictable_leaves = set()
        cache.root_node = TreeNode(priority=-(10**9))
        cache.root_node.key = RadixKey(array("q"))
        cache.root_node.value = []
        cache.root_node.host_value = []
        cache.root_node.hash_value = []
        cache.root_node.lock_ref = 1
        cache._update_leaf_status = lambda node: None
        cache.inc_lock_ref = lambda node: None
        cache.dec_lock_ref = lambda node: None
        cache.evict_host = lambda num_tokens: 0
        cache._all_reduce_attn_groups = lambda tensor, op: None

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 5, 6])),
                value=torch.tensor([20, 21, 22, 23], dtype=torch.int64),
            )
        )

        self.assertEqual(_cpu_events(cache), [])

        cache.writing_check()

        events = _cpu_events(cache)
        prefix_hash = events[0].block_hashes[0]
        self.assertEqual(
            [(event.parent_block_hash, tuple(event.token_ids)) for event in events],
            [
                (None, (1, 2)),
                (prefix_hash, (3, 4)),
                (prefix_hash, (5, 6)),
            ],
        )

    def test_unified_split_pending_write_through_publishes_cpu_prefix(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_components = (ComponentType.FULL,)
        cache._components_tuple = (FakeUnifiedFullComponent(),)
        cache.components = {ComponentType.FULL: cache._components_tuple[0]}
        cache.page_size = 2
        cache.enable_storage = False
        cache.enable_kv_cache_events = True
        cache.cache_controller = FakeWriteThroughController()
        cache.sidecar_pool_specs = ()
        cache.ongoing_write_through = {}
        cache.kv_event_queue = []
        cache.tp_world_size = 1
        cache.tp_group = None
        cache.root_node = UnifiedTreeNode(cache.tree_components)
        cache.root_node.key = RadixKey(array("q"))
        cache.root_node.hash_value = []
        cache.root_node.component_data[BASE_COMPONENT_TYPE].lock_ref = 1
        cache._for_each_component_lru = lambda *args, **kwargs: None
        cache._update_evictable_leaf_sets = lambda node: None
        cache.inc_lock_ref = lambda node: SimpleNamespace(
            to_dec_params=lambda: None
        )
        cache.dec_lock_ref = lambda node, params=None: None
        cache.evict_host = lambda num_tokens: 0

        child = UnifiedTreeNode(cache.tree_components)
        child.parent = cache.root_node
        child.key = RadixKey(array("q", [1, 2, 3, 4]))
        child.component_data[BASE_COMPONENT_TYPE].value = torch.tensor(
            [10, 11, 12, 13], dtype=torch.int64
        )
        child.hash_value = compute_node_hash_values(child, cache.page_size)
        cache.root_node.children[child.key.child_key(cache.page_size)] = child

        cache.write_backup(child)
        cache._split_node(child.key, child, 2)

        self.assertEqual(_cpu_events(cache), [])

        cache.writing_check()

        events = _cpu_events(cache)
        prefix_hash = events[0].block_hashes[0]
        self.assertEqual(
            [(event.parent_block_hash, tuple(event.token_ids)) for event in events],
            [
                (None, (1, 2)),
                (prefix_hash, (3, 4)),
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
