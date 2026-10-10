from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


class _Key:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def _component_data():
    return [SimpleNamespace(value=None, host_value=None) for _ in range(3)]


def test_swa_load_back_splits_large_host_node_to_one_window():
    root = SimpleNamespace(parent=None)
    node = SimpleNamespace(
        key=_Key(16), id=1, parent=root, component_data=_component_data()
    )
    node.component_data[ComponentType.SWA].host_value = torch.arange(16)

    cache = SimpleNamespace(root_node=root, cache_controller=object())

    def split_node(key, child, split_len):
        parent = SimpleNamespace(
            key=_Key(split_len),
            parent=child.parent,
            component_data=_component_data(),
        )
        old = child.component_data[ComponentType.SWA].host_value
        parent.component_data[ComponentType.SWA].host_value = old[:split_len].clone()
        child.component_data[ComponentType.SWA].host_value = old[split_len:].clone()
        child.key = _Key(len(key) - split_len)
        child.parent = parent
        return parent, None

    cache._split_node = split_node
    component = SWAComponent.__new__(SWAComponent)
    component.cache = cache
    component.tree_core = SimpleNamespace(
        root_node=root, has_swa_host_pool=True, enable_hicache=True, _split_node=split_node
    )
    component.sliding_window_size = 6
    component._swa_kv_pool_host = object()

    transfers = component.build_hicache_transfers(
        node, CacheTransferPhase.LOAD_BACK
    )

    assert len(transfers) == 1
    transfer = transfers[0]
    assert transfer.nodes_to_load == [node.id]
    torch.testing.assert_close(transfer.host_indices, torch.arange(10, 16))
    assert len(node.key) == 6
    assert len(node.parent.key) == 10
