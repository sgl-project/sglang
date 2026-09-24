"""Regression tests for physical SWA eviction accounting."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.unified_cache.components.base import (
    BASE_COMPONENT_TYPE,
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _component_with_mapping(mapping: torch.Tensor):
    allocator = MagicMock()
    allocator.page_size = 1
    allocator.translate_loc_from_full_to_swa.return_value = mapping

    component = object.__new__(SWAComponent)
    component.cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    component.tree_core = SimpleNamespace(
        component_evictable_size_={ComponentType.SWA: mapping.numel()},
        host_lru_lists={ComponentType.SWA: MagicMock()},
    )
    return component, allocator


def test_swa_evict_reports_only_live_physical_slots():
    """Tombstoned mappings must not satisfy the physical eviction quota."""
    current_mapping = torch.tensor([0, 0, 31, 32], dtype=torch.int64)
    component, allocator = _component_with_mapping(current_mapping)
    full_value = torch.tensor([11, 12, 13, 14], dtype=torch.int64)
    stale_swa_value = torch.tensor([21, 22, 31, 32], dtype=torch.int64)
    node = SimpleNamespace(
        id=7,
        component_data={
            BASE_COMPONENT_TYPE: SimpleNamespace(value=full_value),
            ComponentType.SWA: SimpleNamespace(
                value=stale_swa_value,
                host_value=None,
            ),
        },
    )
    device_frees = defaultdict(list)

    device_freed, host_freed = component.evict_component(
        node,
        device_frees=device_frees,
        host_frees=defaultdict(list),
        target=EvictLayer.DEVICE,
    )

    assert device_freed == 2
    assert host_freed == 0
    assert component.tree_core.component_evictable_size_[ComponentType.SWA] == 0
    assert node.component_data[ComponentType.SWA].value is None
    assert len(device_frees[ComponentType.SWA]) == 1
    assert torch.equal(
        device_frees[ComponentType.SWA][0],
        torch.tensor([13, 14], dtype=torch.int64),
    )
    allocator.translate_loc_from_full_to_swa.assert_called_once_with(full_value)


def test_internal_ghost_tombstone_reports_structural_progress():
    """A zero-capacity internal tombstone must not look like LRU exhaustion."""
    core = MagicMock()
    core.component_evictable_size_ = {ComponentType.SWA: 2}
    component = MagicMock()

    def tombstone_internal(_tracker, _device_frees, _host_frees):
        core.component_evictable_size_[ComponentType.SWA] = 0
        return None

    component.evict_device_next_node.side_effect = tombstone_internal
    core.components_by_type = {ComponentType.SWA: component}
    core._finish_tracking_unbacked_tokens.return_value = 0

    result = UnifiedTreeCore.evict_device_next_node(
        core,
        ComponentType.SWA,
        {ComponentType.FULL: 0, ComponentType.SWA: 0},
    )

    assert result.node_id is None
    assert not result.tracker
    assert not result.device_frees
    assert result.made_progress


def test_swa_evict_does_not_claim_tombstoned_slots():
    current_mapping = torch.zeros(4, dtype=torch.int64)
    component, _ = _component_with_mapping(current_mapping)
    full_value = torch.tensor([11, 12, 13, 14], dtype=torch.int64)
    node = SimpleNamespace(
        id=8,
        component_data={
            BASE_COMPONENT_TYPE: SimpleNamespace(value=full_value),
            ComponentType.SWA: SimpleNamespace(
                value=torch.tensor([21, 22, 23, 24], dtype=torch.int64),
                host_value=None,
            ),
        },
    )

    device_freed, _ = component.evict_component(
        node,
        device_frees=defaultdict(list),
        host_frees=defaultdict(list),
        target=EvictLayer.DEVICE,
    )

    assert device_freed == 0
    assert component.tree_core.component_evictable_size_[ComponentType.SWA] == 0
