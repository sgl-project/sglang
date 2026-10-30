from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedTreeNode,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures


class UnifiedTreeCoreLoadBackPendingFixtures:
    def _build_core(self, *, is_write_back: bool):
        component_types = (ComponentType.FULL,)
        root = UnifiedTreeNode(component_types)
        shared = UnifiedTreeNode(component_types)
        anchor_a = UnifiedTreeNode(component_types)
        anchor_b = UnifiedTreeNode(component_types)
        shared.parent = root
        anchor_a.parent = shared
        anchor_b.parent = shared

        nodes = {node.id: node for node in (root, shared, anchor_a, anchor_b)}
        core = mock.Mock()
        core.is_write_back = is_write_back
        core.root_node = root
        core.node_by_id.side_effect = nodes.__getitem__
        core.components_by_type = {ComponentType.FULL: mock.Mock()}
        core.full_host_duplicates = {}
        core._is_settled_full_host_duplicate.side_effect = lambda node: (
            UnifiedTreeCore._is_settled_full_host_duplicate(core, node)
        )
        core._update_duplicate_tracking.side_effect = lambda node: (
            UnifiedTreeCore._update_duplicate_tracking(core, node)
        )
        return core, shared, anchor_a, anchor_b

    def _commit_load_back(self, core, anchor, source):
        transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([1], dtype=torch.int64),
            nodes_to_load=[source.id],
        )
        return UnifiedTreeCore.commit_load_back(
            core,
            anchor.id,
            torch.tensor([2], dtype=torch.int64),
            transfer,
            {},
        )


class UnifiedRadixCacheLoadBackFixtures(UnifiedRadixCacheFixtures):
    def _fill_mamba_state(self, req_to_token_pool, indices, marker):
        if not self.cfg.has_mamba:
            return
        mamba_indices = indices.reshape(-1)
        mamba_cache = req_to_token_pool.mamba_pool.mamba_cache
        mamba_cache.temporal[:, mamba_indices] = marker
        for offset, conv_buf in enumerate(mamba_cache.conv, start=1):
            conv_buf[:, mamba_indices] = marker + offset

    def _snapshot_mamba_state(self, req_to_token_pool, indices):
        mamba_indices = indices.reshape(-1)
        mamba_cache = req_to_token_pool.mamba_pool.mamba_cache
        return (
            mamba_cache.temporal[:, mamba_indices].float().cpu().clone(),
            [conv[:, mamba_indices].float().cpu().clone() for conv in mamba_cache.conv],
        )
