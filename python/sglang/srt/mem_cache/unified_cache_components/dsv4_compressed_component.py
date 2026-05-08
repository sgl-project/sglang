from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class DeepSeekV4CompressedComponent(TreeComponent):
    """DeepSeek V4 compressed/cache-state sidecar transfer component.

    UnifiedTree stores the logical skeleton in FULL and the executable suffix
    in SWA. This component has no node-local value; it only emits HiCache
    transfers for V4 pools whose indices are derived from FULL pages or from
    the SWA suffix window.
    """

    component_type = ComponentType.DSV4_COMPRESSED

    _FULL_DERIVED_POOLS = (
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
    )
    _SWA_DERIVED_STATE_POOLS = (
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_INDEXER_STATE,
        PoolName.DEEPSEEK_V4_C128_STATE,
    )

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        super().__init__(cache, params)
        self.sliding_window_size = params.sliding_window_size

    def _has_pool(self, name: PoolName) -> bool:
        controller = self.cache.cache_controller
        if controller is None:
            return False
        return name in controller.mem_pool_host.entry_map

    def _available_full_derived_transfers(self) -> list[PoolTransfer]:
        return [
            PoolTransfer(name=name)
            for name in self._FULL_DERIVED_POOLS
            if self._has_pool(name)
        ]

    def _available_state_transfers(self) -> list[PoolTransfer]:
        return [
            PoolTransfer(name=name)
            for name in self._SWA_DERIVED_STATE_POOLS
            if self._has_pool(name)
        ]

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        return lambda node: True

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        return None

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        return 0, 0

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        return None

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        return None

    @staticmethod
    def _has_transfer_indices(xfer: Optional[PoolTransfer], field: str) -> bool:
        if xfer is None:
            return False
        indices = getattr(xfer, field, None)
        return indices is not None and len(indices) > 0

    def build_hicache_transfers(
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw
    ) -> Optional[list[PoolTransfer]]:
        kv_xfer: Optional[PoolTransfer] = kw.get("kv_xfer")
        peer_transfers: dict = kw.get("peer_transfers", {})
        swa_xfers = peer_transfers.get(ComponentType.SWA)
        swa_xfer = swa_xfers[0] if swa_xfers else None

        transfers = []

        if phase == CacheTransferPhase.BACKUP_HOST:
            if self._has_transfer_indices(kv_xfer, "device_indices"):
                transfers.extend(self._available_full_derived_transfers())
            if self._has_transfer_indices(swa_xfer, "device_indices"):
                transfers.extend(self._available_state_transfers())
            return transfers or None

        if phase == CacheTransferPhase.LOAD_BACK:
            if self._has_transfer_indices(kv_xfer, "host_indices"):
                transfers.extend(self._available_full_derived_transfers())
            if self._has_transfer_indices(swa_xfer, "host_indices"):
                transfers.extend(self._available_state_transfers())
            return transfers or None

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
    ) -> None:
        return None
