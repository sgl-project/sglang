from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import get_dsa_index_kpool, is_kimi_k31
from sglang.srt.hardware_backend.npu.attention.dsa_index_k import get_scored_indexer
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator


def resolve_kimi_k31_index_k_sharding(kvc: KVCacheConfigurator) -> bool:
    """Require sharded Index-K for Ascend Kimi K3.1 DCP targets."""
    hf_config = kvc.model_config.hf_config
    parallel = get_parallel()
    if kvc.is_draft_worker or parallel.attn_dcp_size <= 1 or not is_kimi_k31(hf_config):
        return False

    if (
        kvc.kv_cache_dtype != torch.bfloat16
        or get_dsa_index_kpool(hf_config) != 1
        or get_memory().enable_hisparse
    ):
        raise ValueError(
            "Ascend Kimi K3.1 DCP requires sharded Index-K with BF16 KV "
            "and index_kpool=1, without the HiSparse KV pool."
        )
    if get_scored_indexer() is None:
        raise RuntimeError(
            "Ascend Kimi K3.1 DCP requires the scored lightning_indexer_v2 "
            "binding from sgl-kernel-npu for CANN 9.2 on A3."
        )
    return True


class NPUKimiK31PoolConfigurator(DefaultPoolConfigurator):
    """Size the Ascend hybrid target with one Index-K cell per local token."""

    def __init__(self, kvc: KVCacheConfigurator):
        self._target_index_k_budget_delta = 0
        super().__init__(kvc)
        self._cell_size += self._target_index_k_budget_delta

    def _compute_cell_size(
        self, kvc: KVCacheConfigurator, num_layers: int, dcp_size: int | None = None
    ) -> int:
        # MLA KV bytes per token are independent of DCP width. Passing 1
        # charges a single Index-K cell before adding draft costs.
        cell_size = super()._compute_cell_size(kvc, num_layers, dcp_size=1)
        if kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone():
            # The common draft estimator subtracts Index-K using its own layout.
            # Apply the target correction after it has sized the draft.
            common_cell_size = super()._compute_cell_size(kvc, num_layers, dcp_size)
            self._target_index_k_budget_delta = cell_size - common_cell_size
            return common_cell_size
        return cell_size
