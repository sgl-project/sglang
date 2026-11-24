from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import get_dsa_index_kpool, is_kimi_k31
from sglang.srt.hardware_backend.npu.attention.dsa_index_k import get_scored_indexer
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator


def kimi_k31_index_k_sharding(kvc: KVCacheConfigurator) -> bool | None:
    """Return the Ascend hybrid target's Index-K layout, or None for other pools."""
    if not kvc.use_mla_backend or kvc.mambaish_config is None or kvc.is_draft_worker:
        return None
    hf_config = kvc.model_config.hf_config
    if not is_kimi_k31(hf_config) or get_dsa_index_kpool(hf_config) != 1:
        return None

    parallel = get_parallel()
    if parallel.attn_dcp_size <= 1:
        return None
    memory = get_memory()
    # Ascend PD, host cache, and prefill CP use global Index-K.
    if (
        parallel.attn_cp_size > 1
        or kvc.kv_cache_dtype != torch.bfloat16
        or kvc.modelslim_cache_layout is not None
        or get_disagg().disaggregation_mode != "null"
        or memory.enable_hierarchical_cache
        or memory.enable_hisparse
    ):
        return False
    return get_scored_indexer() is not None


class NPUKimiK31PoolConfigurator(DefaultPoolConfigurator):
    """Size the Ascend hybrid target using its actual Index-K layout."""

    def __init__(self, kvc: KVCacheConfigurator, *, shard_index_k: bool):
        self._shard_index_k = shard_index_k
        self._target_index_k_budget_delta = 0
        super().__init__(kvc)
        self._cell_size += self._target_index_k_budget_delta

    def _compute_cell_size(
        self, kvc: KVCacheConfigurator, num_layers: int, dcp_size: int | None = None
    ) -> int:
        # MLA KV bytes per token are independent of DCP width. At width 1
        # the common calculation includes one Index-K copy under either flag.
        cell_size = super()._compute_cell_size(kvc, num_layers, dcp_size=1)
        if not self._shard_index_k and kvc.modelslim_cache_layout is None:
            index_k_size = self._compute_dsa_indexer_cell_size(
                kvc=kvc, num_layers=num_layers
            )
            width = get_parallel().attn_dcp_size if dcp_size is None else dcp_size
            cell_size += (width - 1) * index_k_size

        if kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone():
            # The common draft estimator subtracts Index-K using its own layout.
            # Apply the target correction after it has sized the draft.
            common_cell_size = super()._compute_cell_size(kvc, num_layers, dcp_size)
            self._target_index_k_budget_delta = cell_size - common_cell_size
            return common_cell_size
        return cell_size
