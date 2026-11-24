from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import get_dsa_index_kpool, is_kimi_k31
from sglang.srt.hardware_backend.npu.attention.dsa_index_k import get_scored_indexer
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator


def uses_sharded_index_k(kvc: KVCacheConfigurator) -> bool:
    if not kvc.use_mla_backend or kvc.mambaish_config is None or kvc.is_draft_worker:
        return False
    hf_config = kvc.model_config.hf_config
    if not is_kimi_k31(hf_config) or get_dsa_index_kpool(hf_config) != 1:
        return False

    parallel = get_parallel()
    memory = get_memory()
    # Ascend PD, host cache, and prefill CP use global Index-K.
    if (
        parallel.attn_dcp_size <= 1
        or parallel.attn_cp_size > 1
        or kvc.kv_cache_dtype != torch.bfloat16
        or kvc.modelslim_cache_layout is not None
        or get_disagg().disaggregation_mode != "null"
        or memory.enable_hierarchical_cache
        or memory.enable_hisparse
    ):
        return False
    return get_scored_indexer() is not None


class NPUShardedIndexKPoolConfigurator(DefaultPoolConfigurator):
    """Size the Ascend hybrid pool with one Index-K cell per local token."""

    def __init__(self, kvc: KVCacheConfigurator):
        self._replicated_index_k_bytes = 0
        super().__init__(kvc)
        self._cell_size -= self._replicated_index_k_bytes

    def _compute_cell_size(
        self, kvc: KVCacheConfigurator, num_layers: int, dcp_size: int | None = None
    ) -> int:
        # MLA KV bytes per token are independent of dcp_size. Passing 1 leaves
        # a single Index-K copy in the budget before adding any draft costs.
        sharded_cell_size = super()._compute_cell_size(kvc, num_layers, dcp_size=1)
        if kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone():
            # The default draft estimator subtracts replicated target Index-K.
            # Keep that estimate intact and remove the extra target copies below it.
            replicated_cell_size = super()._compute_cell_size(kvc, num_layers, dcp_size)
            self._replicated_index_k_bytes = replicated_cell_size - sharded_cell_size
            return replicated_cell_size
        return sharded_cell_size
