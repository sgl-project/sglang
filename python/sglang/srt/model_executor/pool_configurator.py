"""Memory pool configurators for profiling and sizing KV cache pools.

Each model architecture has its own configurator that computes pool sizes
from available GPU memory using a unified coeff+bias model:

    available_bytes = max_tokens * coeff + bias
    max_tokens = (available_bytes - bias) / coeff

Two entry points, same core computation:
- calculate_pool_sizes(available_bytes, page_size): profiling path
- calculate_pool_sizes_from_max_tokens(max_tokens, page_size): constraint path
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import (
    AttentionArch,
    dsa_layer_skips_topk,
    get_dsa_index_head_dim,
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    is_deepseek_dsa,
    is_deepseek_v4,
    is_minimax_sparse,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.fp4_logits_workspace import (
    FP4LogitsWorkspacePlan,
    fp4_logits_width_for_context,
    plan_fp4_logits_workspace,
)
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    get_compress_state_ring_size,
    get_compress_state_write_pad,
    get_dsv4_indexer_bytes_per_token,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
    max_prefill_buffer_tokens,
    max_speculative_num_draft_tokens,
)
from sglang.srt.utils.common import (
    ceil_align,
    ceil_div,
    is_float4_e2m1fn_x2,
    is_hip,
    spec_decode_alloc_len_per_request,
)

_is_hip = is_hip()


@dataclass
class MemoryPoolConfig:
    """Resolved memory pool config, shared between target and draft workers."""

    max_total_num_tokens: int
    max_running_requests: Optional[int] = None
    full_max_total_num_tokens: Optional[int] = None
    swa_max_total_num_tokens: Optional[int] = None

    # DSV4 compressed-attention pool sizes (target only; draft workers leave at 0).
    c4_max_total_num_tokens: int = 0
    c128_max_total_num_tokens: int = 0
    c4_state_pool_size: int = 0
    c128_state_pool_size: int = 0
    dsv4_fp4_logits_workspace_bytes: int = 0
    dsv4_fp4_logits_workspace_max_seq_len: int = 0

    mem_fraction_static: Optional[float] = None

    # Unified pool only: the PROFILED byte budget for the token-granular
    # sub-pools. Set, the factories size the buffer from it directly instead of
    # re-summing ratio-derived token counts, which keeps the re-sum's floor
    # losses out of the buffer; the token counts stay boot labels / conserve
    # caps. None on the token-capped path -- a user token cap IS the budget.
    unified_total_bytes: Optional[int] = None

    def __post_init__(self):
        if self.dsv4_fp4_logits_workspace_bytes < 0:
            raise ValueError("dsv4_fp4_logits_workspace_bytes must be non-negative")
        if self.dsv4_fp4_logits_workspace_max_seq_len < 0:
            raise ValueError(
                "dsv4_fp4_logits_workspace_max_seq_len must be non-negative"
            )
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

logger = logging.getLogger(__name__)


def _dflash_draft_cell_size(kvc: KVCacheConfigurator) -> int:
    """Bytes/token the DFLASH draft KV pool adds to the target's budget, 0 if none.

    Unlike an EAGLE draft, which reuses the target's attention config and is
    therefore priced by layer count, a DFLASH draft has its own geometry and is
    a flat additive term. Under DCP, the target pool is sharded while the draft
    pool spans the allocator's widened virtual location space, so the draft
    term is replicated across DCP ranks.
    """
    if kvc.is_draft_worker or not kvc.spec_algorithm.is_dflash_family():
        return 0
    cell_size = kvc.spec_aux_config.dflash_draft_cell_size_per_token
    if cell_size is None or int(cell_size) <= 0:
        return 0
    return int(cell_size) * get_parallel().attn_dcp_size


def _get_dsa_cache_layer_ids(kvc: KVCacheConfigurator, num_layers: int) -> list[int]:
    """Global layer ids represented by the local DSA pool's dense layer slots."""
    if kvc.mambaish_config and not kvc.is_draft_worker:
        layer_ids = [
            layer_id
            for layer_id in kvc.mambaish_config.full_attention_layer_ids
            if kvc.layer_info.start_layer <= layer_id < kvc.layer_info.end_layer
        ]
    else:
        layer_ids = list(range(kvc.layer_info.start_layer, kvc.layer_info.end_layer))
    # Draft pools and a few platform-specific pools may expose a synthetic layer
    # count. They do not use indexShare, so only the length matters for sizing.
    if len(layer_ids) != num_layers:
        return list(range(num_layers))
    return layer_ids


def _get_dsv4_compress_state_dtype_sizes() -> tuple[int, int]:
    dtype_name = envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get().strip().lower()
    if dtype_name in ("float32", "fp32"):
        return 4, 4
    if dtype_name in ("bfloat16", "bf16"):
        return 2, 2
    raise ValueError(
        "Unsupported SGLANG_DSV4_COMPRESS_STATE_DTYPE="
        f"{dtype_name!r}. Expected one of: float32, fp32, bfloat16, bf16."
    )


class MemoryPoolConfigurator:
    """Base class for memory pool configurators.

    Subclasses compute pool sizes for their architecture via coeff+bias model.
    Both entry points return MemoryPoolConfig (with max_running_requests=None,
    to be filled by the consumer).
    """

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        """Profiling path: compute pool sizes from available bytes."""
        raise NotImplementedError

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Constraint path: recalculate pool sizes from a constrained max_tokens."""
        raise NotImplementedError

    def finalize_with_max_running_requests(
        self, config: MemoryPoolConfig
    ) -> MemoryPoolConfig:
        return config

    @staticmethod
    def validate_swa_pool_size(
        swa_tokens: int, sliding_window_size: Optional[int], page_size: int
    ) -> None:
        """Reject an SWA pool too small to ever admit a request.

        Prefill charges min(extend + decode, window) + page_size of SWA headroom
        per request, so a pool at or below that floor rejects every request no
        matter how far it drains: the scheduler spins in the waiting queue and
        the server hangs at warmup instead of failing here.
        """
        if sliding_window_size is None:
            return
        if sliding_window_size + page_size >= swa_tokens:
            raise ValueError(
                f"SWA pool ({swa_tokens} tokens) cannot hold even one request: "
                f"the prefill admission floor is sliding_window_size "
                f"({sliding_window_size}) + page_size ({page_size}). "
                f"Increase --swa-full-tokens-ratio or the total KV budget."
            )


class DefaultPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for standard models: MHA, MLA, DSA, FP4.

    coeff = cell_size (bytes per token across all layers)
    bias = 0
    """

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
        # Determine effective number of layers for KV cache
        if mambaish := mambaish_config(kvc.model_config):
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if kvc.layer_info.start_layer <= i < kvc.layer_info.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = kvc.layer_info.num_effective_layers

        self._cell_size = self._compute_cell_size(kvc, num_layers)
        has_kv_on_another_pp_stage = (
            self._cell_size == 0
            and mambaish is not None
            and bool(mambaish.full_attention_layer_ids)
            and kvc.ps.pp_size > 1
        )
        self._zero_kv_max_tokens = (
            torch.iinfo(torch.int64).max
            if has_kv_on_another_pp_stage
            else get_schedule().max_total_tokens or kvc.model_config.context_len
        )

        # EAGLE/STANDALONE: scale cell_size to account for draft model KV cache.
        # Assumes draft and target share the same per-layer KV size (head_dim,
        # num_kv_heads, dtype), which holds for EAGLE/MTP draft models that
        # reuse the target architecture's attention config.
        if (
            kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone()
        ) and not kvc.is_draft_worker:
            eagle_draft_num_layers = kvc.spec_aux_config.eagle_draft_num_layers
            if (
                eagle_draft_num_layers is not None
                and int(eagle_draft_num_layers) > 0
                and int(num_layers) > 0
            ):
                draft_num_layers = int(eagle_draft_num_layers)
                if is_deepseek_dsa(kvc.model_config.hf_config):
                    target_indexer_size = self._compute_dsa_indexer_cell_size(
                        kvc=kvc,
                        num_layers=num_layers,
                    )
                    target_kv_size = self._cell_size - target_indexer_size
                    from sglang.srt.layers.cp.utils import (
                        get_glm_dsa_layer_split_effective_num_layers,
                    )

                    target_kv_num_layers = get_glm_dsa_layer_split_effective_num_layers(
                        kvc, num_layers
                    )
                    draft_kv_size = int(
                        target_kv_size * draft_num_layers / target_kv_num_layers
                    )
                    draft_indexer_size = self._compute_dsa_indexer_cell_size(
                        kvc=kvc,
                        num_layers=draft_num_layers,
                        allocate_all_layers=True,
                    )
                    self._cell_size += draft_kv_size + draft_indexer_size
                else:
                    self._cell_size = int(
                        self._cell_size * (1 + draft_num_layers / int(num_layers))
                    )

        # DFLASH/DSPARK: reserve the draft runner's *actual* per-token KV cost.
        # The draft allocates its own KV pool at the target's
        # max_total_num_tokens, whose per-token footprint can differ from the
        # target's (e.g. an MLA-latent target paired with a full per-head K/V
        # draft), so size from the draft config rather than the layer ratio.
        if kvc.spec_algorithm.is_dflash_family() and not kvc.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                scale_kv_cell_size_per_token_for_dflash,
            )

            draft_num_layers = kvc.spec_aux_config.dflash_draft_num_layers
            if (
                draft_num_layers is not None
                and int(draft_num_layers) > 0
                and int(num_layers) > 0
            ):
                self._cell_size = scale_kv_cell_size_per_token_for_dflash(
                    target_cell_size_per_token=self._cell_size,
                    target_num_layers=int(num_layers),
                    draft_num_layers=int(draft_num_layers)
                    * get_parallel().attn_dcp_size,
                    draft_cell_size_per_token=_dflash_draft_cell_size(kvc) or None,
                )

    def _compute_cell_size(self, kvc: KVCacheConfigurator, num_layers: int) -> int:
        """Compute per-token KV cache cost in bytes. Subclasses can override."""
        # args to config cell size
        model_config = kvc.model_config
        kv_cache_dtype = kvc.kv_cache_dtype
        from sglang.srt.layers.cp.utils import (
            get_glm_dsa_layer_split_effective_num_layers,
        )

        effective_num_layers = (
            num_layers
            if kvc.server_args.enable_hisparse
            else get_glm_dsa_layer_split_effective_num_layers(kvc, num_layers)
        )

        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_parallel().attn_tp_size
        dcp_size = get_parallel().attn_dcp_size

        if kvc.use_mla_backend:
            from sglang.srt.mem_cache.kv_cache_configurator import (
                calculate_mla_kv_cache_dim,
            )

            cell_size = (
                calculate_mla_kv_cache_dim(
                    model_config=model_config,
                    kv_cache_dtype=kv_cache_dtype,
                )
                * effective_num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                        // scale_block_size
                    )
                    * effective_num_layers
                    * kv_size
                )

            # Add indexer KV cache overhead for DSA models (DeepSeek V3.2)
            if is_deepseek_dsa(model_config.hf_config):
                cell_size += self._compute_dsa_indexer_cell_size(
                    kvc=kvc,
                    num_layers=num_layers,
                )
        elif is_minimax_sparse(model_config.hf_config):
            # Mirrors MiniMaxSparseKVPool: main pool (K+V all layers) + indexer pool
            # (sparse-only, single-head; kv layers store K+V, k-only layers store K).
            sparse_cfg = get_minimax_sparse_attention_config(model_config.hf_config)
            dense_layer_ids, sparse_layer_ids = get_minimax_sparse_layer_ids(sparse_cfg)
            indexer_k_only_layer_ids = set(
                get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
            )

            local_dense_layer_ids = [
                l
                for l in dense_layer_ids
                if kvc.layer_info.start_layer <= l < kvc.layer_info.end_layer
            ]
            local_sparse_layer_ids = [
                l
                for l in sparse_layer_ids
                if kvc.layer_info.start_layer <= l < kvc.layer_info.end_layer
            ]
            num_dense = len(local_dense_layer_ids)
            num_sparse = len(local_sparse_layer_ids)
            num_indexer_k_only = sum(
                1 for l in local_sparse_layer_ids if l in indexer_k_only_layer_ids
            )
            num_indexer_kv = num_sparse - num_indexer_k_only

            kv_heads = model_config.get_num_kv_heads(get_parallel().attn_tp_size)
            head_dim = model_config.head_dim
            indexer_head_dim = sparse_cfg["sparse_index_dim"]
            indexer_dtype_size = torch._utils._element_size(kvc.model_dtype)

            main_pool_bytes = (
                (num_dense + num_sparse) * 2 * kv_heads * head_dim * kv_size
            )
            indexer_bytes = (
                (num_indexer_kv * 2 + num_indexer_k_only)
                * indexer_head_dim
                * indexer_dtype_size
            )
            # FP4 scale buffer adjustment doesn't apply to MiniMax sparse:
            # cell_size is already a sum over heterogeneous sub-pools.
            return main_pool_bytes + indexer_bytes
        else:
            n = model_config.get_num_kv_heads(tp_size, dcp_size)
            cell_size = (
                n
                * (model_config.head_dim + model_config.v_head_dim)
                * effective_num_layers
                * kv_size
            )

            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                k = model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * effective_num_layers * 2 * kv_size) // scale_block_size
                )
                # FP4 prefill uses one shared FP8 dequant workspace across layers.
                cell_size += n * k * 2 * kv_size
            elif self.kv_cache_dtype_str == "mxfp8":
                scale_block_size = 32
                cell_size += (
                    n * (model_config.head_dim + model_config.v_head_dim) * num_layers
                ) // scale_block_size

        return cell_size

    def _compute_dsa_indexer_cell_size(
        self,
        *,
        kvc: KVCacheConfigurator,
        num_layers: int,
        allocate_all_layers: bool = False,
    ) -> int:
        index_head_dim = get_dsa_index_head_dim(kvc.model_config.hf_config)
        indexer_size_per_token = (
            index_head_dim + index_head_dim // DSATokenToKVPool.quant_block_size * 4
        )
        element_size = torch._utils._element_size(
            DSATokenToKVPool.index_k_with_scale_buffer_dtype
        )
        memory_config = get_memory()
        indexer_ratio = 1
        if memory_config.enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            indexer_ratio = parse_hisparse_config().host_to_device_ratio

        from sglang.srt.mem_cache.kv_cache_configurator import (
            _should_elide_dsa_index_k,
        )

        if (
            allocate_all_layers
            or kvc.server_args.enable_hisparse
            or not _should_elide_dsa_index_k(is_draft_worker=kvc.is_draft_worker)
        ):
            num_indexer_layers = num_layers
        else:
            from sglang.srt.layers.cp.utils import (
                get_glm_dsa_cp_layer_shard_info,
                get_layer_shard_range,
            )

            _, shard_size = get_glm_dsa_cp_layer_shard_info(kvc)
            if shard_size > 1:
                # Preserve the existing LayerSplit sizing semantics. GLM-5.3
                # hybrid-layer support is intentionally limited to the normal
                # (non-LayerSplit) pool below.
                active_indexer_layers = [
                    layer_id
                    for layer_id in range(
                        kvc.layer_info.start_layer, kvc.layer_info.end_layer
                    )
                    if not dsa_layer_skips_topk(kvc.model_config.hf_config, layer_id)
                ]
                active_set = set(active_indexer_layers)
                max_owned = 0
                for rank in range(shard_size):
                    start, end = get_layer_shard_range(rank, shard_size, num_layers)
                    max_owned = max(
                        max_owned,
                        sum(
                            kvc.layer_info.start_layer + i in active_set
                            for i in range(start, end)
                        ),
                    )
                num_indexer_layers = max_owned + 1
            else:
                num_indexer_layers = sum(
                    not dsa_layer_skips_topk(kvc.model_config.hf_config, layer_id)
                    for layer_id in _get_dsa_cache_layer_ids(kvc, num_layers)
                )

        return int(
            indexer_size_per_token * num_indexer_layers * element_size * indexer_ratio
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = (
            available_bytes // self._cell_size
            if self._cell_size
            else self._zero_kv_max_tokens
        )
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)


class HybridSWAPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for MHA or MLA models with sliding-window layers.

    Splits available memory between full attention and SWA pools.
    Does NOT inherit DefaultPoolConfigurator — different coeff model.
    """

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
        model_config = kvc.model_config
        kv_cache_dtype = kvc.kv_cache_dtype
        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_parallel().attn_tp_size

        self._full_layers_num = len(model_config.full_attention_layer_ids)
        self._swa_layers_num = len(model_config.swa_attention_layer_ids)
        assert self._swa_layers_num > 0, (
            "Hybrid SWA model must have at least one SWA layer"
        )

        self._swa_full_tokens_ratio = get_schedule().swa_full_tokens_ratio
        self._sliding_window_size = kvc.sliding_window_size
        self._page_size = kvc.page_size
        self._enable_unified_memory = get_memory().enable_unified_memory

        if model_config.attention_arch == AttentionArch.MLA:
            # MLA pool sizing uses latent dimensions rather than MHA heads.
            from sglang.srt.mem_cache.kv_cache_configurator import (
                calculate_mla_kv_cache_dim,
            )

            self._full_per_token = (
                calculate_mla_kv_cache_dim(
                    model_config=model_config,
                    kv_cache_dtype=kv_cache_dtype,
                )
                * kv_size
            )
            if is_deepseek_dsa(model_config.hf_config):
                index_head_dim = get_dsa_index_head_dim(model_config.hf_config)
                index_elements = (
                    index_head_dim
                    + index_head_dim // DSATokenToKVPool.quant_block_size * 4
                )
                self._full_per_token += index_elements * torch._utils._element_size(
                    DSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
            self._swa_per_token = (
                model_config.swa_kv_lora_rank + model_config.swa_qk_rope_head_dim
            ) * kv_size
        else:
            # Full layer per-token memory (bytes)
            self._full_per_token = (
                model_config.get_num_kv_heads(tp_size)
                * (model_config.head_dim + model_config.v_head_dim)
                * kv_size
            )

            # SWA layer per-token memory (bytes)
            self._swa_per_token = (
                model_config.get_swa_num_kv_heads(tp_size)
                * (model_config.swa_head_dim + model_config.swa_v_head_dim)
                * kv_size
            )

        if self.kv_cache_dtype_str == "mxfp8":
            scale_block_size = 32
            self._full_per_token += (
                model_config.get_num_kv_heads(tp_size)
                * (model_config.head_dim + model_config.v_head_dim)
            ) // scale_block_size
            self._swa_per_token += (
                model_config.get_swa_num_kv_heads(tp_size)
                * (model_config.swa_head_dim + model_config.swa_v_head_dim)
            ) // scale_block_size

        # Draft KV tensors use full, SWA, or full-capacity SWA geometry.
        self._draft_full_layers_num = 0
        self._draft_swa_layers_num = 0
        self._draft_swa_full_layers_num = 0
        if (
            kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone()
        ) and not kvc.is_draft_worker:
            draft_layers = kvc.spec_aux_config.eagle_draft_num_layers
            if draft_layers is not None and int(draft_layers) > 0:
                draft_layers = int(draft_layers)
                mtp_local_layer_ids = getattr(
                    getattr(model_config, "hf_text_config", None),
                    "mtp_local_layer_ids",
                    None,
                )
                if mtp_local_layer_ids is not None:
                    local_layer_ids = set(mtp_local_layer_ids)
                    self._draft_swa_full_layers_num = sum(
                        layer_id in local_layer_ids for layer_id in range(draft_layers)
                    )
                else:
                    draft_swa_layers = kvc.spec_aux_config.eagle_draft_swa_num_layers
                    if draft_swa_layers is not None:
                        self._draft_swa_layers_num = min(
                            max(int(draft_swa_layers), 0), draft_layers
                        )
                self._draft_full_layers_num = (
                    draft_layers
                    - self._draft_swa_layers_num
                    - self._draft_swa_full_layers_num
                )

        self._draft_cell_size = _dflash_draft_cell_size(kvc)

        self._recompute_cell_size()

    def _recompute_cell_size(self) -> None:
        # Bytes per token of max_total_num_tokens.
        #
        # Hybrid (full_layers > 0): max_total = full_tokens, so cell_size accounts
        # for both pools: F*nf + r*S*ns (where swa_tokens = full_tokens * r).
        #
        # All-SWA (full_layers == 0): max_total = swa_tokens directly. The ratio
        # is meaningless here -- there is no full pool to relate to, and every
        # token beyond the sliding window can be evicted. So cell_size = S*ns,
        # with no ratio factor applied.
        if self._full_layers_num == 0:
            self._cell_size = (
                self._swa_per_token * self._swa_layers_num
                + self._full_per_token * self._draft_full_layers_num
                + self._swa_per_token * self._draft_swa_layers_num
                + self._swa_per_token * self._draft_swa_full_layers_num
                + self._draft_cell_size
            )
        else:
            self._cell_size = (
                self._full_per_token
                * (self._full_layers_num + self._draft_full_layers_num)
                + self._swa_per_token * self._draft_swa_full_layers_num
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * (self._swa_layers_num + self._draft_swa_layers_num)
                + self._draft_cell_size
            )

    def _draft_pool_bytes_per_token(self) -> int:
        return int(
            self._full_per_token * self._draft_full_layers_num
            + self._swa_per_token
            * (self._draft_swa_layers_num + self._draft_swa_full_layers_num)
            + self._draft_cell_size
        )

    def _max_unified_full_tokens(
        self,
        available_bytes: int,
        page_size: int,
        fixed_swa_tokens: Optional[int] = None,
    ) -> int:
        """Find the largest page-aligned full capacity whose allocations fit."""
        draft_bytes_per_token = self._draft_pool_bytes_per_token()
        target_full_bytes_per_token = self._full_per_token * self._full_layers_num
        target_swa_bytes_per_token = self._swa_per_token * self._swa_layers_num
        assert target_full_bytes_per_token > 0

        def allocation_bytes(full_pages: int) -> int:
            full_tokens = full_pages * page_size
            swa_tokens = (
                fixed_swa_tokens
                if fixed_swa_tokens is not None
                else int(full_tokens * self._swa_full_tokens_ratio)
                // page_size
                * page_size
            )
            target_bytes = (
                full_tokens * target_full_bytes_per_token
                + swa_tokens * target_swa_bytes_per_token
            )
            virtual_span = max(target_bytes // target_full_bytes_per_token - 1, 0)
            draft_tokens = ceil_align(virtual_span, page_size) + page_size
            return target_bytes + draft_tokens * draft_bytes_per_token

        max_pages = available_bytes // target_full_bytes_per_token // page_size
        full_pages = (
            bisect_right(range(max_pages + 1), available_bytes, key=allocation_bytes)
            - 1
        )
        return max(full_pages, 0) * page_size

    def _solve_pool_sizes(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Core computation: split max_total_num_tokens into full/swa pool sizes."""

        def align_page_size(x: int) -> int:
            return (x // page_size) * page_size

        if self._full_layers_num == 0:
            # All-SWA: no full pool, max_total = actual SWA pool size.
            # Ratio is not applied -- see __init__ comment.
            swa_tokens = align_page_size(max_total_num_tokens)
            logger.info(
                f"Use sliding window memory pool (all SWA). "
                f"swa_layer_tokens={swa_tokens}"
            )
            return MemoryPoolConfig(
                max_total_num_tokens=swa_tokens,
                full_max_total_num_tokens=0,
                swa_max_total_num_tokens=swa_tokens,
            )

        # Hybrid: full_tokens = max_total_num_tokens, swa_tokens = full_tokens * ratio
        full_tokens = align_page_size(max_total_num_tokens)
        swa_tokens = align_page_size(int(full_tokens * self._swa_full_tokens_ratio))

        self.validate_swa_pool_size(
            swa_tokens, self._sliding_window_size, self._page_size
        )

        logger.info(
            f"Use sliding window memory pool. "
            f"full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
        )

        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        if (
            self._enable_unified_memory
            and self._full_layers_num > 0
            and self._draft_pool_bytes_per_token() > 0
        ):
            max_total_num_tokens = self._max_unified_full_tokens(
                available_bytes, page_size
            )
        else:
            max_total_num_tokens = int(available_bytes // self._cell_size)
        return self._solve_pool_sizes(max_total_num_tokens, page_size)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._solve_pool_sizes(max_total_num_tokens, page_size)


class SWAChunkCapPoolConfigurator(HybridSWAPoolConfigurator):
    """Hybrid SWA configurator with the SWA pool sized from a fixed token cap.

    When max_running_requests is explicit, the SWA pool's worst-case
    footprint is bounded per request. The SWA pool is sized tightly from that
    cap and the freed memory is redirected to the full pool, instead of sizing
    both pools by swa_full_tokens_ratio.
    """

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
        super().__init__(kvc)
        assert self._full_layers_num > 0

        page_size = kvc.page_size
        window = kvc.sliding_window_size
        draft_tokens = get_spec().speculative_num_draft_tokens or 1
        eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())

        """
        __________[padding][eviction_interval][window]
        Padding to make sure eviction point is page-aligned.
        """
        trailing_tokens = window + eviction_interval * draft_tokens + page_size
        if get_spec().speculative_algorithm is None:
            decode_alloc = page_size
        elif get_schedule().disable_overlap_schedule:
            # spec-v1: new_tokens_required_next_decode per request.
            decode_alloc = spec_decode_alloc_len_per_request(
                page_size=page_size,
                speculative_num_steps=get_spec().speculative_num_steps,
                speculative_eagle_topk=get_spec().speculative_eagle_topk,
                speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            )
        else:
            # spec-v2: the overlap allocator keeps 2 * alloc_len outstanding
            # (eagle_utils.eagle_prepare_for_decode: kv_committed_len + 2 * alloc_len).
            decode_alloc = 2 * get_alloc_len_per_decode()
        per_request = trailing_tokens + decode_alloc

        num_reqs = get_schedule().max_running_requests // kvc.ps.attn_dp_size
        if get_disagg().disaggregation_mode == "decode":
            self._swa_cap = (
                per_request * num_reqs
                + (window + page_size) * get_disagg().disaggregation_decode_extra_slots
            )
        else:
            chunks_in_flight = 1 if get_schedule().disable_overlap_schedule else 2
            self._swa_cap = (
                per_request * num_reqs
                + chunks_in_flight * get_schedule().chunked_prefill_size
                + page_size
            )

    @staticmethod
    def is_applicable(kvc: KVCacheConfigurator) -> bool:
        """True when SWAChunkCache can be sized from explicit max requests."""
        if get_schedule().max_running_requests is None:
            return False
        if not get_memory().disable_radix_cache:
            return False
        if get_schedule().chunked_prefill_size is None:
            return False
        if kvc.sliding_window_size is None:
            return False
        return len(kvc.model_config.full_attention_layer_ids) > 0

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        # SWA pool sized tightly from the cap; the rest of the budget goes to full.
        swa_tokens = ceil_align(self._swa_cap, page_size)
        fixed_swa_bytes = (
            swa_tokens
            * self._swa_per_token
            * (self._swa_layers_num + self._draft_swa_layers_num)
        )
        if self._enable_unified_memory and self._draft_pool_bytes_per_token() > 0:
            full_tokens = self._max_unified_full_tokens(
                available_bytes, page_size, fixed_swa_tokens=swa_tokens
            )
        else:
            full_cell_size = (
                self._full_per_token
                * (self._full_layers_num + self._draft_full_layers_num)
                + self._swa_per_token * self._draft_swa_full_layers_num
            )
            full_tokens = (
                int((available_bytes - fixed_swa_bytes) // full_cell_size) // page_size
            ) * page_size
        if full_tokens <= 0:
            raise RuntimeError(
                f"SWA pool cap ({swa_tokens} tokens, "
                f"{fixed_swa_bytes / (1 << 30):.2f} GiB) leaves no room for the full "
                f"KV pool within the available {available_bytes / (1 << 30):.2f} GiB. "
                f"Reduce --max-running-requests, lower SGLANG_SWA_EVICTION_INTERVAL, "
                f"or increase --mem-fraction-static."
            )
        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        # Constrained max_total goes to the full pool; SWA stays at its cap.
        swa_tokens = ceil_align(self._swa_cap, page_size)
        full_tokens = (max_total_num_tokens // page_size) * page_size
        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=min(swa_tokens, max_total_num_tokens),
        )


@dataclass
class _DSV4PoolSizes:
    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


class DSV4PoolConfigurator(MemoryPoolConfigurator):
    """Configurator for DSV4 compressed-attention models.

    Splits available memory across full / swa / c4 / c128 + c4_state / c128_state
    pools. coeff is bytes_per_full_token (inflated by (T+D)/T when speculative
    decode reserves a draft worker, mirroring dflash's cell_size scaling); bias = 0.
    """

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
        cfg = kvc.model_config
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.indexer_head_dim = cfg.index_head_dim
        # HIP takes the FP4-accurate byte count here. The NVIDIA FP4 path
        # keeps the FP8 estimate.
        self.indexer_bytes_per_token = get_dsv4_indexer_bytes_per_token(
            self.indexer_head_dim,
            _is_hip and kvc.server_args.enable_deepseek_v4_fp4_indexer,
        )
        self.context_len = kvc.model_config.context_len
        # PP-local slice; matches DeepSeekV4TokenToKVPool's stage_ratios.
        self.compression_ratios = cfg.compress_ratios[
            kvc.layer_info.start_layer : kvc.layer_info.end_layer
        ]
        if kvc.ps.pp_size > 1:
            logger.info(
                f"DSV4 pool PP slice: rank={kvc.pp_group.rank_in_group} "
                f"layers=[{kvc.layer_info.start_layer},{kvc.layer_info.end_layer}) "
                f"local={len(self.compression_ratios)}/{len(cfg.compress_ratios)}"
            )
        self.swa_page_size = cfg.window_size
        self.sliding_window_size = kvc.sliding_window_size
        self.swa_ratio = get_schedule().swa_full_tokens_ratio
        self.is_speculative = get_spec().speculative_algorithm is not None
        self.online_c128_mtp_max_draft_tokens = max_speculative_num_draft_tokens() or 0
        self.requested_max_running_requests_per_worker = (
            get_schedule().max_running_requests // kvc.ps.attn_dp_size
            if get_schedule().max_running_requests is not None
            else None
        )
        self.disaggregation_mode = get_disagg().disaggregation_mode
        self.disaggregation_decode_extra_slots = (
            get_disagg().disaggregation_decode_extra_slots or 0
        )
        self.fp4_logits_workspace_enabled = bool(
            _is_hip
            and not kvc.is_draft_worker
            and kvc.server_args.enable_deepseek_v4_fp4_indexer
            and 4 in self.compression_ratios
            and (self.disaggregation_mode != "decode" or self.is_speculative)
        )
        self.fp4_logits_workspace_plan: Optional[FP4LogitsWorkspacePlan] = None
        fp4_workspace_context_len = (
            self.context_len + get_req_to_token_extra_context_len()
        )
        self.fp4_logits_workspace_max_seq_len = (
            fp4_logits_width_for_context(fp4_workspace_context_len, kvc.page_size)
            if self.fp4_logits_workspace_enabled
            else 0
        )
        if get_memory().enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            self.c4_shrink_factor = parse_hisparse_config().host_to_device_ratio
        else:
            self.c4_shrink_factor = 1
        assert self.c4_shrink_factor >= 1
        if self.c4_shrink_factor > 1:
            logger.info(f"HiSparse c4 host-to-device ratio = {self.c4_shrink_factor}")

        self.c4_ring_size = get_compress_state_ring_size(4, self.is_speculative)
        self.c128_ring_size = get_compress_state_ring_size(128, self.is_speculative)

        self.num_layers_total = len(self.compression_ratios)
        self.num_layers_ca4 = sum(1 for r in self.compression_ratios if r == 4)
        self.num_layers_ca128 = sum(1 for r in self.compression_ratios if r == 128)

        if self.is_speculative:
            # Ring is sized once here, so it must serve the largest adaptive tier.
            self._assert_ring_serves_draft_tokens(
                max_speculative_num_draft_tokens() or 0
            )

        self.bytes_per_full_token = self._get_bytes_per_full_token()
        if self.is_speculative:
            # Reserve memory for the speculative draft worker by inflating
            # per-token bytes by (target+draft)/target. Equivalent to dflash's
            # scale_kv_cell_size_per_token_for_dflash but applied to
            # bytes_per_full_token: tokens = avail / (bpft * (T+D)/T).
            draft_layers = 1
            target_layers = self.num_layers_total
            self.bytes_per_full_token *= (target_layers + draft_layers) / target_layers

        # Online c128 keeps a single in-progress (max, sum, kv) state per index
        # and assumes a strict forward-only schedule. Speculative decode (MTP)
        # would need rollback / replay across draft and verify, which the
        # online path doesn't support yet.
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            allow_experimental_online_c128_mtp = (
                envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
                and kvc.spec_algorithm.is_eagle()
            )
            assert kvc.spec_algorithm.is_none() or allow_experimental_online_c128_mtp, (
                "SGLANG_OPT_USE_ONLINE_COMPRESS does not support speculative decode "
                "(MTP) yet, except the experimental EAGLE topk=1 path gated by "
                "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1"
            )
            if allow_experimental_online_c128_mtp:
                assert self.online_c128_mtp_max_draft_tokens > 0, (
                    "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP requires "
                    "speculative_num_draft_tokens to be set."
                )
                logger.warning(
                    "DSV4 compressed attention: experimental online c128 + MTP enabled "
                    f"(EAGLE topk=1 only, "
                    f"draft_banks={self.online_c128_mtp_max_draft_tokens}). "
                    "Validate correctness carefully."
                )
            else:
                logger.info(
                    "DSV4 compressed attention: online c128 enabled (ring_size=1)"
                )

    def _plan_fp4_logits_workspace(
        self, available_bytes: int
    ) -> Optional[FP4LogitsWorkspacePlan]:
        if not self.fp4_logits_workspace_enabled:
            return None

        schedule = get_schedule()
        mem_fraction_static = schedule.mem_fraction_static
        if mem_fraction_static is None:
            mem_fraction_static = 0.9
        # ``available_bytes`` is the static-pool budget after preserving runtime
        # slack. Reconstruct a conservative share of that slack without another
        # device query; the allocation is checked once against live free memory
        # when the runner creates the workspace.
        runtime_headroom_bytes = int(
            available_bytes
            * max(0.0, 1.0 - mem_fraction_static)
            / max(mem_fraction_static, 1.0e-6)
        )
        free_memory_fraction = envs.SGLANG_DSV4_FP4_LOGITS_FREE_MEM_FRACTION.get()
        if not 0.0 < free_memory_fraction <= 1.0:
            raise ValueError(
                "SGLANG_DSV4_FP4_LOGITS_FREE_MEM_FRACTION must be in (0, 1], "
                f"got {free_memory_fraction}"
            )
        # At mem_fraction_static=1 there is no nominal activation headroom, but
        # this persistent workspace is charged directly against the token pool.
        # Reserve at least one maximum-width row from that static budget.
        row_bytes = self.fp4_logits_workspace_max_seq_len * 4
        runtime_headroom_bytes = max(
            runtime_headroom_bytes,
            math.ceil(row_bytes / free_memory_fraction),
        )

        prefill_buffer_rows = max_prefill_buffer_tokens()
        if prefill_buffer_rows <= 0:
            prefill_buffer_rows = schedule.max_prefill_tokens or 1
        max_prefill_rows = (
            1 if self.disaggregation_mode == "decode" else prefill_buffer_rows
        )
        # CP round-robin reindexes query rows before the local indexer runs.
        max_prefill_rows = ceil_div(
            max_prefill_rows, max(get_parallel().attn_cp_size, 1)
        )
        max_query_rows = max_prefill_rows
        if self.requested_max_running_requests_per_worker is not None:
            verify_rows = self.requested_max_running_requests_per_worker * max(
                self.online_c128_mtp_max_draft_tokens, 1
            )
            max_query_rows = max(max_query_rows, verify_rows)

        ceiling_mb = envs.SGLANG_DSV4_FP4_LOGITS_BUDGET_MB.get()
        if ceiling_mb < 0:
            raise ValueError(
                "SGLANG_DSV4_FP4_LOGITS_BUDGET_MB must be non-negative "
                f"(0 selects auto), got {ceiling_mb}"
            )
        max_workspace_bytes = ceiling_mb * (1 << 20) if ceiling_mb else None
        return plan_fp4_logits_workspace(
            max_seq_len=self.fp4_logits_workspace_max_seq_len,
            max_query_rows=max_query_rows,
            runtime_headroom_bytes=runtime_headroom_bytes,
            free_memory_fraction=free_memory_fraction,
            max_workspace_bytes=max_workspace_bytes,
        )

    def _assert_ring_serves_draft_tokens(self, num_draft_tokens: int) -> None:
        """A verify batch writes its whole optimistic tail into the ring, so ring
        capacity bounds the draft count."""
        for compress_ratio, ring_size, num_layers in (
            (4, self.c4_ring_size, self.num_layers_ca4),
            (128, self.c128_ring_size, self.num_layers_ca128),
        ):
            if num_layers == 0:
                continue
            if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
                # Online c128 keeps per-draft state instead of a ring; sized separately.
                continue
            max_draft_tokens = get_compress_state_write_pad(compress_ratio, ring_size)
            assert num_draft_tokens <= max_draft_tokens, (
                f"speculative_num_draft_tokens={num_draft_tokens} exceeds what the c{compress_ratio} "
                f"compress state ring can keep resident (ring_size={ring_size} serves at most "
                f"{max_draft_tokens} draft tokens). Lower the draft count, or grow the ring in "
                f"get_compress_state_ring_size()."
            )

    def _get_bytes_per_full_token(self) -> float:
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        c4_state_dtype_size, c128_state_dtype_size = (
            _get_dsv4_compress_state_dtype_sizes()
        )
        c4_state_bytes = 2 * 2 * attn_head_dim * c4_state_dtype_size
        # Online c128 stores (max, sum, kv) per slot (3*head_dim) instead of
        # raw (kv, score) (2*head_dim). Combined with ring_size=1 this still
        # nets a large reduction (~3/256x) but the per-slot bytes go up.
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        c128_state_bytes = (
            (3 if c128_online else 2 * 1) * attn_head_dim * c128_state_dtype_size
        )
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * c4_state_dtype_size

        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        # C128 state is request-scoped and is finalized after
        # max_running_requests is known, so it should not scale with
        # full-token capacity here.
        c128_state_ratio = 0

        c4_frac = 1 / (4 * self.c4_shrink_factor)
        return (
            self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * self.indexer_bytes_per_token * self.num_layers_ca4
            + self.swa_ratio * c4_state_ratio * c4_state_bytes * self.num_layers_ca4
            + c128_state_ratio * c128_state_bytes * self.num_layers_ca128
            + self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )

    def _compute_dsv4_sizes(self, full_token: int, page_size: int) -> _DSV4PoolSizes:
        full_token = full_token // page_size * page_size
        swa_tokens = int(full_token * self.swa_ratio) // page_size * page_size
        self.validate_swa_pool_size(swa_tokens, self.sliding_window_size, page_size)
        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            c4_state_pool_size=swa_tokens // self.swa_page_size * self.c4_ring_size,
            c128_state_pool_size=0,
        )

    def _get_num_req_slots(self, max_running_requests: int) -> int:
        if self.disaggregation_mode == "decode":
            return max_running_requests + self.disaggregation_decode_extra_slots + 1
        return max_running_requests + 1

    def _get_c128_state_fixed_bytes(self, max_running_requests: int) -> int:
        if self.num_layers_ca128 == 0:
            return 0

        _, c128_state_dtype_size = _get_dsv4_compress_state_dtype_sizes()
        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        num_req_slots = self._get_num_req_slots(max_running_requests)

        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            state_rows = num_req_slots + self.c128_ring_size + 1
            state_rows *= 1 + self.online_c128_mtp_max_draft_tokens
            state_last_dim = 3 * attn_head_dim
        else:
            state_pool_size = num_req_slots * self.c128_ring_size
            state_rows = state_pool_size + self.c128_ring_size + 1
            state_rows = ceil_div(state_rows, 128) * 128
            state_last_dim = 2 * attn_head_dim

        return (
            state_rows * state_last_dim * c128_state_dtype_size * self.num_layers_ca128
        )

    def _get_c128_state_fixed_bytes_for_token_capacity(
        self, token_capacity: int
    ) -> int:
        if self.requested_max_running_requests_per_worker is not None:
            return self._get_c128_state_fixed_bytes(
                self.requested_max_running_requests_per_worker
            )

        estimated = int(token_capacity / self.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)
        max_running_requests = min(estimated, token_capacity // 2)
        return self._get_c128_state_fixed_bytes(max_running_requests)

    def _to_config(self, sizes: _DSV4PoolSizes) -> MemoryPoolConfig:
        full = sizes.full_max_total_num_tokens
        fp4_plan = self.fp4_logits_workspace_plan
        swa = sizes.swa_max_total_num_tokens
        logger.info(
            f"DSV4 pool sizes: full={full}, swa={swa}, "
            f"c4={sizes.c4_max_total_num_tokens}, "
            f"c128={sizes.c128_max_total_num_tokens}, "
            f"c4_state={sizes.c4_state_pool_size}, "
            f"c128_state={sizes.c128_state_pool_size}"
        )
        return MemoryPoolConfig(
            max_total_num_tokens=full,
            full_max_total_num_tokens=full,
            swa_max_total_num_tokens=swa,
            c4_max_total_num_tokens=sizes.c4_max_total_num_tokens,
            c128_max_total_num_tokens=sizes.c128_max_total_num_tokens,
            c4_state_pool_size=sizes.c4_state_pool_size,
            c128_state_pool_size=sizes.c128_state_pool_size,
            dsv4_fp4_logits_workspace_bytes=(
                fp4_plan.capacity_bytes if fp4_plan is not None else 0
            ),
            dsv4_fp4_logits_workspace_max_seq_len=(
                fp4_plan.max_seq_len if fp4_plan is not None else 0
            ),
        )

    def finalize_with_max_running_requests(
        self, config: MemoryPoolConfig
    ) -> MemoryPoolConfig:
        assert config.max_running_requests is not None
        num_req_slots = self._get_num_req_slots(config.max_running_requests)
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            config.c128_state_pool_size = num_req_slots
        else:
            config.c128_state_pool_size = num_req_slots * self.c128_ring_size
        return config

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        assert page_size % 128 == 0, (
            "page_size must be multiple of 128 for compressed attention"
        )

        if self.requested_max_running_requests_per_worker is not None:
            c128_state_fixed_bytes = self._get_c128_state_fixed_bytes(
                self.requested_max_running_requests_per_worker
            )
        else:
            full_token = int(available_bytes / self.bytes_per_full_token)
            c128_state_fixed_bytes = (
                self._get_c128_state_fixed_bytes_for_token_capacity(full_token)
            )

        self.fp4_logits_workspace_plan = self._plan_fp4_logits_workspace(
            available_bytes
        )
        fp4_logits_workspace_bytes = (
            self.fp4_logits_workspace_plan.capacity_bytes
            if self.fp4_logits_workspace_plan is not None
            else 0
        )
        available_bytes_for_tokens = max(
            available_bytes - c128_state_fixed_bytes - fp4_logits_workspace_bytes,
            0,
        )
        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)

        sizes = self._compute_dsv4_sizes(full_token, page_size)
        logger.info(
            f"DSV4 memory calculation: "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"c128_state_fixed={c128_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"fp4_logits_workspace={fp4_logits_workspace_bytes / (1 << 20):.2f} MiB, "
            f"full_token={sizes.full_max_total_num_tokens}"
        )
        return self._to_config(sizes)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        assert page_size % 128 == 0, (
            "page_size must be multiple of 128 for compressed attention"
        )
        # A token-capped configuration bypasses byte profiling. Still resolve
        # the workspace from the equivalent token-pool footprint so the runner
        # receives the same explicit capacity contract.
        if self.fp4_logits_workspace_plan is None:
            estimated_available_bytes = max(
                int(max_total_num_tokens * self.bytes_per_full_token), 1
            )
            self.fp4_logits_workspace_plan = self._plan_fp4_logits_workspace(
                estimated_available_bytes
            )
        sizes = self._compute_dsv4_sizes(max_total_num_tokens, page_size)
        return self._to_config(sizes)


def create_memory_pool_configurator(
    kvc: KVCacheConfigurator,
) -> MemoryPoolConfigurator:
    """Factory: select the right configurator for the model architecture."""
    if is_deepseek_v4(kvc.model_config.hf_config) and kvc.is_hybrid_swa:
        return DSV4PoolConfigurator(kvc)
    if kvc.is_hybrid_swa:
        if SWAChunkCapPoolConfigurator.is_applicable(kvc):
            return SWAChunkCapPoolConfigurator(kvc)
        return HybridSWAPoolConfigurator(kvc)
    # Future: MambaPoolConfigurator
    return DefaultPoolConfigurator(kvc)
