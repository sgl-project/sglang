"""Memory pool configurators for profiling and sizing KV cache pools.

Each model architecture has its own configurator that computes pool sizes
from available GPU memory using a unified coeff+bias model:

    available_bytes = max_tokens * coeff + bias
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.arg_groups.overrides import resolving_view
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
from sglang.srt.mem_cache.allocation_sizing import get_alloc_len_per_decode
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    collect_sources_by_ratio,
    get_compress_state_write_pad,
    get_dsv4_indexer_bytes_per_token,
    get_swa_ring_size,
    resolve_compress_state_ring_size,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    get_minimax_sparse_index_dtype,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
    max_speculative_num_draft_tokens,
)
from sglang.srt.utils.common import (
    ceil_align,
    ceil_div,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
    spec_decode_alloc_len_per_request,
)

_is_hip = is_hip()
_is_npu = is_npu()


@dataclass
class MemoryPoolConfig:
    """Resolved memory pool config, shared between target and draft workers."""

    max_total_num_tokens: int
    max_running_requests: Optional[int] = None
    full_max_total_num_tokens: Optional[int] = None
    swa_max_total_num_tokens: Optional[int] = None
    unified_memory_pool_bytes: Optional[int] = None

    # DSV4 compressed-attention pool sizes (target only; draft workers leave at 0).
    c4_max_total_num_tokens: int = 0
    c128_max_total_num_tokens: int = 0
    c4_state_pool_size: int = 0
    c128_state_pool_size: int = 0

    mem_fraction_static: Optional[float] = None

    # Unified pool only: the profiled byte budget the buffer is sized from,
    # instead of re-summing token counts; None when a user token cap is the budget.
    unified_total_bytes: Optional[int] = None

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

logger = logging.getLogger(__name__)


def _dflash_draft_cell_size(kvc: KVCacheConfigurator) -> int:
    """Bytes/token the DFLASH draft KV pool adds, 0 if none; replicated across DCP
    ranks because the draft pool spans the widened virtual location space."""
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
    """Standard models (MHA, MLA, DSA, FP4): coeff = cell_size, bias = 0."""

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
            and kvc.pp_size > 1
        )
        self._zero_kv_max_tokens = (
            torch.iinfo(torch.int64).max
            if has_kv_on_another_pp_stage
            else get_schedule().max_total_tokens or kvc.model_config.context_len
        )

        # EAGLE/STANDALONE: assumes the draft shares the target's per-layer KV size,
        # which holds for EAGLE/MTP drafts that reuse the target's attention config.
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

        # DFLASH/DSPARK: the draft's per-token KV cost can differ from the target's
        # (e.g. MLA target, per-head K/V draft), so size from the draft config.
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
        """Compute per-token KV cache cost in bytes."""
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
            if envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get():
                # NPU sparse KV offload uses an index-only device pool.
                from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
                    get_sparsity_driven_kv_offload_cell_size,
                )

                offload_cell_size = get_sparsity_driven_kv_offload_cell_size(
                    model_config=model_config,
                    use_mla_backend=kvc.use_mla_backend,
                    num_layers=num_layers,
                    element_size=kv_size,
                )
                if offload_cell_size is not None:
                    return offload_cell_size

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
            from sglang.srt.server_args import m3_fp8_attn_gemm_enabled

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
            indexer_dtype_size = torch._utils._element_size(
                get_minimax_sparse_index_dtype(
                    fp8_attn_gemm=m3_fp8_attn_gemm_enabled(
                        resolving_view(kvc.server_args)
                    ),
                    kv_cache_dtype=kvc.kv_cache_dtype,
                    model_dtype=kvc.model_dtype,
                )
            )

            full_pool_ratio = 1
            if get_memory().enable_hisparse:
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                full_pool_ratio = parse_hisparse_config().host_to_device_ratio

            main_pool_bytes = (
                (num_dense * full_pool_ratio + num_sparse)
                * 2
                * kv_heads
                * head_dim
                * kv_size
            )
            indexer_bytes = (
                (num_indexer_kv * 2 + num_indexer_k_only)
                * indexer_head_dim
                * indexer_dtype_size
            )
            # FP4 scale buffer adjustment doesn't apply to MiniMax sparse:
            # cell_size is already a sum over heterogeneous sub-pools.
            return main_pool_bytes + indexer_bytes * full_pool_ratio
        else:
            n = model_config.get_num_kv_heads(tp_size, dcp_size)
            cell_size = (
                n
                * (model_config.head_dim + model_config.v_head_dim)
                * effective_num_layers
                * kv_size
            )

            if is_float4_e2m1fn_x2(kv_cache_dtype):
                from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
                    get_kv_cache_quant_method,
                    resolve_kv_cache_quant,
                )

                quant_name = resolve_kv_cache_quant(kvc.kv_cache_dtype_str)
                if quant_name is None:
                    raise ValueError(
                        "FP4 storage dtype requires an explicit KV recipe name."
                    )
                quant_method = get_kv_cache_quant_method(
                    quant_name,
                    num_layers=effective_num_layers,
                    device=kvc.device,
                    page_size=kvc.page_size,
                )
                quant_method.configure_attention_backends_from_server_args(
                    kvc.server_args
                )
                cell_size = quant_method.compute_cell_size(
                    n,
                    model_config.head_dim,
                    effective_num_layers,
                    kv_size,
                )
            elif self.kv_cache_dtype_str == "mxfp8":
                scale_block_size = 32
                cell_size += (
                    n * (model_config.head_dim + model_config.v_head_dim) * num_layers
                ) // scale_block_size

        cell_size += self._compute_qsa_cell_size(
            hf_config=model_config.hf_config, num_layers=num_layers
        )
        return cell_size

    @staticmethod
    def _compute_qsa_cell_size(*, hf_config, num_layers: int) -> int:
        from sglang.srt.layers.attention.qsa.config import (
            parse_qsa_profile,
        )
        from sglang.srt.mem_cache.qsa_kv_pool import (
            QSATokenToKVPool,
        )

        if num_layers == 0:
            return 0
        qsa_profile = parse_qsa_profile(hf_config)
        if qsa_profile is None:
            return 0
        return QSATokenToKVPool.qsa_bytes_per_token(
            kv_heads=qsa_profile.kv_heads,
            head_dim=qsa_profile.head_dim,
            compress_ratio=qsa_profile.compress_ratio,
            num_layers=num_layers,
        )

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
        if _is_npu:
            from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

            dtype = kvc.kv_cache_dtype
            # GPU sizing above assumes FP8 indexers; NPU also needs BF16 sizing.
            if dtype != torch.float8_e4m3fn:
                indexer_size_per_token = index_head_dim
                element_size = torch._utils._element_size(dtype)
            if not is_npu_arch35():
                allocate_all_layers = True
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
                # GLM-5.3 hybrid-layer support is intentionally limited to the
                # non-LayerSplit pool below.
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
        available_bytes = max(available_bytes, 0)
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
    """Splits memory between the full and SWA pools of MHA/MLA sliding-window models."""

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
        model_config = kvc.model_config
        kv_cache_dtype = kvc.kv_cache_dtype
        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_parallel().attn_tp_size

        self._full_layers_num = len(kvc.layer_info.full_attention_layer_ids)
        self._swa_layers_num = len(kvc.layer_info.swa_attention_layer_ids)
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
        # Bytes per token of max_total_num_tokens: full_tokens when hybrid, else
        # swa_tokens with no ratio, since all-SWA has no full pool to relate to.
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

    def _unified_pool_bytes(self, full_tokens: int, swa_tokens: int) -> int:
        return (
            full_tokens * self._full_per_token * self._full_layers_num
            + swa_tokens * self._swa_per_token * self._swa_layers_num
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
            target_bytes = self._unified_pool_bytes(full_tokens, swa_tokens)
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
        def align_page_size(x: int) -> int:
            return (x // page_size) * page_size

        if self._full_layers_num == 0:
            # All-SWA: no full pool, max_total = actual SWA pool size.
            # Ratio is not applied -- see _recompute_cell_size.
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

        if not self._enable_unified_memory:
            self.validate_swa_pool_size(
                swa_tokens, self._sliding_window_size, self._page_size
            )

        logger.info(
            f"Use sliding window memory pool. "
            f"full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
        )

        return self._make_pool_config(full_tokens, swa_tokens)

    def _make_pool_config(self, full_tokens: int, swa_tokens: int) -> MemoryPoolConfig:
        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
            unified_memory_pool_bytes=(
                self._unified_pool_bytes(full_tokens, swa_tokens)
                if self._enable_unified_memory
                else None
            ),
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        if self._enable_unified_memory and self._full_layers_num > 0:
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


def compute_swa_request_cap(*, page_size: int, window: int, attn_dp_size: int) -> int:
    """Worst-case SWA slots the scheduler holds live at max_running_requests."""
    draft_tokens = get_spec().speculative_num_draft_tokens or 1
    eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())

    # __________[padding][eviction_interval][window]
    # Padding to make sure eviction point is page-aligned.
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

    num_reqs = get_schedule().max_running_requests // attn_dp_size
    if get_disagg().disaggregation_mode == "decode":
        return (
            per_request * num_reqs
            + (window + page_size) * get_disagg().disaggregation_decode_extra_slots
        )
    else:
        chunks_in_flight = 1 if get_schedule().disable_overlap_schedule else 2
        return (
            per_request * num_reqs
            + chunks_in_flight * get_schedule().chunked_prefill_size
            + page_size
        )


class SWAChunkCapPoolConfigurator(HybridSWAPoolConfigurator):
    """Hybrid SWA with the SWA pool sized from the explicit max_running_requests
    worst case instead of swa_full_tokens_ratio; the rest goes to the full pool."""

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
        super().__init__(kvc)
        assert self._full_layers_num > 0

        self._swa_cap = compute_swa_request_cap(
            page_size=kvc.page_size,
            window=kvc.sliding_window_size,
            attn_dp_size=kvc.attn_dp_size,
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
        return len(kvc.layer_info.full_attention_layer_ids) > 0

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
        if self._enable_unified_memory:
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
        return self._make_pool_config(full_tokens, swa_tokens)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        # Constrained max_total goes to the full pool; SWA stays at its cap.
        swa_tokens = ceil_align(self._swa_cap, page_size)
        full_tokens = (max_total_num_tokens // page_size) * page_size
        return self._make_pool_config(
            full_tokens, min(swa_tokens, max_total_num_tokens)
        )


# Used when --swa-full-tokens-ratio is at its default and cap mode is unusable.
DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO = 0.1


def _operator_swa_full_tokens_ratio() -> Optional[float]:
    """The operator's --swa-full-tokens-ratio, or None when it was not given."""
    schedule = get_schedule()
    if not schedule._swa_full_tokens_ratio_explicitly_set:
        return None
    return schedule.swa_full_tokens_ratio


@dataclass
class _DSV4PoolSizes:
    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


class DSV4PoolConfigurator(MemoryPoolConfigurator):
    """DSV4 compressed attention: coeff is bytes_per_full_token, inflated by (T+D)/T
    for a draft worker; bias is the request-scoped pools that do not scale with it."""

    # object.__new__ stubs (SWA floor tests) skip __init__
    _dspark_draft_on_bf16 = False

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
        cfg = kvc.model_config
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.indexer_head_dim = cfg.index_head_dim
        self.attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_fp8,
            is_unified_kv_triton,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            dsv4_unified_row_bytes,
        )

        # Resolve the unified-kv gate before any sizing so the two cannot drift.
        self._unified = is_unified_kv_triton()
        self._unified_fp8 = is_unified_kv_fp8()
        # DSpark draft still allocates a bf16 ring; target fp8 * (T+1)/T would
        # under-count that ring (640 vs 1024). MTP keeps the old inflation.
        self._dspark_draft_on_bf16 = bool(
            self._unified_fp8 and kvc.spec_algorithm.is_dspark()
        )
        # Row width across both unified pools: 1024 B bf16, 640 B fp8.
        self._unified_row_bytes = dsv4_unified_row_bytes(
            self.qk_nope_head_dim, self.qk_rope_head_dim, self._unified_fp8
        )
        if self._unified:
            # Unified_kv stores the whole latent: one bf16 row, or fp8 nope + bf16 rope.
            self.kv_bytes = self._unified_row_bytes
        elif get_exec().kernel.dsv4_attn_backend == "trtllm":
            self.kv_bytes = self.attn_head_dim
        else:
            # One FlashMLA-layout latent slot, in bytes.
            self.kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8
        # HIP takes the FP4-accurate byte count here. The NVIDIA FP4 path
        # keeps the FP8 estimate.
        self.indexer_bytes_per_token = get_dsv4_indexer_bytes_per_token(
            self.indexer_head_dim,
            _is_hip and get_exec().kernel.enable_deepseek_v4_fp4_indexer,
        )
        self.context_len = kvc.model_config.context_len
        # PP-local slice; matches DeepSeekV4TokenToKVPool's stage_ratios.
        stage = range(kvc.layer_info.start_layer, kvc.layer_info.end_layer)
        self.stage_compress_ratios = cfg.compress_ratios[stage.start : stage.stop]
        if kvc.pp_size > 1:
            logger.info(
                f"DSV4 pool PP slice: rank={kvc.pp_group.rank_in_group} "
                f"layers=[{stage.start},{stage.stop}) "
                f"local={len(self.stage_compress_ratios)}/{len(cfg.compress_ratios)}"
            )
        # Layers of this stage that own compressed storage, per ratio. Same rule
        # as the pool, so a ratio budgeted here is one it allocates.
        self.stage_owner_layers = collect_sources_by_ratio(
            cfg.compress_ratios, cfg.hf_config.kv_source_layer_ids, stage
        )
        self.operator_swa_ratio = _operator_swa_full_tokens_ratio()
        self.swa_ratio = (
            self.operator_swa_ratio
            if self.operator_swa_ratio is not None
            else DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO
        )
        self.sliding_window_size = kvc.sliding_window_size
        self.page_size = kvc.page_size
        self.is_speculative = get_spec().speculative_algorithm is not None
        self.online_c128_mtp_max_draft_tokens = max_speculative_num_draft_tokens() or 0
        self.attn_dp_size = kvc.attn_dp_size
        self.requested_max_running_requests_per_worker = (
            get_schedule().max_running_requests // kvc.attn_dp_size
            if get_schedule().max_running_requests is not None
            else None
        )
        self.disaggregation_mode = get_disagg().disaggregation_mode
        self.disaggregation_decode_extra_slots = (
            get_disagg().disaggregation_decode_extra_slots or 0
        )
        if get_memory().enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            self.c4_shrink_factor = parse_hisparse_config().host_to_device_ratio
        else:
            self.c4_shrink_factor = 1
        assert self.c4_shrink_factor >= 1
        if self.c4_shrink_factor > 1:
            logger.info(f"HiSparse c4 host-to-device ratio = {self.c4_shrink_factor}")

        # Ratio 1 keeps no compress state, so it has no ring.
        self.ring_sizes = {
            ratio: resolve_compress_state_ring_size(ratio)
            for ratio in self.stage_owner_layers
            if ratio != 1
        }

        self.num_layers_total = len(self.stage_compress_ratios)
        # The low-ratio indexer pools are built with force_fp4=True
        # (deepseek_v4_memory_pool), so they are fp4 whatever dtype c4 uses.
        self.low_ratio_index_bytes = get_dsv4_indexer_bytes_per_token(
            self.indexer_head_dim, use_fp4_indexer=True
        )

        # kvc.sliding_window_size is None on a runner without SWA layers.
        self._swa_ring_size = get_swa_ring_size(cfg.window_size, self.is_speculative)
        self._spec_infl = 1.0

        # The unified pool ignores --kv-cache-dtype; V4 resolves "auto" to fp8_e4m3
        # (overrides.py _deepseek_v4_kv_cache_dtype), so only bfloat16 is explicit.
        if self._unified_fp8 and self.kv_cache_dtype_str == "bfloat16":
            logger.warning(
                "--kv-cache-dtype=bfloat16 is ignored on the unified_kv path; "
                "SGLANG_DSV4_UNIFIED_KV_FP8=1 stores the latent as fp8. Unset the "
                "env switch to get a bf16 unified pool."
            )

        # get_contiguous_buf_infos prices a row as buf[0].nbytes, which under fp8
        # covers the nope pool only; fail at startup rather than at the first transfer.
        # TODO(danli103): drop this once the transfer ships the rope pool.
        if self._unified_fp8 and self.disaggregation_mode != "null":
            raise ValueError(
                "SGLANG_DSV4_UNIFIED_KV_FP8=1 does not support PD disaggregation "
                f"(disaggregation_mode={self.disaggregation_mode!r}). Unset the fp8 "
                "switch or run without disaggregation."
            )

        if self.is_speculative:
            # Ring is sized once here, so it must serve the largest adaptive tier.
            self._assert_ring_serves_draft_tokens(
                max_speculative_num_draft_tokens() or 0
            )

        self.encoder_replay = get_exec().features.enable_encoder_swa_bounded_replay
        self.paged_draft_layers = 0
        if self.encoder_replay and kvc.spec_algorithm.is_dspark():
            self.paged_draft_layers = int(
                kvc.spec_aux_config.dflash_draft_num_layers or 0
            )
            assert self.paged_draft_layers > 0, "DSpark draft layer count is required"
        self.request_window_bytes = 0
        if self.encoder_replay:
            slots = self.requested_max_running_requests_per_worker + 1
            capacity = ceil_align(
                self.sliding_window_size + self.online_c128_mtp_max_draft_tokens,
                self.page_size,
            )
            layers = self.num_layers_total
            scratch = (
                max(
                    get_schedule().chunked_prefill_size,
                    slots * max(128, self.online_c128_mtp_max_draft_tokens),
                )
                + slots * 128
                + self.page_size
            )
            self.request_window_bytes = (
                (slots * capacity + self.page_size) * layers * (self.kv_bytes + 16)
                + 4 * scratch * (self.kv_bytes + 16)
                + slots * 3 * 16 * self.attn_head_dim * 8
            )
            if not self.paged_draft_layers:
                self.swa_ratio = 0
        self.swa_prefix_tails = self._resolve_swa_prefix_tails()
        self.swa_cap_tokens = (
            0
            if self.encoder_replay and not self.paged_draft_layers
            else self._resolve_swa_cap_tokens()
        )
        self.bytes_per_swa_token = self._get_bytes_per_swa_token()
        self.bytes_per_full_token = self._get_bytes_per_full_token()
        if self.is_speculative and not self.encoder_replay:
            # Reserve the draft worker by inflating per-token bytes by
            # (target+draft)/target, as scale_kv_cell_size_per_token_for_dflash does.
            draft_layers = 1
            target_layers = self.num_layers_total
            self._spec_infl = (target_layers + draft_layers) / target_layers
            self.bytes_per_full_token *= self._spec_infl
            self.bytes_per_swa_token *= self._spec_infl

        # Online c128 keeps one in-progress (max, sum, kv) state per index and
        # assumes forward-only; MTP would need rollback across draft and verify.
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

    def num_layers(self, ratio: int) -> int:
        """Layers of this stage owning ratio's compressed storage."""
        return len(self.stage_owner_layers.get(ratio, ()))

    def _assert_ring_serves_draft_tokens(self, num_draft_tokens: int) -> None:
        """A verify batch writes its whole optimistic tail into the ring, so ring
        capacity bounds the draft count."""
        for compress_ratio in (4, 128):
            if self.num_layers(compress_ratio) == 0:
                continue
            if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
                # Online c128 keeps per-draft state instead of a ring; sized separately.
                continue
            ring_size = self.ring_sizes[compress_ratio]
            max_draft_tokens = get_compress_state_write_pad(compress_ratio, ring_size)
            assert num_draft_tokens <= max_draft_tokens, (
                f"speculative_num_draft_tokens={num_draft_tokens} exceeds what the c{compress_ratio} "
                f"compress state ring can keep resident (ring_size={ring_size} serves at most "
                f"{max_draft_tokens} draft tokens). Lower the draft count, or grow the ring in "
                f"get_compress_state_ring_size()."
            )

    def _resolve_swa_prefix_tails(self) -> int:
        """Cached prefix tails cap mode keeps addressable: a prefix is reusable only
        while its last sliding_window tokens still hold SWA slots."""
        prefix_tails = get_schedule().swa_prefix_tails
        if prefix_tails is not None:
            return prefix_tails
        if get_memory().disable_radix_cache:
            # Nothing is kept for reuse, so the request cap alone bounds the pool.
            return 0
        max_running_requests = self.requested_max_running_requests_per_worker
        return 4 * max_running_requests if max_running_requests is not None else 0

    def _resolve_swa_cap_tokens(self) -> Optional[int]:
        """SWA slots to reserve in cap mode, None to keep ratio sizing. Cap mode
        budgets from the request cap plus radix headroom, not full_tokens."""
        if self.operator_swa_ratio is not None:
            return None
        if self._unified:
            # Ring mode: SWA is a fixed per-request ring, with no paged pool to size.
            return None
        max_running_requests = self.requested_max_running_requests_per_worker
        if max_running_requests is None or self.sliding_window_size is None:
            return None
        chunked_prefill_size = get_schedule().chunked_prefill_size
        if self.disaggregation_mode != "decode" and (
            chunked_prefill_size is None or chunked_prefill_size <= 0
        ):
            return None

        cap = compute_swa_request_cap(
            page_size=self.page_size,
            window=self.sliding_window_size,
            attn_dp_size=self.attn_dp_size,
        )
        headroom = self.swa_prefix_tails * (self.sliding_window_size + self.page_size)
        return ceil_align(cap + headroom, self.page_size)

    def _get_paged_kv_bytes_per_token(self, compress_ratio: int = 0) -> float:
        # Unified rings, the NPU pool and the trtllm uniform-FP8 pool do not go
        # through DeepSeekV4SingleKVPool.create_buffer, so they carry no page pad.
        if self._unified or _is_npu or get_exec().kernel.dsv4_attn_backend == "trtllm":
            return self.kv_bytes
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            resolve_compressed_kv_layout,
            select_dsv4_kv_layout,
        )

        layout, compressed_option = select_dsv4_kv_layout()
        page_size = self.page_size
        if compress_ratio:
            layout = resolve_compressed_kv_layout(
                layout, compress_ratio, compressed_option
            )
            page_size = self.page_size // compress_ratio
        return layout.page_bytes(page_size) / page_size

    def _get_bytes_per_swa_token(self) -> float:
        """Bytes one SWA slot costs across the stage. c4_state_pool_size = swa_tokens
        // page_size * ring, so c4 compress state is priced per SWA slot too."""
        if self.encoder_replay:
            # Target SWA lives in the request window; only the draft owns paged SWA
            # bytes, and its layers carry no compressed state.
            return self.kv_bytes * self.paged_draft_layers
        c4_state_dtype_size, _ = _get_dsv4_compress_state_dtype_sizes()
        c4_state_bytes = 2 * 2 * self.attn_head_dim * c4_state_dtype_size
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * c4_state_dtype_size

        c4_state_ratio = self.ring_sizes.get(4, 0) / self.page_size
        return self._get_paged_kv_bytes_per_token() * self.num_layers_total + (
            c4_state_ratio
            * (c4_state_bytes + c4_indexer_state_bytes)
            * self.num_layers(4)
        )

    def _compressed_bytes_per_full_token(self, ratio: int) -> float:
        """Compressed KV (+ indexer) bytes one full token adds per layer of `ratio`;
        compress state is priced per SWA slot (c4) or as fixed bytes (c128, ratio 2)."""
        if ratio in (1, 2):
            return (self.kv_bytes + self.low_ratio_index_bytes) / ratio
        if ratio == 4:
            c4_frac = 1 / (4 * self.c4_shrink_factor)
            return (
                c4_frac * self._get_paged_kv_bytes_per_token(4)
                + 1 / 4 * self.indexer_bytes_per_token
            )
        assert ratio == 128, f"unsupported compression ratio: {ratio}"
        return 1 / 128 * self._get_paged_kv_bytes_per_token(128)

    def _get_bytes_per_full_token(self) -> float:
        # Cap mode and ring mode both move the SWA pool and the c4 state that
        # follows it out of the coefficient and into fixed bytes.
        swa_ratio = (
            0 if self._unified or self.swa_cap_tokens is not None else self.swa_ratio
        )
        return swa_ratio * self.bytes_per_swa_token + sum(
            self._compressed_bytes_per_full_token(ratio) * len(layers)
            for ratio, layers in self.stage_owner_layers.items()
        )

    def _get_swa_fixed_bytes(self) -> float:
        """Bias bytes of the encoder-replay request window plus the cap-mode pool."""
        paged_bytes = (
            0
            if self.swa_cap_tokens is None
            else self.swa_cap_tokens * self.bytes_per_swa_token
        )
        return self.request_window_bytes + paged_bytes

    def _get_swa_tokens(self, full_token: int, page_size: int) -> int:
        # swa_cap_tokens was already page-aligned at resolve time.
        if self.swa_cap_tokens is None:
            return int(full_token * self.swa_ratio) // page_size * page_size
        return self.swa_cap_tokens

    def _compute_dsv4_sizes(self, full_token: int, page_size: int) -> _DSV4PoolSizes:
        full_token = full_token // page_size * page_size
        swa_tokens = self._get_swa_tokens(full_token, page_size)
        if self.swa_cap_tokens is None:
            # Only ratio sizing can under-size a request: cap mode sizes from the
            # request floor, and encoder replay deliberately runs swa_tokens == 0.
            if not self._unified:
                self.validate_swa_pool_size(
                    swa_tokens, self.sliding_window_size, page_size
                )
            source = "explicit" if self.operator_swa_ratio is not None else "default"
            mode = (
                "ring (paged swa_tokens vestigial)"
                if self._unified
                else f"ratio ({source})"
            )
            logger.info(
                f"DSV4 SWA sizing: mode={mode}, swa_tokens={swa_tokens}, "
                f"swa_full_tokens_ratio={self.swa_ratio}"
            )
        else:
            logger.info(
                f"DSV4 SWA sizing: mode=cap, swa_tokens={swa_tokens}, "
                f"request_cap+headroom={self.swa_cap_tokens}, "
                f"prefix_tails={self.swa_prefix_tails}"
            )
        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            # Unified_kv: request-scoped, finalized once concurrency is known.
            # Otherwise the ring is addressed per SWA page (swa_loc // page_size).
            c4_state_pool_size=(
                0
                if self._unified
                else swa_tokens // page_size * self.ring_sizes.get(4, 0)
            ),
            c128_state_pool_size=0,
        )

    def _get_num_req_slots(self, max_running_requests: int) -> int:
        if self.disaggregation_mode == "decode":
            return max_running_requests + self.disaggregation_decode_extra_slots + 1
        return max_running_requests + 1

    def _get_c128_state_fixed_bytes(self, max_running_requests: int) -> int:
        num_layers = self.num_layers(128)
        if num_layers == 0:
            return 0

        _, c128_state_dtype_size = _get_dsv4_compress_state_dtype_sizes()
        num_req_slots = self._get_num_req_slots(max_running_requests)
        ring_size = self.ring_sizes[128]

        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            state_rows = num_req_slots + ring_size + 1
            state_rows *= 1 + self.online_c128_mtp_max_draft_tokens
            state_last_dim = 3 * self.attn_head_dim
        else:
            state_pool_size = num_req_slots * ring_size
            state_rows = state_pool_size + ring_size + 1
            state_rows = ceil_div(state_rows, 128) * 128
            state_last_dim = 2 * self.attn_head_dim

        return state_rows * state_last_dim * c128_state_dtype_size * num_layers

    def _get_c2_state_fixed_bytes(self, max_running_requests: int) -> int:
        """Ratio-2 pending-pair state, one fp32 (kv, score) ring per request slot
        on each ratio-2 kv_source layer; mirrors _make_pair_state_pool."""
        num_layers = self.num_layers(2)
        if num_layers == 0:
            return 0

        num_req_slots = self._get_num_req_slots(max_running_requests)
        ring_size = self.ring_sizes[2]
        # CompressStatePool allocates `size + ring_size + 1` rows, padded to the ratio.
        state_rows = num_req_slots * ring_size + ring_size + 1
        state_rows = ceil_div(state_rows, 2) * 2
        state_last_dim = 2 * self.attn_head_dim
        return state_rows * state_last_dim * torch.float32.itemsize * num_layers

    def _unified_c4_state_pool_size(self, max_running_requests: int) -> int:
        # Unified C4 state loc is req_pool_idx * c4_ring_size + pos % c4_ring_size.
        num_req_slots = self._get_num_req_slots(max_running_requests)
        return num_req_slots * self.ring_sizes[4]

    def _fixed_c4_state_bytes(self, max_running_requests: int) -> int:
        num_layers = self.num_layers(4)
        if not self._unified or num_layers == 0:
            return 0

        c4_state_dtype_size, _ = _get_dsv4_compress_state_dtype_sizes()
        # Mirror CompressStatePool.__init__: it allocates `size + ring_size + 1`
        # rows, padded to the compress ratio.
        state_rows = self._unified_c4_state_pool_size(max_running_requests)
        state_rows = ceil_div(state_rows + self.ring_sizes[4] + 1, 4) * 4
        # overlap c4: last_dim = 2 * (1 + overlap) * head_dim = 4 * head_dim.
        core_bytes = 4 * self.attn_head_dim * c4_state_dtype_size
        indexer_bytes = 4 * self.indexer_head_dim * c4_state_dtype_size
        return state_rows * (core_bytes + indexer_bytes) * num_layers

    def _resolve_max_running_requests_per_worker(self, available_bytes: int) -> int:
        # Approximates ModelRunner._resolve_max_num_reqs. Over-estimating is safe:
        # a larger fixed bias yields a smaller full_token.
        if self.requested_max_running_requests_per_worker is not None:
            return self.requested_max_running_requests_per_worker

        full_token = int(available_bytes / self.bytes_per_full_token)
        estimated = int(full_token / self.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)
        return min(estimated, full_token // 2)

    def _fixed_swa_bytes(self, max_running_requests: int) -> int:
        """Unified_kv SWA ring bytes, sized by req slots; 0 off unified, where SWA is
        priced per token. DSpark+fp8 adds a bf16 draft ring instead of _spec_infl."""
        if not self._unified:
            return 0
        num_req_slots = self._get_num_req_slots(max_running_requests)
        target_ring = (
            num_req_slots
            * self._swa_ring_size
            * self._unified_row_bytes
            * self.num_layers_total
        )
        if self._dspark_draft_on_bf16:
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                dsv4_unified_row_bytes,
            )

            draft_row = dsv4_unified_row_bytes(
                self.qk_nope_head_dim, self.qk_rope_head_dim, fp8=False
            )
            # 1 layer is what shipped DSpark drafts allocate; a multi-layer draft
            # under-counts ~9 MB/layer, covered by the (T+1)/T inflation.
            draft_ring = num_req_slots * self._swa_ring_size * draft_row
            return int(target_ring + draft_ring)
        return int(target_ring * self._spec_infl)

    def _to_config(self, sizes: _DSV4PoolSizes) -> MemoryPoolConfig:
        full = sizes.full_max_total_num_tokens
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
        )

    def finalize_with_max_running_requests(
        self, config: MemoryPoolConfig
    ) -> MemoryPoolConfig:
        assert config.max_running_requests is not None
        num_req_slots = self._get_num_req_slots(config.max_running_requests)
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            config.c128_state_pool_size = num_req_slots
        else:
            config.c128_state_pool_size = num_req_slots * self.ring_sizes.get(128, 0)
        # Ring mode: C4 state is request-scoped, so size it from the known concurrency.
        if self._unified and self.num_layers(4) > 0:
            config.c4_state_pool_size = self._unified_c4_state_pool_size(
                config.max_running_requests
            )
        return config

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        assert page_size % 128 == 0, (
            "page_size must be multiple of 128 for compressed attention"
        )

        max_running_requests_per_worker = self._resolve_max_running_requests_per_worker(
            available_bytes
        )
        c128_state_fixed_bytes = self._get_c128_state_fixed_bytes(
            max_running_requests_per_worker
        )
        c2_state_fixed_bytes = self._get_c2_state_fixed_bytes(
            max_running_requests_per_worker
        )
        swa_ring_fixed_bytes = self._fixed_swa_bytes(max_running_requests_per_worker)
        c4_state_fixed_bytes = self._fixed_c4_state_bytes(
            max_running_requests_per_worker
        )

        swa_fixed_bytes = self._get_swa_fixed_bytes()
        fixed_bytes = (
            c128_state_fixed_bytes
            + c2_state_fixed_bytes
            + swa_fixed_bytes
            + swa_ring_fixed_bytes
            + c4_state_fixed_bytes
        )
        available_bytes_for_tokens = max(available_bytes - fixed_bytes, 0)
        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)
        if full_token <= 0 and self.swa_cap_tokens is not None:
            raise RuntimeError(
                f"The DSV4 SWA pool cap ({self.swa_cap_tokens} tokens, "
                f"{swa_fixed_bytes / (1 << 30):.2f} GB) leaves no room for the full "
                f"KV pool within the available {available_bytes / (1 << 30):.2f} GB. "
                f"Reduce --max-running-requests, lower --swa-prefix-tails "
                f"or SGLANG_SWA_EVICTION_INTERVAL, or increase --mem-fraction-static."
            )

        sizes = self._compute_dsv4_sizes(full_token, page_size)
        logger.info(
            f"DSV4 memory calculation: unified={self._unified}, "
            f"unified_fp8={self._unified_fp8}, "
            f"dspark_draft_bf16={self._dspark_draft_on_bf16}, "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"c128_state_fixed={c128_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"c2_state_fixed={c2_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"swa_fixed={swa_fixed_bytes / (1 << 30):.2f} GB, "
            f"swa_ring_fixed={swa_ring_fixed_bytes / (1 << 30):.2f} GB, "
            f"c4_state_fixed={c4_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"full_token={sizes.full_max_total_num_tokens}"
        )
        return self._to_config(sizes)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        # Token count, not a byte budget: the fixed pools are not re-subtracted, so
        # the input must not exceed what calculate_pool_sizes derived for it.
        assert page_size % 128 == 0, (
            "page_size must be multiple of 128 for compressed attention"
        )
        sizes = self._compute_dsv4_sizes(max_total_num_tokens, page_size)
        return self._to_config(sizes)


def create_memory_pool_configurator(
    kvc: KVCacheConfigurator,
) -> MemoryPoolConfigurator:
    if is_deepseek_v4(kvc.model_config.hf_config) and kvc.is_hybrid_swa:
        return DSV4PoolConfigurator(kvc)
    if kvc.is_hybrid_swa:
        if SWAChunkCapPoolConfigurator.is_applicable(kvc):
            return SWAChunkCapPoolConfigurator(kvc)
        return HybridSWAPoolConfigurator(kvc)
    # Future: MambaPoolConfigurator
    return DefaultPoolConfigurator(kvc)
