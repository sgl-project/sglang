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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import (
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
from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import (
    ceil_align,
    ceil_div,
    is_float4_e2m1fn_x2,
    spec_decode_alloc_len_per_request,
)


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

    mem_fraction_static: Optional[float] = None

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

logger = logging.getLogger(__name__)


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


class DefaultPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for standard models: MHA, MLA, DSA, FP4.

    coeff = cell_size (bytes per token across all layers)
    bias = 0
    """

    def __init__(self, kvc: KVCacheConfigurator):
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
                self._cell_size = int(
                    self._cell_size
                    * (1 + int(eagle_draft_num_layers) / int(num_layers))
                )

        # DFLASH/DSPARK: scale cell_size to account for draft model KV cache
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
                    draft_num_layers=int(draft_num_layers),
                )

    def _compute_cell_size(self, kvc: KVCacheConfigurator, num_layers: int) -> int:
        """Compute per-token KV cache cost in bytes. Subclasses can override."""
        # args to config cell size
        model_config = kvc.model_config
        kv_cache_dtype = kvc.kv_cache_dtype
        from sglang.srt.layers.cp.utils import (
            get_glm_dsa_layer_split_effective_num_layers,
        )

        effective_num_layers = get_glm_dsa_layer_split_effective_num_layers(
            kvc, num_layers
        )

        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_parallel().attn_tp_size

        if kvc.use_mla_backend:
            cell_size = (
                (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
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
                index_head_dim = get_dsa_index_head_dim(model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // DSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    DSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += (
                    indexer_size_per_token * effective_num_layers * element_size
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
            cell_size = (
                model_config.get_num_kv_heads(tp_size)
                * (model_config.head_dim + model_config.v_head_dim)
                * effective_num_layers
                * kv_size
            )

            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                n = model_config.get_num_kv_heads(tp_size)
                k = model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * effective_num_layers * 2 * kv_size) // scale_block_size
                )
                # FP4 prefill uses one shared FP8 dequant workspace across layers.
                cell_size += n * k * 2 * kv_size
            elif kvc.server_args.kv_cache_dtype == "mxfp8":
                scale_block_size = 32
                n = model_config.get_num_kv_heads(tp_size)
                cell_size += (
                    n * (model_config.head_dim + model_config.v_head_dim) * num_layers
                ) // scale_block_size

        return cell_size

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = available_bytes // self._cell_size
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)


class HybridSWAPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for hybrid sliding window attention models (Gemma2, Command-R, MiMo).

    Splits available memory between full attention and SWA pools.
    Does NOT inherit DefaultPoolConfigurator — different coeff model.
    """

    def __init__(self, kvc: KVCacheConfigurator):
        model_config = kvc.model_config
        kv_cache_dtype = kvc.kv_cache_dtype
        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_parallel().attn_tp_size

        self._full_layers_num = len(model_config.full_attention_layer_ids)
        self._swa_layers_num = len(model_config.swa_attention_layer_ids)
        assert (
            self._swa_layers_num > 0
        ), "Hybrid SWA model must have at least one SWA layer"

        self._swa_full_tokens_ratio = kvc.server_args.swa_full_tokens_ratio
        self._sliding_window_size = kvc.sliding_window_size
        self._page_size = kvc.page_size

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

        if kvc.server_args.kv_cache_dtype == "mxfp8":
            scale_block_size = 32
            self._full_per_token += (
                model_config.get_num_kv_heads(tp_size)
                * (model_config.head_dim + model_config.v_head_dim)
            ) // scale_block_size
            self._swa_per_token += (
                model_config.get_swa_num_kv_heads(tp_size)
                * (model_config.swa_head_dim + model_config.swa_v_head_dim)
            ) // scale_block_size

        # EAGLE/STANDALONE draft KV pool inherits max_total tokens with its
        # full-attn layers; budget into the full term. A banded MTP depth
        # (Inkling mtp_local_layer_ids) instead allocates an swa-geometry ring
        # at FULL draft capacity, so budget those depths at swa_per_token.
        self._draft_full_layers_num = 0
        self._draft_swa_full_layers_num = 0
        if (
            kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone()
        ) and not kvc.is_draft_worker:
            draft_layers = kvc.spec_aux_config.eagle_draft_num_layers
            if draft_layers is not None and int(draft_layers) > 0:
                draft_layers = int(draft_layers)
                banded_depths = 0
                if (
                    model_config.hf_config.architectures[0]
                    == "InklingForConditionalGeneration"
                ):
                    banded_depths = len(
                        [
                            i
                            for i in model_config.hf_text_config.mtp_local_layer_ids
                            if i < draft_layers
                        ]
                    )
                self._draft_swa_full_layers_num = banded_depths
                self._draft_full_layers_num = draft_layers - banded_depths

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
                + self._swa_per_token * self._draft_swa_full_layers_num
            )
        else:
            self._cell_size = (
                self._full_per_token
                * (self._full_layers_num + self._draft_full_layers_num)
                + self._swa_per_token * self._draft_swa_full_layers_num
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
            )

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

        if (
            self._sliding_window_size is not None
            and self._sliding_window_size + self._page_size >= swa_tokens
        ):
            raise ValueError(
                f"SWA pool ({swa_tokens} tokens) cannot hold even one request: "
                f"the prefill admission floor is sliding_window_size "
                f"({self._sliding_window_size}) + page_size ({self._page_size}). "
                f"Increase --swa-full-tokens-ratio or the total KV budget."
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
        super().__init__(kvc)
        assert self._full_layers_num > 0

        sa = kvc.server_args
        page_size = kvc.page_size
        window = kvc.sliding_window_size
        draft_tokens = sa.speculative_num_draft_tokens or 1
        eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())

        """
        __________[padding][eviction_interval][window]
        Padding to make sure eviction point is page-aligned.
        """
        trailing_tokens = window + eviction_interval * draft_tokens + page_size
        if sa.speculative_algorithm is None:
            decode_alloc = page_size
        elif sa.disable_overlap_schedule:
            # spec-v1: new_tokens_required_next_decode per request.
            decode_alloc = spec_decode_alloc_len_per_request(sa)
        else:
            # spec-v2: the overlap allocator keeps 2 * alloc_len outstanding
            # (eagle_utils.eagle_prepare_for_decode: kv_committed_len + 2 * alloc_len).
            decode_alloc = 2 * get_alloc_len_per_decode(sa)
        per_request = trailing_tokens + decode_alloc

        num_reqs = sa.max_running_requests // kvc.ps.attn_dp_size
        if sa.disaggregation_mode == "decode":
            self._swa_cap = (
                per_request * num_reqs
                + (window + page_size) * sa.disaggregation_decode_extra_slots
            )
        else:
            chunks_in_flight = 1 if sa.disable_overlap_schedule else 2
            self._swa_cap = (
                per_request * num_reqs
                + chunks_in_flight * sa.chunked_prefill_size
                + page_size
            )

    @staticmethod
    def is_applicable(kvc: KVCacheConfigurator) -> bool:
        """True when SWAChunkCache can be sized from explicit max requests."""
        sa = kvc.server_args
        if sa.max_running_requests is None:
            return False
        if not sa.disable_radix_cache:
            return False
        if sa.chunked_prefill_size is None:
            return False
        if kvc.sliding_window_size is None:
            return False
        return len(kvc.model_config.full_attention_layer_ids) > 0

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        # SWA pool sized tightly from the cap; the rest of the budget goes to full.
        swa_tokens = ceil_align(self._swa_cap, page_size)
        fixed_swa_bytes = swa_tokens * self._swa_per_token * self._swa_layers_num
        full_cell_size = (
            self._full_per_token * (self._full_layers_num + self._draft_full_layers_num)
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
    decode reserves a draft worker, mirroring dflash's cell_size scaling). The
    bias is the sum of request-scoped fixed pools that do not scale with
    full_token: the c128 state pool and, on the unified_kv path, the fixed SWA
    per-request ring (bf16, see _fixed_swa_bytes).
    """

    def __init__(self, kvc: KVCacheConfigurator):
        cfg = kvc.model_config
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.indexer_head_dim = cfg.index_head_dim
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
        self.swa_ratio = kvc.server_args.swa_full_tokens_ratio
        self.is_speculative = kvc.server_args.speculative_algorithm is not None
        self.online_c128_mtp_max_draft_tokens = (
            kvc.server_args.max_speculative_num_draft_tokens or 0
        )
        self.requested_max_running_requests_per_worker = (
            kvc.server_args.max_running_requests // kvc.ps.attn_dp_size
            if kvc.server_args.max_running_requests is not None
            else None
        )
        self.disaggregation_mode = kvc.server_args.disaggregation_mode
        self.disaggregation_decode_extra_slots = (
            kvc.server_args.disaggregation_decode_extra_slots or 0
        )
        if kvc.server_args.enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            self.c4_shrink_factor = parse_hisparse_config(
                kvc.server_args
            ).host_to_device_ratio
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

        # Unified-KV uses a different physical layout than the fp8 path:
        #  * KV is stored bf16 over the full latent (attn_head_dim * 2 bytes),
        #    not the fp8(nope) + bf16(rope) + scales 584-byte cell.
        #  * SWA is a fixed per-request ring (num_req_slots * ring_size),
        #    independent of full_token, so it is a fixed *bias* rather than a
        #    per-token term. Gate on the same switch the pool itself uses so the
        #    sizing and the allocation never drift apart.
        from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        self._unified = is_unified_kv_triton()
        self.attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # Mirror DeepSeekV4TokenToKVPool: swa_ring_size = sliding_window +
        # (speculative_num_draft_tokens - 1).
        spec_num_draft = mr.server_args.speculative_num_draft_tokens or 1
        self._swa_ring_size = self.swa_page_size + (
            (spec_num_draft - 1) if self.is_speculative else 0
        )
        self._spec_infl = 1.0

        self.bytes_per_full_token = self._get_bytes_per_full_token()
        if self.is_speculative:
            # Reserve memory for the speculative draft worker by inflating
            # per-token bytes by (target+draft)/target. Equivalent to dflash's
            # scale_kv_cell_size_per_token_for_dflash but applied to
            # bytes_per_full_token: tokens = avail / (bpft * (T+D)/T).
            draft_layers = 1
            target_layers = self.num_layers_total
            self._spec_infl = (target_layers + draft_layers) / target_layers
            self.bytes_per_full_token *= self._spec_infl

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

    def _get_bytes_per_full_token(self) -> float:
        if self._unified:
            # Unified_kv stores the whole latent in bf16.
            kv_bytes = self.attn_head_dim * 2
        else:
            kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        quant_block_size = 128
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )

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
            # Unified_kv: SWA is a fixed per-request ring (see _fixed_swa_bytes),
            # not a per-token pool, so it is excluded from the per-token coeff.
            (
                0.0
                if self._unified
                else self.swa_ratio * kv_bytes * self.num_layers_total
            )
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            # Unified_kv: the c4 (attn + indexer) compress-state is a ring buffer
            # addressed off the SWA slot ((swa_loc // swa_page_size) * ring_size),
            # and the unified SWA pool is a fixed per-request ring
            # (swa_pages = num_req_slots * swa_ring_size), so the state ring is
            # request-scoped, not full_token-scoped. It is therefore a fixed bias
            # (see _fixed_c4_state_bytes), not a per-token term. On the non-unified
            # path the SWA pool scales with full_token, so it stays per-token.
            + (
                0.0
                if self._unified
                else self.swa_ratio
                * c4_state_ratio
                * c4_state_bytes
                * self.num_layers_ca4
            )
            + c128_state_ratio * c128_state_bytes * self.num_layers_ca128
            + (
                0.0
                if self._unified
                else self.swa_ratio
                * c4_state_ratio
                * c4_indexer_state_bytes
                * self.num_layers_ca4
            )
        )

    def _compute_dsv4_sizes(self, full_token: int, page_size: int) -> _DSV4PoolSizes:
        full_token = full_token // page_size * page_size
        swa_tokens = int(full_token * self.swa_ratio) // page_size * page_size
        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            # Unified_kv sizes the c4 state ring from the fixed SWA ring
            # (request-scoped), finalized once max_running_requests is known -- so
            # it must not scale with full_token here (mirrors c128_state below).
            c4_state_pool_size=(
                0
                if self._unified
                else swa_tokens // self.swa_page_size * self.c4_ring_size
            ),
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

    def _unified_c4_state_pool_size(self, max_running_requests: int) -> int:
        """Request-scoped c4 state-ring slot count on the unified_kv path.

        The c4 compress-state is addressed by
        ``(swa_loc // swa_page_size) * c4_ring_size + swa_loc % c4_ring_size``.
        Under unified_kv the SWA pool is a fixed per-request ring with
        ``swa_pages = num_req_slots * swa_ring_size`` slots, so ``swa_loc`` is
        bounded by that and the required state slots are
        ``ceil(num_req_slots * swa_ring_size / swa_page_size) * c4_ring_size``.
        (Non-speculative: swa_ring_size == swa_page_size, so this reduces to
        ``num_req_slots * c4_ring_size`` -- exactly the c128 pattern.)
        """
        num_req_slots = self._get_num_req_slots(max_running_requests)
        swa_pages = ceil_div(
            num_req_slots * self._swa_ring_size, self.swa_page_size
        )
        return swa_pages * self.c4_ring_size

    def _fixed_c4_state_bytes(self, max_running_requests: int) -> int:
        """Unified_kv c4 (attn + indexer) compress-state is a fixed per-request
        ring, sized by concurrency rather than full_token. Return its byte
        footprint across all c4 layers. Returns 0 on the non-unified path (where
        the c4 state pool scales with the SWA pool and is accounted per-token)."""
        if not self._unified or self.num_layers_ca4 == 0:
            return 0

        c4_state_dtype_size, _ = _get_dsv4_compress_state_dtype_sizes()
        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # CompressStatePool allocates `size + ring_size + 1` rows, padded to the
        # compress ratio (see CompressStatePool.__init__). Mirror that here so the
        # reserved bias covers the real allocation.
        state_rows = self._unified_c4_state_pool_size(max_running_requests)
        state_rows = ceil_div(state_rows + self.c4_ring_size + 1, 4) * 4
        # overlap c4: last_dim = 2 * (1 + overlap) * head_dim = 4 * head_dim.
        core_bytes = 4 * attn_head_dim * c4_state_dtype_size
        indexer_bytes = 4 * self.indexer_head_dim * c4_state_dtype_size
        return state_rows * (core_bytes + indexer_bytes) * self.num_layers_ca4

    def _resolve_max_running_requests_per_worker(self, available_bytes: int) -> int:
        """Approximate ModelRunner._resolve_max_num_reqs closely enough to size
        the request-scoped fixed pools (c128 state, unified SWA ring). Over-
        estimating is safe: a larger fixed bias yields a smaller full_token."""
        if self.requested_max_running_requests_per_worker is not None:
            return self.requested_max_running_requests_per_worker

        full_token = int(available_bytes / self.bytes_per_full_token)
        estimated = int(full_token / self.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)
        return min(estimated, full_token // 2)

    def _fixed_swa_bytes(self, max_running_requests: int) -> int:
        """Unified_kv SWA is a fixed per-request ring, sized by concurrency
        (num_req_slots) rather than by full_token. Return its bf16 byte
        footprint across all full layers, inflated for the draft worker the same
        way as the per-token coeff. Returns 0 on the non-unified path (where SWA
        is already accounted per-token)."""
        if not self._unified:
            return 0
        num_req_slots = self._get_num_req_slots(max_running_requests)
        ring_bytes = (
            num_req_slots
            * self._swa_ring_size
            * self.attn_head_dim
            * 2  # bf16
            * self.num_layers_total
        )
        return int(ring_bytes * self._spec_infl)

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
            config.c128_state_pool_size = num_req_slots * self.c128_ring_size
        # Unified_kv: the c4 state ring is request-scoped (fixed SWA pool), so
        # finalize it here from the now-known concurrency. On the non-unified path
        # it was already sized from full_token in _compute_dsv4_sizes.
        if self._unified and self.num_layers_ca4 > 0:
            config.c4_state_pool_size = self._unified_c4_state_pool_size(
                config.max_running_requests
            )
        return config

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"

        max_running_requests_per_worker = self._resolve_max_running_requests_per_worker(
            available_bytes
        )
        c128_state_fixed_bytes = self._get_c128_state_fixed_bytes(
            max_running_requests_per_worker
        )
        swa_ring_fixed_bytes = self._fixed_swa_bytes(max_running_requests_per_worker)
        c4_state_fixed_bytes = self._fixed_c4_state_bytes(
            max_running_requests_per_worker
        )

        available_bytes_for_tokens = max(
            available_bytes
            - c128_state_fixed_bytes
            - swa_ring_fixed_bytes
            - c4_state_fixed_bytes,
            0,
        )
        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)

        sizes = self._compute_dsv4_sizes(full_token, page_size)
        logger.info(
            f"DSV4 memory calculation: unified={self._unified}, "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"c128_state_fixed={c128_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"swa_ring_fixed={swa_ring_fixed_bytes / (1 << 30):.2f} GB, "
            f"c4_state_fixed={c4_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"full_token={sizes.full_max_total_num_tokens}"
        )
        return self._to_config(sizes)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"
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
