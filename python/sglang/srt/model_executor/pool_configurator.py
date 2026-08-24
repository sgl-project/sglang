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
    can_use_compact_npu_dsa_indexer_cache,
    can_use_npu_quant_lightning_indexer,
    get_dsa_index_head_dim,
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    is_deepseek_dsa,
    is_deepseek_v4,
    is_minimax_sparse,
    resolve_dsa_indexer_layer_ids,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocation_sizing import get_alloc_len_per_decode
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    get_compress_state_ring_size,
    get_compress_state_write_pad,
)
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


def _is_npu_pool_device(device: object) -> bool:
    return str(device).split(":", 1)[0].casefold() == "npu"


def _get_npu_dsa_indexer_size_per_token(
    model_config,
    kv_cache_dtype: torch.dtype,
    *,
    has_scale_cache: bool,
) -> int:
    """Bytes used by one physical Ascend DSA Indexer layer per token."""
    size = get_dsa_index_head_dim(model_config.hf_config) * torch._utils._element_size(
        kv_cache_dtype
    )
    if has_scale_cache:
        size += torch.float32.itemsize
    return size


def _get_npu_dsa_indexer_layer_count(kvc: KVCacheConfigurator, num_layers: int) -> int:
    is_nextn = kvc.is_draft_worker and bool(kvc.model_config.num_nextn_predict_layers)
    return len(
        resolve_dsa_indexer_layer_ids(
            kvc.model_config.hf_config,
            kvc.layer_info.start_layer,
            kvc.layer_info.end_layer,
            is_nextn=is_nextn,
        )
    )


def _npu_dsa_indexer_has_scale_cache(
    kvc: KVCacheConfigurator, kv_cache_dtype: Optional[torch.dtype] = None
) -> bool:
    if kv_cache_dtype is None:
        kv_cache_dtype = kvc.kv_cache_dtype
    return can_use_npu_quant_lightning_indexer(
        kvc.server_args,
        kvc.model_config.hf_config,
        kv_cache_dtype,
        kvc.gpu_id,
    )


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
        self.kvc = kvc
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
            else kvc.server_args.max_total_tokens or kvc.model_config.context_len
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
                if (
                    kvc.spec_algorithm.is_eagle()
                    and _is_npu_pool_device(kvc.device)
                    and is_deepseek_dsa(kvc.model_config.hf_config)
                ):
                    from sglang.srt.mem_cache.kv_cache_configurator import (
                        calculate_mla_kv_cache_dim,
                    )

                    draft_num_layers = int(eagle_draft_num_layers)
                    draft_kv_cache_dtype = (
                        kvc.spec_aux_config.eagle_draft_kv_cache_dtype
                        or kvc.kv_cache_dtype
                    )
                    draft_main_size_per_layer = calculate_mla_kv_cache_dim(
                        model_config=kvc.model_config,
                        kv_cache_dtype=draft_kv_cache_dtype,
                        server_args=kvc.server_args,
                    ) * torch._utils._element_size(draft_kv_cache_dtype)
                    draft_indexer_layers = len(
                        resolve_dsa_indexer_layer_ids(
                            kvc.model_config.hf_config,
                            0,
                            draft_num_layers,
                            is_nextn=True,
                        )
                    )
                    self._cell_size += (
                        draft_num_layers * draft_main_size_per_layer
                        + draft_indexer_layers
                        * _get_npu_dsa_indexer_size_per_token(
                            kvc.model_config,
                            draft_kv_cache_dtype,
                            has_scale_cache=_npu_dsa_indexer_has_scale_cache(
                                kvc, draft_kv_cache_dtype
                            ),
                        )
                    )
                else:
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
                    draft_num_layers=int(draft_num_layers)
                    * get_parallel().attn_dcp_size,
                    draft_cell_size_per_token=_dflash_draft_cell_size(kvc) or None,
                )

    def _local_selective_layer_ids(self) -> tuple[int, ...]:
        """Return target-worker selective layers owned by this PP stage."""
        kvc = self.kvc
        configured = getattr(
            kvc.server_args, "npu_selective_hisparse_layer_ids", None
        )
        if (
            not configured
            or kvc.is_draft_worker
            or not _is_npu_pool_device(kvc.device)
        ):
            return ()
        return tuple(
            sorted(
                {
                    layer_id
                    for layer_id in configured
                    if kvc.layer_info.start_layer
                    <= layer_id
                    < kvc.layer_info.end_layer
                }
            )
        )

    def _compute_cell_size(self, kvc: KVCacheConfigurator, num_layers: int) -> int:
        """Compute per-token KV cache cost in bytes. Subclasses can override."""
        # args to config cell size
        model_config = kvc.model_config
        kv_cache_dtype = kvc.kv_cache_dtype
        from sglang.srt.layers.cp.utils import (
            get_glm_dsa_layer_split_effective_num_layers,
        )

        if _is_npu_pool_device(kvc.device):
            # The Ascend MLA pool is not wrapped by LayerSplitDSATokenToKVPool.
            effective_num_layers = num_layers
        else:
            effective_num_layers = get_glm_dsa_layer_split_effective_num_layers(
                kvc, num_layers
            )

        # Selective HiSparse: reduce main KV layer count by number of offloaded layers
        num_selective = len(self._local_selective_layer_ids())
        main_kv_layers = effective_num_layers - num_selective

        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_parallel().attn_tp_size
        dcp_size = get_parallel().attn_dcp_size

        if kvc.use_mla_backend:
            from sglang.srt.mem_cache.kv_cache_configurator import (
                calculate_mla_kv_cache_dim,
            )

            # Main KV: only resident layers
            cell_size = (
                calculate_mla_kv_cache_dim(
                    model_config=model_config,
                    kv_cache_dtype=kv_cache_dtype,
                    server_args=kvc.server_args,
                )
                * max(main_kv_layers, 0)
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
                if _is_npu_pool_device(kvc.device):
                    cell_size += _get_npu_dsa_indexer_size_per_token(
                        model_config,
                        kv_cache_dtype,
                        has_scale_cache=_npu_dsa_indexer_has_scale_cache(kvc),
                    ) * _get_npu_dsa_indexer_layer_count(kvc, effective_num_layers)
                else:
                    index_head_dim = get_dsa_index_head_dim(model_config.hf_config)
                    indexer_size_per_token = (
                        index_head_dim
                        + index_head_dim // DSATokenToKVPool.quant_block_size * 4
                    )
                    element_size = torch._utils._element_size(
                        DSATokenToKVPool.index_k_with_scale_buffer_dtype
                    )
                    indexer_ratio = 1
                    if kvc.server_args.enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        indexer_ratio = parse_hisparse_config(
                            kvc.server_args
                        ).host_to_device_ratio
                    cell_size += int(
                        indexer_size_per_token
                        * effective_num_layers
                        * element_size
                        * indexer_ratio
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

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        # Selective HiSparse: subtract fixed staging/workspace bias
        if self._local_selective_layer_ids():
            fixed_bias = self._compute_selective_fixed_bias()
            usable_bytes = max(available_bytes - fixed_bias, 0)
        else:
            usable_bytes = available_bytes

        max_total_num_tokens = (
            usable_bytes // self._cell_size
            if self._cell_size
            else self._zero_kv_max_tokens
        )
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)

    def _compute_selective_fixed_bias(self) -> int:
        """Compute fixed HBM overhead for selective HiSparse staging buffers."""
        from sglang.srt.layers.dp_attention import get_attention_dp_size

        kvc = self.kvc
        max_running = kvc.server_args.max_running_requests or 256
        attn_dp = get_attention_dp_size()
        bcap = (max_running + attn_dp - 1) // attn_dp
        # Graph batch sizes raise the staging capacity when graphs are
        # enabled. Under --disable-cuda-graph the DEFAULT bs list (max 512)
        # would inflate the bias ~14GB, but an EXPLICIT --cuda-graph-bs
        # still reflects the real decode batch — eager decode runs the same
        # batches as graph replay would (router-side batching is identical),
        # so honor an explicit list and ignore only the implicit default.
        # Without this, eager 19L under-estimates the bias, the pool eats
        # the difference, and alloc_memory_pool OOMs at startup.
        _explicit_bs = (
            getattr(kvc.server_args, "cuda_graph_bs_decode", None) or None
        )
        if not getattr(kvc.server_args, "disable_cuda_graph", False):
            decode_graph_config = getattr(
                getattr(kvc.server_args, "cuda_graph_config", None),
                "decode",
                None,
            )
            cuda_graph_bs = getattr(decode_graph_config, "decode_bs", None) or getattr(decode_graph_config, "bs", None)
            if cuda_graph_bs:
                bcap = max(bcap, max(cuda_graph_bs))
        elif _explicit_bs:
            bcap = max(bcap, max(_explicit_bs))
        if kvc.spec_algorithm.is_speculative():
            from sglang.srt.speculative.spec_utils import (
                resolve_num_tokens_per_req,
            )

            verify_width = resolve_num_tokens_per_req(
                phase="target_verify",
                server_args=kvc.server_args,
                spec_algorithm=kvc.spec_algorithm,
                is_draft_worker=False,
            )
        else:
            verify_width = 1
        hf_text_config = getattr(
            kvc.model_config,
            "hf_text_config",
            kvc.model_config.hf_config,
        )
        topk = int(getattr(hf_text_config, "index_topk", 2048) or 2048)
        record_bytes = 656
        kv_lora_rank = kvc.model_config.kv_lora_rank
        qk_rope_head_dim = kvc.model_config.qk_rope_head_dim

        tcap = bcap * verify_width
        rcap = tcap * topk

        # packed_staging: [Tcap, K, 656] uint8 — N slices (staging
        # ping-pong, parity-indexed by selected layer; matches
        # _alloc_staging_buffers in selective_hisparse.py, same env)
        import os as _os
        _n_staging_slices = int(
            _os.getenv("SGLANG_SELECTIVE_STAGING_SLICES", "2")
        )
        staging = rcap * record_bytes * _n_staging_slices
        # unpack_k_nope_bf16: [Tcap, K, 512] BF16 — n_ws sets (matches
        # _alloc_staging_buffers in selective_hisparse.py, same env)
        _n_ws = int(_os.getenv("SGLANG_SELECTIVE_UNPACK_WS", "2"))
        unpack_nope = rcap * kv_lora_rank * 2 * _n_ws
        # unpack_k_rope_bf16: [Tcap, K, 64] BF16
        unpack_rope = rcap * qk_rope_head_dim * 2 * _n_ws
        # publish_new_packed_kv retains one packed output per selected layer.
        packed_outputs = (
            len(self._local_selective_layer_ids()) * tcap * record_bytes
        )
        # Persistent unpack/cast workspaces omitted by the original estimate.
        fp8_nope = rcap * kv_lora_rank * _n_ws
        scales = rcap * (kv_lora_rank // 128) * 4 * _n_ws
        # Per-record metadata and mf_offload pointer arrays.
        record_meta = (
            tcap * topk * 8  # host_locs int64
            + tcap * topk * 8  # current_source_row int64
            + tcap * topk * 4  # sparse_indices int32
            + tcap * topk * (8 + 8 + 4)  # H2D src/dst/lens
            + tcap * topk * 8  # preset H2D dst pointers
        )
        token_meta = tcap * (4 + 4 + 4 + 8 + 8 + 4)
        fixed_meta = topk * 4 + 12  # arange_k + three scalar counters

        total = (
            staging
            + unpack_nope
            + unpack_rope
            + fp8_nope
            + scales
            + packed_outputs
            + record_meta
            + token_meta
            + fixed_meta
        )
        logger.info(
            f"Selective HiSparse fixed bias: {total / (1024*1024):.1f} MiB "
            f"(Bcap={bcap}, Tcap={tcap})"
        )
        return total

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
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
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

        self._draft_cell_size = _dflash_draft_cell_size(kvc)

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
                + self._draft_cell_size
            )
        else:
            self._cell_size = (
                self._full_per_token
                * (self._full_layers_num + self._draft_full_layers_num)
                + self._swa_per_token * self._draft_swa_full_layers_num
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
                + self._draft_cell_size
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
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
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
    decode reserves a draft worker, mirroring dflash's cell_size scaling); bias = 0.
    """

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
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

        if self.is_speculative:
            # Ring is sized once here, so it must serve the largest adaptive tier.
            self._assert_ring_serves_draft_tokens(
                kvc.server_args.max_speculative_num_draft_tokens or 0
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
            self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
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
        return config

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"

        if self.requested_max_running_requests_per_worker is not None:
            c128_state_fixed_bytes = self._get_c128_state_fixed_bytes(
                self.requested_max_running_requests_per_worker
            )
        else:
            full_token = int(available_bytes / self.bytes_per_full_token)
            c128_state_fixed_bytes = (
                self._get_c128_state_fixed_bytes_for_token_capacity(full_token)
            )

        available_bytes_for_tokens = max(available_bytes - c128_state_fixed_bytes, 0)
        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)

        sizes = self._compute_dsv4_sizes(full_token, page_size)
        logger.info(
            f"DSV4 memory calculation: "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"c128_state_fixed={c128_state_fixed_bytes / (1 << 30):.2f} GB, "
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
