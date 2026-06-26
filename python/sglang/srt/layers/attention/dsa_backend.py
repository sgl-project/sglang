from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
)

import torch

from sglang.srt.configs.model_config import get_dsa_index_topk, is_deepseek_dsa
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    DeepseekSparseAttnBackendMTPPrecomputeMixin,
    PrecomputedMetadata,
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.dsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_prefill_cp_round_robin_split,
    compute_dsa_seqlens,
    dsa_cp_round_robin_split_data,
    dsa_cp_round_robin_split_q_seqs,
    is_dsa_enable_prefill_cp,
    pad_dsa_cache_seqlens,
)
from sglang.srt.layers.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    seqlens_expand_triton,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_sm100_supported,
)

# Opt-in (default off): route the fp8 sparse-MLA prefill path through the Triton
# per-query flash kernel instead of TileLang. Validated on gfx950 (GLM-5.1 @
# TP4: 16 heads, d_v=512, tail=64). Reads q_nope/q_rope directly (skips the
# concat). Enable with SGLANG_DSA_TRITON_PREFILL=1. Decode stays on TileLang.
_DSA_TRITON_PREFILL = get_bool_env_var("SGLANG_DSA_TRITON_PREFILL")
_IS_GFX95 = is_gfx95_supported()

if is_cuda():
    import deep_gemm

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


_is_hip = is_hip()

if _is_hip:
    from sglang.srt.layers.attention.dsa.triton_kernel import get_valid_kv_indices
    from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

    try:
        from aiter import (  # noqa: F401
            flash_attn_varlen_func,
            get_mla_metadata_info_v1,
            get_mla_metadata_v1,
            mha_batch_prefill_func,
            paged_attention_ragged,
        )
        from aiter.mla import mla_decode_fwd, mla_prefill_fwd  # noqa: F401
    except ImportError:
        print(
            "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
        )
else:
    from sglang.jit_kernel.flash_attention import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )


def _to_2d_context_lens(seqlens_32: torch.Tensor, batch_size: int) -> torch.Tensor:
    # Always normalize to (N_total, 1) layout, to avoid deadlock at deep_gemm.fp8_paged_mqa_logits
    if seqlens_32.dim() == 2:
        if seqlens_32.size(1) == 1:
            return seqlens_32
        # Fall through and re-flatten if the caller already gave us a (bs, next_n)
        # view — we want (N_total, 1) regardless.
        seqlens_32 = seqlens_32.reshape(-1)
    return seqlens_32.contiguous().view(-1, 1)


# Reuse this workspace buffer across all DSA backend instances
global_workspace_buffer = None

# Control whether to use fused metadata copy kernel for cuda graph replay (default: enabled)
# Set SGLANG_USE_FUSED_METADATA_COPY=0 or false to disable
_USE_FUSED_METADATA_COPY = envs.SGLANG_USE_FUSED_METADATA_COPY.get() and not _is_hip


@dataclass(frozen=True)
class DSAFlashMLAMetadata:
    """Metadata only needed by FlashMLA"""

    flashmla_metadata: torch.Tensor
    num_splits: torch.Tensor

    def slice(self, sli):
        return DSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=self.num_splits[sli],
        )

    def copy_(self, other: DSAFlashMLAMetadata):
        self.flashmla_metadata.copy_(other.flashmla_metadata)
        self.num_splits.copy_(other.num_splits)


@dataclass(frozen=True)
class DSAMetadata:
    page_size: int

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor
    # Maximum sequence length for query
    max_seq_len_q: int
    # Maximum sequence length for key
    max_seq_len_k: int
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor
    # Page table, the index of KV Cache Tables/Blocks
    # this table is always with page_size = 1
    page_table_1: torch.Tensor

    # NOTE(dark): This will property be used in:
    # 1. dense decode/prefill, we use paged flash attention, need real_page_table
    # 2. sparse decode/prefill, indexer need real_page_table to compute the score
    real_page_table: torch.Tensor

    # DSA metadata (dsa prefill are expanded)
    dsa_cache_seqlens_int32: torch.Tensor  # this seqlens is clipped to `topk`
    dsa_cu_seqlens_q: torch.Tensor  # must be arange(0, len(dsa_cu_seqlens_k))
    dsa_cu_seqlens_k: torch.Tensor  # cumsum of `dsa_cache_seqlens_int32`
    dsa_extend_seq_lens_list: List[int]
    dsa_seqlens_expanded: torch.Tensor  # expanded, unclipped `seqlens`
    dsa_max_seqlen_q: Literal[1] = 1  # always 1 for decode, variable for extend

    flashmla_metadata: Optional[DSAFlashMLAMetadata] = None
    # DeepGEMM schedule metadata for paged MQA logits (decode/target_verify/draft_extend only).
    # Precomputed once per forward batch and reused across layers.
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    # 2D context_lens used to build the schedule above; the indexer reuses it
    # as DG's `context_lens` arg so the broadcast doesn't rebuild per layer.
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None
    # The sum of sequence lengths for key, prefill only
    seq_lens_sum: Optional[int] = None
    # The flattened 1D page table with shape (seq_lens_sum,), prefill only
    # this table is always with page_size = 1
    page_table_1_flattened: Optional[torch.Tensor] = None
    # The offset of topk indices in ragged kv, prefill only
    # shape: (seq_lens_sum,)
    topk_indices_offset: Optional[torch.Tensor] = None

    # k_start and k_end in kv cache for each token.
    indexer_k_start_end: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # seq lens for each batch.
    indexer_seq_lens_cpu: Optional[torch.Tensor] = None
    # seq lens for each batch.
    indexer_seq_lens: Optional[torch.Tensor] = None
    # batch index for each token.
    token_to_batch_idx: Optional[torch.Tensor] = None


@torch.compile
def _compiled_cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


def _cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Concatenate two tensors along the last dimension.
    Use this function to concatenate q_nope and q_rope or k_nope and k_rope.
    """
    assert len(tensors) == 2

    qk_nope, qk_rope = tensors
    assert qk_nope.ndim == 3 and qk_rope.ndim == 3

    torch._dynamo.mark_dynamic(qk_nope, 0)
    torch._dynamo.mark_dynamic(qk_rope, 0)

    return _compiled_cat([qk_nope, qk_rope], dim=dim)


@dataclass(frozen=True)
class DSAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: DSAMetadata
    topk_transform_method: TopkTransformMethod
    topk_backend: DSATopKBackend = DSATopKBackend.SGL_KERNEL
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None
    force_unfused_topk: bool = False

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.dsa_seqlens_expanded

    def get_cu_seqlens_k(self) -> torch.Tensor:
        return self.attn_metadata.cu_seqlens_k

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn_metadata.indexer_k_start_end

    def get_indexer_seq_len(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens_cpu

    def get_dsa_extend_len_cpu(self) -> List[int]:
        return self.attn_metadata.dsa_extend_seq_lens_list

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        ke_offset: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if topk_indices_offset_override is not None:
            cu_topk_indices_offset = topk_indices_offset_override
            cu_seqlens_q_topk = None
        elif cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)
            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,
            )
        else:
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset
        if ke_offset is not None:
            seq_lens_topk = ke_offset
        else:
            seq_lens_topk = self.get_seqlens_expanded()
        return self.topk_backend.topk_transform(
            logits=logits,
            lengths=seq_lens_topk,
            topk=topk,
            topk_transform_method=self.topk_transform_method,
            attn_metadata=self.attn_metadata,
            cu_seqlens_q_topk=cu_seqlens_q_topk,
            topk_indices_offset=cu_topk_indices_offset,
            row_starts=ks,
            batch_idx_list=batch_idx_list,
            force_unfused_topk=self.force_unfused_topk,
        )


_DSA_IMPL_T: TypeAlias = Literal[
    "flashmla_sparse", "flashmla_kv", "fa3", "tilelang", "trtllm"
]


class DeepseekSparseAttnBackend(
    DeepseekSparseAttnBackendMTPPrecomputeMixin, AttentionBackend
):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.forward_metadata: DSAMetadata
        self.device = model_runner.device
        assert isinstance(model_runner.page_size, int)
        self.real_page_size = model_runner.page_size
        self.num_splits = (
            1 if model_runner.server_args.enable_deterministic_inference else 0
        )
        self.use_dsa = is_deepseek_dsa(model_runner.model_config.hf_config)
        assert self.use_dsa, "DSA backend only supports DeepSeek DSA"
        self.dsa_kv_cache_store_fp8 = (
            model_runner.token_to_kv_pool.dsa_kv_cache_store_fp8
        )
        self.dsa_index_topk = get_dsa_index_topk(model_runner.model_config.hf_config)
        self.max_context_len = model_runner.model_config.context_len
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.kv_cache_dim = model_runner.token_to_kv_pool.kv_cache_dim
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim

        assert model_runner.req_to_token_pool is not None
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.hisparse_coordinator = model_runner.hisparse_coordinator
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mha: bool = False
        self.dsa_prefill_impl: _DSA_IMPL_T = (
            model_runner.server_args.dsa_prefill_backend
        )
        self.dsa_decode_impl: _DSA_IMPL_T = model_runner.server_args.dsa_decode_backend
        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend(
            model_runner.server_args.dsa_topk_backend
        )
        if self.num_q_heads <= 64:
            self.flashmla_kv_num_q_heads = 64
        elif self.num_q_heads <= 128:
            self.flashmla_kv_num_q_heads = 128
        else:
            # Keep original head count if it exceeds current padded variants.
            self.flashmla_kv_num_q_heads = self.num_q_heads
        self.enable_auto_select_prefill_impl = self.dsa_prefill_impl == "flashmla_auto"

        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)

        if _is_hip:
            max_bs = model_runner.req_to_token_pool.size

            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.kv_indices = torch.zeros(
                max_bs * self.dsa_index_topk,
                dtype=torch.int32,
                device=self.device,
            )
            # Aiter mla_decode_fwd supports num_heads multiples of 16 in range [16, 128].
            # For models with fewer heads per GPU (e.g. GLM-5 64 heads / TP8 = 8), need to pad the heads to 16.
            self.need_pad_heads = self.num_q_heads < 16
            self.head_repeat_factor = (
                16 // self.num_q_heads if self.num_q_heads < 16 else 1
            )
            self.num_head_padded = self.num_q_heads * self.head_repeat_factor
            self.aiter_dsa_max_split_per_batch = 64
            self.aiter_dsa_metadata_capacity = 0
            self.aiter_dsa_metadata_max_seqlen_q = 0
            self.aiter_dsa_metadata_q_dtype = None
            self.aiter_dsa_metadata_kv_dtype = None
            self.aiter_dsa_kv_last_page_lens = None
            self.aiter_dsa_work_metadata = None

            if (
                self.dsa_prefill_impl == "aiter" or self.dsa_decode_impl == "aiter"
            ) and model_runner.kv_cache_dtype == fp8_dtype:
                self._ensure_aiter_dsa_decode_metadata_buffer(
                    max_seqlen_q=1,
                    batch_size=max_bs,
                    q_dtype=torch.bfloat16,
                    kv_dtype=fp8_dtype,
                )

        # Speculative decoding
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        self.device_capability = torch.cuda.get_device_capability()
        self.device_sm_major = self.device_capability[0]
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        # Allocate global workspace buffer for TRT-LLM kernels (ragged attention on SM100/B200, or trtllm decode)
        if self.device_sm_major >= 10 or self.dsa_decode_impl == "trtllm":
            global global_workspace_buffer
            if global_workspace_buffer is None:
                global_workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_workspace_buffer
        else:
            self.workspace_buffer = None

    def _make_aiter_dsa_decode_metadata_buffer(
        self,
        max_seqlen_q: int,
        batch_size: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ):
        (
            (work_metadata_size, work_metadata_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            batch_size,
            max_seqlen_q,
            self.num_head_padded,
            q_dtype,
            kv_dtype,
            is_sparse=True,
            fast_mode=False,
            num_kv_splits=self.aiter_dsa_max_split_per_batch,
            intra_batch_mode=True,
        )

        return (
            torch.empty(
                work_metadata_size, dtype=work_metadata_type, device=self.device
            ),
            torch.empty(work_indptr_size, dtype=work_indptr_type, device=self.device),
            torch.empty(
                work_info_set_size, dtype=work_info_set_type, device=self.device
            ),
            torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_type, device=self.device
            ),
            torch.empty(
                reduce_final_map_size, dtype=reduce_final_map_type, device=self.device
            ),
            torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_type,
                device=self.device,
            ),
        )

    def _ensure_aiter_dsa_decode_metadata_buffer(
        self,
        max_seqlen_q: int,
        batch_size: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> None:
        if (
            self.aiter_dsa_work_metadata is not None
            and self.aiter_dsa_metadata_capacity >= batch_size
            and self.aiter_dsa_metadata_max_seqlen_q == max_seqlen_q
            and self.aiter_dsa_metadata_q_dtype == q_dtype
            and self.aiter_dsa_metadata_kv_dtype == kv_dtype
        ):
            return

        (
            self.aiter_dsa_work_metadata,
            self.aiter_dsa_work_indptr,
            self.aiter_dsa_work_info_set,
            self.aiter_dsa_reduce_indptr,
            self.aiter_dsa_reduce_final_map,
            self.aiter_dsa_reduce_partial_map,
        ) = self._make_aiter_dsa_decode_metadata_buffer(
            max_seqlen_q=max_seqlen_q,
            batch_size=batch_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )
        self.aiter_dsa_kv_last_page_lens = torch.ones(
            (batch_size,), dtype=torch.int32, device=self.device
        )
        self.aiter_dsa_metadata_capacity = batch_size
        self.aiter_dsa_metadata_max_seqlen_q = max_seqlen_q
        self.aiter_dsa_metadata_q_dtype = q_dtype
        self.aiter_dsa_metadata_kv_dtype = kv_dtype

    def _prepare_aiter_dsa_decode_metadata(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        bs: int,
        max_seqlen_q: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> dict:
        self._ensure_aiter_dsa_decode_metadata_buffer(
            max_seqlen_q=max_seqlen_q,
            batch_size=bs,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )
        self.aiter_dsa_kv_last_page_lens[:bs].fill_(1)
        kv_last_page_lens = self.aiter_dsa_kv_last_page_lens[:bs]

        get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_lens,
            self.num_head_padded,
            1,
            False,
            self.aiter_dsa_work_metadata,
            self.aiter_dsa_work_info_set,
            self.aiter_dsa_work_indptr,
            self.aiter_dsa_reduce_indptr,
            self.aiter_dsa_reduce_final_map,
            self.aiter_dsa_reduce_partial_map,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=max_seqlen_q,
            uni_seqlen_qo=max_seqlen_q,
            fast_mode=False,
            topk=self.dsa_index_topk,
            max_split_per_batch=self.aiter_dsa_max_split_per_batch,
            intra_batch_mode=True,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
        )

        return {
            "kv_last_page_lens": kv_last_page_lens,
            "work_meta_data": self.aiter_dsa_work_metadata,
            "work_indptr": self.aiter_dsa_work_indptr,
            "work_info_set": self.aiter_dsa_work_info_set,
            "reduce_indptr": self.aiter_dsa_reduce_indptr,
            "reduce_final_map": self.aiter_dsa_reduce_final_map,
            "reduce_partial_map": self.aiter_dsa_reduce_partial_map,
            "intra_batch_mode": True,
            "num_kv_splits": self.aiter_dsa_max_split_per_batch,
        }

    def _build_paged_mqa_schedule_2d_ctx_lens(
        self,
        forward_mode: ForwardMode,
        cache_seqlens_int32: torch.Tensor,
        seqlens_expanded: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        # target_verify with next_n>=2 uses DG-native q=[B,next_n,H,D] which
        # needs a [B, next_n] schedule; everything else stays per-token.
        # TODO: SM90 supports DG-native next_n in {1,2} too — enable once
        # validated; for now DG-native is SM100+ only.
        next_n = self.speculative_num_draft_tokens
        if (
            forward_mode.is_target_verify()
            and next_n
            and next_n >= 2
            and is_sm100_supported()
        ):
            return cache_seqlens_int32.view(-1, 1).expand(-1, next_n).contiguous()
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            return _to_2d_context_lens(seqlens_expanded, batch_size)
        return _to_2d_context_lens(cache_seqlens_int32, batch_size)

    def _get_fused_topk_page_table(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if (
            self.dsa_topk_backend.is_sgl_kernel()
            or self.dsa_topk_backend.is_flashinfer()
        ):
            return topk_indices
        raise RuntimeError(
            f"Unsupported {self.dsa_topk_backend = } for SGLANG_DSA_FUSE_TOPK."
        )

    def get_device_int32_arange(self, length: int) -> torch.Tensor:
        if length > len(self._arange_buf):
            next_pow_of_2 = 1 << (length - 1).bit_length()
            self._arange_buf = torch.arange(
                next_pow_of_2, device=self.device, dtype=torch.int32
            )
        return self._arange_buf[:length]

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        seq_lens_cpu = (
            forward_batch.seq_lens.cpu() if in_capture else forward_batch.seq_lens_cpu
        )
        self._apply_cuda_graph_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            actual_forward_mode=getattr(forward_batch, "actual_forward_mode", None),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = self.speculative_num_draft_tokens
        else:
            draft_token_num = 0

        cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        assert forward_batch.seq_lens_cpu is not None
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item() + draft_token_num)
        # [b, max_seqlen_k]
        page_table = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        page_table_1_flattened = None
        topk_indices_offset = None

        # Centralized dispatch: decide all strategies for this batch
        self.set_dsa_prefill_impl(forward_batch)
        dsa_impl_for_batch = (
            self.dsa_decode_impl
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
            else self.dsa_prefill_impl
        )
        use_flashmla_kv = (not self.use_mha) and dsa_impl_for_batch == "flashmla_kv"
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        # Batch indices selected when cp enabled: After splitting multiple sequences,
        # a certain cp rank may not have some of these sequences.
        # We use bs_idx_cpu to mark which sequences are finally selected by the current cp rank,
        # a default value of None indicates that all sequences are selected.
        bs_idx_cpu = None
        # seq_len_cpu of selected sequences
        indexer_seq_lens_cpu = forward_batch.seq_lens_cpu
        indexer_seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            cu_seqlens_q = self.get_device_int32_arange(batch_size + 1)
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_target_verify():
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                batch_size * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
            forward_batch.extend_seq_lens_cpu = extend_seq_lens_cpu

            seqlens_expanded = seqlens_expand_triton(
                torch.tensor(extend_seq_lens_cpu, dtype=torch.int32, device=device),
                cache_seqlens_int32,
                self.speculative_num_draft_tokens * batch_size,
                self.speculative_num_draft_tokens,
            )
            page_table = torch.repeat_interleave(
                page_table, repeats=self.speculative_num_draft_tokens, dim=0
            )
        elif forward_batch.forward_mode.is_draft_extend_v2():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"

            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None

            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                forward_batch.extend_num_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )

            seqlens_expanded = seqlens_expand_triton(
                forward_batch.extend_seq_lens,
                cache_seqlens_int32,
                sum(extend_seq_lens_cpu),
                self.speculative_num_draft_tokens,
            )
            if forward_batch.forward_mode.is_draft_extend_v2():
                # DRAFT_EXTEND_V2: V2 worker pre-fills draft KV cache with ALL speculated
                # tokens upfront. All requests extend by the same fixed
                # (speculative_num_draft_tokens). Use scalar to avoid GPU sync.
                page_table = torch.repeat_interleave(
                    page_table, repeats=self.speculative_num_draft_tokens, dim=0
                )
            else:
                # DRAFT_EXTEND: the draft worker extends by (num_correct_drafts + 1)
                # per request after verification. Lengths vary per request based on
                # how many tokens were accepted.
                page_table = torch.repeat_interleave(
                    page_table, repeats=forward_batch.extend_seq_lens, dim=0
                )
        elif forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None
            extend_seq_lens = forward_batch.extend_seq_lens

            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )

            if can_dsa_prefill_cp_round_robin_split(forward_batch):
                seqlens_expanded = dsa_cp_round_robin_split_data(seqlens_expanded)
                extend_seq_lens_cpu, extend_seq_lens, bs_idx_cpu, bs_idx = (
                    dsa_cp_round_robin_split_q_seqs(
                        extend_seq_lens_cpu, extend_seq_lens
                    )
                )
                indexer_seq_lens_cpu = indexer_seq_lens_cpu[bs_idx_cpu]
                indexer_seq_lens = indexer_seq_lens[bs_idx]
                cache_seqlens_int32 = cache_seqlens_int32[bs_idx]
                cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
                max_seqlen_k = (
                    int(indexer_seq_lens_cpu.max().item() + draft_token_num)
                    if len(indexer_seq_lens_cpu) != 0
                    else 0
                )
                page_table = page_table[bs_idx, :max_seqlen_k]

            if any(forward_batch.extend_prefix_lens_cpu) or bs_idx_cpu is not None:
                max_seqlen_q = (
                    max(extend_seq_lens_cpu) if len(extend_seq_lens_cpu) != 0 else 1
                )
                cu_seqlens_q = compute_cu_seqlens(extend_seq_lens.to(torch.int32))
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k

            # Check if MHA FP8 dequantization is needed
            mha_dequantize_needed = (
                self.use_mha and self.token_to_kv_pool.dtype == torch.float8_e4m3fn
            )
            forward_batch.using_mha_one_shot_fp8_dequant = mha_dequantize_needed

            # page_table_1_flattened is only used when prefix sharing is enabled:
            has_prefix_sharing = any(forward_batch.extend_prefix_lens_cpu)
            if has_prefix_sharing and (
                topk_transform_method == TopkTransformMethod.RAGGED
                or mha_dequantize_needed
            ):
                page_table_1_flattened = torch.cat(
                    [
                        page_table[i, :kv_len]
                        for i, kv_len in enumerate(
                            indexer_seq_lens_cpu.tolist(),
                        )
                    ]
                )
                assert page_table_1_flattened.shape[0] == sum(
                    indexer_seq_lens_cpu
                ), f"{page_table_1_flattened.shape[0] = } must be the same as {sum(indexer_seq_lens_cpu) = }"

                # Validate indices when logical tokens exceed physical capacity
                # This is likely to be triggered by PP with high kv reuse & parallelism
                kv_cache_capacity = (
                    self.token_to_kv_pool.size + self.token_to_kv_pool.page_size
                )
                if forward_batch.seq_lens_sum > kv_cache_capacity:
                    max_idx = page_table_1_flattened.max().item()
                    assert max_idx < kv_cache_capacity, (
                        f"Invalid page table index: max={max_idx}, "
                        f"kv_cache_capacity={kv_cache_capacity}"
                    )

            if topk_transform_method == TopkTransformMethod.RAGGED:
                topk_indices_offset = torch.repeat_interleave(
                    cu_seqlens_k[:-1],
                    extend_seq_lens,
                )
        else:
            assert False, f"Unsupported {forward_batch.forward_mode = }"

        indexer_k_start_end, token_to_batch_idx = self._cal_indexer_k_start_end(
            forward_batch, bs_idx_cpu
        )
        # 1D, expanded seqlens (1D means cheap to compute, so always compute it)
        dsa_cache_seqlens_int32 = compute_dsa_seqlens(
            original_seq_lens=seqlens_expanded,
            dsa_index_topk=self.dsa_index_topk,
        )
        dsa_cache_seqlens_int32 = pad_dsa_cache_seqlens(
            forward_batch, dsa_cache_seqlens_int32
        )
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens_int32)
        dsa_cu_seqlens_q = self.get_device_int32_arange(len(dsa_cu_seqlens_k))

        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_batch.forward_mode,
                cache_seqlens_int32,
                seqlens_expanded,
                forward_batch.batch_size,
            )
            # NOTE: block_kv arg must be 64 here — DG computes SPLIT_KV =
            # block_kv * 4 and both DG's and the indexer's compute kernels
            # require SPLIT_KV = 256; this is independent of the cache page size.
            paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
            )

        metadata = DSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seq_lens_sum=forward_batch.seq_lens_sum,
            page_table_1=page_table,
            page_table_1_flattened=page_table_1_flattened,
            flashmla_metadata=(
                self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens_int32,
                    seq_len_q=1,
                )
                if use_flashmla_kv
                else None
            ),
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            paged_mqa_ctx_lens_2d=paged_mqa_ctx_lens_2d,
            dsa_cache_seqlens_int32=dsa_cache_seqlens_int32,
            dsa_cu_seqlens_q=dsa_cu_seqlens_q,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            dsa_seqlens_expanded=seqlens_expanded,
            dsa_extend_seq_lens_list=extend_seq_lens_cpu,
            real_page_table=self._transform_table_1_to_real(page_table),
            dsa_max_seqlen_q=1,
            topk_indices_offset=topk_indices_offset,
            indexer_k_start_end=indexer_k_start_end,
            indexer_seq_lens_cpu=indexer_seq_lens_cpu,
            indexer_seq_lens=indexer_seq_lens,
            token_to_batch_idx=token_to_batch_idx,
        )
        self.forward_metadata = metadata

    def _cal_indexer_k_start_end(
        self,
        forward_batch: ForwardBatch,
        bs_idx: Optional[List[int]] = None,
    ):
        if not forward_batch.forward_mode.is_extend_without_speculative():
            return None, None
        if forward_batch.batch_size == 0 or (bs_idx is not None and len(bs_idx) == 0):
            empty_t = torch.empty(0, dtype=torch.int32, device=self.device)
            return (empty_t, empty_t), empty_t

        # Suppose there are two requests, with extend_seq_len = [3, 2]
        # and seq_lens = [10, 4]
        # The logits matrix looks like this, with * representing the valid logits
        # and - representing the invalid logits:
        #
        #  ********--|----
        #  *********-|----
        #  **********|----
        #  ----------|***-
        #  ----------|****
        #
        # ks = [0, 0, 0, 10, 10]
        # ke = [8, 9, 10, 13, 14]
        ks_list = []
        ke_list = []
        token_to_batch_idx = []

        q_offset = 0
        k_offset = 0

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device=self.device
            )
            kv_len = seq_len
            if forward_batch.forward_mode.is_target_verify():
                kv_len += self.speculative_num_draft_tokens
            seq_lens_expanded = torch.arange(
                kv_len - extend_seq_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=self.device,
            )
            ke = ks + seq_lens_expanded
            ks_list.append(ks)
            ke_list.append(ke)

            # bi: The index within the selected batch bs_idx. Entries that were not selected are ignored.
            bi = bs_idx.index(i) if (bs_idx is not None and i in bs_idx) else i
            tb = torch.full(
                (extend_seq_len,), bi, dtype=torch.int32, device=self.device
            )
            token_to_batch_idx.append(tb)

            if bs_idx is None or i in bs_idx:  # skip batch not included in bs_idx
                q_offset += extend_seq_len
                k_offset += seq_len

        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        token_to_batch_idx = torch.cat(token_to_batch_idx, dim=0)
        if bs_idx is not None:
            assert can_dsa_prefill_cp_round_robin_split(forward_batch)
            ks = dsa_cp_round_robin_split_data(ks)
            ke = dsa_cp_round_robin_split_data(ke)
            token_to_batch_idx = dsa_cp_round_robin_split_data(token_to_batch_idx)
        return (ks, ke), token_to_batch_idx

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.decode_cuda_graph_metadata: Dict = {
            "cache_seqlens": torch.ones(
                max_num_tokens, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            # fake page_table for sparse_prefill
            # Match req_to_token's width exactly. It is over-allocated beyond
            # context_len because spec decoding lets seq_len transiently overshoot.
            "page_table": torch.zeros(
                max_num_tokens,
                self.req_to_token.shape[1],
                dtype=torch.int32,
                device=self.device,
            ),
            "flashmla_metadata": (
                self._compute_flashmla_metadata(
                    cache_seqlens=torch.ones(
                        max_num_tokens, dtype=torch.int32, device=self.device
                    ),
                    seq_len_q=1,
                )
                if self.dsa_decode_impl == "flashmla_kv"
                else None
            ),
        }

    def _build_forward_metadata_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
        actual_forward_mode: Optional[ForwardMode] = None,
    ):
        """Create and store DSAMetadata for a new batch size during CUDA graph capture."""
        self.set_dsa_prefill_impl(forward_batch=None)

        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # Get sequence information
            cache_seqlens_int32 = seq_lens.to(torch.int32)
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)

            # Use max context length for seq_len_k
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][:bs, :]
            max_seqlen_q = 1
            max_seqlen_k = page_table_1.shape[1]

            # Precompute page table
            # Precompute cumulative sequence lengths

            # NOTE(dark): this is always arange, since we are decoding
            cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][: bs + 1]
            dsa_cache_seqlens_int32 = compute_dsa_seqlens(
                cache_seqlens_int32, dsa_index_topk=self.dsa_index_topk
            )

            seqlens_expanded = cache_seqlens_int32
            dsa_extend_seq_lens_list = [1] * bs
            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, bs + 1))
                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=dsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            cache_seqlens_int32 = (seq_lens + self.speculative_num_draft_tokens).to(
                torch.int32
            )
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
            max_seqlen_q = 1
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][
                : bs * self.speculative_num_draft_tokens, :
            ]
            max_seqlen_k = page_table_1.shape[1]

            cu_seqlens_q = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=self.device,
            )

            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs

            seqlens_int32_cpu = [
                self.speculative_num_draft_tokens + kv_len
                for kv_len in seq_lens.tolist()
            ]
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seqlens_int32_cpu,
                        strict=True,
                    )
                ]
            )
            dsa_cache_seqlens_int32 = compute_dsa_seqlens(
                seqlens_expanded, dsa_index_topk=self.dsa_index_topk
            )
            dsa_extend_seq_lens_list = [1] * bs * self.speculative_num_draft_tokens

            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, bs * self.speculative_num_draft_tokens + 1))

                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=dsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None

        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens_int32)
        dsa_cu_seqlens_q = self.get_device_int32_arange(len(dsa_cu_seqlens_k))
        real_page_table = self._transform_table_1_to_real(page_table_1)

        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_mode, cache_seqlens_int32, seqlens_expanded, bs
            )
            paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
            )

        metadata = DSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            flashmla_metadata=flashmla_metadata,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            paged_mqa_ctx_lens_2d=paged_mqa_ctx_lens_2d,
            dsa_cache_seqlens_int32=dsa_cache_seqlens_int32,
            dsa_cu_seqlens_q=dsa_cu_seqlens_q,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            dsa_seqlens_expanded=seqlens_expanded,
            real_page_table=real_page_table,
            dsa_extend_seq_lens_list=dsa_extend_seq_lens_list,
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
        actual_forward_mode: Optional[ForwardMode] = None,
    ):
        """Shared capture+replay body for the cuda-graph init path.

        Public entry: :py:meth:`init_forward_metadata_out_graph`. Spec runners
        also call this directly via _apply_cuda_graph_metadata when they
        need to pass out_cache_loc / actual_forward_mode explicitly.
        """
        # NOTE: the decode / target_verify / draft_extend replay branches below
        # derive the page-table width statically (metadata.page_table_1.shape[1])
        # rather than from seq_lens_cpu.max(), so seq_lens_cpu may be None here
        # once the backend opts out of the host seq-len mirror.

        if bs not in self.decode_cuda_graph_metadata:
            self._build_forward_metadata_cuda_graph(
                bs,
                None,
                req_pool_indices,
                seq_lens,
                seq_lens_cpu,
                forward_mode,
                spec_info,
                out_cache_loc,
                actual_forward_mode,
            )
            return

        self.set_dsa_prefill_impl(forward_batch=None)

        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Normal Decode
        metadata: DSAMetadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # Static page-table width (= captured buffer width) instead of a
            # seq_lens_cpu.max() host read. The kernel bounds each row's reads by
            # the GPU cache_seqlens, so copying the full width is correct and keeps
            # this path independent of the host seq-len mirror (no per-step D2H).
            max_len = metadata.page_table_1.shape[1]

            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            page_indices = self.req_to_token[req_pool_indices, :max_len]
            metadata.page_table_1[:, :max_len].copy_(page_indices)
            dsa_cache_seqlens = compute_dsa_seqlens(
                cache_seqlens, dsa_index_topk=self.dsa_index_topk
            )
            metadata.dsa_cache_seqlens_int32.copy_(dsa_cache_seqlens)
            seqlens_expanded = cache_seqlens
        elif forward_mode.is_target_verify():
            # Static width (captured buffer already reserves +num_draft_tokens);
            # avoids the seq_lens_cpu.max() host read. See the decode branch note.
            max_seqlen_k = metadata.page_table_1.shape[1]

            cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(
                torch.int32
            )
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            page_indices = torch.repeat_interleave(
                page_indices, repeats=self.speculative_num_draft_tokens, dim=0
            )
            metadata.page_table_1[:, :max_seqlen_k].copy_(page_indices)
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs

            seqlens_expanded = seqlens_expand_triton(
                torch.tensor(
                    extend_seq_lens_cpu, dtype=torch.int32, device=self.device
                ),
                cache_seqlens,
                self.speculative_num_draft_tokens * bs,
                self.speculative_num_draft_tokens,
            )
            metadata.dsa_seqlens_expanded.copy_(seqlens_expanded)
            dsa_cache_seqlens = compute_dsa_seqlens(
                seqlens_expanded, self.dsa_index_topk
            )
            metadata.dsa_cache_seqlens_int32.copy_(dsa_cache_seqlens)
        elif forward_mode.is_draft_extend_v2():
            # Static width; avoids the seq_lens_cpu.max() host read. See the decode
            # branch note. (The spec_info.num_accept_tokens .tolist() below is a
            # separate host dependency, tracked by the draft-extend cuda-graph work.)
            max_seqlen_k = metadata.page_table_1.shape[1]
            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )

            extend_seq_lens = spec_info.num_accept_tokens[:bs]
            extend_seq_lens_cpu = extend_seq_lens.tolist()

            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            page_indices = torch.repeat_interleave(
                page_indices, repeats=extend_seq_lens, dim=0
            )
            metadata.page_table_1[: page_indices.shape[0], :max_seqlen_k].copy_(
                page_indices
            )

            seqlens_expanded = seqlens_expand_triton(
                extend_seq_lens,
                cache_seqlens,
                sum(extend_seq_lens_cpu),
                self.speculative_num_draft_tokens,
            )
            metadata.dsa_seqlens_expanded[: seqlens_expanded.shape[0]].copy_(
                seqlens_expanded
            )
            dsa_cache_seqlens = compute_dsa_seqlens(
                seqlens_expanded, self.dsa_index_topk
            )
            metadata.dsa_cache_seqlens_int32[: seqlens_expanded.shape[0]].copy_(
                dsa_cache_seqlens
            )

        # Update DeepGEMM paged MQA schedule metadata outside the captured graph.
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            if forward_mode.is_draft_extend_v2():
                schedule_seqlens_expanded = metadata.dsa_seqlens_expanded
            else:
                schedule_seqlens_expanded = seqlens_expanded
            seqlens_32_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_mode,
                metadata.cache_seqlens_int32,
                schedule_seqlens_expanded,
                bs,
            )
            new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                seqlens_32_2d, 64, deep_gemm.get_num_sms()
            )
            if metadata.paged_mqa_schedule_metadata is None:
                object.__setattr__(
                    metadata, "paged_mqa_schedule_metadata", new_schedule
                )
            else:
                metadata.paged_mqa_schedule_metadata.copy_(new_schedule)
            # `copy_` preserves the buffer's data_ptr that the captured graph captured.
            if metadata.paged_mqa_ctx_lens_2d is None:
                object.__setattr__(metadata, "paged_mqa_ctx_lens_2d", seqlens_32_2d)
            else:
                metadata.paged_mqa_ctx_lens_2d.copy_(seqlens_32_2d)
        seqlens_expanded_size = seqlens_expanded.shape[0]
        assert (
            metadata.dsa_cache_seqlens_int32 is not None
            and metadata.dsa_cu_seqlens_k is not None
            and self.dsa_index_topk is not None
        )

        metadata.dsa_cu_seqlens_k[1 : 1 + seqlens_expanded_size].copy_(
            torch.cumsum(dsa_cache_seqlens, dim=0, dtype=torch.int32)
        )
        # NOTE(dark): (dsa-) cu_seqlens_q is always arange, no need to copy

        assert self.real_page_size == metadata.page_size
        if self.real_page_size > 1:
            real_table = self._transform_table_1_to_real(page_indices)
            new_rows = real_table.shape[0]
            new_cols = real_table.shape[1]
            metadata.real_page_table[:new_rows, :new_cols].copy_(real_table)
        else:
            assert metadata.real_page_table is metadata.page_table_1

        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = metadata.flashmla_metadata.slice(
                slice(0, seqlens_expanded_size + 1)
            )
            flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens,
                    seq_len_q=1,
                )
            )

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph_from_precomputed(
        self,
        bs: int,
        precomputed: PrecomputedMetadata,
        forward_mode: ForwardMode,
    ):
        """Fast path: copy precomputed metadata to this backend's metadata.

        This function only performs copy operations, no computation.

        Args:
            bs: Batch size
            precomputed: Precomputed metadata to copy from
            forward_mode: Forward mode
        """
        self.set_dsa_prefill_impl(forward_batch=None)

        metadata = self.decode_cuda_graph_metadata[bs]

        # Track whether fused kernel succeeded
        fused_kernel_succeeded = False

        # Use fused CUDA kernel for all copy operations
        if _USE_FUSED_METADATA_COPY:
            try:
                from sglang.jit_kernel.fused_metadata_copy import (
                    fused_metadata_copy_cuda,
                )

                # Map forward_mode to integer enum
                if forward_mode.is_decode_or_idle():
                    mode_int = 0  # DECODE
                elif forward_mode.is_target_verify():
                    mode_int = 1  # TARGET_VERIFY
                else:
                    raise ValueError(f"Unsupported forward_mode: {forward_mode}")

                # Prepare FlashMLA tensors if needed
                flashmla_num_splits_src = None
                flashmla_num_splits_dst = None
                flashmla_metadata_src = None
                flashmla_metadata_dst = None
                if precomputed.flashmla_metadata is not None:
                    flashmla_num_splits_src = precomputed.flashmla_metadata.num_splits
                    flashmla_num_splits_dst = metadata.flashmla_metadata.num_splits
                    flashmla_metadata_src = (
                        precomputed.flashmla_metadata.flashmla_metadata
                    )
                    flashmla_metadata_dst = metadata.flashmla_metadata.flashmla_metadata

                # Call fused kernel
                fused_metadata_copy_cuda(
                    # Source tensors
                    precomputed.cache_seqlens,
                    precomputed.cu_seqlens_k,
                    precomputed.page_indices,
                    precomputed.dsa_cache_seqlens,
                    precomputed.seqlens_expanded,
                    precomputed.dsa_cu_seqlens_k,
                    precomputed.real_page_table,
                    flashmla_num_splits_src,
                    flashmla_metadata_src,
                    # Destination tensors
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table_1,
                    metadata.dsa_cache_seqlens_int32,
                    metadata.dsa_seqlens_expanded,
                    metadata.dsa_cu_seqlens_k,
                    (
                        metadata.real_page_table
                        if precomputed.real_page_table is not None
                        else None
                    ),
                    flashmla_num_splits_dst,
                    flashmla_metadata_dst,
                    # Parameters
                    mode_int,
                    bs,
                    precomputed.max_len,
                    precomputed.max_seqlen_k,
                    precomputed.seqlens_expanded_size,
                )

                # Successfully used fused kernel
                fused_kernel_succeeded = True

            except ImportError:
                print(
                    "Warning: Fused metadata copy kernel not available, falling back to individual copies."
                )
            except Exception as e:
                print(
                    f"Warning: Fused metadata copy kernel failed with error: {e}, falling back to individual copies."
                )

        # Fallback to individual copy operations if fused kernel disabled or failed
        if not fused_kernel_succeeded:
            # Copy basic seqlens
            metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

            # Mode-specific copy logic
            if forward_mode.is_decode_or_idle():
                # Decode mode
                metadata.page_table_1[:, : precomputed.max_len].copy_(
                    precomputed.page_indices
                )
                metadata.dsa_cache_seqlens_int32.copy_(precomputed.dsa_cache_seqlens)
                # seqlens_expanded is same as cache_seqlens (already copied)

            elif forward_mode.is_target_verify():
                # Target verify mode
                metadata.page_table_1[:, : precomputed.max_seqlen_k].copy_(
                    precomputed.page_indices
                )
                metadata.dsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
                metadata.dsa_cache_seqlens_int32.copy_(precomputed.dsa_cache_seqlens)

            # Copy DSA cu_seqlens
            size = precomputed.seqlens_expanded_size
            metadata.dsa_cu_seqlens_k[1 : 1 + size].copy_(
                precomputed.dsa_cu_seqlens_k[1 : 1 + size]
            )

            # Copy real page table
            if precomputed.real_page_table is not None:
                rows, cols = precomputed.real_page_table.shape
                metadata.real_page_table[:rows, :cols].copy_(
                    precomputed.real_page_table
                )

            # Copy FlashMLA metadata in fallback path
            if precomputed.flashmla_metadata is not None:
                size = precomputed.seqlens_expanded_size
                flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, size + 1))
                flashmla_metadata.copy_(precomputed.flashmla_metadata)

        # Refresh DeepGEMM paged MQA schedule metadata for the actual seqlens of
        # this replay (the captured graph holds stale data otherwise, which can
        # deadlock the kernel when the runtime work decomposition diverges from
        # the captured one).
        if is_cuda():
            if forward_mode.is_decode_or_idle():
                seqlens_32_2d = _to_2d_context_lens(metadata.cache_seqlens_int32, bs)
            else:
                seqlens_32_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                    forward_mode,
                    metadata.cache_seqlens_int32,
                    metadata.dsa_seqlens_expanded,
                    bs,
                )
            new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                seqlens_32_2d, 64, deep_gemm.get_num_sms()
            )
            if metadata.paged_mqa_schedule_metadata is None:
                object.__setattr__(
                    metadata, "paged_mqa_schedule_metadata", new_schedule
                )
            else:
                metadata.paged_mqa_schedule_metadata.copy_(new_schedule)
            if metadata.paged_mqa_ctx_lens_2d is None:
                object.__setattr__(metadata, "paged_mqa_ctx_lens_2d", seqlens_32_2d)
            else:
                metadata.paged_mqa_ctx_lens_2d.copy_(seqlens_32_2d)

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        causal = not layer.is_cross_attention
        metadata = self.forward_metadata
        assert causal, "DSA is causal only"

        dsa_impl = (
            self.dsa_decode_impl
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
            else self.dsa_prefill_impl
        )

        if dsa_impl == "trtllm" and not self.use_mha:
            return self._forward_trtllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                metadata.dsa_cache_seqlens_int32,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
                is_prefill=True,
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                self.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        # Use MHA kernel if in MHA_ONE_SHOT mode
        if self.use_mha:
            assert k is not None and v is not None
            assert q_rope is None, "MHA_ONE_SHOT path should not pass q_rope"
            assert (
                layer.tp_k_head_num == layer.tp_q_head_num > 1
            ), "MHA_ONE_SHOT requires dense multi-head config"
            return self._forward_standard_mha(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
            )

        # Do absorbed multi-latent attention (MLA path)
        assert q_rope is not None
        kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        # Align topk_indices with q dimensions
        # This handles cases where q is padded (TP + partial DP attention)
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # NOTE(dark): here, we use page size = 1
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        if envs.SGLANG_DSA_FUSE_TOPK.get():
            page_table_1 = self._get_fused_topk_page_table(topk_indices)
        else:
            if topk_transform_method == TopkTransformMethod.RAGGED:
                topk_indices_offset = metadata.topk_indices_offset
                assert topk_indices_offset is not None
                mask = topk_indices != -1
                topk_indices_offset = (
                    topk_indices_offset.unsqueeze(1)
                    if topk_indices_offset.ndim == 1
                    else topk_indices_offset
                )
                topk_indices = torch.where(
                    mask, topk_indices + topk_indices_offset, topk_indices
                )
            elif topk_transform_method == TopkTransformMethod.PAGED:
                assert metadata.dsa_extend_seq_lens_list is not None
                page_table_1 = transform_index_page_table_prefill(
                    page_table=metadata.page_table_1,
                    topk_indices=topk_indices,
                    extend_lens_cpu=metadata.dsa_extend_seq_lens_list,
                    page_size=1,
                )

        # todo hisparse: to cover more backends
        if self.hisparse_coordinator is not None:
            # flash_mla_sparse_fwd / tilelang require int32 page indices.
            page_table_1 = self.token_to_kv_pool.translate_loc_to_hisparse_device(
                page_table_1
            ).to(torch.int32)

        if dsa_impl == "tilelang":
            if q_rope is not None:
                # Triton prefill kernel reads q_nope/q_rope directly, skipping
                # the concat (it splits q into main/tail internally anyway).
                # Gated to gfx950 + the validated shape (16 heads, d_v=512,
                # tail=64, topk=2048); everything else uses TileLang.
                if (
                    _DSA_TRITON_PREFILL
                    and _IS_GFX95
                    and kv_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                    and layer.tp_q_head_num == 16
                    and layer.v_head_dim == 512
                    and (layer.head_dim - layer.v_head_dim) == 64
                    and page_table_1.shape[-1] == 2048
                    and q_nope.shape[0] >= 512
                ):
                    from sglang.srt.layers.attention.dsa.triton_sparse_mla import (
                        triton_sparse_mla_fwd,
                    )

                    return triton_sparse_mla_fwd(
                        q_nope=q_nope,
                        q_rope=q_rope,
                        kv=kv_cache,
                        indices=page_table_1.unsqueeze(1),
                        sm_scale=layer.scaling,
                        d_v=layer.v_head_dim,
                    )
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif dsa_impl == "flashmla_sparse":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)

            if topk_transform_method == TopkTransformMethod.RAGGED:
                if any(forward_batch.extend_prefix_lens_cpu):
                    page_table_1_flattened = (
                        self.forward_metadata.page_table_1_flattened
                    )
                    assert page_table_1_flattened is not None
                    kv_cache = dequantize_k_cache_paged(
                        kv_cache, page_table_1_flattened
                    )
                else:
                    kv_cache = _cat([k, k_rope], dim=-1)
                page_table_1 = topk_indices

            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif dsa_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif dsa_impl == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.dsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.dsa_cu_seqlens_q,
                cu_seqlens_k=metadata.dsa_cu_seqlens_k,
                max_seqlen_q=metadata.dsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif dsa_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter_extend(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
            )
        else:
            raise ValueError(
                f"Unsupported {dsa_impl = } for forward_extend. Consider using an other attention backend."
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        causal = not layer.is_cross_attention
        metadata = self.forward_metadata
        assert causal, "DSA is causal only"

        if self.dsa_decode_impl == "trtllm":
            return self._forward_trtllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                metadata.cache_seqlens_int32,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                self.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        # Do absorbed multi-latent attention
        kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            # Caller passed split q_nope / q_rope; we'll need to concat below if
            # the chosen impl wants q_all.
            q_all = None
        else:
            # Caller passed already-concatenated q (q_all = q). Reuse it directly
            # via a zero-copy view; the impl-specific blocks below will skip the
            # otherwise redundant concat_mla_absorb_q_general call.
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        # Align topk_indices with q dimensions
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        if self.hisparse_coordinator is not None:
            page_table_1 = self.hisparse_coordinator.swap_in_selected_pages(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                topk_indices,
                layer.layer_id,
            )
        elif envs.SGLANG_DSA_FUSE_TOPK.get():
            page_table_1 = self._get_fused_topk_page_table(topk_indices)
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        if self.dsa_decode_impl == "flashmla_sparse":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.dsa_decode_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif self.dsa_decode_impl == "tilelang":
            # Cat-skip (HIP-only): when caller passes q_rope=None on HIP, q_all
            # has already been set to a zero-copy view of q in the else branch
            # above and we can reuse it directly. The `not _is_hip` clause keeps
            # CUDA / MUSA paths byte-identical to pre-patch by always re-cat.
            if q_all is None or not _is_hip:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.dsa_decode_impl == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.dsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.dsa_cu_seqlens_q,
                cu_seqlens_k=metadata.dsa_cu_seqlens_k,
                max_seqlen_q=metadata.dsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif self.dsa_decode_impl == "aiter":
            if q_all is None or not _is_hip:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        else:
            assert False, f"Unsupported {self.dsa_decode_impl = }"

    def _forward_fa3(
        self,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        q_nope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        sm_scale: float,
        logit_cap: float,
        page_size: int,
    ) -> torch.Tensor:
        k_rope_cache = kv_cache[:, :, v_head_dim:]
        c_kv_cache = kv_cache[:, :, :v_head_dim]
        qk_rope_dim = k_rope_cache.shape[-1]
        k_rope_cache = k_rope_cache.view(-1, page_size, 1, qk_rope_dim)
        c_kv_cache = c_kv_cache.view(-1, page_size, 1, v_head_dim)
        o = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=sm_scale,
            causal=True,
            softcap=logit_cap,
            return_softmax_lse=False,
            num_splits=self.num_splits,
        )
        return o  # type: ignore

    def _forward_flashmla_sparse(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        # FlashMLA sparse kernel requires num_heads to be a multiple of 64 (Hopper) or 128 (Blackwell)
        # When using TP, num_heads might be smaller (e.g., 256//8=32)
        num_tokens, num_heads, head_dim = q_all.shape

        # Determine required padding based on GPU architecture (use cached value)
        required_padding = 128 if self.device_sm_major >= 10 else 64

        need_padding = num_heads % required_padding != 0

        if need_padding:
            assert required_padding % num_heads == 0, (
                f"num_heads {num_heads} cannot be padded to {required_padding}. "
                f"TP size may be too large for this model."
            )

            # Pad q to required size
            q_padded = q_all.new_zeros((num_tokens, required_padding, head_dim))
            q_padded[:, :num_heads, :] = q_all
            q_input = q_padded
        else:
            q_input = q_all

        # indices shape must be (s_q, h_kv=1, topk), keep h_kv=1 unchanged
        indices_input = page_table_1.unsqueeze(1)

        o, _, _ = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv_cache,
            indices=indices_input,
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )

        # Trim output back to original num_heads if we padded
        if need_padding:
            o = o[:, :num_heads, :]

        return o

    def _forward_flashmla_kv(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        sm_scale: float,
        layer,
        metadata: DSAMetadata,
        page_table_1,
    ) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_with_kvcache

        cache_seqlens = metadata.dsa_cache_seqlens_int32
        assert metadata.flashmla_metadata is not None

        # TODO the 2nd dim is seq_len_q, need to be >1 when MTP
        q_all = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        num_q_heads = q_all.shape[2]
        target_q_heads = self.flashmla_kv_num_q_heads
        if target_q_heads != num_q_heads:
            # Pad q heads to match FlashMLA decode supported head-count variants.
            q_input = q_all.new_zeros(
                q_all.shape[0], q_all.shape[1], target_q_heads, q_all.shape[3]
            )
            q_input[:, :, :num_q_heads, :] = q_all
        else:
            q_input = q_all

        kv_cache = kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)
        assert self.real_page_size == 64, "only page size 64 is supported"

        if not self.dsa_kv_cache_store_fp8:
            # inefficiently quantize the whole cache
            kv_cache = quantize_k_cache(kv_cache)

        indices = page_table_1.unsqueeze(1)
        assert (
            indices.shape[-1] == self.dsa_index_topk
        )  # requirement of FlashMLA decode kernel

        o, _ = flash_mla_with_kvcache(
            q=q_input,
            k_cache=kv_cache,
            cache_seqlens=cache_seqlens,
            head_dim_v=v_head_dim,
            tile_scheduler_metadata=metadata.flashmla_metadata.flashmla_metadata,
            num_splits=metadata.flashmla_metadata.num_splits,
            softmax_scale=sm_scale,
            indices=indices,
            # doc says it is not used, but if pass in None then error
            block_table=torch.empty(
                (q_all.shape[0], 0), dtype=torch.int32, device=q_all.device
            ),
            is_fp8_kvcache=True,
        )

        if target_q_heads != num_q_heads:
            o = o[:, :, :num_q_heads, :]

        return o

    def _forward_standard_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: DSAMetadata,
    ) -> torch.Tensor:
        """Standard MHA using FlashAttention varlen for MHA_ONE_SHOT mode."""
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        # MHA_ONE_SHOT: k/v include all tokens (prefix + current)
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        max_seqlen_k = metadata.max_seq_len_k
        causal = True

        # Verify batch sizes match (length of cu_seqlens should be batch_size + 1)
        assert len(cu_seqlens_q) == len(cu_seqlens_k), (
            f"batch_size mismatch: cu_seqlens_q has {len(cu_seqlens_q)-1} requests, "
            f"cu_seqlens_k has {len(cu_seqlens_k)-1} requests"
        )

        # Use TRTLLm ragged attention for SM100 (Blackwell/B200) to avoid FA4 accuracy issues
        if self.device_sm_major >= 10:
            import flashinfer

            seq_lens = metadata.cache_seqlens_int32
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=self.workspace_buffer,
                seq_lens=seq_lens,
                max_q_len=metadata.max_seq_len_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=forward_batch.batch_size,
                window_left=-1,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                enable_pdl=False,
                is_causal=causal,
                return_lse=False,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
            )

        # Use FA3 for SM90 (Hopper/H200)
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=layer.scaling,
            causal=causal,
        )

    def _forward_tilelang(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        return tilelang_sparse_fwd(
            q=q_all,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )

    def _forward_aiter(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
        metadata: DSAMetadata,
        bs: int,
    ) -> torch.Tensor:
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if self.need_pad_heads:
            q_kernel = q.view(
                -1, layer.tp_q_head_num, layer.head_dim
            ).repeat_interleave(self.head_repeat_factor, dim=1)
            o_kernel = q.new_empty(
                (
                    q.shape[0],
                    layer.tp_q_head_num * self.head_repeat_factor,
                    layer.v_head_dim,
                )
            )
        else:
            q_kernel = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            o_kernel = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        q_scale = None
        kv_scale = None
        aiter_persistent_kwargs = {}
        if kv_cache.dtype == fp8_dtype:
            kv_scale = torch.ones((), dtype=torch.float32, device=q_kernel.device)

        kv_indptr = self.kv_indptr

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)
        kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

        kv_indices = self.kv_indices
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)

        kv_last_page_lens = metadata.cu_seqlens_q
        if kv_cache.dtype == fp8_dtype:
            aiter_persistent_kwargs = self._prepare_aiter_dsa_decode_metadata(
                metadata.cu_seqlens_q,
                kv_indptr,
                bs,
                metadata.max_seq_len_q,
                q_kernel.dtype,
                kv_cache.dtype,
            )
            kv_last_page_lens = aiter_persistent_kwargs.pop("kv_last_page_lens")

        mla_decode_fwd(
            q_kernel,
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o_kernel,
            metadata.cu_seqlens_q,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            metadata.max_seq_len_q,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            q_scale=q_scale,
            kv_scale=kv_scale,
            **aiter_persistent_kwargs,
        )

        if self.need_pad_heads:
            o = o_kernel[:, :: self.head_repeat_factor, :]

        return o

    def _forward_aiter_extend(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
    ) -> torch.Tensor:
        num_tokens = q_all.shape[0]
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((num_tokens, layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if self.need_pad_heads:
            q_kernel = q.view(
                -1, layer.tp_q_head_num, layer.head_dim
            ).repeat_interleave(self.head_repeat_factor, dim=1)
            o_kernel = q.new_empty(
                (
                    num_tokens,
                    layer.tp_q_head_num * self.head_repeat_factor,
                    layer.v_head_dim,
                )
            )
        else:
            q_kernel = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            o_kernel = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        q_scale = None
        kv_scale = None
        aiter_persistent_kwargs = {}
        if kv_cache.dtype == fp8_dtype:
            kv_scale = torch.ones((), dtype=torch.float32, device=q_kernel.device)

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)

        kv_indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = torch.cumsum(non_minus1_counts, dim=0)

        # Allocate kv_indices with upper-bound size (num_tokens * topk)
        topk = page_table_1.shape[1]
        kv_indices = torch.zeros(
            num_tokens * topk, dtype=torch.int32, device=self.device
        )

        # Use get_valid_kv_indices kernel to extract valid indices
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, num_tokens)

        # Build cu_seqlens_q for extend: each token is treated as seq_len_q=1
        cu_seqlens_q = torch.arange(
            0, num_tokens + 1, dtype=torch.int32, device=self.device
        )
        kv_last_page_lens = cu_seqlens_q
        if kv_cache.dtype == fp8_dtype:
            aiter_persistent_kwargs = self._prepare_aiter_dsa_decode_metadata(
                cu_seqlens_q,
                kv_indptr,
                num_tokens,
                1,
                q_kernel.dtype,
                kv_cache.dtype,
            )
            kv_last_page_lens = aiter_persistent_kwargs.pop("kv_last_page_lens")

        # TODO support more forward_mode
        mla_decode_fwd(
            q_kernel,
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o_kernel,
            cu_seqlens_q,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            1,  # max_seq_len_q = 1 for per-token attention
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            q_scale=q_scale,
            kv_scale=kv_scale,
            **aiter_persistent_kwargs,
        )

        if self.need_pad_heads:
            o = o_kernel[:, :: self.head_repeat_factor, :]

        return o

    def _forward_trtllm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """Forward using TRT-LLM sparse MLA kernel."""
        import flashinfer.decode

        metadata = self.forward_metadata

        merge_query = q_rope is not None
        if self.kv_cache_dtype == torch.float8_e4m3fn:
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            assert q_rope is not None, "For FP8 path q_rope should not be None."
            assert k_rope is not None, "For FP8 path k_rope should not be None."
            assert (
                cos_sin_cache is not None
            ), "For FP8 path cos_sin_cache should not be None."

            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

            # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.real_page_size, self.kv_cache_dim).unsqueeze(1)

        if merge_query:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q_all = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            q_all = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Align topk_indices with q dimensions
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q.shape[0])

        if envs.SGLANG_DSA_FUSE_TOPK.get():
            page_table_1 = self._get_fused_topk_page_table(topk_indices)
        elif is_prefill:
            page_table_1 = transform_index_page_table_prefill(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                extend_lens_cpu=metadata.dsa_extend_seq_lens_list,
                page_size=1,
            )
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling

        batch_size = page_table_1.shape[0]
        _, num_heads, head_dim = q_all.shape

        q = q_all.view(batch_size, 1, num_heads, head_dim)
        kv = kv_cache.view(-1, 1, self.real_page_size, self.kv_cache_dim)
        block_tables = page_table_1.unsqueeze(1)
        seq_lens = metadata.cache_seqlens_int32 if seq_lens is None else seq_lens

        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            sparse_mla_top_k=self.dsa_index_topk,
            bmm1_scale=bmm1_scale,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        )

        return out

    def _pad_topk_indices(
        self, topk_indices: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        current_tokens = topk_indices.shape[0]
        if current_tokens == num_tokens:
            return topk_indices

        assert current_tokens <= num_tokens, (
            f"topk_indices rows ({current_tokens}) > num_tokens ({num_tokens}); "
            "this indicates a mismatch between indexer output and q layout."
        )

        pad_size = num_tokens - current_tokens
        padding = torch.full(
            (pad_size, topk_indices.shape[1]),
            -1,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        return torch.cat([topk_indices, padding], dim=0)

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def set_dsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
        """
        Decide all attention prefill dispatch strategies for this batch.
        """
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            is_in_tc_piecewise_cuda_graph,
        )
        from sglang.srt.utils import get_device_sm, is_blackwell

        # Decide MHA vs MLA
        if is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph():
            # Can't branch on seq_lens_cpu in graph replay, force MHA off to
            # guarantee correctness.
            self.use_mha = False
        elif (
            forward_batch and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            # Check if sequence meets criteria for MHA_ONE_SHOT
            assert forward_batch.seq_lens_cpu is not None
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            sum_seq_lens = sum(forward_batch.seq_lens_cpu)
            device_sm = get_device_sm()

            # Requirements: H200/B200, short sequences, supported dtype, fits in chunk
            self.use_mha = (
                (
                    device_sm == 90 or (device_sm >= 100 and device_sm < 110)
                )  # SM90/SM100 only
                and max_kv_len
                <= envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()  # Short enough for MHA
                and self.token_to_kv_pool.dtype in [torch.bfloat16, torch.float8_e4m3fn]
                and sum_seq_lens
                <= forward_batch.get_max_chunk_capacity()  # Fits in chunk
                and (not is_dsa_enable_prefill_cp())  # CP not enabled
                and (self.hisparse_coordinator is None)
            )
        else:
            self.use_mha = False  # Decode/verify always use MLA

        # Set MLA implementation only if not using MHA
        if not self.use_mha and self.enable_auto_select_prefill_impl:
            if self.dsa_kv_cache_store_fp8:
                if (
                    is_blackwell()
                    and forward_batch is not None
                    and forward_batch.forward_mode == ForwardMode.EXTEND
                ):
                    total_kv_tokens = forward_batch.seq_lens_sum
                    total_q_tokens = forward_batch.extend_num_tokens
                    # Heuristic based on benchmarking flashmla_kv vs flashmla_sparse + dequantize_k_cache_paged
                    if total_kv_tokens < total_q_tokens * 512:
                        self.dsa_prefill_impl = "flashmla_sparse"
                        return
                self.dsa_prefill_impl = "flashmla_kv"
            else:
                # bf16 kv cache
                self.dsa_prefill_impl = "flashmla_sparse"

    def get_topk_transform_method(
        self, forward_mode: Optional[ForwardMode] = None
    ) -> TopkTransformMethod:
        """
        SGLANG_DSA_FUSE_TOPK controls whether to fuse the topk transform into the topk kernel.
        This method is used to select the topk transform method which can be fused or unfused.
        """
        if (
            # disable for MTP
            self.dsa_kv_cache_store_fp8
            and self.dsa_prefill_impl == "flashmla_sparse"
            and forward_mode == ForwardMode.EXTEND
        ):
            topk_transform_method = TopkTransformMethod.RAGGED
        else:
            topk_transform_method = TopkTransformMethod.PAGED
        return topk_transform_method

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> DSAIndexerMetadata:
        force_unfused = (
            self.hisparse_coordinator is not None
            and forward_batch.forward_mode.is_decode_or_idle()
        )
        return DSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            topk_transform_method=self.get_topk_transform_method(
                forward_batch.forward_mode
            ),
            topk_backend=self.dsa_topk_backend,
            paged_mqa_schedule_metadata=self.forward_metadata.paged_mqa_schedule_metadata,
            paged_mqa_ctx_lens_2d=self.forward_metadata.paged_mqa_ctx_lens_2d,
            force_unfused_topk=force_unfused,
        )

    def _compute_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        from sgl_kernel.flash_mla import get_mla_metadata

        num_heads_q = self.flashmla_kv_num_q_heads

        flashmla_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            num_q_tokens_per_head_k=seq_len_q * num_heads_q // 1,
            num_heads_k=1,
            num_heads_q=num_heads_q,
            is_fp8_kvcache=True,
            topk=self.dsa_index_topk,
        )

        return DSAFlashMLAMetadata(
            flashmla_metadata=flashmla_metadata,
            num_splits=num_splits,
        )


class DeepseekSparseAttnMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                DeepseekSparseAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if in_capture:
            inner_fb = build_inner_fb_view(
                forward_batch,
                bs=forward_batch.batch_size,
                forward_mode=ForwardMode.DECODE,
            )
            for i in range(self.speculative_num_steps - 1):
                self.attn_backends[i].init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )
            return

        bs = forward_batch.batch_size
        if envs.SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA.get():
            # Precompute metadata once (shared across all backends)
            precomputed = self.attn_backends[0]._precompute_replay_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                forward_mode=ForwardMode.DECODE,
            )

            # Use multi-backend fused copy when we have 3 or more backends
            # This is 3x faster than calling the single-backend copy 3 times
            if self.speculative_num_steps > 3:
                try:
                    from sglang.jit_kernel.fused_metadata_copy import (
                        fused_metadata_copy_multi_cuda,
                    )

                    metadata0 = self.attn_backends[0].decode_cuda_graph_metadata[bs]
                    metadata1 = self.attn_backends[1].decode_cuda_graph_metadata[bs]
                    metadata2 = self.attn_backends[2].decode_cuda_graph_metadata[bs]

                    # Set dsa_prefill_impl for first 3 backends (required by the method)
                    for i in range(3):
                        self.attn_backends[i].set_dsa_prefill_impl(forward_batch=None)

                    # Prepare FlashMLA tensors if needed
                    flashmla_num_splits_src = None
                    flashmla_metadata_src = None
                    flashmla_num_splits_dst0 = None
                    flashmla_num_splits_dst1 = None
                    flashmla_num_splits_dst2 = None
                    flashmla_metadata_dst0 = None
                    flashmla_metadata_dst1 = None
                    flashmla_metadata_dst2 = None

                    if precomputed.flashmla_metadata is not None:
                        flashmla_num_splits_src = (
                            precomputed.flashmla_metadata.num_splits
                        )
                        flashmla_metadata_src = (
                            precomputed.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_num_splits_dst0 = (
                            metadata0.flashmla_metadata.num_splits
                        )
                        flashmla_num_splits_dst1 = (
                            metadata1.flashmla_metadata.num_splits
                        )
                        flashmla_num_splits_dst2 = (
                            metadata2.flashmla_metadata.num_splits
                        )
                        flashmla_metadata_dst0 = (
                            metadata0.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_metadata_dst1 = (
                            metadata1.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_metadata_dst2 = (
                            metadata2.flashmla_metadata.flashmla_metadata
                        )

                    # Call the multi-backend fused kernel for first 3 backends
                    fused_metadata_copy_multi_cuda(
                        # Source tensors
                        precomputed.cache_seqlens,
                        precomputed.cu_seqlens_k,
                        precomputed.page_indices,
                        precomputed.dsa_cache_seqlens,
                        precomputed.dsa_cu_seqlens_k,
                        precomputed.real_page_table,
                        flashmla_num_splits_src,
                        flashmla_metadata_src,
                        # Destination tensors for backend 0
                        metadata0.cache_seqlens_int32,
                        metadata0.cu_seqlens_k,
                        metadata0.page_table_1,
                        metadata0.dsa_cache_seqlens_int32,
                        metadata0.dsa_cu_seqlens_k,
                        (
                            metadata0.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst0,
                        flashmla_metadata_dst0,
                        # Destination tensors for backend 1
                        metadata1.cache_seqlens_int32,
                        metadata1.cu_seqlens_k,
                        metadata1.page_table_1,
                        metadata1.dsa_cache_seqlens_int32,
                        metadata1.dsa_cu_seqlens_k,
                        (
                            metadata1.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst1,
                        flashmla_metadata_dst1,
                        # Destination tensors for backend 2
                        metadata2.cache_seqlens_int32,
                        metadata2.cu_seqlens_k,
                        metadata2.page_table_1,
                        metadata2.dsa_cache_seqlens_int32,
                        metadata2.dsa_cu_seqlens_k,
                        (
                            metadata2.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst2,
                        flashmla_metadata_dst2,
                        # Parameters
                        bs,
                        precomputed.max_len,
                        precomputed.seqlens_expanded_size,
                    )

                    # Copy remaining backends one by one (if > 3 backends)
                    for i in range(3, self.speculative_num_steps - 1):
                        self.attn_backends[
                            i
                        ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                            bs=bs,
                            precomputed=precomputed,
                            forward_mode=ForwardMode.DECODE,
                        )
                except (ImportError, Exception) as e:
                    # Fallback to loop if multi-backend kernel not available or fails
                    if isinstance(e, ImportError):
                        print(
                            "Warning: Multi-backend fused metadata copy kernel not available, falling back to loop."
                        )
                    else:
                        print(
                            f"Warning: Multi-backend fused metadata copy kernel failed with error: {e}, falling back to loop."
                        )
                    for i in range(self.speculative_num_steps - 1):
                        self.attn_backends[
                            i
                        ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                            bs=bs,
                            precomputed=precomputed,
                            forward_mode=ForwardMode.DECODE,
                        )
            else:
                # Less than 3 backends: copy to each backend individually
                for i in range(self.speculative_num_steps - 1):
                    self.attn_backends[
                        i
                    ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                        bs=bs,
                        precomputed=precomputed,
                        forward_mode=ForwardMode.DECODE,
                    )
        else:
            for i in range(self.speculative_num_steps - 1):
                self.attn_backends[i]._apply_cuda_graph_metadata(
                    bs=bs,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                    out_cache_loc=None,
                )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_in_graph(forward_batch)


# Backward-compat aliases (deprecated: use DSA class names)
DeepseekSparseAttnBackend = DeepseekSparseAttnBackend
DeepseekSparseAttnMultiStepBackend = DeepseekSparseAttnMultiStepBackend
DSAMetadata = DSAMetadata
DSAFlashMLAMetadata = DSAFlashMLAMetadata
DSAIndexerMetadata = DSAIndexerMetadata
