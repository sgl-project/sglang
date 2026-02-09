from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, TypeAlias

import torch

from sglang.srt.configs.model_config import get_nsa_index_topk, is_deepseek_nsa
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.srt.layers.attention.nsa.nsa_backend_mtp_precompute import (
    NativeSparseAttnBackendMTPPrecomputeMixin,
    PrecomputedMetadata,
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.nsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_prefill_cp_round_robin_split,
    compute_nsa_seqlens,
    is_nsa_enable_prefill_cp,
    nsa_cp_round_robin_split_data,
    nsa_cp_round_robin_split_q_seqs,
    pad_nsa_cache_seqlens,
)
from sglang.srt.layers.attention.trtllm_mla_backend import _concat_mla_absorb_q_general
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_cuda, is_hip

# from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


_is_hip = is_hip()

if _is_hip:
    try:
        from aiter import (  # noqa: F401
            flash_attn_varlen_func,
            mha_batch_prefill_func,
            paged_attention_ragged,
        )
        from aiter.mla import mla_decode_fwd, mla_prefill_fwd  # noqa: F401
    except ImportError:
        print(
            "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
        )
else:
    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


# Reuse this workspace buffer across all NSA backend instances
global_workspace_buffer = None


@dataclass(frozen=True)
class NSAFlashMLAMetadata:
    """Metadata only needed by FlashMLA"""

    flashmla_metadata: torch.Tensor
    num_splits: torch.Tensor

    def slice(self, sli):
        return NSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=self.num_splits[sli],
        )

    def copy_(self, other: "NSAFlashMLAMetadata"):
        self.flashmla_metadata.copy_(other.flashmla_metadata)
        self.num_splits.copy_(other.num_splits)


@dataclass(frozen=True)
class NSAMetadata:
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

    # NSA metadata (nsa prefill are expanded)
    nsa_cache_seqlens_int32: torch.Tensor  # this seqlens is clipped to `topk`
    nsa_cu_seqlens_q: torch.Tensor  # must be arange(0, len(nsa_cu_seqlens_k))
    nsa_cu_seqlens_k: torch.Tensor  # cumsum of `nsa_cache_seqlens_int32`
    nsa_extend_seq_lens_list: List[int]
    nsa_seqlens_expanded: torch.Tensor  # expanded, unclipped `seqlens`
    nsa_max_seqlen_q: Literal[1] = 1  # always 1 for decode, variable for extend

    flashmla_metadata: Optional[NSAFlashMLAMetadata] = None
    # DeepGEMM schedule metadata for paged MQA logits (decode/target_verify/draft_extend only).
    # Precomputed once per forward batch and reused across layers.
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
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
    # batch index for each token.
    token_to_batch_idx: Optional[torch.Tensor] = None


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()


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
class NSAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: NSAMetadata
    topk_transform_method: TopkTransformMethod
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.nsa_seqlens_expanded

    def get_cu_seqlens_k(self) -> torch.Tensor:
        return self.attn_metadata.cu_seqlens_k

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn_metadata.indexer_k_start_end

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens_cpu

    def get_nsa_extend_len_cpu(self) -> List[int]:
        return self.attn_metadata.nsa_extend_seq_lens_list

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: torch.Tensor = None,
        ke_offset: torch.Tensor = None,
        batch_idx_list: List[int] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sgl_kernel import (
            fast_topk_transform_fused,
            fast_topk_transform_ragged_fused,
            fast_topk_v2,
        )

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
        if batch_idx_list is not None:
            page_table_size_1 = self.attn_metadata.page_table_1[batch_idx_list]
        else:
            page_table_size_1 = self.attn_metadata.page_table_1

        if not envs.SGLANG_NSA_FUSE_TOPK.get():
            return fast_topk_v2(logits, seq_lens_topk, topk, row_starts=ks)
        elif self.topk_transform_method == TopkTransformMethod.PAGED:
            # NOTE(dark): if fused, we return a transformed page table directly
            return fast_topk_transform_fused(
                score=logits,
                lengths=seq_lens_topk,
                page_table_size_1=page_table_size_1,
                cu_seqlens_q=cu_seqlens_q_topk,
                topk=topk,
                row_starts=ks,
            )
        elif self.topk_transform_method == TopkTransformMethod.RAGGED:
            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=seq_lens_topk,
                topk_indices_offset=cu_topk_indices_offset,
                topk=topk,
                row_starts=ks,
            )
        else:
            assert False, f"Unsupported {self.topk_transform_method = }"


_NSA_IMPL_T: TypeAlias = Literal[
    "flashmla_sparse", "flashmla_kv", "fa3", "tilelang", "trtllm"
]


class NativeSparseAttnBackend(
    NativeSparseAttnBackendMTPPrecomputeMixin, AttentionBackend
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
        self.forward_metadata: NSAMetadata
        self.device = model_runner.device
        assert isinstance(model_runner.page_size, int)
        self.real_page_size = model_runner.page_size
        self.num_splits = (
            1 if model_runner.server_args.enable_deterministic_inference else 0
        )
        self.use_nsa = is_deepseek_nsa(model_runner.model_config.hf_config)
        assert self.use_nsa, "NSA backend only supports DeepSeek NSA"
        self.nsa_kv_cache_store_fp8 = (
            model_runner.token_to_kv_pool.nsa_kv_cache_store_fp8
        )
        self.nsa_index_topk = get_nsa_index_topk(model_runner.model_config.hf_config)
        self.max_context_len = model_runner.model_config.context_len
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_cache_dim = model_runner.token_to_kv_pool.kv_cache_dim
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim

        assert model_runner.req_to_token_pool is not None
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mha: bool = False
        self.nsa_prefill_impl: _NSA_IMPL_T = (
            model_runner.server_args.nsa_prefill_backend
        )
        self.nsa_decode_impl: _NSA_IMPL_T = model_runner.server_args.nsa_decode_backend
        self.enable_auto_select_prefill_impl = self.nsa_prefill_impl == "flashmla_auto"

        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)

        if _is_hip:
            max_bs = model_runner.req_to_token_pool.size

            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
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

        # Allocate global workspace buffer for TRT-LLM kernels (ragged attention on SM100/B200, or trtllm decode)
        if self.device_sm_major >= 10 or self.nsa_decode_impl == "trtllm":
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

    def get_device_int32_arange(self, l: int) -> torch.Tensor:
        if l > len(self._arange_buf):
            next_pow_of_2 = 1 << (l - 1).bit_length()
            self._arange_buf = torch.arange(
                next_pow_of_2, device=self.device, dtype=torch.int32
            )
        return self._arange_buf[:l]

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

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
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        page_table_1_flattened = None
        topk_indices_offset = None

        # Centralized dispatch: decide all strategies for this batch
        self.set_nsa_prefill_impl(forward_batch)
        topk_transform_method = self.get_topk_transform_method()
        # Batch indices selected when cp enabled: After splitting multiple sequences,
        # a certain cp rank may not have some of these sequences.
        # We use bs_idx_cpu to mark which sequences are finally selected by the current cp rank,
        # a default value of None indicates that all sequences are selected.
        bs_idx_cpu = None
        # seq_len_cpu of selected sequences
        indexer_seq_lens_cpu = forward_batch.seq_lens_cpu

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

            seqlens_int32_cpu = [
                self.speculative_num_draft_tokens + kv_len
                for kv_len in forward_batch.seq_lens_cpu.tolist()
            ]
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seqlens_int32_cpu,
                        strict=True,
                    )
                ]
            )
            page_table = torch.repeat_interleave(
                page_table, repeats=self.speculative_num_draft_tokens, dim=0
            )
        elif forward_batch.forward_mode.is_draft_extend(include_v2=True):
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
            if forward_batch.forward_mode.is_draft_extend_v2():
                # DRAFT_EXTEND_V2: V2 worker pre-fills draft KV cache with ALL speculated
                # tokens upfront. All requests extend by the same fixed
                # (speculative_num_draft_tokens). Use scalar to avoid GPU sync.
                page_table = torch.repeat_interleave(
                    page_table, repeats=self.speculative_num_draft_tokens, dim=0
                )
            else:
                # DRAFT_EXTEND (v1): V1 worker extends by (accept_length + 1) per request
                # after verification. Lengths vary per request based on how many tokens
                # were accepted.
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

            if can_nsa_prefill_cp_round_robin_split(forward_batch):
                seqlens_expanded = nsa_cp_round_robin_split_data(seqlens_expanded)
                extend_seq_lens_cpu, extend_seq_lens, bs_idx_cpu, bs_idx = (
                    nsa_cp_round_robin_split_q_seqs(
                        extend_seq_lens_cpu, extend_seq_lens
                    )
                )
                indexer_seq_lens_cpu = indexer_seq_lens_cpu[bs_idx_cpu]
                cache_seqlens_int32 = cache_seqlens_int32[bs_idx]
                cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
                max_seqlen_k = (
                    int(indexer_seq_lens_cpu.max().item() + draft_token_num)
                    if len(indexer_seq_lens_cpu) != 0
                    else 0
                )
                page_table = page_table[bs_idx, :max_seqlen_k]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
                or bs_idx_cpu is not None
            ):
                max_seqlen_q = (
                    max(extend_seq_lens_cpu) if len(extend_seq_lens_cpu) != 0 else 1
                )
                cu_seqlens_q = compute_cu_seqlens(extend_seq_lens.to(torch.int32))
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k

            # Check if MHA FP8 dequantization is needed
            mha_dequantize_needed = (
                self.use_mha
                and forward_batch.token_to_kv_pool.dtype == torch.float8_e4m3fn
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
                    forward_batch.token_to_kv_pool.size
                    + forward_batch.token_to_kv_pool.page_size
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
        nsa_cache_seqlens_int32 = compute_nsa_seqlens(
            original_seq_lens=seqlens_expanded,
            nsa_index_topk=self.nsa_index_topk,
        )
        nsa_cache_seqlens_int32 = pad_nsa_cache_seqlens(
            forward_batch, nsa_cache_seqlens_int32
        )
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))

        paged_mqa_schedule_metadata = None
        # DeepGEMM paged MQA logits path needs a schedule metadata tensor.
        # Compute it once per forward batch and reuse it across layers.
        if is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                # NOTE: DeepGEMM paged path uses block_size=64.
                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_batch.forward_mode.is_target_verify()
                        or forward_batch.forward_mode.is_draft_extend()
                    )
                    else cache_seqlens_int32
                )
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
            except (ImportError, ModuleNotFoundError):
                paged_mqa_schedule_metadata = None

        metadata = NSAMetadata(
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
                    cache_seqlens=nsa_cache_seqlens_int32,
                    seq_len_q=1,
                )
                if self.nsa_decode_impl == "flashmla_kv"
                else None
            ),
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            nsa_extend_seq_lens_list=extend_seq_lens_cpu,
            real_page_table=self._transform_table_1_to_real(page_table),
            nsa_max_seqlen_q=1,
            topk_indices_offset=topk_indices_offset,
            indexer_k_start_end=indexer_k_start_end,
            indexer_seq_lens_cpu=indexer_seq_lens_cpu,
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
            assert can_nsa_prefill_cp_round_robin_split(forward_batch)
            ks = nsa_cp_round_robin_split_data(ks)
            ke = nsa_cp_round_robin_split_data(ke)
            token_to_batch_idx = nsa_cp_round_robin_split_data(token_to_batch_idx)
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
            "page_table": torch.zeros(
                max_num_tokens,
                self.max_context_len,
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
                if self.nsa_decode_impl == "flashmla_kv"
                else None
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        self.set_nsa_prefill_impl(forward_batch=None)

        """Initialize forward metadata for capturing CUDA graph."""
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
            nsa_cache_seqlens_int32 = compute_nsa_seqlens(
                cache_seqlens_int32, nsa_index_topk=self.nsa_index_topk
            )

            seqlens_expanded = cache_seqlens_int32
            nsa_extend_seq_lens_list = [1] * num_tokens
            if self.nsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, num_tokens + 1))
                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=nsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend(
            include_v2=True
        ):
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
            nsa_cache_seqlens_int32 = compute_nsa_seqlens(
                seqlens_expanded, nsa_index_topk=self.nsa_index_topk
            )
            nsa_extend_seq_lens_list = [1] * bs * self.speculative_num_draft_tokens

            if self.nsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, bs * self.speculative_num_draft_tokens + 1))

                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=nsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None

        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))
        real_page_table = self._transform_table_1_to_real(page_table_1)

        paged_mqa_schedule_metadata = None
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_mode.is_target_verify()
                        or forward_mode.is_draft_extend()
                    )
                    else cache_seqlens_int32
                )
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
            except (ImportError, ModuleNotFoundError):
                paged_mqa_schedule_metadata = None

        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            flashmla_metadata=flashmla_metadata,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            real_page_table=real_page_table,
            nsa_extend_seq_lens_list=nsa_extend_seq_lens_list,
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        assert seq_lens_cpu is not None

        self.set_nsa_prefill_impl(forward_batch=None)

        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Normal Decode
        metadata: NSAMetadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            max_len = int(seq_lens_cpu.max().item())

            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            page_indices = self.req_to_token[req_pool_indices, :max_len]
            metadata.page_table_1[:, :max_len].copy_(page_indices)
            nsa_cache_seqlens = compute_nsa_seqlens(
                cache_seqlens, nsa_index_topk=self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
            seqlens_expanded = cache_seqlens
        elif forward_mode.is_target_verify():
            max_seqlen_k = int(
                seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
            )

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

            seqlens_int32_cpu = [
                self.speculative_num_draft_tokens + kv_len
                for kv_len in seq_lens_cpu.tolist()
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
            metadata.nsa_seqlens_expanded.copy_(seqlens_expanded)
            nsa_cache_seqlens = compute_nsa_seqlens(
                seqlens_expanded, self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
        elif forward_mode.is_draft_extend(include_v2=True):
            max_seqlen_k = int(seq_lens_cpu.max().item())
            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )

            extend_seq_lens = spec_info.accept_length[:bs]
            extend_seq_lens_cpu = extend_seq_lens.tolist()

            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            page_indices = torch.repeat_interleave(
                page_indices, repeats=extend_seq_lens, dim=0
            )
            metadata.page_table_1[: page_indices.shape[0], :max_seqlen_k].copy_(
                page_indices
            )

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
                        seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )
            metadata.nsa_seqlens_expanded[: seqlens_expanded.shape[0]].copy_(
                seqlens_expanded
            )
            nsa_cache_seqlens = compute_nsa_seqlens(
                seqlens_expanded, self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32[: seqlens_expanded.shape[0]].copy_(
                nsa_cache_seqlens
            )

        # Update DeepGEMM paged MQA schedule metadata outside the captured graph.
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_mode.is_target_verify()
                        or forward_mode.is_draft_extend()
                    )
                    else metadata.cache_seqlens_int32
                )
                new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
                if metadata.paged_mqa_schedule_metadata is None:
                    metadata.paged_mqa_schedule_metadata = new_schedule
                else:
                    metadata.paged_mqa_schedule_metadata.copy_(new_schedule)
            except (ImportError, ModuleNotFoundError):
                metadata.paged_mqa_schedule_metadata = None
        seqlens_expanded_size = seqlens_expanded.shape[0]
        assert (
            metadata.nsa_cache_seqlens_int32 is not None
            and metadata.nsa_cu_seqlens_k is not None
            and self.nsa_index_topk is not None
        )

        metadata.nsa_cu_seqlens_k[1 : 1 + seqlens_expanded_size].copy_(
            torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32)
        )
        # NOTE(dark): (nsa-) cu_seqlens_q is always arange, no need to copy

        assert self.real_page_size == metadata.page_size
        if self.real_page_size > 1:
            real_table = self._transform_table_1_to_real(page_indices)
            new_rows = real_table.shape[0]
            new_cols = real_table.shape[1]
            metadata.real_page_table[:new_rows, :new_cols].copy_(real_table)
        else:
            assert metadata.real_page_table is metadata.page_table_1

        if self.nsa_decode_impl == "flashmla_kv":
            flashmla_metadata = metadata.flashmla_metadata.slice(
                slice(0, seqlens_expanded_size + 1)
            )
            flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens,
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
        self.set_nsa_prefill_impl(forward_batch=None)

        metadata = self.decode_cuda_graph_metadata[bs]

        # Copy basic seqlens
        metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)
        metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

        # Mode-specific copy logic
        if forward_mode.is_decode_or_idle():
            # Decode mode
            metadata.page_table_1[:, : precomputed.max_len].copy_(
                precomputed.page_indices
            )
            metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)
            # seqlens_expanded is same as cache_seqlens (already copied)

        elif forward_mode.is_target_verify():
            # Target verify mode
            metadata.page_table_1[:, : precomputed.max_seqlen_k].copy_(
                precomputed.page_indices
            )
            metadata.nsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
            metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)

        elif forward_mode.is_draft_extend():
            # Draft extend mode
            rows = precomputed.page_indices.shape[0]
            cols = precomputed.max_seqlen_k
            metadata.page_table_1[:rows, :cols].copy_(precomputed.page_indices)

            size = precomputed.seqlens_expanded_size
            metadata.nsa_seqlens_expanded[:size].copy_(precomputed.seqlens_expanded)
            metadata.nsa_cache_seqlens_int32[:size].copy_(precomputed.nsa_cache_seqlens)

        # Copy NSA cu_seqlens
        size = precomputed.seqlens_expanded_size
        metadata.nsa_cu_seqlens_k[1 : 1 + size].copy_(
            precomputed.nsa_cu_seqlens_k[1 : 1 + size]
        )

        # Copy real page table
        if precomputed.real_page_table is not None:
            rows, cols = precomputed.real_page_table.shape
            metadata.real_page_table[:rows, :cols].copy_(precomputed.real_page_table)
        else:
            # real_page_table is same as page_table_1 (already copied)
            pass

        # Copy FlashMLA metadata
        if precomputed.flashmla_metadata is not None:
            flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, size + 1))
            flashmla_metadata.copy_(precomputed.flashmla_metadata)

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
    ) -> torch.Tensor:

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "NSA is causal only"

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
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

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
        topk_transform_method = self.get_topk_transform_method()
        if envs.SGLANG_NSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
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
                assert metadata.nsa_extend_seq_lens_list is not None
                page_table_1 = transform_index_page_table_prefill(
                    page_table=metadata.page_table_1,
                    topk_indices=topk_indices,
                    extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                    page_size=1,
                )

        nsa_impl = (
            self.nsa_decode_impl
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            )
            else self.nsa_prefill_impl
        )

        if nsa_impl == "tilelang":
            if q_rope is not None:
                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif nsa_impl == "flashmla_sparse":
            if q_rope is not None:
                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)

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
        elif nsa_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
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
        elif nsa_impl == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif nsa_impl == "trtllm":
            assert forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend(
                include_v2=True
            ), "TRT-LLM NSA only supports target_verify/draft_extend; normal extend untested."
            if q_rope is not None:
                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
            # Use expanded seq_lens for per-token decode in target_verify/draft_extend.
            return self._forward_trtllm(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                metadata=metadata,
                sm_scale=layer.scaling,
                seq_lens=metadata.nsa_cache_seqlens_int32,
            )
        else:
            raise ValueError(f"Unsupported {nsa_impl = }")

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
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "NSA is causal only"

        # Do absorbed multi-latent attention
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
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
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        if envs.SGLANG_NSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        if self.nsa_decode_impl == "flashmla_sparse":
            if q_rope is not None:
                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.nsa_decode_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
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
        elif self.nsa_decode_impl == "tilelang":
            if q_rope is not None:
                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.nsa_decode_impl == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif self.nsa_decode_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        elif self.nsa_decode_impl == "trtllm":
            if q_rope is not None:
                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_trtllm(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                metadata=metadata,
                sm_scale=layer.scaling,
                seq_lens=metadata.cache_seqlens_int32,
            )

        else:
            assert False, f"Unsupported {self.nsa_decode_impl = }"

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
        metadata: NSAMetadata,
        page_table_1,
    ) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_with_kvcache

        cache_seqlens = metadata.nsa_cache_seqlens_int32

        # TODO the 2nd dim is seq_len_q, need to be >1 when MTP
        q_all = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        kv_cache = kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)
        assert self.real_page_size == 64, "only page size 64 is supported"

        if not self.nsa_kv_cache_store_fp8:
            # inefficiently quantize the whole cache
            kv_cache = quantize_k_cache(kv_cache)

        indices = page_table_1.unsqueeze(1)
        assert (
            indices.shape[-1] == self.nsa_index_topk
        )  # requirement of FlashMLA decode kernel

        o, _ = flash_mla_with_kvcache(
            q=q_all,
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
        return o

    def _forward_standard_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: NSAMetadata,
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
            )

        # Use FA3 for SM90 (Hopper/H200)
        fa_version = 3

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
            ver=fa_version,
        )

    def _forward_tilelang(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.tilelang_kernel import tilelang_sparse_fwd

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
        metadata: NSAMetadata,
        bs: int,
    ) -> torch.Tensor:
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        kv_indptr = self.kv_indptr

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)
        kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

        kv_indices = page_table_1[page_table_1 != -1]

        mla_decode_fwd(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            metadata.cu_seqlens_q,
            kv_indptr,
            kv_indices,
            metadata.cu_seqlens_q,
            metadata.max_seq_len_q,
            layer.scaling,
            layer.logit_cap,
        )
        # kv_cache = kv_cache.view(-1, 1, layer.head_dim)
        return o

    def _forward_trtllm(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        metadata: NSAMetadata,
        sm_scale: float,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using TRT-LLM sparse MLA kernel."""
        import flashinfer.decode

        batch_size = page_table_1.shape[0]
        _, num_heads, head_dim = q_all.shape

        q = q_all.view(batch_size, 1, num_heads, head_dim)
        kv = kv_cache.view(-1, 1, self.real_page_size, self.kv_cache_dim)
        block_tables = page_table_1.unsqueeze(1)

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
            sparse_mla_top_k=self.nsa_index_topk,
            bmm1_scale=sm_scale,
            backend="trtllm-gen",
        )
        # Output: [batch, q_len=1, heads, v_dim] -> [batch, heads, v_dim]
        return out.squeeze(1)

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

    def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
        """
        Decide all attention prefill dispatch strategies for this batch.
        """
        from sglang.srt.utils import get_device_sm, is_blackwell

        # Decide MHA vs MLA
        if forward_batch and forward_batch.forward_mode.is_extend_without_speculative():
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
                and max_kv_len <= self.nsa_index_topk  # Short enough for MHA
                and forward_batch.token_to_kv_pool.dtype
                in [torch.bfloat16, torch.float8_e4m3fn]
                and sum_seq_lens
                <= forward_batch.get_max_chunk_capacity()  # Fits in chunk
                and (not is_nsa_enable_prefill_cp())  # CP not enabled
            )
        else:
            self.use_mha = False  # Decode/verify always use MLA

        # Set MLA implementation only if not using MHA
        if not self.use_mha and self.enable_auto_select_prefill_impl:
            if self.nsa_kv_cache_store_fp8:
                if (
                    is_blackwell()
                    and forward_batch is not None
                    and forward_batch.forward_mode == ForwardMode.EXTEND
                ):
                    total_kv_tokens = forward_batch.seq_lens_sum
                    total_q_tokens = forward_batch.extend_num_tokens
                    # Heuristic based on benchmarking flashmla_kv vs flashmla_sparse + dequantize_k_cache_paged
                    if total_kv_tokens < total_q_tokens * 512:
                        self.nsa_prefill_impl = "flashmla_sparse"
                        return
                self.nsa_prefill_impl = "flashmla_kv"
            else:
                # bf16 kv cache
                self.nsa_prefill_impl = "flashmla_sparse"

    def get_topk_transform_method(self) -> TopkTransformMethod:
        """
        SGLANG_NSA_FUSE_TOPK controls whether to fuse the topk transform into the topk kernel.
        This method is used to select the topk transform method which can be fused or unfused.
        """
        if (
            # disable for MTP
            self.nsa_kv_cache_store_fp8
            and self.nsa_prefill_impl == "flashmla_sparse"
        ):
            topk_transform_method = TopkTransformMethod.RAGGED
        else:
            topk_transform_method = TopkTransformMethod.PAGED
        return topk_transform_method

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> NSAIndexerMetadata:
        return NSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            topk_transform_method=self.get_topk_transform_method(),
            paged_mqa_schedule_metadata=self.forward_metadata.paged_mqa_schedule_metadata,
        )

    def _compute_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        from sgl_kernel.flash_mla import get_mla_metadata

        flashmla_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            num_q_tokens_per_head_k=seq_len_q * self.num_q_heads // 1,
            num_heads_k=1,
            num_heads_q=self.num_q_heads,
            is_fp8_kvcache=True,
            topk=self.nsa_index_topk,
        )

        return NSAFlashMLAMetadata(
            flashmla_metadata=flashmla_metadata,
            num_splits=num_splits,
        )


class NativeSparseAttnMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                NativeSparseAttnBackend(
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
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        if envs.SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA.get():
            # Precompute metadata once (shared across all backends)
            precomputed = self.attn_backends[0]._precompute_replay_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

            # Fast copy to each backend (1-2x faster than computing N times)
            for i in range(self.speculative_num_steps):
                self.attn_backends[
                    i
                ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                    bs=bs,
                    precomputed=precomputed,
                    forward_mode=ForwardMode.DECODE,
                )
        else:
            # Fallback: compute metadata separately for each backend
            for i in range(self.speculative_num_steps):
                self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                    bs=bs,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    seq_lens_sum=forward_batch.seq_lens_sum,
                    encoder_lens=None,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    out_cache_loc=None,
                )
