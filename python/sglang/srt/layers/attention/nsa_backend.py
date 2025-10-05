from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, TypeAlias

import torch

from sglang.srt.configs.model_config import get_nsa_index_topk, is_deepseek_nsa
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.nsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.nsa.utils import (
    NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
    NSA_FUSE_TOPK,
    compute_nsa_seqlens,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_hip

# from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

_is_hip = is_hip()

if _is_hip:
    try:
        from aiter import (
            flash_attn_varlen_func,
            mha_batch_prefill_func,
            paged_attention_ragged,
        )
        from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    except ImportError:
        print(
            "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
        )
else:
    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


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


@dataclass(frozen=True)
class NSAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: NSAMetadata

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.nsa_seqlens_expanded

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        from sgl_kernel import fast_topk_transform_fused, fast_topk_v2

        if not NSA_FUSE_TOPK:
            return fast_topk_v2(logits, self.get_seqlens_expanded(), topk)

        # NOTE(dark): if fused, we return a transformed page table directly
        return fast_topk_transform_fused(
            score=logits,
            lengths=self.get_seqlens_expanded(),
            page_table_size_1=self.attn_metadata.page_table_1,
            cu_seqlens_q=self.attn_metadata.cu_seqlens_q,
            topk=topk,
        )


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    assert seqlens.dtype == torch.int32 and seqlens.is_cuda
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


_NSA_IMPL_T: TypeAlias = Literal[
    "flashmla_prefill", "flashmla_decode", "fa3", "tilelang"
]

NSA_PREFILL_IMPL: _NSA_IMPL_T
NSA_DECODE_IMPL: _NSA_IMPL_T


class NativeSparseAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
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

        assert model_runner.req_to_token_pool is not None
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        global NSA_PREFILL_IMPL, NSA_DECODE_IMPL
        NSA_PREFILL_IMPL = model_runner.server_args.nsa_prefill
        NSA_DECODE_IMPL = model_runner.server_args.nsa_decode

        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)

        if _is_hip:
            max_bs = model_runner.req_to_token_pool.size

            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

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

        assert (
            forward_batch.spec_info is None
        ), "Spec decoding is not supported for NSA backend now"
        cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        assert forward_batch.seq_lens_cpu is not None
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            cu_seqlens_q = self.get_device_int32_arange(batch_size + 1)
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None
            if any(forward_batch.extend_prefix_lens_cpu):
                max_seqlen_q = max(extend_seq_lens_cpu)
                cu_seqlens_q = compute_cu_seqlens(
                    forward_batch.extend_seq_lens.to(torch.int32)
                )
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k
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
        else:
            assert False, f"Unsupported {forward_batch.forward_mode = }"

        # 1D, expanded seqlens (1D means cheap to compute, so always compute it)
        nsa_cache_seqlens_int32 = compute_nsa_seqlens(
            original_seq_lens=seqlens_expanded,
            nsa_index_topk=self.nsa_index_topk,
        )
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))

        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table,
            flashmla_metadata=(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens_int32,
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
                if NSA_DECODE_IMPL == "flashmla_decode"
                else None
            ),
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            nsa_extend_seq_lens_list=extend_seq_lens_cpu,
            real_page_table=self._transform_table_1_to_real(page_table),
        )

        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.decode_cuda_graph_metadata: Dict = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            # fake page_table for sparse_prefill
            "page_table": torch.zeros(
                max_bs,
                self.max_context_len,
                dtype=torch.int32,
                device=self.device,
            ),
            "flashmla_metadata": (
                self._compute_flashmla_metadata(
                    cache_seqlens=torch.ones(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
                if NSA_DECODE_IMPL == "flashmla_decode"
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
        """Initialize forward metadata for capturing CUDA graph."""
        assert forward_mode.is_decode_or_idle(), "Only support decode for now"
        assert (
            spec_info is None
        ), "Speculative decoding is not supported for NSA backend now"

        # Normal Decode
        # Get sequence information
        cache_seqlens_int32 = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)

        # Use max context length for seq_len_k
        page_table_1 = self.decode_cuda_graph_metadata["page_table"][:bs, :]
        max_seq_len_k = page_table_1.shape[1]

        # Precompute page table
        # Precompute cumulative sequence lengths

        # NOTE(dark): this is always arange, since we are decoding
        cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][: bs + 1]
        nsa_cache_seqlens_int32 = compute_nsa_seqlens(
            cache_seqlens_int32, nsa_index_topk=self.nsa_index_topk
        )
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))
        real_page_table = self._transform_table_1_to_real(page_table_1)

        if NSA_DECODE_IMPL == "flashmla_decode":
            flashmla_metadata = self.decode_cuda_graph_metadata[
                "flashmla_metadata"
            ].slice(slice(0, bs + 1))
            flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens_int32,
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
            )
        else:
            flashmla_metadata = None

        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=1,
            max_seq_len_k=max_seq_len_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            flashmla_metadata=flashmla_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=cache_seqlens_int32,
            real_page_table=real_page_table,
            nsa_extend_seq_lens_list=[1] * bs,
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
        assert forward_mode.is_decode_or_idle(), "Only support decode for now"
        assert (
            spec_info is None
        ), "Speculative decoding is not supported for NSA backend now"
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Normal Decode
        metadata: NSAMetadata = self.decode_cuda_graph_metadata[bs]
        max_len = int(seq_lens_cpu.max().item())

        cache_seqlens = seq_lens.to(torch.int32)
        metadata.cache_seqlens_int32.copy_(cache_seqlens)
        metadata.cu_seqlens_k[1:].copy_(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
        )
        page_indices = self.req_to_token[req_pool_indices, :max_len]
        metadata.page_table_1[:, :max_len].copy_(page_indices)
        assert (
            metadata.nsa_cache_seqlens_int32 is not None
            and metadata.nsa_cu_seqlens_k is not None
            and self.nsa_index_topk is not None
        )
        nsa_cache_seqlens = compute_nsa_seqlens(cache_seqlens, self.nsa_index_topk)
        metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
        metadata.nsa_cu_seqlens_k[1:].copy_(
            torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32)
        )
        # NOTE(dark): (nsa-) cu_seqlens_q is always arange, no need to copy

        assert self.real_page_size == metadata.page_size
        if self.real_page_size > 1:
            real_table = self._transform_table_1_to_real(page_indices)
            new_len = real_table.shape[1]
            metadata.real_page_table[:, :new_len].copy_(real_table)
        else:
            assert metadata.real_page_table is metadata.page_table_1

        if NSA_DECODE_IMPL == "flashmla_decode":
            metadata.flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens,
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
            )

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
        assert (
            not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        ), "NSA backend doesn't support speculative decoding"
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

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}

        # Do absorbed multi-latent attention
        assert q_rope is not None
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        # when store in fp8 and compute in fp8, no need to convert dtype
        if not (
            NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 and self.nsa_kv_cache_store_fp8
        ):
            kv_cache = kv_cache.to(q.dtype)

        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        # NOTE(dark): here, we use page size = 1

        if NSA_FUSE_TOPK:
            page_table_1 = topk_indices
        else:
            assert metadata.nsa_extend_seq_lens_list is not None
            page_table_1 = transform_index_page_table_prefill(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                page_size=1,
            )
        if NSA_PREFILL_IMPL == "tilelang":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif NSA_PREFILL_IMPL == "flashmla_prefill":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_prefill(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif NSA_PREFILL_IMPL == "flashmla_decode":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_decode(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif NSA_PREFILL_IMPL == "fa3":
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
        else:
            raise ValueError(f"Unsupported {NSA_PREFILL_IMPL = }")

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

        if NSA_FUSE_TOPK:
            page_table_1 = topk_indices
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        if NSA_DECODE_IMPL == "flashmla_prefill":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_prefill(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif NSA_DECODE_IMPL == "flashmla_decode":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_decode(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif NSA_DECODE_IMPL == "tilelang":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif NSA_DECODE_IMPL == "fa3":
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
        elif NSA_DECODE_IMPL == "aiter":
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

        else:
            assert False, f"Unsupported {NSA_DECODE_IMPL = }"

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

    def _forward_flashmla_prefill(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from flash_mla import flash_mla_sparse_fwd

        o, _, _ = flash_mla_sparse_fwd(
            q=q_all,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )
        return o

    def _forward_flashmla_decode(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        sm_scale: float,
        layer,
        metadata: NSAMetadata,
        page_table_1,
    ) -> torch.Tensor:
        from flash_mla import flash_mla_with_kvcache

        cache_seqlens = metadata.nsa_cache_seqlens_int32

        # TODO the 2nd dim is seq_len_q, need to be >1 when MTP
        q_all = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        kv_cache = kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)
        assert self.real_page_size == 64, "only page size 64 is supported"

        if NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 and not self.nsa_kv_cache_store_fp8:
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
            is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
        )
        return o

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

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> NSAIndexerMetadata:
        return NSAIndexerMetadata(attn_metadata=self.forward_metadata)

    def _compute_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        from flash_mla import get_mla_metadata

        flashmla_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            num_q_tokens_per_head_k=seq_len_q * self.num_q_heads // 1,
            num_heads_k=1,
            num_heads_q=self.num_q_heads,
            is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
            topk=self.nsa_index_topk,
        )

        return NSAFlashMLAMetadata(
            flashmla_metadata=flashmla_metadata,
            num_splits=num_splits,
        )
