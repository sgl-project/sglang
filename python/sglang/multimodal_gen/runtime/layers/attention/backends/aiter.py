# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Optional

import aiter
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.models.deepseek_common.utils import _use_aiter_gfx95

logger = logging.getLogger(__name__)

_use_fp8_attn = os.environ.get("SGLANG_DIFFUSION_AITER_FP8_ATTN", "0") == "1"
_fp8_dtype = torch.float8_e4m3fn

# ── MLA prefill ASM kernel constraints ──────────────────────────────
# The only available FP8 prefill kernel is the pre-compiled ASM binary
# mla_pfl_qh192_vh128_m32x8_n128x1_causal{0,1}.co, originally built for
# DeepSeek-style MLA.  Four hard constraints:
#
# 1. GPU arch must be gfx950 (MI350/MI355).  The ASM binary is compiled
#    exclusively for gfx950; it will crash or fail to load on other archs.
#
# 2. qk_head_dim baked at 192.  Models with smaller head dims (e.g.
#    Wan's 128) are handled by zero-padding Q/K — extra dims contribute
#    0 to dot products, preserving correctness.
#
# 3. v_head_dim baked at 128.  Models with V head dim != 128 cannot use
#    this kernel.
#
# 4. Kernel tiles over heads in groups of 8 ("m32x8" = 32 tokens x 8
#    heads per tile).  num_heads not divisible by 8 causes OOB reads.
#    E.g. Ulysses SP degree=4 with 40 heads -> 10 heads/rank -> crash.
_MLA_PREFILL_QK_HEAD_DIM = 192
_MLA_PREFILL_V_HEAD_DIM = 128
_MLA_PREFILL_HEAD_TILE = 8


if _use_fp8_attn:
    logger.info("DiT FP8 attention enabled via SGLANG_DIFFUSION_AITER_FP8_ATTN=1")


def _can_use_mla_prefill(v_head_dim: int, num_heads: int) -> bool:
    """Check if the MLA prefill ASM kernel supports the given shape and GPU."""
    return (
        _use_aiter_gfx95
        and v_head_dim == _MLA_PREFILL_V_HEAD_DIM
        and num_heads % _MLA_PREFILL_HEAD_TILE == 0
    )


class AITerBackend(AttentionBackend):
    """
    Backend for AITemplate attention implementation.
    """

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER

    @staticmethod
    def get_impl_cls() -> type["AITerImpl"]:
        return AITerImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        # AITer backend does not require special metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError("AITer backend does not have a metadata builder.")


def _build_mla_prefill_metadata(
    batch_size: int,
    seq_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    is_causal: bool,
    block_size: int = 1,
    tile_q: int = 256,
    tile_kv: int = 128,
    kv_seq_lens: Optional[torch.Tensor] = None,
) -> dict:
    """
    Build persistent-scheduling metadata required by mla_prefill_ps_asm_fwd.

    Args:
        batch_size: number of sequences in the batch.
        seq_lens: [batch_size] int tensor with per-sequence Q lengths (on CPU).
        num_heads: number of query heads.
        num_kv_heads: number of KV heads.
        is_causal: whether causal masking is used.
        block_size: KV page size (1 for non-paged token-level layout).
        tile_q: Q tile size used by the kernel.
        tile_kv: KV tile granularity.
        kv_seq_lens: [batch_size] int tensor with per-sequence KV lengths (on CPU).
            If None, defaults to seq_lens (self-attention).

    Returns:
        dict with all metadata tensors needed by the kernel + reduce.
    """
    if kv_seq_lens is None:
        kv_seq_lens = seq_lens

    device = "cuda"
    gqa_ratio = num_heads // num_kv_heads

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)

    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens, dim=0)
    actual_blocks = (kv_seq_lens + block_size - 1) // block_size
    kv_indptr[1 : batch_size + 1] = torch.cumsum(actual_blocks, dim=0)
    num_blocks = int(kv_indptr[-1])

    kv_indices = torch.arange(num_blocks, dtype=torch.int32)

    max_qlen = seq_lens.max()

    qhead_granularity = gqa_ratio
    qlen_granularity = tile_q // qhead_granularity
    kvlen_granularity = max(tile_kv, block_size)

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_size, work_info_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_kv_heads,
        max_qlen=max_qlen,
        qlen_granularity=qlen_granularity,
    )

    work_metadata_ptrs = torch.empty(
        work_meta_data_size, dtype=work_meta_data_type, device=device
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device=device
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device=device
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
    )

    aiter.get_ps_metadata_v1(
        qo_indptr.cpu(),
        kv_indptr.cpu(),
        seq_lens.cpu().int(),
        gqa_ratio,
        num_kv_heads,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        qhead_granularity=qhead_granularity,
        qlen_granularity=qlen_granularity,
        kvlen_granularity=kvlen_granularity,
        block_size=block_size,
        is_causal=is_causal,
    )

    return {
        "qo_indptr": qo_indptr.to(device),
        "kv_indptr": kv_indptr.to(device),
        "kv_indices": kv_indices.to(device),
        "work_indptr": work_indptr,
        "work_info": work_info,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "max_seqlen_q": max_qlen,
        "tile_q": tile_q,
    }


@torch.compiler.disable
def _mla_prefill_ps_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run mla_prefill_ps_asm_fwd + mla_reduce_v1 on 4D batch tensors.

    Reshapes [B, S, H, D] -> varlen [B*S, H, D] with trivial indptr,
    calls the kernel, then reshapes back.

    Supports cross-attention where q has seq_len S_q and k/v have seq_len S_kv.

    The ASM kernel has qk_head_dim=192 baked in at compile time (see module-level
    comments).  If the model's head dim is smaller (e.g. 128), Q and K are
    zero-padded along the last dimension to 192 before calling the kernel.
    The padded zeros contribute nothing to the QK dot product, so attention
    scores are identical to the unpadded case.
    """
    B, S_q, H, D_q = q.shape
    S_kv = k.shape[1]
    D_v = v.shape[-1]
    device = q.device
    num_kv_heads = k.shape[2]

    # Zero-pad Q/K head dim to match the kernel's compiled qk_head_dim=192.
    # Padding with zeros preserves dot-product correctness.
    pad_qk = _MLA_PREFILL_QK_HEAD_DIM - D_q
    if pad_qk > 0:
        q = torch.nn.functional.pad(q, (0, pad_qk))
        k = torch.nn.functional.pad(k, (0, pad_qk))
    D_q_kernel = q.shape[-1]

    q_varlen = q.reshape(B * S_q, H, D_q_kernel).contiguous()
    k_varlen = k.reshape(B * S_kv, num_kv_heads, D_q_kernel).contiguous()
    v_varlen = v.reshape(B * S_kv, num_kv_heads, D_v).contiguous()

    q_seq_lens = torch.full((B,), S_q, dtype=torch.int32)
    kv_seq_lens = torch.full((B,), S_kv, dtype=torch.int32)

    meta = _build_mla_prefill_metadata(
        batch_size=B,
        seq_lens=q_seq_lens,
        kv_seq_lens=kv_seq_lens,
        num_heads=H,
        num_kv_heads=num_kv_heads,
        is_causal=is_causal,
        block_size=1,
    )

    total_s = B * S_q
    tile_q = meta["tile_q"]

    output = torch.empty((total_s, H, D_v), dtype=torch.bfloat16, device=device)
    logits = torch.empty(
        (meta["reduce_partial_map"].size(0) * tile_q, H, D_v),
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(
        (meta["reduce_partial_map"].size(0) * tile_q, H),
        dtype=torch.float32,
        device=device,
    )
    final_lse = torch.empty((total_s, H), dtype=torch.float32, device=device)

    aiter.mla_prefill_ps_asm_fwd(
        q_varlen,
        k_varlen,
        v_varlen,
        meta["qo_indptr"],
        meta["kv_indptr"],
        meta["kv_indices"],
        meta["work_indptr"],
        meta["work_info"],
        meta["max_seqlen_q"],
        softmax_scale,
        is_causal,
        logits,
        attn_lse,
        output,
        q_scale,
        k_scale,
        v_scale,
    )

    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        meta["reduce_indptr"],
        meta["reduce_final_map"],
        meta["reduce_partial_map"],
        tile_q,
        output,
        final_lse,
    )

    return output.view(B, S_q, H, D_v)


class AITerImpl(AttentionImpl):
    """
    Implementation of attention using AITemplate.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError(
                "AITer backend does not support Grouped Query Attention yet."
            )
        self.causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale

    @torch.compiler.disable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs attention using one of:
          - _mla_prefill_ps_attention (FP8, SGLANG_DIFFUSION_AITER_FP8_ATTN=1)
          - flash_attn_func (BF16, default or FP8 fallback for unsupported shapes)

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        if _use_fp8_attn:
            if query.dtype != _fp8_dtype:
                q_fp8, q_scale = aiter.per_tensor_quant(query, quant_dtype=_fp8_dtype)
                k_fp8, k_scale = aiter.per_tensor_quant(key, quant_dtype=_fp8_dtype)
                v_fp8, v_scale = aiter.per_tensor_quant(value, quant_dtype=_fp8_dtype)
            else:
                q_fp8, k_fp8, v_fp8 = query, key, value
                one = torch.tensor(1.0, dtype=torch.float32, device=query.device)
                q_scale = k_scale = v_scale = one

            if _can_use_mla_prefill(v_fp8.shape[-1], q_fp8.shape[2]):
                return _mla_prefill_ps_attention(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    softmax_scale=self.softmax_scale,
                    is_causal=self.causal,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

            logger.warning_once(
                "FP8 MLA prefill kernel unsupported "
                "(need gfx950, v_head_dim=%d, num_heads divisible by %d; "
                "got v_head_dim=%d, num_heads=%d). Falling back to BF16.",
                _MLA_PREFILL_V_HEAD_DIM,
                _MLA_PREFILL_HEAD_TILE,
                v_fp8.shape[-1],
                q_fp8.shape[2],
            )

        # BF16 path
        output, _ = aiter.flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout_p,
            causal=self.causal,
            return_attn_probs=False,
            return_lse=True,
        )
        return output
