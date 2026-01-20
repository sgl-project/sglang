# SPDX-License-Identifier: Apache-2.0
# Adapted from turbo_layer.py for Attention Backend integration

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SparseLinearAttentionBackend(AttentionBackend):
    """Sparse Linear Attention Backend for efficient attention computation."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SLA_ATTN

    @staticmethod
    def get_impl_cls() -> type["SparseLinearAttentionImpl"]:
        return SparseLinearAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SparseLinearAttentionMetadata"]:
        return SparseLinearAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SparseLinearAttentionMetadataBuilder"]:
        return SparseLinearAttentionMetadataBuilder


@dataclass
class SparseLinearAttentionMetadata(AttentionMetadata):
    """Metadata for Sparse Linear Attention computation."""

    # Basic attention parameters
    current_timestep: int

    # Sparse attention configuration
    topk_ratio: float = 0.1


class SparseLinearAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builder for SparseLinearAttentionMetadata."""

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(
        self,
        current_timestep: int,
        topk_ratio: float = 0.1,
        **kwargs: dict[str, Any],
    ) -> SparseLinearAttentionMetadata:
        return SparseLinearAttentionMetadata(
            current_timestep=current_timestep,
            topk_ratio=topk_ratio,
        )


class SparseLinearAttentionImpl(AttentionImpl, nn.Module):
    """Implementation of sparse linear attention for the backend."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # SLA-specific parameters - matched to TurboDiffusion defaults
        topk_ratio: float = 0.1,  # TurboDiffusion uses topk=0.1
        feature_map: str = "softmax",
        BLKQ: int = 128,  # TurboDiffusion uses BLKQ=128
        BLKK: int = 64,  # TurboDiffusion uses BLKK=64
        use_bf16: bool = True,
        **extra_impl_args,
    ) -> None:
        nn.Module.__init__(self)

        # SLA-specific config
        self.topk_ratio = topk_ratio
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16

        # Learnable linear projection for combining sparse + linear attention
        self.proj_l = nn.Linear(head_size, head_size, dtype=torch.float32)

        # Feature map for linear attention
        # Type annotation for callables
        self.feature_map_q: Callable[[torch.Tensor], torch.Tensor]
        self.feature_map_k: Callable[[torch.Tensor], torch.Tensor]
        if feature_map == "elu":
            self.feature_map_q = lambda x: F.elu(x) + 1
            self.feature_map_k = lambda x: F.elu(x) + 1
        elif feature_map == "relu":
            self.feature_map_q = F.relu
            self.feature_map_k = F.relu
        elif feature_map == "softmax":
            self.feature_map_q = lambda x: F.softmax(x, dim=-1)
            self.feature_map_k = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights to zero for residual-like behavior."""
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)  # type: ignore[arg-type]

    def _calc_linear_attention_with_torch(self, q, k, v):
        kv = torch.matmul(k.transpose(-1, -2), v)
        k_sum = torch.sum(k, dim=-2, keepdim=True)
        return torch.matmul(q, kv) / (1e-5 + torch.matmul(q, k_sum.transpose(-1, -2)))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SparseLinearAttentionMetadata = None,
    ) -> torch.Tensor:
        """Forward pass for sparse linear attention.

        Args:
            query: query tensor of shape (B, H, L, D)
            key: key tensor of shape (B, H, L, D)
            value: value tensor of shape (B, H, L, D)
            attn_metadata: attention metadata containing configuration
        Returns:
            output tensor of shape (B, H, L, D)
        """
        dtype = query.dtype

        # Transpose for computation
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        # Get sparse attention map
        sparse_map, lut, real_topk = get_block_map(
            query, key, topk_ratio=self.topk_ratio, BLKQ=self.BLKQ, BLKK=self.BLKK
        )

        # Convert to computation dtype
        query = query.to(self.dtype)
        key = key.to(self.dtype)
        value = value.to(self.dtype)

        # Sparse attention computation
        o_s = _attention.apply(
            query, key, value, sparse_map, lut, real_topk, self.BLKQ, self.BLKK
        )

        # Apply feature maps
        query = self.feature_map_q(query).contiguous().to(self.dtype)  # c_q
        key = self.feature_map_k(key).contiguous().to(self.dtype)  # c_k
        # Linear attention computation
        o_l = self._calc_linear_attention_with_torch(query, key, value)

        # Apply projection and combine results
        with torch.amp.autocast("cuda", dtype=self.dtype):
            o_l = self.proj_l(o_l)

        # Combine sparse and linear attention
        output = (o_s + o_l).to(dtype).transpose(1, 2)

        return output


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()

        # We recommend the following two settings
        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64

        B, H, L, D = q.shape
        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(L, BLOCK_M)

        o_s = torch.empty_like(v)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        _attn_fwd[grid](
            q,
            k,
            v,
            qk_scale,
            topk,
            lut,
            lse,
            o_s,
            L,
            M_BLOCKS,
            D,
            BLOCK_M,
            BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=3,
        )

        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o_s


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    arg_k = k - torch.mean(
        k, dim=-2, keepdim=True
    )  # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


@triton.jit
def compress_kernel(
    X,
    XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(
        X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L
    )

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    LSE,
    OS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)

    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L)
