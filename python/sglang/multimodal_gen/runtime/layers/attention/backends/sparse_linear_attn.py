"""
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

This implementation is adapted from: from https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/SLA/core.py and https://github.com/thu-ml/SLA/blob/main/SageSLA/core.py
Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

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


# ==================================SLA Functions===================================
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


def _get_cuda_arch(device_index: int) -> str:
    """Get CUDA architecture string for the given device."""
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


# ==================================SLA Class===================================
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


# ==================================SageSLA Class===================================
SAGESLA_ENABLED = True
try:
    import spas_sage_attn._fused as fused
    import spas_sage_attn._qattn as qattn
    from spas_sage_attn.utils import block_map_lut_triton, get_vanilla_qk_quant
except ImportError:
    SAGESLA_ENABLED = False

SAGE2PP_ENABLED = True
try:
    from spas_sage_attn._qattn import (
        qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold,
    )
except ImportError:
    SAGE2PP_ENABLED = False


class SageSparseLinearAttentionBackend(AttentionBackend):
    """Quantized Sparse-Linear Attention backend using SageAttention kernels."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_SLA_ATTN

    @staticmethod
    def get_impl_cls() -> type["SageSparseLinearAttentionImpl"]:
        return SageSparseLinearAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SageSparseLinearAttentionMetadata"]:
        return SageSparseLinearAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SageSparseLinearAttentionMetadataBuilder"]:
        return SageSparseLinearAttentionMetadataBuilder


@dataclass
class SageSparseLinearAttentionMetadata(AttentionMetadata):
    """Metadata for Sage Sparse Linear Attention computation."""

    # Basic attention parameters
    current_timestep: int

    # Sparse attention configuration
    topk_ratio: float = 0.1


class SageSparseLinearAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builder for SageSparseLinearAttentionMetadata."""

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(
        self,
        current_timestep: int,
        topk_ratio: float = 0.1,
        **kwargs: dict[str, Any],
    ) -> SageSparseLinearAttentionMetadata:
        return SageSparseLinearAttentionMetadata(
            current_timestep=current_timestep,
            topk_ratio=topk_ratio,
        )


class SageSparseLinearAttentionImpl(AttentionImpl, nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        topk_ratio: float = 0.5,
        feature_map: str = "softmax",
        use_bf16: bool = True,
        **extra_impl_args,
    ) -> None:
        nn.Module.__init__(self)

        assert (
            SAGESLA_ENABLED
        ), "Install spas_sage_attn(pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation) first to enable SageSLA."

        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale if softmax_scale else head_size**-0.5
        self.causal = causal
        self.prefix = prefix

        self.topk_ratio = topk_ratio
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

    def _calc_linear_attention_with_torch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        kv = torch.matmul(k.transpose(-1, -2), v)
        k_sum = torch.sum(k, dim=-2, keepdim=True)
        return torch.matmul(q, kv) / (1e-5 + torch.matmul(q, k_sum.transpose(-1, -2)))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass for Sage Sparse Linear attention with quantized kernels.
        Args:
            query: query tensor of shape (B, L, H, D)
            key: key tensor of shape (B, L, H, D)
            value: value tensor of shape (B, L, H, D)
            attn_metadata: attention metadata containing configuration
        Returns:
            output tensor of shape (B, L, H, D)
        """
        dtype = query.dtype

        # Transpose from (B, L, H, D) to SLA format (B, H, L, D)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Determine block sizes based on GPU architecture
        arch = _get_cuda_arch(q.device.index)

        if arch == "sm90":
            BLKQ = 64
            BLKK = 128
        else:
            BLKQ = 128
            BLKK = 64
        # Compute block-sparse attention pattern
        sparse_map, lut, real_topk = get_block_map(
            q, k, topk_ratio=self.topk_ratio, BLKQ=BLKQ, BLKK=BLKK
        )

        # Convert to compute dtype
        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        ########## SPARGE BEGIN ##########
        km = k.mean(dim=-2, keepdim=True)
        headdim = q.size(-1)
        assert headdim in [
            64,
            128,
        ], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

        # Quantize Q, K to INT8
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, BLKQ, BLKK)
        lut, valid_block_num = block_map_lut_triton(sparse_map)
        scale = 1.0 / (headdim**0.5)

        o_s = torch.empty_like(q)

        if arch in ("sm80", "sm86", "sm87"):
            pvthreshold = torch.full(
                (q.shape[-3],), 1e6, dtype=torch.float32, device=q.device
            )
            v_fp16 = v.to(torch.float16)
            qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
                q_int8,
                k_int8,
                v_fp16,
                o_s,
                lut,
                valid_block_num,
                pvthreshold,
                q_scale,
                k_scale,
                1,
                False,
                1,
                scale,
                0,
            )
        else:
            b, h_kv, kv_len, head_dim = v.shape
            padded_len = (kv_len + 127) // 128 * 128
            v_transposed_permutted = torch.empty(
                (b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device
            )
            fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
            v_fp8 = torch.empty(
                v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device
            )
            v_scale = torch.empty(
                (b, h_kv, head_dim), dtype=torch.float32, device=v.device
            )
            fused.scale_fuse_quant_cuda(
                v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1
            )

            if arch == "sm90":
                qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
                    q_int8,
                    k_int8,
                    v_fp8,
                    o_s,
                    lut,
                    valid_block_num,
                    q_scale,
                    k_scale,
                    v_scale,
                    1,
                    False,
                    1,
                    scale,
                )
            else:
                pvthreshold = torch.full(
                    (q.shape[-3],), 1e6, dtype=torch.float32, device=q.device
                )
                if SAGE2PP_ENABLED:
                    qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8,
                        k_int8,
                        v_fp8,
                        o_s,
                        lut,
                        valid_block_num,
                        pvthreshold,
                        q_scale,
                        k_scale,
                        v_scale,
                        1,
                        False,
                        1,
                        scale,
                        0,
                    )
                else:
                    qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8,
                        k_int8,
                        v_fp8,
                        o_s,
                        lut,
                        valid_block_num,
                        pvthreshold,
                        q_scale,
                        k_scale,
                        v_scale,
                        1,
                        False,
                        1,
                        scale,
                        0,
                    )

        ########## SPARGE END ##########

        # Linear attention with feature maps
        q_linear = self.feature_map_q(q).contiguous().to(self.dtype)
        k_linear = self.feature_map_k(k).contiguous().to(self.dtype)
        o_l = self._calc_linear_attention_with_torch(q_linear, k_linear, v)

        # Project linear attention output and combine
        with torch.amp.autocast("cuda", dtype=self.dtype):
            o_l = self.proj_l(o_l)

        # Combine sparse and linear outputs
        output = (o_s + o_l).to(dtype).transpose(1, 2)

        return output
