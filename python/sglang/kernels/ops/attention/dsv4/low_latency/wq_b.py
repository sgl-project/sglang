"""Low-latency MXFP8 wq_b GEMM fused with the q RoPE for DeepSeek-V4.1-Flash TP4 decode (SM100), and the same GEMM
for the indexer's wq_b, plain or with the indexer's fp4 query packing fused. See
``deepseek_v4/low_latency/wq_b_rope.cuh`` for the design."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

N_OUT = 8192  # 16 local heads x 512
K_IN = 1280
NUM_HEADS = 16
HEAD_DIM = 512
MAX_TOKENS = 128
INDEXER_N_OUT = 4096  # 32 index heads x 128
# Measured on B200 (rmsnorm_mxfp8 -> indexer queries): the fused fp4 epilogue (4 warps per CTA) beats indexer_wq_b +
# index_q_rope_pack_weights up to 32 tokens, loses from 64.
_FUSED_INDEX_Q_MAX_TOKENS = 32


@cache_once
def _n_tile(num_tokens: int) -> int:
    """MMA N: the smallest of 8 / 16 / 32 / 64 / 128 that holds every token."""
    for n in (8, 16, 32, 64, 128):
        if num_tokens <= n:
            return n
    raise ValueError(f"wq_b_rope supports up to {MAX_TOKENS} tokens, got {num_tokens}")


def _pipeline(n: int) -> str:
    """``kStages, kInflight``: the weight ring depth and the weight subtiles in flight per CTA."""
    return "5, 4" if n == 128 else "5, 5"


@cache_once
def _jit_module(n: int) -> Module:
    return load_jit(
        f"dsv41_wq_b_rope_ll_n{n}",
        cuda_files=["deepseek_v4/low_latency/wq_b_rope.cuh"],
        cuda_wrappers=[("run", f"dsv41_wq_b_rope_ll<{n}, {_pipeline(n)}>")],
        extra_cuda_cflags=["-O3"],
    )


def prepare_weight(weight: torch.Tensor, block_scale: torch.Tensor):
    """``(weight, weight_scale)`` as sglang's FlashInfer MXFP8 wq_b keeps them: the plain [8192, 1280] e4m3 weight,
    and its [256, 40] UE8M0 32x32 block scales expanded per row and interleaved by FlashInfer's
    ``block_scale_interleave`` (the layer's ``weight_scale_inv_swizzled``). For tests; the kernel reads exactly
    these tensors."""
    from flashinfer import block_scale_interleave

    assert weight.shape == (N_OUT, K_IN) and weight.dtype == torch.float8_e4m3fn
    assert block_scale.shape == (N_OUT // 32, K_IN // 32)
    per_row = block_scale.contiguous().view(torch.uint8).repeat_interleave(32, dim=0)
    return weight, block_scale_interleave(per_row.contiguous()).contiguous().view(-1)


def wq_b_rope(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``q = bf16(x @ W^T)`` as [T, 16, 512] with RoPE on the last 64 dims of every head, T <= 128: ``x`` [T, 1280]
    e4m3 with ``x_scale``, its UE8M0 per-32 scales in FlashInfer's 128x4 layout (``rmsnorm_mxfp8``'s output);
    ``weight`` [8192, 1280] e4m3 with ``weight_scale``, the 128x4 image of its per-row-per-32 scales; ``freqs``
    the [max_pos, 64] fp32 ``view_as_real(freqs_cis).flatten(1)`` table; ``positions`` [T] int32 / int64.
    The arithmetic of ``mm_mxfp8`` -> ``fused_rope_inplace(q[..., -64:], None, freqs_cis, positions)`` up to the
    fp32 order of the two K halves (split-K 2)."""
    num_tokens = x.shape[0]
    n = _n_tile(num_tokens)
    if out is None:
        out = x.new_empty((num_tokens, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    _jit_module(n).run(
        x,
        x_scale.view(-1),
        weight,
        weight_scale.view(-1),
        freqs,
        positions,
        out.view(num_tokens, N_OUT),
    )
    return out


@cache_once
def _jit_indexer_module(n: int) -> Module:
    return load_jit(
        f"dsv41_indexer_wq_b_ll_n{n}",
        cuda_files=["deepseek_v4/low_latency/wq_b_rope.cuh"],
        cuda_wrappers=[("run", f"dsv41_indexer_wq_b_ll<{n}, {_pipeline(n)}>")],
        extra_cuda_cflags=["-O3"],
    )


def indexer_wq_b(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The indexer's ``wq_b``: ``bf16(x @ W^T)`` as [T, 32, 128], T <= 128, W [4096, 1280] e4m3 with its 128x4
    scale image. ``x`` / ``x_scale`` are the q_lora ``rmsnorm_mxfp8`` output the main ``wq_b_rope`` takes, which
    is bitwise the indexer's own ``mxfp8_quantize`` of the normalized q_lora. ``mm_mxfp8`` up to the fp32 order of
    the two K halves."""
    num_tokens = x.shape[0]
    if out is None:
        out = x.new_empty((num_tokens, INDEXER_N_OUT // 128, 128), dtype=torch.bfloat16)
    _jit_indexer_module(_n_tile(num_tokens)).run(
        x,
        x_scale.view(-1),
        weight,
        weight_scale.view(-1),
        out.view(num_tokens, INDEXER_N_OUT),
    )
    return out


@cache_once
def _jit_indexer_q_module(n: int) -> Module:
    return load_jit(
        f"dsv41_indexer_q_fp4_ll_n{n}",
        cuda_files=["deepseek_v4/low_latency/wq_b_rope.cuh"],
        cuda_wrappers=[("run", f"dsv41_indexer_q_fp4_ll<{n}, {_pipeline(n)}>")],
        extra_cuda_cflags=["-O3"],
    )


def indexer_q_fp4(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    head_weights: torch.Tensor,
    weight_scale_mult: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The indexer's fp4 queries: ``indexer_wq_b`` then ``index_q_rope_pack_weights`` (RoPE on dims [64, 128) of
    every head at ``positions``, both fp4 stages, packed), in one launch up to 32 tokens (the bf16 q never leaves
    the SM; split-K 2), as the two launches above that. Returns ``(payload [T * 32, 64] int8, scale [T * 32]
    int32, weights [T, 32] fp32)``, bitwise ``index_q_rope_pack_weights(indexer_wq_b(x, ...), freqs,
    positions, head_weights, weight_scale_mult)``; ``freqs`` is the whole [max_pos, 64] fp32 table and
    ``head_weights`` the raw ``weights_proj`` output, written before this call."""
    num_tokens = x.shape[0]
    if num_tokens > _FUSED_INDEX_Q_MAX_TOKENS:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
            index_q_rope_pack_weights,
        )

        q = indexer_wq_b(x, x_scale, weight, weight_scale)
        return index_q_rope_pack_weights(
            q, freqs, positions, head_weights, weight_scale_mult
        )
    heads = INDEXER_N_OUT // 128
    payload = x.new_empty((num_tokens * heads, 64), dtype=torch.int8)
    scale = x.new_empty((num_tokens * heads,), dtype=torch.int32)
    weights = x.new_empty((num_tokens, heads), dtype=torch.float32)
    _jit_indexer_q_module(_n_tile(num_tokens)).run(
        x,
        x_scale.view(-1),
        weight,
        weight_scale.view(-1),
        freqs,
        positions,
        payload,
        scale,
        head_weights,
        weights,
        float(weight_scale_mult),
    )
    return payload, scale, weights
