"""No-sort B200 decode kernels for Inkling routed MoE LoRA.

The contract is BF16 CUDA, M<=32, top-k 6, rank 32, up to four slots, TP-local
intermediate 384 or 768, and non-EP expert IDs in ``[0, E)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ROUTER_TOPK = 6
RANK = 32

_BLOCK_M = 16
_GATE_BLOCK_N = 128
_DOWN_BLOCK_K = 128


@triton.jit
def _direct_gate_expand_kernel(
    shared_a_ptr,  # [M, 2R] or [S, M, 2R]
    gate_b_ptr,  # [S, E, 2I, R]
    topk_ids_ptr,  # [M, topk]
    token_lora_mapping_ptr,  # [M]
    output_ptr,  # [M, topk, 2I]
    stride_as,
    stride_am,
    stride_ar,
    stride_bs,
    stride_be,
    stride_bn,
    stride_br,
    stride_oq,
    stride_on,
    ROUTER_TOPK: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    GATE_WIDTH: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    route = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    token = route // ROUTER_TOPK
    slot = tl.load(token_lora_mapping_ptr + token).to(tl.int64)
    active = (slot >= 0) & (slot < NUM_SLOTS)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RANK)
    n_mask = offs_n < GATE_WIDTH

    # BLOCK_N=128 divides both supported I values, so no output tile straddles
    # the gate/up boundary. Gate tiles read A[0:R]; up tiles read A[R:2R].
    a_half = tl.where(pid_n * BLOCK_N >= INTERMEDIATE_SIZE, RANK, 0)
    a = tl.load(
        shared_a_ptr
        + slot * stride_as
        + token * stride_am
        + (a_half + offs_r[None, :]) * stride_ar,
        mask=active & (offs_m[:, None] == 0),
        other=0.0,
    )
    b = tl.load(
        gate_b_ptr
        + slot * stride_bs
        + expert * stride_be
        + offs_r[:, None] * stride_br
        + offs_n[None, :] * stride_bn,
        mask=active & n_mask[None, :],
        other=0.0,
    )
    accumulator = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(
        output_ptr
        + route * stride_oq
        + offs_m[:, None] * 0
        + offs_n[None, :] * stride_on,
        accumulator,
        mask=(offs_m[:, None] == 0) & n_mask[None, :],
    )


@triton.jit
def _direct_down_shrink_kernel(
    activation_ptr,  # [M*topk, I]
    down_a_ptr,  # [S, E, R, I]
    topk_ids_ptr,  # [M, topk]
    token_lora_mapping_ptr,  # [M]
    output_ptr,  # [M, topk, R]
    stride_xq,
    stride_xk,
    stride_as,
    stride_ae,
    stride_ar,
    stride_ak,
    stride_oq,
    stride_or,
    ROUTER_TOPK: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    token = route // ROUTER_TOPK
    slot = tl.load(token_lora_mapping_ptr + token).to(tl.int64)
    active = (slot >= 0) & (slot < NUM_SLOTS)
    offs_m = tl.arange(0, BLOCK_M)
    offs_r = tl.arange(0, RANK)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

    for base_k in tl.static_range(0, INTERMEDIATE_SIZE, BLOCK_K):
        k = base_k + offs_k
        k_mask = k < INTERMEDIATE_SIZE
        x = tl.load(
            activation_ptr + route * stride_xq + k[None, :] * stride_xk,
            mask=(offs_m[:, None] == 0) & k_mask[None, :],
            other=0.0,
        )
        a = tl.load(
            down_a_ptr
            + slot * stride_as
            + expert * stride_ae
            + offs_r[None, :] * stride_ar
            + k[:, None] * stride_ak,
            mask=active & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(x, a, out_dtype=tl.float32)

    tl.store(
        output_ptr
        + route * stride_oq
        + offs_m[:, None] * 0
        + offs_r[None, :] * stride_or,
        accumulator,
        mask=offs_m[:, None] == 0,
    )


def direct_decode_gate_expand(
    shared_intermediate: torch.Tensor,
    gate_b: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Expand shared gate/up rank vectors through directly selected experts.

    ``gate_b`` keeps the standard layout ``[S, E, 2I, 32]`` and
    ``output`` is ``[M, 6, 2I]``, where I is TP-local 384 or 768.
    Expert-id range is a runner invariant and is intentionally not read back
    from the GPU here.
    """

    num_tokens = topk_ids.shape[0]
    if shared_intermediate.ndim == 2:
        shared_slot_stride = 0
        shared_token_stride = shared_intermediate.stride(0)
        shared_rank_stride = shared_intermediate.stride(1)
    else:
        shared_slot_stride = shared_intermediate.stride(0)
        shared_token_stride = shared_intermediate.stride(1)
        shared_rank_stride = shared_intermediate.stride(2)
    gate_width = gate_b.shape[2]
    intermediate_size = gate_width // 2
    output_flat = output.view(num_tokens * ROUTER_TOPK, gate_width)
    num_routes = topk_ids.numel()
    _direct_gate_expand_kernel[(num_routes, triton.cdiv(gate_width, _GATE_BLOCK_N))](
        shared_intermediate,
        gate_b,
        topk_ids,
        token_lora_mapping,
        output_flat,
        shared_slot_stride,
        shared_token_stride,
        shared_rank_stride,
        gate_b.stride(0),
        gate_b.stride(1),
        gate_b.stride(2),
        gate_b.stride(3),
        output_flat.stride(0),
        output_flat.stride(1),
        ROUTER_TOPK=ROUTER_TOPK,
        INTERMEDIATE_SIZE=intermediate_size,
        RANK=RANK,
        GATE_WIDTH=gate_width,
        NUM_SLOTS=gate_b.shape[0],
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_GATE_BLOCK_N,
        num_warps=4,
        num_stages=1,
    )


def direct_decode_down_shrink(
    activation: torch.Tensor,
    down_a: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Shrink directly routed activations through per-expert LoRA-A.

    ``activation`` is flattened by route as ``[M*6, I]``; ``down_a`` keeps the
    standard layout ``[S, E, 32, I]``; and ``output`` is ``[M, 6, 32]``.
    """

    num_tokens = topk_ids.shape[0]
    intermediate_size = activation.shape[1]
    output_flat = output.view(num_tokens * ROUTER_TOPK, RANK)
    num_routes = topk_ids.numel()
    _direct_down_shrink_kernel[(num_routes,)](
        activation,
        down_a,
        topk_ids,
        token_lora_mapping,
        output_flat,
        activation.stride(0),
        activation.stride(1),
        down_a.stride(0),
        down_a.stride(1),
        down_a.stride(2),
        down_a.stride(3),
        output_flat.stride(0),
        output_flat.stride(1),
        ROUTER_TOPK=ROUTER_TOPK,
        INTERMEDIATE_SIZE=intermediate_size,
        RANK=RANK,
        NUM_SLOTS=down_a.shape[0],
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_DOWN_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
