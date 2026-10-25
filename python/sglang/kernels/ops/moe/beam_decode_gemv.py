# SPDX-License-Identifier: Apache-2.0
"""SM103 single-token Beam MoE with weighted expert outputs.

Gate/up and activation retain BF16 rounding boundaries. The down projection
multiplies FP32 routing coefficients before rounding to BF16. The caller owns
expert reduction, routed scaling and TP communication. Vector reductions use
a different FP32 summation order from the tensor-core reference.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.activation._triton_silu import _silu_fast_math


@triton.jit
def _gate_up_gemv_kernel(
    HiddenStates,
    Weight,
    TopkIds,
    Intermediate,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    expert = tl.load(TopkIds + route).to(tl.int64)
    accumulator = tl.full((BLOCK_N, BLOCK_K), 0, tl.float32)
    for start in range(tl.cdiv(HIDDEN_SIZE, BLOCK_K)):
        kk = start * BLOCK_K + k
        w = tl.load(
            Weight
            + expert * (2 * INTERMEDIATE_SIZE * HIDDEN_SIZE)
            + n[:, None] * HIDDEN_SIZE
            + kk[None, :],
            (n[:, None] < 2 * INTERMEDIATE_SIZE) & (kk[None, :] < HIDDEN_SIZE),
            0,
        ).to(tl.float32)
        x = tl.load(HiddenStates + kk, kk < HIDDEN_SIZE, 0).to(tl.float32)
        accumulator = accumulator + w * x[None, :]
    y = tl.sum(accumulator, 1).to(tl.bfloat16).to(tl.float32)
    gate, up = tl.split(tl.reshape(y, (BLOCK_N // 2, 2)))
    silu = _silu_fast_math(gate)
    column = tl.program_id(1) * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    tl.store(
        Intermediate + route * INTERMEDIATE_SIZE + column,
        silu * up,
        column < INTERMEDIATE_SIZE,
    )


@triton.jit
def _down_gemv_kernel(
    Intermediate,
    Weight,
    TopkIds,
    TopkWeights,
    Output,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    expert = tl.load(TopkIds + route).to(tl.int64)
    w = tl.load(
        Weight
        + expert * (HIDDEN_SIZE * INTERMEDIATE_SIZE)
        + n[:, None] * INTERMEDIATE_SIZE
        + k[None, :],
        (n[:, None] < HIDDEN_SIZE) & (k[None, :] < INTERMEDIATE_SIZE),
        0,
    ).to(tl.float32)
    a = tl.load(
        Intermediate + route * INTERMEDIATE_SIZE + k, k < INTERMEDIATE_SIZE, 0
    ).to(tl.float32)
    y = tl.sum(w * a[None, :], 1) * tl.load(TopkWeights + route)
    tl.store(Output + route * HIDDEN_SIZE + n, y, n < HIDDEN_SIZE)


def beam_decode_gemv(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Return weighted BF16 expert outputs for Beam's single-token TP4 path.

    Weights use the interleaved gate/up layout. Expert reduction, routed scaling
    and TP communication remain with the caller.
    """
    assert hidden_states.shape == (1, 5120)
    assert w13_weight.shape == (256, 1280, 5120)
    assert w2_weight.shape == (256, 5120, 640)
    assert topk_ids.shape == topk_weights.shape == (1, 8)
    assert hidden_states.dtype == w13_weight.dtype == w2_weight.dtype == torch.bfloat16
    assert topk_weights.dtype == torch.float32
    assert topk_ids.dtype in (torch.int32, torch.int64)
    assert hidden_states.is_cuda
    assert all(
        t.device == hidden_states.device and t.is_contiguous()
        for t in (hidden_states, w13_weight, w2_weight, topk_weights, topk_ids)
    )

    hidden_size = hidden_states.shape[1]
    top_k = topk_ids.shape[1]
    intermediate_size = w2_weight.shape[2]
    intermediate = torch.empty(
        (top_k, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output = torch.empty(
        (1, top_k, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    _gate_up_gemv_kernel[(top_k, triton.cdiv(2 * intermediate_size, 8))](
        hidden_states,
        w13_weight,
        topk_ids,
        intermediate,
        hidden_size,
        intermediate_size,
        8,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _down_gemv_kernel[(top_k, triton.cdiv(hidden_size, 8))](
        intermediate,
        w2_weight,
        topk_ids,
        topk_weights,
        output,
        hidden_size,
        intermediate_size,
        8,
        triton.next_power_of_2(intermediate_size),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
