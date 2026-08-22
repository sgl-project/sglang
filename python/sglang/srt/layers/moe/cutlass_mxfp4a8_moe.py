# SPDX-License-Identifier: Apache-2.0
"""Cutlass MXFP4A8 MoE kernel.

This is the MXFP4 (weight E2M1 + block=32 E8M0 scale, activation FP8 e4m3)
counterpart of ``cutlass_w4a8_moe.py``. It reuses the exact same host-side
reorder / quantization / permutation plumbing and the same CUTLASS grouped-GEMM
data preparation as the int4a8 path; the only differences are:

  * the weight operand is MXFP4 (E2M1) instead of two's-complement int4, and
  * ``cutlass_mxfp4a8_moe_mm`` is called with ``chunk_size=32`` (E8M0 block) and
    the b_scale is the E8M0 scale pre-expanded to bf16 on the host.

The int4a8 entry (``cutlass_w4a8_moe.py``) is left completely untouched; both
formats are supported side by side.
"""

from typing import Optional

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_cuda_alike

_is_cuda = is_cuda()
_is_cuda_alike = is_cuda_alike()

if _is_cuda_alike:
    from sgl_kernel import (
        cutlass_mxfp4a8_moe_mm,
        get_cutlass_w4a8_moe_mm_data,
    )


from sglang.kernels.ops.moe.ep_moe_kernels import (
    cutlass_w4_run_moe_ep_preproess,
    deepep_ll_get_cutlass_w4a8_moe_mm_data,
    deepep_permute_triton_kernel,
    deepep_post_reorder_triton_kernel,
    deepep_run_moe_deep_preprocess,
    fp8_per_token_to_per_tensor_quant_triton,
    post_reorder_for_cutlass_moe,
    pre_reorder_for_cutlass_moe,
    silu_and_mul_masked_post_per_tensor_quant_fwd,
)
from sglang.srt.layers.mxfp4a8_utils import (
    build_grouped_act_block_scale,
)
from sglang.srt.layers.mxfp4a8_utils import (
    quantize_activation_mxfp8_native as quantize_activation_mxfp8_blockwise,
)
from sglang.srt.layers.mxfp4a8_utils import (
    silu_and_mul_mxfp8_quant_native,
)

# MXFP4 K-wise block size (E8M0 block). int4a8 uses 128; mxfp4a8 uses 32.
MXFP4_CHUNK_SIZE = 32


def _silu_mul_quant(c1, n, swiglu_limit):
    """Fused SwiGLU + silu_and_mul + mxfp8 quant of the GEMM1 output.

    Always uses the native hand-tuned CUDA kernel (7-8x faster than the Triton
    fused kernel). The native kernel has no built-in swiglu clamp, so when a
    ``swiglu_limit`` is set (DeepSeek-V4 uses 10.0) the clamp is applied first as
    two cheap in-place ops on ``c1`` (gate = min(gate, L); up = clamp(up, -L, L))
    -- this is numerically equivalent to the Triton fused clamp path (verified
    recon-exact) while keeping the 8x-faster native quant."""
    if swiglu_limit is not None:
        lim = float(swiglu_limit)
        c1[:, :n].clamp_(max=lim)
        c1[:, n:].clamp_(min=-lim, max=lim)
    return silu_and_mul_mxfp8_quant_native(c1, n, block_size=MXFP4_CHUNK_SIZE)


def cutlass_mxfp4a8_moe(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    routed_scaling_factor: float = 1.0,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """MXFP4A8 fused MoE. Mirrors ``cutlass_w4a8_moe`` but:

    - ``w1_q`` / ``w2_q`` are MXFP4 (E2M1) packed 4-bit weights (stored as int8),
      shapes ``[E, N*2, K//2]`` / ``[E, K, N//2]`` (transposed + packed).
    - ``w1_scale`` / ``w2_scale`` are the E8M0 block=32 scales pre-expanded to
      bf16 on the host (4-wide interleaved, same layout as the int4a8 bf16 scale).
    - the grouped GEMM is invoked with ``chunk_size=32``.
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"
    num_local_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids.size(1)

    if apply_router_weight_on_input:
        assert topk == 1, "apply_router_weight_on_input is only implemented for topk=1"

    device = a.device
    if get_parallel().moe_ep_size > 1:
        topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)

    src2dst = cutlass_w4_run_moe_ep_preproess(
        topk_ids,
    )

    # MXFP8 activation: reorder to bf16 first (identity scale), then per-token +
    # per-block (block=32) fp8 quant. The block scale rides the kernel's 4th TMA
    # and the epilogue alpha is 1.0 (activation scale applied inside the mainloop).
    gateup_input_bf16 = torch.empty(
        (m * topk, k),
        device=device,
        dtype=torch.bfloat16,
    )
    ones_scale = torch.ones(1, dtype=torch.float32, device=device)

    pre_reorder_for_cutlass_moe(
        a,
        gateup_input_bf16,
        src2dst,
        topk_ids,
        ones_scale,
        num_local_experts,
        topk,
        m,
        k,
    )

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    get_cutlass_w4a8_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_local_experts,
        n,
        k,
    )

    # Per-token + per-block fp8 quant of the reordered activation, then build the
    # per-expert-concatenated (even-padded) block-scale buffer + strides.
    gateup_input, a1_blk_scale = quantize_activation_mxfp8_blockwise(
        gateup_input_bf16, block_size=MXFP4_CHUNK_SIZE
    )
    a1_as_packed, a1_as_strides = build_grouped_act_block_scale(
        a1_blk_scale, expert_offsets, block_size=MXFP4_CHUNK_SIZE
    )

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)

    cutlass_mxfp4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        ones_scale,
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        MXFP4_CHUNK_SIZE,
        topk,
        a1_as_packed,
        a1_as_strides,
        MXFP4_CHUNK_SIZE,
    )

    # GEMM2 activation: fused SwiGLU (+clamp) + silu_and_mul + per-token/per-block
    # fp8 quant (native kernel when no clamp, Triton fallback for the clamp path).
    intermediate_q, a2_blk_scale = _silu_mul_quant(c1, n, swiglu_limit)
    a2_as_packed, a2_as_strides = build_grouped_act_block_scale(
        a2_blk_scale, expert_offsets, block_size=MXFP4_CHUNK_SIZE
    )

    cutlass_mxfp4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        ones_scale,
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
        MXFP4_CHUNK_SIZE,
        topk,
        a2_as_packed,
        a2_as_strides,
        MXFP4_CHUNK_SIZE,
    )

    output = torch.empty_like(a)

    post_reorder_for_cutlass_moe(
        c2,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        num_local_experts,
        topk,
        m,
        k,
        routed_scaling_factor,
    )
    return output


def cutlass_mxfp4a8_moe_deepep_normal(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids_: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """MXFP4A8 DeepEP-normal fused MoE. Mirrors ``cutlass_w4a8_moe_deepep_normal``
    with MXFP4 weights and ``chunk_size=32``."""
    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"
    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids_.size(1)
    device = a.device

    reorder_topk_ids, src2dst, _ = deepep_run_moe_deep_preprocess(
        topk_ids_, num_experts
    )
    num_total_tokens = reorder_topk_ids.numel()
    gateup_input_pre_reorder = torch.empty(
        (int(num_total_tokens), a.shape[1]),
        device=device,
        dtype=a.dtype,
    )
    deepep_permute_triton_kernel[(a.shape[0],)](
        a,
        gateup_input_pre_reorder,
        src2dst,
        topk_ids_.to(torch.int64),
        None,
        topk,
        a.shape[1],
        BLOCK_SIZE=512,
    )
    local_topk_ids = topk_ids_
    local_topk_ids = (
        torch.where(local_topk_ids == -1, num_experts, topk_ids_).to(torch.int32)
    ).contiguous()

    a_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
    get_cutlass_w4a8_moe_mm_data(
        local_topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    ones_scale = torch.ones(1, dtype=torch.float32, device=device)
    gateup_input, a1_blk_scale = quantize_activation_mxfp8_blockwise(
        gateup_input_pre_reorder, block_size=MXFP4_CHUNK_SIZE
    )
    del gateup_input_pre_reorder
    a1_as_packed, a1_as_strides = build_grouped_act_block_scale(
        a1_blk_scale, expert_offsets, block_size=MXFP4_CHUNK_SIZE
    )

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.zeros((m * topk, k), device=device, dtype=torch.bfloat16)

    cutlass_mxfp4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        ones_scale,
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        MXFP4_CHUNK_SIZE,
        topk,
        a1_as_packed,
        a1_as_strides,
        MXFP4_CHUNK_SIZE,
    )
    intermediate_q, a2_blk_scale = _silu_mul_quant(c1, n, swiglu_limit)
    a2_as_packed, a2_as_strides = build_grouped_act_block_scale(
        a2_blk_scale, expert_offsets, block_size=MXFP4_CHUNK_SIZE
    )

    cutlass_mxfp4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        ones_scale,
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
        MXFP4_CHUNK_SIZE,
        topk,
        a2_as_packed,
        a2_as_strides,
        MXFP4_CHUNK_SIZE,
    )
    num_tokens = src2dst.shape[0] // topk
    output = torch.empty(
        (num_tokens, c2.shape[1]),
        device=c2.device,
        dtype=torch.bfloat16,
    )
    deepep_post_reorder_triton_kernel[(num_tokens,)](
        c2,
        output,
        src2dst,
        topk_ids_,
        topk_weights,
        topk,
        c2.shape[1],
        BLOCK_SIZE=512,
    )

    return output


def cutlass_mxfp4a8_moe_deepep_ll(
    a_states: torch.Tensor,
    a_scales: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids_: torch.Tensor,
    masked_m: torch.Tensor,
    a_strides1: torch.Tensor,
    b_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides13: torch.Tensor,
    s_strides2: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """MXFP4A8 DeepEP-low-latency fused MoE. Mirrors
    ``cutlass_w4a8_moe_deepep_ll`` with MXFP4 weights and ``chunk_size=32``.

    NOTE (activation-quant granularity): unlike the main / deepep-normal entries
    (which do per-token + per-block(32) mxfp8 quant and feed the block-scale via
    the mainloop's 4th TMA), this low-latency decode path quantizes activations
    **per-tensor** (``fp8_per_token_to_per_tensor_quant_triton`` /
    ``silu_and_mul_masked_post_per_tensor_quant_fwd``) and applies the scalar
    scale through the epilogue ``alpha`` (``a1_scale``/``a2_scale``); no
    activation block-scale is passed. This is intentional: at decode M is tiny,
    so per-tensor scaling avoids the block-scale build/TMA overhead. The weight
    side stays per-block(32) mxfp4 in both paths, so only the activation scale
    granularity differs (per-tensor here vs per-block on the prefill path)."""
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a_states.shape[2] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"
    num_experts = w1_q.size(0)
    m = a_states.size(1)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids_.size(1)

    device = a_states.device

    problem_sizes1, problem_sizes2 = deepep_ll_get_cutlass_w4a8_moe_mm_data(
        masked_m,
        problem_sizes1,
        problem_sizes2,
        num_experts,
        n,
        k,
    )

    gateup_input = torch.empty(a_states.shape, dtype=torch.float8_e4m3fn, device=device)
    fp8_per_token_to_per_tensor_quant_triton(
        x=a_states,
        x_scale=a_scales,
        masked_m=masked_m,
        output_scale=a1_scale,
        output=gateup_input,
    )
    c1 = torch.empty((num_experts, m, n * 2), device=device, dtype=torch.bfloat16)
    c2 = torch.empty((num_experts, m, k), device=device, dtype=torch.bfloat16)

    cutlass_mxfp4a8_moe_mm(
        c1,
        gateup_input,
        w1_q,
        a1_scale.float(),
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides13,
        MXFP4_CHUNK_SIZE,
        topk,
    )

    intermediate_q = torch.empty(
        (num_experts, m, n), device=a_states.device, dtype=torch.float8_e4m3fn
    )
    if swiglu_limit is not None:
        # DeepSeek-V4 swiglu clamp: gate=min(gate, L), up=clamp(up, -L, L).
        lim = float(swiglu_limit)
        c1[..., :n].clamp_(max=lim)
        c1[..., n:].clamp_(min=-lim, max=lim)
    silu_and_mul_masked_post_per_tensor_quant_fwd(
        c1, intermediate_q, masked_m, a2_scale
    )
    cutlass_mxfp4a8_moe_mm(
        c2,
        intermediate_q,
        w2_q,
        a2_scale.float(),
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
        MXFP4_CHUNK_SIZE,
        topk,
    )

    return c2
