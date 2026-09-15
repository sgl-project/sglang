# SPDX-License-Identifier: Apache-2.0
"""Hopper W4AFP8 DeepEP low-latency MoE with per-token-block scales."""

from typing import Optional

import torch


def cutedsl_w4afp8_moe_deepep_ll(
    a_states: torch.Tensor,
    a_scales: Optional[torch.Tensor],
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
    expected_m: Optional[int] = None,
) -> torch.Tensor:
    """Consume DeepEP FP8 [E,M,K] and scales [E,M,K/128] without requantizing.

    The signature matches the CUTLASS LL implementation for explicit backend
    selection. CUTLASS metadata and static activation scales are unused here:
    GEMM2 uses dynamic per-token-block quantization after SiLU(gate) * up.
    Weight scales are per output channel and K-group, interleaved by
    ``interleave_scales``. Routing weights are applied by DeepEP combine.

    Valid counts in masked_m must be in [0,M]. Invalid rows are zero in the
    BF16 output. The production wrapper requires SM90 and never falls back
    to the host-synchronizing Torch reference.
    """
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        silu_and_mul_masked_post_quant_fwd,
    )
    from sglang.srt.layers.moe.cutedsl_w4afp8_gemm_hopper import (
        hopper_w4afp8_gemm_per_token_block,
    )
    from sglang.srt.layers.moe.cutedsl_w4afp8_gemm_per_token_block import (
        deinterleave_w_scale,
    )

    if a_states.ndim != 3 or w1_q.ndim != 3 or w2_q.ndim != 3:
        raise ValueError("LL activations and weights must be rank-three tensors")
    e, rows, hidden = a_states.shape
    intermediate = w2_q.shape[2] * 2
    if hidden <= 0 or intermediate <= 0 or hidden % 128 or intermediate % 128:
        raise ValueError(
            "LL hidden and intermediate dimensions must be positive multiples of 128"
        )
    if w1_q.shape != (e, 2 * intermediate, hidden // 2) or w2_q.shape != (
        e,
        hidden,
        intermediate // 2,
    ):
        raise ValueError("LL gate/up and down weight shapes do not match activations")
    if (
        a_states.dtype != torch.float8_e4m3fn
        or w1_q.dtype != torch.int8
        or w2_q.dtype != torch.int8
    ):
        raise ValueError("LL requires FP8 activations and packed int8 weights")
    if (
        a_scales is None
        or a_scales.shape != (e, rows, hidden // 128)
        or not a_scales.is_floating_point()
    ):
        raise ValueError("LL activation scales must be floating point [E, M, K/128]")
    for scale, channels, contract in (
        (w1_scale, 2 * intermediate, hidden),
        (w2_scale, hidden, intermediate),
    ):
        blocks = contract // 128
        alignment = 4 if blocks % 4 == 0 else 1
        if (
            scale.shape != (e, blocks // alignment, channels * alignment)
            or not scale.is_floating_point()
        ):
            raise ValueError(
                "LL weight scales must use the interleaved [E, groups, channels] layout"
            )
    if masked_m.shape != (e,) or masked_m.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "LL masked_m must be an int32/int64 vector with one count per expert"
        )
    if any(
        t.device != a_states.device
        for t in (a_scales, w1_q, w2_q, w1_scale, w2_scale, masked_m)
    ):
        raise ValueError("LL tensors must be on the same device")
    if not a_states.is_cuda or torch.cuda.get_device_capability(a_states.device) != (
        9,
        0,
    ):
        raise ValueError("CuTe W4AFP8 LL requires an SM90 CUDA device")

    device = a_states.device
    c2 = torch.empty((e, rows, hidden), device=device, dtype=torch.bfloat16)
    if rows == 0 or e == 0:
        return c2
    c1 = torch.empty((e, rows, 2 * intermediate), device=device, dtype=torch.bfloat16)
    w1_logical = deinterleave_w_scale(w1_scale.float(), 2 * intermediate, hidden)
    hopper_w4afp8_gemm_per_token_block(
        a_states, a_scales, w1_q, w1_logical, c1, masked_m
    )
    intermediate_q = torch.empty(
        (e, rows, intermediate), device=device, dtype=torch.float8_e4m3fn
    )
    intermediate_scale = torch.empty(
        (e, rows, intermediate // 128), device=device, dtype=torch.float32
    )
    silu_and_mul_masked_post_quant_fwd(
        c1, intermediate_q, intermediate_scale, 128, masked_m
    )
    w2_logical = deinterleave_w_scale(w2_scale.float(), hidden, intermediate)
    hopper_w4afp8_gemm_per_token_block(
        intermediate_q, intermediate_scale, w2_q, w2_logical, c2, masked_m
    )
    return c2
