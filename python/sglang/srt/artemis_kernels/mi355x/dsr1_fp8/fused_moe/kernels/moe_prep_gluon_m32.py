"""Exact M32 prep with compact-row-aligned route slots for deterministic W2.

This is the selected grid-512, two-band stride-2 M32 preparation schedule.
It preserves every legacy quantization, clear, and route-metadata store while
also writing ``route_slots[257, 32]``. Routed rows record slots 0 through 7,
the shared expert records slot 8, and unused compact rows record -1.
"""

from __future__ import annotations

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

M = gl.constexpr(32)
H = gl.constexpr(7168)
TOPK = gl.constexpr(9)
EXPERTS = gl.constexpr(257)
ROUTED_EXPERTS = gl.constexpr(256)
GROUP_K = gl.constexpr(128)
GROUPS_PER_TOKEN = gl.constexpr(56)
QUANT_GROUPS = gl.constexpr(4)
QUANT_CTAS = gl.constexpr(448)
FP8_MAX_RECIPROCAL = gl.constexpr(0.0022321429569274187)

@gluon.jit
def _compact_route_m32_exact_slots(
    topk_ids_ptr,
    topk_weights_ptr,
    route_tokens_ptr,
    route_weights_ptr,
    route_counts_ptr,
    route_slots_ptr,
    expert_id,
):
    route_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    rows = gl.arange(0, M, layout=route_layout)
    if expert_id == ROUTED_EXPERTS:
        shared_weights = gl.amd.cdna4.buffer_load(
            topk_weights_ptr,
            rows * TOPK + (TOPK - 1),
        )
        shared_base: gl.constexpr = ROUTED_EXPERTS * M
        gl.amd.cdna4.buffer_store(rows, route_tokens_ptr, shared_base + rows)
        gl.amd.cdna4.buffer_store(
            shared_weights,
            route_weights_ptr,
            shared_base + rows,
        )
        gl.amd.cdna4.buffer_store(
            gl.full((M,), TOPK - 1, gl.int32, layout=route_layout),
            route_slots_ptr,
            shared_base + rows,
        )
        gl.store(route_counts_ptr + ROUTED_EXPERTS, M)
    else:
        route_matrix_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[32, 2],
            warps_per_cta=[1, 1],
            order=[1, 0],
        )
        matrix_rows = gl.arange(
            0,
            M,
            layout=gl.SliceLayout(1, route_matrix_layout),
        )
        matrix_slots = gl.arange(
            0,
            TOPK - 1,
            layout=gl.SliceLayout(0, route_matrix_layout),
        )
        matrix_offsets = matrix_rows[:, None] * TOPK + matrix_slots[None, :]
        matrix_ids = gl.amd.cdna4.buffer_load(topk_ids_ptr, matrix_offsets)
        matrix_matches = matrix_ids == expert_id
        match = gl.convert_layout(
            gl.max(matrix_matches.to(gl.int32), axis=1).to(gl.int1),
            route_layout,
        )
        selected_slot = gl.convert_layout(
            gl.max(
                gl.where(matrix_matches, matrix_slots[None, :], 0),
                axis=1,
            ),
            route_layout,
        )
        selected_weight = gl.amd.cdna4.buffer_load(
            topk_weights_ptr,
            rows * TOPK + selected_slot,
            mask=match,
        )
        match_i32 = match.to(gl.int32)
        row_bits = match_i32.to(gl.uint32) << rows
        match_bits = gl.reduce_or(row_bits, axis=0)
        lower_mask = gl.full(
            (M,),
            0xFFFFFFFF,
            gl.uint32,
            layout=route_layout,
        ) >> (31 - rows)
        prefix_bits = match_bits & lower_mask
        positions = gl.inline_asm_elementwise(
            "v_bcnt_u32_b32 $0, $1, 0",
            "=v,v",
            [prefix_bits],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        ).to(gl.int32)
        route_base = expert_id * M
        gl.amd.cdna4.buffer_store(
            gl.full((M,), M, gl.int32, layout=route_layout),
            route_tokens_ptr,
            route_base + rows,
        )
        gl.amd.cdna4.buffer_store(
            gl.zeros((M,), gl.float32, layout=route_layout),
            route_weights_ptr,
            route_base + rows,
        )
        gl.amd.cdna4.buffer_store(
            gl.full((M,), -1, gl.int32, layout=route_layout),
            route_slots_ptr,
            route_base + rows,
        )
        destination = route_base + positions - 1
        gl.amd.cdna4.buffer_store(
            rows,
            route_tokens_ptr,
            destination,
            mask=match,
        )
        gl.amd.cdna4.buffer_store(
            selected_weight,
            route_weights_ptr,
            destination,
            mask=match,
        )
        gl.amd.cdna4.buffer_store(
            selected_slot,
            route_slots_ptr,
            destination,
            mask=match,
        )
        gl.store(route_counts_ptr + expert_id, gl.sum(match_i32, axis=0))


@gluon.jit
def _moe_prep_m32_exact_slots_kernel(
    hidden_ptr,
    quantized_ptr,
    scales_ptr,
    output_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    route_tokens_ptr,
    route_weights_ptr,
    route_counts_ptr,
    route_slots_ptr,
):
    program_id = gl.program_id(0)

    # Preserve the selected prep's reversed physical quant-tile issue order.
    quant_valid = program_id < QUANT_CTAS
    quant_program = QUANT_CTAS - 1 - program_id
    quant_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    quant_rows = gl.arange(
        0,
        QUANT_GROUPS,
        layout=gl.SliceLayout(1, quant_layout),
    )
    quant_cols = gl.arange(
        0,
        GROUP_K,
        layout=gl.SliceLayout(0, quant_layout),
    )
    global_group = quant_program * QUANT_GROUPS + quant_rows
    quant_offsets = global_group[:, None] * GROUP_K + quant_cols[None, :]
    quant_group_valid = quant_valid & (global_group < M * GROUPS_PER_TOKEN)
    quant_element_valid = quant_group_valid[:, None]
    values = gl.amd.cdna4.buffer_load(
        hidden_ptr,
        quant_offsets,
        mask=quant_element_valid,
    ).to(gl.float32)
    absmax = gl.maximum(
        gl.max(gl.abs(values), axis=1, keep_dims=True),
        1.0e-10,
    )
    group_scales = absmax * FP8_MAX_RECIPROCAL
    quantized = (values * (1.0 / group_scales)).to(quantized_ptr.type.element_ty)
    gl.amd.cdna4.buffer_store(
        quantized,
        quantized_ptr,
        quant_offsets,
        mask=quant_element_valid,
    )
    token = global_group // GROUPS_PER_TOKEN
    k_group = global_group - token * GROUPS_PER_TOKEN
    scale_offsets = k_group * M + token
    scale_values = gl.convert_layout(
        gl.reshape(group_scales, (QUANT_GROUPS,)),
        gl.SliceLayout(1, quant_layout),
    )
    gl.amd.cdna4.buffer_store(
        scale_values,
        scales_ptr,
        scale_offsets,
        mask=quant_group_valid,
    )
    gl.amd.cdna4.buffer_store(
        gl.zeros(
            (QUANT_GROUPS, GROUP_K),
            output_ptr.type.element_ty,
            layout=quant_layout,
        ),
        output_ptr,
        quant_offsets,
        mask=quant_element_valid,
    )

    routed = ((program_id & 1) == 0) & (program_id < 512)
    shared = program_id == 511
    if routed | shared:
        expert_id = gl.where(
            routed,
            (program_id // 2 + 128) & 255,
            ROUTED_EXPERTS,
        )
        _compact_route_m32_exact_slots(
            topk_ids_ptr,
            topk_weights_ptr,
            route_tokens_ptr,
            route_weights_ptr,
            route_counts_ptr,
            route_slots_ptr,
            expert_id,
        )


def moe_prep_gluon_m32(
    hidden: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    route_tokens: torch.Tensor,
    route_weights: torch.Tensor,
    route_counts: torch.Tensor,
    route_slots: torch.Tensor,
) -> None:
    """Run selected M32 prep and emit aligned compact-row route slots."""
    if hidden.shape != (int(M), int(H)) or hidden.dtype is not torch.bfloat16:
        raise ValueError("hidden must be contiguous BF16 [32,7168]")
    if quantized.shape != hidden.shape or quantized.dtype is not torch.float8_e4m3fn:
        raise ValueError("quantized must be contiguous FP8 E4M3FN [32,7168]")
    if scales.shape != (int(M), int(GROUPS_PER_TOKEN)) or scales.dtype is not torch.float32:
        raise ValueError("scales must be contiguous FP32 [32,56]")
    if output.shape != hidden.shape or output.dtype is not torch.bfloat16:
        raise ValueError("output must be contiguous BF16 [32,7168]")
    if topk_ids.shape != (int(M), int(TOPK)) or topk_ids.dtype is not torch.int32:
        raise ValueError("topk_ids must be contiguous INT32 [32,9]")
    if topk_weights.shape != (int(M), int(TOPK)) or topk_weights.dtype is not torch.float32:
        raise ValueError("topk_weights must be contiguous FP32 [32,9]")
    if route_tokens.shape != (int(EXPERTS), int(M)) or route_tokens.dtype is not torch.int32:
        raise ValueError("route_tokens must be contiguous INT32 [257,32]")
    if route_weights.shape != (int(EXPERTS), int(M)) or route_weights.dtype is not torch.float32:
        raise ValueError("route_weights must be contiguous FP32 [257,32]")
    if route_counts.shape != (int(EXPERTS),) or route_counts.dtype is not torch.int32:
        raise ValueError("route_counts must be contiguous INT32 [257]")
    if route_slots.shape != (int(EXPERTS), int(M)) or route_slots.dtype is not torch.int32:
        raise ValueError("route_slots must be contiguous INT32 [257,32]")
    tensors = (
        hidden,
        quantized,
        scales,
        output,
        topk_ids,
        topk_weights,
        route_tokens,
        route_weights,
        route_counts,
        route_slots,
    )
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous on GPU")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must be on one GPU")
    if route_slots.data_ptr() % 16:
        raise ValueError("route_slots must be at least 16-byte aligned")

    _moe_prep_m32_exact_slots_kernel[(512,)](
        hidden,
        quantized,
        scales,
        output,
        topk_ids,
        topk_weights,
        route_tokens,
        route_weights,
        route_counts,
        route_slots,
        num_warps=1,
        waves_per_eu=1,
    )


__all__ = ["moe_prep_gluon_m32"]
