"""Add LoRA before activation and expose pair-major inputs for down-A.

pair_to_row addresses either base row layout; invalid pair outputs are zero.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.kernels.ops.lora.moe.fp8_quant import quantize_fp8_groups
from sglang.kernels.ops.lora.moe.quant_blockscaled import (
    BlockScaledRows,
    act_quant_args,
    div_rn_ieee,
    quantize_tile,
    store_quantized_rows,
)


@triton.jit
def apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "relu2":
        value = tl.maximum(x, 0.0)
        return value * value
    else:
        raise ValueError(f"unsupported activation {ACTIVATION_TYPE}")


@triton.jit
def _act_delta_kernel(
    gateup_ptr,  # [rows, slices * inter] bf16, flat base rows
    delta_ptr,  # [num_tokens, top_k, slices * inter] [gate | up]
    act_out_ptr,  # [rows, inter] bf16, flat base rows
    act_lora_in_ptr,  # [num_tokens, top_k, inter] bf16
    act_q_ptr,  # [rows, inter] fp8e4m3 (QUANT_GROUP > 0), or block-scaled codes
    act_s_ptr,  # [rows, inter // QUANT_GROUP] fp32 row-major, or scale atoms
    pair_to_row_ptr,  # [num_tokens * top_k] int32
    topk_ids_ptr,  # [num_tokens, top_k] (local ids; < 0 = invalid)
    num_local_experts,
    inter,
    stride_q_row,  # block-scaled codes only
    rows_per_slab,
    row_atoms,
    k_atoms,
    global_scale,
    row_scale_ptr,  # [rows] fp32 stored per-row scales (QUANT_BLOCKSCALED, PER_ROW)
    row_denominator,
    HAS_DELTA: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    QUANT_GROUP: tl.constexpr,  # 0 disables the additional FP8 output.
    WHOLE_ROW: tl.constexpr,  # Match upstream's whole-row rounding.
    QUANT_BLOCKSCALED: tl.constexpr,  # Block-scaled codes + scale atoms instead.
    VEC: tl.constexpr,
    ELEM_MAX: tl.constexpr,
    ELEM_MAX_LOG2: tl.constexpr,
    PACK_E2M1: tl.constexpr,
    NVFP4: tl.constexpr,
    PER_ROW: tl.constexpr,
    RECIPE_SELECT: tl.constexpr,
    ERR_MSE: tl.constexpr,
    ERR_FAST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_idx_int32 = tl.program_id(0)
    pair_idx = pair_idx_int32.to(tl.int64)

    expert_id = tl.load(topk_ids_ptr + pair_idx)
    valid = (expert_id >= 0) & (expert_id < num_local_experts)
    dst_row = tl.load(pair_to_row_ptr + pair_idx, mask=valid, other=0).to(tl.int64)

    total_inter = NUM_SLICES * inter
    gateup_row = gateup_ptr + dst_row * total_inter
    delta_row = delta_ptr + pair_idx * total_inter
    act_out_row = act_out_ptr + dst_row * inter
    lora_in_row = act_lora_in_ptr + pair_idx * inter

    vec = tl.arange(0, BLOCK_SIZE)
    row_amax = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in tl.range(0, inter, BLOCK_SIZE):
        offs = start + vec
        mask = offs < inter

        if NUM_SLICES == 1:
            gate_offs = offs
            up_offs = offs
        elif INTERLEAVED:
            gate_offs = 2 * offs
            up_offs = 2 * offs + 1
        else:
            gate_offs = offs
            up_offs = inter + offs
        if not GATE_FIRST:
            gate_offs, up_offs = up_offs, gate_offs

        g = tl.load(gateup_row + gate_offs, mask=mask & valid, other=0.0).to(tl.float32)
        if HAS_DELTA:
            # The delta is always contiguous [gate | up].
            dg = tl.load(delta_row + offs, mask=mask & valid, other=0.0).to(tl.float32)
            g += dg
        act = apply_activation(g, ACTIVATION_TYPE)
        if NUM_SLICES == 2:
            u = tl.load(gateup_row + up_offs, mask=mask & valid, other=0.0).to(
                tl.float32
            )
            if HAS_DELTA:
                du = tl.load(
                    delta_row + inter + offs,
                    mask=mask & valid,
                    other=0.0,
                ).to(tl.float32)
                u += du
            act = act * u
        act_bf16 = act.to(act_out_ptr.dtype.element_ty)

        tl.store(act_out_row + offs, act_bf16, mask=mask & valid)
        lora_in = tl.where(valid, act, 0.0)
        tl.store(
            lora_in_row + offs, lora_in.to(act_lora_in_ptr.dtype.element_ty), mask=mask
        )
        if QUANT_BLOCKSCALED and PER_ROW:
            # The row's amax over the BF16-rounded values it stores.
            row_amax = tl.maximum(
                row_amax, tl.abs(tl.where(mask & valid, act_bf16.to(tl.float32), 0.0))
            )
        elif QUANT_BLOCKSCALED:
            # Quantize the BF16-rounded value, matching the unfused path.
            codes, sf_bytes = quantize_tile(
                tl.where(mask & valid, act_bf16.to(tl.float32), 0.0)[None, :],
                global_scale,
                tl.zeros((1,), tl.float32),
                VEC,
                ELEM_MAX,
                ELEM_MAX_LOG2,
                PACK_E2M1,
                NVFP4,
                False,
            )
            store_quantized_rows(
                codes,
                sf_bytes,
                dst_row + tl.zeros((1,), tl.int64),
                (tl.arange(0, 1) == 0) & valid,
                start,
                inter,
                act_q_ptr,
                stride_q_row,
                act_s_ptr,
                rows_per_slab,
                row_atoms,
                k_atoms,
                VEC,
                PACK_E2M1,
            )
        elif QUANT_GROUP > 0:
            # Quantize the BF16-rounded value, matching the unfused path.
            # Admission ensures each quantization group is fully valid or masked.
            QG: tl.constexpr = QUANT_GROUP if QUANT_GROUP > 0 else BLOCK_SIZE
            grouped = tl.reshape(
                tl.where(mask & valid, act_bf16.to(tl.float32), 0.0),
                (BLOCK_SIZE // QG, QG),
            )
            q, scale = quantize_fp8_groups(grouped, WHOLE_ROW)
            tl.store(
                act_q_ptr + dst_row * inter + offs,
                tl.reshape(q, (BLOCK_SIZE,)).to(act_q_ptr.dtype.element_ty),
                mask=mask & valid,
            )
            gvec = tl.arange(0, BLOCK_SIZE // QG)
            goffs = start // QG + gvec
            if WHOLE_ROW:
                # The block spans the row, so its single group is the row's scale.
                n_groups = 1
            else:
                n_groups = inter // QG
            tl.store(
                act_s_ptr + dst_row * n_groups + goffs,
                scale,
                mask=(goffs < n_groups) & valid,
            )
    if QUANT_BLOCKSCALED and PER_ROW:
        # Second pass: the row's scale from the BF16 values it stored, then
        # those values re-read and quantized against it, so the standalone
        # row quantizer reproduces the codes bitwise.
        one = tl.zeros((1,), tl.float32)
        row_scale = div_rn_ieee(tl.max(row_amax, axis=0) + one, one + row_denominator)
        tl.store(
            row_scale_ptr + dst_row + tl.arange(0, 1),
            row_scale,
            mask=(tl.arange(0, 1) == 0) & valid,
        )
        tl.debug_barrier()
        for start in tl.range(0, inter, BLOCK_SIZE):
            offs = start + vec
            mask = offs < inter
            stored = tl.load(act_out_row + offs, mask=mask & valid, other=0.0)
            codes, sf_bytes = quantize_tile(
                stored.to(tl.float32)[None, :],
                global_scale,
                row_scale,
                VEC,
                ELEM_MAX,
                ELEM_MAX_LOG2,
                PACK_E2M1,
                NVFP4,
                PER_ROW,
                RECIPE_SELECT,
                ERR_MSE,
                ERR_FAST,
                row_denominator,
            )
            store_quantized_rows(
                codes,
                sf_bytes,
                dst_row + tl.zeros((1,), tl.int64),
                (tl.arange(0, 1) == 0) & valid,
                start,
                inter,
                act_q_ptr,
                stride_q_row,
                act_s_ptr,
                rows_per_slab,
                row_atoms,
                k_atoms,
                VEC,
                PACK_E2M1,
            )


def act_delta(
    gateup_rows: torch.Tensor,
    gate_up_delta: torch.Tensor | None,  # [num_tokens, top_k, slices * inter]
    act_out_rows: torch.Tensor,
    activation_lora_input: torch.Tensor,  # [num_tokens, top_k, inter] bf16
    pair_to_row: torch.Tensor,  # [num_tokens * top_k] int32
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    *,
    num_local_experts: int,
    gate_first: bool = True,
    interleaved: bool = False,
    activation: str = "silu",
    act_quant: tuple[torch.Tensor, torch.Tensor, int] | BlockScaledRows | None = None,
) -> None:
    """Add pair-major LoRA deltas and activate either provider row layout.

    ``act_quant`` also writes the activation as FP8 groups (fp8 rows, fp32
    scales, group) or as block-scaled rows for the down GEMM; a group equal to
    the intermediate size quantizes whole rows, one scale per row.
    """
    ActivationFn.parse(activation)
    num_pairs = topk_ids.numel()
    if num_pairs == 0:
        return
    gateup_rows = gateup_rows.view(-1, gateup_rows.shape[-1])
    act_out_rows = act_out_rows.view(-1, act_out_rows.shape[-1])
    inter = act_out_rows.shape[-1]
    num_slices = gateup_rows.shape[-1] // inter
    blockscaled = act_quant_args(None, fallback=act_out_rows)
    block_size, whole_row = 512, False
    if isinstance(act_quant, BlockScaledRows):
        assert act_quant.k == inter and inter % 128 == 0
        act_q, act_s, group = act_quant.data, act_quant.scales, 0
        blockscaled = act_quant_args(act_quant, fallback=act_out_rows)
    elif act_quant is not None:
        act_q, act_s, group = act_quant
        whole_row = group == inter
        if whole_row:
            # One block covers the row; the masked tail quantizes as zeros.
            block_size = max(block_size, triton.next_power_of_2(inter))
            group = block_size
        else:
            assert inter % group == 0 and block_size % group == 0
    else:
        act_q, act_s, group = act_out_rows, act_out_rows, 0
    _act_delta_kernel[(num_pairs,)](
        gateup_rows,
        gate_up_delta if gate_up_delta is not None else gateup_rows,
        act_out_rows,
        activation_lora_input,
        act_q,
        act_s,
        pair_to_row,
        topk_ids,
        num_local_experts,
        inter,
        HAS_DELTA=gate_up_delta is not None,
        NUM_SLICES=num_slices,
        ACTIVATION_TYPE=activation,
        GATE_FIRST=gate_first,
        INTERLEAVED=interleaved,
        QUANT_GROUP=group,
        WHOLE_ROW=whole_row,
        BLOCK_SIZE=block_size,
        **blockscaled,
    )
