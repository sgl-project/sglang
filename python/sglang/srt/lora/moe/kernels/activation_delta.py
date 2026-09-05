"""Add LoRA before activation and expose pair-major inputs for down-A.

pair_to_row addresses either base row layout; invalid pair outputs are zero.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.activation import ActivationFn


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
    pair_to_row_ptr,  # [num_tokens * top_k] int32
    topk_ids_ptr,  # [num_tokens, top_k] (local ids; < 0 = invalid)
    top_k,
    num_local_experts,
    inter,
    HAS_DELTA: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONSUME_BASE_PDL: tl.constexpr,
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
    if CONSUME_BASE_PDL:
        # Wait just before the first read of the gate/up GEMM's output.
        tl.extra.cuda.gdc_wait()
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


def _launch_act_delta(
    gateup_rows: torch.Tensor,  # [rows, slices * inter] bf16, flat
    gate_up_delta: torch.Tensor | None,  # [num_tokens, top_k, slices * inter]
    act_out_rows: torch.Tensor,  # [rows, inter] bf16, flat
    activation_lora_input: torch.Tensor,  # [num_tokens, top_k, inter] bf16
    pair_to_row: torch.Tensor,  # [num_tokens * top_k] int32
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    *,
    num_local_experts: int,
    gate_first: bool,
    interleaved: bool,
    activation: str,
    consume_base_pdl: bool,
) -> None:
    ActivationFn.parse(activation)
    num_pairs = topk_ids.numel()
    if num_pairs == 0:
        return
    inter = act_out_rows.shape[-1]
    num_slices = gateup_rows.shape[-1] // inter
    _act_delta_kernel[(num_pairs,)](
        gateup_rows,
        gate_up_delta if gate_up_delta is not None else gateup_rows,
        act_out_rows,
        activation_lora_input,
        pair_to_row,
        topk_ids,
        topk_ids.shape[1],
        num_local_experts,
        inter,
        HAS_DELTA=gate_up_delta is not None,
        NUM_SLICES=num_slices,
        ACTIVATION_TYPE=activation,
        GATE_FIRST=gate_first,
        INTERLEAVED=interleaved,
        CONSUME_BASE_PDL=consume_base_pdl,
        BLOCK_SIZE=512,
        **({"launch_pdl": True} if consume_base_pdl else {}),
    )


def act_delta_masked(
    gateup_output: torch.Tensor,  # [E_local, m_max, slices * inter] bf16
    gate_up_delta: torch.Tensor | None,  # [num_tokens, top_k, slices * inter]
    act_out: torch.Tensor,  # [E_local, m_max, inter] bf16
    activation_lora_input: torch.Tensor,  # [num_tokens, top_k, inter] bf16
    pair_to_row: torch.Tensor,  # [num_tokens * top_k] int32 slab rows
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    gate_first: bool = True,
    interleaved: bool = False,
    activation: str = "silu",
    consume_base_pdl: bool = False,
) -> None:
    _launch_act_delta(
        gateup_output.view(-1, gateup_output.shape[-1]),
        gate_up_delta,
        act_out.view(-1, act_out.shape[-1]),
        activation_lora_input,
        pair_to_row,
        topk_ids,
        num_local_experts=gateup_output.shape[0],
        gate_first=gate_first,
        interleaved=interleaved,
        activation=activation,
        consume_base_pdl=consume_base_pdl,
    )


def act_delta_contiguous(
    gateup_output: torch.Tensor,  # [m_pad_ceiling, slices * inter] bf16
    gate_up_delta: torch.Tensor | None,  # [num_tokens, top_k, slices * inter]
    act_out: torch.Tensor,  # [m_pad_ceiling, inter] bf16
    activation_lora_input: torch.Tensor,  # [num_tokens, top_k, inter] bf16
    pair_to_row: torch.Tensor,  # [num_tokens * top_k] int32 COMPACT rows
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    num_local_experts: int,
    gate_first: bool = True,
    interleaved: bool = False,
    activation: str = "silu",
    consume_base_pdl: bool = False,
) -> None:
    _launch_act_delta(
        gateup_output,
        gate_up_delta,
        act_out,
        activation_lora_input,
        pair_to_row,
        topk_ids,
        num_local_experts=num_local_experts,
        gate_first=gate_first,
        interleaved=interleaved,
        activation=activation,
        consume_base_pdl=consume_base_pdl,
    )
