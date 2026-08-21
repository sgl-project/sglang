"""The activation stage of the masked BF16 pipeline. It adds the LoRA delta
before the activation. It also writes ``activation_lora_input``, which the
down-projection LoRA shrink reads.

The base gate/up GEMM writes ``[E_local, m_max, slices * inter]``. The row of
pair ``(t, k)`` is ``src2dst[t * top_k + k]``. The LoRA delta is contiguous
``[gate | up]`` and uses the pair index. This kernel joins the two index
spaces. It multiplies the activated gate by the up half only when
``NUM_SLICES`` is 2.

A pair with ``topk_ids[t, k] < 0`` is unrouted or padding. The kernel writes no
masked row for it. It writes zero to ``activation_lora_input``, so the shrink
reads exact zeros.
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
def _activation_delta_masked_kernel(
    gateup_ptr,  # [E_local * m_max, slices * inter] bf16
    delta_ptr,  # [num_tokens, top_k, slices * inter] [gate | up]
    act_out_ptr,  # [E_local * m_max, inter] bf16 (masked layout, flat)
    act_lora_in_ptr,  # [num_tokens, top_k, inter] bf16
    src2dst_ptr,  # [num_tokens * top_k] int32
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
    dst_row = tl.load(src2dst_ptr + pair_idx, mask=valid, other=0).to(tl.int64)

    total_inter = NUM_SLICES * inter
    gateup_row = gateup_ptr + dst_row * total_inter
    delta_row = delta_ptr + pair_idx * total_inter
    act_out_row = act_out_ptr + dst_row * inter
    lora_in_row = act_lora_in_ptr + pair_idx * inter

    vec = tl.arange(0, BLOCK_SIZE)
    if CONSUME_BASE_PDL:
        # The route and the addresses above do not depend on the gate/up GEMM.
        # The kernel waits here, just before it loads that GEMM's output.
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


def act_delta_masked(
    gateup_output: torch.Tensor,  # [E_local, m_max, slices * inter] bf16
    gate_up_delta: torch.Tensor | None,  # [num_tokens, top_k, slices * inter]
    act_out: torch.Tensor,  # [E_local, m_max, inter] bf16
    activation_lora_input: torch.Tensor,  # [num_tokens, top_k, inter] bf16
    src2dst: torch.Tensor,  # [num_tokens * top_k] int32
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    gate_first: bool = True,
    interleaved: bool = False,
    activation: str = "silu",
    consume_base_pdl: bool = False,
) -> None:
    ActivationFn.parse(activation)
    num_pairs = topk_ids.numel()
    inter = act_out.shape[-1]
    num_slices = gateup_output.shape[-1] // inter
    if num_slices not in (1, 2) or num_slices * inter != gateup_output.shape[-1]:
        raise ValueError(
            f"gate/up width {gateup_output.shape[-1]} is not 1x or 2x "
            f"intermediate {inter}"
        )
    if gate_up_delta is not None and gate_up_delta.shape != (
        *topk_ids.shape,
        num_slices * inter,
    ):
        raise ValueError(
            f"gate_up_delta must be {(*topk_ids.shape, num_slices * inter)}"
        )
    if activation_lora_input.shape != (*topk_ids.shape, inter):
        raise ValueError(f"activation_lora_input must be {(*topk_ids.shape, inter)}")

    _activation_delta_masked_kernel[(num_pairs,)](
        gateup_output.view(-1, num_slices * inter),
        gate_up_delta if gate_up_delta is not None else gateup_output,
        act_out.view(-1, inter),
        activation_lora_input,
        src2dst,
        topk_ids,
        topk_ids.shape[1],
        gateup_output.shape[0],
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


def act_delta_contiguous(
    gateup_output: torch.Tensor,  # [m_pad_ceiling, slices * inter] bf16
    gate_up_delta: torch.Tensor | None,  # [num_tokens, top_k, slices * inter]
    act_out: torch.Tensor,  # [m_pad_ceiling, inter] bf16
    activation_lora_input: torch.Tensor,  # [num_tokens, top_k, inter] bf16
    src2dst: torch.Tensor,  # [num_tokens * top_k] int32 COMPACT rows
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    num_local_experts: int,
    gate_first: bool = True,
    interleaved: bool = False,
    activation: str = "silu",
    consume_base_pdl: bool = False,
) -> None:
    """Run the masked activation kernel over the compact rows.

    The launch, the grid and the per-pair arithmetic match
    :func:`activation_delta.act_delta_masked`. Only the physical row behind
    each ``src2dst`` entry differs. The kernel writes a zero into
    ``activation_lora_input`` once for each invalid pair.

    ``num_local_experts`` is a parameter here because the compact buffer is 2-D
    and has no expert dimension to read it from.
    """
    ActivationFn.parse(activation)  # reject an unknown activation name
    num_pairs = topk_ids.numel()
    inter = act_out.shape[-1]
    if gateup_output.ndim != 2:
        raise ValueError(
            f"base gate/up must be compact 2-D, got {tuple(gateup_output.shape)}"
        )
    # The slice count comes from the weight shape, not from the activation.
    num_slices = gateup_output.shape[-1] // inter
    if num_slices not in (1, 2) or num_slices * inter != gateup_output.shape[-1]:
        raise ValueError(
            f"gate/up width {gateup_output.shape[-1]} is not 1x or 2x "
            f"intermediate {inter}"
        )
    if act_out.ndim != 2 or act_out.shape[0] != gateup_output.shape[0]:
        raise ValueError("gate/up and activation compact buffers must share rows")
    if gate_up_delta is not None and gate_up_delta.shape != (
        *topk_ids.shape,
        num_slices * inter,
    ):
        raise ValueError(
            f"gate_up_delta must be {(*topk_ids.shape, num_slices * inter)}"
        )
    if activation_lora_input.shape != (*topk_ids.shape, inter):
        raise ValueError(f"activation_lora_input must be {(*topk_ids.shape, inter)}")
    if num_pairs == 0:
        return
    _activation_delta_masked_kernel[(num_pairs,)](
        gateup_output,
        gate_up_delta if gate_up_delta is not None else gateup_output,
        act_out,
        activation_lora_input,
        src2dst,
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
