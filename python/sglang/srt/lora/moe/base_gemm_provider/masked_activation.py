"""S3 activation-join kernel for the MoE LoRA BF16 pipeline.

SwiGLU or ReLU2 with the LoRA delta added PRE-activation, plus the
``activation_lora_input`` side output consumed by the down-proj LoRA shrink.

Layout: the base GEMM1 output is the provider *masked* layout
``[E_local, m_max, slices * inter]`` viewed flat by the kernel;
``src2dst[t * top_k + k]`` is the flat row for expanded pair (t, k)
(``expert * m_max + offset``, produced by ``moe_ep_deepgemm_preprocess``).
The LoRA delta is canonical contiguous ``[gate | up]`` indexed by the EXPANDED
(t, k) — the two index spaces meet here, exactly like the trtllm
``fused_activation_quant`` kernel.

Invalid pairs (``topk_ids[t, k] < 0``: EP-unrouted / padding) mirror
``post_reorder_deepgemm``: skip the store into the masked layout and zero
``activation_lora_input`` so the down-LoRA shrink sees exact zeros.

Layout flags (design §3): ``GATE_FIRST`` / ``INTERLEAVED`` are compile-time
specializations; Phase 1a wires the standard (gate-first, contiguous) layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_ACTIVATIONS = ("silu", "relu2")


@triton.jit
def apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    """Elementwise activation of the gate half; gating is the caller's shape."""
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
    delta_ptr,  # [num_tokens, top_k, slices * inter] canonical
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
        # Route decoding and address construction are independent of GEMM1.
        # Wait only at the first producer-owned base load.
        tl.extra.cuda.gdc_wait()
    for start in tl.range(0, inter, BLOCK_SIZE):
        offs = start + vec
        mask = offs < inter

        if NUM_SLICES == 1:
            gate_offs = offs
            up_offs = offs
        elif INTERLEAVED:
            # base gemm out columns are (g0, u0, g1, u1, ...)
            gate_offs = 2 * offs
            up_offs = 2 * offs + 1
        else:
            gate_offs = offs
            up_offs = inter + offs
        if not GATE_FIRST:
            gate_offs, up_offs = up_offs, gate_offs

        g = tl.load(gateup_row + gate_offs, mask=mask & valid, other=0.0).to(tl.float32)
        if HAS_DELTA:
            # delta is always canonical contiguous [gate | up]
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
        # activation_lora_input is written for every (t, k): zeros when invalid.
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
    """Join base and LoRA delta, then activate; gating comes from the shapes."""
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation={activation!r} is not one of {_ACTIVATIONS}")
    num_pairs = topk_ids.numel()
    inter = act_out.shape[-1]
    # Gating is a resident-shape property, independent of the activation.
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
            f"gate_up_delta must be canonical {(*topk_ids.shape, num_slices * inter)}"
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
