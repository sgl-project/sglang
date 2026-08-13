"""S3 activation-join kernel for the SGL LoRA BF16 pipeline.

SwiGLU with the gate_up LoRA delta added PRE-activation, plus the
``activation_lora_input`` side output consumed by the down-proj LoRA shrink.

Layout: the base GEMM1 output is the DeepGEMM *masked* layout
``[E_local, m_max, 2 * inter]`` viewed flat as ``[E_local * m_max, 2 * inter]``;
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


@triton.jit
def _silu_mul_delta_masked_kernel(
    gateup_ptr,  # [E_local * m_max, 2 * inter] bf16 (masked layout, flat)
    delta_ptr,  # [num_tokens, top_k, 2 * inter] bf16, contiguous [gate|up]
    act_out_ptr,  # [E_local * m_max, inter] bf16 (masked layout, flat)
    act_lora_in_ptr,  # [num_tokens, top_k, inter] bf16
    src2dst_ptr,  # [num_tokens * top_k] int32
    topk_ids_ptr,  # [num_tokens, top_k] (local ids; < 0 = invalid)
    top_k,
    inter,
    HAS_DELTA: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_idx_int32 = tl.program_id(0)
    pair_idx = pair_idx_int32.to(tl.int64)

    expert_id = tl.load(topk_ids_ptr + pair_idx)
    valid = expert_id >= 0
    dst_row = tl.load(src2dst_ptr + pair_idx, mask=valid, other=0).to(tl.int64)

    two_inter = 2 * inter
    gateup_row = gateup_ptr + dst_row * two_inter
    delta_row = delta_ptr + pair_idx * two_inter
    act_out_row = act_out_ptr + dst_row * inter
    lora_in_row = act_lora_in_ptr + pair_idx * inter

    vec = tl.arange(0, BLOCK_SIZE)
    for start in tl.range(0, inter, BLOCK_SIZE):
        offs = start + vec
        mask = offs < inter

        if INTERLEAVED:
            # base gemm out columns are (g0, u0, g1, u1, ...)
            gate_offs = 2 * offs
            up_offs = 2 * offs + 1
        else:
            gate_offs = offs
            up_offs = inter + offs
        if not GATE_FIRST:
            gate_offs, up_offs = up_offs, gate_offs

        g = tl.load(gateup_row + gate_offs, mask=mask & valid, other=0.0).to(tl.float32)
        u = tl.load(gateup_row + up_offs, mask=mask & valid, other=0.0).to(tl.float32)
        if HAS_DELTA:
            # delta is always canonical contiguous [gate | up]
            dg = tl.load(delta_row + offs, mask=mask & valid, other=0.0).to(tl.float32)
            du = tl.load(delta_row + inter + offs, mask=mask & valid, other=0.0).to(
                tl.float32
            )
            g += dg
            u += du

        act = g * tl.sigmoid(g) * u
        act_bf16 = act.to(act_out_ptr.dtype.element_ty)

        tl.store(act_out_row + offs, act_bf16, mask=mask & valid)
        # activation_lora_input is written for every (t, k): zeros when invalid.
        lora_in = tl.where(valid, act, 0.0)
        tl.store(
            lora_in_row + offs, lora_in.to(act_lora_in_ptr.dtype.element_ty), mask=mask
        )


def silu_mul_delta_masked(
    gateup_output: torch.Tensor,  # [E_local, m_max, 2 * inter] bf16
    gate_up_delta: torch.Tensor | None,  # [num_tokens, top_k, 2 * inter] bf16
    act_out: torch.Tensor,  # [E_local, m_max, inter] bf16
    activation_lora_input: torch.Tensor,  # [num_tokens, top_k, inter] bf16
    src2dst: torch.Tensor,  # [num_tokens * top_k] int32
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    gate_first: bool = True,
    interleaved: bool = False,
) -> None:
    num_pairs = topk_ids.numel()
    inter = act_out.shape[-1]
    assert gateup_output.shape[-1] == 2 * inter
    assert activation_lora_input.shape[-1] == inter

    _silu_mul_delta_masked_kernel[(num_pairs,)](
        gateup_output.view(-1, 2 * inter),
        gate_up_delta if gate_up_delta is not None else gateup_output,
        act_out.view(-1, inter),
        activation_lora_input,
        src2dst,
        topk_ids,
        topk_ids.shape[1],
        inter,
        HAS_DELTA=gate_up_delta is not None,
        GATE_FIRST=gate_first,
        INTERLEAVED=interleaved,
        BLOCK_SIZE=512,
    )
