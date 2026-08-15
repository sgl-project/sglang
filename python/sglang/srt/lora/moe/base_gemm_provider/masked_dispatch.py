"""Fused S1 dispatch for the MoE LoRA BF16 pipeline.

One Triton launch replacing the two-kernel ``moe_ep_deepgemm_preprocess``
composition (``fused_moe_dispatch_index`` + ``fill_gateup_input``): each
(token, k) pair reserves its masked-layout slot with one atomic, stores its
``src2dst`` entry, and copies its token row into the ``[E_local, m_max, H]``
slab in the same pass.  Engine-local by design — the shared no-LoRA
``ep_moe_kernels`` path is untouched (precedent: ``masked_activation``).

Contract (drop-in for the bf16 branch of ``moe_ep_deepgemm_preprocess``):

- Returns the same 5-tuple ``(masked_m, expected_m, src2dst, gateup_input,
  None)`` with the same host formulas ``m_max = (T // 256 + 1) * 256`` and
  ``expected_m = ceil(T * top_k / E_local)``.
- Sentinel: pairs with ``topk_ids < 0`` (EP-unrouted / padding) take no
  atomic, get NO ``src2dst`` store (entries stay uninitialized), and copy
  nothing; every consumer gates on ``topk_ids >= 0``.
- Slot order within an expert is atomic-arrival nondeterministic, exactly as
  in the two-kernel composition; all stages share the one ``src2dst``, so the
  final combine is bitwise-independent of the slot permutation.
- Graph safety: the grid derives only from host-static shapes, ``masked_m``
  is zeroed by a stream-ordered memset before the launch, and there is no
  host sync or readback.

BF16 only: the engine path always runs ``output_dtype=torch.bfloat16`` with
no quantization block shape.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_dispatch_fill_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    gateup_input_ptr,  # [E_local, m_max, hidden] bf16 slab, viewed flat
    topk_ids_ptr,  # [num_tokens * topk]; < 0 = padding / EP-unrouted
    src2dst_ptr,  # [num_tokens * topk] int32 out (valid lanes only)
    masked_m_ptr,  # [E_local] int32 cursor + count, PRE-ZEROED by caller
    m_max,
    hidden_size,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)  # token
    k = tl.program_id(1)  # k-slot
    pair = t * TOPK + k
    expert = tl.load(topk_ids_ptr + pair)
    if expert >= 0:
        # The atomic returns this pair's destination row in-register, so the
        # same CTA reserves the slot, stores the mapping, and copies the row.
        slot = tl.atomic_add(masked_m_ptr + expert, 1)
        dst = expert.to(tl.int64) * m_max + slot
        tl.store(src2dst_ptr + pair, dst.to(tl.int32))
        src = input_ptr + t.to(tl.int64) * hidden_size
        out = gateup_input_ptr + dst * hidden_size
        vec = tl.arange(0, BLOCK_H)
        for off in tl.range(0, hidden_size, BLOCK_H):
            mask = off + vec < hidden_size
            tl.store(out + off + vec, tl.load(src + off + vec, mask=mask), mask=mask)


def fused_masked_preprocess(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    hidden_states: torch.Tensor,
    top_k: int,
    block_shape=None,
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    masked_m_out: torch.Tensor | None = None,
    src2dst_out: torch.Tensor | None = None,
    gateup_input_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, None]:
    """One-launch masked dispatch; same signature/return as the bf16 branch
    of ``moe_ep_deepgemm_preprocess`` so ``prepare()`` binds it unchanged."""
    if block_shape is not None:
        raise ValueError("fused masked preprocess takes no quantization block shape")
    if output_dtype != torch.bfloat16 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("fused masked preprocess is BF16-only")
    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous [num_tokens, hidden]")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    num_tokens = hidden_states.size(0)
    num_pairs = topk_ids.numel()
    if num_pairs != num_tokens * top_k:
        raise ValueError(
            f"topk_ids carries {num_pairs} pairs for {num_tokens} tokens x "
            f"top_k={top_k}"
        )
    m_max = (hidden_states.size(0) // 256 + 1) * 256
    expected_m = (topk_ids.numel() - 1) // num_local_experts + 1

    if src2dst_out is None:
        src2dst = torch.empty(num_pairs, device=topk_ids.device, dtype=torch.int32)
    else:
        if (
            src2dst_out.shape != (num_pairs,)
            or src2dst_out.dtype != torch.int32
            or src2dst_out.device != topk_ids.device
            or not src2dst_out.is_contiguous()
        ):
            raise ValueError(
                f"src2dst_out must be contiguous int32 [{num_pairs}] on "
                f"{topk_ids.device}"
            )
        src2dst = src2dst_out
    if masked_m_out is None:
        masked_m = torch.empty(
            num_local_experts, device=topk_ids.device, dtype=torch.int32
        )
    else:
        if (
            masked_m_out.shape != (num_local_experts,)
            or masked_m_out.dtype != torch.int32
            or masked_m_out.device != topk_ids.device
            or not masked_m_out.is_contiguous()
        ):
            raise ValueError(
                f"masked_m_out must be contiguous int32 [{num_local_experts}] "
                f"on {topk_ids.device}"
            )
        masked_m = masked_m_out
    gateup_shape = (num_local_experts, m_max, hidden_states.size(1))
    if gateup_input_out is None:
        gateup_input = torch.empty(
            gateup_shape, device=hidden_states.device, dtype=torch.bfloat16
        )
    else:
        if (
            gateup_input_out.shape != gateup_shape
            or gateup_input_out.dtype != torch.bfloat16
            or gateup_input_out.device != hidden_states.device
            or not gateup_input_out.is_contiguous()
        ):
            raise ValueError(
                f"gateup_input_out must be contiguous bf16 {gateup_shape} on "
                f"{hidden_states.device}"
            )
        gateup_input = gateup_input_out

    # One uniform path: masked_m doubles as the atomic cursor and must be
    # zeroed before any atomic_add. A stream-ordered memset is legal inside
    # CUDA-graph capture (the multi-block dispatch path captures exactly this
    # memset today).
    masked_m.zero_()
    if num_tokens > 0:
        fused_dispatch_fill_kernel[(num_tokens, top_k)](
            hidden_states,
            gateup_input,
            topk_ids.view(-1),
            src2dst,
            masked_m,
            m_max,
            hidden_states.size(1),
            TOPK=top_k,
            BLOCK_H=1024,
        )
    return masked_m, expected_m, src2dst, gateup_input, None
