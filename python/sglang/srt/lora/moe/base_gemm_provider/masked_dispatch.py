"""One kernel that permutes the tokens into the masked rows. It replaces the
``fused_moe_dispatch_index`` plus ``fill_gateup_input`` pair.

This is a drop-in for the BF16 branch of ``moe_ep_deepgemm_preprocess``. The
return tuple and the host formulas are the same:
``m_max = (T // 256 + 1) * 256`` and ``expected_m = ceil(T * top_k / E_local)``.

A pair with ``topk_ids < 0`` is unrouted or padding. The kernel copies no row
for it and writes no ``src2dst`` entry. That entry keeps old data, so every
consumer must test ``topk_ids >= 0`` first.

Atomic arrival order decides which slot a pair gets inside an expert. Every
stage reads the same ``src2dst``, so the final combine gives the same bits
under any slot order.

The grid comes from host-static shapes only. The kernel needs no host sync and
no readback, so a CUDA graph can capture it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_dispatch_fill_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    gateup_input_ptr,  # [E_local, m_max, hidden] bf16 rows, viewed flat
    topk_ids_ptr,  # [num_tokens * topk]; < 0 = padding or EP-unrouted
    src2dst_ptr,  # [num_tokens * topk] int32 out; valid pairs only
    masked_m_ptr,  # [E_local] int32 count and atomic cursor; caller zeroes it
    m_max,
    hidden_size,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.program_id(1)
    pair = t * TOPK + k
    expert = tl.load(topk_ids_ptr + pair)
    if expert >= 0:
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
    if block_shape is not None:
        raise ValueError("fused masked preprocess takes no quantization block shape")
    # The fill kernel downcasts a non-BF16 input and gives no warning. This
    # check rejects one first. ``output_dtype`` is a literal at every call site.
    if hidden_states.dtype != torch.bfloat16:
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

    # ``masked_m`` is also the atomic cursor, so it must start at zero. The
    # memset runs in stream order, and CUDA-graph capture accepts it.
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
