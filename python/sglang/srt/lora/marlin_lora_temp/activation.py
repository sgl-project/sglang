"""Fused ``silu(gate + Δgate) * (up + Δup)`` for the experimental marlin MoE-LoRA path.

Replaces the stock sequence (in-place LoRA delta add over ``[T, 2N]`` +
``silu_and_mul``) with a single kernel: one read of the GEMM output, one read
of the delta, one write of the activation — and it lets the delta live in its
own buffer, which is what makes the side-stream gate_up overlap possible (the
side stream must not race the marlin GEMM writing the shared cache).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_add_delta_kernel(
    x_ptr,  # [T, 2N] gemm1 output, [gate || up]
    d_ptr,  # [T, 2N] LoRA gate_up delta, same layout
    out_ptr,  # [T, N]
    topk_ids_ptr,
    token_lora_mapping_ptr,
    adapter_enabled_ptr,
    N,
    stride_im,
    stride_ik,
    stride_tm,
    stride_ae,
    ROUTER_TOPK: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    HAS_ROUTING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    if HAS_ROUTING:
        token = row // ROUTER_TOPK
        route = row % ROUTER_TOPK
        route_live = tl.load(topk_ids_ptr + token * stride_im + route * stride_ik) >= 0
        slot = tl.load(token_lora_mapping_ptr + token * stride_tm).to(tl.int64)
        if NUM_SLOTS == 1:
            enabled = (slot == 0) & (tl.load(adapter_enabled_ptr) != 0)
        else:
            enabled = (
                tl.load(
                    adapter_enabled_ptr + slot * stride_ae,
                    mask=(slot >= 0) & (slot < NUM_SLOTS),
                    other=0,
                )
                != 0
            )
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    if HAS_ROUTING:
        # Keep every payload load and store behind the original route predicate.
        mask = mask & route_live
    delta_mask = mask
    if HAS_ROUTING:
        delta_mask = delta_mask & enabled

    x_row = x_ptr + row * 2 * N
    d_row = d_ptr + row * 2 * N
    gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row + N + cols, mask=mask, other=0.0).to(tl.float32)
    gate += tl.load(d_row + cols, mask=delta_mask, other=0.0).to(tl.float32)
    up += tl.load(d_row + N + cols, mask=delta_mask, other=0.0).to(tl.float32)

    silu = gate * tl.sigmoid(gate)
    out = silu * up
    tl.store(out_ptr + row * N + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul_add_delta(
    gemm1_out: torch.Tensor,
    gate_up_delta: torch.Tensor,
    out: torch.Tensor,
    *,
    topk_ids: torch.Tensor | None = None,
    token_lora_mapping: torch.Tensor | None = None,
    adapter_enabled: torch.Tensor | None = None,
) -> None:
    """Compute the activation, optionally skipping private inactive route rows.

    Routing metadata is all-or-none. In that mode, disabled or absent adapters
    contribute no delta and dropped routes leave their private output unwritten.
    """
    T, two_n = gemm1_out.shape
    N = two_n // 2
    assert gate_up_delta.shape == (T, two_n) and out.shape == (T, N)
    assert gemm1_out.is_contiguous() and gate_up_delta.is_contiguous()
    assert out.is_contiguous()
    if topk_ids is None:
        assert token_lora_mapping is None and adapter_enabled is None
    else:
        assert token_lora_mapping is not None and adapter_enabled is not None
        assert topk_ids.ndim == 2 and topk_ids.numel() == T
        assert token_lora_mapping.shape == (topk_ids.shape[0],)
        assert adapter_enabled.ndim == 1
        assert topk_ids.dtype in (torch.int32, torch.int64)
        assert token_lora_mapping.dtype in (torch.int32, torch.int64)
        assert adapter_enabled.dtype in (torch.bool, torch.int32, torch.int64)
        assert topk_ids.device == token_lora_mapping.device == gemm1_out.device
        assert adapter_enabled.device == gemm1_out.device
    if T == 0:
        return
    BLOCK = 512
    _silu_mul_add_delta_kernel[(T, triton.cdiv(N, BLOCK))](
        gemm1_out,
        gate_up_delta,
        out,
        topk_ids,
        token_lora_mapping,
        adapter_enabled,
        N,
        topk_ids.stride(0) if topk_ids is not None else 0,
        topk_ids.stride(1) if topk_ids is not None else 0,
        token_lora_mapping.stride(0) if token_lora_mapping is not None else 0,
        adapter_enabled.stride(0) if adapter_enabled is not None else 0,
        ROUTER_TOPK=topk_ids.shape[1] if topk_ids is not None else 1,
        NUM_SLOTS=adapter_enabled.numel() if adapter_enabled is not None else 0,
        HAS_ROUTING=topk_ids is not None,
        BLOCK=BLOCK,
    )
