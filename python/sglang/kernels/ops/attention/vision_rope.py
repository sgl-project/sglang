"""Fused interleaved complex RoPE for vision attention Q/K tensors."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n_pairs"])
def _fused_qk_complex_rope_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    n_pairs,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    q_stride_token: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    freq_stride_token: tl.constexpr,
    freq_stride_pair: tl.constexpr,
    freq_stride_complex: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pair_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pair_offsets < n_pairs
    pairs_per_row = head_dim // 2
    row = pair_offsets // pairs_per_row
    pair = pair_offsets - row * pairs_per_row
    token = row // n_heads
    head = row - token * n_heads

    q_base = token * q_stride_token + head * q_stride_head + pair * 2 * q_stride_dim
    k_base = token * k_stride_token + head * k_stride_head + pair * 2 * k_stride_dim
    freq_base = token * freq_stride_token + pair * freq_stride_pair
    cos = tl.load(freqs_ptr + freq_base, mask=mask).to(tl.float32)
    sin = tl.load(freqs_ptr + freq_base + freq_stride_complex, mask=mask).to(tl.float32)
    q_real = tl.load(q_ptr + q_base, mask=mask).to(tl.float32)
    q_imag = tl.load(q_ptr + q_base + q_stride_dim, mask=mask).to(tl.float32)
    k_real = tl.load(k_ptr + k_base, mask=mask).to(tl.float32)
    k_imag = tl.load(k_ptr + k_base + k_stride_dim, mask=mask).to(tl.float32)

    out_base = row * head_dim + pair * 2
    tl.store(q_out_ptr + out_base, q_real * cos - q_imag * sin, mask=mask)
    tl.store(q_out_ptr + out_base + 1, q_real * sin + q_imag * cos, mask=mask)
    tl.store(k_out_ptr + out_base, k_real * cos - k_imag * sin, mask=mask)
    tl.store(k_out_ptr + out_base + 1, k_real * sin + k_imag * cos, mask=mask)


def can_use_fused_qk_complex_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> bool:
    """Whether the NVIDIA fused path supports these vision RoPE tensors."""

    if not (
        q.is_cuda
        and k.is_cuda
        and freqs_cis.is_cuda
        and q.device == k.device == freqs_cis.device
    ):
        return False
    if q.dtype != k.dtype or q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if freqs_cis.dtype != torch.complex64 or q.shape != k.shape or q.ndim < 3:
        return False
    if q.shape[-1] % 2 != 0:
        return False
    if freqs_cis.shape != q.shape[:-2] + (q.shape[-1] // 2,):
        return False
    major, _ = torch.cuda.get_device_capability(q.device)
    return major >= 9


def apply_fused_qk_complex_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate interleaved Q/K pairs with one kernel.

    ``q`` and ``k`` may be strided views of an interleaved QKV projection. The
    output is contiguous, matching the native complex-multiply implementation.
    The token count remains a runtime kernel argument so random image sizes do
    not create new Triton specializations.
    """

    if not can_use_fused_qk_complex_rope(q, k, freqs_cis):
        raise ValueError(
            "Unsupported fused vision RoPE inputs: "
            f"q={q.shape}/{q.dtype}/{q.device}, "
            f"k={k.shape}/{k.dtype}/{k.device}, "
            f"freqs={freqs_cis.shape}/{freqs_cis.dtype}/{freqs_cis.device}"
        )

    original_shape = q.shape
    # Preserve the interleaved QKV token stride when the token dimension is 1.
    # ``view(-1, ...)`` is otherwise free to collapse that singleton stride,
    # producing a different Triton specialization from real image requests.
    q_flat = q if q.ndim == 3 else q.view(-1, q.shape[-2], q.shape[-1])
    k_flat = k if k.ndim == 3 else k.view(-1, k.shape[-2], k.shape[-1])
    freqs = torch.view_as_real(freqs_cis).view(-1, q.shape[-1] // 2, 2)
    q_out = torch.empty(q_flat.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k_flat.shape, dtype=k.dtype, device=k.device)

    block = 128
    n_pairs = q_flat.numel() // 2
    _fused_qk_complex_rope_kernel[(triton.cdiv(n_pairs, block),)](
        q_flat,
        k_flat,
        freqs,
        q_out,
        k_out,
        n_pairs,
        q_flat.shape[1],
        q_flat.shape[2],
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        freqs.stride(0),
        freqs.stride(1),
        freqs.stride(2),
        BLOCK=block,
        num_warps=4,
    )
    return q_out.view(original_shape), k_out.view(original_shape)


def precompile_fused_qk_complex_rope(
    *,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Compile the dynamic-token QKV-view specialization before serving."""

    if device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16):
        return False
    qkv = torch.empty((1, 3, num_heads, head_dim), dtype=dtype, device=device)
    q, k, _ = torch.unbind(qkv, dim=1)
    freqs = torch.ones((1, head_dim // 2), dtype=torch.complex64, device=device)
    if not can_use_fused_qk_complex_rope(q, k, freqs):
        return False
    apply_fused_qk_complex_rope(q, k, freqs)
    return True
