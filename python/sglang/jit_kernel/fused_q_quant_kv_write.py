from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fused_q_quant_kv_write_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_q_quant_kv_write",
        *args,
        cuda_files=["attention/fused_q_quant_kv_write.cuh"],
        cuda_wrappers=[("fused_q_quant_kv_write", f"fused_q_quant_kv_write<{args}>")],
    )


def fused_q_quant_kv_write(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    inv_k_scale: float,
    inv_v_scale: float,
    bmm1_extra: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Q FP8 quant + paged KV FP8 write for the TRTLLM-GEN MHA path.

    In a single kernel launch this:
      - dynamically per-tensor quantizes ``q`` (BF16/FP16) to FP8 E4M3,
      - emits ``bmm1_scale = (amax_q / 448) * bmm1_extra`` (a 1-element f32
        tensor; ``bmm1_extra`` is the host-side ``k_scale * attn_scaling``),
      - statically quantizes ``k``/``v`` to FP8 with ``inv_*_scale`` and scatters
        them into the slot-major NHD paged cache at ``cache_loc``.

    Args:
        q: [num_tokens, num_q_heads * head_dim] query, contiguous.
        k, v: [num_tokens, num_kv_heads * head_dim]; within-row contiguous.
        k_cache, v_cache: [total_slots, num_kv_heads, head_dim] FP8 paged cache.
        cache_loc: [num_tokens] int64 slot indices.
        inv_k_scale, inv_v_scale: 1 / k_scale, 1 / v_scale (host floats).
        bmm1_extra: k_scale * attn_scaling (host float).

    Returns:
        (q_fp8, bmm1_scale): q_fp8 has q's shape; bmm1_scale is f32[1].
    """
    num_tokens = q.shape[0]
    kv_row = k_cache.shape[-1] * k_cache.shape[-2]

    # Per-tensor amax + elementwise quant are order-independent, so a flat
    # row-major view is enough; reshape is a free view for the contiguous q
    # this path sees (it only copies for the rare non-contiguous input).
    q_flat = q.reshape(-1)
    q_out = torch.empty_like(q_flat, dtype=torch.float8_e4m3fn)
    bmm1_scale = torch.empty(1, device=q.device, dtype=torch.float32)

    # Head/dim are contiguous within a token even when the token stride is
    # non-canonical (QKV split + RoPE invariant), so this 2D view is valid and
    # the kernel reads each row with stride(0) + a contiguous inner span.
    k2d = k.view(num_tokens, kv_row)
    v2d = v.view(num_tokens, kv_row)

    module = _jit_fused_q_quant_kv_write_module(q.dtype)
    module.fused_q_quant_kv_write(
        q_flat,
        q_out,
        bmm1_scale,
        k2d,
        v2d,
        k_cache.view(-1, kv_row),
        v_cache.view(-1, kv_row),
        cache_loc,
        float(inv_k_scale),
        float(inv_v_scale),
        float(bmm1_extra),
        num_tokens,
        kv_row,
        k2d.stride(0),
        v2d.stride(0),
        kv_row,
    )
    return q_out.view(q.shape), bmm1_scale
