from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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
    amax_buf: Optional[torch.Tensor] = None,
    done_counter: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = q.shape[0]
    q_row = q.numel() // num_tokens
    kv_row = k_cache.shape[-1] * k_cache.shape[-2]

    # amax_buf/done_counter are persistent scratch kept at zero by the kernel's
    # last-block reset; pass long-lived buffers to avoid a per-call fill.
    if amax_buf is None:
        amax_buf = torch.zeros(1, device=q.device, dtype=torch.float32)
    if done_counter is None:
        done_counter = torch.zeros(1, device=q.device, dtype=torch.int32)

    q2d = q.view(num_tokens, q_row)
    k2d = k.view(num_tokens, kv_row)
    v2d = v.view(num_tokens, kv_row)
    q_out = torch.empty(num_tokens, q_row, device=q.device, dtype=torch.float8_e4m3fn)
    bmm1_scale = torch.empty(1, device=q.device, dtype=torch.float32)

    module = _jit_fused_q_quant_kv_write_module(q.dtype)
    module.fused_q_quant_kv_write(
        q2d,
        q_out.view(-1),
        amax_buf,
        done_counter,
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
        q_row,
        kv_row,
        q2d.stride(0),
        k2d.stride(0),
        v2d.stride(0),
        kv_row,
    )
    return q_out.view(q.shape), bmm1_scale
