"""SM70 dense MXFP8 e4m3+UE8M0 linear (DeepSeek-V4.1-Flash).

Decode M<=4 is a packed GEMV. Prefill M>4 dequants into a transient fp16
weight and uses ``F.linear``. marlin_v100 FP8 has no group-32; this is not
Marlin W8A16.

Engram ``wkv`` layers can stash the fp16 unpack on ``owner._sm70_prefill_fp16_w``
so a 2048-token chunked prefill does not re-allocate ~300 MiB against a
fragmented caching allocator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    dequant_mxfp8_ue8m0_to_fp16,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_K_MAX_M = 4
_K_GROUP = 32
_PREFILL_UNPACK_HEADROOM_B = 1024 * 1024 * 1024


@cache_once
def _module() -> Module:
    cap = torch.cuda.get_device_capability()
    if cap[0] != 7:
        raise RuntimeError(
            f"sm70_dsv41_mxfp8_linear requires SM70 (Volta); got SM{cap[0]}{cap[1]}"
        )
    return load_jit(
        "sm70_dsv41_mxfp8_linear",
        cuda_files=["sm70_dsv41_mxfp8_linear.cuh"],
        cuda_wrappers=[("linear", "sm70_dsv41_mxfp8::linear")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
        extra_cuda_cflags=["-O3"],
    )


def sm70_dsv41_mxfp8_gemv_enabled() -> bool:
    return bool(envs.SGLANG_DSV41_MXFP8_W8A16.get())


def _pad_m(m: int) -> int:
    if m <= 1:
        return 1
    if m <= 2:
        return 2
    return 4


def _dequant_fp16(
    weight: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        if free_b < _PREFILL_UNPACK_HEADROOM_B:
            torch.cuda.empty_cache()
    return dequant_mxfp8_ue8m0_to_fp16(
        weight, scales, (1, 32), out_dtype=out_dtype
    )


def sm70_dsv41_mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    owner: Any = None,
) -> torch.Tensor:
    """``F.linear``-shaped MXFP8: y = x @ W.dequant.T + bias."""
    if isinstance(x, tuple):
        x = x[0]
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    m, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    out_shape = orig_shape[:-1] + (n,)
    if m == 0:
        return x.new_empty(out_shape)

    use_gemv = (
        sm70_dsv41_mxfp8_gemv_enabled()
        and x2.dtype == torch.float16
        and 1 <= m <= _K_MAX_M
        and k % _K_GROUP == 0
        and int(weight.shape[1]) == k
        and int(scales.numel()) == n * (k // _K_GROUP)
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 7
    )
    if use_gemv:
        mp = _pad_m(m)
        x_in = x2 if x2.is_contiguous() else x2.contiguous()
        if mp != m:
            padded = x_in.new_zeros((mp, k))
            padded[:m].copy_(x_in)
            x_in = padded
        y = torch.empty((mp, n), dtype=torch.float16, device=x_in.device)
        w_u8 = (
            weight.view(torch.uint8)
            if weight.dtype == torch.float8_e4m3fn
            else weight
        )
        s_u8 = (
            scales.view(torch.uint8)
            if scales.dtype == torch.float8_e8m0fnu
            else scales
        )
        if not w_u8.is_contiguous():
            w_u8 = w_u8.contiguous()
        s_view = s_u8.reshape(n, k // _K_GROUP)
        if not s_view.is_contiguous():
            s_view = s_view.contiguous()
        _module().linear(
            y,
            x_in,
            w_u8,
            s_view,
        )
        y = y[:m]
        if bias is not None:
            y = y + bias.to(dtype=y.dtype)
        return y.view(out_shape)

    fp16_w = getattr(owner, "_sm70_prefill_fp16_w", None) if owner is not None else None
    if fp16_w is None:
        fp16_w = _dequant_fp16(weight, scales, x.dtype)
    # Pad M so T=6 TARGET_VERIFY and T=25 EXTEND share one F.linear tile.
    # Decode M<=4 stays on GEMV above and is not padded.
    mp = (m + 31) // 32 * 32
    if mp != m:
        x_pad = x2.new_zeros((mp, k))
        x_pad[:m].copy_(x2)
        y = F.linear(x_pad, fp16_w, bias)
        return y[:m].view(out_shape)
    return F.linear(x, fp16_w, bias)
