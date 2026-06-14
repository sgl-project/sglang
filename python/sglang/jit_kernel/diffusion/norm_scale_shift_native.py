"""Native-CUDA fast path for the fused norm-scale-shift diffusion ops.

In-tree dispatch glue: routes verified production operand patterns of
``fused_norm_scale_shift`` / ``fused_scale_residual_norm_scale_shift`` to the
native CUDA kernel in ``csrc/diffusion/norm_scale_shift.cuh`` (built through
``load_jit``); every other pattern returns ``None`` so the caller's original
CuTe-DSL body continues unchanged. The public custom-op registration is
untouched — only the op bodies consult this module.

Install location: ``python/sglang/jit_kernel/diffusion/norm_scale_shift_native.py``.

``set_native_enabled(False)`` (or env ``SGLANG_NSS_NATIVE_DISABLE=1``) turns
the fast path off at runtime, which gives a perfectly symmetric A/B through
the identical public op.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch

_BF16 = torch.bfloat16
_FP32 = torch.float32
_ALIGN = 32

_ENABLED = os.environ.get("SGLANG_NSS_NATIVE_DISABLE", "0").lower() not in (
    "1",
    "true",
    "yes",
)


def set_native_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = enabled


def native_enabled() -> bool:
    return _ENABLED


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % _ALIGN == 0


def _classify_operand(t, B, S, D, device):
    if t is None:
        return "absent", None
    if not isinstance(t, torch.Tensor) or t.dtype not in (_BF16, _FP32):
        return None
    if not t.is_cuda or t.device != device:
        return None
    if t.ndim >= 1 and t.stride(-1) != 1:
        return None
    if t.ndim == 1:
        if t.numel() == 1:
            return ("scalar", t) if _aligned(t) else None
        if t.shape[0] == D:
            return ("row", t) if _aligned(t) else None
        return None
    if t.ndim == 2:
        if t.shape == (1, D):
            v = t.reshape(D)
            return ("row", v) if v.is_contiguous() and _aligned(v) else None
        return None
    if t.ndim == 3:
        s0, s1, s2 = t.shape
        if s2 != D or s0 != 1 or B != 1:
            return None
        if s1 == 1:
            v = t.reshape(D)
            return ("row", v) if v.is_contiguous() and _aligned(v) else None
        if s1 == S:
            if not t.is_contiguous():
                return None
            v = t.reshape(B * S, D)
            return ("token", v) if _aligned(v) else None
        return None
    return None


def _activation_ok(t, D):
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.dtype == _BF16
        and t.ndim == 3
        and t.numel() > 0
        and t.is_contiguous()
        and _aligned(t)
        and t.shape[-1] == D
    )


def _combo_vec_bytes(sc_dtype, gate_dtype, has_wb) -> int:
    if sc_dtype == _FP32 or gate_dtype == _FP32 or has_wb:
        return 16
    return 32


def _geometry_ok(D, vec_bytes):
    elems = vec_bytes // 2
    block = D // elems if D % elems == 0 else 0
    return D % 256 == 0 and D <= 8192 and block % 32 == 0 and 32 <= block <= 1024


_NS = "sglang_norm_scale_shift"
_CLS = {"absent": 0, "scalar": 1, "row": 2, "token": 3}
_CPP_DT = {_BF16: "bf16_t", _FP32: "fp32_t"}


def _flags(vec_bytes):
    return f"false, true, false, {vec_bytes}"  # layer-only, two-pass, no PDL


def _wrapper_table():
    t = {}

    def flags(key):
        return _flags(_combo_vec_bytes(key[2], key[4], key[5]))

    for sc_class in ("row", "token"):
        for sc_dt in (_BF16, _FP32):
            key = ("nss", sc_class, sc_dt, "absent", None, False)
            t[key] = (
                f"nss_{sc_class}_{_CPP_DT[sc_dt][:-2]}",
                f"{_NS}::NormScaleShiftKernel<bf16_t, {_CPP_DT[sc_dt]}, "
                f"{_CLS[sc_class]}, {flags(key)}>::run",
            )
    key = ("srnss", "row", _BF16, "row", _BF16, False)
    t[key] = (
        "srnss_grow_bf16_row_bf16",
        f"{_NS}::ScaleResidualNormScaleShiftKernel<bf16_t, bf16_t, bf16_t, "
        f"{_CLS['row']}, {_CLS['row']}, {flags(key)}>::run",
    )
    for sc_class, sc_dt, export in (
        ("row", _BF16, "srnss_gnone_row_bf16"),
        ("row", _FP32, "srnss_gnone_row_fp32"),
        ("token", _FP32, "srnss_gnone_token_fp32"),
    ):
        key = ("srnss", sc_class, sc_dt, "absent", None, False)
        t[key] = (
            export,
            f"{_NS}::ScaleResidualNormScaleShiftKernel<bf16_t, bf16_t, "
            f"{_CPP_DT[sc_dt]}, {_CLS['absent']}, {_CLS[sc_class]}, {flags(key)}>::run_nogate",
        )
    for gate_class, export in (
        ("row", "srnss_grow_fp32_wb_scalar_bf16"),
        ("token", "srnss_gtoken_fp32_wb_scalar_bf16"),
    ):
        key = ("srnss", "scalar", _BF16, gate_class, _FP32, True)
        t[key] = (
            export,
            f"{_NS}::ScaleResidualNormScaleShiftAffineKernel<bf16_t, fp32_t, "
            f"fp32_t, bf16_t, {_CLS[gate_class]}, {_CLS['scalar']}, {flags(key)}>::run",
        )
    return t


_WRAPPERS = _wrapper_table()
_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        from sglang.jit_kernel.utils import load_jit

        cuh = Path(__file__).resolve().parents[1] / "csrc" / "diffusion" / "norm_scale_shift.cuh"
        import hashlib

        src_hash = hashlib.sha1(cuh.read_bytes()).hexdigest()[:12]
        _MOD = load_jit(
            "diffusion_norm_scale_shift_native",
            src_hash,
            cuda_files=[str(cuh)],
            cuda_wrappers=sorted(set(_WRAPPERS.values())),
        )
    return _MOD


def _native_fn(key):
    entry = _WRAPPERS.get(key)
    if entry is None:
        return None
    return getattr(_module(), entry[0])


def try_fused_norm_scale_shift(
    x, weight, bias, scale, shift, norm_type, eps
) -> Optional[torch.Tensor]:
    if not _ENABLED or norm_type != "layer" or weight is not None or bias is not None:
        return None
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.ndim == 3):
        return None
    if not (isinstance(scale, torch.Tensor) and isinstance(shift, torch.Tensor)):
        return None
    B, S, D = x.shape
    if not _activation_ok(x, D):
        return None
    sc = _classify_operand(scale, B, S, D, x.device)
    sh = _classify_operand(shift, B, S, D, x.device)
    if (
        sc is None
        or sh is None
        or sc[0] != sh[0]
        or sc[0] not in ("row", "token")
        or scale.dtype != shift.dtype
    ):
        return None
    if not _geometry_ok(D, _combo_vec_bytes(scale.dtype, None, False)):
        return None
    fn = _native_fn(("nss", sc[0], scale.dtype, "absent", None, False))
    if fn is None:
        return None
    y = torch.empty_like(x)
    fn(y.view(B * S, D), x.view(B * S, D), sc[1], sh[1], float(eps))
    return y


def try_fused_scale_residual_norm_scale_shift(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not _ENABLED or norm_type != "layer":
        return None
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(residual, torch.Tensor)
        and x.is_cuda
        and x.ndim == 3
        and residual.shape == x.shape
        and residual.dtype == x.dtype
    ):
        return None
    B, S, D = x.shape
    if not (_activation_ok(x, D) and _activation_ok(residual, D)):
        return None
    has_wb = weight is not None or bias is not None
    if has_wb:
        if not (
            isinstance(weight, torch.Tensor)
            and isinstance(bias, torch.Tensor)
            and weight.dtype == _FP32
            and bias.dtype == _FP32
            and weight.shape == (D,)
            and bias.shape == (D,)
            and weight.is_contiguous()
            and bias.is_contiguous()
            and _aligned(weight)
            and _aligned(bias)
            and weight.is_cuda
            and weight.device == x.device
            and bias.device == x.device
        ):
            return None
    if not (isinstance(scale, torch.Tensor) and isinstance(shift, torch.Tensor)):
        return None
    g = _classify_operand(gate, B, S, D, x.device)
    sc = _classify_operand(scale, B, S, D, x.device)
    sh = _classify_operand(shift, B, S, D, x.device)
    if (
        g is None
        or sc is None
        or sh is None
        or sc[0] != sh[0]
        or scale.dtype != shift.dtype
    ):
        return None
    gate_dtype = gate.dtype if isinstance(gate, torch.Tensor) else None
    key = ("srnss", sc[0], scale.dtype, g[0], gate_dtype, has_wb)
    if not _geometry_ok(D, _combo_vec_bytes(scale.dtype, gate_dtype, has_wb)):
        return None
    fn = _native_fn(key)
    if fn is None:
        return None
    y = torch.empty_like(x)
    res_out = torch.empty_like(x)
    y2, ro2 = y.view(B * S, D), res_out.view(B * S, D)
    r2, x2 = residual.view(B * S, D), x.view(B * S, D)
    e = float(eps)
    if has_wb:
        fn(y2, ro2, r2, x2, g[1], weight, bias, sc[1], sh[1], e)
    elif g[0] == "absent":
        fn(y2, ro2, r2, x2, sc[1], sh[1], e)
    else:
        fn(y2, ro2, r2, x2, g[1], sc[1], sh[1], e)
    return y, res_out
