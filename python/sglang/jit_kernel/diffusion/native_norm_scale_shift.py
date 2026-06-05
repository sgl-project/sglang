from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

_BF16 = torch.bfloat16
_FP32 = torch.float32
_USE_PDL = False
_TWO_PASS_VARIANCE = True
_VEC_BYTES_BF16 = int(os.environ.get("KDA_VEC_BYTES_BF16", "16"))
_VEC_BYTES_FP32_OPERANDS = 16
_EARLY_SCALE_SHIFT = os.environ.get("KDA_EARLY_OPS", "0") == "1"

assert _VEC_BYTES_BF16 in (16, 32), "bf16 vector width must be 16 or 32 bytes"


def _is_h200(t: torch.Tensor) -> bool:
    return torch.cuda.get_device_capability(t.device) == (9, 0)


def _aligned(t: torch.Tensor, nbytes: int = 32) -> bool:
    return t.data_ptr() % nbytes == 0


def _activation_ok(t: torch.Tensor, d: int) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and _is_h200(t)
        and not torch.is_grad_enabled()
        and not t.requires_grad
        and t.dtype == _BF16
        and t.ndim == 3
        and t.numel() > 0
        and t.is_contiguous()
        and _aligned(t)
        and t.shape[-1] == d
    )


def _classify_operand(
    t: Optional[torch.Tensor], b: int, s: int, d: int, device: torch.device
):
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
        if t.shape[0] == d:
            return ("row", t) if _aligned(t) else None
        return None
    if t.ndim == 2:
        if t.shape == (1, d):
            v = t.reshape(d)
            return ("row", v) if v.is_contiguous() and _aligned(v) else None
        return None
    if t.ndim == 3:
        s0, s1, s2 = t.shape
        if s2 != d or s0 != 1 or b != 1:
            return None
        if s1 == 1:
            v = t.reshape(d)
            return ("row", v) if v.is_contiguous() and _aligned(v) else None
        if s1 == s:
            if not t.is_contiguous():
                return None
            v = t.reshape(b * s, d)
            return ("token", v) if _aligned(v) else None
        return None
    return None


def _geometry_ok(d: int, vec_bytes: int) -> bool:
    elems = vec_bytes // 2
    block = d // elems if d % elems == 0 else 0
    return d % 256 == 0 and d <= 8192 and block % 32 == 0 and 32 <= block <= 1024


_NS = "kda_norm_scale_shift"
_CLS = {"absent": 0, "scalar": 1, "row": 2, "token": 3}
_CPP_DT = {_BF16: "bf16_t", _FP32: "fp32_t"}


def _combo_vec_bytes(sc_dtype, gate_dtype, has_wb: bool) -> int:
    if sc_dtype == _FP32 or gate_dtype == _FP32 or has_wb:
        return _VEC_BYTES_FP32_OPERANDS
    return _VEC_BYTES_BF16


def _flags(vec_bytes: int, early: bool) -> str:
    tp = "true" if _TWO_PASS_VARIANCE else "false"
    pdl = "true" if _USE_PDL else "false"
    eo = "true" if early else "false"
    return f"false, {tp}, {pdl}, {eo}, {vec_bytes}"


def _wrapper_table():
    table = {}

    def add(key, base_name: str, symbol_fmt: str, widths: tuple[int, ...]):
        earlies = (False, True) if key[1] in ("row", "token") else (False,)
        table[key] = {
            (vec, early): (
                f"{base_name}_v{vec}{'_eo' if early else ''}",
                symbol_fmt.format(flags=_flags(vec, early)),
            )
            for vec in widths
            for early in earlies
        }

    both = (16, 32)
    narrow = (16,)
    for sc_class in ("row", "token"):
        for sc_dt in (_BF16, _FP32):
            key = ("nss", sc_class, sc_dt, "absent", None, False)
            widths = both if sc_dt == _BF16 else narrow
            add(
                key,
                f"nss_{sc_class}_{_CPP_DT[sc_dt][:-2]}",
                f"{_NS}::NormScaleShiftKernel<bf16_t, {_CPP_DT[sc_dt]}, "
                f"{_CLS[sc_class]}, {{flags}}>::run",
                widths,
            )
    add(
        ("srnss", "row", _BF16, "row", _BF16, False),
        "srnss_grow_bf16_row_bf16",
        f"{_NS}::ScaleResidualNormScaleShiftKernel<bf16_t, bf16_t, bf16_t, "
        f"{_CLS['row']}, {_CLS['row']}, {{flags}}>::run",
        both,
    )
    for sc_class, sc_dt, export, widths in (
        ("row", _BF16, "srnss_gnone_row_bf16", both),
        ("row", _FP32, "srnss_gnone_row_fp32", narrow),
        ("token", _FP32, "srnss_gnone_token_fp32", narrow),
    ):
        add(
            ("srnss", sc_class, sc_dt, "absent", None, False),
            export,
            f"{_NS}::ScaleResidualNormScaleShiftKernel<bf16_t, bf16_t, "
            f"{_CPP_DT[sc_dt]}, {_CLS['absent']}, {_CLS[sc_class]}, {{flags}}>::run_nogate",
            widths,
        )
    for gate_class, export in (
        ("row", "srnss_grow_fp32_wb_scalar_bf16"),
        ("token", "srnss_gtoken_fp32_wb_scalar_bf16"),
    ):
        add(
            ("srnss", "scalar", _BF16, gate_class, _FP32, True),
            export,
            f"{_NS}::ScaleResidualNormScaleShiftAffineKernel<bf16_t, fp32_t, "
            f"fp32_t, bf16_t, {_CLS[gate_class]}, {_CLS['scalar']}, {{flags}}>::run",
            narrow,
        )
    return table


_WRAPPERS = _wrapper_table()
_ROUTED_TO_CUTEDSL = {
    ("nss", "row", _FP32, "absent", None, False),
}


@cache_once
def _module():
    wrappers = sorted(
        {entry for combos in _WRAPPERS.values() for entry in combos.values()}
    )
    extra = os.environ.get("KDA_EXTRA_CUDA_CFLAGS", "").split()
    tag = f"x{abs(hash(tuple(extra))) % 10**8}" if extra else "clean"
    return load_jit(
        "diffusion_native_norm_scale_shift_h200",
        tag,
        cuda_files=["diffusion/norm_scale_shift_h200.cuh"],
        cuda_wrappers=wrappers,
        extra_cuda_cflags=extra,
    )


def _native_fn(key, vec_bytes: int):
    combos = _WRAPPERS.get(key)
    if combos is None:
        return None
    entry = combos.get((vec_bytes, _EARLY_SCALE_SHIFT and key[1] in ("row", "token")))
    if entry is None:
        return None
    return getattr(_module(), entry[0])


@register_custom_op(op_name="diffusion_native_norm_scale_shift_cuda", out_shape="x")
def _native_norm_scale_shift_cuda(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    scale_class: str,
) -> torch.Tensor:
    b, s, d = (int(v) for v in x.shape)
    y = torch.empty_like(x)
    sc = _classify_operand(scale, b, s, d, x.device)
    sh = _classify_operand(shift, b, s, d, x.device)
    assert sc is not None and sh is not None and sc[0] == sh[0] == scale_class
    vec = _combo_vec_bytes(scale.dtype, None, False)
    key = ("nss", scale_class, scale.dtype, "absent", None, False)
    fn = _native_fn(key, vec)
    assert fn is not None
    fn(y.view(b * s, d), x.view(b * s, d), sc[1], sh[1], float(eps))
    return y


def _srnss_fake(
    residual, x, gate, weight, bias, scale, shift, eps, scale_class, gate_class, has_wb
):
    return x.new_empty(x.shape), x.new_empty(x.shape)


@register_custom_op(
    op_name="diffusion_native_scale_residual_norm_scale_shift_cuda",
    fake_impl=_srnss_fake,
)
def _native_scale_residual_norm_scale_shift_cuda(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    scale_class: str,
    gate_class: str,
    has_wb: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, d = (int(v) for v in x.shape)
    y = torch.empty_like(x)
    res_out = torch.empty_like(x)
    g = _classify_operand(gate, b, s, d, x.device)
    sc = _classify_operand(scale, b, s, d, x.device)
    sh = _classify_operand(shift, b, s, d, x.device)
    assert g is not None and sc is not None and sh is not None
    assert sc[0] == sh[0] == scale_class and g[0] == gate_class
    gate_dtype = gate.dtype if isinstance(gate, torch.Tensor) else None
    key = ("srnss", scale_class, scale.dtype, gate_class, gate_dtype, bool(has_wb))
    vec = _combo_vec_bytes(scale.dtype, gate_dtype, bool(has_wb))
    fn = _native_fn(key, vec)
    assert fn is not None
    y2, ro2 = y.view(b * s, d), res_out.view(b * s, d)
    r2, x2 = residual.view(b * s, d), x.view(b * s, d)
    if has_wb:
        assert gate is not None and weight is not None and bias is not None
        fn(y2, ro2, r2, x2, g[1], weight, bias, sc[1], sh[1], float(eps))
    elif gate_class == "absent":
        fn(y2, ro2, r2, x2, sc[1], sh[1], float(eps))
    else:
        assert gate is not None
        fn(y2, ro2, r2, x2, g[1], sc[1], sh[1], float(eps))
    return y, res_out


def try_native_fused_norm_scale_shift(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> Optional[torch.Tensor]:
    if not (
        isinstance(x, torch.Tensor)
        and x.ndim == 3
        and x.shape[0] == 1
        and norm_type == "layer"
        and weight is None
        and bias is None
    ):
        return None
    b, s, d = (int(v) for v in x.shape)
    if not _activation_ok(x, d):
        return None
    sc = _classify_operand(scale, b, s, d, x.device)
    sh = _classify_operand(shift, b, s, d, x.device)
    if sc is None or sh is None or sc[0] != sh[0] or sc[0] not in ("row", "token"):
        return None
    if scale.dtype != shift.dtype:
        return None
    key = ("nss", sc[0], scale.dtype, "absent", None, False)
    if key in _ROUTED_TO_CUTEDSL:
        return None
    vec = _combo_vec_bytes(scale.dtype, None, False)
    if not _geometry_ok(d, vec) or _native_fn(key, vec) is None:
        return None
    try:
        return _native_norm_scale_shift_cuda(x, scale, shift, float(eps), sc[0])
    except Exception:
        return None


def try_native_fused_scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(residual, torch.Tensor)
        and x.ndim == 3
        and x.shape[0] == 1
        and norm_type == "layer"
        and residual.shape == x.shape
        and residual.dtype == x.dtype
    ):
        return None
    b, s, d = (int(v) for v in x.shape)
    if not (_activation_ok(x, d) and _activation_ok(residual, d)):
        return None
    has_wb = weight is not None or bias is not None
    if has_wb:
        if not (
            isinstance(weight, torch.Tensor)
            and isinstance(bias, torch.Tensor)
            and weight.dtype == _FP32
            and bias.dtype == _FP32
            and weight.shape == (d,)
            and bias.shape == (d,)
            and weight.is_contiguous()
            and bias.is_contiguous()
            and _aligned(weight)
            and _aligned(bias)
            and weight.is_cuda
            and weight.device == x.device
            and bias.is_cuda
            and bias.device == x.device
        ):
            return None
    g = _classify_operand(gate, b, s, d, x.device)
    sc = _classify_operand(scale, b, s, d, x.device)
    sh = _classify_operand(shift, b, s, d, x.device)
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
    vec = _combo_vec_bytes(scale.dtype, gate_dtype, has_wb)
    if not _geometry_ok(d, vec) or _native_fn(key, vec) is None:
        return None
    try:
        return _native_scale_residual_norm_scale_shift_cuda(
            residual,
            x,
            gate,
            weight,
            bias,
            scale,
            shift,
            float(eps),
            sc[0],
            g[0],
            has_wb,
        )
    except Exception:
        return None


__all__ = [
    "try_native_fused_norm_scale_shift",
    "try_native_fused_scale_residual_norm_scale_shift",
]
