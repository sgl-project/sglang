"""Lazy dtype resolution and checkpoint scale/payload decoding for NPU."""

import torch

_E8M0_DTYPE = None


def _get_float8_e8m0fnu_dtype():
    # Resolve lazily rather than as a module-level constant: this module is
    # imported early (during quant-scheme registration), so reading the dtype at
    # call time keeps it correct regardless of import order / platform.
    return getattr(torch, "float8_e8m0fnu", None)


def _get_float4_e2m1fn_x2_dtype():
    # The packed-FP4 dtype MUST come from torch_npu (an int enum, e.g. 296), not
    # from torch. The NPU ops that consume it -- npu_dynamic_mx_quant(dst_type=),
    # npu_quant_matmul(x2_dtype=), npu_format_cast(input_dtype=) -- REJECT the
    # torch dtype object torch.float4_e2m1fn_x2 in op-plugin on recent torch_npu
    # builds (it raises, or with None gives "output y must be same shape as input
    # x"), even though torch.float4_e2m1fn_x2 exists. This is fp4-specific: fp8 /
    # float8_e8m0fnu is accepted from torch either way. Verified on A5 /
    # torch_npu 2.10.0.post2.dev20260704 (see llm/probe_fp4_w4a8_chain.py: dst=296
    # passes the full quant->format_cast->matmul chain, dst=torch dtype fails).
    #
    # Lazy import so this NPU-only path keeps the module importable on
    # CUDA/CPU/AMD/XPU CI (no top-level torch_npu; see AGENTS.md known pitfalls).
    from sglang.srt.utils import is_npu

    if is_npu():
        import torch_npu

        npu_dtype = getattr(torch_npu, "float4_e2m1fn_x2", None)
        if npu_dtype is not None:
            return npu_dtype
    return getattr(torch, "float4_e2m1fn_x2", None)


def _require_e8m0_dtype():
    """Resolve the e8m0 block-scale dtype, failing loudly if it is unavailable.

    The grouped matmuls validate their scale-dtype arguments against torch_npu's
    own dtype enum (``torch_npu.float8_e8m0fnu``, 293 on A5) and reject the torch
    dtype object with "weight_scale_dtype only supports float8_e8m0fnu or None,
    but the actual value is Float8_e8m0fnu" — hence torch_npu first, torch only
    as a fallback. Dense ``npu_quant_matmul`` accepts either, which is why
    ``_get_float8_e8m0fnu_dtype`` reads it off torch.

    The MXFP8 ops take the scale dtype explicitly; passing None silently changes
    how they interpret the scales, so a missing dtype must raise rather than
    propagate.

    torch_npu is imported lazily (and cached) so this module stays importable on
    CUDA/CPU/AMD/XPU CI.
    """
    global _E8M0_DTYPE
    if _E8M0_DTYPE is None:
        from sglang.srt.utils import is_npu

        if is_npu():
            import torch_npu

            _E8M0_DTYPE = getattr(torch_npu, "float8_e8m0fnu", None)
        if _E8M0_DTYPE is None:
            _E8M0_DTYPE = _get_float8_e8m0fnu_dtype()
        if _E8M0_DTYPE is None:
            raise RuntimeError(
                "float8_e8m0fnu dtype not found — MXFP8 MoE requires Ascend A5 "
                "with a torch_npu build exposing float8_e8m0fnu (torch_npu >= 2.9)."
            )
    return _E8M0_DTYPE


def _decode_fp32_mxfp_scale(scale: torch.Tensor) -> torch.Tensor:
    """Recover E8M0 bytes from the two HF float32 scale conventions."""
    # Two e8m0 storage conventions in the fp32 scales: (a) the value
    # IS the e8m0 byte (int in 0..255, MiMo-V2.5-Pro); (b) the value
    # is 2^(e-127) -> recover e from the exponent field. Never
    # re-encode with round(log2)+127: it re-biases the exponent.
    s_flat = scale.detach().float().flatten()
    is_int_like = bool(
        torch.all(s_flat >= 0)
        and torch.all(s_flat <= 255)
        and torch.allclose(s_flat, s_flat.round())
    )
    if is_int_like:
        e8m0 = scale.to(torch.uint8)
    else:
        e8m0 = (scale.view(torch.int32) >> 23 & 0xFF).to(torch.uint8)
    return e8m0


def _dequant_e4m3fn_to_float32(u8: torch.Tensor) -> torch.Tensor:
    """Decode float8_e4m3fn payload bytes to FP32 by explicit bit layout.

    Direct ``.to(torch.float32)`` on NPU fp8 tensors fails on some CANN builds
    (aclnnInplaceCopy error 561103), so decode manually. Only used at
    weight-load time, where the extra elementwise ops are irrelevant.
    """
    bits = u8.view(torch.uint8).to(torch.int32)
    sign = torch.where(bits >= 0x80, -1.0, 1.0)
    e = ((bits >> 3) & 0xF).to(torch.float32)
    m = (bits & 0x7).to(torch.float32)
    mag = torch.where(
        e == 0.0,
        m * (1.0 / 512.0),  # subnormal: m * 2^-9
        (1.0 + m / 8.0) * torch.pow(2.0, e - 7.0),
    )
    # e4m3fn reserves S.1111.111 for NaN (no infinities); max finite is 448.
    mag = torch.where((e == 15.0) & (m == 7.0), float("nan"), mag)
    return sign * mag
