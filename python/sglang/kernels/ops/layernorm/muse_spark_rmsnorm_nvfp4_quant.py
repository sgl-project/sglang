"""Fused RMS norm + per-token NVFP4 quantization for MuseSpark v1.2.

Byte-exact with FlashInfer's per-token ``layout_linear`` CuTe-DSL quantizer on
the same bf16 row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.spec import CapabilityRequirement, KernelBackend

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_BLOCK = 16
_MAX_THREADS = 1024


def _num_threads(hidden_size: int) -> int:
    blocks = hidden_size // _BLOCK
    if hidden_size % _BLOCK or blocks % 32:
        raise RuntimeError(
            f"muse_spark norm_fp4_quant: hidden_size={hidden_size} must be a multiple of 512"
        )
    for threads in (256, 512, _MAX_THREADS, 128, 64, 32):
        if blocks % threads == 0:
            return threads
    raise RuntimeError(
        f"muse_spark norm_fp4_quant: unsupported hidden_size={hidden_size}"
    )


@cache_once
def _jit_module(hidden_size: int, num_threads: int) -> Module:
    args = make_cpp_args(hidden_size, num_threads, is_arch_support_pdl())
    return load_jit(
        "muse_spark_norm_fp4_quant",
        *args,
        cuda_files=["muse_spark/norm_fp4_quant.cuh"],
        cuda_wrappers=[
            ("run_quant", f"MuseSparkNormFp4Quant<{args}, true, false>::run"),
            ("run_quant_normed", f"MuseSparkNormFp4Quant<{args}, true, true>::run"),
            ("run_normed", f"MuseSparkNormFp4Quant<{args}, false, true>::run"),
        ],
    )


def _fused_norm_fp4_quant_jit(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    global_scale_inv: float,
    *,
    quantize: bool = True,
    want_normed: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Return ``(fp4, sf, per_token_scale, normed)`` for a bf16 ``[M, K]`` batch.

    ``fp4`` is ``[M, K/2]`` uint8, ``sf`` is ``[M, K/16]`` uint8 (e4m3 bits,
    row-major), ``per_token_scale`` is ``[M]`` fp32. Entries not requested are
    ``None``. ``weight`` is the already-resolved effective norm weight.
    """
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()
    assert quantize or want_normed
    m, hidden_size = x.shape
    if weight is not None:
        assert weight.dtype == torch.bfloat16 and weight.shape == (hidden_size,)
        weight = weight.contiguous()

    fp4 = sf = pts = normed = None
    if quantize:
        fp4 = torch.empty(m, hidden_size // 2, dtype=torch.uint8, device=x.device)
        sf = torch.empty(m, hidden_size // _BLOCK, dtype=torch.uint8, device=x.device)
        pts = torch.empty(m, dtype=torch.float32, device=x.device)
    if want_normed:
        normed = torch.empty_like(x)
    if m == 0:
        return fp4, sf, pts, normed

    module = _jit_module(hidden_size, _num_threads(hidden_size))
    if quantize and want_normed:
        fn = module.run_quant_normed
    elif quantize:
        fn = module.run_quant
    else:
        fn = module.run_normed
    fn(x, weight, normed, fp4, sf, pts, float(eps), float(global_scale_inv))
    return fp4, sf, pts, normed


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even, saturating E2M1 nibble (sign in bit 3) of fp32 ``x``."""
    ax = x.abs()
    # Midpoints between 0, .5, 1, 1.5, 2, 3, 4, 6; ties go to the even code.
    magnitude = (
        (ax > 0.25).to(torch.uint8)
        + (ax >= 0.75)
        + (ax > 1.25)
        + (ax >= 1.75)
        + (ax > 2.5)
        + (ax >= 3.5)
        + (ax > 5.0)
    )
    return magnitude | (torch.signbit(x).to(torch.uint8) << 3)


def _nvfp4_quantize_per_token_native(
    rows: torch.Tensor, global_scale_inv: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(fp4, sf, per_token_scale)`` of bf16 ``rows`` [M, K], as the kernel lays them out."""
    m, hidden_size = rows.shape
    r = rows.float()
    amax = r.abs().amax(-1)
    nonzero = amax != 0
    per_token_scale = torch.where(nonzero, amax * global_scale_inv, 0.0)
    encode_scale = torch.where(
        nonzero, 1.0 / per_token_scale, torch.finfo(torch.float32).max
    )

    blocks = r.view(m, hidden_size // _BLOCK, _BLOCK)
    block_max = blocks.abs().amax(-1)
    scale = encode_scale[:, None] * (block_max * (1.0 / 6.0))
    sf = scale.clamp(max=torch.finfo(torch.float8_e4m3fn).max).to(torch.float8_e4m3fn)
    sf_f32 = sf.float()
    out_scale = torch.where(
        sf_f32 == 0, 0.0, 1.0 / (sf_f32 * (1.0 / encode_scale[:, None]))
    )
    codes = _e2m1_codes(blocks * out_scale[..., None]).view(m, hidden_size)
    fp4 = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return fp4, sf.view(torch.uint8), per_token_scale


class _FusedNormFP4Quant(BaseFusedOp):
    """Backend dispatcher for the Muse Spark fused norm/quant boundary."""

    op = "layernorm.muse_spark_rmsnorm_nvfp4_quant"
    priority = (KernelBackend.JIT, KernelBackend.HELION)
    capabilities = {
        KernelBackend.HELION: {CapabilityRequirement.CUDA},
        KernelBackend.JIT: {CapabilityRequirement.CUDA},
    }

    def backend_eligible(
        self,
        backend: KernelBackend,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        eps: float,
        global_scale_inv: float,
        *,
        quantize: bool = True,
        want_normed: bool = False,
    ) -> bool:
        if not super().backend_eligible(
            backend,
            x,
            weight,
            eps,
            global_scale_inv,
            quantize=quantize,
            want_normed=want_normed,
        ):
            return False
        if backend is not KernelBackend.HELION:
            return True

        from sglang.kernels.jit.utils.helion_runtime import (
            base_package_version,
            helion_supported,
            warn_helion_unsupported,
        )

        if not helion_supported(x):
            return False
        flashinfer_version = base_package_version("flashinfer-python")
        if flashinfer_version != "0.6.17":
            return warn_helion_unsupported(
                "NVFP4 quantization requires FlashInfer 0.6.17, found "
                f"{flashinfer_version or 'not installed'}"
            )
        return (
            x.shape[0] > 0
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.is_contiguous()
            and x.shape[1] == 4096
            and quantize
            and not want_normed
            and weight is not None
            and weight.dtype == torch.bfloat16
            and weight.shape == (x.shape[1],)
        )

    def forward_native(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        eps: float,
        global_scale_inv: float,
        *,
        quantize: bool = True,
        want_normed: bool = False,
    ):
        """Torch reference for the JIT kernel's outputs.

        Same formula and rounding points as the kernel, but IEEE reciprocals
        where it uses ``rcp.approx``, so a value on a rounding boundary can land
        one fp8/fp4 code apart.
        """
        assert quantize or want_normed
        xf = x.float()
        normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            normed = normed * weight.float()
        normed = normed.to(torch.bfloat16)
        if not quantize:
            return None, None, None, normed
        fp4, sf, per_token_scale = _nvfp4_quantize_per_token_native(
            normed, global_scale_inv
        )
        return fp4, sf, per_token_scale, normed if want_normed else None

    def forward_jit(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        eps: float,
        global_scale_inv: float,
        *,
        quantize: bool = True,
        want_normed: bool = False,
    ):
        return _fused_norm_fp4_quant_jit(
            x,
            weight,
            eps,
            global_scale_inv,
            quantize=quantize,
            want_normed=want_normed,
        )

    def forward_helion(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        eps: float,
        global_scale_inv: float,
        *,
        quantize: bool = True,
        want_normed: bool = False,
    ):
        from sglang.kernels.ops.layernorm.muse_spark_rmsnorm_nvfp4_quant_helion import (
            fused_norm_fp4_quant_helion,
        )

        return fused_norm_fp4_quant_helion(
            x,
            weight,
            eps,
            global_scale_inv,
            quantize=quantize,
            want_normed=want_normed,
        )


_FUSED_NORM_FP4_QUANT = _FusedNormFP4Quant()


def fused_norm_fp4_quant(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    global_scale_inv: float,
    *,
    quantize: bool = True,
    want_normed: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Run the selected backend at the existing fused operation boundary."""
    return _FUSED_NORM_FP4_QUANT(
        x,
        weight,
        eps,
        global_scale_inv,
        quantize=quantize,
        want_normed=want_normed,
    )
