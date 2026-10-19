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


class _FusedNormFP4Quant(BaseFusedOp):
    """Backend dispatcher for the Muse Spark fused norm/quant boundary."""

    op = "muse_spark.norm_fp4_quant"
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

        from sglang.kernels.ops.muse_spark.helion_runtime import helion_supported

        return (
            helion_supported("norm_fp4_quant", x)
            and x.shape[0] > 0
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
        raise NotImplementedError("norm_fp4_quant has no device-independent backend")

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
        from sglang.kernels.ops.muse_spark.helion_norm_fp4_quant import (
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
