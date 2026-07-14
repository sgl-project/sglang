from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.spec import KernelBackend
from sglang.srt.utils.common import (
    get_device_capability,
    is_cuda,
    is_sm100_supported,
)
from sglang.srt.utils.custom_op import register_custom_op_from_extern

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


fp4_quantize = None
try:
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize

    _flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"

    def _round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def _flashinfer_fp4_quantize_impl(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _flashinfer_fp4_quantize(
            input=input,
            global_scale=global_scale,
            sf_vec_size=sf_vec_size,
            sf_use_ue8m0=sf_use_ue8m0,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            is_sf_8x4_layout=is_sf_8x4_layout,
            enable_pdl=enable_pdl,
            backend=_flashinfer_fp4_quantize_backend,
        )

    def _flashinfer_fp4_quantize_fake(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_column_major = input.stride(-2) == 1
        if is_column_major:
            m = input.shape[-1]
            K = input.shape[-2]
        else:
            m = input.numel() // input.shape[-1]
            K = input.shape[-1]
        if is_column_major:
            x_q = input.new_empty((*input.shape[:-2], K // 2, m), dtype=torch.uint8)
        else:
            x_q = input.new_empty((*input.shape[:-1], K // 2), dtype=torch.uint8)
        if is_sf_swizzled_layout:
            row_size = 8 if is_sf_8x4_layout else 128
            sf_rows = _round_up(m, row_size)
            sf_cols = _round_up(K // sf_vec_size, 4)
        else:
            sf_rows = m
            sf_cols = K // sf_vec_size
        if is_column_major:
            sf = input.new_empty((sf_cols, sf_rows), dtype=torch.uint8)
        else:
            sf = input.new_empty((sf_rows, sf_cols), dtype=torch.uint8)
        return x_q, sf

    fp4_quantize = register_custom_op_from_extern(
        _flashinfer_fp4_quantize_impl,
        op_name="flashinfer_fp4_quantize",
        fake_impl=_flashinfer_fp4_quantize_fake,
    )
except ImportError:
    fp4_quantize = None


class Fp4GemmRunnerBackend(Enum):
    """Enum for FP4 GEMM runner backend selection."""

    AUTO = "auto"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    MARLIN = "marlin"

    def is_auto(self) -> bool:
        return self == Fp4GemmRunnerBackend.AUTO

    def is_flashinfer_cudnn(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUDNN

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutedsl(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL

    def is_marlin(self) -> bool:
        return self == Fp4GemmRunnerBackend.MARLIN

    def is_flashinfer(self) -> bool:
        return self.value.startswith("flashinfer_")

    def get_flashinfer_backend(self) -> str:
        """Get the backend string to pass to FlashInfer's mm_fp4 API.

        This remaps SGLang's user-facing backend names to FlashInfer's API names.
        Examples:
            'flashinfer_trtllm' -> 'trtllm'
            'flashinfer_cutlass' -> 'cutlass'
            'flashinfer_cudnn' -> 'cudnn'
            'flashinfer_cutedsl' -> 'cute-dsl'
        """
        if self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL:
            return "cute-dsl"
        if self.value.startswith("flashinfer_"):
            return self.value.removeprefix("flashinfer_")
        else:
            return self.value


FP4_GEMM_RUNNER_BACKEND: Fp4GemmRunnerBackend | None = None


def initialize_fp4_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP4 GEMM configuration from server args."""
    global FP4_GEMM_RUNNER_BACKEND

    backend = server_args.fp4_gemm_runner_backend
    if backend == "auto":
        if is_sm100_supported():
            backend = "flashinfer_cutedsl"
        elif is_cuda() and (10, 0) > get_device_capability() >= (8, 0):
            backend = "marlin"
        else:
            backend = "flashinfer_cutlass"

    FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)


def get_fp4_gemm_runner_backend() -> Fp4GemmRunnerBackend:
    """Get the current FP4 GEMM runner backend."""
    global FP4_GEMM_RUNNER_BACKEND
    if FP4_GEMM_RUNNER_BACKEND is None:
        FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.AUTO
    return FP4_GEMM_RUNNER_BACKEND


try:
    from flashinfer import mm_fp4 as flashinfer_fp4_gemm

    enable_flashinfer_fp4_gemm = True
except ImportError:
    flashinfer_fp4_gemm = None
    enable_flashinfer_fp4_gemm = False


class Fp4GemmOp(BaseFusedOp):
    """NVFP4 dense GEMM: one FlashInfer ``mm_fp4`` sub-kernel per
    ``--fp4-gemm-backend`` value. The Marlin (weight-only) fallback is a
    structurally different, dequant-based code path (`apply_fp4_marlin_linear`
    in `marlin_utils_fp4.py`) selected upstream at the call site via
    `get_fp4_gemm_runner_backend().is_marlin()`, so it is not a backend of
    this op.
    """

    op = "gemm.nvfp4"
    priority = (
        KernelBackend.FLASHINFER_CUTEDSL,
        KernelBackend.FLASHINFER_CUTLASS,
        KernelBackend.FLASHINFER_TRTLLM,
        KernelBackend.FLASHINFER_CUDNN,
    )
    descriptions = {
        KernelBackend.FLASHINFER_CUTEDSL: "FlashInfer CuTe DSL NVFP4 GEMM (SM100).",
        KernelBackend.FLASHINFER_CUTLASS: "FlashInfer CUTLASS NVFP4 GEMM.",
        KernelBackend.FLASHINFER_TRTLLM: "FlashInfer TRTLLM NVFP4 GEMM.",
        KernelBackend.FLASHINFER_CUDNN: "FlashInfer cuDNN NVFP4 GEMM.",
    }

    def backend_eligible(self, backend: KernelBackend, *args, **kwargs) -> bool:
        return enable_flashinfer_fp4_gemm

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_sf: torch.Tensor,
        weight_sf: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
        out_features: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "gemm.nvfp4: no pure-torch reference; NVFP4 GEMM requires flashinfer's "
            "mm_fp4 (or the Marlin weight-only fallback, selected separately)."
        )

    def _call_flashinfer(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_sf: torch.Tensor,
        weight_sf: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
        out_features: int,
        backend: str,
    ) -> torch.Tensor:
        # `out_features` only feeds the fake-mode shape inference on the
        # `fp4_gemm` custom op (see modelopt_quant.py); flashinfer's `mm_fp4`
        # does not take it.
        return flashinfer_fp4_gemm(
            input, weight, input_sf, weight_sf, alpha, out_dtype, backend=backend
        )

    def forward_flashinfer_cutedsl(self, *args, **kwargs) -> torch.Tensor:
        return self._call_flashinfer(*args, **kwargs, backend="cute-dsl")

    def forward_flashinfer_cutlass(self, *args, **kwargs) -> torch.Tensor:
        return self._call_flashinfer(*args, **kwargs, backend="cutlass")

    def forward_flashinfer_trtllm(self, *args, **kwargs) -> torch.Tensor:
        return self._call_flashinfer(*args, **kwargs, backend="trtllm")

    def forward_flashinfer_cudnn(self, *args, **kwargs) -> torch.Tensor:
        return self._call_flashinfer(*args, **kwargs, backend="cudnn")


_FP4_GEMM = Fp4GemmOp()

_FP4_RUNNER_TO_KERNEL_BACKEND = {
    Fp4GemmRunnerBackend.FLASHINFER_CUDNN: KernelBackend.FLASHINFER_CUDNN,
    Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL: KernelBackend.FLASHINFER_CUTEDSL,
    Fp4GemmRunnerBackend.FLASHINFER_CUTLASS: KernelBackend.FLASHINFER_CUTLASS,
    Fp4GemmRunnerBackend.FLASHINFER_TRTLLM: KernelBackend.FLASHINFER_TRTLLM,
}


def dispatch_fp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    """Dispatch a dense NVFP4 GEMM to the FlashInfer sub-kernel selected by
    ``--fp4-gemm-backend`` (via `Fp4GemmOp`, a `sglang.kernels.fused_op.BaseFusedOp`).
    """
    if not enable_flashinfer_fp4_gemm:
        raise RuntimeError(
            "NVFP4 GEMM requires flashinfer's mm_fp4; please install flashinfer."
        )
    fp4_backend = get_fp4_gemm_runner_backend()
    kernel_backend = _FP4_RUNNER_TO_KERNEL_BACKEND.get(fp4_backend)
    if kernel_backend is None:
        raise ValueError(
            f"fp4_gemm: unsupported --fp4-gemm-backend={fp4_backend} "
            "(expected one of the flashinfer_* backends)"
        )
    return _FP4_GEMM.forward(
        input,
        weight,
        input_sf,
        weight_sf,
        alpha,
        out_dtype,
        out_features,
        backend=kernel_backend,
    )
