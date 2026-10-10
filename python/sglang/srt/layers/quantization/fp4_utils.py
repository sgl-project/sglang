from __future__ import annotations

import logging
import warnings
from enum import Enum
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_exec,
    get_platform,
)
from sglang.srt.utils.common import (
    get_device_capability,
    is_cuda,
    is_gfx95_supported,
    is_hip,
)
from sglang.srt.utils.custom_op import register_custom_op_from_extern

logger = logging.getLogger(__name__)


fp4_quantize = None
try:
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize

    _flashinfer_fp4_quantize_backend = "cute-dsl" if get_platform().is_sm100 else "cuda"

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
    AITER = "aiter"
    TRITON = "triton"

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

    def is_aiter(self) -> bool:
        return self == Fp4GemmRunnerBackend.AITER

    def is_triton(self) -> bool:
        return self == Fp4GemmRunnerBackend.TRITON

    def is_flashinfer(self) -> bool:
        return self.value.startswith("flashinfer_")

    def supports_swiglu_fusion(self) -> bool:
        """Whether weight preparation builds the fused SwiGLU layout."""
        return not (self.is_marlin() or self.is_flashinfer_trtllm())

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


_ROCM_ONLY_FP4_GEMM_BACKENDS = ("aiter", "triton")
_LEGACY_AITER_FP4_ASM_GEMM_ENV = "SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM"


def _warn_legacy_aiter_fp4_asm_gemm_env(message: str) -> None:
    # DeprecationWarning is hidden by Python's default filters, so also log it.
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    logger.warning(message)


def resolve_rocm_fp4_gemm_backend(
    *,
    backend: str,
    is_gfx95: bool,
    legacy_use_aiter_asm: Optional[bool],
) -> str:
    """Resolve --fp4-gemm-backend for ROCm.

    ``legacy_use_aiter_asm`` is the value of the deprecated
    SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM, or None when it is unset.
    """
    if legacy_use_aiter_asm is not None:
        message = (
            f"Environment variable {_LEGACY_AITER_FP4_ASM_GEMM_ENV} is deprecated "
            "and will be removed in a future release. Please use "
            "'--fp4-gemm-backend aiter' (or 'triton') instead."
        )
        if backend != "auto":
            _warn_legacy_aiter_fp4_asm_gemm_env(
                f"{message} It is ignored because --fp4-gemm-backend={backend} "
                "is set explicitly."
            )
        else:
            _warn_legacy_aiter_fp4_asm_gemm_env(message)
            # The env var was a silent no-op off gfx95; keep it that way.
            return "aiter" if legacy_use_aiter_asm and is_gfx95 else "triton"

    if backend == "auto":
        # AITER stays opt-in: its preshuffled weights cannot serve the Quark
        # tuple-input fusion paths (e.g. DeepSeek MLA projections) yet.
        return "triton"
    if backend == "aiter" and not is_gfx95:
        raise ValueError(
            "--fp4-gemm-backend=aiter requires an AMD gfx95 GPU (MI350/MI355). "
            "Use --fp4-gemm-backend triton instead."
        )
    return backend


def initialize_fp4_gemm_config() -> None:
    """Initialize the FP4 GEMM backend from the published configuration."""
    global FP4_GEMM_RUNNER_BACKEND

    backend = get_exec().kernel.fp4_gemm_runner_backend
    if is_hip():
        legacy_env = envs.SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM
        backend = resolve_rocm_fp4_gemm_backend(
            backend=backend,
            is_gfx95=is_gfx95_supported(),
            legacy_use_aiter_asm=legacy_env.get() if legacy_env.is_set() else None,
        )
    elif backend in _ROCM_ONLY_FP4_GEMM_BACKENDS:
        raise ValueError(f"--fp4-gemm-backend={backend} is only supported on ROCm.")
    elif backend == "auto":
        if get_platform().is_sm100:
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
