from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.utils.common import (
    get_device_capability,
    is_cuda,
    is_sm100_supported,
)
from sglang.srt.utils.custom_op import register_custom_op_from_extern

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


NVFP4_SF_VEC_SIZE = 16
fp4_quantize = None
try:
    from flashinfer import SfLayout
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize
    from flashinfer import nvfp4_quantize as _flashinfer_nvfp4_quantize

    _flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"

    def _round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def _flashinfer_fp4_quantize_impl(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = NVFP4_SF_VEC_SIZE,
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
        sf_vec_size: int = NVFP4_SF_VEC_SIZE,
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

    def _flashinfer_nvfp4_quantize_per_token_impl(
        input: torch.Tensor,
        global_scale: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not is_sf_swizzled_layout:
            sf_layout = SfLayout.layout_linear
        elif is_sf_8x4_layout:
            sf_layout = SfLayout.layout_8x4
        else:
            sf_layout = SfLayout.layout_128x4
        return _flashinfer_nvfp4_quantize(
            input,
            global_scale,
            sfLayout=sf_layout,
            sf_vec_size=NVFP4_SF_VEC_SIZE,
            enable_pdl=enable_pdl,
            backend=_flashinfer_fp4_quantize_backend,
            per_token_activation=True,
        )

    def _flashinfer_nvfp4_quantize_per_token_fake(
        input: torch.Tensor,
        global_scale: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_q, sf = _flashinfer_fp4_quantize_fake(
            input,
            global_scale,
            sf_vec_size=NVFP4_SF_VEC_SIZE,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            is_sf_8x4_layout=is_sf_8x4_layout,
            enable_pdl=enable_pdl,
        )
        per_token_scale = input.new_empty((input.shape[0],), dtype=torch.float32)
        return x_q, sf, per_token_scale

    _fp4_quantize = register_custom_op_from_extern(
        _flashinfer_fp4_quantize_impl,
        op_name="flashinfer_fp4_quantize",
        fake_impl=_flashinfer_fp4_quantize_fake,
    )
    _nvfp4_quantize_per_token = register_custom_op_from_extern(
        _flashinfer_nvfp4_quantize_per_token_impl,
        op_name="flashinfer_nvfp4_quantize_per_token",
        fake_impl=_flashinfer_nvfp4_quantize_per_token_fake,
    )

    def fp4_quantize(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = NVFP4_SF_VEC_SIZE,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
        per_token_activation: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if per_token_activation:
            if global_scale is None:
                raise ValueError(
                    "Per-token NVFP4 quantization requires a global scale."
                )
            if sf_vec_size != NVFP4_SF_VEC_SIZE or sf_use_ue8m0:
                raise ValueError(
                    "Per-token NVFP4 quantization requires sf_vec_size=16 "
                    "and sf_use_ue8m0=False."
                )
            return _nvfp4_quantize_per_token(
                input,
                global_scale,
                is_sf_swizzled_layout,
                is_sf_8x4_layout,
                enable_pdl,
            )
        return _fp4_quantize(
            input,
            global_scale,
            sf_vec_size,
            sf_use_ue8m0,
            is_sf_swizzled_layout,
            is_sf_8x4_layout,
            enable_pdl,
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
