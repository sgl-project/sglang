from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

NVFP4_E2M1_MAX = 6.0
NVFP4_FP8_E4M3_MAX = 448.0


class Fp4GemmRunnerBackend(Enum):
    """Enum for FP4 GEMM runner backend selection."""

    AUTO = "auto"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"

    def is_auto(self) -> bool:
        return self == Fp4GemmRunnerBackend.AUTO

    def is_flashinfer_cudnn(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUDNN

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_TRTLLM

    def get_flashinfer_backend(self) -> str:
        """Get the backend string to pass to FlashInfer's mm_fp4 API.

        This remaps SGLang's user-facing backend names to FlashInfer's API names.
        Examples:
            'flashinfer_trtllm' -> 'trtllm'
            'flashinfer_cutlass' -> 'cutlass'
            'flashinfer_cudnn' -> 'cudnn'
        """
        if self.value.startswith("flashinfer_"):
            return self.value.removeprefix("flashinfer_")
        else:
            return self.value


FP4_GEMM_RUNNER_BACKEND: Fp4GemmRunnerBackend | None = None


def initialize_fp4_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP4 GEMM configuration from server args."""
    global FP4_GEMM_RUNNER_BACKEND

    backend = server_args.fp4_gemm_runner_backend

    # Handle deprecated env var for backward compatibility
    # TODO: Remove this in a future version
    if envs.SGLANG_FLASHINFER_FP4_GEMM_BACKEND.is_set():
        env_backend = envs.SGLANG_FLASHINFER_FP4_GEMM_BACKEND.get()
        if backend == "auto":
            logger.warning(
                "SGLANG_FLASHINFER_FP4_GEMM_BACKEND is deprecated. "
                f"Please use '--fp4-gemm-backend={env_backend}' instead."
            )
            if not env_backend.startswith("flashinfer_"):
                env_backend = "flashinfer_" + env_backend
            backend = env_backend
        else:
            logger.warning(
                f"FP4 GEMM backend set to '{backend}' via --fp4-gemm-backend overrides "
                "environment variable SGLANG_FLASHINFER_FP4_GEMM_BACKEND. "
                "Using server argument value."
            )

    FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)


def get_fp4_gemm_runner_backend() -> Fp4GemmRunnerBackend:
    """Get the current FP4 GEMM runner backend."""
    global FP4_GEMM_RUNNER_BACKEND
    if FP4_GEMM_RUNNER_BACKEND is None:
        FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.AUTO
    return FP4_GEMM_RUNNER_BACKEND


def nvfp4_online_scale_enabled() -> bool:
    return envs.SGLANG_NVFP4_ONLINE_SCALE.get()


def nvfp4_compute_input_scale_and_inv(
    x: torch.Tensor,
    input_scale: torch.Tensor,
    input_scale_inv: torch.Tensor,
) -> None:
    if x.numel() == 0:
        input_scale.zero_()
        input_scale_inv.zero_()
        return

    global_amax = x.abs().max().to(torch.float32)
    input_scale_inv.copy_(global_amax)
    input_scale_inv.reciprocal_()
    input_scale_inv.mul_(NVFP4_FP8_E4M3_MAX * NVFP4_E2M1_MAX)
    input_scale_inv.clamp_(max=torch.finfo(torch.float32).max)
    input_scale_inv.masked_fill_(input_scale_inv == 0.0, 1.0)
    input_scale.copy_(input_scale_inv)
    input_scale.reciprocal_()
