from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class Fp4GemmRunnerBackend(Enum):
    """Enum for FP4 GEMM runner backend selection."""

    AUTO = "auto"
    CUDNN = "cudnn"
    CUTLASS = "cutlass"
    TRTLLM = "trtllm"

    def is_auto(self) -> bool:
        return self == Fp4GemmRunnerBackend.AUTO

    def is_cudnn(self) -> bool:
        return self == Fp4GemmRunnerBackend.CUDNN

    def is_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.CUTLASS

    def is_trtllm(self) -> bool:
        return self == Fp4GemmRunnerBackend.TRTLLM


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
