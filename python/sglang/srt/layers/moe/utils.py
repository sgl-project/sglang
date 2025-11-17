from __future__ import annotations

import logging
from contextlib import contextmanager
from enum import Enum, IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.layers.dp_attention import (
    get_attention_dp_size,
    is_dp_attention_enabled,
)
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MoeA2ABackend(Enum):

    NONE = "none"
    DEEPEP = "deepep"
    MOONCAKE = "mooncake"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        for member in cls:
            if value == member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_none(self):
        return self == MoeA2ABackend.NONE

    def is_deepep(self):
        return self == MoeA2ABackend.DEEPEP

    def is_mooncake(self):
        return self == MoeA2ABackend.MOONCAKE


class MoeRunnerBackend(Enum):

    AUTO = "auto"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    TRITON_KERNELS = "triton_kernel"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    CUTLASS = "cutlass"

    def is_auto(self):
        return self == MoeRunnerBackend.AUTO

    def is_deep_gemm(self):
        return self == MoeRunnerBackend.DEEP_GEMM

    def is_triton(self):
        return self == MoeRunnerBackend.TRITON

    def is_triton_kernels(self):
        return self == MoeRunnerBackend.TRITON_KERNELS

    def is_flashinfer_trtllm(self):
        return self == MoeRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutlass(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_mxfp4(self):
        return self == MoeRunnerBackend.FLASHINFER_MXFP4

    def is_cutlass(self):
        return self == MoeRunnerBackend.CUTLASS


class DeepEPMode(Enum):

    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self) -> bool:
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self) -> bool:
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY

    def is_normal(self) -> bool:
        return self == DeepEPMode.NORMAL

    def is_low_latency(self) -> bool:
        return self == DeepEPMode.LOW_LATENCY

    def is_auto(self) -> bool:
        return self == DeepEPMode.AUTO


MOE_A2A_BACKEND: Optional[MoeA2ABackend] = None
MOE_RUNNER_BACKEND: Optional[MoeRunnerBackend] = None
SPECULATIVE_MOE_RUNNER_BACKEND: Optional[MoeRunnerBackend] = None
DEEPEP_MODE: Optional[DeepEPMode] = None
IS_TBO_ENABLED: Optional[bool] = None
IS_SBO_ENABLED: Optional[bool] = None
TBO_TOKEN_DISTRIBUTION_THRESHOLD: Optional[float] = None
DEEPEP_CONFIG: Optional[str] = None
DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER: Optional[bool] = None


def initialize_moe_config(server_args: ServerArgs):
    global MOE_A2A_BACKEND
    global MOE_RUNNER_BACKEND
    global SPECULATIVE_MOE_RUNNER_BACKEND
    global DEEPEP_MODE
    global DEEPEP_CONFIG
    global IS_TBO_ENABLED
    global IS_SBO_ENABLED
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER

    MOE_A2A_BACKEND = MoeA2ABackend(server_args.moe_a2a_backend)
    MOE_RUNNER_BACKEND = MoeRunnerBackend(server_args.moe_runner_backend)
    SPECULATIVE_MOE_RUNNER_BACKEND = (
        MoeRunnerBackend(server_args.speculative_moe_runner_backend)
        if server_args.speculative_moe_runner_backend is not None
        else MOE_RUNNER_BACKEND
    )
    DEEPEP_MODE = DeepEPMode(server_args.deepep_mode)
    DEEPEP_CONFIG = server_args.deepep_config or ""
    IS_TBO_ENABLED = server_args.enable_two_batch_overlap
    IS_SBO_ENABLED = server_args.enable_single_batch_overlap
    TBO_TOKEN_DISTRIBUTION_THRESHOLD = server_args.tbo_token_distribution_threshold
    DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = (
        server_args.disable_flashinfer_cutlass_moe_fp4_allgather
    )


def get_moe_a2a_backend() -> MoeA2ABackend:
    global MOE_A2A_BACKEND
    if MOE_A2A_BACKEND is None:
        MOE_A2A_BACKEND = MoeA2ABackend.NONE
    return MOE_A2A_BACKEND


def get_moe_runner_backend() -> MoeRunnerBackend:
    global MOE_RUNNER_BACKEND
    if MOE_RUNNER_BACKEND is None:
        log_info_on_rank0(
            logger,
            "MOE_RUNNER_BACKEND is not initialized, the backend will be automatically selected",
        )
        MOE_RUNNER_BACKEND = MoeRunnerBackend.AUTO
    return MOE_RUNNER_BACKEND


def get_speculative_moe_runner_backend() -> MoeRunnerBackend:
    global SPECULATIVE_MOE_RUNNER_BACKEND
    if SPECULATIVE_MOE_RUNNER_BACKEND is None:
        logger.warning(
            "SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend"
        )
        SPECULATIVE_MOE_RUNNER_BACKEND = MoeRunnerBackend.AUTO
    return SPECULATIVE_MOE_RUNNER_BACKEND


def get_deepep_mode() -> DeepEPMode:
    global DEEPEP_MODE
    if DEEPEP_MODE is None:
        logger.warning("DEEPEP_MODE is not initialized, using auto mode")
        DEEPEP_MODE = DeepEPMode.AUTO
    return DEEPEP_MODE


def get_deepep_config() -> str:
    global DEEPEP_CONFIG
    if DEEPEP_CONFIG is None:
        logger.warning("DEEPEP_CONFIG is not initialized, using default config")
        DEEPEP_CONFIG = ""
    return DEEPEP_CONFIG


def is_tbo_enabled() -> bool:
    global IS_TBO_ENABLED
    if IS_TBO_ENABLED is None:
        IS_TBO_ENABLED = False
    return IS_TBO_ENABLED


def is_sbo_enabled() -> bool:
    global IS_SBO_ENABLED
    if IS_SBO_ENABLED is None:
        IS_SBO_ENABLED = False
    return IS_SBO_ENABLED


def get_tbo_token_distribution_threshold() -> float:
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    if TBO_TOKEN_DISTRIBUTION_THRESHOLD is None:
        logger.warning(
            "TBO_TOKEN_DISTRIBUTION_THRESHOLD is not initialized, using 0.48"
        )
        TBO_TOKEN_DISTRIBUTION_THRESHOLD = 0.48
    return TBO_TOKEN_DISTRIBUTION_THRESHOLD


@lru_cache(maxsize=1)
def should_use_flashinfer_cutlass_moe_fp4_allgather():
    """
    Perform FP4 quantize before all-gather for flashinfer cutlass moe to reduce communication cost for high-throughput serving.
    """
    return (
        not DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
        and get_moe_runner_backend().is_flashinfer_cutlass()
        and is_dp_attention_enabled()
        and get_moe_expert_parallel_world_size() == get_attention_dp_size()
    )


@contextmanager
def speculative_moe_backend_context():
    """
    Context manager to temporarily use the speculative MoE backend for draft model operations.
    This ensures that draft models in speculative decoding use the configured speculative backend.
    """
    global MOE_RUNNER_BACKEND
    original_backend = MOE_RUNNER_BACKEND
    try:
        MOE_RUNNER_BACKEND = get_speculative_moe_runner_backend()
        yield
    finally:
        MOE_RUNNER_BACKEND = original_backend


# The type of method in top-K routing, for use in torch custom op
# Please keep this in sync with the counterpart defined in https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/trtllm/fused_moe/runner.h
class RoutingMethodType(IntEnum):
    # Default: Softmax -> TopK
    Default = (0,)
    # Renormalize: TopK -> Softmax
    Renormalize = (1,)
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups
    DeepSeekV3 = (2,)
    # Llama4: Top1 -> Sigmoid
    Llama4 = (3,)
    # Qwen3: Softmax -> TopK -> Renormalize
    RenormalizeNaive = (4,)
    # TopK only (no softmax)
    TopK = (5,)
    # Unspecified
    Unspecified = 6
