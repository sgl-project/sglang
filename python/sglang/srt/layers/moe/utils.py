from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.layers.dp_attention import (
    get_attention_dp_size,
    is_dp_attention_enabled,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MoeA2ABackend(Enum):

    NONE = "none"
    DEEPEP = "deepep"
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    MORI = "mori"
    ASCEND_FUSEEP = "ascend_fuseep"
    FLASHINFER = "flashinfer"
    CUSTOMIZED = "customized"

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

    def is_nixl(self):
        return self == MoeA2ABackend.NIXL

    def is_flashinfer(self):
        return self == MoeA2ABackend.FLASHINFER

    def is_ascend_fuseep(self):
        return self == MoeA2ABackend.ASCEND_FUSEEP

    def is_mori(self):
        return self == MoeA2ABackend.MORI

    def is_customized(self):
        return self == MoeA2ABackend.CUSTOMIZED


class MoeRunnerBackend(Enum):

    AUTO = "auto"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    TRITON_KERNELS = "triton_kernel"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_TRTLLM_ROUTED = "flashinfer_trtllm_routed"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    CUTLASS = "cutlass"
    MARLIN = "marlin"

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

    def is_flashinfer_trtllm_routed(self):
        return self == MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED

    def is_flashinfer_cutlass(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_mxfp4(self):
        return self == MoeRunnerBackend.FLASHINFER_MXFP4

    def is_cutlass(self):
        return self == MoeRunnerBackend.CUTLASS

    def is_marlin(self):
        return self == MoeRunnerBackend.MARLIN


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
SPECULATIVE_MOE_A2A_BACKEND: Optional[MoeA2ABackend] = None
RECORD_NOLORA_GRAPH: bool = False
DEEPEP_MODE: Optional[DeepEPMode] = None
IS_TBO_ENABLED: Optional[bool] = None
IS_SBO_ENABLED: Optional[bool] = None
TBO_TOKEN_DISTRIBUTION_THRESHOLD: Optional[float] = None
DEEPEP_CONFIG: Optional[str] = None
DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER: Optional[bool] = None
MOE_QUANTIZATION: Optional[str] = None


def initialize_moe_config(server_args: ServerArgs):
    global MOE_A2A_BACKEND
    global MOE_RUNNER_BACKEND
    global SPECULATIVE_MOE_RUNNER_BACKEND
    global SPECULATIVE_MOE_A2A_BACKEND
    global RECORD_NOLORA_GRAPH
    global DEEPEP_MODE
    global DEEPEP_CONFIG
    global IS_TBO_ENABLED
    global IS_SBO_ENABLED
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
    global MOE_QUANTIZATION

    MOE_A2A_BACKEND = MoeA2ABackend(server_args.moe_a2a_backend)
    MOE_RUNNER_BACKEND = MoeRunnerBackend(server_args.moe_runner_backend)
    # Dual CUDA graphs only validated for triton MoE backends.
    _triton_ok = MOE_RUNNER_BACKEND in (
        MoeRunnerBackend.TRITON,
        MoeRunnerBackend.TRITON_KERNELS,
    )
    if (
        bool(server_args.record_nolora_graph)
        and bool(server_args.enable_lora)
        and not _triton_ok
    ):
        logger.warning(
            f"record_nolora_graph only validated for triton MoE backend, "
            f"but moe_runner_backend={server_args.moe_runner_backend}. Disabling."
        )
    RECORD_NOLORA_GRAPH = (
        bool(server_args.record_nolora_graph)
        and bool(server_args.enable_lora)
        and _triton_ok
    )
    SPECULATIVE_MOE_RUNNER_BACKEND = (
        MoeRunnerBackend(server_args.speculative_moe_runner_backend)
        if server_args.speculative_moe_runner_backend is not None
        else MOE_RUNNER_BACKEND
    )
    SPECULATIVE_MOE_A2A_BACKEND = (
        MoeA2ABackend(server_args.speculative_moe_a2a_backend)
        if server_args.speculative_moe_a2a_backend is not None
        else MOE_A2A_BACKEND
    )
    DEEPEP_MODE = DeepEPMode(server_args.deepep_mode)
    DEEPEP_CONFIG = server_args.deepep_config or ""
    IS_TBO_ENABLED = server_args.enable_two_batch_overlap
    IS_SBO_ENABLED = server_args.enable_single_batch_overlap
    TBO_TOKEN_DISTRIBUTION_THRESHOLD = server_args.tbo_token_distribution_threshold
    DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = (
        server_args.disable_flashinfer_cutlass_moe_fp4_allgather
    )
    MOE_QUANTIZATION = server_args.quantization


def get_moe_a2a_backend() -> MoeA2ABackend:
    global MOE_A2A_BACKEND
    if MOE_A2A_BACKEND is None:
        MOE_A2A_BACKEND = MoeA2ABackend.NONE
    return MOE_A2A_BACKEND


def get_moe_runner_backend() -> MoeRunnerBackend:
    global MOE_RUNNER_BACKEND
    if MOE_RUNNER_BACKEND is None:
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


def get_speculative_moe_a2a_backend() -> MoeA2ABackend:
    global SPECULATIVE_MOE_A2A_BACKEND
    if SPECULATIVE_MOE_A2A_BACKEND is None:
        logger.warning(
            "SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend"
        )
        SPECULATIVE_MOE_A2A_BACKEND = MoeA2ABackend.NONE
    return SPECULATIVE_MOE_A2A_BACKEND


def should_record_nolora_graph() -> bool:
    return RECORD_NOLORA_GRAPH


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


def is_deepep_class_backend() -> bool:
    """Check if the MoE backend is DeepEP-family (DeepEP, Mooncake, or Mori)."""
    b = get_moe_a2a_backend()
    return b.is_deepep() or b.is_mooncake() or b.is_mori()


def is_flashinfer_cutedsl_v1_path() -> bool:
    """CuteDSL v1 + DeepEP low-latency path (no MoeRunner, no autotune)."""
    return (
        get_moe_runner_backend().is_flashinfer_cutedsl()
        and get_moe_a2a_backend().is_deepep()
    )


def get_tbo_token_distribution_threshold() -> float:
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    if TBO_TOKEN_DISTRIBUTION_THRESHOLD is None:
        logger.warning(
            "TBO_TOKEN_DISTRIBUTION_THRESHOLD is not initialized, using 0.48"
        )
        TBO_TOKEN_DISTRIBUTION_THRESHOLD = 0.48
    return TBO_TOKEN_DISTRIBUTION_THRESHOLD


def filter_moe_weight_param_global_expert(name, x, num_local_experts):
    """
    Filter out for MoE expert parameters that requires global expert.
    """
    return (
        not getattr(x, "_sglang_require_global_experts", False)
        and x.data.ndim > 0
        and x.data.shape[0] == num_local_experts
    )


def should_use_flashinfer_cutlass_moe_fp4_allgather():
    """
    Perform FP4 quantize before all-gather for flashinfer cutlass moe to reduce communication cost for high-throughput serving.
    """
    return (
        not DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
        and get_moe_a2a_backend().is_none()
        and get_moe_runner_backend().is_flashinfer_cutlass()
        and is_dp_attention_enabled()
        and MOE_QUANTIZATION == "modelopt_fp4"
        and get_moe_expert_parallel_world_size() == get_attention_dp_size()
    )


def should_use_dp_reduce_scatterv():
    """
    Use reduce_scatterv in the standard dispatcher's combine() for DP attention
    with EP, replacing the default all-reduce + dp_scatter path.
    Only changes the combine (post-kernel) communication; dispatch is unchanged.
    """
    return (
        not should_use_flashinfer_cutlass_moe_fp4_allgather()
        and get_moe_a2a_backend().is_none()
        and is_dp_attention_enabled()
        and get_attention_dp_size() > 1
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


@contextmanager
def speculative_moe_a2a_backend_context():
    """
    Context manager to temporarily use the speculative MoE A2A backend for draft model operations.
    This ensures that draft models in speculative decoding use the configured speculative A2A backend.
    """
    global MOE_A2A_BACKEND
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
    original_backend = MOE_A2A_BACKEND
    original_disable_flashinfer_cutlass_moe_fp4_allgather = (
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER
    )
    try:
        MOE_A2A_BACKEND = get_speculative_moe_a2a_backend()
        # Disable FP4 allgather for spec decode since MTP layers are unquantized
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = True
        yield
    finally:
        MOE_A2A_BACKEND = original_backend
        DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = (
            original_disable_flashinfer_cutlass_moe_fp4_allgather
        )


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


AITER_PADDING_SIZE = 128
TRITON_PADDING_SIZE = 128


# Unit of padding - context dependent
def get_moe_padding_size(is_aiter_moe):
    if is_aiter_moe:
        return AITER_PADDING_SIZE
    else:
        return (
            TRITON_PADDING_SIZE
            if bool(int(os.getenv("SGLANG_MOE_PADDING", "0")))
            else 0
        )


def get_moe_weight_sizes(inter_dim, is_concat, is_packed, is_aiter_moe):
    """
    Calculate dimensions for MoE weight tensors.

    Args:
        inter_dim: Base intermediate dimension.
        is_concat: If True, fusions W1 (gate) and W3 (up) projections.
        is_packed: If True, uses 4-bit quantization (two FP4 elements per byte).
        is_aiter_moe: If True, applies Aiter-specific kernel padding alignment.
    """
    # w2_down_dim is the packing rank, but w13_up_dim not (of matrix to matmul)
    w13_up_dim = 2 * inter_dim if is_concat else inter_dim
    w2_down_dim = inter_dim // 2 if is_packed else inter_dim

    if is_aiter_moe:
        padding_size = get_moe_padding_size(True)
        align_aiter = lambda n: ((n + padding_size - 1) // padding_size) * padding_size
        is_padded = (w2_down_dim % padding_size) > 0
        if is_padded:
            # w2_down_dim, padding & aligned, unit: parameter dtype
            w2_down_dim = align_aiter(w2_down_dim)
        # up proj + gate fusion : 2x
        if is_concat:
            w13_up_dim = w2_down_dim * 2
        # packed
        if hasattr(torch, "float4_e2m1fn_x2") and is_packed:
            # w13_up_dim (row rank of matmul matrix) is not packing dim, *2 to recover
            w13_up_dim *= 2

    return (w13_up_dim, w2_down_dim, False if not is_aiter_moe else is_padded)
