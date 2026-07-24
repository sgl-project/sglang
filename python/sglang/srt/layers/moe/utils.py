from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.runtime_context import get_flags, get_forward, get_parallel
from sglang.srt.utils import is_cuda, is_npu

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.runtime_context import get_server_args

logger = logging.getLogger(__name__)


class MoeA2ABackend(Enum):

    NONE = "none"
    DEEPEP = "deepep"
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    MORI = "mori"
    ASCEND_FUSEEP = "ascend_fuseep"
    ASCEND_TP = "ascend_tp"
    FLASHINFER = "flashinfer"
    MEGAMOE = "megamoe"
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

    def is_ascend_tp(self):
        return self == MoeA2ABackend.ASCEND_TP

    def is_mori(self):
        return self == MoeA2ABackend.MORI

    def is_megamoe(self):
        return self == MoeA2ABackend.MEGAMOE

    def is_customized(self):
        return self == MoeA2ABackend.CUSTOMIZED

    def supports_aiter(self) -> bool:
        return self in (
            MoeA2ABackend.NONE,
            MoeA2ABackend.DEEPEP,
            MoeA2ABackend.MOONCAKE,
            MoeA2ABackend.NIXL,
            MoeA2ABackend.MORI,
        )


class MoeRunnerBackend(Enum):

    AUTO = "auto"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    TRITON_KERNELS = "triton_kernel"
    ASCEND = "ascend"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    EXPERIMENTAL_SGL_TRTLLM = "experimental_sgl_trtllm"
    FLASHINFER_TRTLLM_ROUTED = "flashinfer_trtllm_routed"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    CUTLASS = "cutlass"
    MARLIN = "marlin"
    HUMMING = "humming"
    EXPERIMENTAL_SGL_MARLIN = "experimental_sgl_marlin"
    AITER = "aiter"

    def is_auto(self):
        return self == MoeRunnerBackend.AUTO

    def is_deep_gemm(self):
        return self == MoeRunnerBackend.DEEP_GEMM

    def is_triton(self):
        return self == MoeRunnerBackend.TRITON

    def is_ascend(self):
        return self == MoeRunnerBackend.ASCEND

    def is_triton_kernels(self):
        return self == MoeRunnerBackend.TRITON_KERNELS

    def is_flashinfer_trtllm(self):
        # experimental_sgl_trtllm shares the TRT-LLM FP8 kernels + layout, so it inherits
        # trtllm weight-prep here; divergent sites check is_experimental_sgl_trtllm() first.
        return self in (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
        )

    def is_experimental_sgl_trtllm(self):
        return self == MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM

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
        # experimental_sgl_marlin shares the marlin weight repack, quant-method
        # selection, and base fused path; divergent sites (the LoRA MoE dispatch)
        # check is_experimental_sgl_marlin() first.
        return self in (
            MoeRunnerBackend.MARLIN,
            MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN,
        )

    def is_experimental_sgl_marlin(self):
        return self == MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN

    def is_humming(self):
        return self == MoeRunnerBackend.HUMMING

    def is_aiter(self):
        return self == MoeRunnerBackend.AITER


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


class DispatcherOutputDtype(Enum):
    """
    Describes the dispatch output data type for DeepEP.

    - BF16: dispatch hidden states in bf16
    - FP8: dispatch hidden states in fp8
    - INT8: dispatch hidden states in int8
    - NVFP4: dispatch hidden states in nvfp4
    """

    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    NVFP4 = "nvfp4"


def get_deepep_output_dtype(self) -> DispatcherOutputDtype:
    """
    Automatically choose the dispatch output dtype for DeepEP.

    The decision follows several checks in priority order:
    0. Parse server argument.
    1. Parse deprecated environment variables.
    2. If quant_config contains input_global_scale → NVFP4 path.
    3. Parse quant config
    4. If flashinfer_cutedsl or is_cutlass backend is active → BF16 (it quantizes hidden_states internally).
    5. Otherwise default for NPU → BF16 (the default for NPU).
    6. Otherwise → FP8 (the default for most models like DeepSeek-V3).
    """

    # 0. Parse server argument.
    server_args = get_server_args()
    if server_args and server_args.deepep_dispatcher_output_dtype != "auto":
        return DispatcherOutputDtype(server_args.deepep_dispatcher_output_dtype)

    # 1. Parse deprecated environment variables.
    if envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
        logger.warning_once(
            "Warning: The env variable SGLANG_DEEPEP_BF16_DISPATCH deprecated "
            "and will be removed in future releases. Please use a new "
            "`--deepep-dispatcher-output-dtype bf16` argument instead."
        )
        return DispatcherOutputDtype.BF16

    # 2. NVFP4 is detected inside dispatch_a / _dispatch_core via quant_config; no need to infer here.
    if self.quant_config is not None:
        input_global_scale = self.quant_config.get("input_global_scale", None)
        if input_global_scale is not None:
            return DispatcherOutputDtype.NVFP4

        # 3. Parse quant config to determine the output dtype of dispatcher
        dispatcher_output_dtype = self.quant_config.get("dispatcher_output_dtype", None)
        if dispatcher_output_dtype is not None:
            return DispatcherOutputDtype(dispatcher_output_dtype)

    # 4. flashinfer_cutedsl / cutlass / humming expects BF16 dispatch
    if (
        get_moe_runner_backend().is_flashinfer_cutedsl()
        or get_moe_runner_backend().is_cutlass()
        or get_moe_runner_backend().is_humming()
    ):
        return DispatcherOutputDtype.BF16

    # 5. Default on NPU → BF16
    if _is_npu:
        return DispatcherOutputDtype.BF16

    # 6. Default → FP8
    return DispatcherOutputDtype.FP8


def get_ascend_dispatcher_output_dtype(dispatcher):
    """
    Automatically choose the dispatch output dtype for Ascend.
    """

    # 1. Parse quant config to determine the output dtype of dispatcher
    if dispatcher.quant_config is not None:
        dispatcher_output_dtype = dispatcher.quant_config.get(
            "dispatcher_output_dtype", None
        )
        if dispatcher_output_dtype is not None:
            return DispatcherOutputDtype(dispatcher_output_dtype)

    # 2. Ascend dispatch defaults to BF16
    return DispatcherOutputDtype.BF16


def initialize_moe_config(server_args: ServerArgs):
    moe = get_flags().moe
    moe.a2a_backend = MoeA2ABackend(server_args.moe_a2a_backend)
    moe.runner_backend = MoeRunnerBackend(server_args.moe_runner_backend)
    moe.speculative_runner_backend = (
        MoeRunnerBackend(server_args.speculative_moe_runner_backend)
        if server_args.speculative_moe_runner_backend is not None
        else moe.runner_backend
    )
    moe.speculative_a2a_backend = (
        MoeA2ABackend(server_args.speculative_moe_a2a_backend)
        if server_args.speculative_moe_a2a_backend is not None
        else moe.a2a_backend
    )
    moe.deepep_mode = DeepEPMode(server_args.deepep_mode)
    moe.deepep_config = server_args.deepep_config or ""
    moe.tbo_enabled = server_args.enable_two_batch_overlap
    moe.sbo_enabled = server_args.enable_single_batch_overlap
    if moe.sbo_enabled and is_cuda():
        if torch.cuda.get_device_capability()[0] == 9:
            raise ValueError(
                "SBO (single batch overlap) is not supported on SM90 GPUs with latest sgl-deep-gemm wheel. Please try removing --enable-single-batch-overlap argument."
            )
    moe.tbo_token_distribution_threshold = server_args.tbo_token_distribution_threshold
    moe.disable_fp4_allgather = server_args.disable_flashinfer_cutlass_moe_fp4_allgather
    moe.quantization = server_args.quantization


def get_moe_a2a_backend() -> MoeA2ABackend:
    moe = get_flags().moe
    if moe.a2a_backend is None:
        moe.a2a_backend = MoeA2ABackend.NONE
    return moe.a2a_backend


def get_moe_runner_backend() -> MoeRunnerBackend:
    moe = get_flags().moe
    if moe.runner_backend is None:
        moe.runner_backend = MoeRunnerBackend.AUTO
    return moe.runner_backend


def get_speculative_moe_runner_backend() -> MoeRunnerBackend:
    moe = get_flags().moe
    if moe.speculative_runner_backend is None:
        logger.warning(
            "SPECULATIVE_MOE_RUNNER_BACKEND is not initialized, using auto backend"
        )
        moe.speculative_runner_backend = MoeRunnerBackend.AUTO
    return moe.speculative_runner_backend


def get_speculative_moe_a2a_backend() -> MoeA2ABackend:
    moe = get_flags().moe
    if moe.speculative_a2a_backend is None:
        logger.warning(
            "SPECULATIVE_MOE_A2A_BACKEND is not initialized, using none backend"
        )
        moe.speculative_a2a_backend = MoeA2ABackend.NONE
    return moe.speculative_a2a_backend


def get_deepep_mode() -> DeepEPMode:
    moe = get_flags().moe
    if moe.deepep_mode is None:
        logger.warning("DEEPEP_MODE is not initialized, using auto mode")
        moe.deepep_mode = DeepEPMode.AUTO
    return moe.deepep_mode


def get_deepep_config() -> str:
    moe = get_flags().moe
    if moe.deepep_config is None:
        logger.warning("DEEPEP_CONFIG is not initialized, using default config")
        moe.deepep_config = ""
    return moe.deepep_config


def is_tbo_enabled() -> bool:
    moe = get_flags().moe
    if moe.tbo_enabled is None:
        moe.tbo_enabled = False
    return moe.tbo_enabled


def is_sbo_enabled() -> bool:
    moe = get_flags().moe
    if moe.sbo_enabled is None:
        moe.sbo_enabled = False
    return moe.sbo_enabled


def is_deepep_class_backend() -> bool:
    """Check if the MoE backend is DeepEP-family (DeepEP, Mooncake, or Mori)."""
    b = get_moe_a2a_backend()
    return b.is_deepep() or b.is_mooncake() or b.is_mori()


def uses_per_rank_fused_shared_slots() -> bool:
    """Check whether fused shared experts use per-rank physical slots."""
    return is_deepep_class_backend() or get_moe_a2a_backend().is_megamoe()


def has_per_rank_fused_shared_slots(num_fused_shared_experts: int) -> bool:
    """Check whether this layer has fused shared experts in per-rank slots."""
    return num_fused_shared_experts > 0 and uses_per_rank_fused_shared_slots()


def is_flashinfer_cutedsl_v1_path() -> bool:
    """CuteDSL v1 + DeepEP low-latency path (no MoeRunner, no autotune)."""
    return (
        get_moe_runner_backend().is_flashinfer_cutedsl()
        and get_moe_a2a_backend().is_deepep()
    )


def get_tbo_token_distribution_threshold() -> float:
    moe = get_flags().moe
    if moe.tbo_token_distribution_threshold is None:
        logger.warning(
            "TBO_TOKEN_DISTRIBUTION_THRESHOLD is not initialized, using 0.48"
        )
        moe.tbo_token_distribution_threshold = 0.48
    return moe.tbo_token_distribution_threshold


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
        not get_flags().moe.disable_fp4_allgather
        and get_moe_a2a_backend().is_none()
        and get_moe_runner_backend().is_flashinfer_cutlass()
        and is_dp_attention_enabled()
        and get_flags().moe.quantization == "modelopt_fp4"
        and get_parallel().moe_ep_size == get_parallel().attn_dp_size
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
        and get_parallel().attn_dp_size > 1
        and get_parallel().moe_ep_size == get_parallel().attn_dp_size
    )


def should_skip_mlp_all_reduce() -> bool:
    """Whether dense MLP / row-parallel projections should skip their all-reduce.

    True when the decoder published ``fuse_mlp_allreduce`` (next residual+LN
    absorbs the AR) or ``mlp_reduce_scatter`` (postprocess will reduce-scatter)
    on ``get_forward()``.
    """
    f = get_forward()
    return f.fuse_mlp_allreduce or f.mlp_reduce_scatter


def should_skip_post_experts_all_reduce(*, is_tp_path: bool) -> bool:
    """Whether to skip the post-experts all-reduce (EP or TP) because a
    downstream component will fuse, replace, or absorb it.

    Skip reasons, in order:
      - ``get_forward().fuse_mlp_allreduce``: LayerCommunicator will fuse the
        all-reduce with the next layer's residual all-reduce.
      - ``get_forward().mlp_reduce_scatter``: LayerCommunicator's post-attention
        scatter will do reduce-scatter, which would double-reduce on top of
        an all-reduce.
      - ``should_use_dp_reduce_scatterv()``: the standard dispatcher's combine
        path replaces the all-reduce with a reduce-scatterv.
      - ``should_use_flashinfer_cutlass_moe_fp4_allgather()`` (TP path only):
        the flashinfer cutlass FP4 kernel performs an all-gather that absorbs
        the post-experts TP all-reduce. Not relevant to the EP all-reduce.
      - ``get_moe_a2a_backend().is_flashinfer()``: the flashinfer A2A
        dispatcher's ``MoeAlltoAll.combine`` already alltoall-reduces partial
        MoE outputs back to the source rank, so any further EP/TP all-reduce
        would double-count and overflow BF16. Mirrors TRTLLM's
        ``not enable_alltoall`` gate
        (``tensorrt_llm/_torch/modules/fused_moe/interface.py:879``).

    The first two reasons come from per-layer ``ForwardFlags`` published by
    the decoder via ``get_forward().scoped(...)``. Pass ``is_tp_path=True``
    for the post-experts TP all-reduce, ``False`` for the EP all-reduce.
    """
    if should_skip_mlp_all_reduce():
        return True
    if should_use_dp_reduce_scatterv():
        return True
    if is_tp_path and should_use_flashinfer_cutlass_moe_fp4_allgather():
        return True
    if get_moe_a2a_backend().is_flashinfer():
        return True
    return False


@contextmanager
def speculative_moe_backend_context():
    """
    Context manager to temporarily use the speculative MoE backend for draft model operations.
    This ensures that draft models in speculative decoding use the configured speculative backend.
    """
    moe = get_flags().moe
    original_backend = moe.runner_backend
    try:
        moe.runner_backend = get_speculative_moe_runner_backend()
        yield
    finally:
        moe.runner_backend = original_backend


@contextmanager
def speculative_moe_a2a_backend_context():
    """
    Context manager to temporarily use the speculative MoE A2A backend for draft model operations.
    This ensures that draft models in speculative decoding use the configured speculative A2A backend.
    """
    moe = get_flags().moe
    original_backend = moe.a2a_backend
    original_disable_fp4_allgather = moe.disable_fp4_allgather
    try:
        moe.a2a_backend = get_speculative_moe_a2a_backend()
        # Disable FP4 allgather for spec decode since MTP layers are unquantized
        moe.disable_fp4_allgather = True
        yield
    finally:
        moe.a2a_backend = original_backend
        moe.disable_fp4_allgather = original_disable_fp4_allgather


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
    # SigmoidRenorm: Sigmoid -> TopK -> Renormalize
    SigmoidRenorm = (6,)
    # MiniMax2
    MiniMax2 = (7,)
    # Sigmoid: Sigmoid -> TopK (no renormalize)
    Sigmoid = (8,)
    # Unspecified
    Unspecified = 9


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
