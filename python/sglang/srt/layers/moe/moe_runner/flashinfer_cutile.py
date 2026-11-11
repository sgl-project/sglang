# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    DispatchMoeRunnerCore,
    MoeQuantInfo,
    MoeRunnerConfig,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_a2a_backend
from sglang.srt.runtime_context import get_buffer

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


class FlashInferCuTileMoeQuantInfo(MoeQuantInfo):
    """One layer's prepared weights, separate from shared execution workspaces.

    This is not an nn.Module: the quantization scheme owns the pack and rejects
    parameter-based offload, online updates and checkpoint export.
    """

    def __init__(self, *, weight_pack: Any, device: torch.device):
        self.weight_pack = weight_pack
        self.device = torch.device(device)


class _CuTileExecution:
    """A public MoELayer owned by one host thread and one CUDA stream."""

    def __init__(
        self, *, config: MoeRunnerConfig, device: torch.device, max_num_tokens: int
    ):
        self.device = device
        self.layer = _create_moe_layer(
            config=config, device=device, max_num_tokens=max_num_tokens
        )
        self.warmed_token_counts: set[int] = set()

    def __call__(self, act_pack: Any, weight_pack: Any) -> torch.Tensor:
        num_tokens = act_pack.hidden_states_q.shape[0]
        capturing = _is_capturing(self.device)
        # Track successful eager calls, not just allocated workspace buckets:
        # capture must not be the first call that needs a compiled shape.
        if capturing and num_tokens not in self.warmed_token_counts:
            raise RuntimeError(
                "flashinfer_cutile requires an eager warmup of this token count "
                "on the same CUDA stream before CUDA graph capture."
            )
        output = self.layer(act_pack, weight_pack)
        if not capturing:
            self.warmed_token_counts.add(num_tokens)
        return output


class FlashInferCuTileRunnerCore(DispatchMoeRunnerCore):
    """Share execution workspaces for the lifetime of the current resources.

    FlashInfer caches workspaces for visited token buckets in each execution.
    Different streams need independent executions, so their allocations add up.
    The token capacity bounds inputs, not the sum of resident workspace memory.
    Keep these allocations until the runtime resources are reset: individual
    eviction could invalidate a captured CUDA graph.
    """

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        validate_flashinfer_cutile_config(config)
        self._geometry = (
            config.num_experts,
            config.hidden_size,
            config.intermediate_size_per_partition,
            config.top_k,
            *_activation_parameters(config),
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.FLASHINFER_CUTILE

    def _get_execution(self, device: torch.device, num_tokens: int) -> _CuTileExecution:
        max_num_tokens = _get_max_num_tokens()
        if num_tokens > max_num_tokens:
            raise ValueError(
                f"flashinfer_cutile received {num_tokens} tokens, exceeding "
                f"the configured capacity {max_num_tokens}."
            )
        # Weights are supplied per call, so equal geometries can share an
        # execution across layers. A separate execution per thread/stream avoids
        # concurrent calls overwriting the same temporary buffers.
        key = (
            threading.get_ident(),
            device,
            torch.cuda.current_stream(device=device).cuda_stream,
            self._geometry,
            max_num_tokens,
        )
        # Do not cache this pool or its executions on the runner. reset_context()
        # replaces Resources while existing runners may still be alive.
        pool: dict[tuple, _CuTileExecution] = get_buffer(
            "flashinfer_cutile_execution_pool", dict
        )
        execution = pool.get(key)
        if execution is None:
            if _is_capturing(device):
                raise RuntimeError(
                    "flashinfer_cutile must initialize and warm up its "
                    "execution stream before CUDA graph capture."
                )
            _check_device_support(device)
            execution = _CuTileExecution(
                config=self.config,
                device=device,
                max_num_tokens=max_num_tokens,
            )
            pool[key] = execution
        return execution

    def run_from_dispatch(
        self,
        dispatch_output: StandardDispatchOutput,
        quant_info: FlashInferCuTileMoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks=None,
    ) -> StandardCombineInput:
        from flashinfer.fused_moe import MoEActivationPack
        from flashinfer.fused_moe.core import RoutingInputMode

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        if hooks is not None:
            raise ValueError("flashinfer_cutile does not support LoRA hooks.")
        if not isinstance(dispatch_output, StandardDispatchOutput):
            raise ValueError("flashinfer_cutile requires the standard dispatcher.")
        if not isinstance(quant_info, FlashInferCuTileMoeQuantInfo):
            raise TypeError("flashinfer_cutile requires prepared cuTile NVFP4 weights.")
        x = dispatch_output.hidden_states
        if x.dtype != torch.bfloat16:
            raise TypeError("flashinfer_cutile requires BF16 hidden states.")
        if x.ndim != 2 or x.shape[1] != self.config.hidden_size:
            raise ValueError("flashinfer_cutile hidden-state shape is inconsistent.")
        if x.device != quant_info.device:
            raise ValueError(
                "flashinfer_cutile inputs and weights must share a device."
            )
        if dispatch_output.hidden_states_scale is not None:
            raise ValueError("flashinfer_cutile quantizes BF16 activations internally.")
        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(
                "flashinfer_cutile requires separate top-k ids and weights."
            )
        ids, weights = topk_output.topk_ids, topk_output.topk_weights
        expected = (x.shape[0], self.config.top_k)
        if tuple(ids.shape) != expected or tuple(weights.shape) != expected:
            raise ValueError(
                f"flashinfer_cutile top-k tensors must have shape {expected}."
            )
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("flashinfer_cutile top-k ids must be int32 or int64.")
        if not weights.is_floating_point():
            raise TypeError("flashinfer_cutile top-k weights must be floating point.")
        if ids.device != x.device or weights.device != x.device:
            raise ValueError(
                "flashinfer_cutile routing and activations must share a device."
            )
        if x.shape[0] == 0:
            return StandardCombineInput(hidden_states=torch.empty_like(x))

        # Graph padding can contain unrouted slots (expert id -1). Replace
        # each with an in-range id AND zero weight: changing only the id would
        # add expert 0's output. Keep this on-device to avoid a capture-breaking
        # host synchronization; routing normalization already happened upstream.
        valid = ids >= 0
        ids = torch.where(valid, ids, 0).to(dtype=torch.int32).contiguous()
        weights = torch.where(valid, weights, 0).to(dtype=torch.float32).contiguous()
        act_pack = MoEActivationPack(
            hidden_states_q=x.contiguous(),
            hidden_states_scale=None,
            topk_ids=ids,
            topk_weights=weights,
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
        execution = self._get_execution(device=x.device, num_tokens=x.shape[0])
        output = execution(act_pack, quant_info.weight_pack)
        # MiniMaxM3MoE configures TopK to include model scaling in its weights.
        # Preserve those weights and apply only an additional factor carried by
        # the runner contract, at the same output-side boundary as Marlin.
        if runner_config.routed_scaling_factor is not None:
            output = output * runner_config.routed_scaling_factor
        return StandardCombineInput(hidden_states=output)


def validate_flashinfer_cutile_config(config: MoeRunnerConfig) -> None:
    if not get_moe_a2a_backend().is_none():
        raise ValueError("flashinfer_cutile requires --moe-a2a-backend none.")
    if config.layer is not None and config.moe_ep_size != 1:
        raise ValueError("flashinfer_cutile requires expert parallel size 1.")
    if config.num_local_experts not in (None, config.num_experts):
        raise ValueError("flashinfer_cutile does not support expert parallelism.")
    if config.num_fused_shared_experts:
        raise ValueError("flashinfer_cutile does not support fused shared experts.")
    if config.params_dtype not in (None, torch.bfloat16):
        raise ValueError("flashinfer_cutile requires BF16 model activations.")
    if config.activation != "silu" or not config.is_gated:
        raise ValueError("flashinfer_cutile currently supports gated SwiGLU only.")
    unsupported = (
        config.apply_router_weight_on_input,
        config.no_combine,
        config.no_combine_keep_router_weight,
        config.use_tp_all_gather_activation,
        config.silu_mul_keep_fp32,
    )
    if any(unsupported):
        raise ValueError(
            "flashinfer_cutile requires output-side routing, combined output, "
            "standard dispatch and native NVFP4 activation arithmetic."
        )
    for name in (
        "num_experts",
        "hidden_size",
        "intermediate_size_per_partition",
        "top_k",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"flashinfer_cutile requires a positive {name}.")
    if config.top_k > config.num_experts:
        raise ValueError("flashinfer_cutile top_k must not exceed num_experts.")
    if config.hidden_size % 64 or config.intermediate_size_per_partition % 64:
        raise ValueError(
            "flashinfer_cutile hidden and intermediate sizes must be divisible by 64."
        )
    _activation_parameters(config)
    factor = config.routed_scaling_factor
    if factor is not None and not math.isfinite(factor):
        raise ValueError("flashinfer_cutile routed_scaling_factor must be finite.")


def prepare_flashinfer_cutile_nvfp4_weights(
    layer: torch.nn.Module, config: MoeRunnerConfig
) -> FlashInferCuTileMoeQuantInfo:
    """Adapt loaded MiniMax experts to FlashInfer's canonical NVFP4 layout.

    With E experts, hidden size H and local intermediate size I, the loader's
    packed w13 has shape [E, 2*I, H/2]: two FP4 values share each byte, and
    output rows are [gate, up]. FlashInfer's w1 argument is this fused first
    projection (not the checkpoint's gate-only w1) and expects [up, gate].
    Reorder the matching scales with the rows; w2 is the down projection and
    keeps its order. FlashInfer then owns the backend-specific preparation.
    """
    from flashinfer.fused_moe import CuTileNvfp4Config, MoEWeightPack, SwiGLU

    validate_flashinfer_cutile_config(config)
    device = layer.w13_weight_packed.device
    _check_device_support(device)
    # cuTile quantizes both GEMM inputs dynamically. This adapter does not pass
    # checkpoint activation global scales, so accepting a non-unit value would
    # silently change the checkpoint's arithmetic.
    for name in ("w13_input_global_scale", "w2_input_global_scale"):
        scale = getattr(layer, name)
        if not torch.equal(scale, torch.ones_like(scale)):
            raise ValueError(
                "flashinfer_cutile currently requires activation global scales "
                f"equal to 1, but {name} is non-unit."
            )
    for name in ("w13_weight_global_scale", "w2_weight_global_scale"):
        scale = getattr(layer, name)
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"flashinfer_cutile requires finite positive {name}.")
    # Block scales are row-major E4M3, one per 16 values along the input
    # dimension. They follow the output rows, so swapping only packed weights
    # would pair each branch with the other branch's quantization scales.
    gate, up = layer.w13_weight_packed.chunk(2, dim=1)
    gate_scale, up_scale = layer.w13_weight_scale.chunk(2, dim=1)
    global_scale = layer.w13_weight_global_scale
    if tuple(global_scale.shape) not in (
        (config.num_experts,),
        (config.num_experts, 2),
    ):
        raise ValueError(
            "flashinfer_cutile w13_weight_global_scale must have shape "
            "[num_experts] or [num_experts, 2]."
        )
    # A single global scale per expert is shared by both branches. With
    # two columns, [gate, up] must become [up, gate] just like the weight rows.
    if global_scale.ndim == 2:
        global_scale = global_scale[:, [1, 0]].contiguous()
    alpha, beta, limit = _activation_parameters(config)
    # Pass checkpoint scales unchanged apart from branch order. The
    # compressed_tensors source format handles their global-scale convention
    # and swizzles the row-major block scales; do not apply the Marlin reciprocal
    # conversion or pre-swizzle them here.
    view = CuTileNvfp4Config.prepare_weights(
        w1_fp4=torch.cat((up, gate), dim=1),
        w1_block_scale=torch.cat((up_scale, gate_scale), dim=1),
        w1_global_scale=global_scale,
        w2_fp4=layer.w2_weight_packed,
        w2_block_scale=layer.w2_weight_scale,
        w2_global_scale=layer.w2_weight_global_scale,
        num_local_experts=config.num_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size_per_partition,
        activation=SwiGLU(alpha=alpha, beta=beta, limit=limit),
        source_format="compressed_tensors",
        device=device,
    )
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("cutile_nvfp4", view)
    return FlashInferCuTileMoeQuantInfo(weight_pack=weight_pack, device=device)


def _activation_parameters(config: MoeRunnerConfig) -> tuple[float, float, float]:
    # Share one activation mapping between weight preparation and execution.
    # MiniMax forwards its configured alpha/limit through gemm1_alpha and
    # gemm1_clamp_limit, and sets gemm1_beta=1.0. Preserve these values instead
    # of relying on FlashInfer's default activation parameters.
    limit = (
        config.gemm1_clamp_limit
        if config.gemm1_clamp_limit is not None
        else config.swiglu_limit
    )
    parameters = (config.gemm1_alpha, config.gemm1_beta, limit)
    if any(value is None or not math.isfinite(value) for value in parameters):
        raise ValueError(
            "flashinfer_cutile requires finite SwiGLU alpha, beta and limit."
        )
    if limit <= 0:
        raise ValueError("flashinfer_cutile SwiGLU limit must be positive.")
    return tuple(float(value) for value in parameters)


def _check_device_support(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("flashinfer_cutile requires an SM120 or SM121 CUDA device.")
    major, minor = torch.cuda.get_device_capability(device=device)
    if (major, minor) not in ((12, 0), (12, 1)):
        raise ValueError(
            f"flashinfer_cutile requires SM120/SM121, got SM{major}{minor}."
        )


def _is_capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _get_max_num_tokens() -> int:
    from sglang.srt.runtime_context import cutedsl_moe_max_num_tokens

    # Reuse the scheduler's published capacity for both prefill and decode.
    # This is an input bound, not a byte budget for cached workspaces.
    capacity = cutedsl_moe_max_num_tokens()
    if capacity <= 0:
        raise ValueError("flashinfer_cutile requires a positive token capacity.")
    return capacity


def _create_moe_layer(
    *, config: MoeRunnerConfig, device: torch.device, max_num_tokens: int
):
    from flashinfer.fused_moe import (
        BackendOptions,
        CuTileNvfp4Config,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoELayer,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )

    alpha, beta, limit = _activation_parameters(config)
    moe_config = MoEConfig(
        routing=RoutingConfig(num_experts=config.num_experts, top_k=config.top_k),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=config.intermediate_size_per_partition),
        activation=SwiGLU(alpha=alpha, beta=beta, limit=limit),
        backend=BackendOptions(candidates=(CuTileNvfp4Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=max_num_tokens),
    )
    return MoELayer(config=moe_config, device=device)
