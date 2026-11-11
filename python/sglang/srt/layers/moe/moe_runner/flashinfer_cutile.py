"""FlashInfer cuTile NVFP4 W4A4 fused MoE runner for SM120/SM121."""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from packaging.version import InvalidVersion, Version

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.quantization.utils import reorder_w1w3_to_w3w1
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

# Compute capabilities FlashInfer's CuTileNvfp4Config advertises (RTX 50xx / RTX PRO
# Blackwell, and GB10 / DGX Spark); SM100/SM110 have no cuTile NVFP4 W4A4 kernels.
CUTILE_NVFP4_DEVICE_SMS = (120, 121)
# First FlashInfer release shipping CuTileNvfp4Config.
_MIN_FLASHINFER_VERSION = Version("0.7.2")
# FlashInfer buckets its graph-stable workspace by token count and rounds up to a
# power of two past 8192, so one call of 8193 tokens allocates the 16384 bucket.
# Larger batches run as chunks of this size, which bounds the workspace per layer.
CUTILE_NVFP4_MAX_TOKENS_PER_CALL = 8192
_SCALE_BLOCK_SIZE = 16
# cuTile NVFP4 GEMM tiles need both reduction dimensions in whole 64-element tiles.
_DIM_ALIGNMENT = 64


# flashinfer.fused_moe.MoELayer per (device, layer shape, activation).
_SHARED_MOE_LAYERS: dict[tuple, Any] = {}


@dataclass
class FlashInferCutileNvfp4MoeQuantInfo(MoeQuantInfo):
    """Prepared weights and the FlashInfer layer that owns the kernel workspaces."""

    # flashinfer.fused_moe.MoELayer, shared by same-shape layers; it caches one
    # workspace per token bucket, so two calls must never overlap in time.
    moe_layer: Any
    # flashinfer.fused_moe.MoEWeightPack holding the "cutile_nvfp4" view.
    weight_pack: Any
    max_tokens_per_call: int = CUTILE_NVFP4_MAX_TOKENS_PER_CALL


def cutile_nvfp4_platform_unsupported_reason() -> Optional[str]:
    """Why this machine cannot run the cuTile NVFP4 runner, or None if it can.

    Covers the device architecture, the FlashInfer release, and the CUDA Tile
    compiler; it never raises, so it is safe to call while resolving defaults.
    """
    platform = get_platform()
    if not platform.is_cuda:
        return "it requires an NVIDIA CUDA device"
    device_sm = platform.device_sm
    if device_sm not in CUTILE_NVFP4_DEVICE_SMS:
        return (
            f"it supports compute capability {CUTILE_NVFP4_DEVICE_SMS} only, "
            f"got sm_{device_sm}"
        )
    try:
        installed = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        return "flashinfer-python is not installed"
    try:
        if Version(installed) < _MIN_FLASHINFER_VERSION:
            return (
                f"it requires flashinfer-python >= {_MIN_FLASHINFER_VERSION}, "
                f"got {installed}"
            )
    except InvalidVersion:
        return f"flashinfer-python reports an unparsable version {installed!r}"
    try:
        from flashinfer.cutile import is_cuda_tile_available
        from flashinfer.fused_moe import CuTileNvfp4Config
    except ImportError as exc:
        return f"the installed FlashInfer has no cuTile NVFP4 MoE ({exc})"
    if not CuTileNvfp4Config.supported(device_sm):
        return (
            f"the installed FlashInfer has no cuTile NVFP4 kernels for sm_{device_sm}"
        )
    if not is_cuda_tile_available():
        return (
            "the CUDA Tile toolchain is unavailable: install cuda-tile and a "
            "tileiras compiler (nvidia-cuda-tileiras) whose major.minor matches "
            "nvidia-cuda-nvcc and nvidia-nvvm"
        )
    return None


def cutile_nvfp4_config_unsupported_reason(config: MoeRunnerConfig) -> Optional[str]:
    """Why this MoE layer cannot run on the cuTile NVFP4 runner, or None if it can."""
    if config.num_local_experts != config.num_experts:
        # The kernel indexes weights with the routed id and defines no behavior
        # for the -1 the standard dispatcher writes for non-local experts.
        return (
            "expert parallelism is not supported "
            f"({config.num_local_experts} local of {config.num_experts} experts)"
        )
    if config.num_fused_shared_experts:
        return "fused shared experts are not supported"
    if not config.is_gated or config.activation != "silu":
        return (
            "only the gated SiLU (SwiGLU) activation is supported, got "
            f"activation={config.activation!r}, is_gated={config.is_gated}"
        )
    if (
        config.gemm1_alpha is None
        or config.gemm1_beta is None
        or config.gemm1_clamp_limit is None
    ):
        return (
            "the SwiGLU alpha, beta and clamp limit must all be set, got "
            f"alpha={config.gemm1_alpha}, beta={config.gemm1_beta}, "
            f"limit={config.gemm1_clamp_limit}"
        )
    if config.apply_router_weight_on_input or config.no_combine:
        return "apply_router_weight_on_input and no_combine are not supported"
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size_per_partition
    if hidden_size % _DIM_ALIGNMENT or intermediate_size % _DIM_ALIGNMENT:
        return (
            f"hidden_size and the per-rank intermediate size must be multiples of "
            f"{_DIM_ALIGNMENT}, got {hidden_size} and {intermediate_size}"
        )
    return None


def _unit_scale_violation(name: str, scale: torch.Tensor) -> Optional[str]:
    scale = scale.float()
    if bool(torch.all(scale == 1.0)):
        return None
    return (
        f"{name} must be exactly 1 because the cuTile kernel quantizes "
        f"activations with a unit global scale, got values in "
        f"[{scale.min().item():g}, {scale.max().item():g}]"
    )


def to_flashinfer_canonical_gemm1(
    w13_packed: torch.Tensor,
    w13_block_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder checkpoint ``[gate, up]`` GEMM1 tensors to FlashInfer's ``[up, gate]``.

    Rows of ``w13_packed`` / ``w13_block_scale`` are ``[w1 (gate); w3 (up)]`` and
    the per-expert encode scales are ``[E, 2]`` in the same order. FlashInfer's
    canonical source layout is ``[up, gate]`` on every one of them.
    """
    w13, scale = reorder_w1w3_to_w3w1(w13_packed, w13_block_scale, dim=-2)
    if w13_global_scale.dim() != 2 or w13_global_scale.shape[1] != 2:
        raise ValueError(
            "GEMM1 weight global scale must be [num_experts, 2] in [gate, up] "
            f"order, got shape {tuple(w13_global_scale.shape)}"
        )
    global_scale = w13_global_scale.float().flip(1).contiguous()
    return w13, scale, global_scale


def prepare_cutile_nvfp4_weight_view(
    *,
    w13_packed: torch.Tensor,
    w13_block_scale: torch.Tensor,
    w13_weight_global_scale: torch.Tensor,
    w13_input_global_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_block_scale: torch.Tensor,
    w2_weight_global_scale: torch.Tensor,
    w2_input_global_scale: torch.Tensor,
    activation: Any,
) -> dict[str, torch.Tensor]:
    """Build FlashInfer's ``cutile_nvfp4`` weight view from compressed-tensors experts.

    Weights are packed E2M1 (two values per byte) with row-major E4M3 block scales
    over 16 elements and one FP32 encode scale per expert projection, as stored
    by the checkpoint; GEMM1 rows are ``[gate; up]``. ``activation`` is a
    ``flashinfer.fused_moe.SwiGLU``.
    """
    from flashinfer.fused_moe import CuTileNvfp4Config

    violation = _unit_scale_violation(
        "w13_input_global_scale", w13_input_global_scale
    ) or _unit_scale_violation("w2_input_global_scale", w2_input_global_scale)
    if violation is not None:
        raise ValueError(f"cuTile NVFP4 MoE: {violation}")
    for name, scale in (
        ("w13_weight_global_scale", w13_weight_global_scale),
        ("w2_weight_global_scale", w2_weight_global_scale),
    ):
        if not bool(torch.all(torch.isfinite(scale) & (scale > 0))):
            raise ValueError(f"cuTile NVFP4 MoE: {name} must be finite and positive")

    num_experts, gemm1_rows, packed_hidden = w13_packed.shape
    w13, w13_scale, w13_global = to_flashinfer_canonical_gemm1(
        w13_packed, w13_block_scale, w13_weight_global_scale
    )
    # compressed_tensors: global scales are encode scales (FlashInfer inverts them)
    # and block scales are row-major (FlashInfer swizzles them for SM12x).
    return CuTileNvfp4Config.prepare_weights(
        w13,
        w13_scale,
        w13_global,
        w2_packed,
        w2_block_scale,
        w2_weight_global_scale.float().contiguous(),
        num_local_experts=num_experts,
        hidden_size=packed_hidden * 2,
        intermediate_size=gemm1_rows // 2,
        activation=activation,
        source_format="compressed_tensors",
        device=w13_packed.device,
    )


def build_cutile_nvfp4_quant_info(
    *,
    w13_packed: torch.Tensor,
    w13_block_scale: torch.Tensor,
    w13_weight_global_scale: torch.Tensor,
    w13_input_global_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_block_scale: torch.Tensor,
    w2_weight_global_scale: torch.Tensor,
    w2_input_global_scale: torch.Tensor,
    top_k: int,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
) -> FlashInferCutileNvfp4MoeQuantInfo:
    """Prepare the experts and the FlashInfer layer once, at load."""
    from flashinfer.fused_moe import (
        BackendOptions,
        CuTileNvfp4Config,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )

    activation = SwiGLU(alpha=swiglu_alpha, beta=swiglu_beta, limit=swiglu_limit)
    view = prepare_cutile_nvfp4_weight_view(
        w13_packed=w13_packed,
        w13_block_scale=w13_block_scale,
        w13_weight_global_scale=w13_weight_global_scale,
        w13_input_global_scale=w13_input_global_scale,
        w2_packed=w2_packed,
        w2_block_scale=w2_block_scale,
        w2_weight_global_scale=w2_weight_global_scale,
        w2_input_global_scale=w2_input_global_scale,
        activation=activation,
    )
    num_experts, gemm1_rows, packed_hidden = w13_packed.shape
    intermediate_size = gemm1_rows // 2
    device = w13_packed.device
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("cutile_nvfp4", view)
    # FlashInfer keeps its workspaces on the MoELayer (about 1.2 GiB for one
    # 8192-token bucket at 128 experts, hidden 6144, intermediate 3072), and takes
    # the weights per call; layers of one shape run one after another, so they
    # share a MoELayer instead of holding that workspace once per model layer.
    layer_key = (
        device,
        num_experts,
        top_k,
        packed_hidden * 2,
        intermediate_size,
        (swiglu_alpha, swiglu_beta, swiglu_limit),
    )
    moe_layer = _SHARED_MOE_LAYERS.get(layer_key)
    if moe_layer is None:
        moe_layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
                quant=QuantConfig(
                    weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4
                ),
                experts=ExpertConfig(intermediate_size=intermediate_size),
                activation=activation,
                backend=BackendOptions(candidates=(CuTileNvfp4Config(),)),
                execution=ExecutionConfig(
                    tune_max_num_tokens=CUTILE_NVFP4_MAX_TOKENS_PER_CALL
                ),
            ),
            device=device,
        )
        _SHARED_MOE_LAYERS[layer_key] = moe_layer
    return FlashInferCutileNvfp4MoeQuantInfo(
        moe_layer=moe_layer, weight_pack=weight_pack
    )


def _run_cutile_nvfp4(
    *,
    quant_info: FlashInferCutileNvfp4MoeQuantInfo,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    from flashinfer.fused_moe import MoEActivationPack
    from flashinfer.fused_moe.core import RoutingInputMode

    def run(rows: slice) -> torch.Tensor:
        # The cuTile runners take separate int32 ids and fp32 weights under
        # PackedPrecomputed, their only routing mode; nothing is bit-packed.
        return quant_info.moe_layer(
            MoEActivationPack(
                hidden_states[rows],
                None,
                topk_ids[rows],
                topk_weights[rows],
                routing_input_mode=RoutingInputMode.PackedPrecomputed,
            ),
            quant_info.weight_pack,
        )

    num_tokens = hidden_states.shape[0]
    step = quant_info.max_tokens_per_call
    if num_tokens <= step:
        return run(slice(None))
    # Every output row depends on its own token only (activation scales are per
    # 16-element block of one row, the combine gathers one token's top-k rows),
    # so chunking over tokens does not change which values are combined.
    return torch.cat(
        [run(slice(start, start + step)) for start in range(0, num_tokens, step)]
    )


def _drop_unrouted_slots(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TopK writes expert id -1 into CUDA-graph padding rows (num_token_non_padded),
    # and the cuTile kernel defines no behaviour for negative ids. Route those
    # slots to expert 0 with weight 0, on device, so capture needs no host sync.
    valid = topk_ids >= 0
    ids = torch.where(valid, topk_ids, 0).to(torch.int32).contiguous()
    weights = torch.where(valid, topk_weights, 0).to(torch.float32).contiguous()
    return ids, weights


@register_fused_func("none", "flashinfer_cutile")
def fused_experts_none_to_flashinfer_cutile(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferCutileNvfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if not TopKOutputChecker.format_is_standard(topk_output):
        raise TypeError(
            "cuTile NVFP4 MoE needs standard top-k output (ids and weights), got "
            f"{type(topk_output).__name__}"
        )
    if dispatch_output.hidden_states_scale is not None:
        raise ValueError("cuTile NVFP4 MoE quantizes its own input; got a scale")
    if hidden_states.dim() != 2 or hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "cuTile NVFP4 MoE needs 2D bfloat16 hidden states, got "
            f"shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}"
        )
    if hidden_states.shape[0] == 0:
        return StandardCombineInput(hidden_states=hidden_states.clone())

    topk_ids, topk_weights = _drop_unrouted_slots(
        topk_output.topk_ids, topk_output.topk_weights
    )
    output = _run_cutile_nvfp4(
        quant_info=quant_info,
        hidden_states=hidden_states.contiguous(),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    # The kernel applies the router weights but no routed scaling factor.
    scaling = runner_config.routed_scaling_factor
    if scaling is not None and scaling != 1.0:
        output.mul_(scaling)
    return StandardCombineInput(hidden_states=output)
