# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic FlashInfer MegaMOE backend (moe_ep.MoEEpMegaLayer).

Wraps FlashInfer's fused EP all-to-all + expert-compute mega kernel so it can
be selected as a model-agnostic MoE runner backend through the standard
FusedMoE dispatch -> run_moe_core -> combine flow. The mega kernel does its EP
communication internally via the deep_gemm symmetric buffer, so the dispatcher
and combine stay pure no-ops; this module owns the layer build + forward.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

logger = logging.getLogger(__name__)


def _format_megakernel_config(config: Any) -> str:
    """Readable one-line repr of a mega kernel config.

    The config dataclasses carry per-expert tensor fields (e.g. fc1_alpha /
    fc2_alpha / fc1_norm_const); their default repr dumps every element, so
    abbreviate tensors to shape/dtype/device instead.
    """
    import dataclasses

    if not dataclasses.is_dataclass(config):
        return repr(config)

    parts = []
    for field, value in zip(dataclasses.fields(config), dataclasses.astuple(config)):
        if isinstance(value, torch.Tensor):
            value = (
                f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"device={value.device})"
            )
        else:
            value = repr(value)
        parts.append(f"{field.name}={value}")
    return f"{type(config).__name__}({', '.join(parts)})"


if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        DispatchOutput,
        StandardCombineInput,
    )


@contextmanager
def _capture_safe_ue8m0_pack() -> Generator[None, None, None]:
    """Make deep_gemm's UE8M0 scale packing safe under CUDA graph capture.

    The deep_gemm mega staging path (block-FP8 models such as DeepSeek-V4-Flash)
    runs ``per_token_cast_to_fp8(..., use_packed_ue8m0=True)`` on every forward,
    which calls ``deep_gemm.utils.math.pack_ue8m0_to_int``. Its upstream
    implementation carries two debug assertions::

        assert (x_int >= 0).all() and (x_int & 0x7fffff == 0).all()

    ``.all()`` forces a device->host sync, which is illegal while a CUDA graph is
    capturing. Replace the helper only around the mega forward that runs during
    capture, then restore the exact upstream function.

    TODO(deepseek-ai/DeepGEMM#414): remove once upstream provides a capture-safe
    helper: https://github.com/deepseek-ai/DeepGEMM/issues/414
    """
    if not torch.cuda.is_available() or not torch.cuda.is_current_stream_capturing():
        yield
        return

    try:
        import deep_gemm.utils.math as _dgm
    except ImportError:
        # deep_gemm is only needed by the block-FP8 mega path; NVFP4/MXFP8 mega
        # runs on cutedsl and does not import it. Nothing to patch here.
        yield
        return

    def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
        x_int = x.view(torch.int)
        return (x_int >> 23).to(torch.uint8).view(torch.int)

    original = _dgm.pack_ue8m0_to_int
    _dgm.pack_ue8m0_to_int = _pack_ue8m0_to_int
    try:
        yield
    finally:
        _dgm.pack_ue8m0_to_int = original


@dataclass
class FlashInferMegaMoeQuantInfo(MoeQuantInfo):
    mega: Any
    mega_forward: Callable[[Any, Any], torch.Tensor] | None = None
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    fc1_norm_const: torch.Tensor | None = None
    apply_routed_scaling_factor: bool = False

    def __post_init__(self) -> None:
        if self.mega_forward is None:
            self.mega_forward = _select_megamoe_forward(self.mega)


def _forward_megamoe_with_workspace_view(mega: Any, tensors: Any) -> torch.Tensor:
    return mega.forward(tensors, return_workspace_view=True)


def _forward_megamoe_legacy(mega: Any, tensors: Any) -> torch.Tensor:
    return mega.forward(tensors)


def _select_megamoe_forward(mega: Any) -> Callable[[Any, Any], torch.Tensor]:
    import inspect

    if "return_workspace_view" in inspect.signature(mega.forward).parameters:
        return _forward_megamoe_with_workspace_view
    return _forward_megamoe_legacy


def _resolve_max_tokens_per_rank() -> int:
    """Per-rank symmetric-buffer sizing for the mega kernel.

    Honors the explicit env override; otherwise derives the largest per-(DP)rank
    token count a single MoE forward can route (same bound the cutedsl A2A path
    uses), falling back to 1024 if it cannot be determined.
    """
    configured = envs.SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK.get()
    if configured > 0:
        return configured

    from sglang.srt.runtime_context import cutedsl_moe_max_num_tokens

    derived = cutedsl_moe_max_num_tokens()
    return derived if derived > 0 else 1024


def resolve_flashinfer_megamoe_combine_dtype() -> str:
    combine_dtype = envs.SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE.get().strip().lower()
    if combine_dtype not in ("bf16", "mxfp8", "nvfp4"):
        raise ValueError(
            "SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE must be one of "
            f"'bf16', 'mxfp8', or 'nvfp4', got {combine_dtype!r}."
        )
    if (
        combine_dtype != "bf16"
        and envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get()
    ):
        raise ValueError(
            "SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE="
            f"{combine_dtype!r} is incompatible with "
            "SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE=1."
        )
    return combine_dtype


def _layer_ep_world_rank(layer: FusedMoE) -> tuple[int, int]:
    world_size = int(layer.moe_ep_size)
    rank = int(layer.moe_ep_rank)
    if world_size <= 0:
        raise ValueError(f"moe_ep_size must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"moe_ep_rank must be in [0, {world_size}), got {rank}.")
    return world_size, rank


def _scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(torch.float32).max())
    return float(value)


def _local_expert_vector(value: torch.Tensor, num_local_experts: int) -> torch.Tensor:
    value = value.detach().to(torch.float32)
    if value.dim() == 0:
        return value.expand(num_local_experts).contiguous()
    if value.shape != (num_local_experts,):
        raise ValueError(
            f"expected per-local-expert vector of shape ({num_local_experts},), "
            f"got {tuple(value.shape)}"
        )
    return value.contiguous()


def _validate_nvfp4_fc1_alpha(layer: FusedMoE) -> None:
    """MegaMOE reuses ``g1_alphas`` as fc1_alpha; the kernel takes one alpha per
    expert, so the gate and up FC1 alphas must agree."""
    if not layer.moe_runner_config.is_gated:
        return
    gate_alpha = _local_expert_vector(layer.g1_alphas, layer.num_local_experts)
    up_alpha = _local_expert_vector(layer.g1_alphas_up, layer.num_local_experts)
    if not torch.allclose(gate_alpha, up_alpha):
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires matching gate/up FC1 alpha "
            "values because the kernel accepts one alpha per expert."
        )


def _bind_transformed_weights(
    layer: FusedMoE,
    transformed_weights: Any,
    *,
    w13_scale_name: str,
    w2_scale_name: str,
) -> None:
    from sglang.srt.layers.utils.common import copy_or_rebind_param

    (w13_weight, w13_scale), (w2_weight, w2_scale) = transformed_weights
    copy_or_rebind_param(layer, "w13_weight", w13_weight)
    copy_or_rebind_param(layer, w13_scale_name, w13_scale)
    copy_or_rebind_param(layer, "w2_weight", w2_weight)
    copy_or_rebind_param(layer, w2_scale_name, w2_scale)


def _init_flashinfer_megamoe_layer_state(layer: FusedMoE) -> None:
    layer._flashinfer_megamoe_layer = None
    layer._flashinfer_megamoe_forward = None
    layer._flashinfer_megamoe_input_norm_const = None


def _get_or_init_flashinfer_megamoe_layer_state(layer: FusedMoE) -> Any:
    if not hasattr(layer, "_flashinfer_megamoe_layer"):
        _init_flashinfer_megamoe_layer_state(layer)
    return layer._flashinfer_megamoe_layer


def _ensure_flashinfer_megamoe_layer(
    layer: FusedMoE,
    *,
    megakernel_config: Any,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> Any:
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
    )

    transformed_weights = (
        (layer.w13_weight.data, w13_scale.data),
        (layer.w2_weight.data, w2_scale.data),
    )
    world_size, rank = _layer_ep_world_rank(layer)

    max_tokens_per_rank = _resolve_max_tokens_per_rank()
    logger.debug(
        "FlashInfer MegaMOE layer[%s] build: megakernel_config=%s "
        "(world_size=%d, num_experts=%d, max_tokens_per_rank=%d, hidden_size=%d)",
        layer.layer_id,
        _format_megakernel_config(megakernel_config),
        world_size,
        layer.num_experts,
        max_tokens_per_rank,
        layer.hidden_size,
    )

    mega = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size, rank=rank, device=torch.cuda.current_device()
        ),
        fleet_params=FleetParams(
            num_experts=layer.num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=layer.hidden_size,
        ),
        # weights already preprocessed in prepare_*; with transformed_weights set
        # the kernel never reads `weights` (see MoEEpMegaLayer), so pass None.
        weights=None,
        backend=MegaConfig(
            megakernel=megakernel_config,
            preprocess_weights=False,
            transformed_weights=transformed_weights,
        ),
    )
    layer._flashinfer_megamoe_layer = mega
    layer._flashinfer_megamoe_forward = _select_megamoe_forward(mega)
    return mega


def ensure_fp4_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import DeepGemmMegaMoeConfig

    return _ensure_flashinfer_megamoe_layer(
        layer,
        megakernel_config=DeepGemmMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            activation_clamp=layer.moe_runner_config.swiglu_limit,
        ),
        w13_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
    )


def ensure_nvfp4_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import Nvfp4CutedslMegaMoeConfig

    input_norm_const = layer._flashinfer_megamoe_input_norm_const
    if input_norm_const is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer NVFP4 MegaMOE layer must be initialized before "
                "CUDA graph capture."
            )
        logger.warning(
            "FlashInfer NVFP4 MegaMOE layer[%s]: input_norm_const was not "
            "precomputed at weight-load time; computing it lazily now via a "
            "blocking device sync. This should only happen once per layer, "
            "but if it happens during warmup it will look like a stall.",
            layer.layer_id,
        )
        input_norm_const = _scalar_float(layer.w13_input_scale_quant)
        layer._flashinfer_megamoe_input_norm_const = input_norm_const

    return _ensure_flashinfer_megamoe_layer(
        layer,
        megakernel_config=Nvfp4CutedslMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            gate_up_clamp=layer.moe_runner_config.swiglu_limit,
            apply_topk_in_fc1=True,
            in_kernel_fc2_reduce=envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get(),
            combine_dtype=resolve_flashinfer_megamoe_combine_dtype(),
            input_norm_const=input_norm_const,
            fc1_alpha=layer.g1_alphas,
            fc2_alpha=layer.g2_alphas,
            fc1_norm_const=layer.w2_input_scale_quant,
        ),
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
    )


def ensure_mxfp8_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import Mxfp8CutedslMegaMoeConfig

    return _ensure_flashinfer_megamoe_layer(
        layer,
        megakernel_config=Mxfp8CutedslMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            kind="mxfp8_e4m3",
            gate_up_clamp=layer.moe_runner_config.swiglu_limit,
            in_kernel_fc2_reduce=envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get(),
        ),
        w13_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
    )


def prepare_fp4_moe_weights_for_flashinfer_megamoe(
    layer: FusedMoE,
) -> None:
    """Prepare loaded FP4 weights for MegaMOE.

    SGLang loads FP4-packed expert weights plus raw block scales. FlashInfer's
    current moe_ep API owns backend-specific weight preprocessing, including
    DeepGEMM scale layout transforms.
    """
    _init_flashinfer_megamoe_layer_state(layer)

    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_mega_weights,
    )

    weights = MoEWeightPack(
        w13=layer.w13_weight.data,
        w2=layer.w2_weight.data,
        w13_scale=layer.w13_weight_scale_inv.data,
        w2_scale=layer.w2_weight_scale_inv.data,
    )
    transformed_weights = preprocess_mega_weights(
        weights,
        intermediate_size=layer.intermediate_size_per_partition,
        hidden_size=layer.hidden_size,
    )
    _bind_transformed_weights(
        layer,
        transformed_weights,
        w13_scale_name="w13_weight_scale_inv",
        w2_scale_name="w2_weight_scale_inv",
    )


def prepare_nvfp4_moe_weights_for_flashinfer_megamoe(
    layer: FusedMoE,
) -> None:
    _init_flashinfer_megamoe_layer_state(layer)

    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_nvfp4_cutedsl_mega_weights,
    )

    if layer.hidden_size % 128 != 0:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires hidden_size to be a multiple "
            f"of 128, got {layer.hidden_size}."
        )
    if layer.quant_config.use_per_token_activation:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE does not support per-token activation "
            "scaling. Use flashinfer_trtllm/flashinfer_trtllm_routed for "
            "ModelOpt NVFP4 per-token activation."
        )
    if layer.intermediate_size_per_partition % 128 != 0:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires intermediate_size_per_partition "
            f"to be a multiple of 128, got {layer.intermediate_size_per_partition}."
        )
    if layer.num_experts % layer.moe_ep_size != 0:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires num_experts to be divisible by "
            f"ep_size, got {layer.num_experts=} and {layer.moe_ep_size=}."
        )

    _validate_nvfp4_fc1_alpha(layer)
    layer._flashinfer_megamoe_input_norm_const = _scalar_float(
        layer.w13_input_scale_quant
    )

    gate_up_clamp = layer.moe_runner_config.swiglu_limit

    weights = MoEWeightPack(
        w13=layer.w13_weight.data,
        w2=layer.w2_weight.data,
        w13_scale=layer.w13_weight_scale.data,
        w2_scale=layer.w2_weight_scale.data,
    )
    transformed_weights = preprocess_nvfp4_cutedsl_mega_weights(
        weights,
        intermediate_size=layer.intermediate_size_per_partition,
        hidden_size=layer.hidden_size,
        gate_up_clamp=gate_up_clamp,
        activation_clamp=None,
    )
    _bind_transformed_weights(
        layer,
        transformed_weights,
        w13_scale_name="w13_weight_scale",
        w2_scale_name="w2_weight_scale",
    )


def prepare_mxfp8_moe_weights_for_flashinfer_megamoe(
    layer: FusedMoE,
) -> None:
    _init_flashinfer_megamoe_layer_state(layer)

    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_mxfp8_cutedsl_mega_weights,
    )

    if layer.hidden_size % 128 != 0:
        raise ValueError(
            "FlashInfer MXFP8 MegaMOE requires hidden_size to be a multiple "
            f"of 128, got {layer.hidden_size}."
        )
    if layer.intermediate_size_per_partition % 128 != 0:
        raise ValueError(
            "FlashInfer MXFP8 MegaMOE requires intermediate_size_per_partition "
            f"to be a multiple of 128, got {layer.intermediate_size_per_partition}."
        )
    if layer.num_experts % layer.moe_ep_size != 0:
        raise ValueError(
            "FlashInfer MXFP8 MegaMOE requires num_experts to be divisible by "
            f"ep_size, got {layer.num_experts=} and {layer.moe_ep_size=}."
        )

    weights = MoEWeightPack(
        w13=layer.w13_weight.data,
        w2=layer.w2_weight.data,
        w13_scale=layer.w13_weight_scale_inv.data,
        w2_scale=layer.w2_weight_scale_inv.data,
    )
    transformed_weights = preprocess_mxfp8_cutedsl_mega_weights(
        weights,
        intermediate_size=layer.intermediate_size_per_partition,
        hidden_size=layer.hidden_size,
        kind="mxfp8_e4m3",
        gate_up_clamp=layer.moe_runner_config.swiglu_limit,
        activation_clamp=None,
    )
    _bind_transformed_weights(
        layer,
        transformed_weights,
        w13_scale_name="w13_weight_scale_inv",
        w2_scale_name="w2_weight_scale_inv",
    )


def _ensure_shared_workspace(mega: Any) -> None:
    """Point this layer at a runtime-context-owned shared symmetric buffer.

    FlashInfer's own workspace_pool shares by a key that includes
    ``epilogue_pool_key(fc1_alpha/fc2_alpha/fc1_norm_const)`` (see
    ``flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.backend
    ._workspace_pool_key``), and that key is identity-keyed for tensors, not
    value-keyed. sglang binds distinct per-layer alpha/norm-const tensor
    objects on every FusedMoE layer, so FlashInfer's pool never actually hits
    across layers and every layer independently pays a full CuteDSL compile
    (removed in b07d9012a3 "fix: rely on FlashInfer MegaMoE workspace
    pooling", which assumed FlashInfer's pooling alone was equivalent).

    All layers share identical fleet/kernel geometry, and the alpha/norm-const
    values are already re-staged into the workspace per forward via
    ``stage_inputs()`` -- they are not baked into the compiled kernel -- so it
    is safe to key sharing on geometry alone and skip FlashInfer's per-tensor
    identity check entirely.

    The first mega layer's first forward creates it (collective; safe because
    warmup runs the same layer on all ranks in lockstep); later layers reuse it.
    """
    if getattr(mega, "_workspace", None) is not None:
        return
    fp = mega._fleet_params
    kc = mega._megakernel_config
    mc = mega._mega_config
    from sglang.srt.runtime_context import get_resources

    key = (
        getattr(kc, "kernel_name", kc.__class__.__name__),
        mega._bootstrap.world_size,
        fp.num_experts,
        fp.max_tokens_per_rank,
        fp.token_hidden_size,
        kc.top_k,
        kc.intermediate_size,
        getattr(kc, "gate_up_clamp", None),
        getattr(kc, "activation_clamp", None),
        getattr(kc, "apply_topk_in_fc1", None),
        getattr(kc, "kind", None),
        getattr(kc, "in_kernel_fc2_reduce", None),
        getattr(kc, "combine_dtype", None),
        getattr(kc, "token_back_by_dispatch", None),
        getattr(kc, "fast_math", None),
        mc.quantize_input,
    )
    workspaces = get_resources().flashinfer_megamoe_workspaces
    shared = workspaces.get(key)
    if shared is None:
        workspaces[key] = mega._ensure_workspace()
    else:
        mega._workspace = shared


@register_fused_func("flashinfer_megamoe", "flashinfer_megamoe")
def run_flashinfer_megamoe(
    dispatch_output: DispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Run the fused mega kernel and return per-rank outputs (no combine)."""
    from flashinfer.moe_ep import MoEEpTensors

    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    assert isinstance(quant_info, FlashInferMegaMoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_megamoe: {type(quant_info)}"
    )

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    mega = quant_info.mega
    _ensure_shared_workspace(mega)

    t = MoEEpTensors(
        hidden_states=x.to(torch.bfloat16),
        # FlashInfer's fused staging accepts the int32 router output and widens
        # directly into its final int64 workspace buffer. Keep this path copy-free.
        topk_ids=topk_ids,
        topk_weights=topk_weights.to(torch.float32),
        fc1_alpha=quant_info.fc1_alpha,
        fc2_alpha=quant_info.fc2_alpha,
        fc1_norm_const=quant_info.fc1_norm_const,
    )
    with _capture_safe_ue8m0_pack():
        assert quant_info.mega_forward is not None
        y = quant_info.mega_forward(mega, t)

    if quant_info.apply_routed_scaling_factor:
        rsf = runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0:
            y.mul_(rsf)

    return StandardCombineInput(hidden_states=y)


def warmup_all_flashinfer_megamoe_layers(model: torch.nn.Module) -> None:
    """Force every FlashInfer MegaMOE layer to fully build before CUDA graph
    capture starts.

    ``ensure_*_moe_layer_for_flashinfer_megamoe`` build lazily on the layer's
    first forward (see "Initialize MegaMOE layer state lazily"). Nothing
    actually guarantees that first forward happens during warmup, outside of
    any graph capture -- if a layer's first real forward instead happens
    while a CUDA graph is being captured (e.g. it wasn't exercised by the
    warmup dummy batch/shape), its lazily-computed state (nvfp4's
    ``input_norm_const``) is still ``None`` when capture reaches it, and
    ``ensure_nvfp4_moe_layer_for_flashinfer_megamoe`` raises -- capture
    forbids the fallback's blocking device sync, so it can't silently
    recover the way warmup can.

    Walk every FusedMoE layer once, explicitly, right before capture begins
    (see ``ModelRunner.init_cuda_graphs``), so this always happens eagerly
    and outside any graph, regardless of whether warmup's dummy batches
    happened to route through every layer.

    Only implements the nvfp4 path (this repro is nvfp4-only); extend the
    dispatch below if fp4/mxfp8 MegaMOE hits the same gap.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    n_built = 0
    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        if not hasattr(module, "_flashinfer_megamoe_layer"):
            # This layer's quant method never went through one of the
            # prepare_*_moe_weights_for_flashinfer_megamoe hooks -- not a
            # MegaMOE layer (or not on the flashinfer_megamoe backend).
            continue
        if getattr(module, "_flashinfer_megamoe_layer", None) is not None:
            continue  # already built (e.g. warmup's dummy batch hit it)

        # Dispatch mirrors modelopt_quant.py's apply(): only the nvfp4
        # method is wired up here today.
        if type(module.quant_method).__name__ == "ModelOptNvFp4FusedMoEMethod":
            ensure_nvfp4_moe_layer_for_flashinfer_megamoe(module)
            n_built += 1
        else:
            logger.warning(
                "warmup_all_flashinfer_megamoe_layers: layer[%s] uses "
                "quant_method=%s, which this eager pre-capture warmup does "
                "not know how to build. If capture then fails with "
                "'must be initialized before CUDA graph capture', add a "
                "branch here for that quant method.",
                getattr(module, "layer_id", "?"),
                type(module.quant_method).__name__,
            )

    if n_built:
        logger.info(
            "warmup_all_flashinfer_megamoe_layers: eagerly built %d "
            "FlashInfer MegaMOE layer(s) before CUDA graph capture.",
            n_built,
        )
