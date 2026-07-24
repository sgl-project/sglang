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
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
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


_UE8M0_PACK_PATCHED = False


def _install_capture_safe_ue8m0_pack() -> None:
    """Make deep_gemm's UE8M0 scale packing safe under CUDA graph capture.

    The deep_gemm mega staging path (block-FP8 models such as DeepSeek-V4-Flash)
    runs ``per_token_cast_to_fp8(..., use_packed_ue8m0=True)`` on every forward,
    which calls ``deep_gemm.utils.math.pack_ue8m0_to_int``. Its upstream
    implementation carries two debug assertions::

        assert (x_int >= 0).all() and (x_int & 0x7fffff == 0).all()

    ``.all()`` forces a device->host sync, which is illegal while a CUDA graph is
    capturing and aborts sglang's decode cuda-graph capture with
    ``cudaErrorStreamCaptureUnsupported``. The assertions are pure sanity checks;
    the packing (``x_int >> 23``) is deterministic and unaffected by dropping
    them, so we swap in a capture-safe variant.

    TODO(flashinfer/deep_gemm): remove once upstream gates these asserts behind a
    debug env or moves them off the capture path.
    """
    global _UE8M0_PACK_PATCHED
    if _UE8M0_PACK_PATCHED:
        return

    try:
        import deep_gemm.utils.math as _dgm
    except ImportError:
        # deep_gemm is only needed by the block-FP8 mega path; NVFP4/MXFP8 mega
        # runs on cutedsl and does not import it. Nothing to patch here.
        return

    def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
        x_int = x.view(torch.int)
        return (x_int >> 23).to(torch.uint8).view(torch.int)

    _dgm.pack_ue8m0_to_int = _pack_ue8m0_to_int
    _UE8M0_PACK_PATCHED = True


# Selecting the flashinfer_megamoe backend imports this module (see MoeRunner /
# the fp8 / modelopt quant methods), so install the capture-safe shim here.
_install_capture_safe_ue8m0_pack()


@dataclass
class FlashInferMegaMoeQuantInfo(MoeQuantInfo):
    mega: Any
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    fc1_norm_const: torch.Tensor | None = None
    apply_routed_scaling_factor: bool = False


def _resolve_max_tokens_per_rank() -> int:
    """Per-rank symmetric-buffer sizing for the mega kernel.

    Honors the explicit env override; otherwise derives the largest per-(DP)rank
    token count a single MoE forward can route (same bound the cutedsl A2A path
    uses), falling back to 1024 if it cannot be determined.
    """
    configured = envs.SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK.get()
    if configured > 0:
        return configured

    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    derived = 0
    if server_args is not None:
        derived = server_args.cutedsl_moe_max_num_tokens()
    return derived if derived > 0 else 1024


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
        return float(value.detach().to(torch.float32).max().item())
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


def _ensure_flashinfer_megamoe_layer(
    layer: FusedMoE,
    *,
    megakernel_config: Any,
    w13_scale_name: str,
    w2_scale_name: str,
) -> Any:
    mega = getattr(layer, "_flashinfer_megamoe_layer", None)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
    )

    w13_scale = getattr(layer, w13_scale_name)
    w2_scale = getattr(layer, w2_scale_name)
    transformed_weights = (
        (layer.w13_weight.data, w13_scale.data),
        (layer.w2_weight.data, w2_scale.data),
    )
    world_size, rank = _layer_ep_world_rank(layer)

    max_tokens_per_rank = _resolve_max_tokens_per_rank()
    logger.info(
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
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
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
    return mega


def ensure_fp4_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = getattr(layer, "_flashinfer_megamoe_layer", None)
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
        w13_scale_name="w13_weight_scale_inv",
        w2_scale_name="w2_weight_scale_inv",
    )


def ensure_nvfp4_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = getattr(layer, "_flashinfer_megamoe_layer", None)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import Nvfp4CutedslMegaMoeConfig

    return _ensure_flashinfer_megamoe_layer(
        layer,
        megakernel_config=Nvfp4CutedslMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            gate_up_clamp=layer.moe_runner_config.swiglu_limit,
            apply_topk_in_fc1=True,
            in_kernel_fc2_reduce=envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get(),
            input_norm_const=_scalar_float(layer.w13_input_scale_quant),
            fc1_alpha=layer.g1_alphas,
            fc2_alpha=layer.g2_alphas,
            fc1_norm_const=layer.w2_input_scale_quant,
        ),
        w13_scale_name="w13_weight_scale",
        w2_scale_name="w2_weight_scale",
    )


def ensure_mxfp8_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = getattr(layer, "_flashinfer_megamoe_layer", None)
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
        w13_scale_name="w13_weight_scale_inv",
        w2_scale_name="w2_weight_scale_inv",
    )


def prepare_fp4_moe_weights_for_flashinfer_megamoe(
    layer: FusedMoE,
) -> None:
    """Prepare loaded FP4 weights for MegaMOE.

    SGLang loads FP4-packed expert weights plus raw block scales. FlashInfer's
    current moe_ep API owns backend-specific weight preprocessing, including
    DeepGEMM scale layout transforms.
    """
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
    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_nvfp4_cutedsl_mega_weights,
    )

    if layer.hidden_size % 128 != 0:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires hidden_size to be a multiple "
            f"of 128, got {layer.hidden_size}."
        )
    if getattr(layer.quant_config, "use_per_token_activation", False):
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


# One symmetric-memory workspace shared across all mega layers. FlashInfer's
# MoEEpMegaLayer allocates a workspace per instance; MoE layers run
# sequentially and the workspace is weight-independent, so a per-layer buffer
# would multiply device memory by the layer count (and OOM). Keyed by the
# durable sizing so distinct shapes still get distinct buffers. Mirrors the
# shared symm-buffer cache in the deepgemm mega_moe path.
_SHARED_MEGA_WORKSPACE: dict = {}


def _ensure_shared_workspace(mega) -> None:
    """Point this layer's workspace at the process-wide shared symm buffer.

    The first mega layer's first forward creates it (collective; safe because
    warmup runs the same layer on all ranks in lockstep); later layers reuse it.
    """
    if getattr(mega, "_workspace", None) is not None:
        return
    fp = mega._fleet_params
    kc = mega._megakernel_config
    mc = mega._mega_config
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
        getattr(kc, "token_back_by_dispatch", None),
        getattr(kc, "fast_math", None),
        mc.quantize_input,
    )
    shared = _SHARED_MEGA_WORKSPACE.get(key)
    if shared is None:
        _SHARED_MEGA_WORKSPACE[key] = mega._ensure_workspace()
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

    assert isinstance(
        quant_info, FlashInferMegaMoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_megamoe: {type(quant_info)}"

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    mega = quant_info.mega
    _ensure_shared_workspace(mega)

    t = MoEEpTensors(
        hidden_states=x.to(torch.bfloat16),
        topk_ids=topk_ids.to(torch.int64),
        topk_weights=topk_weights.to(torch.float32),
        fc1_alpha=quant_info.fc1_alpha,
        fc2_alpha=quant_info.fc2_alpha,
        fc1_norm_const=quant_info.fc1_norm_const,
    )
    y = mega.forward(t)

    if quant_info.apply_routed_scaling_factor:
        rsf = runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0:
            y.mul_(rsf)

    return StandardCombineInput(hidden_states=y)
