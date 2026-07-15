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

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        DispatchOutput,
        StandardCombineInput,
    )


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


def _global_or_local_expert_vector(
    value: torch.Tensor, layer: FusedMoE, *, name: str
) -> torch.Tensor:
    value = value.detach().to(torch.float32)
    if value.dim() == 0:
        return value.expand(layer.num_local_experts).contiguous()
    if value.shape == (layer.num_local_experts,):
        return value.contiguous()
    if value.shape == (layer.num_experts,):
        start = layer.moe_ep_rank * layer.num_local_experts
        end = start + layer.num_local_experts
        return value[start:end].contiguous()
    raise ValueError(
        f"{name} must be scalar, local shape ({layer.num_local_experts},), "
        f"or global shape ({layer.num_experts},); got {tuple(value.shape)}"
    )


def _build_nvfp4_megamoe_scales(layer: FusedMoE) -> None:
    gate_alpha = _local_expert_vector(layer.g1_alphas, layer.num_local_experts)
    if layer.moe_runner_config.is_gated:
        up_alpha = _local_expert_vector(layer.g1_alphas_up, layer.num_local_experts)
        if not torch.allclose(gate_alpha, up_alpha):
            raise ValueError(
                "FlashInfer NVFP4 MegaMOE requires matching gate/up FC1 alpha "
                "values because the kernel accepts one alpha per expert."
            )
        fc1_alpha = gate_alpha.contiguous()
    else:
        fc1_alpha = gate_alpha

    fc2_input_scale = _global_or_local_expert_vector(
        layer.w2_input_scale, layer, name="w2_input_scale"
    )
    fc2_weight_scale_2 = _local_expert_vector(
        layer.w2_weight_scale_2, layer.num_local_experts
    )

    layer.flashinfer_megamoe_fc1_alpha = fc1_alpha
    layer.flashinfer_megamoe_fc2_alpha = (
        (fc2_input_scale * fc2_weight_scale_2).to(torch.float32).contiguous()
    )
    layer.flashinfer_megamoe_fc1_norm_const = (
        (1 / fc2_input_scale).to(torch.float32).contiguous()
    )


def _build_flashinfer_cutedsl_megamoe_layer(
    layer: FusedMoE, *, megakernel_config: object, weights: object
) -> None:
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
    )

    world_size, rank = _layer_ep_world_rank(layer)

    layer.flashinfer_megamoe_layer = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=layer.num_experts,
            max_tokens_per_rank=_resolve_max_tokens_per_rank(),
            token_hidden_size=layer.hidden_size,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=megakernel_config,
            preprocess_weights=True,
        ),
    )


def build_flashinfer_megamoe_layer(layer: FusedMoE) -> None:
    """Construct + cache a MoEEpMegaLayer from the layer's loaded FP4 weights.

    SGLang loads FP4-packed expert weights plus raw block scales. FlashInfer's
    current moe_ep API owns backend-specific weight preprocessing, including
    DeepGEMM scale layout transforms.
    """
    if getattr(layer, "flashinfer_megamoe_layer", None) is not None:
        return

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
    )

    world_size, rank = _layer_ep_world_rank(layer)

    layer.flashinfer_megamoe_layer = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=layer.num_experts,
            max_tokens_per_rank=_resolve_max_tokens_per_rank(),
            token_hidden_size=layer.hidden_size,
        ),
        weights=MoEWeightPack(
            w13=layer.w13_weight.data,
            w2=layer.w2_weight.data,
            w13_scale=layer.w13_weight_scale_inv.data,
            w2_scale=layer.w2_weight_scale_inv.data,
        ),
        backend=MegaConfig(
            megakernel=DeepGemmMegaMoeConfig(
                intermediate_size=layer.intermediate_size_per_partition,
                top_k=layer.top_k,
                activation_clamp=layer.moe_runner_config.swiglu_limit,
            ),
            preprocess_weights=True,
        ),
    )


def build_flashinfer_nvfp4_megamoe_layer(layer: FusedMoE) -> None:
    if getattr(layer, "flashinfer_megamoe_layer", None) is not None:
        return

    from flashinfer.moe_ep import (
        MoEWeightPack,
        Nvfp4CutedslMegaMoeConfig,
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

    _build_nvfp4_megamoe_scales(layer)

    input_norm_const = _scalar_float(layer.w13_input_scale_quant)
    layer.flashinfer_megamoe_input_norm_const = input_norm_const
    gate_up_clamp = layer.moe_runner_config.swiglu_limit

    _build_flashinfer_cutedsl_megamoe_layer(
        layer,
        weights=MoEWeightPack(
            w13=layer.w13_weight.data,
            w2=layer.w2_weight.data,
            w13_scale=layer.w13_weight_scale.data,
            w2_scale=layer.w2_weight_scale.data,
        ),
        megakernel_config=Nvfp4CutedslMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            gate_up_clamp=gate_up_clamp,
            apply_topk_in_fc1=True,
            input_norm_const=input_norm_const,
            fc1_alpha=layer.flashinfer_megamoe_fc1_alpha,
            fc2_alpha=layer.flashinfer_megamoe_fc2_alpha,
            fc1_norm_const=layer.flashinfer_megamoe_fc1_norm_const,
        ),
    )


def build_flashinfer_mxfp8_megamoe_layer(layer: FusedMoE) -> None:
    if getattr(layer, "flashinfer_megamoe_layer", None) is not None:
        return

    from flashinfer.moe_ep import MoEWeightPack, Mxfp8CutedslMegaMoeConfig

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

    _build_flashinfer_cutedsl_megamoe_layer(
        layer,
        weights=MoEWeightPack(
            w13=layer.w13_weight.data,
            w2=layer.w2_weight.data,
            w13_scale=layer.w13_weight_scale_inv.data,
            w2_scale=layer.w2_weight_scale_inv.data,
        ),
        megakernel_config=Mxfp8CutedslMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            kind="mxfp8_e4m3",
            gate_up_clamp=layer.moe_runner_config.swiglu_limit,
        ),
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


def run_flashinfer_megamoe(
    layer: FusedMoE, dispatch_output: DispatchOutput
) -> StandardCombineInput:
    """Run the fused mega kernel and return per-rank outputs (no combine)."""
    from flashinfer.moe_ep import MoEEpTensors

    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    mega = layer.flashinfer_megamoe_layer
    _ensure_shared_workspace(mega)

    t = MoEEpTensors(
        hidden_states=x.to(torch.bfloat16),
        topk_ids=topk_ids.to(torch.int64),
        topk_weights=topk_weights.to(torch.float32),
        fc1_alpha=getattr(layer, "flashinfer_megamoe_fc1_alpha", None),
        fc2_alpha=getattr(layer, "flashinfer_megamoe_fc2_alpha", None),
        fc1_norm_const=getattr(layer, "flashinfer_megamoe_fc1_norm_const", None),
    )
    y = mega.forward(t)

    # The kernel's combine applies topk_weights but not routed_scaling_factor.
    # When it isn't fused into topk upstream (HashTopK layers can't fuse it),
    # apply it here -- mirrors the deepgemm megamoe path in mega_moe.py.
    if not layer.should_fuse_routed_scaling_factor_in_topk:
        rsf = layer.moe_runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0:
            y.mul_(rsf)

    return StandardCombineInput(hidden_states=y)
