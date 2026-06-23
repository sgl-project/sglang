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
"""Generic FlashInfer MegaMOE backend (moe_ep_v2.MoEEpMegaLayer).

Wraps FlashInfer's fused EP all-to-all + expert-compute mega kernel so it can
be selected as a model-agnostic MoE runner backend through the standard
FusedMoE dispatch -> run_moe_core -> combine flow. The mega kernel does its EP
communication internally via the deep_gemm symmetric buffer, so the dispatcher
and combine stay pure no-ops; this module owns the layer build + forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def build_flashinfer_megamoe_layer(layer: FusedMoE) -> None:
    """Construct + cache a MoEEpMegaLayer from the layer's loaded FP4 weights.

    SGLang loads FP4-packed expert weights (int8, k//2) plus fp32 block-32
    scales. FlashInfer's pre-quantized weight path feeds (weight, scale)
    straight into ``transform_weights_for_mega_moe`` without first calling
    ``transform_sf_into_required_layout``, so we transform the scales here to
    the UE8M0 (1, 32) layout -- mirroring ``build_mega_moe_experts_weights``.
    """
    if getattr(layer, "flashinfer_megamoe_layer", None) is not None:
        return

    import torch.distributed as dist
    from deep_gemm import transform_sf_into_required_layout
    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
    )

    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_sf_fp32 = layer.w13_weight_scale_inv.data
    w2_sf_fp32 = layer.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    _, n2, half_k2 = w2.shape
    k2 = half_k2 * 2

    w13_sf = transform_sf_into_required_layout(
        w13_sf_fp32,
        mn=n1,
        k=k1,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    w2_sf = transform_sf_into_required_layout(
        w2_sf_fp32,
        mn=n2,
        k=k2,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # The kernel hardcodes dist.group.WORLD, so EP == world (TP == EP == world).
    layer.flashinfer_megamoe_layer = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=layer.num_experts,
            max_tokens_per_rank=_resolve_max_tokens_per_rank(),
            token_hidden_size=layer.hidden_size,
            weights=MoEWeightPack(w13=w13, w2=w2, w13_scale=w13_sf, w2_scale=w2_sf),
        ),
        backend=MegaConfig(
            megakernel=DeepGemmMegaMoeConfig(
                intermediate_size=layer.intermediate_size_per_partition,
                top_k=layer.top_k,
                activation_clamp=layer.moe_runner_config.swiglu_limit,
            ),
            stage_inputs=True,
            preprocess_weights=True,
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
    key = (
        mega._bootstrap.world_size,
        fp.num_experts,
        fp.max_tokens_per_rank,
        fp.token_hidden_size,
        kc.top_k,
        kc.intermediate_size,
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
    from flashinfer.moe_ep_v2 import MoEEpTensors

    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    mega = layer.flashinfer_megamoe_layer
    _ensure_shared_workspace(mega)
    num_tokens = x.shape[0]

    hidden = x.to(torch.bfloat16)
    ids = topk_ids.to(torch.int64)
    weights = topk_weights.to(torch.float32)

    if num_tokens == 0:
        # The mega kernel is collective (symm-buffer all-to-all), so every rank
        # must launch it even with no local tokens. Pad one dummy token (routed
        # to expert 0) and drop it from the output to keep ranks in lockstep.
        hidden = x.new_zeros((1, layer.hidden_size), dtype=torch.bfloat16)
        ids = ids.new_zeros((1, layer.top_k))
        weights = weights.new_zeros((1, layer.top_k))

    y = mega.forward(
        MoEEpTensors(hidden_states=hidden, topk_ids=ids, topk_weights=weights)
    )

    # The kernel's combine applies topk_weights but not routed_scaling_factor.
    # When it isn't fused into topk upstream (HashTopK layers can't fuse it),
    # apply it here -- mirrors the deepgemm megamoe path in mega_moe.py.
    if not layer.should_fuse_routed_scaling_factor_in_topk:
        rsf = layer.moe_runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0:
            y.mul_(rsf)

    if num_tokens == 0:
        y = y[:0]

    return StandardCombineInput(hidden_states=y)
