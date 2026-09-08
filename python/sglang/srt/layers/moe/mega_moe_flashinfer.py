# Copyright 2026 SGLang Team
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
"""FlashInfer `sm100_nvfp4_nvfp4_bf16_cutedsl` MegaMoE for ModelOpt NVFP4 experts.

Selected by `--moe-a2a-backend megamoe --megamoe-backend flashinfer_cutedsl`.
The fused kernel (`flashinfer.moe_ep.MoEEpMegaLayer`) owns EP dispatch, both
grouped GEMMs, SwiGLU and combine over an NVSHMEM symmetric heap; sglang only
hands it the checkpoint's NVFP4 weights and the ModelOpt scales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens

if TYPE_CHECKING:
    from flashinfer.moe_ep import MoEEpMegaLayer

# ModelOpt NVFP4 convention: a tensor x is stored as q * sf * s, where sf is the
# per-16 UE4M3 block scale and s the per-tensor "scale_2" / "input_scale". The
# FlashInfer kernel quantizes activations with sf = amax/6 * norm_const, so the
# dequantized GEMM operands carry x * norm_const; alpha then restores x * w:
#   fc1: norm_const = 1 / s_x1, alpha = s_x1 * s_w13
#   fc2: norm_const = 1 / s_x2, alpha = s_x2 * s_w2
# fc1's norm_const is one float for the whole session, so s_x1 is scalarized
# the same way the CuteDSL fused-MoE runner does (max over experts).


class FlashInferMegaMoeScales(NamedTuple):
    input_norm_const: float
    fc1_alpha: torch.Tensor
    fc1_norm_const: torch.Tensor
    fc2_alpha: torch.Tensor


class FlashInferMegaMoe(NamedTuple):
    # (token capacity, session) pairs, smallest capacity first; a batch runs in
    # the first session whose capacity fits it.
    sessions: tuple[tuple[int, MoEEpMegaLayer], ...]
    scales: FlashInferMegaMoeScales


def compute_flashinfer_mega_moe_scales(
    *,
    w13_input_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
    w2_input_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    ep_rank: int,
    num_local_experts: int,
) -> FlashInferMegaMoeScales:
    w13_input_scale = w13_input_scale.to(torch.float32).reshape(-1)
    used_input_scale = w13_input_scale.max()

    if w13_weight_scale_2.dim() == 2 and w13_weight_scale_2.shape[1] >= 2:
        gate_scale = w13_weight_scale_2[:, 0]
        up_scale = w13_weight_scale_2[:, 1]
        if not torch.allclose(gate_scale, up_scale):
            raise ValueError(
                "FlashInfer MegaMoE applies one fc1 alpha per expert; this "
                "checkpoint has different gate and up weight_scale_2 values."
            )
    else:
        gate_scale = w13_weight_scale_2.reshape(-1)
    gate_scale = gate_scale.to(torch.float32)

    w2_input_scale = w2_input_scale.to(torch.float32).reshape(-1)
    if w2_input_scale.numel() != num_local_experts:
        start = ep_rank * num_local_experts
        w2_input_scale = w2_input_scale[start : start + num_local_experts]
    w2_weight_scale_2 = w2_weight_scale_2.to(torch.float32).reshape(-1)

    return FlashInferMegaMoeScales(
        input_norm_const=float(1.0 / used_input_scale.item()),
        fc1_alpha=(used_input_scale * gate_scale).contiguous(),
        fc1_norm_const=(1.0 / w2_input_scale).contiguous(),
        fc2_alpha=(w2_input_scale * w2_weight_scale_2).contiguous(),
    )


def build_flashinfer_mega_moe_layer(
    layer: torch.nn.Module,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    activation_clamp: float | None,
) -> FlashInferMegaMoe:
    # `layer` is a FusedMoE holding raw ModelOpt NVFP4 params: packed e2m1
    # weights [E_local, N, K/2] (uint8) and UE4M3 block scales [E_local, N, K/16].
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MoEEpMegaLayer
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.modes.config import MegaConfig
    from flashinfer.moe_ep.weights import PrequantizedMoEWeights

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    scales = compute_flashinfer_mega_moe_scales(
        w13_input_scale=layer.w13_input_scale,
        w13_weight_scale_2=layer.w13_weight_scale_2,
        w2_input_scale=layer.w2_input_scale,
        w2_weight_scale_2=layer.w2_weight_scale_2,
        ep_rank=layer.moe_ep_rank,
        num_local_experts=layer.num_local_experts,
    )
    weights = PrequantizedMoEWeights(
        w13=layer.w13_weight.data.view(torch.uint8),
        w2=layer.w2_weight.data.view(torch.uint8),
        w13_scale=layer.w13_weight_scale.data.view(torch.float8_e4m3fn),
        w2_scale=layer.w2_weight_scale.data.view(torch.float8_e4m3fn),
    )
    # The per-expert alphas go in per forward (MoEEpTensors), not in the
    # config: config-level epilogue tensors are part of the workspace pool key,
    # so baking them in would give every MoE layer its own multi-GB symmetric
    # workspace and compiled session instead of one shared per geometry.
    kernel_config = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=intermediate_size,
        top_k=top_k,
        activation_clamp=activation_clamp,
        input_norm_const=scales.input_norm_const,
    )
    ep_group = get_moe_ep_group().device_group
    bootstrap = BootstrapConfig(
        world_size=layer.moe_ep_size,
        rank=layer.moe_ep_rank,
        device=torch.cuda.current_device(),
        process_group=ep_group,
    )
    # One kernel-ready weight copy shared by every session of this layer.
    transformed = preprocess_mega_weights(
        weights,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        activation_clamp=activation_clamp,
    )
    # The kernel keeps its own interleaved / swizzled copies; drop the
    # checkpoint-layout params so the experts are not held twice.
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        param = getattr(layer, name)
        param.data = param.data.new_empty(0)

    full_capacity = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    decode_capacity = envs.SGLANG_FLASHINFER_MEGAMOE_DECODE_MAX_TOKENS_PER_RANK.get()
    capacities = [full_capacity]
    if 0 < decode_capacity < full_capacity:
        capacities.insert(0, decode_capacity)
    sessions = []
    for capacity in capacities:
        fleet = FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden_size,
        )
        session = MoEEpMegaLayer(
            bootstrap,
            fleet,
            weights,
            MegaConfig(megakernel=kernel_config, transformed_weights=transformed),
        )
        # Collective: allocates the symmetric workspace and compiles the kernel
        # on every EP rank before any CUDA graph capture can reach forward().
        session.warmup()
        sessions.append((capacity, session))
    return FlashInferMegaMoe(sessions=tuple(sessions), scales=scales)


def run_flashinfer_mega_routed(
    mega: FlashInferMegaMoe,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    *,
    top_k: int,
    num_tokens: int,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    from flashinfer.moe_ep.tensors import MoEEpTensors

    # The fused staging kernel reads the routing as raw int64 / fp32 buffers;
    # sglang's TopK emits int32 ids. A rank with no tokens still joins the
    # collective launch with an empty batch.
    if num_tokens > 0:
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(torch.float32)
    else:
        topk_ids = hidden_states.new_empty((0, top_k), dtype=torch.int64)
        topk_weights = hidden_states.new_empty((0, top_k), dtype=torch.float32)
    # The launch is collective, so every EP rank must pick the same session:
    # size it by the largest per-rank batch in this step (DP attention gives
    # ranks different token counts), not by the local one.
    global_num_tokens = get_dp_global_num_tokens()
    step_tokens = max(global_num_tokens) if global_num_tokens else num_tokens
    session = mega.sessions[-1][1]
    for capacity, candidate in mega.sessions:
        if step_tokens <= capacity:
            session = candidate
            break
    y = session(
        MoEEpTensors(
            hidden_states=hidden_states[:num_tokens].contiguous(),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            fc1_alpha=mega.scales.fc1_alpha,
            fc2_alpha=mega.scales.fc2_alpha,
            fc1_norm_const=mega.scales.fc1_norm_const,
        )
    )
    if routed_scaling_factor != 1.0:
        y.mul_(routed_scaling_factor)
    return y
