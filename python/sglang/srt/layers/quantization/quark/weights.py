# SPDX-License-Identifier: Apache-2.0

import math
import re

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


def load_gptoss_weight_quark(
    model,
    weights,
    *,
    is_nextn: bool,
    weight_name_mapping,
) -> None:
    # Regex matching `model.layers.{L}.mlp.experts.{N}.{gate_up_proj|down_proj}.{suffix}`
    # used by the AMD Quark GPT-OSS per-expert checkpoint layout.
    quark_expert_pat = re.compile(
        r"^(.*\.mlp\.experts)\.(\d+)\.(gate_up_proj|down_proj)\."
        r"(weight|weight_scale|input_scale|bias)$"
    )
    quark_experts_weights = []
    normal_weights = []

    for name, weight in weights:
        if quark_expert_pat.match(name) is not None:
            quark_experts_weights.append((name, weight))
        else:
            normal_weights.append((name, weight))

    quark_loaded = _load_gptoss_quark_expert_weights(
        model, quark_experts_weights, quark_expert_pat
    )
    model._load_normal_weights(
        normal_weights,
        is_nextn=is_nextn,
        weight_name_mapping=weight_name_mapping,
        other_loaded_param_names=quark_loaded,
    )


def _load_gptoss_quark_expert_weights(model, weights, quark_expert_pat):
    """GPT-OSS per-expert style loader for Quark MoE tensors into padded fused buffers.

    Quark stores each expert separately:
        experts.{N}.gate_up_proj.{weight,weight_scale,input_scale,bias}
        experts.{N}.down_proj.{weight,weight_scale,input_scale,bias}

    We mirror the static MXFP4 expert loader: slice the checkpoint along
    the TP-sharded dimension (intermediate axis) and copy into a window
    of the padded ``w13_*`` / ``w2_*`` parameters allocated by
    :class:`QuarkW4A8MXFp4MoE`. Down-proj bias is loaded only on
    ``moe_tp_rank == 0`` to avoid double-counting after all-reduce.
    """
    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()
    mxfp4_block = 32

    moe_tp_rank = get_parallel().moe_tp_rank
    moe_tp_size = get_parallel().moe_tp_size
    moe_ep_rank = get_parallel().moe_ep_rank
    moe_ep_size = get_parallel().moe_ep_size

    intermediate_size = model.config.intermediate_size
    assert (
        intermediate_size % mxfp4_block == 0
    ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"
    intermediate_size_block = intermediate_size // mxfp4_block

    per_rank_intermediate_size_block = math.ceil(intermediate_size_block / moe_tp_size)

    per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

    # Calculate common slicing bounds for current rank
    assert model.config.num_local_experts % moe_ep_size == 0
    moe_num_local_experts = model.config.num_local_experts // moe_ep_size

    moe_tp_rank_start = moe_tp_rank * per_rank_intermediate_size
    moe_tp_rank_end = min(
        (moe_tp_rank + 1) * per_rank_intermediate_size, intermediate_size
    )

    moe_ep_rank_start = moe_ep_rank * moe_num_local_experts
    moe_ep_rank_end = (moe_ep_rank + 1) * moe_num_local_experts

    for name, weight in weights:
        # Quark stores experts separately as
        # `experts.{N}.{gate_up_proj|down_proj}.{suffix}`; pull the
        # expert id out of the name (mxfp4 has it as axis 0 instead).
        m = quark_expert_pat.match(name)
        if m is None:
            continue
        prefix, expert_str, proj, suffix = m.groups()
        global_expert_id = int(expert_str)
        if global_expert_id < moe_ep_rank_start or global_expert_id >= moe_ep_rank_end:
            continue
        local_expert_id = global_expert_id - moe_ep_rank_start

        if _is_cuda:
            weight = weight.cuda()

        dispatch_key = f"{proj}.{suffix}"

        if dispatch_key == "gate_up_proj.weight":
            # Handle MLP gate and up projection weights
            new_name = f"{prefix}.w13_weight"

            # De-interleave gate/up rows ([g0,u0,g1,u1,...] -> [g..., u...])
            # then slice the TP window. Each half is written into its own
            # slot of the padded fused buffer; the gap between halves is
            # pre-zeroed by `create_weights` and must not be overwritten.
            narrow_gate = weight[0::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()
            narrow_up = weight[1::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()

            param = params_dict[new_name]
            intermediate_pad = param.data.shape[1] // 2
            g0, g1 = narrow_gate.shape
            u0, u1 = narrow_up.shape
            param.data[local_expert_id, :g0, :g1].copy_(
                narrow_gate.to(param.data.dtype)
            )
            param.data[
                local_expert_id,
                intermediate_pad : intermediate_pad + u0,
                :u1,
            ].copy_(narrow_up.to(param.data.dtype))
            loaded_params.add(new_name)

        elif dispatch_key == "down_proj.weight":
            # Handle MLP down projection weights
            # packed FP4 -> halve the TP bound on the contracting K dim
            new_name = f"{prefix}.w2_weight"

            narrow_weight = weight[
                ...,
                moe_tp_rank_start // 2 : moe_tp_rank_end // 2,
            ]

            param = params_dict[new_name]
            d0, d1 = narrow_weight.shape
            param.data[local_expert_id, :d0, :d1].copy_(
                narrow_weight.to(param.data.dtype)
            )
            loaded_params.add(new_name)

        elif dispatch_key == "gate_up_proj.weight_scale":
            # Handle MLP gate and up projection weight scales
            new_name = f"{prefix}.w13_weight_scale"

            narrow_gate = weight[0::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()
            narrow_up = weight[1::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()

            param = params_dict[new_name]
            intermediate_pad = param.data.shape[1] // 2
            g0, g1 = narrow_gate.shape
            u0, u1 = narrow_up.shape
            param.data[local_expert_id, :g0, :g1].copy_(
                narrow_gate.to(param.data.dtype)
            )
            param.data[
                local_expert_id,
                intermediate_pad : intermediate_pad + u0,
                :u1,
            ].copy_(narrow_up.to(param.data.dtype))
            loaded_params.add(new_name)

        elif dispatch_key == "down_proj.weight_scale":
            # Handle MLP down projection weight scales
            # 32 fp4 values per block -> slice by mxfp4_block
            new_name = f"{prefix}.w2_weight_scale"

            narrow_weight = weight[
                ...,
                moe_tp_rank_start // mxfp4_block : moe_tp_rank_end // mxfp4_block,
            ]

            param = params_dict[new_name]
            d0, d1 = narrow_weight.shape
            param.data[local_expert_id, :d0, :d1].copy_(
                narrow_weight.to(param.data.dtype)
            )
            loaded_params.add(new_name)

        elif dispatch_key == "gate_up_proj.bias":
            # Handle MLP gate and up projection biases
            new_name = f"{prefix}.w13_weight_bias"

            narrow_gate = weight[0::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()
            narrow_up = weight[1::2][moe_tp_rank_start:moe_tp_rank_end].contiguous()

            param = params_dict[new_name]
            intermediate_pad = param.data.shape[1] // 2
            param.data[local_expert_id, : narrow_gate.shape[0]].copy_(
                narrow_gate.to(param.data.dtype)
            )
            param.data[
                local_expert_id,
                intermediate_pad : intermediate_pad + narrow_up.shape[0],
            ].copy_(narrow_up.to(param.data.dtype))
            loaded_params.add(new_name)

        elif dispatch_key == "down_proj.bias":
            # Handle MLP down projection bias
            # Only TP rank 0 owns the bias; others zero out so the
            # post-MoE all-reduce sums to the correct value once.
            narrow_weight = weight
            if moe_tp_rank != 0:
                narrow_weight = torch.zeros_like(narrow_weight)

            new_name = f"{prefix}.w2_weight_bias"
            param = params_dict[new_name]
            d0 = narrow_weight.shape[0]
            param.data[local_expert_id, :d0].copy_(narrow_weight.to(param.data.dtype))
            loaded_params.add(new_name)

        elif dispatch_key == "gate_up_proj.input_scale":
            # Handle MLP gate/up FP8 activation scale (per-tensor scalar)
            new_name = f"{prefix}.w13_input_scale"
            if new_name not in params_dict:
                # Scheme didn't allocate the parameter (e.g. W4A16); skip.
                continue

            param = params_dict[new_name]
            param.data[local_expert_id].copy_(weight.to(param.data.dtype).reshape(()))
            loaded_params.add(new_name)

        elif dispatch_key == "down_proj.input_scale":
            # Handle MLP down FP8 activation scale (per-tensor scalar)
            new_name = f"{prefix}.w2_input_scale"
            if new_name not in params_dict:
                # Scheme didn't allocate the parameter (e.g. W4A16); skip.
                continue

            param = params_dict[new_name]
            param.data[local_expert_id].copy_(weight.to(param.data.dtype).reshape(()))
            loaded_params.add(new_name)

    return loaded_params
