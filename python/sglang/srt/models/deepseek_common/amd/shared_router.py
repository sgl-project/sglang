# SPDX-License-Identifier: Apache-2.0
"""Fail-closed dispatch for the isolated DSV4.1 shared/router optimization."""

import logging

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def try_shared_router(moe, x, forward_batch, skip_shared_experts):
    """Return (shared, standard top-k), or None without changing native execution.

    Only target verification is supported. Hash/draft routing, EP, expert location
    remapping, shared TP1 and other quantization formats keep their native paths.
    No persistent tensor handoff or process-global workspace is used.
    """
    if not envs.SGLANG_DSV41_SHARED_ROUTER_FUSION.get():
        return None
    from sglang.srt.runtime_context import get_exec
    from sglang.srt.utils import is_gfx95_supported

    if (
        not is_gfx95_supported()
        or forward_batch is None
        or not forward_batch.forward_mode.is_target_verify()
        or skip_shared_experts
        or moe.is_hash
        or moe.is_nextn
        or moe.tp_size != 4
        or moe.moe_ep_size != 1
        or moe._shared_expert_tp1
        or moe.num_fused_shared_experts != 0
        or moe._enable_a2a_moe
        or moe._fuse_shared_experts_inside_sbo
        or get_exec().moe.enable_eplb
        or envs.SGLANG_OPT_MOE_QUANT_ONCE.get()
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
        or getattr(moe.config, "model_type", None) != "deepseek_v41"
        or getattr(moe.config, "n_routed_experts", None) != 384
        or getattr(moe.config, "num_experts_per_tok", None) != 6
        or getattr(moe.config, "scoring_func", None) != "sqrtsoftplus"
        or not getattr(moe.config, "norm_topk_prob", False)
        or moe.routed_scaling_factor != 1.5
        or x.ndim != 2
        or x.shape[0] not in (6, 12)
        or x.shape[1] != 5120
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        return None
    mlp = getattr(moe, "shared_experts", None)
    if mlp is None or mlp.swiglu_limit != 10.0:
        return None
    gate, down = mlp.gate_up_proj, mlp.down_proj
    bias = moe.gate.e_score_correction_bias
    if (
        not getattr(gate, "mxfp8_native_ready", False)
        or not getattr(down, "mxfp8_native_ready", False)
        or moe.gate.e_score_correction_bias_vl is not None
    ):
        return None
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        _m_bucket,
        _select_config,
    )

    cfg = _select_config(_m_bucket(x.shape[0]), 1152, 5120)
    if (cfg.waves, cfg.steps, cfg.rows, cfg.tokens) != (
        8,
        1 if x.shape[0] == 6 else 2,
        16,
        16,
    ):
        return None
    operands = (
        (gate.weight, (72, 40, 2048), torch.float8_e4m3fn),
        (gate.weight_scale_mx_e8m0, (36, 160), torch.uint8),
        (moe.gate.weight, (384, 5120), torch.bfloat16),
        (down.weight, (320, 5, 2048), torch.float8_e4m3fn),
        (down.weight_scale_mx_e8m0, (160, 20), torch.uint8),
        (bias, (384,), torch.bfloat16),
    )
    if any(
        t is None
        or t.shape != shape
        or t.dtype != dtype
        or t.device != x.device
        or not t.is_contiguous()
        for t, shape, dtype in operands
    ):
        return None
    from sglang.kernels.ops.moe.shared_router_gfx950 import (
        shared_router,
        unpack_shared_down,
    )
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput, _post_process_topk_ids

    weight_key = (
        down.weight.data_ptr(),
        down.weight._version,
        down.weight_scale_mx_e8m0.data_ptr(),
        down.weight_scale_mx_e8m0._version,
    )
    if getattr(moe, "_shared_router_weight_key", None) != weight_key:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm shared/router fusion before graph capture")
        moe._shared_router_down_bf16 = unpack_shared_down(
            down.weight, down.weight_scale_mx_e8m0
        )
        moe._shared_router_weight_key = weight_key
    shared, weights, ids, logits = shared_router(
        x,
        gate.weight.view(torch.uint8),
        gate.weight_scale_mx_e8m0,
        moe.gate.weight,
        moe._shared_router_down_bf16,
        bias,
    )
    # Preserve upstream's padded-row masking, routed-expert capture and
    # distribution recording. The fusion must not bypass these side effects.
    ids, weights, recorder_ids = _post_process_topk_ids(
        ids,
        weights,
        moe.topk.topk_config,
        logits,
        moe.layer_id,
        num_token_non_padded=forward_batch.moe_num_token_non_padded(),
    )
    if recorder_ids is not None:
        get_global_expert_distribution_recorder().on_select_experts(
            topk_ids=recorder_ids
        )
    if not getattr(moe, "_shared_router_logged", False):
        logger.info(
            "DSV41_SHARED_ROUTER_SELECTED layer=%s M=%s", moe.layer_id, x.shape[0]
        )
        moe._shared_router_logged = True
    return shared, StandardTopKOutput(weights, ids, logits)
