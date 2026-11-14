"""Opt-in FP8 input all-gather for GLM's interleaved CP / TP MoE.

Route the original BF16 shard locally and gather its exact router results with
the FP8 values/scales. Expert weights, output reduction and residual ownership
remain with the existing model and layer boundary. Unsupported layers gather
BF16 through the original model path instead. No EP/A2A or MTP support is added.
"""

import logging
import os

import torch

_ENABLED = os.environ.get("SGLANG_CP_MOE_FP8_TRANSPORT", "0") == "1"
_LAYERS = {}
_LOGGED = False
logger = logging.getLogger(__name__)


def cp_moe_fp8_transport_enabled():
    if not _ENABLED:
        return False
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    return (
        p.enable_prefill_cp
        and p.enable_cp_tp_group_sharing
        and p.cp_strategy == "interleave"
        and p.attn_cp_size == p.tp_size
        and p.attn_cp_size > 1
        and p.attn_dp_size == 1
        and p.moe_dp_size == 1
        and p.moe_ep_size == 1
        and p.pp_size == 1
        and get_moe_a2a_backend().value == "none"
        and (
            get_moe_runner_backend().is_triton()
            or get_moe_runner_backend().is_deep_gemm()
        )
    )


def register_cp_moe_fp8_layer(layer):
    from sglang.srt.runtime_context import get_exec

    layer._cp_fp8_transport_eligible = False
    if (
        not cp_moe_fp8_transport_enabled()
        or layer.is_nextn
        or layer.reduce_results
        or layer.moe_ep_size != 1
        or getattr(layer, "is_hash", False)
        or layer.gate.e_score_correction_bias_vl is not None
        or get_exec().moe.enable_eplb
    ):
        return
    # The quantization runner is finalized only after weight loading.
    layer._cp_fp8_transport_eligible = None
    _LAYERS[layer.layer_id] = layer


def cp_moe_fp8_layer_eligible(layer):
    if layer._cp_fp8_transport_eligible is None:
        eligible, reason = layer._compute_moe_quant_once_enabled(
            ignore_env=True, allow_deep_gemm=True, allow_fused_shared=True
        )
        layer._cp_fp8_transport_eligible = eligible
        if not eligible:
            logger.info(
                "CP FP8 MoE layer %s uses BF16 fallback: %s", layer.layer_id, reason
            )
    return layer._cp_fp8_transport_eligible


@torch.library.custom_op(
    "sglang::cp_moe_fp8_transport", mutates_args=(), device_types="cuda"
)
def cp_moe_fp8_transport(
    local_hidden: torch.Tensor, layer_id: int, cp_size: int
) -> torch.Tensor:
    from sglang.kernels.ops.quantization.moe_fp8_packet import (
        pack_moe_fp8,
        unpack_moe_fp8,
    )
    from sglang.kernels.ops.quantization.per_token_group_quant import (
        per_token_group_quant,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        maybe_fuse_routed_scale_and_shared_add,
    )
    from sglang.srt.runtime_context import get_parallel

    layer = _LAYERS[layer_id]
    # Router must not consume a lossy dequantized FP8 approximation.
    logits = layer.gate(local_hidden)
    topk = layer.topk(local_hidden, logits, num_token_non_padded=None)
    assert TopKOutputChecker.format_is_standard(topk)
    if layer.experts.quant_method.runner.runner_backend.is_deep_gemm():
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8_row_padded,
        )

        q, scale = sglang_per_token_group_quant_fp8_row_padded(local_hidden, 128)
    else:
        q, scale = per_token_group_quant(local_hidden, group_size=128)
    payload = pack_moe_fp8(q, scale, topk.topk_weights, topk.topk_ids)
    gathered = payload.new_empty((payload.shape[0] * cp_size, payload.shape[1]))
    get_parallel().attn_cp_group.all_gather_into_tensor(gathered, payload)
    q, scale, weights, ids = unpack_moe_fp8(
        gathered, local_hidden.shape[1], topk.topk_ids.shape[1]
    )
    # All consumers below use pre_quant. This is only output storage / metadata,
    # never an uninitialized BF16 GEMM operand.
    shape_only = local_hidden.new_empty(q.shape)
    shared = layer._forward_shared_experts(shape_only, pre_quant_input=(q, scale))
    routed = layer.experts.forward_impl(
        shape_only,
        StandardTopKOutput(weights, ids, None),
        pre_quant_input=(q, scale),
    )
    output = maybe_fuse_routed_scale_and_shared_add(
        layer.experts,
        routed,
        None if layer._shared_expert_tp1 else shared,
        layer.routed_scaling_factor,
    )
    if (
        shared is not None
        and layer._shared_expert_tp1
        and layer._adds_replicated_output()
    ):
        output += shared
    global _LOGGED
    if not _LOGGED:
        logger.info(
            "CP FP8 MoE transport: local_rows=%s, payload_row_bytes=%s, BF16_row_bytes=%s",
            local_hidden.shape[0],
            payload.shape[1],
            local_hidden.shape[1] * 2,
        )
        _LOGGED = True
    return output


@cp_moe_fp8_transport.register_fake
def _cp_moe_fp8_transport_fake(local_hidden, layer_id, cp_size):
    return local_hidden.new_empty(
        (local_hidden.shape[0] * cp_size, local_hidden.shape[1])
    )
