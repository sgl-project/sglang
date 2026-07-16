from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.batch_overlap import operations
from sglang.srt.batch_overlap.operations import Operation
from sglang.srt.layers.moe.token_dispatcher import DeepEPConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@dataclass
class OperationsStrategy:
    operations: List[Operation]
    deep_gemm_num_sms: Optional[int] = None
    tbo_delta_stages: Optional[int] = None

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            deep_gemm_num_sms=_assert_all_same(
                [item.deep_gemm_num_sms for item in items]
            ),
            tbo_delta_stages=_assert_all_same(
                [item.tbo_delta_stages for item in items]
            ),
        )

    @staticmethod
    def init_new_tbo(
        layers: torch.nn.ModuleList,
        forward_mode: ForwardMode,
    ) -> "OperationsStrategy":
        layer_name = layers[0].__class__.__name__
        if layer_name == "DeepseekV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "Qwen3MoeDecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_qwen3_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "MiMoV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_mimov2_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "DeepseekV4DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_v4_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


# -------------------------------- Strategy for DeepSeek ---------------------------------------


# TODO can refactor to make it more fancy if we have more complex strategies
def _compute_moe_deepseek_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "dense layer TBO not yet implemented"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_deepseek_blog_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_deepseek_blog_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_deepseek_blog_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = None
    if not _is_hip:
        deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_deepseek_blog_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            layer.mlp.op_shared_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            operations.YieldOperation(),
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


# -------------------------------- Strategy for DeepSeek V4 ---------------------------------------


# DSV4 prefill TBO (EP / mori path). Cross-layer mHC fusion is disabled under
# TBO, so each layer is self-contained: attn-side mHC pre+norm -> attn ->
# ffn-side mHC pre+norm -> MoE (a2a dispatch/combine overlapped) -> mHC post.
# The MoE ops are reused from self.mlp (DeepseekV2MoE) and decompose
# forward_deepep; the layer-level op_mhc_* wrap DSV4's hc_pre / hc_post.
def _compute_moe_deepseek_v4_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_deepseek_v4_prefill(layer)
    elif forward_mode == ForwardMode.DECODE:
        return _compute_moe_deepseek_v4_decode(layer)
    else:
        raise NotImplementedError(
            f"DeepseekV4 TBO only supports EXTEND/DECODE, got {forward_mode=}"
        )


def _compute_moe_deepseek_v4_decode(layer):
    from sglang.srt.layers.moe import get_moe_a2a_backend

    # The first stage is intentionally only the attention-side prepare.  Both
    # children therefore launch HiSparse swap-in before either child enters
    # the long attention+MegaMoE stage.  Do not insert a YieldOperation after
    # op_attn_compute: that would switch to child B before child A's MoE has
    # had a chance to hide child B's swap-in.
    if get_moe_a2a_backend().is_megamoe():
        # MegaMoE is a fused routed-MoE call, not the DeepEP dispatcher
        # decomposition below. Keep it in the same stage as attention so the
        # other child's swap-in can overlap the whole MegaMoE invocation.
        ops = [
            layer.op_mhc_prepare_attn,
            layer.self_attn.op_attn_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_attn_compute,
            layer.op_mhc_post_attn_pre_mlp,
            layer.op_megamoe,
            operations.YieldOperation(),
            layer.op_mhc_postprocess,
        ]
    elif get_moe_a2a_backend().is_none():
        ops = [
            layer.op_mhc_prepare_attn,
            layer.self_attn.op_attn_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_attn_compute,
            layer.op_mhc_post_attn_pre_mlp,
            layer.op_gather_a,
            layer.op_gather_b,
            layer.op_moe,
            layer.op_combine_a,
            operations.YieldOperation(),
            layer.op_combine_b,
            layer.op_mhc_postprocess,
        ]
    else:
        ops = [
            layer.op_mhc_prepare_attn,
            layer.self_attn.op_attn_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_attn_compute,
            layer.op_mhc_post_attn_pre_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_mhc_postprocess,
        ]
    return OperationsStrategy(
        operations=ops,
        deep_gemm_num_sms=None,
        # A0 prepare/swap-in, B0 prepare/swap-in, A1 attention+MegaMoE,
        # B1 attention+MegaMoE.
        tbo_delta_stages=0,
    )


def _compute_moe_deepseek_v4_prefill(layer):
    from sglang.srt.layers.moe import get_moe_a2a_backend

    if get_moe_a2a_backend().is_none():
        # Non-EP DP TP-MoE: overlap the DP all_gatherv (gather) + reduce_scatterv
        # (combine) with the other ubatch's attn+MoE compute (ATOM's DSV4 path).
        ops = [
            layer.op_mhc_prepare_attn,
            layer.self_attn.op_attn,
            layer.op_mhc_post_attn_pre_mlp,
            layer.op_gather_a,
            operations.YieldOperation(),
            layer.op_gather_b,
            layer.op_moe,
            layer.op_combine_a,
            operations.YieldOperation(),
            layer.op_combine_b,
            layer.op_mhc_postprocess,
        ]
    else:
        # EP / mori a2a: reuse DeepseekV2MoE's deepep dispatch/combine ops.
        ops = [
            layer.op_mhc_prepare_attn,
            layer.self_attn.op_attn,
            layer.op_mhc_post_attn_pre_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_mhc_postprocess,
        ]
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=0,
        operations=ops,
    )


# -------------------------------- Strategy for Qwen3 ---------------------------------------


# TODO: unstable, current strategy is almost the same as DeepSeek, keep redundant code here for
# convenience to adjust strategy
def _compute_moe_qwen3_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "qwen3 moe only support sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_qwen3_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_qwen3_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_qwen3_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = None
    if not _is_hip:
        deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_qwen3_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )


# -------------------------------- Strategy for MiMoV2DecoderLayer ---------------------------------------


# TODO: unstable; current strategy matches DeepSeek for the common operations (MiMoV2 has no op_shared_experts),
# so we keep this redundant code here for convenience when adjusting the strategy
def _compute_moe_mimov2_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "MiMoV2DecoderLayer moe only support sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_mimov2_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_mimov2_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_mimov2_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_mimov2_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )
