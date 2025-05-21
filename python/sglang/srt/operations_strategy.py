from dataclasses import dataclass
from typing import List, Optional

import torch
from sglang.srt import operations
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.operations import Operation


@dataclass
class OperationsStrategy:
    operations: List[Operation]
    deep_gemm_num_sms: Optional[int]
    tbo_delta_stages: Optional[int]

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            deep_gemm_num_sms=_assert_all_same([item.deep_gemm_num_sms for item in items]),
            tbo_delta_stages=_assert_all_same([item.tbo_delta_stages for item in items]),
        )


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


def compute_layers_operations(
    layers: torch.nn.ModuleList,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    return OperationsStrategy.concat([_compute_layer_operations(layer, forward_mode) for layer in layers])


def _compute_layer_operations(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    if not layer.is_layer_sparse:
        return [
            layer.op_comm_prepare_attn,
            layer.op_attn,
            layer.op_comm_prepare_mlp,
            layer.op_mlp,
            layer.op_comm_postprocess_layer,
        ]

    if these_layers_of_this_batch_needs_tbo:
        if forward_mode == ForwardMode.EXTEND:
            return [
                layer.op_comm_prepare_attn,
                layer.op_attn,
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
            ]
        elif forward_mode == ForwardMode.DECODE:
            return [
                layer.op_comm_prepare_attn,
                layer.op_decode_attn_0,  # TODO
                operations.YieldOperation(),
                layer.op_decode_attn_1,  # TODO
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
                layer.mlp.op_output,
                layer.op_comm_postprocess_layer,
                operations.YieldOperation(),
            ]
        else:
            raise NotImplementedError(f"Unsupported {forward_mode=}")

    return [
        layer.op_comm_prepare_attn,
        layer.op_attn,
        layer.op_comm_prepare_mlp,
        layer.mlp.op_gate,
        layer.mlp.op_shared_experts,
        layer.mlp.op_select_experts,
        layer.mlp.op_dispatch_a,
        layer.mlp.op_dispatch_b,
        layer.mlp.op_experts,
        layer.mlp.op_combine_a,
        layer.mlp.op_combine_b,
        layer.mlp.op_output,
        layer.op_comm_postprocess_layer,
    ]
