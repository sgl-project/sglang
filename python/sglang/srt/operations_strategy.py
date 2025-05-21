import torch

from sglang.srt import operations
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def compute_layers_operations(
    layers: torch.nn.ModuleList,
    forward_mode: ForwardMode,
):
    return [
        op for layer in layers for op in compute_layer_operations(layer, forward_mode)
    ]


# TODO refactor this if there are more overlapping strategies
def compute_layer_operations(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
):
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
