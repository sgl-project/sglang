import torch

from sglang.srt import two_batch_overlap, operations
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def compute_layer_operations(
    layer: torch.nn.Module,
):
    if not layer.is_layer_sparse:
        return [
            layer.op_comm_prepare_attn,
            layer.op_attn,
            layer.op_comm_prepare_mlp,
            layer.op_mlp,
            layer.op_comm_postprocess_layer,
        ]

    if enable_two_batch_overlap:
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
                layer.op_decode_attn_0, # TODO
                operations.YieldOperation(),
                layer.op_decode_attn_1, # TODO
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

