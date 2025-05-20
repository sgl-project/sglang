import torch


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

    # Will add TBO operation orders here
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
