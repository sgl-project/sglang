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

    if forward_mode == ForwardMode.EXTEND:
        return [
            self.op_comm_prepare_attn,
            self.op_attn,
            self.op_comm_prepare_mlp,
            self.mlp.op_gate,
            self.mlp.op_select_experts,
            self.mlp.op_dispatch_a,
            two_batch_overlap.YieldOperation(),
            self.mlp.op_dispatch_b,
            self.mlp.op_experts,
            self.mlp.op_combine_a,
            two_batch_overlap.YieldOperation(),
            self.mlp.op_shared_experts,
            self.mlp.op_combine_b,
            self.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ]
    elif forward_mode == ForwardMode.DECODE:
        return [
            self.op_comm_prepare_attn,
            self.op_decode_attn_0, # TODO
            two_batch_overlap.YieldOperation(),
            self.op_decode_attn_1, # TODO
            self.op_comm_prepare_mlp,
            self.mlp.op_gate,
            self.mlp.op_select_experts,
            two_batch_overlap.YieldOperation(),
            self.mlp.op_dispatch_a,
            self.mlp.op_shared_experts,
            two_batch_overlap.YieldOperation(),
            self.mlp.op_dispatch_b,
            self.mlp.op_experts,
            self.mlp.op_combine_a,
            two_batch_overlap.YieldOperation(),
            self.mlp.op_combine_b,
            self.mlp.op_output,
            layer.op_comm_postprocess_layer,
            two_batch_overlap.YieldOperation(),
        ]
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")
