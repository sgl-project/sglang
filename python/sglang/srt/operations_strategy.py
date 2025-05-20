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
            self._forward_tbo_op_input_layernorm,
            self._forward_tbo_op_prefill_attn,
            self._forward_tbo_op_post_attn_layernorm,
            self.mlp._forward_tbo_op_gate,
            self.mlp._forward_tbo_op_dispatch_a_part_one,
            self.mlp._forward_tbo_op_dispatch_a_part_two,
            two_batch_overlap.YieldOperation(),
            partial(
                self.mlp._forward_tbo_op_dispatch_b, tbo_child_index=tbo_child_index
            ),
            self.mlp._forward_tbo_op_mlp,
            self.mlp._forward_tbo_op_combine_a,
            two_batch_overlap.YieldOperation(),
            self.mlp._forward_tbo_op_shared,
            self.mlp._forward_tbo_op_combine_b,
            self._forward_tbo_op_compute_layer_output,
        ]
    elif forward_mode == ForwardMode.DECODE:
        return [
            self._forward_tbo_op_input_layernorm,
            self._forward_tbo_op_decode_attn_0,
            two_batch_overlap.YieldOperation(),
            self._forward_tbo_op_decode_attn_1,
            self._forward_tbo_op_post_attn_layernorm,
            self.mlp._forward_tbo_op_gate,
            self.mlp._forward_tbo_op_dispatch_a_part_one,
            two_batch_overlap.YieldOperation(),
            self.mlp._forward_tbo_op_dispatch_a_part_two,
            self.mlp._forward_tbo_op_shared,
            two_batch_overlap.YieldOperation(),
            partial(
                self.mlp._forward_tbo_op_dispatch_b, tbo_child_index=tbo_child_index
            ),
            self.mlp._forward_tbo_op_mlp,
            self.mlp._forward_tbo_op_combine_a,
            two_batch_overlap.YieldOperation(),
            self.mlp._forward_tbo_op_combine_b,
            self._forward_tbo_op_compute_layer_output,
            two_batch_overlap.YieldOperation(),
        ]
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")
