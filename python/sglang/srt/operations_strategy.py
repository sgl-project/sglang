from abc import ABC

import torch
from sglang.srt import operations
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class OperationStrategy(ABC):
    @staticmethod
    def init_new(
        forward_mode: ForwardMode,
        is_layer_sparse: bool,
    ) -> "OperationStrategy":
        if not is_layer_sparse:
            return _MlpNormalOperationStrategy()

        if global_server_args_dict["enable_two_batch_overlap"]:
            if forward_mode == ForwardMode.EXTEND:
                return _TboDeepseekClassicalExtendOperationStrategy()
            elif forward_mode == ForwardMode.DECODE:
                return _TboDeepseekClassicalDecodeOperationStrategy()
            else:
                raise NotImplementedError(f"Unsupported {forward_mode=}")

        return _MoeNormalOperationStrategy()

    def compute_layers_operations(self, layers: torch.nn.ModuleList):
        return [
            op for layer in layers for op in self.compute_layer_operations(layer)
        ]

    def compute_layer_operations(self, layer: torch.nn.Module):
        raise NotImplementedError


class _MlpNormalOperationStrategy(OperationStrategy):
    def compute_layer_operations(self, layer: torch.nn.Module):
        return [
            layer.op_comm_prepare_attn,
            layer.op_attn,
            layer.op_comm_prepare_mlp,
            layer.op_mlp,
            layer.op_comm_postprocess_layer,
        ]


class _MoeNormalOperationStrategy(OperationStrategy):
    def compute_layer_operations(self, layer: torch.nn.Module):
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


class _TboDeepseekClassicalExtendOperationStrategy(OperationStrategy):
    def compute_layer_operations(self, layer: torch.nn.Module):
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


class _TboDeepseekClassicalDecodeOperationStrategy(OperationStrategy):
    def compute_layer_operations(self, layer: torch.nn.Module):
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
