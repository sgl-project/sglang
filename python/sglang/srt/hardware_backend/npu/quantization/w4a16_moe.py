from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

class NPUW4A8Int4DynamicMoEMethod(FusedMoEMethodBase):

    def __init__(self) -> None:
        self.group_size = 256
        self.tp_size = 1

    def pack_to_int32(self, weight: torch.Tensor):
        assert weight.dim() == 3
        if weight.dtype == torch.int32:
            # pack 8 int4 to int32, we use a int32 to represent a int4
            assert (
                weight.shape[-1] % 8 == 0
            ), "the last dim of weight needs to be divided by 8"
            new_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
                weight.flatten(0, 1)
            )
            new_weight = new_weight.view(weight.shape[0], weight.shape[1], -1)
        elif weight.dtype == torch.int8:
            # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
            assert (
                weight.shape[-1] % 4 == 0
            ), "the last dim of weight needs to be divided by 4"
            new_weight = weight.view(torch.int32).contiguous()
        else:
            raise ValueError(f"{weight.dtype=} is not supported !")
        return new_weight

    def unpack_from_int32(
        self,
        value: torch.Tensor,
        num_bits: int,
        shape: torch.Size = None,
        packed_dim=1,
    ) -> torch.Tensor:
        """
        Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
        original bit range.

        Return tensors in int8

        :param value: tensor to unpack
        :param num_bits: number of bits to unpack each data point into
        :param shape: shape to unpack into, used to remove padding
        :returns: unpacked int8 tensor
        """
        if value.dtype is not torch.int32:
            raise ValueError(
                f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
            )

        if num_bits > 8:
            raise ValueError("Unpacking is only supported for less than 8 bits")

        pack_factor = 32 // num_bits

        # unpack
        mask = (1 << num_bits) - 1

        if packed_dim == 1:
            unpacked = torch.zeros(
                (value.shape[0], value.shape[1] * pack_factor),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

            # remove padding
            if shape is not None:
                original_row_size = int(shape[1])
                unpacked = unpacked[:, :original_row_size]
        else:
            unpacked = torch.zeros(
                (value.shape[0] * pack_factor, value.shape[1]),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

            # remove padding
            original_row_size = int(shape[0])
            unpacked = unpacked[:original_row_size, :]

        # bits are packed in unsigned format, reformat to signed
        # update the value range from unsigned to signed
        offset = pow(2, num_bits) // 2
        unpacked = (unpacked - offset).to(torch.int8)

        return unpacked

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_weight_scale = layer.w13_weight_scale.data.transpose(-1, -2).contiguous()
        w2_weight_scale = layer.w2_weight_scale.data.transpose(-1, -2).contiguous()
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)

        layer.w13_weight_offset = torch.nn.Parameter(
            layer.w13_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_offset = torch.nn.Parameter(
            layer.w2_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )

        # w = [n, k // 8]  --> [k, n // 8]
        # w13_weight = layer.w13_weight.data.transpose(1, 2).contiguous()
        # w2_weight = layer.w2_weight.data.transpose(1, 2).contiguous()
        unpacked_w13_weight = (
            self.unpack_from_int32(layer.w13_weight.data.flatten(0, 1), 4)
            .view(layer.w13_weight.data.shape[0], layer.w13_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        unpacked_w2_weight = (
            self.unpack_from_int32(layer.w2_weight.data.flatten(0, 1), 4)
            .view(layer.w2_weight.data.shape[0], layer.w2_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )

        w13_weight = self.pack_to_int32(unpacked_w13_weight)
        w2_weight = self.pack_to_int32(unpacked_w2_weight)

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        # FIXME W4A8 only support with deepep
        raise NotImplementedError(
            f"W4A8 only support with deepep for now, please enable --moe-a2a-backend deepep"
        )

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[self.w13_weight],
            scale=[self.w13_weight_scale],
            bias=[self.w13_scale_bias],
            per_token_scale=[hidden_states_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )[0]

        # act_fn: swiglu
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[self.w2_weight],
            scale=[self.w2_weight_scale],
            bias=[self.w2_scale_bias],
            per_token_scale=[swiglu_out_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )[0]

        return hidden_states
