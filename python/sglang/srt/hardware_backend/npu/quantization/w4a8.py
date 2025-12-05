from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

class NPUW4A8Int4DynamicMoEMethod(FusedMoEMethodBase):

    def __init__(self) -> None:
        self.group_size = 256
        self.tp_size = 1

    def process_scale(self, weight: torch.Tensor, scale, per_group_scale):
        scale = scale.transpose(1, 2).contiguous()
        per_group_scale = per_group_scale.transpose(1, 2).contiguous()
        group_num, k, n = weight.shape
        # the weight of the new version is reduced by half by pack n, so it needs to be restored
        n = n * 2
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape
        bias = None

        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2), dtype=np.uint32)

        sscale_uint64[..., ::2] = scale_fp32_np

        sscale_uint64_buffer = np.frombuffer(
            sscale_uint64.tobytes(), dtype=np.int64
        ).copy()
        sscale_uint64_tensor = torch.from_numpy(sscale_uint64_buffer).reshape(
            group_num, quantgroup_num, n
        )
        sscale_uint64_tensor = sscale_uint64_tensor.npu()
        return sscale_uint64_tensor, bias

    def update_bias(self, layer, w13_bias, w2_bias):
        layer.w13_scale_bias.data = (
            layer.w13_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        )
        layer.w2_scale_bias.data = (
            layer.w2_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        )

    def pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        assert (
            weight.shape[-1] % 4 == 0
        ), "the last dim of weight needs to be divided by 4"
        return weight.view(torch.int32).contiguous()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )

        w13_weight_scale_second = (
            layer.w13_weight_scale_second.data
            if hasattr(layer, "w13_weight_scale_second")
            else None
        )
        w2_weight_scale_second = (
            layer.w2_weight_scale_second.data
            if hasattr(layer, "w2_weight_scale_second")
            else None
        )
        layer.w13_weight_scale.data, w13_bias = self.process_scale(
            layer.w13_weight, layer.w13_weight_scale.data, w13_weight_scale_second
        )
        layer.w2_weight_scale.data, w2_bias = self.process_scale(
            layer.w2_weight, layer.w2_weight_scale.data, w2_weight_scale_second
        )
        if hasattr(layer, "w13_weight_scale_second"):
            # scale_second is no longer used, release this part of the memory
            del layer.w13_weight_scale_second
            del layer.w2_weight_scale_second
            del layer.w13_weight_offset_second
            del layer.w2_weight_offset_second

        self.update_bias(layer, w13_bias, w2_bias)

        layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)
        layer.w13_weight.data = self.pack_to_int32(layer.w13_weight.data)
        layer.w2_weight.data = self.pack_to_int32(layer.w2_weight.data)

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
