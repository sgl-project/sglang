from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.hardware_backend.npu.quantization.torch_npu_kernels import TorchNpuKernelsQuantInfo


def npu_fused_moe_without_routing_weights_bf16(
    layer, hidden_states, group_list_type, group_list, output_dtype
):
    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[layer.w13_weight.permute(0, 2, 1)],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[layer.w2_weight.permute(0, 2, 1)],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    return hidden_states


class _NPUFusedMoEMethodBase(FusedMoEMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW8A8Int8DynamicMoEMethod(_NPUFusedMoEMethodBase):

    def _release_weight_cache(self, weight: torch.Tensor):
        # .contiguous() introduces additional memory overhead and needs to be released using resize_(0)
        origin_weight = weight.data.transpose(1, 2)
        new_weight = origin_weight.contiguous()
        origin_weight.untyped_storage().resize_(0)
        return new_weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_data = self._release_weight_cache(layer.w13_weight.data)
        layer.w13_weight = torch.nn.Parameter(weight_data, requires_grad=False)

        weight_data = self._release_weight_cache(layer.w2_weight.data)
        layer.w2_weight = torch.nn.Parameter(weight_data, requires_grad=False)

        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.squeeze(-1).contiguous().to(torch.float32),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.squeeze(-1).contiguous(), requires_grad=False
        )
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "w13_weight_offset"):
            layer.w13_weight_offset = torch.nn.Parameter(
                layer.w13_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )
        else:
            layer.w13_weight_offset = None
        if hasattr(layer, "w2_weight_offset"):
            layer.w2_weight_offset = torch.nn.Parameter(
                layer.w2_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )
        else:
            layer.w2_weight_offset = None

        layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = (
                MoeRunnerBackend.TORCH_NPU_KERNELS
            )
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        backend = self.runner.runner_backend
        quant_info = TorchNpuKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w13_offset=layer.w13_weight_offset,
                w2_offset=layer.w2_weight_offset,
            )
        return self.runner.run(dispatch_output, quant_info)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.int32,
        )[0]

        # act_fn: swiglu
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=layer.w13_weight_scale,
            activation_scale=hidden_states_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
        )

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale.to(output_dtype)],
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
        )[0]
        return hidden_states


class NPUW4A8Int8DynamicMoEMethod(_NPUFusedMoEMethodBase):

    def _process_scale(
        self, weight: torch.Tensor, scale, per_group_scale, is_per_channel_weight
    ):
        scale = scale.transpose(1, 2).contiguous()

        if is_per_channel_weight:
            scale_np = scale.cpu().numpy()
            scale_np.dtype = np.uint32
            scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
            return scale_uint64_tensor, None

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

    def _update_bias(self, layer):
        layer.w13_scale_bias.data = (
            layer.w13_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        )
        layer.w2_scale_bias.data = (
            layer.w2_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        )

    def _pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        assert (
            weight.shape[-1] % 4 == 0
        ), "the last dim of weight needs to be divided by 4"
        return weight.view(torch.int32).contiguous()

    def process_weights_after_loading(
        self, layer: torch.nn.Module, is_per_channel_weight, activation_use_clip
    ) -> None:
        if not activation_use_clip:
            self._process_weights_without_clip(layer, is_per_channel_weight)
        else:
            self._process_weights_with_clip(layer)

        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.transpose(1, 2).contiguous(), requires_grad=False
        )

        layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)

        layer.w13_weight.data = self._pack_to_int32(layer.w13_weight.data)
        layer.w2_weight.data = self._pack_to_int32(layer.w2_weight.data)

    def _process_weights_without_clip(
        self, layer: torch.nn.Module, is_per_channel_weight
    ) -> None:
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
        layer.w13_weight_scale.data, layer.w13_bias = self._process_scale(
            layer.w13_weight,
            layer.w13_weight_scale.data,
            w13_weight_scale_second,
            is_per_channel_weight,
        )
        layer.w2_weight_scale.data, layer.w2_bias = self._process_scale(
            layer.w2_weight,
            layer.w2_weight_scale.data,
            w2_weight_scale_second,
            is_per_channel_weight,
        )
        if hasattr(layer, "w13_weight_scale_second"):
            # scale_second is no longer used, release this part of the memory
            del layer.w13_weight_scale_second
            del layer.w2_weight_scale_second
            del layer.w13_weight_offset_second
            del layer.w2_weight_offset_second

        self._update_bias(layer)

    def _process_weights_with_clip(self, layer: torch.nn.Module) -> None:
        w13_weight_scale = (
            layer.w13_weight_scale.data.squeeze(-1).contiguous().unsqueeze(1)
        )
        w2_weight_scale = (
            layer.w2_weight_scale.data.squeeze(-1).contiguous().unsqueeze(1)
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
        layer.w13_scale_bias = layer.w13_bias
        layer.w2_scale_bias = layer.w2_bias

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = (
                MoeRunnerBackend.TORCH_NPU_KERNELS
            )
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        backend = self.runner.runner_backend
        quant_info = TorchNpuKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_scale=layer.w2_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w13_scale_bias=layer.w13_scale_bias,
                w2_scale_bias=layer.w2_scale_bias,
            )
        return self.runner.run(dispatch_output, quant_info)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale],
            bias=[layer.w13_scale_bias],
            per_token_scale=[hidden_states_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )[0]

        hidden_states, swiglu_out_scale = swiglu_quant(
            hidden_states, group_list, group_list_type
        )

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale],
            bias=[layer.w2_scale_bias],
            per_token_scale=[swiglu_out_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )[0]

        return hidden_states


class NPUW4A16Int4DynamicMoEMethod(_NPUFusedMoEMethodBase):

    def _pack_to_int32(self, weight: torch.Tensor):
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

    def _unpack_from_int32(
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
            self._unpack_from_int32(layer.w13_weight.data.flatten(0, 1), 4)
            .view(layer.w13_weight.data.shape[0], layer.w13_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )
        unpacked_w2_weight = (
            self._unpack_from_int32(layer.w2_weight.data.flatten(0, 1), 4)
            .view(layer.w2_weight.data.shape[0], layer.w2_weight.data.shape[1], -1)
            .transpose(1, 2)
            .contiguous()
            .int()
        )

        w13_weight = self._pack_to_int32(unpacked_w13_weight)
        w2_weight = self._pack_to_int32(unpacked_w2_weight)

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

    def apply(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        backend = self.runner.runner_backend
        quant_info = TorchNpuKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w13_bias=layer.w13_offset,
                w2_offset=layer.w2_offset,
            )
        return self.runner.run(dispatch_output, quant_info)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        if hidden_states_scale is None:
            # gmm1: gate_up_proj
            hidden_states = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w13_weight],
                antiquant_scale=[layer.w13_weight_scale],
                antiquant_offset=[layer.w13_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]

            # act_fn: swiglu
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

            # gmm2: down_proj
            out_hidden = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w2_weight],
                antiquant_scale=[layer.w2_weight_scale],
                antiquant_offset=[layer.w2_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]
        else:
            raise ValueError(
                "when weight is int4, hidden_states only supports non-quant dtype!"
            )

        return out_hidden
