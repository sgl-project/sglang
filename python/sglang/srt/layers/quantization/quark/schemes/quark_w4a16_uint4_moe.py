# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.layers.utils.common import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


__all__ = ["QuarkW4A16Uint4MoE", "convert_quark_uint4_tensor"]

# Quark's "reorder" packing puts the eight codes of an int32 in AutoAWQ order.
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def convert_quark_uint4_tensor(
    tensor: torch.Tensor, *, tensor_type: str
) -> torch.Tensor:
    """Quark real-quantized uint4 tensor -> the layout of the Triton WNA16 MoE kernel.

    weight:     int32 (K, N // 8)              -> uint8 (N, K // 2)
    zero_point: int32 (K // group_size, N // 8) -> uint8 (N // 2, K // group_size)
    """
    size0 = tensor.size(0)
    tensor = tensor.contiguous().view(torch.uint8)
    shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
    tensor = ((tensor[:, :, None] >> shifter) & 0xF).view(-1, 8)
    tensor = tensor[:, _REVERSE_AWQ_PACK_ORDER]
    tensor = tensor.reshape(size0, -1).T.contiguous()
    if tensor_type == "weight":
        return tensor[:, 1::2] * 16 + tensor[:, ::2]
    if tensor_type == "zero_point":
        return tensor[1::2, :] * 16 + tensor[::2, :]
    raise ValueError(f"Unknown tensor type {tensor_type!r}")


class QuarkW4A16Uint4MoE(QuarkMoEScheme):
    """Quark packed uint4 weight-only (W4A16) MoE on the Triton WNA16 kernel.

    Asymmetric weights with per-group zero points, pack_method "reorder".
    """

    def __init__(self, weight_config: dict[str, Any]):
        self.group_size = int(weight_config["group_size"])
        self.bit8_pack_factor = 2

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if layer.use_presharded_weights:
            raise ValueError(
                "Quark UINT4 MoE weights are sliced per rank at load; "
                "pre-sharded checkpoints are not supported"
            )
        if extra_weight_attrs.get("with_bias", False):
            raise ValueError("Quark UINT4 MoE does not support expert biases")
        assert "weight_loader" in extra_weight_attrs
        if extra_weight_attrs["weight_loader"].__name__ == "weight_loader_fused":
            raise ValueError(
                "Quark UINT4 MoE loads one expert projection at a time; "
                "stacked expert tensors (use_weight_loader_fused) are not supported"
            )
        group_size = self.group_size
        intermediate_size = intermediate_size_per_partition * layer.moe_tp_size
        if hidden_size % group_size or intermediate_size % group_size:
            raise ValueError(
                "Quark UINT4 MoE needs hidden_size and intermediate_size to be "
                f"multiples of the checkpoint group size {group_size}, got "
                f"{hidden_size} and {intermediate_size}"
            )
        group_size_div_factor = 1
        # A TP shard may split a checkpoint group:
        # refine the group and repeat the per-group tensors at load.
        while intermediate_size_per_partition % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            if group_size < 32:
                raise ValueError(
                    "Quark UINT4 MoE needs hidden_size and the per-rank "
                    "intermediate size to be multiples of 32, got "
                    f"{hidden_size} and {intermediate_size_per_partition}"
                )
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor

        extra_weight_attrs.update(
            {
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
                "is_transposed": False,
            }
        )
        extra_weight_attrs["weight_loader"] = self._get_weight_loader(
            layer=layer,
            weight_loader=extra_weight_attrs["weight_loader"],
            hidden_size=hidden_size,
        )

        pack = self.bit8_pack_factor

        gate_up, down = 2 * intermediate_size_per_partition, hidden_size
        shapes = {
            "w13_weight": (gate_up, hidden_size // pack),
            "w2_weight": (down, intermediate_size_per_partition // pack),
            "w13_weight_scale": (gate_up, hidden_size // group_size),
            "w2_weight_scale": (down, intermediate_size_per_partition // group_size),
            "w13_weight_zero_point": (gate_up // pack, hidden_size // group_size),
            "w2_weight_zero_point": (
                down // pack,
                intermediate_size_per_partition // group_size,
            ),
        }
        for name, shape in shapes.items():
            dtype = params_dtype if name.endswith("_scale") else torch.uint8
            param = torch.nn.Parameter(
                torch.zeros((num_experts, *shape), dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def _get_weight_loader(
        self, *, layer: torch.nn.Module, weight_loader, hidden_size: int
    ):
        # Checkpoint tensors are [inputs or input groups, outputs or output words]
        # of the whole expert; a wrong extent would broadcast into the parameter.
        intermediate_size = layer.intermediate_size_per_partition * layer.moe_tp_size
        group_size = self.group_size

        def checkpoint_shape(weight_name: str, shard_id: str) -> tuple:
            inputs, outputs = (
                (intermediate_size, hidden_size)
                if shard_id == "w2"
                else (hidden_size, intermediate_size)
            )
            if weight_name.endswith("_weight_scale"):
                return (inputs // group_size, outputs)
            if weight_name.endswith("_weight_zero_point"):
                return (inputs // group_size, outputs // 8)
            return (inputs, outputs // 8)

        def quark_uint4_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            is_scale = weight_name.endswith("_weight_scale")
            expected = checkpoint_shape(weight_name, shard_id)
            dtype_ok = (
                loaded_weight.is_floating_point()
                if is_scale
                else loaded_weight.dtype == torch.int32
            )
            if tuple(loaded_weight.shape) != expected or not dtype_ok:
                raise ValueError(
                    f"{weight_name} ({shard_id}): expected "
                    f"{'floating-point' if is_scale else 'int32'} {expected}, got "
                    f"{loaded_weight.dtype} {tuple(loaded_weight.shape)}"
                )
            loaded_weight = loaded_weight.to(param.data.device)

            if weight_name.endswith("_weight_zero_point"):
                loaded_weight = convert_quark_uint4_tensor(
                    loaded_weight, tensor_type="zero_point"
                )
            elif weight_name.endswith("_weight_scale"):
                loaded_weight = loaded_weight.T
            else:
                loaded_weight = convert_quark_uint4_tensor(
                    loaded_weight, tensor_type="weight"
                )

            if layer.group_size_div_factor > 1 and (
                "weight_zero_point" in weight_name or "weight_scale" in weight_name
            ):
                loaded_weight = loaded_weight.repeat_interleave(
                    layer.group_size_div_factor, 1
                )

            # Zero points are packed two rows per byte and sharded like the scales.
            weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)

        return quark_uint4_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The loader already wrote the kernel layout.
        return

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        # The loader writes gate and up as two halves; the alpha variant of gated SiLU
        # reads them interleaved by default.
        if (
            moe_runner_config.activation != "silu"
            or not moe_runner_config.is_gated
            or moe_runner_config.gemm1_alpha is not None
        ):
            raise ValueError(
                "Quark UINT4 MoE supports plain gated SiLU experts only, got "
                f"activation {moe_runner_config.activation!r}, "
                f"is_gated {moe_runner_config.is_gated}, "
                f"gemm1_alpha {moe_runner_config.gemm1_alpha}"
            )
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int4_w4a16=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_zp=layer.w13_weight_zero_point,
            w2_zp=layer.w2_weight_zero_point,
            block_shape=[0, layer.group_size],
        )
        return self.runner.run(dispatch_output, quant_info)
