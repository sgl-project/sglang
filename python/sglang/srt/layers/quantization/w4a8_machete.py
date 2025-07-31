# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/moe_wna16.py
from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from sgl_kernel import machete_mm, machete_prepack_B
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.awq import AWQConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import (
    machete_repack,
    parse_machete_config,
    replace_parameter,
)
from sglang.srt.utils import get_device_capability, set_weight_attrs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class W4A8MacheteConfig(QuantizationConfig):
    """Config class for AWQ Machete (W4A8) quantization."""

    def __init__(
        self,
        linear_quant_method: str,
        weight_bits: int,
        group_size: int,
        has_zp: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[List[str]],
        full_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits
        self.pack_factor = 32 // self.weight_bits
        self.lm_head_quantized = lm_head_quantized
        self.full_config = full_config
        self.quant_type = scalar_types.uint4 if weight_bits == 4 else scalar_types.uint8
        if modules_to_not_convert is None:
            self.modules_to_not_convert = []
        else:
            self.modules_to_not_convert = modules_to_not_convert

    @classmethod
    def get_name(cls) -> str:
        return "w4a8_machete"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W4A8MacheteConfig":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        has_zp = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )

        return cls(
            quant_method,
            weight_bits,
            group_size,
            has_zp,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if user_quant == "w4a8_machete" and cls.is_w4a8_machete_compatible(
            hf_quant_cfg
        ):
            return cls.get_name()
        return None

    @classmethod
    def is_w4a8_machete_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        desc_act = quant_config.get("desc_act")

        capability_tuple = get_device_capability()
        device_capability = (
            -1
            if all(capability is None for capability in capability_tuple)
            else capability_tuple[0] * 10 + capability_tuple[1]
        )
        awq_min_capability = AWQConfig.get_min_capability()

        awq_compatible = (
            quant_method == "awq"
            and num_bits == 4
            and device_capability >= awq_min_capability
        )
        return awq_compatible

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        # avoid circular import
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            if "fused_qkv_a_proj_with_mqa" in layer.prefix:
                from sglang.srt.layers.quantization.awq import AWQMarlinLinearMethod

                return AWQMarlinLinearMethod(self)
            return AWQW4A8MacheteLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return MoeW4A8MacheteMethod(self)
        return None


class MoeW4A8MacheteMethod:
    """Linear method for MOE AWQ Machete (W4A8) quantization.

    Args:
        quant_config: The MOE AWQ Machete (W4A8) quantization config.
    """

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config: W4A8MacheteConfig):
        self.quant_config = quant_config
        self.act_type = torch.float8_e4m3fn
        self.quant_type = scalar_types.uint4b8
        self.scale_type = torch.bfloat16

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

        layer.quant_config = self.quant_config
        bit8_pack_factor = self.quant_config.bit8_pack_factor
        group_size = self.quant_config.group_size
        group_size_div_factor = 1
        self.scale_type = params_dtype

        # make intermediate_size and hidden_size diviable by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size_per_partition % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": False})

        assert "weight_loader" in extra_weight_attrs
        weight_loader = extra_weight_attrs["weight_loader"]
        wrapped_weight_loader = MoeW4A8MacheteMethod.get_weight_loader(
            layer, weight_loader
        )
        extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        # shape: [num_experts, out_channels, in_channels]
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        if self.quant_config.has_zp:
            w13_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition // bit8_pack_factor,
                    hidden_size // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)

            w2_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size // bit8_pack_factor,
                    intermediate_size_per_partition // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_qweight(self, qweight):
        G, out_dim, in_dim = qweight.shape
        qweight = qweight.reshape([-1, in_dim]).view(torch.int32).T
        qweight_processed = machete_prepack_B(
            qweight, self.act_type, self.quant_type, self.scale_type
        )
        return qweight_processed.reshape([G, -1, out_dim])

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_qweight = self.process_qweight(layer.w13_qweight)
        w2_qweight = self.process_qweight(layer.w2_qweight)

        w13_scales = (
            layer.w13_scales.reshape([-1, layer.w13_scales.shape[-1]])
            .permute([1, 0])
            .contiguous()
            .to(self.scale_type)
        )
        w2_scales = (
            layer.w2_scales.reshape([-1, layer.w2_scales.shape[-1]])
            .permute([1, 0])
            .contiguous()
            .to(self.scale_type)
        )

        layer.register_parameter(
            "w13_qweight", torch.nn.Parameter(w13_qweight, requires_grad=False)
        )
        layer.register_parameter(
            "w2_qweight", torch.nn.Parameter(w2_qweight, requires_grad=False)
        )
        layer.register_parameter(
            "w13_scales", torch.nn.Parameter(w13_scales, requires_grad=False)
        )
        layer.register_parameter(
            "w2_scales", torch.nn.Parameter(w2_scales, requires_grad=False)
        )

        if self.quant_config.has_zp:
            layer.register_parameter(
                "w13_qzeros",
                torch.nn.Parameter(layer.w13_qzeros.data, requires_grad=False),
            )
            layer.register_parameter(
                "w2_qzeros",
                torch.nn.Parameter(layer.w2_qzeros.data, requires_grad=False),
            )

        torch.cuda.empty_cache()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output: TopKOutput,
        *,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_machete_impl

        topk_weights, topk_ids, _ = topk_output

        return fused_experts_machete_impl(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w1_zp=layer.w13_qzeros if self.quant_config.has_zp else None,
            w2_zp=layer.w2_qzeros if self.quant_config.has_zp else None,
            no_combine=no_combine,
            has_zp=self.quant_config.has_zp,
            routed_scaling_factor=routed_scaling_factor,
        )

    @staticmethod
    def get_weight_loader(layer, weight_loader):

        def convert_awq_tensor(tensor, tensor_type):
            # convert awq qweight/qzeros to a standard format (assume int4)
            # qweight: (k, n // pack_factor_bit32) -> (n, k // pack_factor_bit8)
            # qzeros: (k // group_size, n // pack_factor_bit32) ->
            #         (n // pack_factor_bit8, k // group_size)
            # pack_factor_bit32 = 32 // weight_bits
            # pack_factor_bit8 = 8 // weight_bits

            # 0. suppose origin shape (a, b), dtype int32
            # 1. convert to uint8, shape (a, b) -> (a, 4 * b)
            size0 = tensor.size(0)
            tensor = tensor.view(torch.uint8)

            # 2. unpack to uint4 (only when weight_bits == 4)
            #    shape (a, 4 * b) -> (a, 4 * b, 2)
            shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF

            # 3. change order, see
            # https://github.com/casper-hansen/AutoAWQ/blob/v0.2.8/awq/utils/quant_utils.py
            # shape -> (a, 4 * b * pack_factor_bit8)
            reverse_awq_pack_order = [0, 4, 1, 5, 2, 6, 3, 7]
            tensor = tensor.view(-1, 8)[:, reverse_awq_pack_order]
            tensor = tensor.view(size0, -1)

            # 4. transpose, shape -> (4 * b * pack_factor_bit8, a)
            tensor = tensor.T.contiguous()

            # 5. repack (only when weight_bits == 4)
            # qweight shape -> (4 * b * pack_factor_bit8, a // pack_factor_bit8)
            # qzeros shape -> (4 * b, a)

            if tensor_type == "qweight":
                tensor = tensor[:, 1::2] * 16 + tensor[:, ::2]
            elif tensor_type == "qzeros":
                tensor = tensor[1::2, :] * 16 + tensor[::2, :]
            return tensor

        def w4a8_machete_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            if "g_idx" in weight_name:
                return
            if not layer.quant_config.has_zp and "qzeros" in weight_name:
                return

            device = get_tp_group().device
            tp_rank = get_tensor_model_parallel_rank()
            loaded_weight = loaded_weight.to(device)
            shard_size = layer.intermediate_size_per_partition

            assert layer.quant_config.weight_bits == 4
            if "weight" in weight_name:
                loaded_weight = convert_awq_tensor(loaded_weight, "qweight")
            elif "zeros" in weight_name:
                loaded_weight = convert_awq_tensor(loaded_weight, "qzeros")
            else:
                loaded_weight = loaded_weight.T

            if "scales" in weight_name:
                loaded_weight = loaded_weight.to(param.dtype)

            # repeat the qzeros/scales to fit new group size
            if (
                layer.group_size_div_factor > 1
                and "qzeros" in weight_name
                or "scales" in weight_name
            ):
                loaded_weight = loaded_weight.repeat_interleave(
                    layer.group_size_div_factor, 1
                )

            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(layer.tp_size, -1, loaded_weight.size(1))[
                    tp_rank
                ]
                if shard_id == "w1":
                    param.data[expert_id, : shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2 :] = tensor
            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.tp_size, -1
                )[:, tp_rank]
            else:
                weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)

        return w4a8_machete_weight_loader


class AWQW4A8MacheteLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: W4A8MacheteConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        self.schedules = parse_machete_config(
            input_size_per_partition, output_size_per_partition
        )

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        qweight_processed = machete_repack(layer.qweight.data)
        replace_parameter(layer, "qweight", qweight_processed)

        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        group_size = self.quant_config.group_size

        schedule = None
        if self.schedules is not None:
            bs_thresholds = list(self.schedules.keys())
            idx = min(
                bisect.bisect_left(bs_thresholds, x.shape[0]), len(bs_thresholds) - 1
            )
            schedule = self.schedules[bs_thresholds[idx]]

        out_shape = x.shape[:-1] + (scales.shape[-1],)

        a_fp8 = x.clamp(min=-448, max=448).to(torch.float8_e4m3fn)

        with torch.cuda.device(x.device.index):
            out = machete_mm(
                a_fp8,
                qweight,
                b_type=scalar_types.uint4b8,
                b_group_scales=scales,
                b_group_zeros=None,
                b_group_size=group_size,
                out_type=x.dtype,
                schedule=schedule,
            )

        if bias is not None:
            out.add_(bias)
        out = out.reshape(out_shape)

        return out
