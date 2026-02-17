# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.online_quantization import CopyNumelCounter
from sglang.srt.layers.quantization.utils import all_close_1d, per_tensor_dequantize
from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_hip,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig

logger = logging.getLogger(__name__)

_is_shuffle_moe_mxfp4 = is_gfx95_supported()

__all__ = ["QuarkMoEMethod", "QuarkW4A4MXFp4MoEMethod"]

_is_fp8_fnuz = is_fp8_fnuz()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    from sglang.srt.layers.moe.rocm_moe_utils import rocm_fused_experts_tkw1


if _is_hip:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
else:
    dynamic_mxfp4_quant = None

OCP_MX_BLOCK_SIZE = 32

if TYPE_CHECKING:
    from sglang.srt.layers.quantization import QuarkConfig


class QuarkMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: QuarkConfig):
        self.quant_config = quant_config

    @staticmethod
    def get_moe_method(
        quant_config: QuarkConfig,  # type: ignore # noqa E501 # noqa F821
        module: torch.nn.Module,
        layer_name: str,
    ) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(layer_name, module)

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if quant_config._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFp4MoEMethod(
                weight_config,
                input_config,
                is_checkpoint_mxfp4_serialized=quant_config.is_prequantized,
            )
        elif quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8FP8MoEMethod(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")


class QuarkW4A4MXFp4MoEMethod(QuarkMoEMethod):

    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        is_checkpoint_mxfp4_serialized: bool = True,
    ):
        self.weight_quant = weight_config
        self.input_quant = input_config
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group" and input_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}"
            )  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        self.with_bias = False

        if not self.is_checkpoint_mxfp4_serialized:
            logger.info_once(
                "Using online MXFP4 quantization for MoE layers from a higher precision checkpoint. "
                "Beware that this optimization may degrade prediction quality - please validate your model accuracy. "
                "More details at https://docs.sglang.io/advanced_features/quantization.html#online-quantization."
            )

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

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        original_weight_loader = extra_weight_attrs.get("weight_loader")

        if self.is_checkpoint_mxfp4_serialized:
            weight_loader = original_weight_loader
            weight_device = torch.get_default_device()
            weight_dtype = torch.uint8
        else:
            # Online quantization: use original dtype and meta device
            weight_loader = self.get_online_weight_loader(layer, original_weight_loader)
            weight_device = torch.device("meta")
            weight_dtype = params_dtype

        params_dtype = torch.uint8

        layer._load_device = torch.get_default_device()
        layer._w13_loaded_numel = 0
        layer._w2_loaded_numel = 0

        extra_weight_attrs["weight_loader"] = weight_loader

        # WEIGHTS
        w13_shape = (
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2 if self.is_checkpoint_mxfp4_serialized else hidden_size,
        )
        w13_weight = torch.nn.Parameter(
            torch.empty(
                w13_shape,
                dtype=weight_dtype,
                device=weight_device,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_shape = (
            num_experts,
            hidden_size,
            (
                intermediate_size_per_partition // 2
                if self.is_checkpoint_mxfp4_serialized
                else intermediate_size_per_partition
            ),
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                w2_shape,
                dtype=weight_dtype,
                device=weight_device,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

    def get_online_weight_loader(self, layer, original_weight_loader):
        """
        Wrap the original weight loader to perform online MXFP4 quantization for MoE layers.
        """

        def online_mxfp4_moe_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            if dynamic_mxfp4_quant is None:
                raise NotImplementedError(
                    "Online MXFP4 quantization for MoE is only supported on AMD GPUs."
                )

            # Determine which weight parameter we're loading (w13 or w2)
            is_w13 = "w13" in weight_name
            is_w2 = "w2" in weight_name

            # Initialize weight on device if first load
            if is_w13 and layer._w13_loaded_numel == 0:
                layer.w13_weight = torch.nn.Parameter(
                    torch.empty_like(param.data, device=layer._load_device),
                    requires_grad=False,
                )
                param = layer.w13_weight
            elif is_w2 and layer._w2_loaded_numel == 0:
                layer.w2_weight = torch.nn.Parameter(
                    torch.empty_like(param.data, device=layer._load_device),
                    requires_grad=False,
                )
                param = layer.w2_weight

            # Move to device for faster quantization
            loaded_weight = loaded_weight.to(layer._load_device)

            if is_w13:
                param = layer.w13_weight
            elif is_w2:
                param = layer.w2_weight

            # In case TP>1, the weight loader logic uses narrow so we cannot directly rely on `param.shape` or `loaded_weight.shape`.
            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                original_weight_loader(
                    param, loaded_weight, weight_name, shard_id, expert_id
                )

            if is_w13:
                layer._w13_loaded_numel += copy_numel_counter.copied_numel
                target_loaded_numel = layer.w13_weight.numel()
                current_loaded = layer._w13_loaded_numel
            elif is_w2:
                layer._w2_loaded_numel += copy_numel_counter.copied_numel
                target_loaded_numel = layer.w2_weight.numel()
                current_loaded = layer._w2_loaded_numel
            else:
                raise ValueError("Expected w13 or w2.")

            assert (
                current_loaded <= target_loaded_numel
            ), f"target_loaded_numel={target_loaded_numel}, current_loaded={current_loaded}"

            # Delay online quantization until all tensor shards (e.g. w1 and w3) are loaded, to avoid having to re-quantize later on.
            if is_w13 and layer._w13_loaded_numel == target_loaded_numel:
                self._quantize_w13_online(layer, dynamic_mxfp4_quant)
            elif is_w2 and layer._w2_loaded_numel == target_loaded_numel:
                self._quantize_w2_online(layer, dynamic_mxfp4_quant)

        return online_mxfp4_moe_weight_loader

    def _quantize_w13_online(self, layer, dynamic_mxfp4_quant):
        qw13_weight = torch.empty(
            layer.w13_weight.shape[0],
            layer.w13_weight.shape[1],
            layer.w13_weight.shape[2] // 2,
            dtype=torch.uint8,
            device=layer._load_device,
        )

        for expert in range(layer.w13_weight.shape[0]):
            qweight, weight_scale = dynamic_mxfp4_quant(layer.w13_weight.data[expert])
            assert qw13_weight[expert].shape == qweight.shape
            assert qw13_weight[expert].dtype == qweight.dtype
            qw13_weight[expert] = qweight

            assert layer.w13_weight_scale[expert].shape == weight_scale.shape
            assert layer.w13_weight_scale[expert].dtype == weight_scale.dtype
            layer.w13_weight_scale[expert] = weight_scale

        layer.w13_weight = torch.nn.Parameter(qw13_weight, requires_grad=False)

    def _quantize_w2_online(self, layer, dynamic_mxfp4_quant):
        qw2_weight = torch.empty(
            layer.w2_weight.shape[0],
            layer.w2_weight.shape[1],
            layer.w2_weight.shape[2] // 2,
            dtype=torch.uint8,
            device=layer._load_device,
        )

        for expert in range(layer.w2_weight.shape[0]):
            qweight, weight_scale = dynamic_mxfp4_quant(layer.w2_weight.data[expert])
            qw2_weight[expert] = qweight
            layer.w2_weight_scale[expert] = weight_scale

        layer.w2_weight = torch.nn.Parameter(qw2_weight, requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self.is_checkpoint_mxfp4_serialized:
            # Quantization already happened during weight loading via get_online_weight_loader.
            assert layer.w13_weight.dtype == torch.uint8
            assert layer.w2_weight.dtype == torch.uint8
            assert layer.w13_weight_scale.dtype == torch.uint8
            assert layer.w2_weight_scale.dtype == torch.uint8

        # Pre-shuffle weight scales
        s0, s1, _ = layer.w13_weight_scale.shape
        w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)
        w13_weight_scale = e8m0_shuffle(w13_weight_scale)
        # layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale, requires_grad=False)
        layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)

        s0, s1, _ = layer.w2_weight_scale.shape
        w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)
        w2_weight_scale = e8m0_shuffle(w2_weight_scale)
        # layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
        layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)

        # Pre-shuffle weight
        if _is_shuffle_moe_mxfp4:
            layer.w13_weight.data = shuffle_weight(
                layer.w13_weight.contiguous(), (16, 16)
            )
            layer.w2_weight.data = shuffle_weight(
                layer.w2_weight.contiguous(), (16, 16)
            )
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config
        topk_weights, topk_ids, _ = topk_output
        if _is_hip:
            topk_weights = topk_weights.to(
                torch.float32
            )  # aiter's moe_sorting requires topk_weights to be FP32

        if hasattr(torch, "float4_e2m1fn_x2"):
            w13_weight = layer.w13_weight.view(torch.float4_e2m1fn_x2)
            w2_weight = layer.w2_weight.view(torch.float4_e2m1fn_x2)
        else:
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

        if hasattr(layer.w13_weight, "is_shuffled"):
            w13_weight.is_shuffled = True
            w2_weight.is_shuffled = True

        output = fused_moe(
            x,
            w13_weight,
            w2_weight,
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_1x32,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation=(
                ActivationType.Silu
                if moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            doweight_stage1=False,
            expert_mask=layer.expert_mask_gpu,
        )
        return StandardCombineInput(hidden_states=output)


class QuarkW8A8FP8MoEMethod(QuarkMoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
        self.is_static_input_scheme: bool = False
        self.input_qscheme = None

        if input_config is not None:
            self.is_static_input_scheme = not input_config.get("is_dynamic")
            self.input_qscheme = input_config.get("qscheme")

        self.input_per_token = (
            not self.is_static_input_scheme and self.input_qscheme == "per_channel"
        )
        self.weight_qscheme = weight_config.get("qscheme")
        self.is_weight_per_channel = self.weight_qscheme == "per_channel"
        self.out_dtype = torch.get_default_dtype()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

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

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # per-tensor quantization
        if self.weight_qscheme == "per_tensor":
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            weight_quant_method = FusedMoeWeightScaleSupported.TENSOR.value
        elif self.weight_qscheme == "per_channel":
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            weight_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_qscheme}."
            )

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update({"quant_method": weight_quant_method})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.is_static_input_scheme:
            assert (
                self.input_qscheme == "per_tensor"
            ), "Only per-tensor quantization is supported for static input scales"
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.is_static_input_scheme:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if _is_fp8_fnuz:
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )
        if self.weight_qscheme == "per_tensor":
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])

                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )
        elif self.weight_qscheme == "per_channel":
            layer.w13_weight_scale = torch.nn.Parameter(
                layer.w13_weight_scale.unsqueeze(-1), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                layer.w2_weight_scale.unsqueeze(-1), requires_grad=False
            )
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_qscheme}."
            )

        if (
            _use_aiter
            and self.is_weight_per_channel
            and self.moe_runner_config.apply_router_weight_on_input
        ):
            with torch.no_grad():
                # Pre-shuffle weights
                layer.w13_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w13_weight.data, (16, 16)),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w2_weight.data, (16, 16)),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if (
            _use_aiter
            and self.is_weight_per_channel
            and moe_runner_config.apply_router_weight_on_input
        ):
            topk_weights, topk_ids, _ = topk_output
            output = rocm_fused_experts_tkw1(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=moe_runner_config.activation,
                apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,
                use_fp8_w8a8=True,
                per_channel_quant=self.is_weight_per_channel,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_fp8_w8a8=True,
                per_channel_quant=self.is_weight_per_channel,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return self.runner.run(dispatch_output, quant_info)
