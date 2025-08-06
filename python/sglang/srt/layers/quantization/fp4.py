# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast

import aiter
import torch
import torch.nn.functional as F
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.fused_moe_bf16_asm import asm_moe, ck_moe_2stages
from aiter.ops.gemm_op_a4w4 import gemm_a4w4
from aiter.ops.quant import get_torch_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from torch.nn import Module

from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkScheme, QuarkW4A4MXFP4
from sglang.srt.layers.quantization.quark.utils import deep_compare, should_ignore_layer
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    log_info_on_rank0,
    mxfp_supported,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)

use_dynamic_mxfp4_linear = get_bool_env_var("SGLANG_USE_DYNAMIC_MXFP4_linear")

OCP_MX_BLOCK_SIZE = 32


class Mxfp4Config(QuantizationConfig):

    def __init__(self, ignored_layers: Optional[list[str]] = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            raise NotImplementedError("Mxfp4 linear layer is not implemented")
        elif isinstance(layer, FusedMoE):
            return Mxfp4MoEMethod(layer.moe_config)
        elif isinstance(layer, Attention):
            raise NotImplementedError("Mxfp4 attention layer is not implemented")
        return None


class MxFp4LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: MxFp4Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return
        # if self.quantization_config.is_checkpoint_fp4_serialized:
        #    layer.scheme.process_weights_after_loading(layer)
        # else:
        #    #w, w_scales = dynamic_mxfp4_quant(layer.weight.data)
        #    ##log_info_on_rank0(logger, f"w.shape: {w.shape}")

        #    #wshuffle = w#shuffle_weight(w, layout=(16, 16))
        #    #w_scales_shuffle = w_scales#e8m0_shuffle(w_scales).view(dtypes.fp8_e8m0)

        #    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)

        #    w, w_scales_shuffle = quant_func(layer.weight.data, shuffle=True)

        #    wshuffle = shuffle_weight(w, layout=(16, 16))

        #    layer.weight = torch.nn.Parameter(wshuffle,
        #                                      requires_grad=False)
        #    layer.weight_scale = torch.nn.Parameter(w_scales_shuffle,
        #                                            requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.quantization_config.is_checkpoint_fp4_serialized:
            layer.scheme.create_weights(
                layer=layer,
                input_size=input_size,
                input_size_per_partition=input_size_per_partition,
                output_partition_sizes=output_partition_sizes,
                output_size=output_size,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
            )
        else:
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = output_size_per_partition
            layer.orig_dtype = params_dtype

            weight_dtype = params_dtype

            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=weight_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )

            layer.register_parameter("weight", weight)
            layer.register_parameter("weight_scale", None)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        if self.quantization_config.is_checkpoint_fp4_serialized:
            scheme = layer.scheme
            if scheme is None:
                raise ValueError("A scheme must be defined for each layer")
            return scheme.apply_weights(layer, x, bias=bias)
        else:
            out_dtype = x.dtype

            # ck or asm implement
            # M = x.shape[0]
            # N = layer.weight.shape[0]

            # quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)

            # x, x_scales_shuffle = quant_func(x, shuffle=True)

            # y = torch.zeros((M + 255) // 256 * 256, N, device=x.device, dtype=out_dtype)

            # out = gemm_a4w4(x, layer.weight.data, x_scales_shuffle, layer.weight_scale.data, y, bias=bias)

            # return out[:M]

            # triton implement
            x_q, x_s = dynamic_mxfp4_quant(x)
            y = torch.empty(
                x_q.shape[0], layer.weight.shape[0], device=x_q.device, dtype=out_dtype
            )

            out = gemm_afp4wfp4(
                x_q, layer.weight, x_s, layer.weight_scale, out_dtype, y
            )

            return out


class MxFp4MoEMethod:
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

    @staticmethod
    def get_moe_method(
        quant_config: "MxFp4Config",  # type: ignore # noqa E501 # noqa F821
        module: torch.nn.Module,
        layer_name: str,
    ) -> "MxFp4MoEMethod":

        if quant_config.is_checkpoint_fp4_serialized:
            layer_quant_config = quant_config._find_matched_config(layer_name, module)

            if layer_quant_config.get("output_tensors") or layer_quant_config.get(
                "bias"
            ):
                raise NotImplementedError(
                    "Currently, Quark models with "
                    "output_tensors and bias "
                    "quantized are not supported"
                )
            weight_config = layer_quant_config.get("weight")
            input_config = layer_quant_config.get("input_tensors")

            if quant_config._is_mx_fp4(weight_config, input_config):
                return W4A4MXFp4MoEStaticMethod(weight_config, input_config)
            else:
                raise RuntimeError("Unsupported FusedMoe scheme")
        else:
            return W4A4MXFp4MoEDynamicMethod(quant_config)


class W4A4MXFp4MoEDynamicMethod(MxFp4MoEMethod):
    def __init__(self, quant_config):
        self.quant_config = quant_config

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

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def mxfp4_quantize(self, w):
        w_shape = w.shape
        w_need_reshape = True if w.dim() != 2 else False

        if w_need_reshape:
            w_last_dim_size = w_shape[-1]
            w = w.view(-1, w_last_dim_size)

        # log_info_on_rank0(logger, f"[Pre-quant] w.shape: {w.shape}")
        w, mx_scales = dynamic_mxfp4_quant(w)
        # log_info_on_rank0(logger, f"[Post-quant] w.shape: {w.shape} mx_scales.shape: {mx_scales.shape}")

        if w_need_reshape:
            w_new_shape = w_shape[:-1] + (w.shape[-1],)
            w = w.view(w_new_shape)

        # log_info_on_rank0(logger, f"[re-shape] w.shape: {w.shape} mx_scales.shape: {mx_scales.shape}")

        mx_scales = e8m0_shuffle(mx_scales)

        return w, mx_scales

    def process_weights_after_loading(self, layer: Module) -> None:
        w13, w13_mx_scales = self.mxfp4_quantize(layer.w13_weight.data)
        w2, w2_mx_scales = self.mxfp4_quantize(layer.w2_weight.data)

        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_mx_scales, requires_grad=False)

        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_mx_scales, requires_grad=False)

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
        topk_weights, topk_ids, _ = topk_output

        return fused_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_1x32,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation=(
                ActivationType.Silu if activation == "silu" else ActivationType.Gelu
            ),
            doweight_stage1=False,
        )


class W4A4MXFp4MoEStaticMethod(MxFp4MoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group" and input_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme=}, {input_qscheme=}"
            )  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")

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

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
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
                intermediate_size_per_partition // 2,
                dtype=params_dtype,
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        float_dtype = torch.get_default_dtype()

        # Pre-shuffle weight scales
        s0, s1, _ = layer.w13_weight_scale.shape
        w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)
        w13_weight_scale = e8m0_shuffle(w13_weight_scale)
        layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)

        s0, s1, _ = layer.w2_weight_scale.shape
        w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)
        w2_weight_scale = e8m0_shuffle(w2_weight_scale)
        layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)

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
        topk_weights, topk_ids, _ = topk_output

        return fused_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_1x32,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation=(
                ActivationType.Silu if activation == "silu" else ActivationType.Gelu
            ),
            doweight_stage1=False,
        )


class MxFp4KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from quark checkpoints.
    """

    def __init__(self, quant_config: MxFp4Config):
        self.validate_kv_cache_config(quant_config.kv_cache_config)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_config(kv_cache_config: Optional[dict[str, Any]]):
        """
        Validator for the kv cache configuration. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM
        :param kv_cache_config: the quark kv cache scheme
        """
        if kv_cache_config is None:
            return

        dtype = kv_cache_config.get("dtype")
        if dtype != "fp8_e4m3":
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                f"dtype=fp8_e4m3, however received {dtype}"
            )

        qscheme = kv_cache_config.get("qscheme")
        if qscheme != "per_tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for quark KV cache. "
                f"Expected qscheme: per_tensor, found qscheme: {qscheme}"
            )
