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


class MxFp4Config(QuantizationConfig):

    def __init__(
        self,
        is_checkpoint_fp4_serialized: bool = False,
        quant_config: dict[str, Any] = None,
        kv_cache_group: Optional[list[str]] = None,
        kv_cache_config: Optional[dict[str, Any]] = None,
        pack_method: str = "reorder",
        ignored_layers: Optional[List[str]] = None,
    ):
        super().__init__()
        if kv_cache_group is None:
            kv_cache_group = []

        self.is_checkpoint_fp4_serialized = is_checkpoint_fp4_serialized
        self.quant_config = quant_config
        self.kv_cache_group = kv_cache_group
        self.kv_cache_config = kv_cache_config
        self.pack_method = pack_method

        self.packed_modules_mapping = (
            self.quant_config["packed_modules_mapping"]
            if is_checkpoint_fp4_serialized
            else None
        )

        self.ignored_layers = ignored_layers or []

    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "fp4"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        # Check if the layer is skipped for quantization.
        if len(self.ignored_layers) > 0 and should_ignore_layer(
            prefix,
            ignore=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()

        if isinstance(layer, LinearBase):
            if self.is_checkpoint_fp4_serialized:
                scheme = self.get_scheme(layer=layer, layer_name=prefix)
                layer.scheme = scheme
                return MxFp4LinearMethod(self)

            elif use_dynamic_mxfp4_linear:
                return MxFp4LinearMethod(self)
            else:
                return UnquantizedLinearMethod()

        if isinstance(layer, RadixAttention):
            return MxFp4KVCacheMethod(self)

        if isinstance(layer, FusedMoE):
            return MxFp4MoEMethod.get_moe_method(self, module=layer, layer_name=prefix)

        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MxFp4Config":
        if not mxfp_supported():
            platform = torch.cuda.get_device_properties(0).gcnArchName
            raise ValueError(
                f"Current platform {platform} not support mxfp4 computation"
            )
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp4_serialized = (
            True if quant_method else False
        )  # "quark" in quant_method

        kv_cache_group = []
        pack_method = None

        if is_checkpoint_fp4_serialized:
            export_config = config.get("export")
            if export_config is None:
                raise ValueError(
                    "The export key should be included in "
                    "the configurations of Quark quantized model"
                )

            kv_cache_group = cast(list[str], export_config.get("kv_cache_group"))
            pack_method = cast(str, export_config.get("pack_method"))

        # In the export model of quark, the quantization configuration
        # of kv_cache is stored in layer_quant_config. First, it is
        # judged whether kv_cache_group exists, and then it is judged
        # whether layer_quant_config has a quantization configuration
        # that matches kv_cache.
        if len(kv_cache_group) == 0:
            kv_cache_config = None
        else:
            kv_cache_set = set(kv_cache_group)
            layer_quant_config = cast(dict[str, Any], config.get("layer_quant_config"))
            layer_quant_names = list(layer_quant_config.keys())
            layer_quant_set = set(layer_quant_names)

            if not kv_cache_set.issubset(layer_quant_set):
                raise ValueError(
                    "The Quark quantized model has the "
                    "kv_cache_group parameter setting, "
                    "but no kv_cache quantization settings "
                    "were found in the quantization "
                    "configuration."
                )

            q_configs = [
                cast(dict[str, Any], layer_quant_config.get(name))
                for name in kv_cache_group
            ]
            if not all(deep_compare(q_config, q_configs[0]) for q_config in q_configs):
                raise ValueError(
                    "The quantization method used for kv_cache should "
                    "be the same, but the quantization method for the "
                    "kv_cache layer in the config is different."
                )
            kv_cache_config = q_configs[0].get("output_tensors")
            if kv_cache_config is None:
                raise ValueError("The kv_cache quantization configuration is empty.")

            # Since we have already set kv_cache quantization configurations,
            # we will remove the quantization configuration for the
            # output_tensors corresponding to the kv_cache layer.
            for q_config in q_configs:
                q_config["output_tensors"] = None

            # In case q_proj output is also quantized, remove the configuration
            # to keep qkv consistency.
            q_proj_q_config = cast(dict[str, Any], layer_quant_config.get("*q_proj"))
            if q_proj_q_config is not None:
                q_proj_q_config["output_tensors"] = None

        ignored_layers = cls.get_from_keys_or(config, ["exclude"], None)

        return cls(
            is_checkpoint_fp4_serialized=is_checkpoint_fp4_serialized,
            quant_config=config,
            kv_cache_group=kv_cache_group,
            kv_cache_config=kv_cache_config,
            pack_method=pack_method,
            ignored_layers=ignored_layers,
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        capability_tuple = get_device_capability()

        if capability_tuple is not None:
            assert 0 <= capability_tuple[1] < 10
            capability = capability_tuple[0] * 10 + capability_tuple[1]

            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.",
                )
            return supported
        else:
            return False

    def _is_mx_fp4(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            logger.debug(
                "Quark model is not in MX-FP4 format: "
                "weight_quant or input_quant not set"
            )
            return False

        # Input and weight dtype needs to be fp4.
        if weight_quant.get("dtype") != "fp4" or input_quant.get("dtype") != "fp4":
            logger.debug("Quark model is not in MX-FP4 format: dtype not fp4")
            return False

        # Input and weight qscheme needs to be per group.
        if (
            weight_quant.get("qscheme") != "per_group"
            or input_quant.get("qscheme") != "per_group"
        ):
            logger.debug("Quark model is not in MX-FP4 format: not per_group")
            return False

        # Input and weight group size needs to be 32.
        if weight_quant.get("group_size") != 32 or input_quant.get("group_size") != 32:
            logger.debug("Quark model is not in MX-FP4 format: not group_size=32")
            return False

        # Weights need to use static quantization.
        if weight_quant.get("is_dynamic") is True:
            logger.debug("Quark model is not in MX-FP4 format: not weight static")
            return False

        # Activations need to use dynamic quantization.
        if input_quant.get("is_dynamic") is False:
            logger.debug("Quark model is not in MX-FP4 format: not activation dynamic")
            return False

        # Activations and weight scales need to be in e8m0 format.
        if (
            weight_quant.get("scale_format") != "e8m0"
            or input_quant.get("scale_format") != "e8m0"
        ):
            logger.debug("Quark model is not in MX-FP4 format: not scale_format e8m0")
            return False

        return True

    def _find_matched_config(
        self, layer_name: str, module: torch.nn.Module
    ) -> dict[str, Any]:

        proj_name = layer_name.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            shard_proj_names = self.packed_modules_mapping[proj_name]

            # Convert fused_name --> [shard_names]
            shard_names = [
                layer_name.replace(proj_name, shard_proj_name)
                for shard_proj_name in shard_proj_names
            ]
            shard_configs = [
                self._find_matched_config(shard_name, module)
                for shard_name in shard_names
            ]
            if not all(
                deep_compare(q_config, shard_configs[0]) for q_config in shard_configs
            ):
                raise ValueError(
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names=} in {layer_name=}. vLLM "
                    "requires all to use the same scheme."
                )
            return shard_configs[0]
        else:
            layer_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_quant_config")
            )
            for name_pattern in layer_quant_config:
                if fnmatch.fnmatch(layer_name, name_pattern):
                    return layer_quant_config[name_pattern]

            layer_type = cast(str, type(module))
            layer_type_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_type_quant_config")
            )
            if layer_type in layer_type_quant_config:
                return layer_type_quant_config[layer_type]

            global_quant_config = cast(
                dict[str, Any], self.quant_config.get("global_quant_config")
            )
            return global_quant_config

    def _get_scheme_from_config(self, config: dict[str, Any]) -> "QuarkScheme":
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported"
            )
        weight_config = cast(dict[str, Any], config.get("weight"))
        input_config = cast(dict[str, Any], config.get("input_tensors"))

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFP4(weight_config, input_config)

        raise NotImplementedError(
            "No quark compatible scheme was found. "
            f"{weight_config=}, "
            f"{input_config=}"
        )

    def get_scheme(self, layer: torch.nn.Module, layer_name: str) -> "QuarkScheme":

        layer_quant_config = self._find_matched_config(layer_name, layer)

        # Find the quant_scheme
        scheme = self._get_scheme_from_config(layer_quant_config)

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())

        return scheme

    def get_scaled_act_names(self) -> List[str]:
        return []


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
