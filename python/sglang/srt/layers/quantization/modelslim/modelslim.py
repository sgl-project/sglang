from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch
import torch_npu
import numpy as np

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _NPULinearMethodBase,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimMXFP8Scheme,
    ModelSlimW4A4Int4,
    ModelSlimW4A4Int4MoE,
    ModelSlimW4A8Int8MoE,
    ModelSlimW8A8Int8,
    ModelSlimW8A8Int8MoE,
)
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, _ColumnvLLMParameter
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import apply_module_patch

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
    from sglang.srt.layers.quantization.modelslim.schemes import (
        ModelSlimLinearScheme,
        ModelSlimMoEScheme,
    )

logger = logging.getLogger(__name__)


# func refers to RMSNorm.__init__
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size: int, **extra_args) -> None:
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # The Ascend w8a8_int8 quantization requires adding a bias in rmsnorm
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    return init


# func refers to RMSNorm.forward_oot
def npu_wrapper_rmsnorm_forward(func):
    def _rmsnorm_forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias

            out, residual_out = add_rmsnorm_bias(
                x,
                residual,
                self.weight.data,
                self.bias,
                self.variance_epsilon,
            )
            return out.to(x.dtype), residual_out

        out = torch.ops.npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)

    return _rmsnorm_forward_oot


class ModelSlimConfig(QuantizationConfig):
    """
    Config class for ModelSlim Quantization, a NPU-specific quantization type.
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        keys = [k for k in quant_config if isinstance(k, str)]
        is_dsv4 = any(k.startswith("hc_head_") for k in keys)
        if is_dsv4:
            from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

            remap = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
            quant_config = {
                (remap(k) if isinstance(k, str) else k): v
                for k, v in quant_config.items()
            }

        self.quant_description = quant_config
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )

        for name in self.quant_description.keys():
            if "norm.bias" in name:
                apply_module_patch(
                    "sglang.srt.layers.layernorm.RMSNorm",
                    "__init__",
                    [npu_wrapper_rmsnorm_init],
                )
                apply_module_patch(
                    "sglang.srt.layers.layernorm.RMSNorm",
                    "forward_npu",
                    [npu_wrapper_rmsnorm_forward],
                )

    def update_packed_modules_mapping(self, mapping: Dict[str, List[str]]) -> None:
        self.packed_modules_mapping.update(mapping)

    def get_linear_method(self) -> ModelSlimLinearMethod:
        return ModelSlimLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_name(cls) -> str:
        return "modelslim"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = ["quant_model_description.json"]
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelSlimConfig:
        return cls(config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        if isinstance(layer, LinearBase):
            # TODO: we should remove this code and switch to the packed_modules_mapping declared inside the modeling files
            key = "model"
            if "vision_model" in prefix:
                key = "vision_model"
            elif "visual" in prefix:
                key = "visual"
            if "vision_tower" in prefix or "mm_projector" in prefix:
                prefix = prefix.replace(r"attn.qkv_proj", r"wqkv")
                prefix = prefix.replace(r"attn.proj", r"wo")
            packed_modules_mapping_subset = self.packed_modules_mapping.get(key, {})
            prefix_in_quant_config = prefix
            proj_name = prefix.split(".")[-1]
            if proj_name in packed_modules_mapping_subset:
                prefix_in_quant_config = prefix.replace(
                    proj_name, packed_modules_mapping_subset[proj_name][0]
                )
            if self.is_layer_skipped(
                prefix, packed_modules_mapping_subset
            ) or self.is_layer_skipped(prefix, self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            layer.scheme = self.get_linear_scheme(layer, prefix_in_quant_config)
            return ModelSlimLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer, prefix)
            return ModelSlimFusedMoEMethod(self)
        elif isinstance(layer, RadixAttention):
            return ModelSlimKVCacheMethod(self)
        return None

    def get_linear_scheme(
        self, layer: torch.nn.Module, prefix: Optional[str] = None
    ) -> Optional[ModelSlimLinearScheme]:
        """
        get_scheme method adjusted for modelslim, taken from
        python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
        """

        linear_quant_schemes = [
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4),
            ("W8A8", ModelSlimW8A8Int8),
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8),
            ("W8A8_MXFP8", ModelSlimMXFP8Scheme),
        ]

        quant_schemes = [self.quant_description.get(prefix + ".weight", "")]

        for scheme_name, scheme_class in linear_quant_schemes:
            if any(s == scheme_name for s in quant_schemes):
                logger.info_once(f"Using {scheme_class.__name__}")
                return scheme_class(quant_config=self.quant_description, prefix=prefix)

        logger.warning(
            f"Unsupported Linear modelslim scheme: "
            f"{quant_schemes} in layer: {prefix}"
        )
        return None

    def get_moe_scheme(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[ModelSlimMoEScheme]:
        moe_quant_schemes = [
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4MoE),
            ("W4A8_DYNAMIC", ModelSlimW4A8Int8MoE),
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8MoE),
        ]

        moe_weight_suffixes = [".0.gate_proj.weight", ".0.w2.weight"]
        quant_schemes = [
            self.quant_description.get(prefix + suffix, "")
            for suffix in moe_weight_suffixes
        ]

        for scheme_name, scheme_class in moe_quant_schemes:
            if any(s == scheme_name for s in quant_schemes):
                logger.info_once(f"Using {scheme_class.__name__}")
                return scheme_class(self)

        logger.warning(
            f"Unsupported FusedMoe modelslim scheme: "
            f"{quant_schemes} in layer: {prefix}"
        )
        return None

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = (
                    self.quant_description.get(shard_prefix + ".weight", "") == "FLOAT"
                )

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision."
                    )
        else:
            is_skipped = self.quant_description.get(prefix + ".weight", "") == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


class ModelSlimLinearMethod(_NPULinearMethodBase):

    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the ModelSlimLinearScheme associated with the layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Use the output of create_weights and the ModelSlimLinearScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class ModelSlimFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the ModelSlimMoEScheme associated with the layer to create
        the necessary parameters for the layer. See FusedMoEMethodBase for param
        details
        """
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        return layer.scheme.create_moe_runner(layer, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """
        Use the output of create_weights and the ModelSlimMoEScheme
        associated with the layer to apply the forward pass with the
        layer input.  See FusedMoEMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, dispatch_output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        return layer.scheme.apply_without_routing_weights(
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )


class ModelSlimKVCacheMethod(BaseKVCacheMethod):
    def __init__(self, config):
        super().__init__(config)

    def create_weights(
        self,
        layer: nn.Module,
        head_size: int,
        num_kv_heads: int,
        tp_rank: int,
        **extra_weight_attrs,
    ):
        # Deleting scales created in base Attention class to register new ones.
        del layer.k_scale
        del layer.v_scale
        self.kv_size = num_kv_heads * head_size
        self.tp_rank = tp_rank

        k_scale = ChannelQuantScaleParameter(
            data=torch.full([self.kv_size], fill_value=-1, dtype=torch.float32),
            output_dim=0,
            weight_loader=self.weight_loader,
        )
        layer.register_parameter("k_scale", k_scale)
        v_scale = ChannelQuantScaleParameter(
            data=torch.full([self.kv_size], fill_value=-1, dtype=torch.float32),
            output_dim=0,
            weight_loader=self.weight_loader,
        )
        layer.register_parameter("v_scale", v_scale)
        k_offset = ChannelQuantScaleParameter(
            data=torch.full([self.kv_size], fill_value=-1, dtype=torch.float32),
            output_dim=0,
            weight_loader=self.weight_loader,
        )
        layer.register_parameter("k_offset", k_offset)
        v_offset = ChannelQuantScaleParameter(
            data=torch.full([self.kv_size], fill_value=-1, dtype=torch.float32),
            output_dim=0,
            weight_loader=self.weight_loader,
        )
        layer.register_parameter("v_offset", v_offset)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        device = layer.k_scale.device
        dtype = torch.bfloat16

        k_offset = torch.from_numpy(np.frombuffer(layer.k_offset.to(torch.float32).cpu().numpy().tobytes(), dtype=np.int32
        ).copy()).to(device)
        v_offset = torch.from_numpy(np.frombuffer(layer.v_offset.to(torch.float32).cpu().numpy().tobytes(), dtype=np.int32
        ).copy()).to(device)

        layer.k_quant_offset = torch.nn.Parameter(k_offset.to(dtype), requires_grad=False).unsqueeze(0)
        layer.v_quant_offset = torch.nn.Parameter(v_offset.to(dtype), requires_grad=False).unsqueeze(0)
        layer.k_dequant_scale = torch.nn.Parameter(layer.k_scale.to(dtype), requires_grad=False).unsqueeze(0)
        layer.v_dequant_scale = torch.nn.Parameter(layer.v_scale.to(dtype), requires_grad=False).unsqueeze(0)
        layer.k_quant_scale = torch.nn.Parameter(layer.k_scale.reciprocal().to(dtype), requires_grad=False).unsqueeze(0)
        layer.v_quant_scale = torch.nn.Parameter(layer.v_scale.reciprocal().to(dtype), requires_grad=False).unsqueeze(0)

    def anti_quant_int8(self, k_cache, v_cache, layer):
        old_shape = k_cache.shape
        k_cache = k_cache.view(-1, self.kv_size)
        v_cache = v_cache.view(-1, self.kv_size)
        k_cache = torch_npu.npu_anti_quant(
            x=k_cache,
            scale=layer.k_scale,
            dst_dtype=torch.bfloat16
        )
        v_cache = torch_npu.npu_anti_quant(
            x=v_cache,
            scale=layer.v_scale,
            dst_dtype=torch.bfloat16
        )
        k_cache = k_cache.view(old_shape)
        v_cache = v_cache.view(old_shape)
        return k_cache, v_cache

    def apply(self, k_cache, v_cache, layer):
        #TODO: add dynamic quantization support
        old_shape = k_cache.shape
        k_cache = k_cache.view(-1, self.kv_size)
        v_cache = v_cache.view(-1, self.kv_size)

        key_int8 = torch_npu.npu_quantize(
                    k_cache,
                    layer.k_quant_scale.squeeze(),
                    layer.k_quant_offset.squeeze() if hasattr(layer,"k_quant_offset") else None,
                    torch.qint8,
                    -1,
                    False)

        value_int8 = torch_npu.npu_quantize(
                    v_cache,
                    layer.v_quant_scale.squeeze(),
                    layer.v_quant_offset.squeeze() if hasattr(layer,"v_quant_offset") else None,
                    torch.qint8,
                    -1,
                    False)

        key_int8 = key_int8.view(old_shape)
        value_int8 = value_int8.view(old_shape)

        return key_int8, value_int8

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if isinstance(param, _ColumnvLLMParameter):
            logger.info_once(f"Loading kv cache scales...")
            param.load_column_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=False,
            )
        