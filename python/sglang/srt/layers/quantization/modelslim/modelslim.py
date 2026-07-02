from __future__ import annotations

import logging
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch

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


_DEFAULT_PACKED_MODULES_MAPPING: Mapping[str, Mapping[str, List[str]]] = {
    "model": {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # MiniMax-M3's sparse index branch has no index_v_proj in current
        # ModelSlim W8A8 checkpoints; value-enabled checkpoints can still load
        # index_v_proj weights via the model weight loader when present.
        "index_qkv_proj": ["index_q_proj", "index_k_proj"],
    }
}


def _require_modelslim_scheme(layer: torch.nn.Module, layer_kind: str):
    scheme = getattr(layer, "scheme", None)
    if scheme is None:
        raise ValueError(
            f"ModelSlim {layer_kind} quantization scheme is missing. "
            "Check quant_model_description.json and packed_modules_mapping for "
            "this layer prefix."
        )
    return scheme


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

    def update_packed_modules_mapping(
        self, mapping: Mapping[str, Mapping[str, List[str]]]
    ) -> None:
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
            packed_modules_mapping_subset = self.get_packed_modules_mapping_subset(key)
            if self.is_layer_skipped(
                prefix, packed_modules_mapping_subset
            ) or self.is_layer_skipped(prefix, self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            layer.scheme = self.get_linear_scheme(
                layer, prefix, packed_modules_mapping_subset
            )
            return ModelSlimLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer, prefix)
            return ModelSlimFusedMoEMethod(self)
        return None

    def get_packed_modules_mapping_subset(self, key: str) -> Mapping[str, List[str]]:
        default_mapping = _DEFAULT_PACKED_MODULES_MAPPING.get(key, {})
        configured_mapping = self.packed_modules_mapping.get(key, {})
        if not default_mapping:
            return configured_mapping
        if not configured_mapping:
            return default_mapping
        return {**default_mapping, **configured_mapping}

    @staticmethod
    def iter_packed_linear_prefixes(
        prefix: str,
        packed_modules_mapping_subset: Mapping[str, List[str]],
    ) -> Iterable[str]:
        for linear_prefix in ModelSlimConfig.iter_linear_prefix_aliases(prefix):
            yield linear_prefix
            proj_name = linear_prefix.split(".")[-1]
            if proj_name not in packed_modules_mapping_subset:
                continue
            for shard_proj_name in packed_modules_mapping_subset[proj_name]:
                yield linear_prefix.replace(proj_name, shard_proj_name)

    @staticmethod
    def iter_linear_prefix_aliases(prefix: str) -> Iterable[str]:
        yield prefix
        if ".mlp.shared_experts" in prefix:
            yield prefix.replace(
                ".mlp.shared_experts", ".block_sparse_moe.shared_experts"
            )
        if ".block_sparse_moe.shared_experts" in prefix:
            yield prefix.replace(
                ".block_sparse_moe.shared_experts", ".mlp.shared_experts"
            )

    @staticmethod
    def iter_moe_prefix_aliases(prefix: str) -> Iterable[str]:
        yield prefix
        if ".mlp.experts" in prefix:
            yield prefix.replace(".mlp.experts", ".block_sparse_moe.experts")
        if ".block_sparse_moe.experts" in prefix:
            yield prefix.replace(".block_sparse_moe.experts", ".mlp.experts")

    def get_linear_scheme(
        self,
        layer: torch.nn.Module,
        prefix: Optional[str] = None,
        packed_modules_mapping_subset: Mapping[str, List[str]] = MappingProxyType({}),
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

        quant_prefixes = list(
            dict.fromkeys(
                self.iter_packed_linear_prefixes(prefix, packed_modules_mapping_subset)
            )
        )
        quant_schemes = [
            self.quant_description.get(quant_prefix + ".weight", "")
            for quant_prefix in quant_prefixes
        ]

        for scheme_name, scheme_class in linear_quant_schemes:
            for quant_prefix, quant_scheme in zip(quant_prefixes, quant_schemes):
                if quant_scheme == scheme_name:
                    logger.info_once(f"Using {scheme_class.__name__}")
                    return scheme_class(
                        quant_config=self.quant_description, prefix=quant_prefix
                    )

        logger.warning(
            f"Unsupported Linear modelslim scheme: "
            f"{quant_schemes} in layer: {prefix}, candidates: {quant_prefixes}"
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

        moe_weight_suffixes = [
            ".0.gate_proj.weight",
            ".0.up_proj.weight",
            ".0.down_proj.weight",
            ".0.w1.weight",
            ".0.w2.weight",
            ".0.w3.weight",
        ]
        quant_prefixes = list(dict.fromkeys(self.iter_moe_prefix_aliases(prefix)))
        quant_schemes = [
            self.quant_description.get(quant_prefix + suffix, "")
            for quant_prefix in quant_prefixes
            for suffix in moe_weight_suffixes
        ]

        for scheme_name, scheme_class in moe_quant_schemes:
            if any(s == scheme_name for s in quant_schemes):
                logger.info_once(f"Using {scheme_class.__name__}")
                return scheme_class(self)

        logger.warning(
            f"Unsupported FusedMoe modelslim scheme: "
            f"{quant_schemes} in layer: {prefix}, candidates: {quant_prefixes}"
        )
        return None

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        for linear_prefix in self.iter_linear_prefix_aliases(prefix):
            if proj_name in fused_mapping:
                shard_prefixes = [
                    linear_prefix.replace(proj_name, shard_proj_name)
                    for shard_proj_name in fused_mapping[proj_name]
                ]
            else:
                shard_prefixes = [linear_prefix]

            present_schemes = [
                self.quant_description[shard_prefix + ".weight"]
                for shard_prefix in shard_prefixes
                if shard_prefix + ".weight" in self.quant_description
            ]
            if not present_schemes:
                continue

            is_skipped = present_schemes[0] == "FLOAT"
            if any((scheme == "FLOAT") != is_skipped for scheme in present_schemes):
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
            return is_skipped

        return False

    def get_scaled_act_names(self) -> List[str]:
        return []


class ModelSlimLinearMethod(_NPULinearMethodBase):

    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _require_modelslim_scheme(layer, "Linear").process_weights_after_loading(layer)

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
        _require_modelslim_scheme(layer, "Linear").create_weights(
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
        _require_modelslim_scheme(layer, "MoE").process_weights_after_loading(layer)

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
        _require_modelslim_scheme(layer, "MoE").create_weights(
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
        return _require_modelslim_scheme(layer, "MoE").create_moe_runner(
            layer, moe_runner_config
        )

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
