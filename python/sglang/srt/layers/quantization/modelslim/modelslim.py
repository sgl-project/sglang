from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _NPULinearMethodBase,
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.ascend import (
    AscendQuantInfo,
)
from sglang.srt.layers.moe.utils import get_moe_runner_backend
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
            if layer.scheme is None:
                return UnquantizedLinearMethod()
            return ModelSlimLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            layer.w13_scheme, layer.w2_scheme = self.get_moe_scheme(layer, prefix)
            layer.w13_kernel, layer.w2_kernel = (
                layer.w13_scheme.kernel,
                layer.w2_scheme.kernel,
            )
            return ModelSlimFusedMoEMethod(self)
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
    ):
        moe_quant_schemes = [
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4MoE),
            ("W4A8_DYNAMIC", ModelSlimW4A8Int8MoE),
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8MoE),
        ]
        # Suffixes for the two weight groups
        w13_suffixes = [".0.gate_proj.weight", ".0.up_proj.weight"]
        w2_suffix = ".0.down_proj.weight"

        # Look up scheme names from the quant description
        w13_names = [
            self.quant_description.get(prefix + suf, "") for suf in w13_suffixes
        ]
        w2_name = self.quant_description.get(prefix + w2_suffix, "")

        # For w13, gate_proj and up_proj must agree on the scheme
        unique_w13 = set(name for name in w13_names if name)  # ignore empty/missing
        if len(unique_w13) > 1:
            logger.warning(
                f"Mismatched quantization for gate_proj/up_proj in {prefix}: "
                f"{w13_names}. Using the first found scheme."
            )
        w13_scheme_name = next((name for name in w13_names if name), "")

        # Map scheme names to classes
        scheme_map = dict(
            moe_quant_schemes
        )  # dict: "W4A4_DYNAMIC" -> ModelSlimW4A4Int4MoE, etc.

        # Instantiate the schemes
        def instantiate(name, weight_group):
            cls = scheme_map.get(name)
            if cls is None:
                logger.warning(f"Unsupported scheme '{name}' for layer {prefix}")
                return None
            return cls(self, weight_group)

        w13_scheme = instantiate(w13_scheme_name, weight_group="w13")
        logger.info_once(
            f"Using {scheme_map[w13_scheme_name].__name__} for gate_up_proj"
        )
        w2_scheme = instantiate(w2_name, weight_group="w2")
        logger.info_once(f"Using {scheme_map[w2_name].__name__} for down_proj")

        return w13_scheme, w2_scheme

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
    """
    Fused MoE method for ModelSlim quantization on Ascend NPU.

    Delegates routing, activation, and finalization to the modular NPU MoE
    components introduced in the hardware backend refactoring.
    """

    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_scheme.process_weights_after_loading(layer)
        layer.w2_scheme.process_weights_after_loading(layer)

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
        layer.w13_scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            weight_prefix="w13",
            **extra_weight_attrs,
        )
        layer.w2_scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            weight_prefix="w2",
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        self.runner = MoeRunner(backend, moe_runner_config)

    # ------------------------------------------------------------------
    # Main apply()
    # ------------------------------------------------------------------
    def apply(
        self,
        layer,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        backend = self.runner.runner_backend
        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_scale=layer.w2_weight_scale,
            w13_weight_offset=layer.w13_weight_offset,
            w2_weight_offset=layer.w2_weight_offset,
        )
        return self.runner.run(dispatch_output, quant_info)
