# SPDX-License-Identifier: Apache-2.0

import fnmatch
import logging
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, cast

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (  # noqa: E501
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.quark.schemes import (
    QuarkLinearScheme,
    QuarkMoEScheme,
    QuarkW4A4MXFP4,
    QuarkW4A4MXFp4MoE,
    QuarkW4A8MXFp4MoE,
    QuarkW8A8Fp8,
    QuarkW8A8FP8MoE,
)
from sglang.srt.layers.quantization.quark.utils import deep_compare, should_ignore_layer
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_device_capability

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

__all__ = ["QuarkLinearMethod", "QuarkFusedMoEMethod"]

logger = logging.getLogger(__name__)

_MOE_SHARED_EXPERT_QUANT_LAYER0_BASES: tuple[str, ...] = (
    "model.layers.0",
    "model.language_model.layers.0",
)

_SHARED_EXPERT_BODY_PROJ_SUFFIXES: tuple[str, ...] = (
    "gate_proj",
    "up_proj",
    "gate_up_proj",
    "down_proj",
)


class QuarkConfig(QuantizationConfig):

    def __init__(
        self,
        quant_config: Optional[dict[str, Any]] = None,
        hf_config: "PretrainedConfig | None" = None,
        kv_cache_group: Optional[list[str]] = None,
        kv_cache_config: Optional[dict[str, Any]] = None,
        pack_method: str = "reorder",
        is_prequantized: bool = False,
        online_scheme: Optional[str] = None,
    ):
        super().__init__()
        if kv_cache_group is None:
            kv_cache_group = []

        if online_scheme is not None:
            assert not is_prequantized
            if online_scheme == "quark_mxfp4":
                quant_config = self._create_online_mxfp4_config(
                    model_type=hf_config.model_type
                )
            else:
                raise ValueError(f"Unsupported online_scheme: {online_scheme}")

        if quant_config is None:
            raise ValueError("Either quant_config or online_scheme must be provided")

        self.quant_config = quant_config
        self.kv_cache_group = kv_cache_group
        self.kv_cache_config = kv_cache_config
        self.pack_method = pack_method
        self.exclude_layers = cast(list[str], self.quant_config.get("exclude", []))
        self.is_prequantized = is_prequantized

        self.packed_modules_mapping = self.quant_config["packed_modules_mapping"]
        self._quantized_layers = set()
        # Dense-FP8 policy registered by the active model; empty include -> no-op.
        self._dense_fp8_include: tuple[str, ...] = ()
        self._dense_fp8_exclude: tuple[str, ...] = ()
        self._dense_fp8_min_output_size: int = 0

    @property
    def quantized_layers(self) -> tuple[list[str], int]:
        # Extract unique layer types (last part after ".")
        layer_types = sorted(
            set(name.split(".")[-1] for name in self._quantized_layers)
        )
        return layer_types, len(self._quantized_layers)

    def get_linear_method(self) -> "QuarkLinearMethod":
        return QuarkLinearMethod(self)

    def register_dense_fp8_modules(
        self,
        include: Iterable[str],
        exclude: Iterable[str] = (),
        min_output_size: int = 0,
    ) -> None:
        """Register a model's dense-FP8 policy (quark owns no names/thresholds):
        ``include``/``exclude`` module-name substrings and a min output size. Only
        excluded (bf16) layers matching the policy are promoted, under --enable-dense-fp8.
        """
        self._dense_fp8_include = tuple(
            dict.fromkeys((*self._dense_fp8_include, *include))
        )
        self._dense_fp8_exclude = tuple(
            dict.fromkeys((*self._dense_fp8_exclude, *exclude))
        )
        self._dense_fp8_min_output_size = max(
            self._dense_fp8_min_output_size, min_output_size
        )

    def _dense_fp8_enabled(self) -> bool:
        from sglang.srt.server_args import get_global_server_args

        try:
            return get_global_server_args().enable_dense_fp8
        except ValueError:
            return False

    def _dense_fp8_eligible(self, prefix: str, layer: torch.nn.Module) -> bool:
        if any(b in prefix for b in self._dense_fp8_exclude):
            return False
        if not any(m in prefix for m in self._dense_fp8_include):
            return False
        n = getattr(layer, "output_size_per_partition", None) or getattr(
            layer, "output_size", None
        )
        return n is None or n >= self._dense_fp8_min_output_size

    def _get_dense_fp8_method(self, prefix: str) -> "Fp8LinearMethod":
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

        cfg = getattr(self, "_dense_fp8_config", None)
        if cfg is None:
            cfg = Fp8Config(
                is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
            )
            self._dense_fp8_config = cfg
        logger.info("[quark] routing excluded dense layer to online FP8: %s", prefix)
        return Fp8LinearMethod(cfg)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "quark"

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
        mapped = hf_to_sglang_mapper.apply_list(self.exclude_layers)
        expanded = []
        for name in mapped:
            expanded.append(name)
            if name.startswith("language_model."):
                expanded.append(name.removeprefix("language_model."))
        self.exclude_layers = list(dict.fromkeys(expanded))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        # Check if the layer is skipped for quantization.

        if should_ignore_layer(
            prefix,
            ignore=self.exclude_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            if isinstance(layer, LinearBase):
                if self._dense_fp8_enabled() and self._dense_fp8_eligible(
                    prefix, layer
                ):
                    return self._get_dense_fp8_method(prefix)
                return UnquantizedLinearMethod()
            elif isinstance(layer, RadixAttention):
                return QuarkKVCacheMethod(self)
            return None

        if isinstance(layer, LinearBase):
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            self._quantized_layers.add(prefix)
            return QuarkLinearMethod(self)

        if isinstance(layer, RadixAttention):
            self._quantized_layers.add(prefix)
            return QuarkKVCacheMethod(self)

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if isinstance(layer, FusedMoE):
            self._quantized_layers.add(prefix)
            layer.scheme = self.get_moe_scheme(layer, prefix)
            return QuarkFusedMoEMethod(self)

        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":
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

        return cls(
            quant_config=config,
            kv_cache_group=kv_cache_group,
            kv_cache_config=kv_cache_config,
            pack_method=pack_method,
            is_prequantized=True,
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @staticmethod
    def _create_online_mxfp4_config(model_type: str) -> dict[str, Any]:
        """
        Create a synthetic quant_config for online MXFP4 quantization.
        """
        # MOE gate/router is typically implemented as a ReplicatedLinear, and skipped for quantization for accuracy reasons.
        # lm_head/embed_tokens is also skipped for accuracy reasons, normally not handled by `QuarkConfig` in any case, but adding them here for safety.
        exclude = [
            "re:.*gate$",
            "re:.*router",
            "re:.*lm_head",
            "re:.*embed_tokens",
        ]

        if model_type == "qwen3_5_moe":
            # Exclusion for accuracy adapted from
            # https://huggingface.co/amd/Qwen3.5-397B-A17B-MXFP4/blob/main/config.json
            exclude.extend(
                [
                    "re:.*n_proj_a",
                    "re:.*in_proj_b",
                    "re:.*in_proj_qkv",
                    "re:.*in_proj_z",
                    "re:.*o_proj",
                    "re:.*out_proj",
                    "re:.*qkv_proj",
                    "re:.*shared_expert",
                ]
            )

        return {
            "packed_modules_mapping": {},
            "exclude": exclude,
            "global_quant_config": {
                "weight": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": False,
                    "scale_format": "e8m0",
                },
                "input_tensors": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": True,
                    "scale_format": "e8m0",
                },
                "output_tensors": None,
                "bias": None,
            },
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "export": {
                "kv_cache_group": [],
                "pack_method": "reorder",
            },
        }

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

    def _is_fp8_w8a8(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported
        is_fp8_dtype = (
            weight_quant.get("dtype") == "fp8_e4m3"
            and input_quant.get("dtype") == "fp8_e4m3"
        )
        is_static_weight = not weight_quant.get("is_dynamic")
        is_per_tensor_or_channel_weight = weight_quant.get("qscheme") in [
            "per_tensor",
            "per_channel",
        ]

        if not (is_fp8_dtype and is_static_weight and is_per_tensor_or_channel_weight):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.get("is_dynamic"):
            return True

        # Confirm activation scheme is supported.
        is_per_tensor_activation = input_quant.get("qscheme") == "per_tensor"
        return is_per_tensor_activation

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

    def _is_mx_w4a8(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        if weight_quant is None or input_quant is None:
            return False

        is_mx_fp4_weight = (
            weight_quant.get("dtype") == "fp4"
            and weight_quant.get("qscheme") == "per_group"
            and weight_quant.get("group_size") == 32
            and not weight_quant.get("is_dynamic")
            and weight_quant.get("scale_format") == "e8m0"
        )
        is_static_fp8_activation = (
            input_quant.get("dtype") in ("fp8_e4m3", "fp8_e4m3fn")
            and input_quant.get("qscheme") == "per_tensor"
            and not input_quant.get("is_dynamic")
        )
        return is_mx_fp4_weight and is_static_fp8_activation

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
                    f"{shard_proj_names} in {layer_name}. vLLM "
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

            layer_type = type(module).__name__
            layer_type_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_type_quant_config")
            )
            if layer_type in layer_type_quant_config:
                return layer_type_quant_config[layer_type]

            global_quant_config = cast(
                dict[str, Any], self.quant_config.get("global_quant_config")
            )
            return global_quant_config

    def _get_scheme_from_config(self, config: dict[str, Any]) -> "QuarkLinearScheme":
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported"
            )
        weight_config = cast(dict[str, Any], config.get("weight"))
        input_config = cast(dict[str, Any], config.get("input_tensors"))

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFP4(
                weight_config,
                input_config,
                is_checkpoint_mxfp4_serialized=self.is_prequantized,
            )
        if self._is_fp8_w8a8(weight_config, input_config):
            is_fp8_w8a8_supported = self._check_scheme_supported(
                QuarkW8A8Fp8.get_min_capability(), error=False
            )
            if is_fp8_w8a8_supported:
                return QuarkW8A8Fp8(weight_config, input_config)

        raise NotImplementedError(
            "No quark compatible scheme was found. "
            f"Weight config: {weight_config}, "
            f"Input config: {input_config}"
        )

    def get_linear_scheme(
        self, layer: torch.nn.Module, layer_name: str
    ) -> "QuarkLinearScheme":

        layer_quant_config = self._find_matched_config(layer_name, layer)

        # Find the quant_scheme
        scheme = self._get_scheme_from_config(layer_quant_config)

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())

        return scheme

    def get_moe_scheme(
        self,
        module: torch.nn.Module,
        layer_name: str,
    ) -> "QuarkMoEScheme":
        layer_quant_config = self._find_matched_config(layer_name, module)

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFp4MoE(
                weight_config,
                input_config,
                is_checkpoint_mxfp4_serialized=self.is_prequantized,
            )
        elif self._is_mx_w4a8(weight_config, input_config):
            logger.info_once("Using Quark MXFP4-W/FP8-A MoE scheme")
            return QuarkW4A8MXFp4MoE(weight_config, input_config)
        elif self._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8FP8MoE(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")

    def get_scaled_act_names(self) -> List[str]:
        return []

    def can_fuse_shared_expert(self) -> bool:
        # Shared-expert body excluded from quant; the gate must not veto fusion.
        if any(
            "shared_expert" in layer
            and "shared_expert_gate" not in layer
            and not layer.startswith("mtp.")
            for layer in self.exclude_layers
        ):
            return False

        # No per-layer config -> uniform spec, nothing to compare.
        layer_quant_config = self.quant_config.get("layer_quant_config") or {}
        if not layer_quant_config:
            return True

        # Compare routed vs shared specs at layer 0 (stub module needed by
        # _find_matched_config; an unmatched name -> ValueError -> cannot fuse).
        lookup_stub = torch.nn.Module()
        try:
            for base in _MOE_SHARED_EXPERT_QUANT_LAYER0_BASES:
                moe_name = f"{base}.mlp.experts"
                moe_cfg = self._find_matched_config(moe_name, lookup_stub)
                for suffix in _SHARED_EXPERT_BODY_PROJ_SUFFIXES:
                    shared_name = f"{base}.mlp.shared_expert.{suffix}"
                    shared_cfg = self._find_matched_config(shared_name, lookup_stub)
                    if not deep_compare(moe_cfg, shared_cfg):
                        return False
        except ValueError:
            return False

        return True


class QuarkLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

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
        Use the QuarkLinearScheme associated with the layer to create
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
        Use the output of create_weights and the QuarkLinearScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class QuarkFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
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
        Use the QuarkMoEScheme associated with the layer to create
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
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        """
        Use the output of create_weights and the QuarkMoEScheme
        associated with the layer to apply the forward pass with the
        fused MoE layer. See FusedMoEMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, dispatch_output)


class QuarkKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from quark checkpoints.
    """

    def __init__(self, quant_config: QuarkConfig):
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
