from __future__ import annotations

import logging
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Union

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported
from sglang.srt.layers.quantization.utils import (
    get_dynamic_override,
    get_linear_quant_method,
    get_scalar_types,
    resolve_dynamic_fused_shards,
)
from sglang.srt.utils import is_hip
from sglang.srt.utils.patch_torch import register_fake_if_exists

from .schemes import (
    GPTQAscendLinearScheme,
    GPTQIntelAMXLinearScheme,
    GPTQIntelAMXMoEScheme,
    GPTQLinearScheme,
    GPTQMarlinLinearScheme,
    GPTQMarlinMoEScheme,
    GPTQMoEAscendScheme,
    GPTQTritonLinearScheme,
    GPTQXPULinearScheme,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.models.utils import WeightsMapper

logger = logging.getLogger(__name__)
_, scalar_types = get_scalar_types()


def check_marlin_format(hf_quant_cfg: Dict[str, Any]) -> bool:
    # compat: gptqmodel and autogptq (eol) main use checkpoint_format: str
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    return hf_quant_cfg.get("checkpoint_format") == "marlin" or hf_quant_cfg.get(
        "is_marlin_format", False
    )


def _canonicalize_gptq_module_prefix(name: str) -> str:
    """Normalize checkpoint and runtime wrapper prefixes without conflating MTP."""
    name = name.lstrip(".").replace(".self_attn.", ".")
    for wrapper in (
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.",
    ):
        if name.startswith(wrapper):
            return name.removeprefix(wrapper)
    return name


class _CheckpointTensorManifest:
    """Immutable index of dense vs GPTQ-quantized checkpoint modules.

    ``__deepcopy__`` returns ``self`` in O(1) because ``get_linear_quant_method``
    deep-copies the quantization config once per linear layer.
    """

    def __init__(
        self,
        weight_modules: Iterable[str],
        qweight_modules: Iterable[str],
    ) -> None:
        self.weight_modules = frozenset(m.lstrip(".") for m in weight_modules if m)
        self.qweight_modules = frozenset(m.lstrip(".") for m in qweight_modules if m)
        self.canon_weight_modules = frozenset(
            _canonicalize_gptq_module_prefix(m) for m in self.weight_modules
        )
        self.canon_qweight_modules = frozenset(
            _canonicalize_gptq_module_prefix(m) for m in self.qweight_modules
        )

    @classmethod
    def from_tensor_names(
        cls, tensor_names: Iterable[str]
    ) -> _CheckpointTensorManifest:
        weight_modules: Set[str] = set()
        qweight_modules: Set[str] = set()
        for name in tensor_names:
            if name.endswith(".weight"):
                weight_modules.add(name.removesuffix(".weight"))
            elif name.endswith(".qweight"):
                qweight_modules.add(name.removesuffix(".qweight"))
        return cls(weight_modules, qweight_modules)

    def __deepcopy__(self, memo: dict[int, Any]) -> _CheckpointTensorManifest:
        return self

    def with_weight_name_mapper(
        self, hf_to_sglang_mapper: WeightsMapper
    ) -> _CheckpointTensorManifest:
        def _map_set(modules: frozenset[str]) -> Set[str]:
            mapped_set = set(modules)
            for mod in modules:
                mapped = hf_to_sglang_mapper._map_name(f"{mod}.weight")
                if mapped is not None and mapped.endswith(".weight"):
                    mapped_set.add(mapped.removesuffix(".weight"))
            return mapped_set

        return _CheckpointTensorManifest(
            _map_set(self.weight_modules),
            _map_set(self.qweight_modules),
        )

    def classify_module(self, target: str) -> Optional[str]:
        raw_target = target.lstrip(".")
        has_dense = raw_target in self.weight_modules
        has_quant = raw_target in self.qweight_modules
        if has_dense and has_quant:
            raise ValueError(
                f"Checkpoint contains both dense '.weight' and quantized "
                f"'.qweight' tensors for '{target}'."
            )
        if has_quant:
            return "quantized"
        if has_dense:
            return "dense"

        canon = _canonicalize_gptq_module_prefix(raw_target)
        has_dense = canon in self.canon_weight_modules
        has_quant = canon in self.canon_qweight_modules
        if has_dense and has_quant:
            raise ValueError(
                f"Checkpoint contains both dense '.weight' and quantized "
                f"'.qweight' tensors for '{target}'."
            )
        if has_quant:
            return "quantized"
        if has_dense:
            return "dense"
        return None


class _GPTQCheckpointMixin:
    """Shared checkpoint-manifest & packed-shard resolution for GPTQ configs."""

    _checkpoint_tensor_manifest: Optional[_CheckpointTensorManifest] = None

    def set_checkpoint_tensor_names(
        self, tensor_names: Optional[Iterable[str]]
    ) -> None:
        if tensor_names is None:
            self._checkpoint_tensor_manifest = None
        else:
            self._checkpoint_tensor_manifest = (
                _CheckpointTensorManifest.from_tensor_names(tensor_names)
            )

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper) -> None:
        if self._checkpoint_tensor_manifest is not None:
            self._checkpoint_tensor_manifest = (
                self._checkpoint_tensor_manifest.with_weight_name_mapper(
                    hf_to_sglang_mapper
                )
            )

    def is_checkpoint_module_unquantized(self, prefix: str) -> bool:
        manifest = getattr(self, "_checkpoint_tensor_manifest", None)
        if not prefix or manifest is None:
            return False

        direct_state = manifest.classify_module(prefix)
        if direct_state is not None:
            return direct_state == "dense"

        proj_name = prefix.rsplit(".", 1)[-1]
        fused_shards = resolve_dynamic_fused_shards(self, proj_name)
        if not fused_shards:
            return False

        parent = prefix[: -len(proj_name)]
        shard_states = [
            manifest.classify_module(f"{parent}{shard}") for shard in fused_shards
        ]
        if "dense" in shard_states and "quantized" in shard_states:
            raise ValueError(
                f"Detected mixed dense and quantized shards for fused layer "
                f"'{prefix}' in checkpoint ({dict(zip(fused_shards, shard_states))}). "
                "All shards of a fused layer must share the same precision."
            )
        return all(state == "dense" for state in shard_states)

    def is_linear_unquantized(self, prefix: str) -> bool:
        return get_dynamic_override(
            self, layer_name=prefix
        ) is False or self.is_checkpoint_module_unquantized(prefix)

    def _resolve_unquantized_linear_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> tuple[bool, Optional[QuantizeMethodBase]]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
            UnquantizedLinearMethod,
        )
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        parallel_lm_head_quantized = (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        )
        if not (isinstance(layer, LinearBase) or parallel_lm_head_quantized):
            return True, None
        if self.is_linear_unquantized(prefix):
            return True, (
                UnquantizedEmbeddingMethod()
                if parallel_lm_head_quantized
                else UnquantizedLinearMethod()
            )
        return False, None


class GPTQConfig(_GPTQCheckpointMixin, QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
        checkpoint_format: str = "",
        true_sequential: bool = False,
        static_groups: bool = False,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        super().__init__()
        if packed_modules_mapping is not None:
            self.packed_modules_mapping = packed_modules_mapping
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = Fraction(32, self.weight_bits)
        # GPTQ v1 and v2 format deals with zero points differently.
        # Currently GPTQModel stores v1 format checkpoints by default,
        # but provides the option to set `format="gptq_v2"` in `QuantizeConfig`.
        self.checkpoint_format = checkpoint_format
        self.true_sequential = true_sequential
        self.static_groups = static_groups
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits."
            )

    def __repr__(self) -> str:
        return (
            f"GPTQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}),"
            f"lm_head_quantized={self.lm_head_quantized}), "
            f"dynamic={self.dynamic},"
            f"checkpoint_format={self.checkpoint_format})"
        )

    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> GPTQConfig:
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        checkpoint_format = cls.get_from_keys_or(
            config, ["checkpoint_format"], default=""
        )
        true_sequential = cls.get_from_keys_or(
            config, ["true_sequential"], default=False
        )
        static_groups = cls.get_from_keys_or(config, ["static_groups"], default=False)
        packed_modules_mapping = cls.get_from_keys_or(
            config, ["packed_modules_mapping"], default=None
        )
        return cls(
            weight_bits,
            group_size,
            desc_act,
            lm_head_quantized,
            dynamic,
            checkpoint_format,
            true_sequential,
            static_groups,
            packed_modules_mapping=packed_modules_mapping,
        )

    def _get_scheme_linear_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None
        handled, unquant_method = self._resolve_unquantized_linear_method(layer, prefix)
        if handled:
            return unquant_method
        quant_method = get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )
        if isinstance(quant_method, GPTQLinearMethod):
            layer.scheme = quant_method.quant_config.get_linear_scheme(layer)
        return quant_method

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            raise TypeError("GPTQ Method does not support MoE, please use gptq_marlin")
        handled, unquant_method = self._resolve_unquantized_linear_method(layer, prefix)
        if handled:
            return unquant_method
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )

    def get_linear_scheme(self, layer: torch.nn.Module):
        if is_hip():
            return GPTQTritonLinearScheme(self)
        return GPTQLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        raise NotImplementedError("GPTQConfig does not support MoE.")


class GPTQAscendConfig(GPTQConfig):
    """Config class for GPTQ on Ascend NPU."""

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            'NPU hardware does not support "get_min_capability" feature.'
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer)
            return GPTQMoEMethod(self)
        return self._get_scheme_linear_quant_method(layer, prefix)

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQAscendLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        return GPTQMoEAscendScheme(self)


class CPUGPTQConfig(GPTQConfig):
    """CPU Config class for GPTQ on Intel CPU with AMX."""

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer)
            return GPTQMoEMethod(self)
        return self._get_scheme_linear_quant_method(layer, prefix)

    def get_linear_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.linear import LinearBase

        assert isinstance(layer, LinearBase)
        return GPTQIntelAMXLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        return GPTQIntelAMXMoEScheme(self)


class GPTQXPUConfig(GPTQConfig):
    """Config class for GPTQ on Intel XPU.

    Dense int4 GPTQ lowers to torch's native
    ``_weight_int4pack_mm_with_scales_and_zeros`` op (no Marlin on XPU). MoE is
    out of scope for the dense phase.
    """

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            raise NotImplementedError(
                "GPTQ MoE is not yet supported on XPU (dense-only phase)."
            )
        handled, unquant_method = self._resolve_unquantized_linear_method(layer, prefix)
        if handled:
            return unquant_method
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQXPULinearScheme(self)


class GPTQMarlinConfig(_GPTQCheckpointMixin, QuantizationConfig):
    """Config class for GPTQ Marlin"""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
        full_config: Dict[str, Any],
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__()
        if packed_modules_mapping is not None:
            self.packed_modules_mapping = packed_modules_mapping
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.is_sym = is_sym

        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.full_config = full_config

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported quantization config: bits={weight_bits}, sym={is_sym}"
            )

        # (num_bits, is_sym) -> quant_type
        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

    def __repr__(self) -> str:
        return (
            f"GPTQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}), "
            f"dynamic={self.dynamic}"
        )

    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> GPTQMarlinConfig:
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        packed_modules_mapping = cls.get_from_keys_or(
            config, ["packed_modules_mapping"], default=None
        )
        return cls(
            weight_bits,
            group_size,
            desc_act,
            is_sym,
            lm_head_quantized,
            dynamic,
            config,
            packed_modules_mapping=packed_modules_mapping,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        is_marlin_format = check_marlin_format(hf_quant_cfg)

        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)

        is_valid_user_quant = (
            user_quant is None or user_quant == "marlin" or user_quant == "gptq_marlin"
        )

        if not is_marlin_format and can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()

        if not is_marlin_format and can_convert and user_quant == "gptq":
            logger.info(
                "Detected that the model can run with gptq_marlin"
                ", however you specified quantization=gptq explicitly,"
                " so forcing gptq. Use quantization=gptq_marlin for"
                " faster inference"
            )
        return None

    def is_linear_unquantized(self, prefix: str) -> bool:
        # Base implementation probes with LinearBase(1, 1), which fails Marlin
        # tile alignment and would falsely report quantized layers as unquantized.
        return super().is_linear_unquantized(prefix)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        # Delay the import to avoid circular dependency
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.marlin_utils import (
            verify_marlin_supports_layer,
        )

        if isinstance(layer, FusedMoE):
            return GPTQMarlinMoEMethod(self)
        handled, unquant_method = self._resolve_unquantized_linear_method(layer, prefix)
        if handled:
            return unquant_method
        quant_method = get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQMarlinLinearMethod
        )
        if isinstance(quant_method, GPTQMarlinLinearMethod) and isinstance(
            layer, LinearBase
        ):
            verify_marlin_supports_layer(
                layer, quant_method.quant_config.group_size, prefix=prefix
            )
        return quant_method

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQMarlinLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        return GPTQMarlinMoEScheme(self)

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        sym = quant_config.get("sym")
        desc_act = quant_config.get("desc_act")

        if quant_method != "gptq":
            return False

        # Marlin conversion is only valid if required properties are found
        if num_bits is None or group_size is None or sym is None or desc_act is None:
            return False

        if (num_bits, sym) not in cls.TYPE_MAP:
            return False

        try:
            return check_marlin_supported(
                quant_type=cls.TYPE_MAP[(num_bits, sym)], group_size=group_size
            )
        except Exception:
            return False


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

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
        if layer.scheme is None:
            layer.scheme = self.quant_config.get_linear_scheme(layer)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer.scheme.apply_weights(layer, x, bias)


class GPTQMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: GPTQConfig):
        super().__init__()
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
        if layer.scheme is None:
            layer.scheme = self.quant_config.get_moe_scheme(layer)
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self,
        layer: torch.nn.Module,
        moe_runner_config: MoeRunnerConfig,
        **extra_weight_attrs,
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return layer.scheme.get_moe_runner_backend(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        return layer.scheme.apply_weights(layer, dispatch_output)


class GPTQMarlinLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Marlin.

    Args:
        quant_config: The GPTQ Marlin quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if layer.scheme is None:
            layer.scheme = self.quant_config.get_linear_scheme(layer)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer.scheme.apply_weights(layer, x, bias)


class GPTQMarlinMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
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
        if layer.scheme is None:
            layer.scheme = self.quant_config.get_moe_scheme(layer)
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return layer.scheme.get_moe_runner_backend(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return layer.scheme.apply_weights(layer, dispatch_output)


# Register fake implementations for torch.compile support. The decorator is a
# no-op when the custom op is unavailable on the current platform.
@register_fake_if_exists("sgl_kernel::gptq_marlin_repack")
def _(b_q_weight, perm, size_k, size_n, num_bits):
    return b_q_weight.new_empty(
        (size_k // 16, size_n * (num_bits // 2)), dtype=b_q_weight.dtype
    )
