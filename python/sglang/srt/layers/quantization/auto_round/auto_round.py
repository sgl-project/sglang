# SPDX-License-Identifier: Apache-2.0

import logging
from fractions import Fraction
from typing import Any, Optional, Union

import torch

from sglang.srt.layers.quantization.auto_round.mx import (
    _from_compressed_tensors_config,
    _from_native_mxfp_config,
)
from sglang.srt.layers.quantization.auto_round.routing import (
    apply_awq_quant_layer,
    apply_gptq_quant_layer,
    get_layer_config,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import cpu_has_amx_support

logger = logging.getLogger(__name__)
ScalarType, scalar_types = get_scalar_types()
_is_cpu_amx_available = cpu_has_amx_support()

_CPU_AMX_REQUIRED_MSG = (
    "SGLang's AutoRound CPU inference path currently requires the Intel AMX "
    "CPU backend. Generic x86, AMD CPU, and other non-AMX CPU backends are "
    "not supported by this SGLang backend."
)

_GPTQ_DEFAULTS = {
    "lm_head_quantized": False,
    "desc_act": False,
    "dynamic": {},
    "checkpoint_format": "",
    "true_sequential": False,
    "static_groups": False,
}


class AutoRoundConfig(QuantizationConfig):
    """Config class for AutoRound.

    CPU support is limited to 4-bit AWQ/GPTQ checkpoints on the
    Intel AMX backend. This is a limitation of SGLang's current CPU backend,
    not a general AutoRound limitation.

    Reference: https://arxiv.org/pdf/2309.05516
    """

    SUPPORTED_BITS = {2, 3, 4, 8}
    SUPPORTED_DTYPES = {"int", "mx_fp"}
    SUPPORTED_FORMATS = {
        "auto_round:auto_gptq",
        "auto_round:auto_awq",
        "auto_round:llm_compressor",
    }
    SUPPORTED_BACKENDS = {"auto", "gptq", "gptq:marlin", "awq", "awq:marlin", "marlin"}

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: Optional[Union[str, list[str]]] = None,
        extra_config: Optional[dict[str, Any]] = None,
        data_type: str = "int",
        backend: str = "auto",
        lm_head_quantized: bool = False,
        desc_act: bool = False,
        dynamic: Optional[dict[str, dict[str, Union[int, bool]]]] = None,
        checkpoint_format: str = "",
        true_sequential: bool = False,
        static_groups: bool = False,
        gptq_defaulted_config_keys: Optional[tuple[str, ...]] = None,
        mx_delegate: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported weight_bits: {weight_bits}, "
                f"currently only support  {self.SUPPORTED_BITS}"
            )
        if data_type not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data_type: {data_type},"
                f" currently only support  {self.SUPPORTED_DTYPES}"
            )
        if packing_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported packing_format: {packing_format}, "
                f"currently only support  {self.SUPPORTED_FORMATS}"
            )
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend},  "
                f"currently only support  {self.SUPPORTED_BACKENDS}"
            )

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.packing_format = packing_format
        self.block_name_to_quantize = (
            block_name_to_quantize.split(",")
            if isinstance(block_name_to_quantize, str)
            else block_name_to_quantize
        )
        self.extra_config = extra_config
        self.data_type = data_type
        self.backend = backend
        self.pack_factor = Fraction(32, weight_bits)
        self.lm_head_quantized = lm_head_quantized
        self.desc_act = desc_act
        self.dynamic = dynamic or {}
        self.checkpoint_format = checkpoint_format
        self.true_sequential = true_sequential
        self.static_groups = static_groups
        self.gptq_defaulted_config_keys = gptq_defaulted_config_keys or ()
        self._logged_gptq_default_assumptions = False
        self.mx_delegate = mx_delegate

    def __repr__(self) -> str:
        return (
            f"AutoRoundConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, sym={self.sym})"
        )

    @classmethod
    def get_name(cls):
        return "auto-round"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoRoundConfig":
        if config.get("config_groups"):
            return cls._from_compressed_tensors_config(config)

        data_type = cls.get_from_keys_or(config, ["data_type"], "int")
        if data_type == "mx_fp":
            return cls._from_native_mxfp_config(config)

        def has_any_key(keys: list[str]) -> bool:
            return any(key in config for key in keys)

        gptq_config_keys = {
            "lm_head_quantized": ["lm_head", "lm_head_quantized"],
            "desc_act": ["desc_act"],
            "dynamic": ["dynamic"],
            "checkpoint_format": ["checkpoint_format"],
            "true_sequential": ["true_sequential"],
            "static_groups": ["static_groups"],
        }
        gptq_defaulted_config_keys = tuple(
            name for name, keys in gptq_config_keys.items() if not has_any_key(keys)
        )

        return cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            sym=cls.get_from_keys(config, ["sym"]),
            packing_format=cls.get_from_keys_or(
                config,
                ["packing_format"],
                "auto_round:auto_gptq",
            ),
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"], None
            ),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(
                config, ["backend", "vllm_backend", "sglang_backend"], "auto"
            ),
            lm_head_quantized=cls.get_from_keys_or(
                config, ["lm_head", "lm_head_quantized"], False
            ),
            desc_act=cls.get_from_keys_or(config, ["desc_act"], False),
            dynamic=cls.get_from_keys_or(config, ["dynamic"], {}) or {},
            checkpoint_format=cls.get_from_keys_or(config, ["checkpoint_format"], ""),
            true_sequential=cls.get_from_keys_or(config, ["true_sequential"], False),
            static_groups=cls.get_from_keys_or(config, ["static_groups"], False),
            gptq_defaulted_config_keys=gptq_defaulted_config_keys,
        )

    @classmethod
    def _from_compressed_tensors_config(cls, config: dict[str, Any]):
        return _from_compressed_tensors_config(cls, config)

    @classmethod
    def _from_native_mxfp_config(cls, config: dict[str, Any]):
        return _from_native_mxfp_config(cls, config)

    def apply_weight_name_mapper(self, hf_to_sglang_mapper) -> None:
        if self.mx_delegate is not None:
            self.mx_delegate.apply_weight_name_mapper(hf_to_sglang_mapper)
        else:
            super().apply_weight_name_mapper(hf_to_sglang_mapper)

    def get_scaled_act_names(self) -> list[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    def get_layer_config(self, layer, layer_name: str):
        return get_layer_config(self, layer, layer_name)

    def check_quantized(self, weight_bits: int) -> bool:
        return weight_bits < 16

    def check_cpu_support(self, weight_bits: int) -> None:
        if weight_bits != 4:
            raise ValueError(
                "SGLang's AutoRound CPU inference path currently supports "
                "only 4-bit AWQ/GPTQ checkpoints because it uses the Intel "
                f"AMX INT4 backend, but got {weight_bits}-bit."
            )
        if not _is_cpu_amx_available:
            raise ValueError(_CPU_AMX_REQUIRED_MSG)

    def log_gptq_default_assumptions_once(self) -> None:
        if self._logged_gptq_default_assumptions or not self.gptq_defaulted_config_keys:
            return
        self._logged_gptq_default_assumptions = True
        default_summary = {
            key: _GPTQ_DEFAULTS[key] for key in self.gptq_defaulted_config_keys
        }
        logger.info(
            "AutoRound GPTQ config does not specify %s; using SGLang defaults %s.",
            ", ".join(self.gptq_defaulted_config_keys),
            default_summary,
        )

    def check_gptq_support(self) -> None:
        if self.desc_act:
            raise ValueError(
                "SGLang's AutoRound GPTQ loader supports desc_act=False only. "
                "AutoRound auto_gptq export does not use act-order/desc_act=True; "
                "if this checkpoint is a GPTQModel act-order checkpoint, use "
                "`--quantization gptq` or `--quantization gptq_marlin` instead."
            )

    def get_gptq_config_kwargs(
        self, weight_bits: int, group_size: int
    ) -> dict[str, Any]:
        self.log_gptq_default_assumptions_once()
        self.check_gptq_support()
        return {
            "weight_bits": weight_bits,
            "group_size": group_size,
            "lm_head_quantized": self.lm_head_quantized,
            "desc_act": self.desc_act,
            "dynamic": self.dynamic,
            "checkpoint_format": self.checkpoint_format,
            "true_sequential": self.true_sequential,
            "static_groups": self.static_groups,
        }

    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        return apply_awq_quant_layer(self, layer, prefix, backend)

    def apply_gptq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        return apply_gptq_quant_layer(self, layer, prefix, backend)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if self.mx_delegate is not None:
            return self.mx_delegate.get_quant_method(layer, prefix)
        if "gptq" in self.packing_format or "gptq" in self.backend:
            return self.apply_gptq_quant_layer(layer, prefix, self.backend)
        if "awq" in self.packing_format or "awq" in self.backend:
            return self.apply_awq_quant_layer(layer, prefix, self.backend)
