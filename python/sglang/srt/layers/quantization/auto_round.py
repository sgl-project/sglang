# SPDX-License-Identifier: Apache-2.0

import logging
import re
from fractions import Fraction
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.layers.quantization.utils import get_scalar_types

ScalarType, scalar_types = get_scalar_types()

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_npu

_is_npu = is_npu()
_is_cpu = is_cpu()
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
    SUPPORTED_DTYPES = {"int"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}
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

    def get_scaled_act_names(self) -> list[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    def get_layer_config(self, layer, layer_name: str):
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        def get_config(name: str, quantized: bool = True):
            if not self.extra_config:
                return (
                    self.weight_bits if quantized else 16,
                    self.group_size if quantized else -1,
                    self.sym if quantized else True,
                )

            # Exact match first
            if name in self.extra_config:
                cfg = self.extra_config[name]
                return (
                    cfg.get("bits", self.weight_bits if quantized else 16),
                    cfg.get("group_size", self.group_size if quantized else -1),
                    cfg.get("sym", self.sym if quantized else True),
                )

            REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
            for pattern, cfg in self.extra_config.items():
                if not isinstance(pattern, str) or not any(
                    c in REGEX_SPECIAL_CHARS for c in pattern
                ):
                    continue

                try:
                    if re.fullmatch(pattern, name):
                        return (
                            cfg.get("bits", self.weight_bits if quantized else 16),
                            cfg.get("group_size", self.group_size if quantized else -1),
                            cfg.get("sym", self.sym if quantized else True),
                        )
                except re.error:
                    # Invalid regex, ignore.
                    continue

            return (
                self.weight_bits if quantized else 16,
                self.group_size if quantized else -1,
                self.sym if quantized else True,
            )

        # 1. Exact match from config
        if self.extra_config and layer_name in self.extra_config:
            return get_config(layer_name)

        # 2. Determine whether layer should be quantized
        quantized = not isinstance(layer, ParallelLMHead)
        if self.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name) for name in self.block_name_to_quantize
            )

        # 3. Handle fused MoE
        if self.extra_config and "fusedmoe" in layer.__class__.__name__.lower():
            moe_configs = [
                get_config(name, quantized)
                for name in self.extra_config
                if name.startswith(layer_name)
            ]
            if moe_configs:
                if len(set(moe_configs)) == 1:
                    return moe_configs[0]
                raise ValueError(
                    f"Fused MoE layer '{layer_name}' requires "
                    f"consistent quant config for all sub-layers"
                )

        # 4. Handle fused QKV or other patterns
        if self.extra_config:
            for fusion_key, sub_keys in self.packed_modules_mapping.items():
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                    sub_names = [
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                    ]
                    sub_configs = [get_config(name, quantized) for name in sub_names]
                    if len(set(sub_configs)) == 1:
                        return sub_configs[0]
                    raise ValueError(
                        f"Fused module '{layer_name}' requires "
                        f"consistent quant config for {sub_names}"
                    )

        # 5. Fallback or try a regular expression match
        return get_config(layer_name, quantized)

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
        if (
            self._logged_gptq_default_assumptions
            or not self.gptq_defaulted_config_keys
        ):
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

    def get_gptq_config_kwargs(self, weight_bits: int, group_size: int) -> dict[str, Any]:
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
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None
        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if _is_cpu:
            self.check_cpu_support(weight_bits)
            from sglang.srt.layers.quantization.awq import (
                AWQCPUConfig,
                AWQLinearMethod,
                AWQMoEMethod,
            )

            quant_args = AWQCPUConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
            )
            if isinstance(layer, FusedMoE):
                layer.scheme = quant_args.get_moe_scheme(layer)
                return AWQMoEMethod(quant_args)
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                layer.scheme = quant_args.get_linear_scheme(layer)
                return AWQLinearMethod(quant_args)
            return None

        if backend == "auto" or "marlin" in backend:
            AWQ_TYPE_MAP = {
                4: scalar_types.uint4,
                8: scalar_types.uint8,
            }
            use_marlin = (weight_bits in AWQ_TYPE_MAP) and check_marlin_supported(
                AWQ_TYPE_MAP[weight_bits], group_size, not sym
            )

            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )
        else:
            use_marlin = False
        if use_marlin:
            from sglang.srt.layers.quantization.awq import (
                AWQLinearMethod,
                AWQMarlinConfig,
                AWQMoEMethod,
            )

            quant_args_marlin = AWQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
                lm_head_quantized=False,
                full_config={},
                modules_to_not_convert=[],
            )
        else:
            from sglang.srt.layers.quantization.awq import AWQConfig, AWQLinearMethod

            quant_args = AWQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                layer.scheme = quant_args_marlin.get_moe_scheme(layer)
                return AWQMoEMethod(quant_args_marlin)
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "awq",
                "bits": weight_bits,
                "group_size": group_size,
                "zero_point": not sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                layer.scheme = quant_args_marlin.get_linear_scheme(layer)
                return AWQLinearMethod(quant_args_marlin)
            else:
                layer.scheme = quant_args.get_linear_scheme(layer)
                return AWQLinearMethod(quant_args)
        return None

    def apply_gptq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.gptq import (
            GPTQAscendConfig,
            GPTQLinearMethod,
            GPTQMoEMethod,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        self.log_gptq_default_assumptions_once()
        if _is_npu:
            quant_args = GPTQAscendConfig(
                **self.get_gptq_config_kwargs(weight_bits, group_size),
            )
            quant_args.sym = sym

            if isinstance(layer, FusedMoE):
                layer.scheme = quant_args.get_moe_scheme(layer)
                return GPTQMoEMethod(quant_args)

            if isinstance(layer, (LinearBase, ParallelLMHead)):
                layer.scheme = quant_args.get_linear_scheme(layer)
                return GPTQLinearMethod(quant_args)

            return None

        if _is_cpu:
            self.check_cpu_support(weight_bits)
            from sglang.srt.layers.quantization.gptq import CPUGPTQConfig

            quant_args = CPUGPTQConfig(
                **self.get_gptq_config_kwargs(weight_bits, group_size),
            )
            quant_args.sym = sym

            if isinstance(layer, FusedMoE):
                layer.scheme = quant_args.get_moe_scheme(layer)
                return GPTQMoEMethod(quant_args)

            if isinstance(layer, (LinearBase, ParallelLMHead)):
                layer.scheme = quant_args.get_linear_scheme(layer)
                return GPTQLinearMethod(quant_args)

            return None

        if backend == "auto" or "marlin" in backend:
            GPTQ_TYPE_MAP = {
                (4, True): scalar_types.uint4b8,
                (8, True): scalar_types.uint8b128,
            }
            use_marlin = (weight_bits, sym) in GPTQ_TYPE_MAP and check_marlin_supported(
                GPTQ_TYPE_MAP[(weight_bits, sym)], group_size, has_zp=not sym
            )
            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )
        else:
            use_marlin = False
        if use_marlin:
            from sglang.srt.layers.quantization.gptq import (
                GPTQMarlinConfig,
                GPTQMarlinLinearMethod,
                GPTQMarlinMoEMethod,
            )

            quant_args_marlin = GPTQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                is_sym=sym,
                lm_head_quantized=self.lm_head_quantized,
                desc_act=self.desc_act,
                dynamic=self.dynamic,
                full_config={},
            )
        else:
            from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod

            quant_args = GPTQConfig(
                **self.get_gptq_config_kwargs(weight_bits, group_size),
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return GPTQMarlinMoEMethod(quant_args_marlin)
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "gptq",
                "bits": weight_bits,
                "group_size": group_size,
                "sym": sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return GPTQMarlinLinearMethod(quant_args_marlin)
            else:
                return GPTQLinearMethod(quant_args)

        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if "gptq" in self.packing_format or "gptq" in self.backend:
            return self.apply_gptq_quant_layer(layer, prefix, self.backend)
        if "awq" in self.packing_format or "awq" in self.backend:
            return self.apply_awq_quant_layer(layer, prefix, self.backend)
