# SPDX-License-Identifier: Apache-2.0

import logging
import re
from fractions import Fraction
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

from sglang.srt.layers.quantization.utils import get_scalar_types

ScalarType, scalar_types = get_scalar_types()

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_npu, set_weight_attrs

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

_MXFP_BLOCK_SIZE = 32


class AutoRoundMxfp4LinearWNA16Method:
    """Load serialized MXFP4 dense weights and run them with A16 activations."""

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
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // _MXFP_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

        dtype = getattr(layer, "orig_dtype", torch.bfloat16)
        weight = MXFP4QuantizeUtil.dequantize(
            layer.weight_packed.data,
            dtype=dtype,
            scale=layer.weight_scale.data,
            block_sizes=[_MXFP_BLOCK_SIZE],
        ).contiguous()

        for name in ("weight", "weight_packed", "weight_scale"):
            if name in layer._parameters:
                del layer._parameters[name]
            elif hasattr(layer, name):
                delattr(layer, name)
        layer.register_parameter("weight", Parameter(weight, requires_grad=False))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


class AutoRoundMxfp8LinearWNA16Method:
    """Fallback for MXFP8 checkpoints when no W8A8 activation kernel is available."""

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
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        scale = BlockQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // _MXFP_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        scale.format_ue8m0 = True
        layer.register_parameter("weight_scale_inv", scale)
        layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.mxfp8_block_convert import (
            dequant_mxfp8_2d_to_bf16,
        )

        weight = dequant_mxfp8_2d_to_bf16(
            layer.weight.data, layer.weight_scale_inv.data
        ).to(getattr(layer, "orig_dtype", torch.bfloat16))
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        del layer.weight_scale_inv
        layer.input_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


class AutoRoundMxfp4MoEWNA16Method(FusedMoEMethodBase):
    """Load serialized MXFP4 experts and run MoE with A16 activations."""

    def __init__(self):
        self.runner = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        del num_experts
        layer.orig_dtype = params_dtype
        self.with_bias = with_bias

        w13_weight = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.full(
                (
                    layer.num_local_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // _MXFP_BLOCK_SIZE,
                ),
                fill_value=127,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "group"

        w2_weight = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.full(
                (
                    layer.num_local_experts,
                    hidden_size,
                    intermediate_size_per_partition // _MXFP_BLOCK_SIZE,
                ),
                fill_value=127,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "group"

        if with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.zeros(
                    layer.num_local_experts,
                    2 * intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

            w2_weight_bias = torch.nn.Parameter(
                torch.zeros(layer.num_local_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

        dtype = getattr(layer, "orig_dtype", torch.bfloat16)
        w13_weight = MXFP4QuantizeUtil.dequantize(
            layer.w13_weight.data,
            dtype=dtype,
            scale=layer.w13_weight_scale.data,
            block_sizes=[_MXFP_BLOCK_SIZE],
        ).contiguous()
        w2_weight = MXFP4QuantizeUtil.dequantize(
            layer.w2_weight.data,
            dtype=dtype,
            scale=layer.w2_weight_scale.data,
            block_sizes=[_MXFP_BLOCK_SIZE],
        ).contiguous()

        del layer.w13_weight
        del layer.w13_weight_scale
        del layer.w2_weight
        del layer.w2_weight_scale
        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):
        del layer
        from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend

        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def get_triton_quant_info(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
        )

    def apply(self, layer: torch.nn.Module, dispatch_output):
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if TopKOutputChecker.format_is_bypassed(topk_output):
            dispatch_output = dispatch_output._replace(
                topk_output=topk_output.to_standard(layer.layer_id)
            )
        return self.runner.run(dispatch_output, self.get_triton_quant_info(layer))


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
    ) -> None:
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported weight_bits: {weight_bits}, "
                f"currently only support  {self.SUPPORTED_BITS}"
            )
        is_mxfp = "mx_fp" in data_type
        if data_type not in self.SUPPORTED_DTYPES and not is_mxfp:
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
        self._logged_mxfp8_w8a8 = False
        self._logged_mxfp8_wna16_fallback = False
        self._logged_mxfp4_moe = False
        self._logged_mxfp4_moe_wna16_fallback = False
        self._logged_mxfp4_dense_fallback = False
        self._mxfp8_quant_config = None
        mxfp_layer_bits = {self.weight_bits}
        for layer_cfg in (self.extra_config or {}).values():
            if isinstance(layer_cfg, dict):
                layer_dtype = str(layer_cfg.get("data_type", self.data_type))
                if "mx_fp" in layer_dtype:
                    mxfp_layer_bits.add(int(layer_cfg.get("bits", self.weight_bits)))
        self.use_mxfp8 = 8 in mxfp_layer_bits
        self.weight_block_size = [1, _MXFP_BLOCK_SIZE] if self.use_mxfp8 else None
        self.activation_scheme = "dynamic"

        if self.is_mxfp:
            self._validate_mxfp_metadata()

    def __repr__(self) -> str:
        return (
            f"AutoRoundConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, sym={self.sym})"
        )

    @classmethod
    def get_name(cls):
        return "auto-round"

    @property
    def is_mxfp(self) -> bool:
        return "mx_fp" in self.data_type

    @classmethod
    def is_mxfp_config(cls, config: dict[str, Any]) -> bool:
        return (
            (
                config.get("quant_method") == "auto-round"
                or config.get("_auto_round_quant_method") == "auto-round"
            )
            and "mx_fp" in str(config.get("data_type", ""))
        )

    @classmethod
    def get_mxfp_quantization_method(cls, config: dict[str, Any]) -> Optional[str]:
        if not cls.is_mxfp_config(config):
            return None
        bits = int(config.get("bits", 0))
        if bits not in (4, 8):
            raise ValueError(
                "SGLang supports AutoRound MXFP checkpoints with bits=4 or bits=8, "
                f"but got bits={bits}."
            )
        return "auto-round"

    @classmethod
    def to_native_mxfp_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        quant_method = cls.get_mxfp_quantization_method(config)
        if quant_method is None:
            return config

        cls._validate_mxfp_config_dict(config)
        normalized = dict(config)
        normalized["_auto_round_quant_method"] = "auto-round"
        normalized["_auto_round_mxfp"] = True
        normalized["_auto_round_mxfp_mixed"] = cls.is_mixed_mxfp_config(config)
        normalized["quant_method"] = "auto-round"
        logger.info(
            "Detected AutoRound MXFP checkpoint; keeping quant_method=auto-round "
            "for per-layer MXFP dispatch."
        )
        return normalized

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: Optional[str]
    ) -> Optional[str]:
        return cls.get_mxfp_quantization_method(hf_quant_cfg)

    @classmethod
    def is_mixed_mxfp_config(cls, config: dict[str, Any]) -> bool:
        if not cls.is_mxfp_config(config):
            return False

        global_bits = int(config.get("bits", 0))
        global_dtype = str(config.get("data_type", ""))
        mxfp_bits = {global_bits} if "mx_fp" in global_dtype else set()
        for layer_cfg in (config.get("extra_config") or {}).values():
            if not isinstance(layer_cfg, dict):
                continue
            layer_dtype = str(layer_cfg.get("data_type", global_dtype))
            if "mx_fp" not in layer_dtype:
                continue
            layer_bits = int(layer_cfg.get("bits", global_bits))
            if layer_bits in (4, 8):
                mxfp_bits.add(layer_bits)
        return len(mxfp_bits) > 1

    def _validate_mxfp_metadata(self) -> None:
        self._validate_mxfp_config_dict(
            {
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "sym": self.sym,
                "packing_format": self.packing_format,
            }
        )

    @classmethod
    def _validate_mxfp_config_dict(cls, config: dict[str, Any]) -> None:
        bits = int(config.get("bits", 0))
        group_size = int(config.get("group_size", 0))
        sym = bool(config.get("sym", False))
        packing_format = config.get("packing_format")

        if bits not in (4, 8):
            raise ValueError(
                "SGLang supports AutoRound MXFP checkpoints with 4-bit or 8-bit "
                f"weights, but got {bits}-bit."
            )
        if group_size != 32:
            raise ValueError(
                "AutoRound MXFP checkpoints require group_size=32, "
                f"but got group_size={group_size}."
            )
        if not sym:
            raise ValueError("AutoRound MXFP checkpoints must be symmetric.")
        if packing_format != "auto_round:llm_compressor":
            raise ValueError(
                "AutoRound MXFP checkpoints require "
                "packing_format='auto_round:llm_compressor', "
                f"but got {packing_format!r}."
            )

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

        def cfg_tuple(cfg: dict[str, Any], quantized: bool):
            bits = int(cfg.get("bits", self.weight_bits if quantized else 16))
            layer_quantized = quantized and self.check_quantized(bits)
            return (
                bits,
                cfg.get("group_size", self.group_size if layer_quantized else -1),
                cfg.get("sym", self.sym if layer_quantized else True),
            )

        def get_config(name: str, quantized: bool = True):
            if not self.extra_config:
                return (
                    self.weight_bits if quantized else 16,
                    self.group_size if quantized else -1,
                    self.sym if quantized else True,
                )

            # Exact match first
            if name in self.extra_config:
                return cfg_tuple(self.extra_config[name], quantized)

            REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
            for pattern, cfg in self.extra_config.items():
                if not isinstance(pattern, str) or not any(
                    c in REGEX_SPECIAL_CHARS for c in pattern
                ):
                    continue

                try:
                    if re.fullmatch(pattern, name):
                        return cfg_tuple(cfg, quantized)
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
            packed_modules_mapping = self.packed_modules_mapping or {
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                "gate_up_proj": ["gate_proj", "up_proj"],
            }
            for fusion_key, sub_keys in packed_modules_mapping.items():
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

    @staticmethod
    def _mxfp8_w8a8_dense_supported() -> bool:
        try:
            from sglang.srt.layers.quantization.fp8_utils import (
                resolve_mxfp8_dense_gemm_backend,
            )

            return not resolve_mxfp8_dense_gemm_backend().is_unsupported()
        except Exception as err:
            logger.warning(
                "MXFP8 W8A8 dense backend detection failed; falling back to "
                "A16 activations where possible. Error: %s",
                err,
            )
            return False

    def _get_mxfp8_quant_config(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        if self._mxfp8_quant_config is None:
            config = {
                "quant_method": "mxfp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [1, _MXFP_BLOCK_SIZE],
                "scale_fmt": "ue8m0",
                "packed_modules_mapping": self.packed_modules_mapping,
            }
            self._mxfp8_quant_config = Fp8Config.from_config(config)
        return self._mxfp8_quant_config

    def apply_mxfp_quant_layer(self, layer, prefix: str):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None

        if group_size != _MXFP_BLOCK_SIZE or not sym:
            raise ValueError(
                "AutoRound MXFP layers require group_size=32 and sym=True, "
                f"but layer {prefix!r} has bits={weight_bits}, "
                f"group_size={group_size}, sym={sym}."
            )

        if weight_bits == 8:
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                if self._mxfp8_w8a8_dense_supported():
                    if not self._logged_mxfp8_w8a8:
                        self._logged_mxfp8_w8a8 = True
                        logger.info(
                            "Using AutoRound MXFP8 W8A8 for dense linear layers "
                            "(dynamic MXFP8 activation quantization enabled)."
                        )
                    return self._get_mxfp8_quant_config().get_quant_method(layer, prefix)
                if not self._logged_mxfp8_wna16_fallback:
                    self._logged_mxfp8_wna16_fallback = True
                    logger.warning(
                        "No MXFP8 W8A8 dense activation kernel is available on "
                        "this device; falling back to A16 dense execution by "
                        "dequantizing MXFP8 weights after load."
                    )
                return AutoRoundMxfp8LinearWNA16Method()

            if isinstance(layer, FusedMoE):
                return self._get_mxfp8_quant_config().get_quant_method(layer, prefix)
            return None

        if weight_bits == 4:
            if isinstance(layer, FusedMoE):
                from sglang.srt.layers.moe.utils import get_moe_runner_backend
                from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
                from sglang.srt.utils import (
                    is_sm90_supported,
                    is_sm100_supported,
                    is_sm120_supported,
                )

                moe_backend = get_moe_runner_backend()
                moe_backend_name = getattr(moe_backend, "value", str(moe_backend))
                is_sm90 = is_sm90_supported()
                is_sm100 = is_sm100_supported()
                is_sm120 = is_sm120_supported()
                device_arch = (
                    "SM90"
                    if is_sm90
                    else "SM100"
                    if is_sm100
                    else "SM120"
                    if is_sm120
                    else "unsupported CUDA architecture"
                )
                native_w4a8_supported = False
                fallback_reason = (
                    f"backend={moe_backend_name} does not provide an AutoRound "
                    "MXFP4 W4A8 MoE path"
                )

                if moe_backend.is_flashinfer_mxfp4():
                    if is_sm100 or is_sm120:
                        native_w4a8_supported = True
                    elif is_sm90:
                        fallback_reason = (
                            "FlashInfer MXFP4 MoE on SM90 is a W4A16/weight-only "
                            "path, not the AutoRound MXFP4 W4A8 target"
                        )
                    else:
                        fallback_reason = (
                            "FlashInfer MXFP4 W4A8 MoE requires SM100 or SM120"
                        )
                elif moe_backend.is_deep_gemm():
                    if is_sm100:
                        native_w4a8_supported = True
                    elif is_sm90:
                        fallback_reason = (
                            "DeepGEMM MXFP4 W4A8 MoE is not supported on SM90"
                        )
                    else:
                        fallback_reason = (
                            "DeepGEMM MXFP4 W4A8 MoE requires SM100"
                        )
                elif moe_backend.is_marlin():
                    fallback_reason = (
                        "Marlin MXFP4 MoE is a W4A16/weight-only path, not the "
                        "AutoRound MXFP4 W4A8 target"
                    )

                if native_w4a8_supported:
                    if not self._logged_mxfp4_moe:
                        self._logged_mxfp4_moe = True
                        logger.info(
                            "Using AutoRound MXFP4 W4A8 MoE for layer %s "
                            "with backend=%s on %s.",
                            prefix,
                            moe_backend_name,
                            device_arch,
                        )
                    return Mxfp4MoEMethod(prefix=prefix)

                if not self._logged_mxfp4_moe_wna16_fallback:
                    self._logged_mxfp4_moe_wna16_fallback = True
                    logger.warning(
                        "AutoRound MXFP4 MoE layer %s cannot use native W4A8 "
                        "execution with backend=%s on %s: %s. Falling back to "
                        "WnA16 MoE by dequantizing MXFP4 expert weights after "
                        "load; MXFP4 weight-only MoE kernels are not used for "
                        "this W4A8 target.",
                        prefix,
                        moe_backend_name,
                        device_arch,
                        fallback_reason,
                    )
                return AutoRoundMxfp4MoEWNA16Method()

            if isinstance(layer, (LinearBase, ParallelLMHead)):
                if not self._logged_mxfp4_dense_fallback:
                    self._logged_mxfp4_dense_fallback = True
                    logger.warning(
                        "AutoRound MXFP4 dense linear layer %s has no CUDA dense "
                        "MXFP4 activation kernel in SGLang; falling back to A16 "
                        "dense execution by dequantizing MXFP4 weights after load.",
                        prefix,
                    )
                return AutoRoundMxfp4LinearWNA16Method()
            return None

        raise ValueError(
            "SGLang supports AutoRound MXFP layers with bits=4 or bits=8, "
            f"but layer {prefix!r} has bits={weight_bits}."
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if self.is_mxfp:
            return self.apply_mxfp_quant_layer(layer, prefix)
        if "gptq" in self.packing_format or "gptq" in self.backend:
            return self.apply_gptq_quant_layer(layer, prefix, self.backend)
        if "awq" in self.packing_format or "awq" in self.backend:
            return self.apply_awq_quant_layer(layer, prefix, self.backend)
