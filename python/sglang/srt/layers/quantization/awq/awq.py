# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.marlin_utils import (
    check_marlin_supported,
    check_marlin_supports_layer,
    check_moe_marlin_supports_layer,
    verify_marlin_supported,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils.patch_torch import register_fake_if_exists

from .schemes import (
    AWQAscendLinearScheme,
    AWQAscendMoEScheme,
    AWQIntelAMXLinearScheme,
    AWQIntelAMXMoEScheme,
    AWQLinearScheme,
    AWQMarlinLinearScheme,
    AWQMoEScheme,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_npu = is_npu()

if not (_is_cuda or _is_hip or _is_xpu or _is_npu):
    warnings.warn(f"Only CUDA, HIP and XPU support AWQ currently.")

logger = logging.getLogger(__name__)


ScalarType, scalar_types = get_scalar_types()


def is_layer_skipped_awq(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16] if not _is_npu else [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        if _is_npu:
            raise NotImplementedError(
                'NPU hardware does not support "get_min_capability" feature.'
            )
        else:
            return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> AWQConfig:
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if _is_npu:
            if isinstance(layer, LinearBase):
                if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                    return UnquantizedLinearMethod()
                layer.scheme = self.get_linear_scheme(layer)
                return AWQLinearMethod(self)
            elif isinstance(layer, FusedMoE):
                layer.scheme = self.get_moe_scheme(layer)
                return AWQMoEMethod(self)
            return None

        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            layer.scheme = self.get_linear_scheme(layer)
            return AWQLinearMethod(self)
        return None

    def get_linear_scheme(self, layer: torch.nn.Module):
        assert isinstance(layer, LinearBase)
        # TODO: move platform-specific AWQ scheme selection into the platform
        # plugin factory once quantization hooks are available there.
        if _is_npu:
            return AWQAscendLinearScheme(self)
        return AWQLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        # This is currently only reached by the NPU path in get_quant_method.
        if _is_npu:
            return AWQAscendMoEScheme(self)
        raise NotImplementedError("AWQConfig only supports MoE scheme on NPU.")


class AWQCPUConfig(AWQConfig):
    """CPU Config class for AWQ, inherit from AWQConfig"""

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            layer.scheme = self.get_linear_scheme(layer)
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer)
            return AWQMoEMethod(self)
        return None

    def get_linear_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.linear import LinearBase

        assert isinstance(layer, LinearBase)
        return AWQIntelAMXLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        return AWQIntelAMXMoEScheme(self)


class AWQMarlinConfig(QuantizationConfig):
    """Config class for AWQ Marlin"""

    # num_bits -> type
    TYPE_MAP = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[list[str]],
        full_config: dict[str, Any],
    ) -> None:
        super().__init__()
        if _is_hip:
            warnings.warn(f"HIP does not support fused_marlin_moe currently.")
        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.zero_point = zero_point
        self.lm_head_quantized = lm_head_quantized
        self.weight_bits = weight_bits
        self.modules_to_not_convert = modules_to_not_convert or []
        self.full_config = full_config

        if self.weight_bits not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported num_bits = {self.weight_bits}. "
                f"Supported num_bits = {self.TYPE_MAP.keys()}"
            )

        self.quant_type = self.TYPE_MAP[self.weight_bits]

        verify_marlin_supported(
            self.quant_type, group_size=self.group_size, has_zp=self.zero_point
        )

    def __repr__(self) -> str:
        return (
            f"AWQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def get_name(cls) -> str:
        return "awq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AWQMarlinConfig:
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        can_convert = cls.is_awq_marlin_compatible(hf_quant_cfg)
        is_valid_user_quant = (
            user_quant is None or user_quant == "marlin" or user_quant == "awq_marlin"
        )

        if can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "awq":
            logger.info(
                "Detected that the model can run with awq_marlin"
                ", however you specified quantization=awq explicitly,"
                " so forcing awq. Use quantization=awq_marlin for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            # Check if the layer is supported by AWQMarlin.
            if not check_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    "Layer '%s' is not supported by AWQMarlin. Falling back to unoptimized AWQ kernels.",  # noqa: E501
                    prefix,
                )
                return AWQConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            layer.scheme = self.get_linear_scheme(layer)
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            layer.scheme = self.get_moe_scheme(layer)
            return AWQMoEMethod(self)
        return None

    def get_linear_scheme(self, layer: torch.nn.Module):
        return AWQMarlinLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        return AWQMoEScheme(self)

    @classmethod
    def is_awq_marlin_compatible(cls, quant_config: dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        zero_point = quant_config.get("zero_point")

        if not _is_cuda:
            return False

        if quant_method != "awq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or group_size is None or zero_point is None:
            return False

        if num_bits not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[num_bits], group_size=group_size, has_zp=zero_point
        )


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

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


class AWQMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: AWQMarlinConfig):
        self.quant_config = quant_config
        self.quant_type = scalar_types.uint4
        if self.quant_config.weight_bits != 4:
            raise ValueError("AWQMoEMethod only supports 4bit now.")

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
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

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return layer.scheme.apply_weights(layer, dispatch_output)


# Register fake implementations for torch.compile support
if _is_cuda:

    @register_fake_if_exists("sgl_kernel::awq_marlin_repack")
    def _(b_q_weight, size_k, size_n, num_bits):
        return b_q_weight.new_empty(
            (size_k // 16, size_n * (num_bits // 2)), dtype=b_q_weight.dtype
        )
