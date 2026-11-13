# SPDX-License-Identifier: Apache-2.0
"""AutoRound ``mx_fp`` W8A8 (``auto_round:llm_compressor``) checkpoints on Intel XPU.

Checkpoint layout per quantized linear: ``weight`` float8_e4m3fn [N, K] and
``weight_scale`` uint8 UE8M0 [N, K/32], which is compressed-tensors' MXFP8 layout;
the layers run through srt's ``CompressedTensorsW8A8MXFp8`` scheme (dynamic MXFP8
activations, ``xpu_mxfp8_scaled_mm_linear``), with diffusion weight parameters.
"""

from typing import Any

import torch
from compressed_tensors.quantization import QuantizationArgs

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs
from sglang.srt.layers.quantization.auto_round import (
    AutoRoundConfig as SRTAutoRoundConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8MXFp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    XPU_MXFP8_GROUP_SIZE as MXFP8_GROUP_SIZE,
)

# AutoRound mx_fp W8A8 in compressed-tensors terms: MXFP8 weights static, activations dynamic.
_MXFP8_WEIGHT_ARGS = QuantizationArgs(
    num_bits=8,
    type="float",
    strategy="group",
    group_size=MXFP8_GROUP_SIZE,
    dynamic=False,
)
_MXFP8_ACTIVATION_ARGS = QuantizationArgs(
    num_bits=8,
    type="float",
    strategy="group",
    group_size=MXFP8_GROUP_SIZE,
    dynamic=True,
)


def is_auto_round_mx_fp(config: dict[str, Any]) -> bool:
    """Any AutoRound mx_fp checkpoint: srt's AutoRound only knows int, so even the
    unsupported mx_fp variants are rejected here with a precise message."""
    return str(config.get("data_type", "")).startswith("mx_fp")


def _flatten_block_names(value: Any) -> tuple[str, ...]:
    """block_name_to_quantize as a ","-joined string, a list, or (multimodal
    exports) a list of lists."""
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(name.strip() for name in value.split(",") if name.strip())
    return tuple(name for item in value for name in _flatten_block_names(item))


class AutoRoundMXFP8Config(QuantizationConfig):
    """Serialized AutoRound MXFP8 W8A8 linears for diffusion models on XPU."""

    checkpoint_uses_native_qkv_layout = True

    def __init__(self, block_names: tuple[str, ...]) -> None:
        super().__init__()
        # Linears under these module prefixes are quantized; all others are bf16.
        self.block_names = block_names

    @classmethod
    def get_name(cls) -> str:
        return "auto-round"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return SRTAutoRoundConfig.get_config_filenames()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoRoundMXFP8Config":
        expected = {
            "data_type": "mx_fp",
            "bits": 8,
            "act_bits": 8,
            "act_data_type": "mx_fp",
            "group_size": MXFP8_GROUP_SIZE,
            "act_group_size": MXFP8_GROUP_SIZE,
            "sym": True,
            "act_dynamic": True,
            "packing_format": "auto_round:llm_compressor",
        }
        mismatched = {
            key: config.get(key)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatched:
            raise ValueError(
                "SGLang diffusion supports AutoRound mx_fp only as dynamic "
                f"W8A8 MXFP8 ({expected}); got {mismatched}."
            )
        if config.get("act_sym", True) is not True:
            raise ValueError(
                "AutoRound MXFP8 requires symmetric activations (act_sym)."
            )
        if config.get("extra_config"):
            raise ValueError(
                "AutoRound MXFP8 per-layer extra_config is not supported yet."
            )
        if not current_platform.is_xpu():
            raise ValueError("AutoRound MXFP8 checkpoints require Intel XPU.")

        block_names = _flatten_block_names(
            config.get("block_name_to_quantize") or config.get("to_quant_block_names")
        )
        if not block_names:
            raise ValueError(
                "AutoRound MXFP8 checkpoint has no block_name_to_quantize; "
                "cannot tell which linears hold MXFP8 weights."
            )
        return cls(block_names=block_names)

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        return input_size_per_partition % MXFP8_GROUP_SIZE == 0

    def _is_quantized_prefix(self, prefix: str) -> bool:
        return any(prefix.startswith(f"{name}.") for name in self.block_names)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if not isinstance(layer, LinearBase):
            return None
        if not self._is_quantized_prefix(prefix):
            return UnquantizedLinearMethod()
        # The llm_compressor packing is compressed-tensors' MXFP8 layout, so the
        # srt scheme loads and runs it (as GPTQ checkpoints reuse srt's GPTQ methods).
        layer.scheme = _DiffusionMXFP8Scheme(
            weight_quant=_MXFP8_WEIGHT_ARGS, input_quant=_MXFP8_ACTIVATION_ARGS
        )
        return CompressedTensorsLinearMethod(quantization_config=self)


class _DiffusionMXFP8Scheme(CompressedTensorsW8A8MXFp8):
    """srt's MXFP8 scheme with diffusion weight parameters: diffusion linears call
    load_*_weight without the tp_rank that srt's parameter classes require."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        del input_size, output_size
        if input_size_per_partition % self.group_size:
            raise ValueError(
                f"MXFP8 input partition must be divisible by {self.group_size}, "
                f"got {input_size_per_partition}"
            )
        output_size_per_partition = sum(output_partition_sizes)
        # [N, K] e4m3 and [N, K/32] UE8M0 bytes, kept as-is by the inherited
        # process_weights_after_loading; apply_weights reads them directly.
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        # A quantized-prefix linear without a scale in the checkpoint is BF16 there;
        # a default-filled scale would silently zero its output.
        set_weight_attrs(weight_scale, {"missing_param_init": "error"})
        layer.register_parameter("weight_scale", weight_scale)
