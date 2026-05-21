import logging
from typing import Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import is_hip, mxfp_supported

logger = logging.getLogger(__name__)
_is_hip = is_hip()

if _is_hip:
    try:
        import aiter
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import dynamic_mxfp4_quant
    except ImportError as e:
        logger.warning(f"aiter MXFP4 kernels not available: {e}")
        aiter = None
        shuffle_weight = None
        dynamic_mxfp4_quant = None
        gemm_a4w4 = None

# The gemm_a4w4 ASM kernel has degraded precision when the output
# dimension (N) is smaller than its minimum tile size.
# Layers with output_size falls below this threshold will stay unquantized
_MXFP4_MIN_OUTPUT_DIM = 256


class Mxfp4Config(QuantizationConfig):
    """
    MXFP4 quantization config for diffusion models.

    Supports online quantization from unquantized BF16/FP16 checkpoints.
    Note: MXFP4 requires ROCm and MI350+ (gfx95x).
    """

    def __init__(
        self,
        is_checkpoint_mxfp4_serialized: bool = False,
        ignored_layers: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 95  # gfx95x, Note: mxfp_supported() is a better check

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # No config file needed for online quantization

    @classmethod
    def from_config(cls, config: dict) -> "Mxfp4Config":
        """Create from model config (for pre-quantized checkpoints)."""
        is_serialized = config.get("quant_method") == "mxfp4"
        return cls(is_checkpoint_mxfp4_serialized=is_serialized)

    def get_quant_method(self, layer, prefix: str):
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                logger.debug(
                    f"MXFP4: Keeping layer {prefix} unquantized (in ignored_layers)"
                )
                return UnquantizedLinearMethod()
            # Skip layers whose output dims are too small, see ASM kernel comment above
            output_size = getattr(layer, "output_size", None)
            if output_size is not None and output_size < _MXFP4_MIN_OUTPUT_DIM:
                logger.info(
                    f"MXFP4: Keeping layer {prefix} unquantized "
                    f"(output_size={output_size} < {_MXFP4_MIN_OUTPUT_DIM})"
                )
                return UnquantizedLinearMethod()
            logger.debug(f"MXFP4: Replacing layer {prefix} with MXFP4 linear method")
            return Mxfp4LinearMethod(self)
        else:
            logger.debug(f"MXFP4: Skipping layer {prefix} (not a LinearBase)")
        return None


class Mxfp4LinearMethod(LinearMethodBase):
    """
    MXFP4 online quantization method for linear layers.

    Quantizes unquantized BF16/FP16 weights to MXFP4 format during
    process_weights_after_loading().
    """

    def __init__(self, quant_config: Mxfp4Config):
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
        """
        Creates BF16/FP16 parameters that will be
        quantized to MXFP4 in process_weights_after_loading().
        """
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
        )
        layer.register_parameter("weight", weight)

        # Placeholder scale (will be created during quantization)
        weight_scale = PerTensorScaleParameter(
            data=torch.empty(1, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """
        Quantize BF16/FP16 weights to MXFP4 after loading from checkpoint.

        Converts weights from unquantized format to:
        - Packed uint8 (2 FP4 values per byte)
        - E8M0 scales (one per 32-element block)
        """
        if not mxfp_supported():
            platform = "unknown"
            if _is_hip:
                try:
                    platform = torch.cuda.get_device_properties(0).gcnArchName
                except:
                    platform = "ROCm (unknown arch)"
            raise RuntimeError(
                f"MXFP4 quantization requires ROCm and MI350+ (gfx95x). "
                f"Current platform: {platform}."
            )

        # Check if weights are already quantized
        if layer.weight.dtype not in [torch.bfloat16, torch.float16]:
            # Already quantized or unexpected dtype
            logger.info("Weights are quantized or unexpected dtype")
            return

        if any(fn is None for fn in (dynamic_mxfp4_quant, shuffle_weight, gemm_a4w4)):
            raise RuntimeError(
                "aiter MXFP4 kernels not available. "
                "Install aiter with MXFP4 support."
            )

        weight_data = layer.weight.data
        was_on_cpu = weight_data.device.type == "cpu"
        if was_on_cpu:
            weight_data = weight_data.cuda()

        w_quant, mx_scales = dynamic_mxfp4_quant(weight_data, shuffle=True)

        w_quant_shuffled = shuffle_weight(w_quant)

        if was_on_cpu:
            w_quant_shuffled = w_quant_shuffled.cpu()
            mx_scales = mx_scales.cpu()

        layer.weight = Parameter(w_quant_shuffled, requires_grad=False)
        layer.weight_scale = Parameter(mx_scales, requires_grad=False)

        logger.debug(
            f"MXFP4: Quantized layer weights - weight {layer.weight.shape} {layer.weight.dtype}, "
            f"scale {layer.weight_scale.shape}"
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if not mxfp_supported():
            raise RuntimeError(
                "MXFP4 inference requires ROCm and MI350+ (gfx95x). "
                "Current platform not supported."
            )

        # Handle 3D input tensors [batch, seq, hidden]
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])

        x_fp4, x_scale = dynamic_mxfp4_quant(x, shuffle=True)

        y = gemm_a4w4(x_fp4, layer.weight, x_scale, layer.weight_scale)

        if bias is not None:
            y = y + bias

        return y.view(*original_shape[:-1], layer.weight.shape[0])
