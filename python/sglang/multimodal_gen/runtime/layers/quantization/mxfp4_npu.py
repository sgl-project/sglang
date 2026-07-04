"""Online MXFP4 quantization for Diffusion models on Ascend NPU.

Provides ``NPUMXFP4Config`` (registered as ``"mxfp4_npu"``) and
``NPUMXFP4DiffusionLinearMethod`` which quantises FP16/BF16 weights to MXFP4
at load time using dual-level MX quantization and uses
``npu_dynamic_dual_level_mx_quant`` + ``npu_dual_level_quant_matmul`` for
inference.

The ``"mxfp4_npu"`` key is distinct from upstream's ROCm ``"mxfp4"``
(``Mxfp4Config`` in ``mxfp4.py``) which targets AMD MI350+ via aiter kernels.

NOTE: Online weight quantization via ``npu_dynamic_dual_level_mx_quant`` is
experimental. MindIE-SD only uses an offline (pre-quantized) path for MXFP4
weights. The online path quantizes FP16/BF16 weights at load time, which may
produce different numerical results than the offline calibrated path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.platforms import current_platform

_is_npu = current_platform.is_npu()

if _is_npu:
    import torch_npu

from sglang.multimodal_gen.runtime.layers.linear import LinearBase, LinearMethodBase
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class NPUMXFP4Config(QuantizationConfig):
    """Config for online MXFP4 quantization on Ascend NPU (Diffusion)."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4_npu"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # NPU, not CUDA

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> NPUMXFP4Config:
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return NPUMXFP4DiffusionLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class NPUMXFP4DiffusionLinearMethod(LinearMethodBase):
    """Ascend NPU MXFP4 linear method for Diffusion models (dual-level).

    Online mode: loads FP16/BF16 weights → quantises to MXFP4 at load time
    via ``npu_dynamic_dual_level_mx_quant``.
    Inference: dynamic dual-level MXFP4 activation quant + dual-level matmul.

    Reference: MindIE-SD ``W4A4MXFP4DualQuantLinear`` (offline path only).
    """

    def __init__(self, quant_config: NPUMXFP4Config):
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
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise later in process_weights_after_loading
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)

        # Move weight to NPU if needed. dit_cpu_offload defaults to True in
        # ServerArgs, which causes fsdp_load to move parameters back to CPU
        # after loading. npu_dynamic_dual_level_mx_quant requires an NPU tensor.
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Online dual-level MXFP4 weight quantisation.
        # NOTE: This is experimental — MindIE-SD only has an offline path for
        # MXFP4 weights. We assume npu_dynamic_dual_level_mx_quant can also
        # quantise weights (not just activations).
        # Returns: (qw, w_dual_scale, w_scale)
        #   qw          — quantized weight in float4_e2m1fn_x2 (2 FP4 packed/byte)
        #   w_dual_scale — L0-level scale (goes to pos 3 in npu_dual_level_quant_matmul)
        #   w_scale      — L1-level scale (goes to pos 5 in npu_dual_level_quant_matmul)
        qw, w_dual_scale, w_scale = torch_npu.npu_dynamic_dual_level_mx_quant(
            weight_fp, smooth_scale=None
        )

        # npu_dual_level_quant_matmul requires x2 (weight) in FRACTAL_NZ format.
        # Reference: MindIE-SD W4A4MXFP4DualQuantLinear._init_dynamic_quant_param
        qw = torch_npu.npu_format_cast(
            qw.view(torch.int8), 29, customize_dtype=torch.int8
        )

        # x2Level0Scale must be [in/level0_block_size, out] — transpose from
        # the [out, in/level0_block_size] shape returned by the quant op.
        # Reference: MindIE-SD layer.py:409
        w_dual_scale = w_dual_scale.squeeze(-1).transpose(0, 1).contiguous()

        layer.weight = Parameter(qw, requires_grad=False)
        layer.weight_dual_scale = Parameter(w_dual_scale, requires_grad=False)
        layer.weight_scale = Parameter(w_scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for the quantization operators
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic dual-level MXFP4 activation quantisation
        qx, act_l0_scale, act_l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(
            x_2d, smooth_scale=None
        )

        # Dual-level MXFP4 matmul
        # Arg order: act_quant, weight_quant, act_l0_scale, weight_dual_scale,
        #            act_l1_scale, weight_scale, bias=, output_dtype=
        # NOTE: weight is NOT transposed (unlike MXFP8's npu_quant_matmul).
        output = torch_npu.npu_dual_level_quant_matmul(
            qx,
            layer.weight,
            act_l0_scale,
            layer.weight_dual_scale,
            act_l1_scale,
            layer.weight_scale,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        output = output.reshape(output_shape)

        return output
