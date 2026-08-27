"""Online MXFP8 quantization for Diffusion models on Ascend NPU.

Provides ``MXFP8Config`` (registered as ``"mxfp8"``) and
``NPUMXFP8DiffusionLinearMethod`` which quantise FP16/BF16 weights to MXFP8
at load time and use ``npu_dynamic_mx_quant`` + ``npu_quant_matmul`` for
inference, mirroring the LLM-side ``NPUMXFP8LinearMethod``.
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

MXFP8_BLOCK_SIZE = 32


class MXFP8Config(QuantizationConfig):
    """Config for online MXFP8 quantization on Ascend NPU (Diffusion)."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> str:
        return "mxfp8"

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
    def from_config(cls, config: Dict[str, Any]) -> "MXFP8Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return NPUMXFP8DiffusionLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class NPUMXFP8DiffusionLinearMethod(LinearMethodBase):
    """Ascend NPU MXFP8 linear method for Diffusion models.

    Online mode: loads FP16/BF16 weights → quantises to MXFP8 at load time.
    Inference: dynamic MXFP8 activation quant + MXFP8 matmul (block_size=32).
    """

    def __init__(self, quant_config: MXFP8Config):
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

        # Move weight to NPU if needed. We intentionally use a conditional
        # move rather than an assert because `dit_cpu_offload` defaults to
        # True in ServerArgs, which causes fsdp_load to move every parameter
        # back to CPU after loading (even when the target device is NPU).
        # npu_dynamic_mx_quant requires an NPU tensor, so we must transfer
        # here. The quantized fp8 weights produced below will remain on NPU
        # for inference; if the model still needs to be offloaded after
        # quantization (e.g. very large model on a small NPU), a higher-level
        # offload pass can move them back afterwards.
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Online MXFP8 quantisation of weights (block_size=32)
        qw, w_scale = torch_npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=torch_npu.float8_e4m3fn
        )
        layer.weight = Parameter(qw, requires_grad=False)
        layer.weight_scale_inv = Parameter(w_scale, requires_grad=False)

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

        # Flatten to 2D [tokens, hidden] so npu_dynamic_mx_quant returns 3D scale
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch_npu.float8_e4m3fn
        )

        # MXFP8 matmul
        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight.transpose(0, 1),
            layer.weight_scale_inv.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        output = output.reshape(output_shape)

        return output
