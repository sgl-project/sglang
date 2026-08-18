# SPDX-License-Identifier: Apache-2.0
"""GGUF quantization for diffusion transformers.

Weights stay packed in their GGML block format on device and are dequantized to
the activation dtype at use time, at the cost of one dequantization per linear
per denoise step.

What this buys is the size of the checkpoint, not the peak VRAM: layerwise
offload already bounds peak by its buffers and by VAE decode rather than by the
weight dtype. The 3.5x smaller file (17.5 vs 61.7 GiB for MiniMax-H3) shrinks the
download, the host memory offload pins, and the bytes streamed per layer.

Unlike the LLM-side GGUF path in ``sglang.srt``, the packed shape of every
tensor is already known here: ``GGUFConfig`` is built from the checkpoint header
before the model is constructed. ``create_weights`` can therefore register a
parameter with its exact packed byte shape, so no lazily-materialized parameter
or weight-loader special case is needed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.loader.gguf_weights import (
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGUFTensorMeta,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

logger = init_logger(__name__)

# GGML types the CUDA dequantize kernel implements. Kept explicit so an
# unsupported checkpoint fails at load with a clear message rather than
# producing garbage pixels.
SUPPORTED_GGML_TYPES = frozenset(
    {
        GGML_F32,
        GGML_F16,
        GGML_BF16,
        2,  # Q4_0
        3,  # Q4_1
        6,  # Q5_0
        7,  # Q5_1
        8,  # Q8_0
        10,  # Q2_K
        11,  # Q3_K
        12,  # Q4_K
        13,  # Q5_K
        14,  # Q6_K
        16,  # IQ2_XXS
        17,  # IQ2_XS
        18,  # IQ3_XXS
        19,  # IQ1_S
        20,  # IQ4_NL
        21,  # IQ3_S
        22,  # IQ2_S
        23,  # IQ4_XS
        29,  # IQ1_M
    }
)


class GGUFConfig(QuantizationConfig):
    """Config for a diffusion transformer whose weights come from a GGUF file.

    Args:
        gguf_file: Local path to the ``.gguf`` holding the transformer weights.
        tensor_meta: Layout of every tensor in ``gguf_file``, keyed by GGUF
            tensor name. Read from the header before model construction.
    """

    def __init__(self, gguf_file: str, tensor_meta: dict[str, GGUFTensorMeta]):
        super().__init__()
        self.gguf_file = gguf_file
        self.tensor_meta = tensor_meta

    def __repr__(self) -> str:
        return (
            f"GGUFConfig(gguf_file={self.gguf_file!r}, tensors={len(self.tensor_meta)})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "gguf"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The GGML kernels target Pascal and newer.
        return 61

    @staticmethod
    def get_config_filenames() -> list[str]:
        # A GGUF checkpoint carries its quantization in the file itself.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GGUFConfig:
        gguf_file = config.get("gguf_file")
        if not gguf_file:
            raise ValueError("GGUF quantization requires a `gguf_file` path.")
        tensor_meta = config.get("tensor_meta")
        if tensor_meta is None:
            from sglang.multimodal_gen.runtime.loader.gguf_weights import (
                read_gguf_tensor_meta,
            )

            tensor_meta = read_gguf_tensor_meta(gguf_file)
        return cls(gguf_file=gguf_file, tensor_meta=tensor_meta)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None

        meta = self.tensor_meta.get(f"{prefix}.weight")
        if meta is None:
            raise ValueError(
                f"Linear layer {prefix!r} has no weight in the GGUF checkpoint "
                f"{self.gguf_file!r}. The checkpoint does not match this model."
            )
        if meta.ggml_type not in SUPPORTED_GGML_TYPES:
            raise ValueError(
                f"GGUF tensor {prefix}.weight uses unsupported GGML type "
                f"{meta.ggml_type}."
            )
        if not meta.is_quantized:
            # Stored at full precision in the checkpoint (H3 keeps its patch
            # projections, timestep MLP and output heads unquantized); the
            # ordinary linear path already loads these correctly.
            return UnquantizedLinearMethod()
        return GGUFLinearMethod(meta=meta, prefix=prefix)


class GGUFLinearMethod(LinearMethodBase):
    """Linear method holding one packed GGML weight.

    One instance per layer: it captures that layer's tensor layout so the packed
    parameter can be registered with an exact shape.
    """

    def __init__(self, *, meta: GGUFTensorMeta, prefix: str):
        self.meta = meta
        self.prefix = prefix

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
        output_size_per_partition = sum(output_partition_sizes)
        logical_shape = self.meta.logical_shape
        if logical_shape != (output_size_per_partition, input_size_per_partition):
            raise ValueError(
                f"GGUF tensor {self.prefix}.weight has shape {logical_shape}, but "
                f"layer {self.prefix!r} expects "
                f"{(output_size_per_partition, input_size_per_partition)}. Tensor "
                "parallelism is not supported for GGUF diffusion checkpoints."
            )

        # Registered with the checkpoint's packed byte shape so the generic
        # weight loader's dtype cast is a no-op and its shape assertion holds.
        qweight = Parameter(
            torch.empty(self.meta.stored_shape, dtype=self.meta.stored_dtype),
            requires_grad=False,
        )
        # output_dim lets the standard loader slice rows; GGML blocks run along
        # the input dim, so a row is always a whole number of blocks.
        set_weight_attrs(qweight, {"output_dim": 0, "ignore_warning": True})
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        layer.ggml_type = self.meta.ggml_type
        layer.logical_weight_shape = logical_shape

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from sgl_kernel.quantization import ggml_dequantize

        out_features, in_features = layer.logical_weight_shape
        weight = ggml_dequantize(
            layer.qweight, layer.ggml_type, out_features, in_features, x.dtype
        )
        # F.linear handles the N-D activations a DiT produces ([batch, seq,
        # hidden]) without an explicit reshape.
        return F.linear(x, weight, bias)
