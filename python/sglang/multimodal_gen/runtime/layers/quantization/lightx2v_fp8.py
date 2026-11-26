# SPDX-License-Identifier: Apache-2.0
"""Load LightX2V Qwen-Image rowwise FP8 checkpoints without repacking."""

import re
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
    RowwiseFp8Config,
)
from sglang.multimodal_gen.runtime.models.parameter import ChannelQuantScaleParameter


class LightX2VFp8LinearMethod(Fp8LinearMethod):
    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        weight_loader = extra_weight_attrs.get("weight_loader")

        def load_weight(
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            if isinstance(param, ChannelQuantScaleParameter):
                if loaded_weight.ndim != 2 or loaded_weight.shape[1] != 1:
                    raise ValueError(
                        "LightX2V weight_scale must have shape [out_features, 1]"
                    )
                # Normalize before the existing loader applies TP/fused sharding.
                loaded_weight = loaded_weight.squeeze(-1).float()
            return weight_loader(param, loaded_weight, *args, **kwargs)

        if weight_loader is not None:
            extra_weight_attrs["weight_loader"] = load_weight
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )


class LightX2VFp8Config(RowwiseFp8Config):
    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Checkpoint modulation linears have a Sequential index; runtime ones do not.
        if prefix.endswith((".img_mod", ".txt_mod")):
            prefix += ".1"
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, Fp8LinearMethod):
            return LightX2VFp8LinearMethod(self)
        return method


_QWEN_IMAGE_LINEAR_PATTERN = re.compile(
    r"transformer_blocks\.\d+\.(?:"
    r"attn\.(?:to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|to_out\.0|to_add_out)"
    r"|(?:img|txt)_mlp\.net\.(?:0\.proj|2)|(?:img|txt)_mod\.1)"
)


def build_lightx2v_fp8_config(
    safetensors_files: list[str],
) -> LightX2VFp8Config | None:
    """Recognize undeclared LightX2V Qwen-Image FP8 tensor layouts.

    Explicit quantization declarations take precedence in the caller. Return
    None for unrelated layouts; reject malformed scales in a recognized layout.
    """
    tensor_metadata: dict[str, dict[str, Any]] = {}
    for path in safetensors_files:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key in tensor_metadata:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                tensor = checkpoint.get_slice(key)
                tensor_metadata[key] = {
                    "shape": tensor.get_shape(),
                    "dtype": tensor.get_dtype(),
                }
    # Architecture marker plus layer names keep this fallback model-specific.
    if tensor_metadata.get("img_in.weight", {}).get("shape") != [3072, 64]:
        return None
    fp8_layers = {
        key.removesuffix(".weight")
        for key, metadata in tensor_metadata.items()
        if key.endswith(".weight") and metadata["dtype"] == "F8_E4M3"
    }
    if not fp8_layers or not all(
        _QWEN_IMAGE_LINEAR_PATTERN.fullmatch(prefix) for prefix in fp8_layers
    ):
        return None
    expected_scales = {f"{prefix}.weight_scale" for prefix in fp8_layers}
    actual_scales = {key for key in tensor_metadata if key.endswith(".weight_scale")}
    if actual_scales != expected_scales or any(
        key.endswith(".input_scale") for key in tensor_metadata
    ):
        raise ValueError("Invalid LightX2V FP8 weight/input scale keys")
    for prefix in sorted(fp8_layers):
        weight = tensor_metadata[f"{prefix}.weight"]
        scale = tensor_metadata[f"{prefix}.weight_scale"]
        if (
            len(weight["shape"]) != 2
            or scale["shape"] != [weight["shape"][0], 1]
            or scale["dtype"] != "BF16"
        ):
            raise ValueError(f"Invalid LightX2V FP8 scale layout for {prefix}")
    for key, metadata in tensor_metadata.items():
        if metadata["dtype"] != "BF16" and not (
            key.endswith(".weight") and key.removesuffix(".weight") in fp8_layers
        ):
            raise ValueError(f"Unsupported LightX2V tensor dtype: {key}")
    # Reuse the internal rowwise runtime contract after normalizing the scales.
    return LightX2VFp8Config(format_version="1.1", fp8_layers=frozenset(fp8_layers))
