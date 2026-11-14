# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py

import logging

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = logging.getLogger(__name__)


class BaseKVCacheMethod(QuantizeMethodBase):
    """
    Quant method that adds `k_scale` and `v_scale` attributes to the
    Attention layer to support loading those scaling factors from checkpoints.
    The k/v_scale will be used to:
        - quantize k/v_cache entries before saving them to the cache
        - dequantize k/v_cache entries before fetching them from the cache

    :param quant_config: the appropriate QuantizationConfig
    """

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """
        Create "weight" (aka k_scale and v_scale) for an attention layer.
        """
        # Initialize the KV cache scales to -1.0, which is an invalid value.
        # If the k/v_scale appears in the checkpoint, it will be
        # overwritten when loading weights.
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        layer.k_scale._skip_weight_check = True
        layer.v_scale._skip_weight_check = True

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer) -> None:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            # We prefer to use separate k_scale and v_scale if present
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale <= 0.0 and layer.v_scale <= 0.0:
            # If no scales were loaded (both scales are invalid non-positive
            # values), use the default value of 1.0
            k_scale = 1.0
            v_scale = 1.0
        else:
            # If we find a single kv_scale in the checkpoint, we remap
            # kv_scale to k_scale during weight loading, and duplicate
            # k_scale to v_scale here
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        # These are used in the final Attention.forward()
        layer.k_scale.copy_(k_scale)
        layer.v_scale.copy_(v_scale)
        layer.k_scale_float = k_scale
        layer.v_scale_float = v_scale


class SharedKVCacheScaleMethod(BaseKVCacheMethod):
    """KV-cache method of an attention layer whose KV cache later layers read.

    KV-shared layers (Gemma 3n, Gemma 4) write no KV cache: they read the cache
    of an earlier layer, the owner, which holds K/V divided by the owner's
    k_scale/v_scale. The owner and its readers share this method. Processing
    the owner gives the readers its processed scales, and processing a reader
    does nothing, so the layers may be processed in any order.
    """

    def __init__(self, method: BaseKVCacheMethod, owner: torch.nn.Module):
        super().__init__(method.quant_config)
        self.method = method
        self.owner = owner
        self.readers: list = []

    def process_weights_after_loading(self, layer) -> None:
        if layer is not self.owner:
            return
        self.method.process_weights_after_loading(layer)
        for reader in self.readers:
            reader.k_scale.copy_(layer.k_scale)
            reader.v_scale.copy_(layer.v_scale)
            reader.k_scale_float = layer.k_scale_float
            reader.v_scale_float = layer.v_scale_float


def share_kv_cache_scales(owner: torch.nn.Module, reader: torch.nn.Module) -> None:
    """Make `reader`, an attention layer that reads the KV cache of the
    attention layer `owner`, use the owner's k/v scales after loading."""
    if not isinstance(reader.quant_method, BaseKVCacheMethod):
        return
    method = owner.quant_method
    if not isinstance(method, SharedKVCacheScaleMethod):
        if not isinstance(method, BaseKVCacheMethod):
            return
        method = SharedKVCacheScaleMethod(method, owner)
        owner.quant_method = method
    method.readers.append(reader)
    reader.quant_method = method
