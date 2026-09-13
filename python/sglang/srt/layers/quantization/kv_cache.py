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
from sglang.srt.runtime_context import get_model

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

    def _checkpoint_kv_scales_apply(self, kv_cache_dtype) -> bool:
        """Whether --kv-cache-dtype is a per-tensor FP8 cache that consumes this
        checkpoint's external k_scale/v_scale: fp8_e4m3/fp8_e5m2, or "auto" under an
        FP8 KV scheme. bf16 and block-scaled formats (mxfp8/mxfp4/nvfp4/...) carry
        their own/no scales; keys on the raw string, not the shared torch dtype.
        """
        if kv_cache_dtype in ("fp8_e4m3", "fp8_e5m2"):
            return True
        if kv_cache_dtype == "auto":
            algo = getattr(self.quant_config, "kv_cache_quant_algo", None)
            return isinstance(algo, str) and algo.upper() == "FP8"
        return False

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

        # Apply the checkpoint's FP8-KV scales only when the runtime KV cache
        # actually is that FP8 cache; otherwise fall back to unit scales so every
        # backend reads an identity descale with no per-backend guard. Mirrors
        # load_kv_cache_scales(), which only loads scales for fp8_e4m3.
        if not self._checkpoint_kv_scales_apply(get_model().kv_cache_dtype):
            k_scale = 1.0
            v_scale = 1.0

        # These are used in the final Attention.forward()
        layer.k_scale.copy_(k_scale)
        layer.v_scale.copy_(v_scale)
        layer.k_scale_float = k_scale
        layer.v_scale_float = v_scale
