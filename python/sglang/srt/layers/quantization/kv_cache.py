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
        layer._kv_scale_checkpoint = None
        layer._kv_scale_runtime = None

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    @staticmethod
    def _checkpoint_scales(layer):
        scales = (layer.k_scale.to("cpu").tolist(), layer.v_scale.to("cpu").tolist())
        if not all(isinstance(scale, float) for scale in scales):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        runtime = layer._kv_scale_runtime
        if runtime is not None:
            # Repeated postprocessing must use checkpoint values, not the
            # default/duplicated/FNUZ values written by the previous call. Check
            # each scale independently so a changed scale is not silently ignored.
            # Loaders MUST still restore before writing: a new checkpoint value
            # can have the same bytes as the previous runtime value.
            scales = tuple(
                raw if current == processed else current
                for current, processed, raw in zip(
                    scales, runtime, layer._kv_scale_checkpoint
                )
            )
        return scales

    @staticmethod
    def _set_scales(layer, k_scale, v_scale):
        # Keep Parameter identity, storage and loader attributes (CUDA graphs
        # and in-place weight loaders can retain references to these parameters).
        layer.k_scale.copy_(k_scale)
        layer.v_scale.copy_(v_scale)
        # Re-assert the exemption: a device move can hand back fresh Parameters,
        # and a k/v scale must never be reported as corruption -- a checkpoint
        # without one is legitimate and the runtime default is synthesized.
        layer.k_scale._skip_weight_check = True
        layer.v_scale._skip_weight_check = True
        layer.k_scale_float = k_scale
        layer.v_scale_float = v_scale

    def restore_weights_before_loading(self, layer: torch.nn.Module) -> None:
        """Restore checkpoint scales before an empty, partial or full reload.

        The caller must not run inference between restore and postprocessing.
        Preserve the original sentinel too: runtime (1, 1) is not sufficient to
        distinguish missing scales from a calibrated checkpoint on FNUZ devices.
        """
        if layer._kv_scale_runtime is None:
            return  # Initial load, or already restored; keep any pending writes.
        self._set_scales(layer, *self._checkpoint_scales(layer))
        layer._kv_scale_runtime = None

    def process_weights_after_loading(self, layer) -> None:
        checkpoint = self._checkpoint_scales(layer)
        k_scale, v_scale = checkpoint
        if k_scale <= 0.0 and v_scale <= 0.0:
            # No checkpoint scales: the runtime default must never be doubled.
            k_scale = v_scale = 1.0
        else:
            # A single kv_scale is remapped to k_scale by the checkpoint loader.
            # Retain v's sentinel in checkpoint state, not the duplicated value,
            # so subsequent k-only reloads continue to update both runtime scales.
            assert k_scale > 0.0
            if v_scale <= 0.0:
                v_scale = k_scale
            if is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        self._set_scales(layer, k_scale, v_scale)
        # Scalars are host-side lifecycle metadata, not additional model tensors.
        layer._kv_scale_checkpoint = checkpoint
        layer._kv_scale_runtime = (k_scale, v_scale)
