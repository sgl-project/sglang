# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py

import logging

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

logger = logging.getLogger(__name__)

_PER_HEAD_KV_SCALE_ATTENTION_BACKENDS = {"fa3"}


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

    def _is_per_head_scale(self, layer: torch.nn.Module) -> bool:
        kv_cache_scheme = getattr(self.quant_config, "kv_cache_scheme", None)
        if not (
            isinstance(kv_cache_scheme, dict)
            and kv_cache_scheme.get("strategy") == "attn_head"
        ):
            return False

        self._validate_per_head_scale_support(layer)
        return True

    def _validate_per_head_scale_support(self, layer: torch.nn.Module):
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        prefill_backend, decode_backend = server_args.get_attention_backends()
        unsupported_backends = {
            backend
            for backend in (prefill_backend, decode_backend)
            if backend not in _PER_HEAD_KV_SCALE_ATTENTION_BACKENDS
        }
        if unsupported_backends:
            raise ValueError(
                "FP8 KV cache attn_head scales are only supported with "
                f"{sorted(_PER_HEAD_KV_SCALE_ATTENTION_BACKENDS)} attention "
                "backend(s), but got "
                f"prefill={prefill_backend}, decode={decode_backend}."
            )

        if getattr(layer, "is_cross_attention", False):
            raise ValueError(
                "FP8 KV cache attn_head scales are not supported for "
                "cross-attention layers."
            )

        attn_type = getattr(layer, "attn_type", None)
        attn_type_value = getattr(attn_type, "value", attn_type)
        if attn_type_value != "decoder":
            raise ValueError(
                "FP8 KV cache attn_head scales are only supported for "
                "decoder self-attention layers, but got "
                f"attn_type={attn_type_value}."
            )

        if getattr(layer, "head_dim", 0) > 256:
            raise ValueError(
                "FP8 KV cache attn_head scales require attention head_dim <= "
                f"256 for the FA3 path, but got head_dim={layer.head_dim}."
            )

    def create_weights(self, layer: torch.nn.Module):
        """
        Create "weight" (aka k_scale and v_scale) for an attention layer.
        """
        # Initialize the KV cache scales to -1.0, which is an invalid value.
        # If the k/v_scale appears in the checkpoint, it will be
        # overwritten when loading weights.
        scale_shape = (layer.tp_k_head_num,) if self._is_per_head_scale(layer) else ()
        layer.k_scale = torch.nn.Parameter(
            torch.full(scale_shape, -1.0, dtype=torch.float32), requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.full(scale_shape, -1.0, dtype=torch.float32), requires_grad=False
        )
        layer.k_scale.weight_loader = self._kv_cache_scale_loader
        layer.v_scale.weight_loader = self._kv_cache_scale_loader
        layer.k_scale._skip_weight_check = True
        layer.v_scale._skip_weight_check = True

    @staticmethod
    def _kv_cache_scale_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
        loaded_weight = loaded_weight.reshape(-1)
        if param.numel() == loaded_weight.numel():
            param.data.copy_(loaded_weight.reshape_as(param))
            return
        if param.numel() == 1 and loaded_weight.numel() == 1:
            param.data.fill_(loaded_weight.item())
            return

        # attn_head scales are saved for all KV heads. Load only this
        # attention-TP rank's shard when the local parameter is sharded.
        from sglang.srt.layers.dp_attention import get_attention_tp_rank

        start = get_attention_tp_rank() * param.numel()
        end = start + param.numel()
        if loaded_weight.numel() >= end:
            param.data.copy_(loaded_weight[start:end].reshape_as(param))
            return

        raise ValueError(
            f"Attempted to load KV cache scale ({loaded_weight.size()}) "
            f"into parameter ({param.size()})"
        )

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer) -> None:
        k_scale_param = layer.k_scale.detach()
        v_scale_param = layer.v_scale.detach()
        if torch.all(k_scale_param > 0.0) and torch.all(v_scale_param > 0.0):
            # We prefer to use separate k_scale and v_scale if present
            k_scale = k_scale_param.to(torch.float32)
            v_scale = v_scale_param.to(torch.float32)
            if is_fp8_fnuz():
                k_scale = k_scale * 2
                v_scale = v_scale * 2
        elif torch.all(k_scale_param < 0.0) and torch.all(v_scale_param < 0.0):
            # If no scales were loaded (both scales are invalid negative
            # values), use the default value of 1.0
            k_scale = torch.ones_like(k_scale_param, dtype=torch.float32)
            v_scale = torch.ones_like(v_scale_param, dtype=torch.float32)
        else:
            # If we find a single kv_scale in the checkpoint, we remap
            # kv_scale to k_scale during weight loading, and duplicate
            # k_scale to v_scale here
            if torch.all(k_scale_param > 0.0):
                scale_to_duplicate = k_scale_param
            else:
                assert torch.all(v_scale_param > 0.0)
                scale_to_duplicate = v_scale_param
            k_scale = scale_to_duplicate.to(torch.float32)
            v_scale = scale_to_duplicate.to(torch.float32)
            if is_fp8_fnuz():
                k_scale = k_scale * 2
                v_scale = v_scale * 2

        # These are used in the final Attention.forward()
        layer.k_scale.copy_(k_scale)
        layer.v_scale.copy_(v_scale)
        layer.k_scale_float = k_scale.item() if k_scale.numel() == 1 else k_scale
        layer.v_scale_float = v_scale.item() if v_scale.numel() == 1 else v_scale
