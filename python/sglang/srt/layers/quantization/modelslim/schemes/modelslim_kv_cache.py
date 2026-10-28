# SPDX-License-Identifier: Apache-2.0
"""ModelSlim KV cache contracts and shared parameter lifecycle."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ModelSlimKVCacheLayout:
    """Allocation contract; independent of checkpoint scale values."""

    layout: str
    key_head_dim: int
    value_head_dim: int
    local_kv_heads: int
    key_dtype: torch.dtype
    value_dtype: torch.dtype
    scale_dtype: Optional[torch.dtype] = None
    group_size: int = 0

    @property
    def bytes_per_token(self) -> int:
        size = self.local_kv_heads * (
            self.key_head_dim * torch._utils._element_size(self.key_dtype)
            + self.value_head_dim * torch._utils._element_size(self.value_dtype)
        )
        if self.scale_dtype is not None:
            size += (
                self.local_kv_heads
                * (self.key_head_dim + self.value_head_dim)
                // self.group_size
                * torch._utils._element_size(self.scale_dtype)
            )
        return size


class ModelSlimKVCacheScheme(ABC):
    """ModelSlim parameter lifecycle and cache encoding, without slot ownership."""

    def __init__(self, prefix, kv_cache_dtype):
        self.prefix = prefix
        if kv_cache_dtype not in self.SUPPORTED_KV_CACHE_DTYPES:
            raise ValueError(
                f"{prefix}: {type(self).__name__} supports KV cache dtype "
                f"{self.SUPPORTED_KV_CACHE_DTYPES}, got {kv_cache_dtype!r}"
            )

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def get_cache_layout(self, *, use_mla_backend) -> ModelSlimKVCacheLayout:
        raise NotImplementedError

    @abstractmethod
    def quantize_query(self, q: torch.Tensor):
        """Return encoded Q and per-token/head (or group) dequantization scales."""
        raise NotImplementedError

    @abstractmethod
    def quantize_cache(self, k, v):
        """Return encoded K, V and optional dynamic K scale."""
        raise NotImplementedError

    def _create_parameter(self, layer, name, shape, dtype, default):
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        parameter = torch.nn.Parameter(
            torch.full(shape, default, dtype=dtype),
            requires_grad=False,
        )
        parameter.modelslim_loaded = False

        def weight_loader(param, loaded_weight):
            if loaded_weight.dtype != dtype:
                raise ValueError(
                    f"{self.prefix}.{name}: expected {dtype}, got {loaded_weight.dtype}"
                )
            default_weight_loader(param, loaded_weight)
            param.modelslim_loaded = True

        parameter.weight_loader = weight_loader
        layer.register_parameter(name, parameter)
        return parameter
