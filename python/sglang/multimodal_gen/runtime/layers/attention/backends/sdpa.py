# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


class SDPABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TORCH_SDPA

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SDPAImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        self.allow_cudnn_sdp = bool(extra_impl_args.get("allow_cudnn_sdp", False))

    def _sdpa_context(self, query: torch.Tensor):
        if self.allow_cudnn_sdp and query.device.type == "cuda":
            return sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
        return nullcontext()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # transpose to bs, heads, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_kwargs = {
            "attn_mask": None,
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale,
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        with self._sdpa_context(query):
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, **attn_kwargs
            )
        output = output.transpose(1, 2)
        return output


class CudnnSDPABackend(SDPABackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TORCH_CUDNN_SDPA

    @staticmethod
    def get_impl_cls() -> type["CudnnSDPAImpl"]:
        return CudnnSDPAImpl


class CudnnSDPAImpl(SDPAImpl):
    def _sdpa_context(self, query: torch.Tensor):
        if query.device.type == "cuda":
            return sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
        return nullcontext()


class DynamicCudnnSDPABackend(SDPABackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.DYNAMIC_CUDNN_SDPA

    @staticmethod
    def get_impl_cls() -> type["DynamicCudnnSDPAImpl"]:
        return DynamicCudnnSDPAImpl


class DynamicCudnnSDPAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
            FlashAttentionImpl,
            set_fa_ver,
        )

        self.causal = causal
        self.head_size = head_size
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            set_fa_ver(4)
        self.cudnn_impl = CudnnSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.cudnn",
            **extra_impl_args,
        )
        self.fa_impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.fa",
            **extra_impl_args,
        )

    def _use_cudnn_sdpa(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> bool:
        if self.causal:
            return False
        if query.device.type != "cuda":
            return False
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False
        if query.shape[2] != key.shape[2]:
            return False
        if query.shape[1] != key.shape[1]:
            return False
        return query.shape[-1] == 64 and query.shape[1] == 1024 and query.shape[0] >= 4

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self._use_cudnn_sdpa(query, key, value):
            return self.cudnn_impl.forward(query, key, value, attn_metadata)
        return self.fa_impl.forward(query, key, value, attn_metadata)
