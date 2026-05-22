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
        sdpa_context = (
            sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
            if self.allow_cudnn_sdp and query.device.type == "cuda"
            else nullcontext()
        )
        with sdpa_context:
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, **attn_kwargs
            )
        output = output.transpose(1, 2)
        return output
