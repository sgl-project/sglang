# SPDX-License-Identifier: Apache-2.0


import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class AITERSageFp8Backend(AttentionBackend):

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_SAGE_FP8

    @staticmethod
    def get_impl_cls() -> type["AITERSageFp8Impl"]:
        return AITERSageFp8Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER Sage FP8 backend does not have a metadata builder."
        )


class AITERSageFp8Impl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:

        try:
            from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func

            self.attn_fn = fav3_sage_wrapper_func
        except ImportError:
            raise ImportError(
                "AITER Sage FP8 attention is not available, please update AITER version."
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs attention using AITER Sage FP8 backend.

        Args:
            query: Query tensor of shape [batch_size, seq_len, head_num, head_dim]
            key: Key tensor of shape [batch_size, seq_len, head_num, head_dim]
            value: Value tensor of shape [batch_size, seq_len, head_num, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, head_num, head_dim]
        """

        output = self.attn_fn(query, key, value)
        return output
