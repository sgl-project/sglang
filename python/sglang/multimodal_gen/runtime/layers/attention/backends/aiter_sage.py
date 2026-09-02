# SPDX-License-Identifier: Apache-2.0


import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class AITERSageBackend(AttentionBackend):

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_SAGE

    @staticmethod
    def get_impl_cls() -> type["AITERSageImpl"]:
        return AITERSageImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        # AITER Sage backend does not require special metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER Sage backend does not have a metadata builder."
        )


class AITERSageImpl(AttentionImpl):

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

            self.aiter_sage_attn_fn = fav3_sage_wrapper_func
        except ImportError:
            raise ImportError(
                "AITER Sage attention is not available, please update AITER version."
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs attention using aiter sage backend.

        Args:
            query: Query tensor of shape [batch_size, seq_len, head_num, head_dim]
            key: Key tensor of shape [batch_size, seq_len, head_num, head_dim]
            value: Value tensor of shape [batch_size, seq_len, head_num, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, head_num, head_dim]
        """

        output = self.aiter_sage_attn_fn(query, key, value)
        return output
