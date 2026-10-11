# SPDX-License-Identifier: Apache-2.0


import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    trailing_padding_used_len,
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
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )
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

        output = self.aiter_sage_attn_fn(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Packed varlen attention: [total_tokens, head_num, head_dim] in and out,
        one kernel call per segment of ``cu_seqlens``."""
        bounds = tuple(
            cu_seqlens_host
            if cu_seqlens_host is not None
            else (int(x) for x in cu_seqlens.tolist())
        )
        # MiniMax-H3 packs one live document as bounds=(0, used, total):
        # [0, used) are real tokens; [used, total) is 64-aligned tail padding.
        if self.packed_trailing_padding:
            used = trailing_padding_used_len(
                total_tokens=query.shape[0], max_seqlen=max_seqlen, bounds=bounds
            )
            if used is not None:
                bounds = (0, used)
        if bounds == (0, query.shape[0]):
            return self.forward(
                query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
            )[0]
        # Keep padded tail rows at zero so downstream masked rows stay inactive.
        output = torch.zeros_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            output[start:stop] = self.forward(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
            )[0]
        return output
