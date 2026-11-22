# SPDX-License-Identifier: Apache-2.0
"""FlashInfer FA2 for dense BSHD attention, including rectangular GQA."""

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class FlashInferFA2Backend(AttentionBackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FLASHINFER_FA2

    @staticmethod
    def get_impl_cls() -> type["FlashInferFA2Impl"]:
        return FlashInferFA2Impl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError


class FlashInferFA2Impl(AttentionImpl):
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
        from flashinfer import single_prefill_with_kv_cache

        self._prefill = single_prefill_with_kv_cache
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # FlashInfer takes one NHD sequence and supports strided Q/K/V. Avoid
        # stacking or copying its output for the common batch-one case.
        def attend(index: int) -> torch.Tensor:
            return self._prefill(
                query[index],
                key[index],
                value[index],
                kv_layout="NHD",
                causal=self.causal,
                sm_scale=self.softmax_scale,
                backend="fa2",
            )

        if query.shape[0] == 1:
            return attend(0).unsqueeze(0)
        return torch.stack([attend(index) for index in range(query.shape[0])])
