# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import math
from typing import Optional

from mindspore import Tensor, nn, ops
from mindspore.ops.operations.nn_ops import (
    FlashAttentionScore,
    PagedAttention,
    ReshapeAndCache,
)


class MsNativeAttnBackend(nn.Cell):
    """MindSpore Attention Manager."""

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        scale_value: Optional[float] = None,
        mla_v_dim: int = 0,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.scale_value = (
            1 / math.sqrt(self.head_dim) if scale_value is None else scale_value
        )
        self.attention_layout = "TH"

        self.flash_attention = FlashAttentionScore(
            head_num=self.n_heads,
            scale_value=self.scale_value,
            next_tokens=0,
            input_layout=self.attention_layout,
        )
        self.paged_attention = PagedAttention(
            head_num=self.n_heads,
            scale_value=self.scale_value,
            kv_head_num=self.n_kv_heads,
            mla_v_dim=mla_v_dim,
        )
        self.reshape_and_cache = ReshapeAndCache()

    # pylint: disable=W0613
    def construct(
        self,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor = None,
        value_cache: Tensor = None,
        out_cache_loc: Tensor = None,
        k_scale: float = None,
        v_scale: float = None,
    ) -> Tensor:
        if k_scale is not None:
            key = key / k_scale
        if v_scale is not None:
            value = value / v_scale
        cache_out = self.reshape_and_cache(
            key, value, key_cache, value_cache, out_cache_loc
        )
        key = ops.depend(key, cache_out)

        return key

    def extend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor = None,
        alibi_mask: Tensor = None,
        prefix=None,
        padding_mask: Tensor = None,
        q_seq_lens: Tensor = None,
        batch_valid_length: Tensor = None,
    ) -> Tensor:
        _, _, _, output = self.flash_attention(
            query,
            key,
            value,
            alibi_mask,
            None,
            padding_mask,
            attn_mask,
            prefix,
            q_seq_lens,
            batch_valid_length,
        )
        return output

    def decode(
        self,
        query: Tensor,
        batch_valid_length: Tensor,
        attn_mask: Tensor = None,
        q_seq_lens: Tensor = None,
        key_cache: Tensor = None,
        value_cache: Tensor = None,
        block_tables: Tensor = None,
    ) -> Tensor:
        output = self.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            batch_valid_length,
            None,
            None,
            attn_mask,
            q_seq_lens,
        )
        return output
