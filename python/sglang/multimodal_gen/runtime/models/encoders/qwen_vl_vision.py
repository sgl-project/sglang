# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen-VL vision attention for multimodal generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.runtime_context import get_parallel

logger = init_logger(__name__)


@dataclass(frozen=True)
class PackedSequenceMetadata:
    cu_seqlens: torch.Tensor
    cu_seqlens_host: tuple[int, ...]
    max_seqlen: int

    @classmethod
    def from_cu_seqlens(cls, cu_seqlens: torch.Tensor) -> PackedSequenceMetadata:
        bounds = tuple(int(value) for value in cu_seqlens.tolist())
        return cls(
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=bounds,
            max_seqlen=max(
                stop - start for start, stop in zip(bounds[:-1], bounds[1:])
            ),
        )


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype = query.dtype
    key_dtype = key.dtype
    query = query.float()
    key = key.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    query = query * cos + _rotate_half(query) * sin
    key = key * cos + _rotate_half(key) * sin
    return query.to(query_dtype), key.to(key_dtype)


class QwenVLVisionAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        prefix: str,
        model_name: str,
        quant_config: Any = None,
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        self.num_heads = config.num_heads // parallel.tp_size
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )
        self.proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )

        backend = get_attn_backend(self.head_dim, torch.get_default_dtype())
        self._attention_impl = None
        if backend.supports_packed_varlen():
            self._attention_impl = backend.get_impl_cls()(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                num_kv_heads=self.num_heads,
                softmax_scale=self.scaling,
                causal=False,
                prefix=prefix,
            )
        else:
            logger.warning_once(
                f"{model_name} vision attention uses torch SDPA because "
                f"{backend.get_enum().name.lower()} does not support packed sequences"
            )

    def _packed_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: PackedSequenceMetadata,
    ) -> torch.Tensor:
        if self._attention_impl is not None:
            return self._attention_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=metadata.cu_seqlens,
                cu_seqlens_host=metadata.cu_seqlens_host,
                max_seqlen=metadata.max_seqlen,
            )

        output = torch.empty_like(query)
        for start, stop in zip(
            metadata.cu_seqlens_host[:-1], metadata.cu_seqlens_host[1:]
        ):
            if start == stop:
                continue
            segment = F.scaled_dot_product_attention(
                query[start:stop].transpose(0, 1).unsqueeze(0),
                key[start:stop].transpose(0, 1).unsqueeze(0),
                value[start:stop].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            output[start:stop] = segment.squeeze(0).transpose(0, 1)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: PackedSequenceMetadata,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = (
            qkv.reshape(sequence_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = _apply_rotary_embedding(query, key, *position_embeddings)
        output = self._packed_attention(query, key, value, metadata)
        output, _ = self.proj(output.reshape(sequence_length, -1).contiguous())
        return output
