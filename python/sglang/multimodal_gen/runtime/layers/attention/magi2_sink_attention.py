# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend


class Magi2SinkAttention(nn.Module):
    """``USPAttention``'s varlen paths cannot pass sink logits, so this drives ``FlashAttentionImpl`` directly."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        sink_token_num: int = 0,
        window_size: tuple[int, int] = (-1, -1),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.softmax_scale = softmax_scale or head_dim**-0.5
        self.window_size = window_size

        # fp32: compared against attention logits, and the reference avoids bf16 here.
        self.sinks = (
            nn.Parameter(torch.zeros(sink_token_num, num_heads, dtype=torch.float32))
            if sink_token_num
            else None
        )
        self._impl = None

    def _local_sinks(self) -> torch.Tensor | None:
        """``_usp_input_all_to_all_packed_qkv`` gives rank *r* heads ``[r * h_local, (r + 1) * h_local)``."""
        if self.sinks is None:
            return None
        if self.sinks.shape[0] != 1:
            raise ValueError(
                "FlashAttention takes one sink logit per head, so "
                f"sink_token_num must be 1; got {self.sinks.shape[0]}"
            )
        # The checkpoint stores [sink_token_num, num_heads]; the kernel wants flat.
        sinks = self.sinks[0]
        world_size = get_ulysses_parallel_world_size()
        if world_size <= 1:
            return sinks
        h_local = self.num_heads // world_size
        start = get_ulysses_parallel_rank() * h_local
        return sinks[start : start + h_local]

    def _ensure_impl(self, dtype: torch.dtype) -> None:
        if self._impl is not None:
            return
        backend = get_attn_backend(
            self.head_dim,
            dtype,
            attention_requirements=AttentionRequirements(packed_varlen=True),
        )
        world_size = get_ulysses_parallel_world_size()
        local_heads = self.num_heads // world_size
        local_kv_heads = max(1, self.num_kv_heads // world_size)
        sinks = self._local_sinks()
        self._impl = backend.get_impl_cls()(
            num_heads=local_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.softmax_scale,
            num_kv_heads=local_kv_heads,
            # Only FlashAttentionImpl forwards these; other backends drop the sink
            # silently. The parameter stays fp32, but FA3 wants sinks in the q dtype.
            sinks=None if sinks is None else sinks.to(dtype).contiguous(),
            window_size=self.window_size,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.layers.usp import (
            _usp_input_all_to_all_packed_qkv,
            _usp_output_all_to_all,
        )

        ulysses = get_ulysses_parallel_world_size() > 1
        if ulysses:
            q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)

        self._ensure_impl(q.dtype)
        out = self._impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if ulysses:
            out = _usp_output_all_to_all(out[None], head_dim=2)[0]
        return out
