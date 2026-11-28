"""Run a model's layer stack over one prefill, outside a ModelRunner.

A caller that only needs hidden states (a diffusion pipeline's text encoder,
for one) can build an srt model under its own parallel groups and call the
model's ``forward`` inside :func:`prefill_only`. Nothing is cached: attention
runs causally within each sequence and writes no KV.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch
from sglang.srt.runtime_context import get_exec

# (q [tokens, q_heads, dim], k [tokens, kv_heads, dim], v [tokens, kv_heads, v_dim],
#  per-sequence lengths, softmax scale) -> [tokens, q_heads, v_dim]
PrefillAttention = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Sequence[int], float], torch.Tensor
]


def sdpa_causal_attention(q, k, v, seq_lens, scale):
    """Causal attention within each sequence, through torch SDPA."""
    out = torch.empty((*q.shape[:2], v.shape[-1]), dtype=q.dtype, device=q.device)
    start = 0
    for seq_len in seq_lens:
        rows = slice(start, start + seq_len)
        out[rows] = F.scaled_dot_product_attention(
            q[rows].transpose(0, 1),
            k[rows].transpose(0, 1),
            v[rows].transpose(0, 1),
            is_causal=True,
            scale=scale,
            enable_gqa=q.shape[1] != k.shape[1],
        ).transpose(0, 1)
        start += seq_len
    return out


class PrefillOnlyAttnBackend(AttentionBackend):
    """Extend-only attention that keeps no KV cache; ``attention`` computes it."""

    def __init__(self, attention: PrefillAttention = sdpa_causal_attention):
        self.attention = attention

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **_):
        out = self.attention(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
            v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            forward_batch.extend_seq_lens_cpu,
            layer.scaling,
        )
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("a prefill-only forward has no decode step")


@contextmanager
def prefill_only(
    seq_lens: Sequence[int],
    positions: torch.Tensor,
    attention: Optional[PrefillAttention] = None,
) -> Iterator[ForwardBatch]:
    """Publish a cache-free extend batch of ``seq_lens`` (concatenated rows).

    ``positions`` is what the model's forward takes: ``[tokens]``, or
    ``[3, tokens]`` under M-RoPE. Pass the yielded batch to the forward.
    """
    seq_lens = [int(n) for n in seq_lens]
    num_tokens = sum(seq_lens)
    device = positions.device
    lens = torch.tensor(seq_lens, dtype=torch.int64)
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        input_ids=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        req_pool_indices=torch.arange(len(seq_lens), device=device),
        seq_lens=lens.to(device),
        out_cache_loc=torch.empty(0, dtype=torch.int64, device=device),
        seq_lens_sum=num_tokens,
        seq_lens_cpu=lens,
        positions=positions,
        extend_num_tokens=num_tokens,
        extend_seq_lens=lens.to(device),
        extend_prefix_lens=torch.zeros_like(lens, device=device),
        extend_seq_lens_cpu=seq_lens,
        extend_prefix_lens_cpu=[0] * len(seq_lens),
        is_prefill_only=True,
    )
    backend = PrefillOnlyAttnBackend(attention or sdpa_causal_attention)
    # Eager: no graph runner, whatever the process configured for serving.
    no_graphs = CudaGraphConfig(decode=PhaseConfig(), prefill=PhaseConfig())
    with (
        get_exec().graph.override(cuda_graph_config=no_graphs),
        forward_context(ForwardContext(backend)),
        set_forward_batch(forward_batch),
    ):
        yield forward_batch
