"""A cache-free prefill outside the ModelRunner (prefill_only)."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.prefill_only import (
    PrefillOnlyAttnBackend,
    prefill_only,
    sdpa_causal_attention,
)
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.runtime_context import get_exec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SEQ_LENS = [5, 3]
Q_HEADS, KV_HEADS, DIM = 4, 2, 8


def _qkv():
    gen = torch.Generator().manual_seed(0)
    n = sum(SEQ_LENS)
    return (
        torch.randn(n, Q_HEADS, DIM, generator=gen),
        torch.randn(n, KV_HEADS, DIM, generator=gen),
        torch.randn(n, KV_HEADS, DIM, generator=gen),
    )


def _reference(q, k, v, scale):
    """Causal attention within each sequence, computed one sequence at a time."""
    out, start = [], 0
    for n in SEQ_LENS:
        rows = slice(start, start + n)
        repeat = Q_HEADS // KV_HEADS
        out.append(
            F.scaled_dot_product_attention(
                q[rows].transpose(0, 1),
                k[rows].repeat_interleave(repeat, 1).transpose(0, 1),
                v[rows].repeat_interleave(repeat, 1).transpose(0, 1),
                is_causal=True,
                scale=scale,
            ).transpose(0, 1)
        )
        start += n
    return torch.cat(out)


def test_sdpa_causal_attention_stays_within_each_sequence():
    q, k, v = _qkv()
    torch.testing.assert_close(
        sdpa_causal_attention(q, k, v, SEQ_LENS, 0.3), _reference(q, k, v, 0.3)
    )


def test_prefill_only_publishes_a_cache_free_extend_batch():
    positions = torch.arange(sum(SEQ_LENS))
    with published_topology():
        graphs = get_exec().graph.cuda_graph_config
        with prefill_only(SEQ_LENS, positions) as batch:
            assert get_forward_batch() is batch
            assert isinstance(get_attn_backend(), PrefillOnlyAttnBackend)
            assert batch.forward_mode == ForwardMode.EXTEND
            assert batch.extend_seq_lens_cpu == SEQ_LENS
            assert batch.positions is positions
            config = get_exec().graph.cuda_graph_config
            assert config.prefill.backend == config.decode.backend == "disabled"
        assert get_exec().graph.cuda_graph_config is graphs


def test_radix_attention_runs_through_the_callers_attention():
    q, k, v = _qkv()
    layer = RadixAttention(Q_HEADS, DIM, 0.3, num_kv_heads=KV_HEADS, layer_id=0)
    calls = []

    def attention(*args):
        calls.append(args[3])
        return sdpa_causal_attention(*args)

    with (
        published_topology(),
        prefill_only(
            SEQ_LENS, torch.arange(sum(SEQ_LENS)), attention=attention
        ) as batch,
    ):
        out = layer(q.flatten(1), k.flatten(1), v.flatten(1), batch)

    assert calls == [SEQ_LENS]
    torch.testing.assert_close(out.view(-1, Q_HEADS, DIM), _reference(q, k, v, 0.3))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
