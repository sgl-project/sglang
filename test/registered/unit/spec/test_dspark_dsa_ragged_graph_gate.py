from types import SimpleNamespace

import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

DeepseekSparseAttnBackend = type("DeepseekSparseAttnBackend", (), {})
OtherBackend = type("OtherBackend", (), {})


def _runner(backend, *, verify_width=8):
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.attn_backend = backend
    runner.num_tokens_per_bs = verify_width
    return runner


def _layout(*, verify_lens, graph_num_tokens=None, total_verify_tokens=None):
    if graph_num_tokens is None:
        graph_num_tokens = sum(verify_lens)
    if total_verify_tokens is None:
        total_verify_tokens = sum(verify_lens)
    return SimpleNamespace(
        verify_lens=torch.tensor(verify_lens, dtype=torch.int32),
        verify_lens_cpu=list(verify_lens),
        graph_num_tokens=graph_num_tokens,
        total_verify_tokens=total_verify_tokens,
    )


def test_dsa_ragged_graph_allows_full_width_layout():
    runner = _runner(DeepseekSparseAttnBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 8]),
    )


def test_dsa_ragged_graph_rejects_non_uniform_compact_layout():
    runner = _runner(DeepseekSparseAttnBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert not runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 1], graph_num_tokens=16),
    )


def test_dsa_ragged_graph_rejects_token_tier_padding():
    runner = _runner(DeepseekSparseAttnBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert not runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 8], graph_num_tokens=24, total_verify_tokens=16),
    )


def test_non_dsa_backend_keeps_generic_ragged_graph_path():
    runner = _runner(OtherBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 1], graph_num_tokens=16),
    )
