"""CPU contracts shared by the full and breakable prefill attention paths."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_batch(tokens=2, rows=4, return_lse=False):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        global_num_token_non_padded_cpu=tokens,
        out_cache_loc=torch.arange(rows),
        positions=torch.arange(rows),
        _attn_output=None,
        mha_return_lse=return_lse,
    )


@pytest.mark.parametrize("breakable", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("tokens", [0, 2, 4])
def test_dense_outputs_padding_and_lse(breakable, return_lse, tokens):
    layer = RadixAttention(2, 3, 1.0, 2, 7)
    batch = make_batch(tokens=tokens, return_lse=return_lse)
    original = batch.out_cache_loc, batch.positions
    calls = []

    def attention(q, k, v, actual_layer, live_batch, save_kv_cache, **kwargs):
        assert actual_layer is layer
        assert live_batch is batch
        assert q.shape[0] == tokens
        assert k.shape[0] == v.shape[0] == 3
        assert batch.positions.shape[0] == batch.out_cache_loc.shape[0] == tokens
        calls.append(q)
        out = torch.full_like(q, 3)
        if return_lse:
            return out, torch.full((tokens, 2), 7.0)
        return out

    q = torch.zeros(4, 2, 3)
    with (
        forward_context(
            ForwardContext(
                SimpleNamespace(forward=attention),
                full_graph=not breakable,
                raw_num_tokens=tokens,
            )
        ),
        patch(
            "sglang.srt.layers.radix_attention.is_in_breakable_cuda_graph",
            return_value=breakable,
        ),
    ):
        result = layer(q, q, q, batch, key_value_num_tokens=3)
    output, lse = result if return_lse else (result, None)
    torch.testing.assert_close(output[:tokens], torch.full_like(output[:tokens], 3))
    assert torch.count_nonzero(output[tokens:]) == 0
    assert len(calls) == bool(tokens)
    assert batch.out_cache_loc is original[0] and batch.positions is original[1]
    assert batch._attn_output is None
    if return_lse:
        assert lse.shape == (4, 2)
        assert torch.all(lse[:tokens] == 7)
        assert torch.count_nonzero(lse[tokens:]) == 0


def test_extra_kwargs_and_exception_restore():
    layer = RadixAttention(2, 3, 1.0, 2, 0)
    batch = make_batch()
    sentinel = torch.empty(1)
    batch._attn_output = sentinel
    original = batch.out_cache_loc, batch.positions
    modifier = lambda score: score
    q = torch.zeros(4, 2, 3)
    aux = [torch.arange(4)]

    def attention(q, k, v, actual_layer, live_batch, save_kv_cache, **kwargs):
        assert kwargs["score_mod"] is modifier
        assert kwargs["aux_tensors"][0].shape == (2,)
        assert kwargs["q_rope"].shape[0] == 2
        assert kwargs["k_rope"].shape[0] == 3
        assert kwargs["q_descale"].shape[0] == 2
        raise RuntimeError("backend failed")

    with forward_context(
        ForwardContext(
            SimpleNamespace(forward=attention), full_graph=True, raw_num_tokens=2
        )
    ):
        with pytest.raises(RuntimeError, match="backend failed"):
            layer(
                q,
                q,
                q,
                batch,
                key_value_num_tokens=3,
                score_mod=modifier,
                aux_tensors=aux,
                q_rope=q,
                k_rope=q,
                q_descale=q,
            )
    assert batch.out_cache_loc is original[0] and batch.positions is original[1]
    assert batch._attn_output is sentinel
    assert aux[0].shape == (4,)


@pytest.mark.parametrize("tokens", [0, 2])
@pytest.mark.parametrize("has_index_value", [False, True])
def test_sparse_full_graph_two_outputs(tokens, has_index_value):
    layer = RadixAttention(2, 3, 1.0, 2, 0)
    batch = make_batch(tokens=tokens)
    q = torch.zeros(4, 2, 3)
    idx = torch.zeros(4, 1, 2)

    def attention(q, k, v, layer, batch, save_kv_cache, **kwargs):
        assert kwargs["idx_q"].shape == (tokens, 1, 2)
        assert kwargs["idx_k"].shape == (tokens, 1, 2)
        return (
            torch.full((tokens, 1, 2), 5.0) if has_index_value else None,
            torch.ones_like(q),
        )

    with forward_context(
        ForwardContext(
            SimpleNamespace(forward=attention), full_graph=True, raw_num_tokens=tokens
        )
    ):
        index_output, output = layer(q, q, q, batch, idx_q=idx, idx_k=idx)
    assert output.shape == (4, 6) and index_output.shape == (4, 2)
    assert torch.all(output[:tokens] == 1)
    assert torch.count_nonzero(output[tokens:]) == 0
    if has_index_value:
        assert torch.all(index_output[:tokens] == 5)
    assert torch.count_nonzero(index_output[tokens:]) == 0


@pytest.mark.parametrize("sparse", [False, True])
def test_direct_backend_dispatch(sparse):
    layer = RadixAttention(2, 3, 1.0, 2, 0)
    batch = make_batch()
    q = torch.zeros(4, 2, 3)
    result = object()
    backend = SimpleNamespace(forward=lambda *args, **kwargs: result)
    with (
        forward_context(ForwardContext(backend)),
        patch(
            "sglang.srt.layers.radix_attention.is_in_breakable_cuda_graph",
            return_value=sparse,
        ),
    ):
        kwargs = {"idx_q": q, "idx_k": q} if sparse else {}
        assert layer(q, q, q, batch, **kwargs) is result


def test_output_dtype_follows_values():
    layer = RadixAttention(2, 3, 1.0, 2, 0)
    batch = make_batch()
    q = torch.zeros(4, 2, 3, dtype=torch.float32)
    v = q.to(torch.bfloat16)
    backend = SimpleNamespace(
        forward=lambda q, k, v, *args, **kwargs: torch.ones_like(v)
    )
    with forward_context(ForwardContext(backend, full_graph=True, raw_num_tokens=2)):
        assert layer(q, q, v, batch).dtype == torch.bfloat16


@pytest.mark.parametrize("return_lse", [False, True])
def test_attention_replay_uses_live_batch_and_static_output_buffers(return_lse):
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        breakable_cuda_graph as bcg,
    )

    layer = RadixAttention(2, 3, 1.0, 2, 0)
    q = torch.zeros(4, 2, 3)
    captured_batch = make_batch(return_lse=return_lse)
    live_batch = make_batch(tokens=1, return_lse=return_lse)
    seen = []

    def attention(query, key, value, actual_layer, batch, save_kv_cache, **kwargs):
        seen.append(batch)
        out = torch.full_like(query, float(len(seen)))
        lse = torch.full((query.shape[0], 2), float(len(seen)))
        return (out, lse) if return_lse else out

    graph = bcg.BreakableCUDAGraph()
    capture = SimpleNamespace(
        cuda_graph=graph,
        _barrier_fn=None,
        _end_current_segment=lambda: None,
        _begin_new_segment=lambda: None,
    )
    backend = SimpleNamespace(forward=attention)
    with (
        forward_context(ForwardContext(backend, raw_num_tokens=2)),
        patch(
            "sglang.srt.layers.radix_attention.is_in_breakable_cuda_graph",
            return_value=True,
        ),
    ):
        token = bcg._current_capture_var.set(capture)
        try:
            result = layer(q, q, q, captured_batch)
        finally:
            bcg._current_capture_var.reset(token)
    output, lse = result if return_lse else (result, None)
    pointer = output.data_ptr()
    with forward_context(ForwardContext(backend, raw_num_tokens=1)):
        graph._break_fns[0](live_batch)
    assert seen == [captured_batch, live_batch]
    assert output.data_ptr() == pointer
    assert torch.all(output[:1] == 2) and torch.count_nonzero(output[1:]) == 0
    if return_lse:
        assert torch.all(lse[:1] == 2) and torch.count_nonzero(lse[1:]) == 0
