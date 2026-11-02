"""SGLang transport tests against AITER's public plan-based PA API."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention import aiter_backend, aiter_utils
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm"
)


def _backend(query_length, partitions=8, dtype=torch.bfloat16):
    backend = object.__new__(aiter_backend.AiterAttnBackend)
    backend.device = torch.device("cuda")
    backend.page_size = 64
    backend.num_head, backend.num_kv_head = 16, 1
    backend.kv_cache_dtype = dtype
    backend.kv_cache_is_vectorized_5d = True
    backend.use_mla = False
    backend.topk = 1
    backend.max_context_len = 1_048_576
    backend.head_dim, backend.v_head_dim = 192, 128
    backend.input_dtype = torch.bfloat16
    backend._flydsl_pa_work_plans = {}
    backend._flydsl_pa_decode_query_length = query_length
    backend._flydsl_pa_decode_workspace_max_bs = 0
    backend._flydsl_pa_decode_compiled = False
    with (
        aiter_backend.envs.SGLANG_AITER_PA_DECODE_IMPL.override("flydsl"),
        patch.dict(os.environ, {"SGLANG_FLYDSL_PA_NUM_PARTITIONS": str(partitions)}),
    ):
        backend._configure_flydsl_pa_decode(4)
    return backend


def _inputs(qlen, kv_heads, head_dim, dtype):
    torch.manual_seed(213)
    batch, pages = 3, 33
    blocks = batch * pages
    k = (torch.randn(blocks, 64, kv_heads, head_dim, device="cuda") * 0.3).to(dtype)
    v = (torch.randn(blocks, 64, kv_heads, 128, device="cuda") * 0.3).to(dtype)
    vector = 8 if dtype == torch.bfloat16 else 16
    kc = k.view(blocks, 64, kv_heads, head_dim // vector, vector)
    kc = kc.permute(0, 2, 3, 1, 4).contiguous()
    vc = v.view(blocks, 64 // vector, vector, kv_heads, 128)
    vc = vc.permute(0, 3, 1, 4, 2).contiguous()
    table = torch.zeros((batch + 1, pages), dtype=torch.int32, device="cuda")
    table[:batch].copy_(torch.randperm(blocks, device="cuda").view(batch, pages))
    q = torch.randn(
        batch * qlen * 2, 16 * head_dim, device="cuda", dtype=torch.bfloat16
    )[::2]
    guarded = torch.full(
        (batch * qlen + 2, 16 * 128), 19.0, device="cuda", dtype=torch.bfloat16
    )
    return k, v, kc, vc, table, q, guarded


def _reference(
    q, k, v, table, lengths, qlen, head_dim, sinks=None, window=0, scale=1.0
):
    batch = len(lengths)
    result = torch.empty(batch * qlen, 16, 128, device=q.device)
    kv_heads = k.shape[2]
    for i, length in enumerate(lengths):
        keys = k[table[i].long()].flatten(0, 1)[:length].float() * scale
        values = v[table[i].long()].flatten(0, 1)[:length].float() * scale
        keys = keys.repeat_interleave(16 // kv_heads, dim=1)
        values = values.repeat_interleave(16 // kv_heads, dim=1)
        queries = q.view(batch, qlen, 16, head_dim)[i].float()
        logits = torch.einsum("qhd,khd->hqk", queries, keys) * head_dim**-0.5
        positions = length - qlen + torch.arange(qlen, device=q.device)
        token = torch.arange(length, device=q.device)
        mask = token[None, :] <= positions[:, None]
        if window:
            mask &= token[None, :] > positions[:, None] - window
        logits.masked_fill_(~mask[None], float("-inf"))
        if sinks is not None:
            logits = torch.cat(
                (logits, sinks[:, None, None].expand(-1, qlen, 1)), dim=-1
            )
        probs = torch.softmax(logits, dim=-1).nan_to_num()
        probs = probs[..., :length]
        result[i * qlen : (i + 1) * qlen] = torch.einsum("hqk,khd->qhd", probs, values)
    return result.flatten(1).to(torch.bfloat16)


@pytest.mark.parametrize("query_length", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, fp8_dtype], ids=["bf16", "fp8"])
@pytest.mark.parametrize("partitions", [8, 64])
def test_full_pa_transport_eager_and_dynamic_graph(query_length, dtype, partitions):
    backend = _backend(query_length, partitions, dtype)
    backend._compile_flydsl_pa_decode()
    k, v, kc, vc, table, q, guarded = _inputs(query_length, 1, 192, dtype)
    output = guarded[1:-1]
    layer = SimpleNamespace(
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        sliding_window_size=-1,
        logit_cap=0.0,
        scaling=192**-0.5,
    )
    scale = 0.75 if dtype == fp8_dtype else 1.0
    if dtype == fp8_dtype:
        # Model scales are zero-dimensional Parameters; the adapter must pass
        # zero-copy length-one views to AITER.
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(scale, device="cuda"), requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(scale, device="cuda"), requires_grad=False
        )
    seq_lens = torch.tensor([0, 67, 2081], dtype=torch.int64, device="cuda")
    batch = SimpleNamespace(batch_size=3, seq_lens=seq_lens)
    backend.forward_metadata = SimpleNamespace(
        max_q_len=query_length, kv_indices=table if query_length == 1 else table[:3]
    )
    decode = query_length == 1
    forward = (
        aiter_utils.forward_decode_vectorized_5d
        if decode
        else aiter_utils.forward_target_verify_flydsl_5d
    )

    def run():
        forward(backend, q, layer, batch, kc, vc, output, sinks=None)

    def check():
        lengths = seq_lens.tolist()
        if not decode:
            lengths = [n + query_length for n in lengths]
        expected = _reference(q, k, v, table, lengths, query_length, 192, scale=scale)
        torch.testing.assert_close(output, expected, atol=0.006, rtol=0.04)
        assert torch.all(guarded[[0, -1]] == 19)

    run()
    check()
    plans = [
        (p, p.work_info.data_ptr(), p.reduce_info.data_ptr())
        for p in backend._flydsl_pa_work_plans.values()
    ]
    scratch = backend._flydsl_pa_decode_pout.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for state in ([61, 1017, 2048], [129, 0, 2051], [0, 67, 2081]):
        seq_lens.copy_(torch.tensor(state, dtype=seq_lens.dtype, device="cuda"))
        table[:3].copy_(table[:3].roll(1, dims=1))
        graph.replay()
        check()
    for plan, work_ptr, reduce_ptr in plans:
        assert plan.work_info.data_ptr() == work_ptr
        assert plan.reduce_info.data_ptr() == reduce_ptr
    assert backend._flydsl_pa_decode_pout.data_ptr() == scratch
    assert len(backend._flydsl_pa_work_plans) == len(plans)


@pytest.mark.parametrize("query_length", [1, 4, 8])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_draft_pa_transport_eager_and_dynamic_graph(
    monkeypatch, query_length, with_sinks
):
    monkeypatch.setenv("SGLANG_AITER_DFLASH_SWA_IMPL", "flydsl")
    backend = _backend(query_length)
    backend.is_draft_worker = True
    backend.head_dim, backend.v_head_dim, backend.num_kv_head = 128, 128, 2
    backend._flydsl_dflash_swa_enabled = True
    aiter_utils.prepare_flydsl_dflash_swa(backend, 4, query_length, 1024)
    k, v, kc, vc, table, q, guarded = _inputs(query_length, 2, 128, torch.bfloat16)
    layer = SimpleNamespace(
        tp_q_head_num=16,
        tp_k_head_num=2,
        qk_head_dim=128,
        v_head_dim=128,
        scaling=128**-0.5,
        sliding_window_size=1024,
    )
    seq_lens = torch.tensor([0, 1021, 2079], dtype=torch.int64, device="cuda")
    batch = SimpleNamespace(batch_size=3)
    backend.forward_metadata = SimpleNamespace(max_q_len=query_length, kv_indices=table)
    sinks = torch.linspace(-1, 1, 16, device="cuda") if with_sinks else None
    output = guarded[1:-1]

    def run():
        aiter_utils.prepare_flydsl_swa_forward_metadata(
            backend, 3, seq_lens, context_includes_query=query_length == 1
        )
        aiter_utils.forward_target_verify_flydsl_swa_5d(
            backend, q, layer, batch, kc, vc, output, sinks
        )

    def check():
        lengths = seq_lens.tolist()
        if query_length != 1:
            lengths = [n + query_length for n in lengths]
        expected = _reference(q, k, v, table, lengths, query_length, 128, sinks, 1024)
        torch.testing.assert_close(output, expected, atol=0.006, rtol=0.04)
        assert torch.all(guarded[[0, -1]] == 19)

    run()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    seq_lens.copy_(torch.tensor([1031, 0, 1091], dtype=seq_lens.dtype, device="cuda"))
    table[:3].copy_(table[:3].roll(1, dims=1))
    graph.replay()
    check()


def test_pa_plan_cache_and_workspace_bounds(monkeypatch):
    backend = _backend(8, 64)
    for batch_size, qlen in ((1, 1), (2, 4), (4, 8)):
        # Use the same large partial buffers for every graph shape.
        backend._flydsl_pa_decode_query_length = qlen
        mx, sums, out, lengths = aiter_utils._get_flydsl_workspace_views(
            backend, batch_size, qlen
        )
        assert out.data_ptr() == backend._flydsl_pa_decode_pout.data_ptr()
        plan = aiter_utils._get_flydsl_work_plan(backend, lengths, 1, 64, qlen)
        assert plan.capacity == batch_size * 64
        again = aiter_utils._get_flydsl_work_plan(backend, lengths, 1, 64, qlen)
        assert again is plan
    with pytest.raises(RuntimeError, match="not initialized for this batch"):
        aiter_utils._get_flydsl_workspace_views(backend, 5, 8)
    with pytest.raises(RuntimeError, match="prepared for query_length"):
        aiter_utils._get_flydsl_workspace_views(backend, 1, 3)
    lengths = backend._flydsl_pa_decode_context_lengths[:3]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="must be warmed"):
        aiter_utils._get_flydsl_work_plan(backend, lengths, 1, 64, 8)


def test_full_pa_empty_batch_does_not_plan_or_launch():
    backend = _backend(4)
    k, v, kc, vc, table, q, guarded = _inputs(4, 1, 192, torch.bfloat16)
    backend.forward_metadata = SimpleNamespace(max_q_len=4, kv_indices=table[:0])
    batch = SimpleNamespace(
        batch_size=0, seq_lens=torch.empty(0, dtype=torch.int64, device="cuda")
    )
    layer = SimpleNamespace(
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        sliding_window_size=-1,
        logit_cap=0.0,
    )
    aiter_utils.forward_target_verify_flydsl_5d(
        backend, q[:0], layer, batch, kc, vc, guarded[:0], None
    )
    assert not backend._flydsl_pa_work_plans


def test_sglang_graph_backend_alternates_batch_shapes_with_shared_scratch():
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    backend = _backend(4, 64)
    backend._compile_flydsl_pa_decode()
    k, v, kc, vc, table, q, guarded = _inputs(4, 1, 192, torch.bfloat16)
    seq_lens = torch.tensor([1, 67, 2081], dtype=torch.int64, device="cuda")
    layer = SimpleNamespace(
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        sliding_window_size=-1,
        logit_cap=0.0,
        scaling=192**-0.5,
    )
    # Exercise the real two-warmup/capture lifecycle; no distributed model is
    # needed to check per-batch plan creation and shared scratch ownership.
    runner = SimpleNamespace(
        device_module=torch.cuda,
        model_runner=SimpleNamespace(tp_group=SimpleNamespace(barrier=lambda: None)),
    )
    graphs = FullCudaGraphBackend(runner)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    batches = {}
    scratch = backend._flydsl_pa_decode_pout.data_ptr()
    try:
        with graphs.capture_session(stream):
            for bs in (3, 2, 1):
                batch = SimpleNamespace(batch_size=bs, seq_lens=seq_lens[:bs])
                batches[bs] = batch
                backend.forward_metadata = SimpleNamespace(
                    max_q_len=4, kv_indices=table[:bs]
                )

                def run(batch=batch):
                    rows = batch.batch_size * 4
                    output = guarded[1 : rows + 1]
                    aiter_utils.forward_target_verify_flydsl_5d(
                        backend, q[:rows], layer, batch, kc, vc, output, None
                    )
                    return output

                graphs.capture_one(bs, run)
        torch.cuda.current_stream().wait_stream(stream)
        cached_plans = len(backend._flydsl_pa_work_plans)
        for bs, state in (
            (1, [1021]),
            (3, [0, 121, 2049]),
            (2, [63, 1091]),
            (3, [11, 17, 2067]),
        ):
            seq_lens[:bs].copy_(torch.tensor(state, dtype=torch.int64, device="cuda"))
            table[:bs].copy_(table[:bs].roll(1, dims=1))
            actual = graphs.replay(bs, batches[bs])
            expected = _reference(
                q[: bs * 4], k, v, table, [n + 4 for n in state], 4, 192
            )
            torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.04)
            assert backend._flydsl_pa_decode_pout.data_ptr() == scratch
            assert len(backend._flydsl_pa_work_plans) == cached_plans
    finally:
        torch.cuda.synchronize()
        graphs.cleanup()


def test_block4_full_pa_high_cache_addresses_and_nan_tail():
    # K extends beyond 4 GiB, as in the serving KV pool. Only two high pages
    # are live, so the Torch reference stays small and checks the real offsets.
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("requires 8 GiB free for the model-size KV address test")
    backend = _backend(4, 16)
    blocks = 2**32 // (192 * 64 * 2) + 2
    k_cache = torch.empty((blocks, 1, 24, 64, 8), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.empty((blocks, 1, 8, 128, 8), dtype=torch.bfloat16, device="cuda")
    assert k_cache.nbytes > 2**32
    table = torch.tensor([[blocks - 2, blocks - 1]], dtype=torch.int32, device="cuda")
    ref_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    torch.manual_seed(5887)
    k = torch.randn((2, 64, 1, 192), dtype=torch.bfloat16, device="cuda") * 0.2
    v = torch.randn((2, 64, 1, 128), dtype=torch.bfloat16, device="cuda") * 0.2
    q = torch.randn((4, 16 * 192), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((4, 16 * 128), dtype=torch.bfloat16, device="cuda")
    lengths = torch.empty(1, dtype=torch.int64, device="cuda")
    batch = SimpleNamespace(batch_size=1, seq_lens=lengths)
    layer = SimpleNamespace(
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        sliding_window_size=-1,
        logit_cap=0.0,
        scaling=192**-0.5,
    )
    backend.forward_metadata = SimpleNamespace(max_q_len=4, kv_indices=table)

    def set_context(length):
        keys, values = k.clone(), v.clone()
        keys[1, length - 64 :] = float("nan")
        values[1, length - 64 :] = float("nan")
        k_cache[-2:].copy_(keys.view(2, 64, 1, 24, 8).permute(0, 2, 3, 1, 4))
        v_cache[-2:].copy_(values.view(2, 8, 8, 1, 128).permute(0, 3, 1, 4, 2))
        lengths.fill_(length - 4)

    def run():
        aiter_utils.forward_target_verify_flydsl_5d(
            backend, q, layer, batch, k_cache, v_cache, output, None
        )

    set_context(65)
    run()
    expected = _reference(q, k, v, ref_table, [65], 4, 192)
    torch.testing.assert_close(output, expected, atol=0.006, rtol=0.04)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for length in (124, 65):
        set_context(length)
        graph.replay()
        expected = _reference(q, k, v, ref_table, [length], 4, 192)
        torch.testing.assert_close(output, expected, atol=0.006, rtol=0.04)
