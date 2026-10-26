"""Dense LoRA wrappers against the legacy backend in eager and graph execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from sglang.srt.lora.backend.triton_backend import (  # noqa: E402
    TritonLoRABackend,
)
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend  # noqa: E402
from sglang.srt.lora.dense.plan import DenseLoraKind  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import reset_context  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="dense LoRA backends need CUDA"
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
SLOTS = 4
POOL_RANK = 32
RANKS = [32, 16, 0, 8]  # slot 2 carries no adapter
SCALINGS = [1.5, 0.5, 1.0, 2.0]
HIDDEN = 256


@pytest.fixture(scope="module", autouse=True)
def _dist():
    """Publish world-size-one groups and the config/rank bundle; reset afterward."""
    import os
    import socket

    from sglang.test.layer_ut_utils import init_single_process_dist

    # Probe on all addresses: a port free on 127.0.0.1 can be busy on another
    # local address (an active connection), and the store binds them all.
    for attempt in range(6):
        with socket.socket() as sock:
            sock.bind(("", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        try:
            init_single_process_dist()
            break
        except Exception as exc:  # torch raises DistNetworkError on EADDRINUSE
            if "EADDRINUSE" not in str(exc) or attempt == 5:
                raise
    yield
    reset_context()


def _pool(stack: int, k: int, width: int, seed: int):
    """Dense pool slots as the memory pool writes them, stale past the rank."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    a = torch.randn(SLOTS, stack * POOL_RANK, k, generator=g, device=DEVICE) * 100
    b = torch.randn(SLOTS, width, POOL_RANK, generator=g, device=DEVICE) * 100
    for slot, rank in enumerate(RANKS):
        a[slot, : stack * rank] = (
            torch.randn(stack * rank, k, generator=g, device=DEVICE) * 0.05
        )
        b[slot, :, :rank] = torch.randn(width, rank, generator=g, device=DEVICE) * 0.05
    return a.to(DTYPE).contiguous(), b.to(DTYPE).contiguous()


def _batch(seq_lens, slots, decode: bool):
    seg_lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    fb = SimpleNamespace(
        forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        extend_seq_lens=seg_lens,
        extend_seq_lens_cpu=list(seq_lens),
        extend_num_tokens=int(sum(seq_lens)),
        spec_info=None,
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
    )
    return fb, list(slots), list(RANKS), list(SCALINGS)


def _layers(kind: str):
    from sglang.srt.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from sglang.srt.lora.layers import get_lora_layer

    common = dict(bias=False, params_dtype=DTYPE)
    if kind == "qkv":
        base = QKVParallelLinear(HIDDEN, 32, 8, 2, **common)
    elif kind == "merged":
        base = MergedColumnParallelLinear(HIDDEN, [96, 96, 160, 160], **common)
    elif kind == "row":
        base = RowParallelLinear(512, HIDDEN, reduce_results=False, **common)
    elif kind == "replicated":
        base = ReplicatedLinear(HIDDEN, 320, **common)
    else:
        base = ColumnParallelLinear(HIDDEN, 192, **common)
    base = base.to(DEVICE)
    with torch.no_grad():
        base.weight.normal_(0, 0.02)
    k = base.weight.shape[1]
    width = base.weight.shape[0]
    stack = {"qkv": 3, "merged": 4, "replicated": 2}.get(kind, 1)
    a, b = _pool(stack, k, width, seed=sum(kind.encode()))

    def wrap(backend):
        layer = get_lora_layer(base, backend)
        if kind == "replicated":
            layer.first_output_dim = 128
        layer.set_lora_info(a, b)
        return layer

    return wrap, k


def _prepare_batch(backend, fb, wi, ranks, scal, *args, **kwargs):
    """Fill backend metadata and record whether any request carries an adapter."""
    backend.prepare_lora_batch(fb, wi, ranks, scal, *args, **kwargs)
    backend.batch_info.has_active_lora = any(ranks[i] > 0 for i in wi)


def _prepare(
    backend, batch, use_decode_cuda_graph=False, *, use_prefill_cuda_graph=False
):
    fb, wi, ranks, scal = batch
    _prepare_batch(
        backend, fb, wi, ranks, scal, use_decode_cuda_graph, use_prefill_cuda_graph
    )


@pytest.mark.parametrize("kind", ["qkv", "merged", "row", "replicated", "column"])
@pytest.mark.parametrize("batch_name", ["decode13", "prefill4", "prefill_one_request"])
def test_wrappers_match_legacy(kind, batch_name):
    wrap, k = _layers(kind)
    if batch_name == "decode13":
        batch = _batch([1] * 13, [i % SLOTS for i in range(13)], decode=True)
    elif batch_name == "prefill4":
        batch = _batch([5, 17, 1, 9], [1, 0, 2, 3], decode=False)
    else:
        batch = _batch([23], [3], decode=False)
    tokens = batch[0].extend_num_tokens
    x = torch.randn(tokens, k, device=DEVICE, dtype=DTYPE)

    legacy = TritonLoRABackend(SLOTS, DEVICE)
    _prepare(legacy, batch)
    expected = wrap(legacy)(x)[0]

    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    _prepare(ours, batch)
    out = wrap(ours)(x)[0]
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)


class _FixedPlanTable:
    def __init__(self, plan):
        self.plan = plan

    def plan_for(self, kind, phase, num_tokens, *geometry):
        return self.plan


def _cached_route_keys(runner):
    return {key[-3:] for routes in runner.workspace.routes.values() for key in routes}


def test_eager_target_verify_batch_uses_the_draft_width():
    """TARGET_VERIFY uses draft-width segments without extend lengths."""
    wrap_col, k_col = _layers("column")
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    layer = wrap_col(ours)
    draft, bs, slots = 8, 3, [0, 1, 3]
    verify = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=bs,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_num_tokens=None,
        spec_info=SimpleNamespace(draft_token_num=draft),
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
    )
    x = torch.randn(bs * draft, k_col, device=DEVICE, dtype=DTYPE)
    _prepare_batch(
        ours, verify, slots, list(RANKS), list(SCALINGS), use_decode_cuda_graph=False
    )
    info = ours.batch_info
    assert info.seg_indptr.diff().tolist() == [draft] * bs
    assert info.seg_indptr[: bs + 1].tolist() == [0, draft, 2 * draft, 3 * draft]
    verified = layer(x)[0].clone()
    _prepare(ours, _batch([draft] * bs, slots, decode=False))
    torch.testing.assert_close(verified, layer(x)[0], rtol=0, atol=0)


@pytest.mark.parametrize("request_capacity", [None, 48])
def test_prefill_graph_admits_any_request_count_within_the_token_bucket(
    request_capacity,
):
    """Replay ragged requests with the captured bucket's plan and route storage."""
    from sglang.kernels.ops.lora.common.routing import _aligned_route_capacity
    from sglang.srt.lora.dense.plan import DensePlan

    wrap_row, k_row = _layers("row")
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    ours.init_prefill_cuda_graph_batch_info(512, request_capacity)
    assert ours.prefill_cuda_graph_max_bs == (request_capacity or 512)
    table = _TokensPlanTable(
        DensePlan(block_size=32), DensePlan(block_size=16), threshold=128
    )
    ours.runner._plan_tables[POOL_RANK] = table
    layer = wrap_row(ours)
    bucket = 256
    x = torch.randn(bucket, k_row, device=DEVICE, dtype=DTYPE)
    out = torch.empty(
        bucket, layer.base_layer.weight.shape[0], device=DEVICE, dtype=DTYPE
    )

    def forward():
        ours.reset_routing_cache()
        out.copy_(layer(x)[0])

    def route_buffers(route):
        # The route pads per slot, so its capacity ignores the request count.
        assert (
            bucket
            <= route.sorted_pair_ids.numel()
            <= _aligned_route_capacity(bucket, route.block_size, SLOTS)
        )
        ids = route.sorted_pair_ids[: route.num_pairs_post_padded.item()]
        valid_ids = ids[(ids >= 0) & (ids < bucket)]
        active_tokens = torch.where(ours.runner.token_slots >= 0)[0]
        assert torch.isin(active_tokens, valid_ids).all().item()
        return (
            route.sorted_pair_ids.data_ptr(),
            route.sorted_pair_ids.numel(),
            route.block_bucket_ids.data_ptr(),
        )

    capture = _batch([bucket], [1], decode=False)
    _prepare(ours, capture, use_prefill_cuda_graph=True)
    assert not ours.runner.workspace.routes
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
        warmup_route = ours.runner.route(table.large.block_size)
    torch.cuda.current_stream().wait_stream(stream)
    key = ("sorted", table.large.block_size, SLOTS)
    assert _cached_route_keys(ours.runner) == {key}
    buffers = route_buffers(warmup_route)
    _prepare(ours, capture, use_prefill_cuda_graph=True)
    assert not ours.runner.workspace.routes
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        forward()
        captured_route = ours.runner.route(table.large.block_size)
    assert _cached_route_keys(ours.runner) == {key}

    legacy = TritonLoRABackend(SLOTS, DEVICE)
    seq_lens = [
        1 + (i * 7) % 9 for i in range(48)
    ]  # 48 ragged requests, 1..9 tokens each
    slots = [(i * 3) % SLOTS for i in range(48)]
    live = sum(seq_lens)
    assert len(seq_lens) > 32 and live <= bucket
    for batch in (
        _batch(seq_lens, slots, decode=False),
        _batch([90, 70, 50, 30], [2, 0, 3, 1], decode=False),
        _batch([60, 40], [3, 0], decode=False),
    ):
        live = batch[0].extend_num_tokens
        x.copy_(torch.randn_like(x))
        _prepare(ours, batch, use_prefill_cuda_graph=True)
        assert ours.runner.token_slots.numel() == live
        assert not ours.runner.workspace.routes
        captured_route.sorted_pair_ids.fill_(-7)
        captured_route.block_bucket_ids.fill_(-7)
        captured_route.num_pairs_post_padded.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert route_buffers(captured_route) == buffers
        _prepare(legacy, batch)
        expected = wrap_row(legacy)(x[:live])[0]
        torch.testing.assert_close(out[:live], expected, rtol=2e-2, atol=2e-2)
    _prepare(ours, batch)
    site = (DenseLoraKind.LINEAR, POOL_RANK, k_row, layer.base_layer.weight.shape[0])
    eager = ours.runner.plan_for(*site, num_tokens=live)
    assert eager.block_size != captured_route.block_size


class _TokensPlanTable:
    def __init__(self, large, small, threshold):
        self.large, self.small, self.threshold = large, small, threshold
        self.queries = []

    def plan_for(self, kind, phase, num_tokens, *geometry):
        self.queries.append(num_tokens)
        return self.large if num_tokens > self.threshold else self.small


@pytest.mark.parametrize("prefill", [False, True], ids=["decode", "prefill"])
def test_variable_bucket_graphs_replay_without_dense_host_work(monkeypatch, prefill):
    """Old graphs read refreshed metadata without Python planning or routing.

    Each graph holds a qkv and a row site, and capture rebuilds the warm-up
    route exactly once, inside the graph.
    """
    from sglang.srt.lora.dense.plan import DensePlan

    wrap_qkv, k_qkv = _layers("qkv")
    wrap_row, k_row = _layers("row")
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    table = _TokensPlanTable(
        DensePlan(block_size=32),
        DensePlan(block_size=16),
        threshold=16,
    )
    ours.runner._plan_tables[POOL_RANK] = table
    if prefill:
        ours.init_prefill_cuda_graph_batch_info(64)
    else:
        ours.init_decode_cuda_graph_batch_info(64, 1)
    layer_qkv, layer_row = wrap_qkv(ours), wrap_row(ours)
    x_qkv = torch.randn(64, k_qkv, device=DEVICE, dtype=DTYPE)
    x_row = torch.randn(64, k_row, device=DEVICE, dtype=DTYPE)
    out_qkv = torch.empty(
        64, layer_qkv.base_layer.weight.shape[0], device=DEVICE, dtype=DTYPE
    )
    out_row = torch.empty(
        64, layer_row.base_layer.weight.shape[0], device=DEVICE, dtype=DTYPE
    )
    build_route = ours.runner._build_route
    builds = []  # per route build: was the current stream capturing?

    def counted_build(*args, **kwargs):
        builds.append(torch.cuda.is_current_stream_capturing())
        return build_route(*args, **kwargs)

    monkeypatch.setattr(ours.runner, "_build_route", counted_build)

    def batch_for(live, offset):
        lengths = [live // 2, live - live // 2] if prefill else [1] * live
        slots = [(i + offset) % SLOTS for i in range(len(lengths))]
        return _batch(lengths, slots, decode=not prefill)

    def prepare(batch):
        _prepare(ours, batch, not prefill, use_prefill_cuda_graph=prefill)

    graphs = {}
    stream = torch.cuda.Stream()  # one stream for every bucket, as in a graph runner
    for bucket in (64, 16):

        def forward(bucket=bucket):
            ours.reset_routing_cache()
            out_qkv[:bucket].copy_(layer_qkv(x_qkv[:bucket])[0])
            out_row[:bucket].copy_(layer_row(x_row[:bucket])[0])

        queries = len(table.queries)
        prepare(batch_for(11, 0))
        assert len(table.queries) == queries
        assert ours.runner._token_slots.numel() == 64
        assert ours.runner.token_slots.numel() == 11
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        count = len(builds)
        graphs[bucket] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[bucket], stream=stream):
            forward()
        # The warm-up route is rebuilt exactly once, inside the capture.
        assert builds[count:] == [True]
        assert ours.runner._token_slots.numel() == 64
        assert ours.runner.token_slots.numel() == bucket
        plan = table.large if bucket > table.threshold else table.small
        assert _cached_route_keys(ours.runner) == {("sorted", plan.block_size, SLOTS)}
    assert table.queries == [64] * 6 + [16] * 6  # one plan per site per forward

    def unexpected_dense_work(*args, **kwargs):
        pytest.fail("preparation and replay must not plan or build dense routes")

    monkeypatch.setattr(ours.runner, "plan_for", unexpected_dense_work)
    monkeypatch.setattr(ours.runner, "_build_route", unexpected_dense_work)
    legacy = TritonLoRABackend(SLOTS, DEVICE)
    # The same local size can replay either bucket, including DP-wide padding.
    for offset, (live, bucket) in enumerate(
        ((11, 64), (11, 16), (40, 64), (16, 16), (64, 64))
    ):
        x_qkv.copy_(torch.randn_like(x_qkv))
        x_row.copy_(torch.randn_like(x_row))
        batch = batch_for(live, offset + 1)
        prepare(batch)
        assert ours.runner.token_slots.numel() == live
        assert not ours.runner.workspace.routes
        graphs[bucket].replay()
        torch.cuda.synchronize()
        assert not ours.runner.workspace.routes
        _prepare(legacy, batch)
        expected_qkv = wrap_qkv(legacy)(x_qkv[:live])[0]
        expected_row = wrap_row(legacy)(x_row[:live])[0]
        torch.testing.assert_close(out_qkv[:live], expected_qkv, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_row[:live], expected_row, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 64])  # a partial and a full route block
def test_decode_lm_head_matches_legacy(tokens):
    # Decode uses the main runner's generic plan; slot 2 has no adapter.
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.lora.layers import get_lora_layer

    vocab, embed = 4096, 2048
    batch = _batch([1] * tokens, [i % SLOTS for i in range(tokens)], decode=True)
    g = torch.Generator(device=DEVICE).manual_seed(5)
    a_head = torch.randn(SLOTS, POOL_RANK, embed, generator=g, device=DEVICE) * 0.05
    b_head = torch.randn(SLOTS, vocab, POOL_RANK, generator=g, device=DEVICE) * 0.05
    for slot, rank in enumerate(RANKS):
        a_head[slot, rank:] = 1e4  # stale past the rank
        b_head[slot, :, rank:] = 1e4
    a_head, b_head = a_head.to(DTYPE).contiguous(), b_head.to(DTYPE).contiguous()
    base_head = ParallelLMHead(vocab, embed, params_dtype=DTYPE).to(DEVICE)
    with torch.no_grad():
        base_head.weight.normal_(0, 0.02, generator=g)
    hidden = torch.randn(tokens, embed, generator=g, device=DEVICE, dtype=DTYPE)

    def run(backend):
        backend.validate_lora_targets(base_head, {"lm_head"})
        _prepare(backend, batch)
        head = get_lora_layer(base_head, backend)
        head.set_lora_info(a_head, b_head)
        return head(hidden)

    expected = run(TritonLoRABackend(SLOTS, DEVICE))
    out = run(TritonV2LoRABackend(SLOTS, DEVICE))
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_embedding_and_lm_head_match_legacy():
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from sglang.srt.lora.layers import get_lora_layer

    vocab, embed = 1000, HIDDEN
    batch = _batch([5, 17, 1, 9], [1, 0, 2, 3], decode=False)
    tokens = batch[0].extend_num_tokens
    g = torch.Generator(device=DEVICE).manual_seed(3)
    ids = torch.randint(0, vocab, (tokens,), generator=g, device=DEVICE)
    a_emb = torch.randn(
        SLOTS, POOL_RANK, vocab, generator=g, device=DEVICE, dtype=torch.float32
    )
    b_emb = torch.randn(
        SLOTS, embed, POOL_RANK, generator=g, device=DEVICE, dtype=torch.float32
    )
    a_head = torch.randn(SLOTS, POOL_RANK, embed, generator=g, device=DEVICE) * 0.05
    b_head = torch.randn(SLOTS, vocab, POOL_RANK, generator=g, device=DEVICE) * 0.05
    for slot, rank in enumerate(RANKS):
        a_emb[slot, rank:] = 0  # the gather kernel keeps the tail zero-filled
        a_emb[slot, :rank] *= 0.05
        b_emb[slot, :, rank:] = 1e4  # stale
        b_emb[slot, :, :rank] *= 0.05
        a_head[slot, rank:] = 1e4
        b_head[slot, :, rank:] = 1e4
    a_emb, b_emb, a_head, b_head = (
        t.to(DTYPE).contiguous() for t in (a_emb, b_emb, a_head, b_head)
    )

    base_embed = VocabParallelEmbedding(vocab, embed, params_dtype=DTYPE).to(DEVICE)
    base_head = ParallelLMHead(vocab, embed, params_dtype=DTYPE).to(DEVICE)
    with torch.no_grad():
        base_embed.weight.normal_(0, 0.02, generator=g)
        base_head.weight.copy_(base_embed.weight)

    def run(backend):
        # Both backends wrap the same base layers, so only the LoRA path differs.
        backend.validate_lora_targets(base_head, {"lm_head", "embed_tokens"})
        _prepare(backend, batch)
        emb = get_lora_layer(base_embed, backend)
        emb.set_lora_info(None, a_emb, b_emb)
        hidden = emb(ids)
        head = get_lora_layer(base_head, backend)
        head.set_lora_info(a_head, b_head)
        pruned = hidden[[4, 21, 22, 31]]  # last token of each request
        logits = head(pruned)
        return hidden, logits

    hidden_l, logits_l = run(TritonLoRABackend(SLOTS, DEVICE))
    hidden_v, logits_v = run(TritonV2LoRABackend(SLOTS, DEVICE))
    torch.testing.assert_close(hidden_v.float(), hidden_l.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(logits_v.float(), logits_l.float(), rtol=2e-2, atol=2e-2)


def test_pruned_chunked_lm_head_numeric_and_single_binding(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.lora.dense.plan import DensePlan, Overlap
    from sglang.srt.lora.layers import get_lora_layer

    monkeypatch.setenv("SGLANG_ENABLE_LOGPROB_CHUNK", "true")
    monkeypatch.setenv("SGLANG_LOGPROB_CHUNK_SIZE", "4")
    assert envs.SGLANG_ENABLE_LOGPROB_CHUNK.get()
    backend = TritonV2LoRABackend(SLOTS, DEVICE)
    backend.validate_lora_targets(None, {"lm_head"})
    backend.lm_head_runner._plan_tables[POOL_RANK] = _FixedPlanTable(
        DensePlan(overlap=Overlap.NONE)  # as the shipped vocab rows
    )
    vocab = 128
    base = ParallelLMHead(vocab, HIDDEN, params_dtype=DTYPE).to(DEVICE)
    with torch.no_grad():
        base.weight.normal_(0, 0.02)
    head = get_lora_layer(base, backend)
    a, b = _pool(1, HIDDEN, vocab, 117)
    head.set_lora_info(a, b)
    for assignments, ranks, scales in (
        ([1, 1, 3], RANKS, SCALINGS),
        ([0, 2, 1], [8, 0, 0, 0], [0.25, 0, 0, 0]),
    ):
        fb, _, _, _ = _batch([5, 3, 4], assignments, False)
        fb.return_logprob = True
        fb.extend_logprob_start_lens_cpu = [2, 1, 1]
        _prepare_batch(backend, fb, assignments, ranks, scales, False)
        token_slots = [
            slot for slot, n in zip(assignments, [3, 2, 3]) for _ in range(n)
        ]
        hidden = torch.randn(8, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.05
        expected = torch.nn.functional.linear(hidden, base.weight).float()
        for i, slot in enumerate(token_slots):
            rank = ranks[slot]
            if rank:
                bridge = (hidden[i].float() @ a[slot, :rank].float().T).to(DTYPE)
                expected[i] += (bridge.float() @ b[slot, :, :rank].float().T) * scales[
                    slot
                ]
        assert len(backend.lm_head_pass_batch_infos) == 2
        for idx in range(2):
            head.set_lm_head_pass(idx)
            result = head(hidden[idx * 4 : (idx + 1) * 4])
            torch.testing.assert_close(
                result.float(),
                expected[idx * 4 : (idx + 1) * 4],
                rtol=2e-2,
                atol=2e-2,
            )
        head.reset_lm_head_pass()
        result = head(hidden)
        torch.testing.assert_close(result.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "site,overlap",
    # ab_delta guards the embedding's integer-input delta dtype; lm_head rows use none
    [
        (site, overlap)
        for site in ("embedding_decode", "embedding_prefill")
        for overlap in ("none", "ab_delta")
    ]
    + [("head_decode", "none")],
)
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
def test_vocab_layers_use_unified_forward(site, overlap, graph_mode):
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from sglang.srt.lora.dense.plan import DensePlan, Overlap
    from sglang.srt.lora.layers import get_lora_layer

    embedding = site.startswith("embedding")
    prefill = site.endswith("prefill")
    # TP1 only: an uneven vocab gives lm_head a padded base output wider than
    # its LoRA B. This checks the padding boundary, not multi-rank collectives.
    vocab, width = 131, HIDDEN
    backend = TritonV2LoRABackend(SLOTS, DEVICE)
    backend.validate_lora_targets(None, {"embed_tokens" if embedding else "lm_head"})
    backend.runner._plan_tables[POOL_RANK] = _FixedPlanTable(
        DensePlan(overlap=Overlap(overlap))
    )
    if graph_mode:
        if prefill:
            backend.init_prefill_cuda_graph_batch_info(64)
        else:
            backend.init_decode_cuda_graph_batch_info(64, 1)
    base_type = VocabParallelEmbedding if embedding else ParallelLMHead
    base = base_type(vocab, width, params_dtype=DTYPE).to(DEVICE)
    with torch.no_grad():
        base.weight.normal_(0, 0.02)
    layer = get_lora_layer(base, backend)
    a, b = _pool(1, vocab if embedding else width, width if embedding else vocab, 219)
    if embedding:
        layer.set_lora_info(None, a, b)
        x = torch.randint(vocab, (64,), device=DEVICE)
    else:
        layer.set_lora_info(a, b)
        x = torch.randn(64, width, device=DEVICE, dtype=DTYPE) * 0.05
    output_width = width if embedding else base.weight.shape[0]
    if not embedding:
        assert output_width > vocab
        assert layer.lora_offsets == (0, vocab)
    output = torch.empty(64, output_width, device=DEVICE, dtype=DTYPE)

    def prepare(lengths, slots, ranks, scales):
        fb, _, _, _ = _batch(lengths, slots, not prefill)
        _prepare_batch(
            backend,
            fb,
            slots,
            ranks,
            scales,
            graph_mode and not prefill,
            graph_mode and prefill,
        )

    def forward(bucket):
        backend.reset_routing_cache()
        output[:bucket].copy_(layer(x[:bucket]))

    graphs = {}
    stream = torch.cuda.Stream()  # one stream for every bucket, as in a graph runner
    for bucket in (64, 16):
        lengths = [bucket // 2, bucket // 2] if prefill else [1] * bucket
        slots = [i % SLOTS for i in range(len(lengths))]
        prepare(lengths, slots, RANKS, SCALINGS)
        forward(bucket)
        if graph_mode:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward(bucket)
            torch.cuda.current_stream().wait_stream(stream)
            prepare(lengths, slots, RANKS, SCALINGS)
            graphs[bucket] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[bucket], stream=stream):
                forward(bucket)

    for step, bucket in enumerate((64, 16, 64, 16)):
        lengths = [5, 0, 3, 1] if prefill else [1] * 9
        slots = [1, 0, 2, 3] if prefill else [1, 0, 2, 3, 1, 3, 0, 2, 1]
        ranks = [8, 4, 0, 0] if step < 2 else [0] * SLOTS
        scales = [0.25, 1.25, 0, 0.5]
        if embedding:
            x.random_(vocab)
        else:
            x.normal_(0, 0.05)
        prepare(lengths, slots, ranks, scales)
        extent = bucket if graph_mode else sum(lengths)
        if graph_mode:
            graphs[bucket].replay()
        else:
            forward(extent)
        if embedding:
            expected = torch.nn.functional.embedding(x[:extent], base.weight).float()
        else:
            expected = torch.nn.functional.linear(x[:extent], base.weight).float()
        row = 0
        for length, slot in zip(lengths, slots):
            rank = ranks[slot]
            if rank and length:
                values = x[row : row + length]
                if embedding:
                    bridge = a[slot, :rank, values].T
                else:
                    bridge = (values.float() @ a[slot, :rank].float().T).to(DTYPE)
                expected[row : row + length, : b.shape[-2]] += (
                    bridge.float() @ b[slot, :, :rank].float().T
                ) * scales[slot]
            row += length
        torch.testing.assert_close(
            output[:extent].float(), expected, rtol=2e-2, atol=2e-2
        )
        if not embedding:
            torch.testing.assert_close(
                output[:extent, vocab:].float(),
                expected[:, vocab:],
                rtol=0,
                atol=0,
            )
        assert output.dtype == DTYPE


@pytest.mark.parametrize("verify", [False, True])
def test_graph_phases_replay_with_changed_ranks_scales_and_old_buckets(verify):
    wrap, k = _layers("column")
    backend = TritonV2LoRABackend(SLOTS, DEVICE)
    backend.init_prefill_cuda_graph_batch_info(64)
    width = 4 if verify else 1
    backend.init_decode_cuda_graph_batch_info(64 // width, width)
    layer = wrap(backend)
    x = torch.randn(64, k, device=DEVICE, dtype=DTYPE) * 0.05
    output = torch.empty(64, 192, device=DEVICE, dtype=DTYPE)

    def prepare(prefill, lengths, slots, ranks, scales):
        fb, _, _, _ = _batch(lengths, slots, not prefill)
        if verify and not prefill:
            fb.forward_mode = ForwardMode.TARGET_VERIFY
            fb.spec_info = SimpleNamespace(draft_token_num=width)
        _prepare_batch(backend, fb, slots, ranks, scales, not prefill, prefill)

    graphs = {}
    stream = torch.cuda.Stream()  # one stream for every bucket, as in a graph runner
    for prefill in (False, True):
        for bucket in (64, 32):
            lengths = [bucket] if prefill else [width] * (bucket // width)
            slots = [i % SLOTS for i in range(len(lengths))]
            prepare(prefill, lengths, slots, RANKS, SCALINGS)

            def forward():
                backend.reset_routing_cache()
                output[:bucket].copy_(layer(x[:bucket])[0])

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            torch.cuda.current_stream().wait_stream(stream)
            prepare(prefill, lengths, slots, RANKS, SCALINGS)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            graphs[prefill, bucket] = graph

    for step, (prefill, bucket) in enumerate(
        ((False, 64), (True, 32), (False, 32), (True, 64)) * 2
    ):
        lengths = [5, 0, 9] if prefill else [width] * 3
        slots = [3, 1, 0] if prefill else [1, 0, 3]
        ranks = [8, 4, 0, 0] if step % 3 else [0] * SLOTS
        scales = [0.25, 1.25, 0.0, 0.5]
        x.normal_(0, 0.05)
        prepare(prefill, lengths, slots, ranks, scales)
        graphs[prefill, bucket].replay()
        expected = torch.nn.functional.linear(
            x[:bucket], layer.base_layer.weight
        ).float()
        row = 0
        for length, slot in zip(lengths, slots):
            rank = ranks[slot]
            if rank:
                bridge = (
                    x[row : row + length].float()
                    @ layer.A_buffer[slot, :rank].float().T
                ).to(DTYPE)
                expected[row : row + length] += (
                    bridge.float() @ layer.B_buffer[slot, :, :rank].float().T
                ) * scales[slot]
            row += length
        torch.testing.assert_close(
            output[:bucket].float(), expected, rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
