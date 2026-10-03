"""Dense LoRA wrappers against the legacy backend in eager and graph execution."""

from __future__ import annotations

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from sglang.srt.lora.backend.triton_backend import (  # noqa: E402
    TritonLoRABackend,
)
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend  # noqa: E402
from sglang.srt.lora.dense.plan import DenseLoraKind  # noqa: E402
from sglang.srt.lora.utils import Phase  # noqa: E402
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
    a, b = _pool(stack, k, width, seed=hash(kind) % 1000)

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


def _prepare(backend, batch, use_decode_cuda_graph=False):
    fb, wi, ranks, scal = batch
    _prepare_batch(backend, fb, wi, ranks, scal, use_decode_cuda_graph)


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


@pytest.mark.parametrize("prefill", [False, True], ids=["decode", "prefill"])
def test_forward_shape_selects_plan_with_full_capacity_metadata(prefill):
    """The executing layer supplies the extent; metadata does not predict a bucket."""
    from sglang.srt.lora.dense.plan import DensePlan

    wrap_row, k_row = _layers("row")
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    table = _TokensPlanTable(
        DensePlan(block_size=32), DensePlan(block_size=16), threshold=16
    )
    ours.runner._plan_tables[POOL_RANK] = table
    if prefill:
        ours.init_prefill_cuda_graph_batch_info(64)
    else:
        ours.init_decode_cuda_graph_batch_info(64, 1)
    layer = wrap_row(ours)
    for live, bucket in ((3, 8), (11, 32), (17, 64), (5, 8)):
        lengths = [live] if prefill else [1] * live
        batch = _batch(lengths, [i % SLOTS for i in range(len(lengths))], not prefill)
        fb, wi, ranks, scal = batch
        queries = len(table.queries)
        _prepare_batch(
            ours,
            fb,
            wi,
            ranks,
            scal,
            use_decode_cuda_graph=not prefill,
            use_prefill_cuda_graph=prefill,
        )
        assert len(table.queries) == queries
        assert ours.runner._token_slots.numel() == 64
        assert ours.runner.token_slots.numel() == live
        layer(torch.randn(bucket, k_row, device=DEVICE, dtype=DTYPE))
        assert table.queries[queries:] == [bucket]
        assert ours.runner.num_tokens == bucket
        plan = table.large if bucket > table.threshold else table.small
        assert _cached_route_keys(ours.runner) == {("sorted", plan.block_size, SLOTS)}


@pytest.mark.parametrize("request_capacity", [None, 4])
def test_prefill_graph_capture_and_replay_after_new_batch(request_capacity):
    """Replay changed request lengths, slots, and live-token counts against legacy."""
    wrap_row, k_row = _layers("row")
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    ours.init_prefill_cuda_graph_batch_info(512, request_capacity)
    layer = wrap_row(ours)
    bucket = 256
    x = torch.randn(bucket, k_row, device=DEVICE, dtype=DTYPE)
    out = torch.empty(
        bucket, layer.base_layer.weight.shape[0], device=DEVICE, dtype=DTYPE
    )

    def forward():
        ours.reset_routing_cache()
        out.copy_(layer(x)[0])

    capture = _batch([100, 60, 40], [0, 1, 3], decode=False)  # 200 tokens: bucket 256
    fb, wi, ranks, scal = capture
    _prepare_batch(
        ours,
        fb,
        wi,
        ranks,
        scal,
        use_decode_cuda_graph=False,
        use_prefill_cuda_graph=True,
    )
    assert ours.runner.token_slots.numel() == 200
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    _prepare_batch(
        ours,
        fb,
        wi,
        ranks,
        scal,
        use_decode_cuda_graph=False,
        use_prefill_cuda_graph=True,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    assert ours.runner.token_slots.numel() == bucket

    legacy = TritonLoRABackend(SLOTS, DEVICE)
    for seq_lens, slots in (([120, 70], [3, 1]), ([30, 90, 50, 20], [1, 0, 3, 2])):
        live = sum(seq_lens)
        x.copy_(torch.randn_like(x))
        batch = _batch(seq_lens, slots, decode=False)
        fb, wi, ranks, scal = batch
        _prepare_batch(
            ours,
            fb,
            wi,
            ranks,
            scal,
            use_decode_cuda_graph=False,
            use_prefill_cuda_graph=True,
        )
        assert ours.runner.token_slots.numel() == live
        graph.replay()
        torch.cuda.synchronize()
        _prepare(legacy, batch)
        expected = wrap_row(legacy)(x[:live])[0]
        torch.testing.assert_close(out[:live], expected, rtol=2e-2, atol=2e-2)


def test_prefill_graph_replay_builds_the_route_its_bucket_selects():
    """Routing runs in the graph with its bucket's plan, not the live-size plan."""
    from sglang.srt.lora.dense.plan import DensePlan

    wrap_row, k_row = _layers("row")
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    ours.runner._plan_tables[POOL_RANK] = _TokensPlanTable(
        DensePlan(block_size=32),
        DensePlan(block_size=16),
        threshold=512,
    )
    ours.init_prefill_cuda_graph_batch_info(1024)
    layer = wrap_row(ours)
    bucket = 1024
    x = torch.randn(bucket, k_row, device=DEVICE, dtype=DTYPE)
    out = torch.empty(
        bucket, layer.base_layer.weight.shape[0], device=DEVICE, dtype=DTYPE
    )

    def forward():
        ours.reset_routing_cache()
        out.copy_(layer(x)[0])

    def prepare(batch, graph):
        fb, wi, ranks, scal = batch
        _prepare_batch(
            ours,
            fb,
            wi,
            ranks,
            scal,
            use_decode_cuda_graph=False,
            use_prefill_cuda_graph=graph,
        )

    capture = _batch([1024], [1], decode=False)  # one request of the whole bucket
    prepare(capture, True)
    assert ours.runner.token_slots.numel() == bucket
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    site = (DenseLoraKind.LINEAR, POOL_RANK, k_row, layer.base_layer.weight.shape[0])
    captured_plan = ours.runner.plan_for(*site, num_tokens=bucket)
    captured_key = ("sorted", captured_plan.block_size)
    prepare(capture, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
        captured_route = ours.runner.route(captured_plan.block_size)

    legacy = TritonLoRABackend(SLOTS, DEVICE)
    for seq_lens, slots in (([300, 212], [3, 0]), ([100, 100, 100, 212], [1, 3, 0, 1])):
        live = sum(seq_lens)  # 512: half the bucket
        x.copy_(torch.randn_like(x))
        batch = _batch(seq_lens, slots, decode=False)
        prepare(batch, True)
        assert not ours.runner.workspace.routes
        assert ours.runner.token_slots.numel() == live
        captured_route.sorted_pair_ids.fill_(-7)
        captured_route.block_bucket_ids.fill_(-7)
        captured_route.num_pairs_post_padded.zero_()
        graph.replay()
        torch.cuda.synchronize()
        _prepare(legacy, batch)
        expected = wrap_row(legacy)(x[:live])[0]
        torch.testing.assert_close(out[:live], expected, rtol=2e-2, atol=2e-2)
    # Eager execution at the live extent selects the smaller plan.
    prepare(_batch([300, 212], [3, 0], decode=False), False)
    eager = ours.runner.plan_for(*site, num_tokens=512)
    assert eager.block_size != captured_key[1]


def test_graph_workspace_phase_follows_buffers_not_target_verify_plans():
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    ours.init_prefill_cuda_graph_batch_info(8)
    ours.init_decode_cuda_graph_batch_info(2, 4)
    fb, wi, ranks, scal = _batch([4, 4], [0, 1], decode=False)
    workspace = ours.lora_workspace
    buffers = {}
    for is_prefill_graph in (True, False, True):
        fb.forward_mode = (
            ForwardMode.EXTEND if is_prefill_graph else ForwardMode.TARGET_VERIFY
        )
        fb.spec_info = SimpleNamespace(draft_token_num=4)
        _prepare_batch(
            ours,
            fb,
            wi,
            ranks,
            scal,
            use_decode_cuda_graph=not is_prefill_graph,
            use_prefill_cuda_graph=is_prefill_graph,
        )
        assert ours.runner.phase is Phase.PREFILL
        scratch = workspace.tensor("phase_probe", (8,), dtype=DTYPE, device=DEVICE)
        key = ("phase_probe", DTYPE, DEVICE, is_prefill_graph)
        assert workspace._graph_storage[key].data_ptr() == scratch.data_ptr()
        assert (
            buffers.setdefault(is_prefill_graph, scratch.data_ptr())
            == scratch.data_ptr()
        )
    assert buffers[True] != buffers[False]


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
    fb, wi, ranks, scal = _batch([draft] * bs, slots, decode=False)
    _prepare_batch(ours, fb, wi, ranks, scal, use_decode_cuda_graph=False)
    torch.testing.assert_close(verified, layer(x)[0], rtol=0, atol=0)


@pytest.mark.parametrize("request_capacity", [None, 48])
def test_prefill_graph_admits_any_request_count_within_the_token_bucket(
    request_capacity,
):
    """Request counts change route data, never the route capacity."""
    from sglang.srt.lora.dense.plan import DensePlan
    from sglang.srt.lora.kernels.routing import _aligned_route_capacity

    wrap_row, k_row = _layers("row")
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    ours.init_prefill_cuda_graph_batch_info(512, request_capacity)
    assert ours.prefill_cuda_graph_max_bs == (request_capacity or 512)
    original_plan_for = ours.runner.plan_for

    def selected_plan(*args, **kwargs):
        original_plan_for(*args, **kwargs)
        return DensePlan(block_size=16)

    ours.runner.plan_for = selected_plan
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

    capture = _batch([100, 60, 40], [0, 1, 3], decode=False)
    fb, wi, ranks, scal = capture
    _prepare_batch(
        ours,
        fb,
        wi,
        ranks,
        scal,
        use_decode_cuda_graph=False,
        use_prefill_cuda_graph=True,
    )
    assert not ours.runner.workspace.routes
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
        warmup_route = ours.runner.route(16)
    torch.cuda.current_stream().wait_stream(stream)
    key = ("sorted", 16, SLOTS)
    assert _cached_route_keys(ours.runner) == {key}
    buffers = route_buffers(warmup_route)
    _prepare_batch(
        ours,
        fb,
        wi,
        ranks,
        scal,
        use_decode_cuda_graph=False,
        use_prefill_cuda_graph=True,
    )
    assert not ours.runner.workspace.routes
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
        captured_route = ours.runner.route(16)
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
    ):
        fb, wi, ranks, scal = batch
        live = fb.extend_num_tokens
        x.copy_(torch.randn_like(x))
        _prepare_batch(
            ours,
            fb,
            wi,
            ranks,
            scal,
            use_decode_cuda_graph=False,
            use_prefill_cuda_graph=True,
        )
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


def test_decode_graph_capture_and_replay_after_new_batch():
    wrap_qkv, k_qkv = _layers("qkv")
    wrap_row, k_row = _layers("row")
    max_bs = 16
    ours = TritonV2LoRABackend(SLOTS, DEVICE)
    ours.init_decode_cuda_graph_batch_info(max_bs, 1)
    layer_qkv, layer_row = wrap_qkv(ours), wrap_row(ours)
    x_qkv = torch.randn(max_bs, k_qkv, device=DEVICE, dtype=DTYPE)
    x_row = torch.randn(max_bs, k_row, device=DEVICE, dtype=DTYPE)
    out_qkv = torch.empty(
        max_bs, layer_qkv.base_layer.weight.shape[0], device=DEVICE, dtype=DTYPE
    )
    out_row = torch.empty(
        max_bs, layer_row.base_layer.weight.shape[0], device=DEVICE, dtype=DTYPE
    )

    def forward():
        ours.reset_routing_cache()
        out_qkv.copy_(layer_qkv(x_qkv)[0])
        out_row.copy_(layer_row(x_row)[0])

    batch = _batch([1] * max_bs, [i % SLOTS for i in range(max_bs)], decode=True)
    _prepare(ours, batch, use_decode_cuda_graph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    _prepare(ours, batch, use_decode_cuda_graph=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()

    legacy = TritonLoRABackend(SLOTS, DEVICE)
    for live, slots in (
        (max_bs, [i % SLOTS for i in range(max_bs)]),
        (11, [3, 3, 1, 0, 2, 1, 1, 0, 3, 2, 0]),
    ):
        # Live rows change; the graph runs at the captured size and the rows
        # past the live batch carry no adapter after the re-prepare.
        x_qkv.copy_(torch.randn_like(x_qkv))
        x_row.copy_(torch.randn_like(x_row))
        live_batch = _batch([1] * live, slots, decode=True)
        _prepare(ours, live_batch, use_decode_cuda_graph=True)
        graph.replay()
        torch.cuda.synchronize()
        _prepare(legacy, live_batch)
        exp_qkv = wrap_qkv(legacy)(x_qkv[:live])[0]
        exp_row = wrap_row(legacy)(x_row[:live])[0]
        torch.testing.assert_close(
            out_qkv[:live].float(), exp_qkv.float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            out_row[:live].float(), exp_row.float(), rtol=2e-2, atol=2e-2
        )


class _TokensPlanTable:
    def __init__(self, large, small, threshold):
        self.large, self.small, self.threshold = large, small, threshold
        self.queries = []

    def plan_for(self, kind, phase, num_tokens, *geometry):
        self.queries.append(num_tokens)
        return self.large if num_tokens > self.threshold else self.small


@pytest.mark.parametrize("prefill", [False, True], ids=["decode", "prefill"])
def test_variable_bucket_graphs_replay_without_dense_host_work(monkeypatch, prefill):
    """Old graphs read refreshed metadata without Python planning or routing."""
    from sglang.srt.lora.dense.plan import DensePlan

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
    layer = wrap_row(ours)
    x = torch.randn(64, k_row, device=DEVICE, dtype=DTYPE)
    out = torch.empty(64, layer.base_layer.weight.shape[0], device=DEVICE, dtype=DTYPE)

    def batch_for(live, offset):
        lengths = [live // 2, live - live // 2] if prefill else [1] * live
        slots = [(i + offset) % SLOTS for i in range(len(lengths))]
        return _batch(lengths, slots, decode=not prefill)

    def prepare(batch):
        fb, wi, ranks, scal = batch
        _prepare_batch(
            ours,
            fb,
            wi,
            ranks,
            scal,
            use_decode_cuda_graph=not prefill,
            use_prefill_cuda_graph=prefill,
        )

    graphs = {}
    for bucket in (64, 16):

        def forward(bucket=bucket):
            ours.reset_routing_cache()
            out[:bucket].copy_(layer(x[:bucket])[0])

        prepare(batch_for(bucket, 0))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        graphs[bucket] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[bucket]):
            forward()
        assert ours.runner._token_slots.numel() == 64
        assert ours.runner.token_slots.numel() == bucket
        plan = table.large if bucket > table.threshold else table.small
        assert _cached_route_keys(ours.runner) == {("sorted", plan.block_size, SLOTS)}
    assert table.queries == [64, 64, 64, 16, 16, 16]

    def unexpected_dense_work(*args, **kwargs):
        pytest.fail("preparation and replay must not plan or build dense routes")

    monkeypatch.setattr(ours.runner, "plan_for", unexpected_dense_work)
    monkeypatch.setattr(ours.runner, "_build_route", unexpected_dense_work)
    legacy = TritonLoRABackend(SLOTS, DEVICE)
    # The same local size can replay either bucket, including DP-wide padding.
    for offset, (live, bucket) in enumerate(
        ((11, 64), (11, 16), (40, 64), (16, 16), (64, 64))
    ):
        x.copy_(torch.randn_like(x))
        batch = batch_for(live, offset + 1)
        prepare(batch)
        assert ours.runner.token_slots.numel() == live
        assert not ours.runner.workspace.routes
        graphs[bucket].replay()
        torch.cuda.synchronize()
        assert not ours.runner.workspace.routes
        _prepare(legacy, batch)
        expected = wrap_row(legacy)(x[:live])[0]
        torch.testing.assert_close(out[:live], expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 8, 16, 64])
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


def test_unrequested_lm_head_does_not_prepare_metadata(monkeypatch):
    import sglang.srt.lora.backend.triton_v2_backend as module

    backend = TritonV2LoRABackend(SLOTS, DEVICE)
    backend.validate_lora_targets(None, {"q_proj"})

    def forbidden(*args, **kwargs):
        raise AssertionError("lm_head preparation must be lazy")

    monkeypatch.setattr(module, "get_lm_head_pruned_lens", forbidden)
    _prepare(backend, _batch([5, 3], [1, 0], False))
    assert backend.lm_head_runner is None
    assert backend.lm_head_batch_info is None
    assert backend.lm_head_pass_batch_infos is None
    assert not hasattr(backend.batch_info, "seg_lens")
    assert not hasattr(backend.batch_info, "req_seg_indptr")
    backend.reset_routing_cache()
    backend.reset_batch_state()
    assert backend.batch_info is None
    assert not backend.runner.active


def test_v2_request_views_and_padding_refresh():
    from sglang.srt.lora.layers import FusedMoEWithLoRA

    backend = TritonV2LoRABackend(SLOTS, DEVICE)
    backend.is_moe_lora = True
    backend.init_prefill_cuda_graph_batch_info(64)
    backend.init_decode_cuda_graph_batch_info(8, 1)
    prefill = backend.prefill_cuda_graph_batch_info
    decode = backend.decode_cuda_graph_batch_info
    assert prefill.packed.data_ptr() != decode.packed.data_ptr()
    assert prefill.token_slots.data_ptr() != decode.token_slots.data_ptr()
    addresses = (prefill.packed.data_ptr(), prefill.token_slots.data_ptr())
    layer = SimpleNamespace(
        lora_backend=backend,
        gate_up_lora_a_weights=None,
        gate_up_lora_b_weights=None,
        down_lora_a_weights=None,
        down_lora_b_weights=None,
    )
    for lengths, slots, ranks in (
        ([3, 0, 5, 1], [0, 1, 2, 3], RANKS),
        ([1, 2], [0, 1], [0, 8, 0, 0]),
        ([0, 0], [0, 3], [0] * SLOTS),
    ):
        fb, _, _, scales = _batch(lengths, slots, False)
        _prepare_batch(backend, fb, slots, ranks, scales, False, True)
        info = backend.batch_info
        expected = [
            slot if ranks[slot] else -1
            for n, slot in zip(lengths, slots)
            for _ in range(n)
        ]
        assert info.token_slots.tolist() == expected + [-1] * (64 - len(expected))
        assert not hasattr(info, "adapter_enabled")
        assert not hasattr(info, "moe_lora_info")
        assert info.num_requests == len(lengths)
        assert info.num_tokens == len(expected)
        assert info.weight_indices[: info.num_requests].tolist() == slots
        assert info.seg_indptr[: info.num_requests + 1].diff().tolist() == lengths
        moe_batch = FusedMoEWithLoRA._get_moe_lora_batch(layer)
        assert moe_batch.token_lora_mapping.tolist() == expected
        assert moe_batch.token_lora_mapping.untyped_storage().data_ptr() == addresses[1]
        assert (info.packed.data_ptr(), info.token_slots.data_ptr()) == addresses
        snapshot = info.packed.clone()
        _prepare(backend, _batch([1, 1], [3, 1], True), use_decode_cuda_graph=True)
        torch.testing.assert_close(snapshot, prefill.packed)
    _prepare(backend, _batch([], [], True), use_decode_cuda_graph=True)
    assert decode.num_requests == decode.num_tokens == 0
    assert decode.token_slots.tolist() == [-1] * 8


@pytest.mark.parametrize("overlap", ["none", "a", "ab_delta"])
def test_pruned_chunked_lm_head_numeric_and_single_binding(monkeypatch, overlap):
    from unittest.mock import patch

    import sglang.srt.lora.backend.triton_v2_backend as module
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
        DensePlan(overlap=Overlap(overlap))
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
        with (
            patch.object(
                backend.lm_head_runner,
                "begin_batch",
                wraps=backend.lm_head_runner.begin_batch,
            ) as bind,
            patch.object(
                module,
                "_compute_token_lora_mapping",
                wraps=module._compute_token_lora_mapping,
            ) as mapping,
            patch.object(
                backend.lm_head_runner,
                "plan_for",
                wraps=backend.lm_head_runner.plan_for,
            ) as choose,
        ):
            for idx in range(2):
                head.set_lm_head_pass(idx)
                result = head(hidden[idx * 4 : (idx + 1) * 4])
                torch.testing.assert_close(
                    result.float(),
                    expected[idx * 4 : (idx + 1) * 4],
                    rtol=2e-2,
                    atol=2e-2,
                )
                assert (
                    bind.call_count
                    == mapping.call_count
                    == choose.call_count
                    == idx + 1
                )
                assert choose.call_args.args[0] is DenseLoraKind.LM_HEAD
            head.reset_lm_head_pass()
            result = head(hidden)
            torch.testing.assert_close(result.float(), expected, rtol=2e-2, atol=2e-2)
            assert bind.call_count == mapping.call_count == choose.call_count == 3
        backend.reset_batch_state()
        assert not hasattr(backend, "_lm_head_bound")
        assert backend._lm_head_pass_idx is None


@pytest.mark.parametrize(
    "site", ["embedding_decode", "embedding_prefill", "head_decode"]
)
@pytest.mark.parametrize("overlap", ["none", "a", "ab_delta"])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
def test_vocab_layers_use_unified_forward(monkeypatch, site, overlap, graph_mode):
    from unittest.mock import patch

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

    def forbidden(*args, **kwargs):
        pytest.fail("V2 vocab layers must not call legacy split A/B entry points")

    for name in ("run_lora_a_sgemm", "run_lora_b_sgemm", "run_lora_a_embedding"):
        monkeypatch.setattr(backend, name, forbidden)

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
    for bucket in (64, 16):
        lengths = [bucket // 2, bucket // 2] if prefill else [1] * bucket
        slots = [i % SLOTS for i in range(len(lengths))]
        prepare(lengths, slots, RANKS, SCALINGS)
        with (
            patch.object(
                backend, "forward_with_base", wraps=backend.forward_with_base
            ) as seam,
            patch.object(
                backend.runner, "plan_for", wraps=backend.runner.plan_for
            ) as choose,
        ):
            forward(bucket)
            assert seam.call_count == choose.call_count == 1
            assert seam.call_args.kwargs["kind"] == (
                "embedding" if embedding else "lm_head"
            )
            assert choose.call_args.args[0] is (
                DenseLoraKind.EMBEDDING if embedding else DenseLoraKind.LM_HEAD
            )
        if graph_mode:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward(bucket)
            torch.cuda.current_stream().wait_stream(stream)
            prepare(lengths, slots, RANKS, SCALINGS)
            graphs[bucket] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[bucket]):
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
        assert not hasattr(backend, "_lm_head_bound")


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
    for prefill in (False, True):
        for bucket in (64, 32):
            lengths = [bucket] if prefill else [width] * (bucket // width)
            slots = [i % SLOTS for i in range(len(lengths))]
            prepare(prefill, lengths, slots, RANKS, SCALINGS)

            def forward():
                backend.reset_routing_cache()
                output[:bucket].copy_(layer(x[:bucket])[0])

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            torch.cuda.current_stream().wait_stream(stream)
            prepare(prefill, lengths, slots, RANKS, SCALINGS)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
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


def test_fill_batch_writes_request_metadata_in_place_and_phase_isolated():
    backend = TritonV2LoRABackend(4, torch.device("cuda"))
    backend.init_decode_cuda_graph_batch_info(64, 1)
    backend.init_prefill_cuda_graph_batch_info(64)
    decode = backend.decode_cuda_graph_batch_info
    prefill = backend.prefill_cuda_graph_batch_info
    assert decode.packed.data_ptr() != prefill.packed.data_ptr()
    backend._fill_batch(decode, [1, 0], [16, 8, 0, 4], [1.0] * 4, None)

    for lengths in ([1] * 33, [3, 0, 2], []):
        slots = [i % 4 for i in range(len(lengths))]
        prefill.seg_indptr.fill_(-12345)
        backend._fill_batch(prefill, slots, [16, 8, 0, 4], [1.0] * 4, lengths)
        assert decode.weight_indices[:2].tolist() == [1, 0]
        assert prefill.weight_indices[: len(slots)].tolist() == slots
        assert prefill.seg_indptr[: len(lengths) + 1].diff().tolist() == lengths
        assert prefill.seg_indptr[: len(lengths) + 1].tolist() == [
            0,
            *accumulate(lengths),
        ]
        assert prefill.seg_indptr[len(lengths) + 1 :].eq(-12345).all().item()
        assert prefill.lora_ranks.tolist() == [16, 8, 0, 4]
        assert prefill.scalings.tolist() == [1.0] * 4
        assert not hasattr(prefill, "adapter_enabled")
