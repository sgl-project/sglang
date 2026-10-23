"""Dense executors against torch, including stale rank tails and no adapter."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.srt.lora.backend.base_backend import _compute_moe_lora_info  # noqa: E402
from sglang.srt.lora.dense.plan import (  # noqa: E402
    AFamily,
    BFamily,
    DensePlan,
    Overlap,
)
from sglang.srt.lora.dense.runner import DenseLoraRunner  # noqa: E402
from sglang.srt.lora.utils import LoRABatchInfo  # noqa: E402
from sglang.srt.lora.utils import Phase  # noqa: E402
from sglang.srt.lora.workspace import LoraWorkspace  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="dense LoRA kernels need CUDA"
)


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
SLOTS = 4
RANK_MAX = 64
RANKS = [64, 32, 0, 16]  # slot 2 is the "no adapter" slot
SCALINGS = [2.0, 0.5, 1.0, 1.25]
K_IN = 384

SITES = {
    "qkv": (0, 512, 640, 768),
    "merged": (0, 384, 768),
    "column": (0, 512),
    "row": (0, 320),
}

BATCHES = {
    # (segment lengths, slot per segment)
    "decode": (
        [1] * 37,
        [0, 1, 2, 3, 0, 0, 1, 2, 3, 3, 1, 0, 2, 3, 1, 0, 0] * 2 + [1, 3, 0],
    ),
    "prefill": ([7, 33, 1, 64, 10, 5], [1, 0, 2, 3, 2, 0]),
    "single": ([1], [3]),
}


def _pool(offsets: tuple[int, ...], generator: torch.Generator):
    """Store slice s at A rows [s*r, (s+1)*r) and B columns [0, r).
    Leave stale values beyond each adapter rank to catch out-of-rank reads.
    """
    stack = len(offsets) - 1
    width = offsets[-1]
    a = (
        torch.randn(SLOTS, stack * RANK_MAX, K_IN, generator=generator, device=DEVICE)
        * 100
    )
    b = torch.randn(SLOTS, width, RANK_MAX, generator=generator, device=DEVICE) * 100
    for slot, rank in enumerate(RANKS):
        a[slot, : stack * rank] = (
            torch.randn(stack * rank, K_IN, generator=generator, device=DEVICE) * 0.05
        )
        b[slot, :, :rank] = (
            torch.randn(width, rank, generator=generator, device=DEVICE) * 0.05
        )
    return a.to(DTYPE), b.to(DTYPE)


def _reference(x, a, b, offsets, token_slots):
    out = torch.zeros(x.shape[0], offsets[-1], dtype=torch.float32, device=DEVICE)
    stack = len(offsets) - 1
    for t in range(x.shape[0]):
        slot = int(token_slots[t])
        if slot < 0:
            continue
        rank = RANKS[slot]
        if rank == 0:
            continue
        for s in range(stack):
            bridge = (x[t].float() @ a[slot, s * rank : (s + 1) * rank].float().T).to(
                DTYPE
            )
            out[t, offsets[s] : offsets[s + 1]] += SCALINGS[slot] * (
                bridge.float() @ b[slot, offsets[s] : offsets[s + 1], :rank].float().T
            )
    return out


def _batch(name):
    seg_lens, seg_slots = BATCHES[name]
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.int32, device=DEVICE)
    seg_indptr = torch.zeros(len(seg_lens) + 1, dtype=torch.int32, device=DEVICE)
    seg_indptr[1:] = torch.cumsum(seg_lens_t, 0)
    weight_indices = torch.tensor(seg_slots, dtype=torch.int32, device=DEVICE)
    token_slots = torch.repeat_interleave(weight_indices, seg_lens_t)
    token_slots = torch.where(
        torch.tensor(
            [RANKS[s] > 0 for s in seg_slots], device=DEVICE
        ).repeat_interleave(seg_lens_t),
        token_slots,
        torch.full_like(token_slots, -1),
    )
    batch_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=len(seg_lens),
        num_segments=len(seg_lens),
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=torch.tensor(RANKS, dtype=torch.int64, device=DEVICE),
        scalings=torch.tensor(SCALINGS, dtype=torch.float32, device=DEVICE),
        max_len=max(seg_lens),
        seg_lens=seg_lens_t,
        permutation=None,
    )
    return batch_info, token_slots


def _engine(batch_info, token_slots, phase):
    engine = DenseLoraRunner(LoraWorkspace(), max_loras=SLOTS, device=DEVICE)
    engine.begin_batch(
        token_slots=token_slots,
        lora_ranks=batch_info.lora_ranks,
        scalings=batch_info.scalings,
        num_tokens=int(batch_info.seg_indptr[-1]),
        phase=phase,
        graph_mode=False,
    )
    return engine


PLANS = {
    "sorted16": DensePlan(block_size=16),
    "sorted64": DensePlan(block_size=64),
    "sorted16_ov": DensePlan(block_size=16, overlap=Overlap.A),
    "sorted16_gA_prB": DensePlan(block_size=16, b_family=BFamily.PER_ROW),
    "sorted16_splitk4": DensePlan(
        block_size=16,
        a_tiles={**DensePlan().a_tiles, "SPLIT_K": 4},
    ),
    "sorted16_splitk8_prB": DensePlan(
        block_size=16,
        b_family=BFamily.PER_ROW,
        a_tiles={**DensePlan().a_tiles, "SPLIT_K": 8},
    ),
    "grouped64_splitk3": DensePlan(
        block_size=64, a_tiles={**DensePlan().a_tiles, "SPLIT_K": 3}
    ),
    "sorted16_splitk4_planes": DensePlan(
        block_size=16,
        a_tiles={**DensePlan().a_tiles, "SPLIT_K": 4, "SPLIT_MODE": "planes"},
    ),
    "grouped64_splitk8_planes_ov": DensePlan(
        block_size=64,
        overlap=Overlap.A,
        a_tiles={**DensePlan().a_tiles, "SPLIT_K": 8, "SPLIT_MODE": "planes"},
    ),
    "sorted16_splitk8_planes_prB_ov": DensePlan(  # planes summed by the per-row expand
        block_size=16,
        b_family=BFamily.PER_ROW,
        overlap=Overlap.A,
        a_tiles={**DensePlan().a_tiles, "SPLIT_K": 8, "SPLIT_MODE": "planes"},
    ),
    "per_row": DensePlan(a_family=AFamily.PER_ROW, b_family=BFamily.PER_ROW),
    "all_slots": DensePlan(a_family=AFamily.ALL_SLOTS, b_family=BFamily.PER_ROW),
    "overlap_ab_delta": DensePlan(overlap=Overlap.AB_DELTA),
    "per_row_ab_delta": DensePlan(
        a_family=AFamily.PER_ROW, b_family=BFamily.PER_ROW, overlap=Overlap.AB_DELTA
    ),
}


@pytest.mark.parametrize(
    "batch_name,site_kind,plan_name",
    [("decode", "qkv", plan) for plan in PLANS]
    # overlap only reschedules streams, so its plans run on decode alone
    + [("prefill", "qkv", plan) for plan in PLANS if not plan.endswith("_ov")]
    + [
        ("prefill", site, plan)
        for site in ("merged", "column")
        for plan in ("sorted16", "per_row", "all_slots")
    ]
    + [("single", "column", plan) for plan in ("sorted16", "per_row", "all_slots")],
)
def test_engine_matches_reference(batch_name, site_kind, plan_name):
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    offsets = SITES[site_kind]
    a, b = _pool(offsets, generator)
    batch_info, token_slots = _batch(batch_name)
    num_tokens = int(batch_info.seg_indptr[-1])
    x = torch.randn(num_tokens, K_IN, generator=generator, device=DEVICE, dtype=DTYPE)
    base = torch.randn(
        num_tokens, offsets[-1], generator=generator, device=DEVICE, dtype=DTYPE
    )

    phase = Phase.PREFILL if batch_name == "prefill" else Phase.DECODE
    engine = _engine(batch_info, token_slots, phase)
    assert torch.equal(engine.token_slots, token_slots)
    out = engine.apply(
        x,
        lambda: base.clone(),
        PLANS[plan_name],
        a=a,
        b=b,
        offsets=offsets,
    )

    expected = base.float() + _reference(x, a, b, offsets, token_slots)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("overlap", [Overlap.NONE, Overlap.AB_DELTA])
def test_row_site_with_all_reduce(overlap):
    """Reduce the complete base + LoRA output exactly once."""
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    offsets = SITES["row"]
    a, b = _pool(offsets, generator)
    batch_info, token_slots = _batch("decode")
    num_tokens = int(batch_info.seg_indptr[-1])
    x = torch.randn(num_tokens, K_IN, generator=generator, device=DEVICE, dtype=DTYPE)
    base = torch.randn(
        num_tokens, offsets[-1], generator=generator, device=DEVICE, dtype=DTYPE
    )

    engine = _engine(batch_info, token_slots, Phase.DECODE)
    plan = DensePlan(overlap=overlap)
    reduced_shapes = []

    def all_reduce(tensor):
        reduced_shapes.append(tensor.shape)
        return tensor * 2

    out = engine.apply(
        x,
        lambda: base.clone(),
        plan,
        all_reduce=all_reduce,
        a=a,
        b=b,
        offsets=offsets,
    )

    expected = 2 * (base.float() + _reference(x, a, b, offsets, token_slots))
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=4e-2)
    assert reduced_shapes == [base.shape]


@pytest.mark.parametrize("overlap", [Overlap.A, Overlap.AB_DELTA])
def test_graph_mode_buffers_are_stable_and_capture_replays(overlap):
    """Warm in graph mode, capture one forward, replay with new inputs."""
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    offsets = SITES["qkv"]
    a, b = _pool(offsets, generator)
    batch_info, token_slots = _batch("decode")
    num_tokens = int(batch_info.seg_indptr[-1])
    x = torch.randn(num_tokens, K_IN, generator=generator, device=DEVICE, dtype=DTYPE)
    base = torch.randn(
        num_tokens, offsets[-1], generator=generator, device=DEVICE, dtype=DTYPE
    )
    out = torch.empty_like(base)

    engine = DenseLoraRunner(LoraWorkspace(), max_loras=SLOTS, device=DEVICE)
    plan = DensePlan(overlap=overlap)
    # The graph-static map is larger than any live batch, as the backend
    # allocates it at graph init.
    static_slots = torch.full((num_tokens + 27,), -1, dtype=torch.int32, device=DEVICE)

    def begin(live_requests: int = num_tokens):
        # Reset inactive slots to -1 but retain full static route capacity,
        # padding sentinel, and pair count as the live batch changes.
        _compute_moe_lora_info(
            live_requests,
            batch_info.seg_indptr[: live_requests + 1],
            batch_info.lora_ranks,
            batch_info.weight_indices[:live_requests],
            None,
            static_slots,
            max_len=batch_info.max_len,
        )
        engine.begin_batch(
            token_slots=static_slots,
            lora_ranks=batch_info.lora_ranks,
            scalings=batch_info.scalings,
            num_tokens=live_requests,
            phase=Phase.DECODE,
            graph_mode=True,
        )

    def forward():
        engine.reset_routes()
        out.copy_(
            engine.apply(
                x,
                lambda: base.clone(),
                plan,
                a=a,
                b=b,
                offsets=offsets,
            )
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        begin()
        forward()  # warmup allocates every workspace buffer
        route_before = engine.route(plan.block_size).sorted_pair_ids.data_ptr()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    begin()
    with torch.cuda.graph(graph, stream=stream):
        forward()
        captured_route = engine.route(plan.block_size)
    assert captured_route.sorted_pair_ids.data_ptr() == route_before

    # Only input metadata changes outside the graph; replay rebuilds routing.
    x.copy_(
        torch.randn(num_tokens, K_IN, generator=generator, device=DEVICE, dtype=DTYPE)
    )
    base.copy_(
        torch.randn(
            num_tokens, offsets[-1], generator=generator, device=DEVICE, dtype=DTYPE
        )
    )
    batch_info.weight_indices.copy_(torch.roll(batch_info.weight_indices, 1))
    begin()
    assert not engine.workspace.routes
    captured_route.sorted_pair_ids.fill_(-7)
    captured_route.block_bucket_ids.fill_(-7)
    captured_route.num_pairs_post_padded.zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected = base.float() + _reference(x, a, b, offsets, static_slots[:num_tokens])
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)

    # A smaller live batch replays the same graph: rows past the live tokens
    # must see no adapter (the map's tail is re-armed to -1 each batch).
    live = num_tokens - 13
    begin(live)
    assert torch.all(static_slots[live:] == -1)
    assert not engine.workspace.routes
    assert captured_route.sorted_pair_ids.data_ptr() == route_before
    captured_route.sorted_pair_ids.fill_(-7)
    captured_route.block_bucket_ids.fill_(-7)
    captured_route.num_pairs_post_padded.zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected = base.float() + _reference(x, a, b, offsets, static_slots[:num_tokens])
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def test_two_sites_on_two_streams_keep_separate_scratch():
    """Concurrent graph sites must not share scratch."""
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    sites = {"wide": SITES["qkv"], "narrow": (0, 64)}
    pools = {name: _pool(offsets, generator) for name, offsets in sites.items()}
    plans = {
        "wide": DensePlan(
            b_family=BFamily.PER_ROW,
            overlap=Overlap.A,
            a_tiles={**DensePlan().a_tiles, "SPLIT_K": 8},
        ),
        "narrow": DensePlan(
            b_family=BFamily.PER_ROW,
            overlap=Overlap.A,
            a_tiles={**DensePlan().a_tiles, "SPLIT_K": 4},
        ),
    }
    batch_info, token_slots = _batch("decode")
    num_tokens = int(batch_info.seg_indptr[-1])
    x = torch.randn(num_tokens, K_IN, generator=generator, device=DEVICE, dtype=DTYPE)
    bases = {
        name: torch.randn(
            num_tokens, offsets[-1], generator=generator, device=DEVICE, dtype=DTYPE
        )
        for name, offsets in sites.items()
    }
    outs = {name: torch.empty_like(base) for name, base in bases.items()}
    # the wider site's base GEMM in a model outlasts the narrower site's shrink;
    # a fixed filler keeps that window, so a shared bridge would be overwritten
    # before the wider expand reads it
    filler = torch.randn(1536, 1536, generator=generator, device=DEVICE)

    def base_fn(name):
        if name == "wide":
            for _ in range(4):
                torch.matmul(filler, filler)
        return bases[name].clone()

    def apply(engine, name):
        a, b = pools[name]
        return engine.apply(
            x, lambda: base_fn(name), plans[name], a=a, b=b, offsets=sites[name]
        )

    def bind(engine, graph_mode):
        engine.begin_batch(
            token_slots=token_slots,
            lora_ranks=batch_info.lora_ranks,
            scalings=batch_info.scalings,
            num_tokens=num_tokens,
            phase=Phase.DECODE,
            graph_mode=graph_mode,
        )

    shared = DenseLoraRunner(LoraWorkspace(), max_loras=SLOTS, device=DEVICE)
    stream, alt = torch.cuda.Stream(), torch.cuda.Stream()

    def forward():
        shared.reset_routes()
        current = torch.cuda.current_stream()
        alt.wait_stream(current)
        outs["wide"].copy_(apply(shared, "wide"))
        with torch.cuda.stream(alt):
            outs["narrow"].copy_(apply(shared, "narrow"))
        current.wait_stream(alt)

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        bind(shared, graph_mode=True)
        forward()  # warm-up on the capture stream allocates both streams' scratch
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    bind(shared, graph_mode=True)
    with torch.cuda.graph(graph, stream=stream):
        forward()

    serial = {
        name: DenseLoraRunner(LoraWorkspace(), max_loras=SLOTS, device=DEVICE)
        for name in sites
    }
    for _ in range(4):
        x.copy_(torch.randn_like(x))
        for base in bases.values():
            base.copy_(torch.randn_like(base))
        bind(shared, graph_mode=True)
        graph.replay()
        torch.cuda.synchronize()
        for name in sites:
            bind(serial[name], graph_mode=False)
            expected = apply(serial[name], name)
            torch.cuda.synchronize()
            assert torch.equal(outs[name], expected), name
        counters = [
            storage
            for key, storage in shared.workspace._graph_storage.items()
            if key[0] == "dense:splitk_counters"
        ]
        assert len(counters) >= 2 and not any(bool(c.any()) for c in counters)


def test_workspace_side_streams_are_keyed_by_the_calling_stream():
    """Different caller streams need independent side streams and fork/join events."""
    workspace = LoraWorkspace()
    main, alt = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(main):
        side_main = workspace.side_stream(DEVICE)
        ready_main = workspace.event(DEVICE, "x:ready")
    with torch.cuda.stream(alt):
        side_alt = workspace.side_stream(DEVICE)
        ready_alt = workspace.event(DEVICE, "x:ready")
    assert side_main is not side_alt and side_main not in (main, alt)
    assert ready_main is not ready_alt
    with torch.cuda.stream(main):
        assert workspace.side_stream(DEVICE) is side_main  # stable per calling stream

    # concurrent forks from both streams, each joining only its own side work
    x = torch.randn(4096, 4096, device=DEVICE, dtype=DTYPE)
    y = torch.randn(4096, 4096, device=DEVICE, dtype=DTYPE)
    ref_main = (x @ y) + x.sum(dim=0, keepdim=True)
    ref_alt = (y @ x) + y.sum(dim=0, keepdim=True)
    outs = {}

    def run(name, a, b):
        holder = {}
        side = lambda: holder.__setitem__("s", a.sum(dim=0, keepdim=True))
        base = workspace.run_parallel(
            name="x", device=DEVICE, compute=lambda: a @ b, side=side
        )
        outs[name] = base + holder["s"]

    for _ in range(3):
        with torch.cuda.stream(main):
            run("main", x, y)
        with torch.cuda.stream(alt):
            run("alt", y, x)
    torch.cuda.synchronize()
    torch.testing.assert_close(outs["main"], ref_main)
    torch.testing.assert_close(outs["alt"], ref_alt)


@pytest.mark.parametrize(
    "zero_sentinel,add_inplace",
    # dense never zeroes the sentinel; MoE zeroes it only when overwriting
    [(False, False), (False, True), (True, False)],
)
@pytest.mark.parametrize("blocks", [9, 36], ids=["short", "oversized"])
def test_grouped_b_retained_graphs_with_different_column_counts(
    zero_sentinel, add_inplace, blocks
):
    from sglang.kernels.ops.lora.common.lora_b import grouped_lora_b, slice_geometry
    from sglang.kernels.ops.lora.common.route_view import RouteView, RouteViewKind

    def make_case(widths):
        tokens, rank, block = 51, 16, 16
        width, slices = sum(widths), len(widths)
        capacity = (blocks - 1) * block + 3
        bridge = torch.empty(
            (tokens, slices * rank), dtype=torch.bfloat16, device="cuda"
        )
        weight = torch.empty((2, width, rank), dtype=torch.bfloat16, device="cuda")
        output = torch.empty((tokens, width), dtype=torch.bfloat16, device="cuda")
        ranks = torch.empty(2, dtype=torch.int32, device="cuda")
        scalings = torch.empty(2, dtype=torch.float32, device="cuda")
        route = RouteView(
            view=RouteViewKind.ALIGNED,
            block_size=block,
            token_slots=torch.full((tokens,), -1, dtype=torch.int32, device="cuda"),
            group_ids=None,
            groups_per_slot=1,
            max_loras=2,
            maybe_sorted_pair_ids=torch.empty(
                capacity, dtype=torch.int32, device="cuda"
            ),
            maybe_block_bucket_ids=torch.empty(
                blocks, dtype=torch.int32, device="cuda"
            ),
            maybe_num_pairs_post_padded=torch.empty(
                1, dtype=torch.int32, device="cuda"
            ),
        )
        starts = [0]
        for size in widths:
            starts.append(starts[-1] + size)
        geometry = slice_geometry(starts, 64, torch.device("cuda"))
        if len(set(widths)) > 1:
            # An extra column tile exercises _slice_of_tile's no-slice return.
            geometry = geometry._replace(num_column_tiles=geometry.num_column_tiles + 1)

        def launch():
            grouped_lora_b(
                bridge,
                weight,
                output,
                route,
                geometry=geometry,
                config={
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 16,
                    "GROUP_SIZE_M": 8,
                    "num_warps": 4,
                    "num_stages": 2,
                },
                add_inplace=add_inplace,
                zero_sentinel=zero_sentinel,
                lora_ranks=ranks,
                scalings=scalings,
            )

        def prepare(live_blocks, iteration):
            generator = torch.Generator().manual_seed(9127 + iteration)
            bridge_cpu = (
                torch.randn(tokens, slices * rank, generator=generator) * 0.25
            ).bfloat16()
            weight_cpu = (
                torch.randn(2, width, rank, generator=generator) * 0.25
            ).bfloat16()
            ranks_cpu = [0 if iteration == 3 else 16, 8 if iteration % 2 == 0 else 16]
            scales_cpu = [0.5 + iteration * 0.25, 1.5 - iteration * 0.125]
            expected = torch.full((tokens, width), 7.0, dtype=torch.bfloat16)
            ids = torch.full((capacity,), -(1 << 20), dtype=torch.int32)
            bucket_ids = torch.full((blocks,), 12345, dtype=torch.int32)
            mapping = torch.full((tokens,), -1, dtype=torch.int32)
            ids[: min(live_blocks * block, capacity)] = tokens
            bucket_ids[:live_blocks] = -1
            active = {}
            if live_blocks:
                active[0] = (1 - iteration % 2, list(range(16)))
            if live_blocks >= 2:
                active[1] = (-1, list(range(16, 32)))
            if live_blocks == 3:
                active[2] = (1, list(range(32, 48)))
            if live_blocks == blocks:
                active[3] = (-1 if iteration == 0 else 0, list(range(32, 48)))
                active[blocks - 1] = (1, list(range(48, 51)))
            for block_id, (bucket_id, rows) in active.items():
                bucket_ids[block_id] = bucket_id
                begin = block_id * block
                ids[begin : begin + len(rows)] = torch.tensor(rows, dtype=torch.int32)
                mapping[rows] = bucket_id
                if bucket_id == -1:
                    if zero_sentinel:
                        expected[rows] = 0
                elif ranks_cpu[bucket_id]:
                    active_rank = ranks_cpu[bucket_id]
                    for slice_id, size in enumerate(widths):
                        start, end = starts[slice_id : slice_id + 2]
                        bridge_start = slice_id * active_rank
                        delta = (
                            bridge_cpu[
                                rows, bridge_start : bridge_start + active_rank
                            ].float()
                            @ weight_cpu[bucket_id, start:end, :active_rank].float().T
                        )
                        delta *= scales_cpu[bucket_id]
                        if add_inplace:
                            delta += expected[rows, start:end].float()
                        expected[rows, start:end] = delta.bfloat16()
            bridge.copy_(bridge_cpu)
            weight.copy_(weight_cpu)
            ranks.copy_(torch.tensor(ranks_cpu, dtype=torch.int32))
            scalings.copy_(torch.tensor(scales_cpu, dtype=torch.float32))
            route.sorted_pair_ids.copy_(ids)
            route.block_bucket_ids.copy_(bucket_ids)
            route.num_pairs_post_padded.fill_(live_blocks * block)
            route.token_slots.copy_(mapping)
            output.fill_(7)
            return expected

        prepare(blocks, 0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch()
            launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        torch.cuda.current_stream().wait_stream(stream)

        return prepare, launch, graph, output

    cases = [make_case(widths) for widths in ((64,), (128, 128), (65, 70))]
    for iteration, live_blocks in enumerate((blocks, 2, 0, blocks, 3, blocks)):
        for prepare, launch, graph, output in cases if iteration % 2 else cases[::-1]:
            expected = prepare(live_blocks, iteration)
            graph.replay()
            torch.testing.assert_close(output.cpu(), expected, atol=5e-3, rtol=2e-2)
            output.fill_(7)
            launch()
            torch.testing.assert_close(output.cpu(), expected, atol=5e-3, rtol=2e-2)


def _poisoned_tail_expand_case(family, planes, seed=4111):
    """Expand inputs whose inactive rank tails hold NaN.

    Pool rank 128 with live ranks 96 and 128 and BLOCK_SIZE_K 64: the live
    rank of slot 0 splits the second K block, so the whole-block loads read
    the poisoned lanes and must drop them. Rows 38 and 39 carry no adapter.
    Returns the launch, the output, the expected output and the inputs.
    """
    from sglang.kernels.ops.lora.common.lora_b import (
        grouped_lora_b,
        per_row_lora_b,
        slice_geometry,
    )
    from sglang.kernels.ops.lora.common.route_view import RouteView, RouteViewKind

    tokens, pool_rank, width, block = 40, 128, 256, 16
    ranks, scalings = [96, 128], [1.0, 0.5]
    generator = torch.Generator().manual_seed(seed)
    slot_rows = {0: list(range(24)), 1: list(range(24, 38))}
    token_slots_cpu = torch.full((tokens,), -1, dtype=torch.int32)
    for slot, rows in slot_rows.items():
        token_slots_cpu[rows] = slot
    if planes == 1:
        bridge_cpu = (torch.randn(tokens, pool_rank, generator=generator) * 0.25).to(
            torch.bfloat16
        )
    else:
        bridge_cpu = torch.randn(planes, tokens, pool_rank, generator=generator) * 0.05
    weight_cpu = (
        torch.randn(2, width, pool_rank, generator=generator) * 0.25
    ).bfloat16()
    expected = torch.full((tokens, width), 7.0, dtype=torch.bfloat16)
    for slot, rows in slot_rows.items():
        rank = ranks[slot]
        if planes == 1:
            summed = bridge_cpu[rows].float()
        else:
            # Planes are summed in order and rounded to bf16 before the expand.
            summed = bridge_cpu[0, rows].clone()
            for p in range(1, planes):
                summed += bridge_cpu[p, rows]
        summed = summed.bfloat16().float()
        expected[rows] = (
            scalings[slot] * (summed[:, :rank] @ weight_cpu[slot, :, :rank].float().T)
        ).bfloat16()
        if rank < pool_rank:
            bridge_cpu[..., rows, rank:] = float("nan")
            weight_cpu[slot, :, rank:] = float("nan")
    bridge_cpu[..., 38:, :] = float("nan")

    bridge = bridge_cpu.to(DEVICE)
    weight = weight_cpu.to(DEVICE)
    output = torch.full((tokens, width), 7.0, dtype=torch.bfloat16, device=DEVICE)
    token_slots = token_slots_cpu.to(DEVICE)
    lora_ranks = torch.tensor(ranks, dtype=torch.int32, device=DEVICE)
    scales = torch.tensor(scalings, dtype=torch.float32, device=DEVICE)
    geometry = slice_geometry([0, width], 128, DEVICE)
    config = {
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }
    if family == "grouped":
        # Three aligned blocks: slot 0 fills blocks 0-1, slot 1 block 2; the
        # pad entries carry the pair count and the base-only rows stay out.
        pad = tokens
        ids = list(range(16)) + list(range(16, 24)) + [pad] * 8
        ids += list(range(24, 38)) + [pad] * 2
        route = RouteView(
            view=RouteViewKind.ALIGNED,
            block_size=block,
            token_slots=token_slots,
            group_ids=None,
            groups_per_slot=1,
            max_loras=2,
            maybe_sorted_pair_ids=torch.tensor(ids, dtype=torch.int32, device=DEVICE),
            maybe_block_bucket_ids=torch.tensor(
                [0, 0, 1], dtype=torch.int32, device=DEVICE
            ),
            maybe_num_pairs_post_padded=torch.tensor(
                [3 * block], dtype=torch.int32, device=DEVICE
            ),
        )
        expand = grouped_lora_b
    else:
        route = RouteView(
            view=RouteViewKind.RAW,
            block_size=block,
            token_slots=token_slots,
            group_ids=None,
            groups_per_slot=1,
            max_loras=2,
        )
        expand = per_row_lora_b

    def launch():
        expand(
            bridge,
            weight,
            output,
            route,
            geometry=geometry,
            config=config,
            add_inplace=False,
            zero_sentinel=False,
            lora_ranks=lora_ranks,
            scalings=scales,
            planes=planes,
        )

    return launch, output, expected, {"bridge": bridge, "weight": weight}


@pytest.mark.parametrize("planes", [1, 8])
@pytest.mark.parametrize("family", ["grouped", "per_row"])
def test_expand_whole_block_loads_drop_poisoned_rank_tails(family, planes):
    launch, output, expected, inputs = _poisoned_tail_expand_case(family, planes)
    assert not torch.isfinite(inputs["bridge"]).all()
    assert not torch.isfinite(inputs["weight"]).all()
    launch()
    torch.cuda.synchronize()
    result = output.float().cpu()
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected.float(), atol=5e-3, rtol=2e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
