"""Inkling sink LoRA against a reference using raw, pre-scaled MoE pool buffers.
Covers both layouts, executor families, TP shards, and graph-visible adapter/base reloads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.moe.inkling_moe import silu_and_mul_triton  # noqa: E402
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend  # noqa: E402
from sglang.srt.lora.dense.plan import (  # noqa: E402
    AFamily,
    BFamily,
    DensePlan,
    Overlap,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.model_executor.runner_utils.capture_mode import (  # noqa: E402
    capture_lora_variant,
    model_capture_mode,
)
from sglang.srt.models.inkling_common.dense_mlp import (  # noqa: E402
    InklingBatchDenseMLP,
)
from sglang.srt.models.inkling_common.lora import (  # noqa: E402
    InklingBatchDenseMLPWithLoRAV2,
)
from sglang.srt.models.inkling_common.util import deinterleave_gate_up  # noqa: E402
from sglang.srt.runtime_context import get_context, get_parallel  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the sink kernels need CUDA"
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
SLOTS, RMAX = 4, 32
RANKS = [32, 16, 0, 8]  # slot 2 holds no adapter
# A pool rank whose shared gate/up bridge (2R = 32 columns) is narrower than
# the shrink's 64-wide column tile.
SMALL_RMAX, SMALL_RANKS = 16, [16, 8, 0, 8]
HIDDEN, N_EXPERTS, F = 256, 2, 128
TOL = dict(rtol=3e-2, atol=3e-2)
V2 = InklingBatchDenseMLPWithLoRAV2


def _fill_slot(bufs, slot: int, rank: int, g: torch.Generator) -> None:
    """One slot as the pool writes it: max-rank spaced, zero tails, pre-scaled B."""
    a_gu, b_gu, a_dn, b_dn = bufs
    outer, f, rmax = a_gu.shape[1], b_gu.shape[2] // 2, b_gu.shape[3]
    for buf in bufs:
        buf[slot].zero_()
    if rank == 0:
        return

    def rnd(*shape):
        return (torch.randn(*shape, generator=g, device=DEVICE) * 0.05).to(DTYPE)

    for half in range(2):
        a_gu[slot, :, half * rmax : half * rmax + rank] = rnd(outer, rank, HIDDEN)
    b_gu[slot, :, :, :rank] = rnd(N_EXPERTS, 2 * f, rank)
    a_dn[slot, :, :rank] = rnd(N_EXPERTS, rank, f)
    b_dn[slot, :, :, :rank] = rnd(outer, HIDDEN, rank)


def _pool(shared_outer: bool, seed: int, f: int = F, ranks=RANKS, rmax: int = RMAX):
    outer = 1 if shared_outer else N_EXPERTS
    bufs = (
        torch.zeros(SLOTS, outer, 2 * rmax, HIDDEN, device=DEVICE, dtype=DTYPE),
        torch.zeros(SLOTS, N_EXPERTS, 2 * f, rmax, device=DEVICE, dtype=DTYPE),
        torch.zeros(SLOTS, N_EXPERTS, rmax, f, device=DEVICE, dtype=DTYPE),
        torch.zeros(SLOTS, outer, HIDDEN, rmax, device=DEVICE, dtype=DTYPE),
    )
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for slot, rank in enumerate(ranks):
        _fill_slot(bufs, slot, rank, g)
    return bufs


def _swiglu_interleaved(y, gammas):
    """Run the fused gated activation for the served interleaved layout."""
    t, n, two_f = y.shape
    return silu_and_mul_triton(y.reshape(t * n, two_f), gammas.reshape(-1)).view(
        t, n, two_f // 2
    )


def _base_w13(w13_lin):
    """Convert each expert to [gate | up]; reference weights stay interleaved."""
    rows = w13_lin.shape[0] // N_EXPERTS
    return torch.stack(
        [deinterleave_gate_up(e, dim=0) for e in w13_lin.view(N_EXPERTS, rows, -1)]
    ).reshape_as(w13_lin)


def _reference(layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer):
    a_gu, b_gu, a_dn, b_dn = (b.float() for b in bufs)
    t, n, f, rmax = x.shape[0], N_EXPERTS, b_gu.shape[2] // 2, b_gu.shape[3]
    y = (x.float() @ w13_lin.float().T).view(t, n, 2 * f)
    for i, s in enumerate(token_slots):
        if s < 0:
            continue
        for e in range(n):
            a = a_gu[s, 0 if shared_outer else e]  # [2R, hidden]
            bridge = (
                (x[i].float() @ a.T).to(DTYPE).float()
            )  # bf16 bridge, as the kernels
            y[i, e, 0::2] += bridge[:rmax] @ b_gu[s, e, :f].T
            y[i, e, 1::2] += bridge[rmax:] @ b_gu[s, e, f:].T
    act = _swiglu_interleaved(y.to(DTYPE), gammas)  # [t, n, f]
    out = act.reshape(t, -1).float() @ w2_lin.float()
    for i, s in enumerate(token_slots):
        if s < 0:
            continue
        if shared_outer:
            bridge = sum(act[i, e].float() @ a_dn[s, e].T for e in range(n))
            out[i] += bridge.to(DTYPE).float() @ b_dn[s, 0].T
        else:
            for e in range(n):
                bridge = (act[i, e].float() @ a_dn[s, e].T).to(DTYPE).float()
                out[i] += bridge @ b_dn[s, e].T
    return out


def _batch(decode: bool, slots: list[int] | None = None, ranks=RANKS):
    seq_lens = [1] * 13 if decode else [5, 17, 1, 9]
    if slots is None:
        slots = [i % SLOTS for i in range(13)] if decode else [1, 0, 2, 3]
    fb = SimpleNamespace(
        forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        extend_seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        extend_seq_lens_cpu=list(seq_lens),
        extend_num_tokens=int(sum(seq_lens)),
        spec_info=None,
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
    )
    token_slots = [
        slot if ranks[slot] > 0 else -1
        for slot, length in zip(slots, seq_lens)
        for _ in range(length)
    ]
    return fb, slots, token_slots


def _inputs(t: int, f: int = F, seed: int = 11):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(t, HIDDEN, generator=g, device=DEVICE, dtype=DTYPE)
    gammas = torch.rand(t, N_EXPERTS, generator=g, device=DEVICE, dtype=DTYPE) + 0.5
    w13_lin = (
        torch.randn(N_EXPERTS * 2 * f, HIDDEN, generator=g, device=DEVICE) * 0.02
    ).to(DTYPE)
    w2_lin = (torch.randn(N_EXPERTS * f, HIDDEN, generator=g, device=DEVICE) * 0.02).to(
        DTYPE
    )
    return x, gammas, w13_lin, w2_lin


def _layer(cls, backend, bufs, *, f: int = F, tp_size: int = 1, tp_rank: int = 0):
    """The LoRA wrapper on a bare sink (no base weights to build)."""
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.n_shared_experts = N_EXPERTS
    layer.tp_group = None
    layer.inference_moe_w13_interleaved = True
    layer._linearized_bf16_enabled = True
    layer._w13_gate_up_contiguous = cls is V2
    layer.moe_tp_size, layer.moe_tp_rank = tp_size, tp_rank
    layer.intermediate_size_per_partition = f
    layer.initialize_lora(backend)
    if bufs is not None:
        layer.set_lora_info(*bufs)
    return layer


def _v2_backend():
    backend = TritonV2LoRABackend(SLOTS, DEVICE)
    backend.is_moe_lora = True
    return backend


SINK_PREFIX = "model.layers.0.mlp.experts.shared_experts"


def _real_sink(
    cls, *, f: int = F, tp_size: int = 1, tp_rank: int = 0, lora_backend=None
):
    """Construct under the serving configuration before any weight is loaded."""
    if lora_backend is None:
        lora_backend = "triton_v2" if cls is V2 else "triton"
    default = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        # The sink partitions its shared experts over the "tp" group, which
        # resolves to the published tp_size and the overridden tp_rank before
        # any process group exists.
        with (
            get_context().override_server_args(
                enable_lora=True, lora_backend=lora_backend, tp_size=tp_size
            ),
            get_parallel().override(tp_rank=tp_rank),
        ):
            layer = cls(
                n_shared_experts=N_EXPERTS,
                d_model=HIDDEN,
                shared_d_mlp=f * tp_size,
                layer_id=0,
                prefix=SINK_PREFIX,
                inference_moe_w13_interleaved=True,
                parallel_group="tp",
                linearized_bf16=True,
            )
    finally:
        torch.set_default_dtype(default)
    return layer.to(DEVICE)


def _load_w13(layer, w13_lin):
    """Load checkpoint [N, 2F, H] gate/up weights through the TP-sliced loader."""
    full = w13_lin.view(N_EXPERTS, -1, HIDDEN)
    layer.weight_loader_fused(
        layer.w13_weight, full, f"{SINK_PREFIX}.w13_weight", "w13"
    )


def _load_w2(layer, w2_lin):
    """Load the linearized [N*F, H] down rows as the checkpoint's [N, H, F]."""
    full = w2_lin.view(N_EXPERTS, -1, HIDDEN).transpose(1, 2).contiguous()
    layer.weight_loader_fused(layer.w2_weight, full, f"{SINK_PREFIX}.w2_weight", "w2")


def _loaded_v2(backend, bufs, w13_lin, w2_lin):
    """Load and post-process base weights before attaching the LoRA wrapper."""
    layer = _real_sink(V2)
    _load_w13(layer, w13_lin)
    _load_w2(layer, w2_lin)
    layer.process_weights_after_loading()
    layer.initialize_lora(backend)
    layer.set_lora_info(*bufs)
    return layer


def _prepare(backend, fb, slots, ranks=RANKS, *, graph=False, prefill_graph=False):
    backend.prepare_lora_batch(
        fb,
        slots,
        list(ranks),
        [1.0] * SLOTS,
        use_decode_cuda_graph=graph,
        use_prefill_cuda_graph=prefill_graph,
    )
    backend.batch_info.has_active_lora = any(
        slot >= 0 and ranks[slot] > 0 for slot in slots
    )


def _forward(layer, x, gammas, w13_lin, w2_lin):
    return layer._forward_bf16_linearized(
        x, gammas, (w13_lin, w2_lin), use_reduce_scatter=True
    )


def _capture(backend, fn):
    """Warm up, then capture one forward and return its graph and static output."""

    def forward():
        backend.reset_routing_cache()
        return fn()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with model_capture_mode(), capture_lora_variant(None):
        with torch.cuda.stream(stream):
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = forward()
    return graph, out


def _tiles(**overrides):
    tiles = dict(DensePlan().a_tiles)
    tiles.update(overrides)
    return tiles


# Overlap.AB_DELTA is site-agnostic apply() plumbing; test_dense_lora_kernels covers it.
PLANS = {
    "grouped": DensePlan(),
    "grouped_sorted_splitk_serial": DensePlan(a_tiles=_tiles(SPLIT_K=4)),
    "grouped_splitk_planes": DensePlan(a_tiles=_tiles(SPLIT_K=2, SPLIT_MODE="planes")),
    "grouped_overlap_a": DensePlan(overlap=Overlap.A),
    "per_row_a_grouped_b": DensePlan(a_family=AFamily.PER_ROW, overlap=Overlap.A),
    "per_row": DensePlan(
        a_family=AFamily.PER_ROW, b_family=BFamily.PER_ROW, overlap=Overlap.A
    ),
    "all_slots": DensePlan(a_family=AFamily.ALL_SLOTS, b_family=BFamily.PER_ROW),
    "all_slots_overlap_a": DensePlan(
        a_family=AFamily.ALL_SLOTS, b_family=BFamily.PER_ROW, overlap=Overlap.A
    ),
}


def _pin_plan(backend, plan: DensePlan) -> None:
    backend.runner.plan_for = (
        lambda kind, max_rank, in_features=0, out_features=0, *, num_tokens: plan
    )


@pytest.mark.parametrize(
    "shared_outer,decode,rmax",
    [(shared, decode, RMAX) for shared in (True, False) for decode in (True, False)]
    + [(True, False, SMALL_RMAX)],
)
def test_v2_tail_width_matches_reference(shared_outer, decode, rmax):
    f = 96
    ranks = RANKS if rmax == RMAX else SMALL_RANKS
    bufs = _pool(shared_outer, seed=3 + shared_outer, f=f, ranks=ranks, rmax=rmax)
    fb, slots, token_slots = _batch(decode, ranks=ranks)
    x, gammas, w13_lin, w2_lin = _inputs(fb.extend_num_tokens, f=f)
    backend = _v2_backend()
    _prepare(backend, fb, slots, ranks)
    layer = _layer(V2, backend, bufs, f=f)
    out = _forward(layer, x, gammas, _base_w13(w13_lin), w2_lin)
    expected = _reference(
        layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
    )
    torch.testing.assert_close(out.float(), expected, **TOL)


@pytest.mark.parametrize(
    "shared_outer,plan_name,decode,rmax",
    [
        (shared, name, decode, RMAX)
        for shared in (True, False)
        for name in sorted(PLANS)
        if shared or not name.startswith("all_slots")
        for decode in (True, False)
    ]
    + [(shared, "grouped", False, SMALL_RMAX) for shared in (True, False)],
)
def test_v2_every_executor_family(shared_outer, plan_name, decode, rmax):
    """Exercise every sink executor with full and partial shrink tiles."""
    ranks = RANKS if rmax == RMAX else SMALL_RANKS
    bufs = _pool(shared_outer, seed=5, ranks=ranks, rmax=rmax)
    fb, slots, token_slots = _batch(decode, ranks=ranks)
    x, gammas, w13_lin, w2_lin = _inputs(fb.extend_num_tokens)
    backend = _v2_backend()
    _prepare(backend, fb, slots, ranks)
    _pin_plan(backend, PLANS[plan_name])
    layer = _layer(V2, backend, bufs)
    out = _forward(layer, x, gammas, _base_w13(w13_lin), w2_lin)
    expected = _reference(
        layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
    )
    torch.testing.assert_close(out.float(), expected, **TOL)


@pytest.mark.parametrize(
    "prefill", [False, True], ids=["decode_graph", "prefill_graph"]
)
@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
def test_v2_graph_replay_follows_adapters_and_slot_swaps(shared_outer, prefill):
    bufs = _pool(shared_outer, seed=9)
    fb, slots, token_slots = _batch(not prefill, slots=[2] * (4 if prefill else 13))
    t = fb.extend_num_tokens
    x, gammas, w13_lin, w2_lin = _inputs(t)
    backend = _v2_backend()
    if prefill:
        backend.init_prefill_cuda_graph_batch_info(t)
        _prepare(backend, fb, slots, prefill_graph=True)
    else:
        backend.init_decode_cuda_graph_batch_info(fb.batch_size, 1)
        _prepare(backend, fb, slots, graph=True)
    assert not backend.batch_info.has_active_lora
    layer = _layer(V2, backend, bufs)
    w13_base = _base_w13(w13_lin)
    graph, out = _capture(backend, lambda: _forward(layer, x, gammas, w13_base, w2_lin))

    def check(token_slots):
        graph.replay()
        expected = _reference(
            layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
        )
        torch.testing.assert_close(out.float(), expected, **TOL)

    # Captured on an empty route: every bucket is -1, so the replay is the base.
    check(token_slots)
    fb, slots, token_slots = _batch(not prefill)
    _prepare(backend, fb, slots, graph=not prefill, prefill_graph=prefill)
    check(token_slots)
    # The next batch routes its requests to other adapters (and one to none).
    new_slots = [3, 2, 1, 0] if prefill else [(i + 1) % SLOTS for i in range(13)]
    fb2, _, token_slots2 = _batch(not prefill, slots=new_slots)
    _prepare(backend, fb2, new_slots, graph=not prefill, prefill_graph=prefill)
    check(token_slots2)
    # A pool slot is replaced (the empty slot 2 takes a rank-8 adapter, slot 0
    # a new rank-32 one) after capture: the replay reads the new weights.
    g = torch.Generator(device=DEVICE).manual_seed(99)
    ranks = list(RANKS)
    for slot, rank in ((2, 8), (0, 32)):
        _fill_slot(bufs, slot, rank, g)
        ranks[slot] = rank
        layer.on_lora_slots_updated({slot})
    fb3, _, token_slots3 = _batch(not prefill, slots=new_slots, ranks=ranks)
    _prepare(backend, fb3, new_slots, ranks, graph=not prefill, prefill_graph=prefill)
    check(token_slots3)


@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
def test_v2_tp4_local_shards_sum_to_the_unsharded_output(shared_outer):
    """Sum four TP-local base/pool shards, sliced by the wrapper on one GPU."""
    tp, f_full = 4, 256
    f_local = f_full // tp
    bufs = _pool(shared_outer, seed=13, f=f_full)
    fb, slots, token_slots = _batch(False)
    x, gammas, w13_lin, w2_lin = _inputs(fb.extend_num_tokens, f=f_full)
    backend = _v2_backend()
    _prepare(backend, fb, slots)
    full = _layer(V2, backend, bufs, f=f_full)
    expected = _reference(
        full, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
    )
    torch.testing.assert_close(
        _forward(full, x, gammas, _base_w13(w13_lin), w2_lin).float(),
        expected,
        **TOL,
    )

    a_gu, b_gu, a_dn, b_dn = bufs
    total = torch.zeros_like(expected)
    for rank in range(tp):
        layer = _layer(V2, backend, None, f=f_local, tp_size=tp, tp_rank=rank)
        b_gu_r = torch.stack(
            [
                layer.slice_moe_lora_b_weights(b_gu[s], rank, "gate_up_proj_moe")
                for s in range(SLOTS)
            ]
        )
        a_dn_r = torch.stack(
            [
                layer.slice_moe_lora_a_weights(a_dn[s], rank, "down_proj_moe")
                for s in range(SLOTS)
            ]
        )
        assert b_gu_r.shape == (SLOTS, N_EXPERTS, 2 * f_local, RMAX)
        assert a_dn_r.shape == (SLOTS, N_EXPERTS, RMAX, f_local)
        layer.set_lora_info(a_gu, b_gu_r, a_dn_r, b_dn)
        # Interleaved gate/up rows of this rank's features, and its w2 rows.
        w13_r = (
            w13_lin.view(N_EXPERTS, 2 * f_full, HIDDEN)[
                :, 2 * rank * f_local : 2 * (rank + 1) * f_local
            ]
            .reshape(-1, HIDDEN)
            .contiguous()
        )
        w2_r = (
            w2_lin.view(N_EXPERTS, f_full, HIDDEN)[
                :, rank * f_local : (rank + 1) * f_local
            ]
            .reshape(-1, HIDDEN)
            .contiguous()
        )
        # The layout conversion follows the shard, as it does on the loaded rows.
        total += _forward(layer, x, gammas, _base_w13(w13_r), w2_r).float()
    torch.testing.assert_close(total, expected, **TOL)


@pytest.mark.parametrize(
    "prefill,shared_outer",
    [(False, True), (True, False)],
    ids=["decode_graph-shared", "prefill_graph-per_expert"],
)
def test_w13_reload_through_the_loader_reaches_the_captured_graph(
    prefill, shared_outer
):
    """Reloads reach the next replay without post-load processing or an eager repair."""
    bufs = _pool(shared_outer, seed=21)
    fb, slots, token_slots = _batch(not prefill)
    t = fb.extend_num_tokens
    x, gammas, w13_lin, w2_lin = _inputs(t)
    backend = _v2_backend()
    if prefill:
        backend.init_prefill_cuda_graph_batch_info(t)
        _prepare(backend, fb, slots, prefill_graph=True)
    else:
        backend.init_decode_cuda_graph_batch_info(fb.batch_size, 1)
        _prepare(backend, fb, slots, graph=True)
    layer = _loaded_v2(backend, bufs, w13_lin, w2_lin)
    assert torch.equal(layer.w13_weight.data.view_as(w13_lin), _base_w13(w13_lin))
    graph, out = _capture(
        backend, lambda: layer.forward(x, gammas, use_reduce_scatter=True)
    )
    addresses = (
        layer.w13_weight.data_ptr(),
        layer.w2_weight.data_ptr(),
        layer._w2_lin.data_ptr(),
    )

    def check(w13_now, w2_now):
        # The loaded rows are converted exactly once (twice would not be
        # [gate | up] either), then the existing graph replays on them.
        assert torch.equal(layer.w13_weight.data.view_as(w13_now), _base_w13(w13_now))
        graph.replay()
        expected = _reference(
            layer, x, gammas, w13_now, w2_now, bufs, token_slots, shared_outer
        )
        torch.testing.assert_close(out.float(), expected, **TOL)
        assert (
            layer.w13_weight.data_ptr(),
            layer.w2_weight.data_ptr(),
            layer._w2_lin.data_ptr(),
        ) == addresses

    check(w13_lin, w2_lin)
    # From here on nothing may repair the weights but the loader itself.
    layer.forward = layer.get_bf16_linearized_weights = lambda *a, **k: pytest.fail(
        "a reload must not need an eager forward or the linearized getter"
    )
    g = torch.Generator(device=DEVICE).manual_seed(31)

    def fresh(like):
        return (torch.randn(like.shape, generator=g, device=DEVICE) * 0.02).to(DTYPE)

    w13_a = fresh(w13_lin)
    _load_w13(layer, w13_a)  # W13 alone
    check(w13_a, w2_lin)
    w2_b, w13_b = fresh(w2_lin), fresh(w13_lin)
    _load_w2(layer, w2_b)  # W2, then W13
    _load_w13(layer, w13_b)
    check(w13_b, w2_b)
    w13_c, w2_c = fresh(w13_lin), fresh(w2_lin)
    _load_w13(layer, w13_c)  # W13, then W2
    _load_w2(layer, w2_c)
    check(w13_c, w2_c)
    w13_d = fresh(w13_lin)
    _load_w13(layer, w13_d)  # and again
    check(w13_d, w2_c)


@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("lora_backend", ["triton_v2", "triton"], ids=["v2", "legacy"])
def test_w13_initial_load_and_reload_use_the_tp_local_layout(tp_rank, lora_backend):
    tp, f_full = 4, 256
    f_local = f_full // tp
    sink = _real_sink(
        InklingBatchDenseMLP,
        f=f_local,
        tp_size=tp,
        tp_rank=tp_rank,
        lora_backend=lora_backend,
    )
    contiguous = lora_backend == "triton_v2"
    assert sink._w13_gate_up_contiguous is contiguous
    _, _, w13_lin, w2_lin = _inputs(4, f=f_full)
    pointer = sink.w13_weight.data_ptr()

    def rank_rows(w13):
        rows = w13.view(N_EXPERTS, 2 * f_full, HIDDEN)[
            :, 2 * tp_rank * f_local : 2 * (tp_rank + 1) * f_local
        ]
        return (
            torch.stack([deinterleave_gate_up(e, dim=0) for e in rows])
            if contiguous
            else rows
        )

    def load_and_check(w13):
        original = w13.clone()
        _load_w13(sink, w13)
        assert torch.equal(w13, original)
        assert torch.equal(sink.w13_weight.data, rank_rows(w13))
        assert sink.w13_weight.data_ptr() == pointer

    load_and_check(w13_lin)
    _load_w2(sink, w2_lin)
    sink.process_weights_after_loading()
    assert torch.equal(sink.w13_weight.data, rank_rows(w13_lin))
    lin = sink._w2_lin.clone()
    g = torch.Generator(device=DEVICE).manual_seed(41)
    for _ in range(2):
        w13_new = (torch.randn(w13_lin.shape, generator=g, device=DEVICE) * 0.02).to(
            DTYPE
        )
        load_and_check(w13_new)
        sink.process_weights_after_loading()
        rows, _ = sink.get_bf16_linearized_weights()
        assert torch.equal(rows, rank_rows(w13_new).reshape_as(rows))
        assert rows.data_ptr() == pointer
        assert torch.equal(sink._w2_lin, lin) and sink._bf16_linearized_ready


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
