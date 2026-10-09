"""Inkling sink LoRA against a reference using raw, pre-scaled MoE pool buffers.
Covers both layouts, executor families, TP shards, and graph-visible adapter/base reloads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.moe.inkling_moe import silu_and_mul_triton  # noqa: E402
from sglang.srt.lora.backend.triton_backend import (  # noqa: E402
    TritonLoRABackend,
)
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend  # noqa: E402
from sglang.srt.lora.dense.plan import (  # noqa: E402
    AFamily,
    BFamily,
    DensePlan,
    Overlap,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.models.inkling_common.dense_mlp import (  # noqa: E402
    InklingBatchDenseMLP,
)
from sglang.srt.models.inkling_common.lora import (  # noqa: E402
    InklingBatchDenseMLPWithLoRA,
    InklingBatchDenseMLPWithLoRAV2,
)
from sglang.srt.models.inkling_common.util import deinterleave_gate_up  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402
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


V2 = InklingBatchDenseMLPWithLoRAV2


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
        s if ranks[s] > 0 else -1 for s, l in zip(slots, seq_lens) for _ in range(l)
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
    cls,
    *,
    f: int = F,
    tp_size: int = 1,
    tp_rank: int = 0,
    lora_backend=None,
    enable_lora=True,
    linearized_bf16=True,
):
    """Construct under the serving configuration before any weight is loaded."""
    if lora_backend is None:
        lora_backend = "triton_v2" if cls is V2 else "triton"
    default = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        with get_context().override_server_args(
            enable_lora=enable_lora, lora_backend=lora_backend
        ):
            layer = cls(
                n_shared_experts=N_EXPERTS,
                d_model=HIDDEN,
                shared_d_mlp=f * tp_size,
                layer_id=0,
                prefix=SINK_PREFIX,
                inference_moe_w13_interleaved=True,
                tp_rank=tp_rank,
                tp_size=tp_size,
                linearized_bf16=linearized_bf16,
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
    backend.batch_info.has_active_lora = True


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
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = forward()
    return graph, out


def _tiles(**overrides):
    tiles = dict(DensePlan().a_tiles)
    tiles.update(overrides)
    return tiles


PLANS = {
    "grouped": DensePlan(),
    "grouped_sorted_splitk_serial": DensePlan(a_tiles=_tiles(SPLIT_K=4)),
    "grouped_splitk_planes": DensePlan(a_tiles=_tiles(SPLIT_K=2, SPLIT_MODE="planes")),
    "grouped_overlap_a": DensePlan(overlap=Overlap.A),
    "grouped_overlap_delta": DensePlan(overlap=Overlap.AB_DELTA),
    "per_row_a_grouped_b": DensePlan(a_family=AFamily.PER_ROW, overlap=Overlap.A),
    "per_row": DensePlan(
        a_family=AFamily.PER_ROW, b_family=BFamily.PER_ROW, overlap=Overlap.A
    ),
    "all_slots": DensePlan(a_family=AFamily.ALL_SLOTS, b_family=BFamily.PER_ROW),
    "all_slots_overlap_a": DensePlan(
        a_family=AFamily.ALL_SLOTS, b_family=BFamily.PER_ROW, overlap=Overlap.A
    ),
    "all_slots_overlap_delta": DensePlan(
        a_family=AFamily.ALL_SLOTS, b_family=BFamily.PER_ROW, overlap=Overlap.AB_DELTA
    ),
}


def _pin_plan(backend, plan: DensePlan) -> None:
    backend.runner.plan_for = (
        lambda kind, max_rank, in_features=0, out_features=0, *, num_tokens: plan
    )


@pytest.mark.parametrize("rmax", [RMAX, SMALL_RMAX], ids=["r32", "r16"])
@pytest.mark.parametrize("f", [F, 96], ids=["f128", "f96_tail"])
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "prefill"])
@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
def test_v2_matches_reference(shared_outer, decode, f, rmax):
    ranks = RANKS if rmax == RMAX else SMALL_RANKS
    bufs = _pool(shared_outer, seed=3 + shared_outer, f=f, ranks=ranks, rmax=rmax)
    fb, slots, token_slots = _batch(decode, ranks=ranks)
    x, gammas, w13_lin, w2_lin = _inputs(fb.extend_num_tokens, f=f)
    backend = _v2_backend()
    _prepare(backend, fb, slots, ranks)
    layer = _layer(V2, backend, bufs, f=f)
    # One derived operand: the down shrink's A_cat (shared B) or the down
    # expand's B_cat (per-expert B); gate/up and the per-expert down shrink
    # read the pool.
    assert set(dict(layer.named_buffers())) == {"_down_cat"}
    out = _forward(layer, x, gammas, _base_w13(w13_lin), w2_lin)
    expected = _reference(
        layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
    )
    torch.testing.assert_close(out.float(), expected, **TOL)


@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
def test_v2_refresh_only_updates_requested_slots(shared_outer):
    bufs = _pool(shared_outer, seed=17)
    layer = _layer(V2, _v2_backend(), bufs)
    refresh = layer._down_refresh
    source, target = refresh
    before = target.clone()
    pointers = source.data_ptr(), target.data_ptr(), layer._down_cat.data_ptr()
    bindings = layer._gate_up, layer._down
    g = torch.Generator(device=DEVICE).manual_seed(18)
    for slot in (0, 2):
        _fill_slot(bufs, slot, 8, g)

    layer.on_lora_slots_updated({2})

    factor = bufs[2] if shared_outer else bufs[3]
    expected = before.clone()
    expected[2].copy_(factor[2].permute(1, 0, 2))
    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    assert not torch.equal(target[2], before[2])
    assert pointers == (
        source.data_ptr(),
        target.data_ptr(),
        layer._down_cat.data_ptr(),
    )
    assert layer._gate_up is bindings[0] and layer._down is bindings[1]
    assert layer._down_refresh is refresh


@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
def test_v2_failed_rebind_keeps_the_active_binding(shared_outer):
    bufs = _pool(shared_outer, seed=19)
    backend = _v2_backend()
    layer = _layer(V2, backend, bufs)
    bindings = layer._gate_up, layer._down, layer._down_refresh
    derived = layer._down_cat
    before = derived.clone()
    incompatible = _pool(shared_outer, seed=20, ranks=SMALL_RANKS, rmax=SMALL_RMAX)

    with pytest.raises(RuntimeError, match="pool shape changed"):
        layer.set_lora_info(*incompatible)

    assert layer.set_lora
    assert layer._gate_up is bindings[0] and layer._down is bindings[1]
    assert layer._down_refresh is bindings[2]
    assert layer._down_cat is derived
    torch.testing.assert_close(derived, before, rtol=0, atol=0)
    fb, slots, token_slots = _batch(True)
    _prepare(backend, fb, slots)
    x, gammas, w13_lin, w2_lin = _inputs(fb.extend_num_tokens)
    out = _forward(layer, x, gammas, _base_w13(w13_lin), w2_lin)
    expected = _reference(
        layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
    )
    torch.testing.assert_close(out.float(), expected, **TOL)


@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
def test_v2_no_active_adapter_is_the_base(shared_outer):
    """An empty adapter route must leave the base output unchanged."""
    bufs = _pool(shared_outer, seed=7)
    fb, slots, token_slots = _batch(True, slots=[2] * 13)
    assert all(s < 0 for s in token_slots)
    x, gammas, w13_lin, w2_lin = _inputs(fb.extend_num_tokens)
    backend = _v2_backend()
    _prepare(backend, fb, slots)
    layer = _layer(V2, backend, bufs)
    out = _forward(layer, x, gammas, _base_w13(w13_lin), w2_lin)
    base = _reference(
        layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
    )
    torch.testing.assert_close(out.float(), base, **TOL)


@pytest.mark.parametrize("rmax", [RMAX, SMALL_RMAX], ids=["r32", "r16"])
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "prefill"])
@pytest.mark.parametrize("plan_name", sorted(PLANS))
@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
def test_v2_every_executor_family(shared_outer, plan_name, decode, rmax):
    """Exercise every sink executor with full and partial shrink tiles."""
    if not shared_outer and plan_name.startswith("all_slots"):
        # The per-expert sink down is a windowed shrink; an all_slots row cannot
        # serve it and the table resolver never pairs them (the forced pairing is
        # test_windowed_shrink_rejects_a_forced_all_slots_plan).
        pytest.skip("all_slots cannot serve the windowed per-expert sink down")
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
    layer = _layer(V2, backend, bufs)
    w13_base = _base_w13(w13_lin)
    graph, out = _capture(backend, lambda: _forward(layer, x, gammas, w13_base, w2_lin))

    def check(token_slots):
        graph.replay()
        expected = _reference(
            layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
        )
        torch.testing.assert_close(out.float(), expected, **TOL)

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


@pytest.mark.parametrize("active", [True, False], ids=["adapters", "no_adapter"])
@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
@pytest.mark.parametrize(
    "prefill", [False, True], ids=["decode_graph", "prefill_graph"]
)
def test_w13_reload_through_the_loader_reaches_the_captured_graph(
    prefill, shared_outer, active
):
    """Reloads reach the next replay without post-load processing or an eager repair."""
    bufs = _pool(shared_outer, seed=21)
    slots = None if active else ([2] * 4 if prefill else [2] * 13)
    fb, slots, token_slots = _batch(not prefill, slots=slots)
    assert active or all(s < 0 for s in token_slots)
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


@pytest.mark.parametrize("linearized_bf16", [False, True])
@pytest.mark.parametrize("enable_lora", [False, True])
@pytest.mark.parametrize("lora_backend", ["triton_v2", "triton"], ids=["v2", "legacy"])
def test_sink_constructor_selects_gate_up_layout(
    lora_backend, enable_lora, linearized_bf16
):
    sink = _real_sink(
        InklingBatchDenseMLP,
        lora_backend=lora_backend,
        enable_lora=enable_lora,
        linearized_bf16=linearized_bf16,
    )
    assert sink._w13_gate_up_contiguous is (
        linearized_bf16 and enable_lora and lora_backend == "triton_v2"
    )


@pytest.mark.parametrize("tp_rank", range(4))
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


def test_contiguous_gate_up_activation_matches_interleaved():
    f = 48
    sink = _real_sink(V2, f=f)
    y = torch.randn(7, N_EXPERTS, 2 * f, device=DEVICE, dtype=DTYPE)
    gammas = torch.rand(7, N_EXPERTS, device=DEVICE, dtype=DTYPE) + 0.5
    y_interleaved = (
        y.view(7, N_EXPERTS, 2, f).transpose(-1, -2).reshape(7, N_EXPERTS, 2 * f)
    )
    torch.testing.assert_close(
        sink._swiglu(y, gammas), _swiglu_interleaved(y_interleaved, gammas)
    )


def test_windowed_shrink_rejects_a_forced_all_slots_plan():
    """A forced all_slots plan for a windowed site must fail before launch."""
    runner = _v2_backend().runner
    x = torch.zeros(4, 8, device=DEVICE, dtype=DTYPE)
    a = torch.zeros(2, 16, 8, device=DEVICE, dtype=DTYPE)
    with pytest.raises(NotImplementedError, match="cannot window"):
        runner.run_a(x, a, stack=1, plan=PLANS["all_slots"], windowed=True)


def test_legacy_wrapper_runs_the_experimental_path_without_engine_operands():
    """The legacy shared-outer wrapper matches the reference using its own operands."""
    bufs = _pool(True, seed=4)
    fb, slots, token_slots = _batch(True)
    x, gammas, w13_lin, w2_lin = _inputs(fb.extend_num_tokens)
    legacy = TritonLoRABackend(SLOTS, DEVICE)
    legacy.is_moe_lora = True
    legacy.prepare_lora_batch(fb, slots, RANKS, [1.0] * SLOTS, use_cuda_graph=False)
    legacy.batch_info.has_active_lora = True
    layer = _layer(InklingBatchDenseMLPWithLoRA, legacy, bufs)
    out = _forward(layer, x, gammas, w13_lin, w2_lin)
    expected = _reference(layer, x, gammas, w13_lin, w2_lin, bufs, token_slots, True)
    torch.testing.assert_close(out.float(), expected, **TOL)
    assert set(dict(layer.named_buffers())) == {"_w1_delta", "_a_cat"}
    assert not any(name.startswith("_sink") for name in vars(layer))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
