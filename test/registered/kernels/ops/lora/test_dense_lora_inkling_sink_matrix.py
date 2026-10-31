"""Extra-tier Inkling sink executor, tail-width, layout and TP-shard matrix.
Core replay and reload guards live in test_dense_lora_inkling_sink.py.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.srt.lora.dense.plan import (  # noqa: E402
    AFamily,
    BFamily,
    DensePlan,
    Overlap,
)
from sglang.srt.models.inkling_common.dense_mlp import (  # noqa: E402
    InklingBatchDenseMLP,
)
from sglang.srt.models.inkling_common.util import deinterleave_gate_up  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.kits.inkling_sink_lora_kit import (
    DEVICE,
    DTYPE,
    HIDDEN,
    N_EXPERTS,
    RANKS,
    RMAX,
    SLOTS,
    TOL,
    V2,
    _base_w13,
    _batch,
    _forward,
    _inputs,
    _layer,
    _load_w2,
    _load_w13,
    _pool,
    _prepare,
    _real_sink,
    _reference,
    _v2_backend,
)

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the sink kernels need CUDA"
)

# A pool rank whose shared gate/up bridge (2R = 32 columns) is narrower than
# the shrink's 64-wide column tile.
SMALL_RMAX, SMALL_RANKS = 16, [16, 8, 0, 8]


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
    expected = _reference(x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer)
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
    expected = _reference(x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer)
    torch.testing.assert_close(out.float(), expected, **TOL)


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
    expected = _reference(x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer)
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
