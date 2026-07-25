"""Correctness tests for the fused MLA KV-scatter + absorbed-q concat kernel.

The fused kernel must be bit-exact against the composition of the two
audited kernels it replaces (``set_mla_kv_buffer`` + ``concat_mla_absorb_q``)
— both are pure byte movement, so any mismatch is a bug, not tolerance.
Inputs use production-like strided views (k halves sliced from one latent
row, q halves sliced from one [B, H, 576] projection) and the graph
capture/replay pattern the serving decode path runs under.

``covered()`` guards are derived alignment properties: the device code does
16-byte vector accesses on kv/nope/q rows and 4-byte accesses on rope rows,
so misaligned bases or row pitches must be rejected (fall back) rather than
fault — the failure class behind the TP16 fused-decode crash (PR #208).
"""

import pytest
import torch

from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    can_use_set_mla_kv_concat_q,
    covered,
    set_mla_kv_concat_q,
)
from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

NOPE_DIM = 512
ROPE_DIM = 64
TOTAL_DIM = NOPE_DIM + ROPE_DIM
PAGES = 2048


def _require_supported():
    if not can_use_set_mla_kv_concat_q(NOPE_DIM * 2, ROPE_DIM * 2):
        pytest.skip("fused set_mla_kv_concat_q requires SM90+")


def _make_inputs(bs: int, heads: int, loc_dtype: torch.dtype, seed: int):
    """Production-like strided inputs: k halves are slices of one latent row,
    q halves are slices of one [B, H, TOTAL_DIM] projection output."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(0.1)
            .to(torch.bfloat16)
        )

    pool = rand(PAGES, TOTAL_DIM)
    latent = rand(bs, TOTAL_DIM)
    q_all = rand(bs, heads, TOTAL_DIM)
    loc = torch.randperm(PAGES, generator=gen, device="cuda")[:bs].to(loc_dtype)
    return (
        pool,
        loc,
        latent[:, :NOPE_DIM],
        latent[:, NOPE_DIM:],
        q_all[..., :NOPE_DIM],
        q_all[..., NOPE_DIM:],
    )


def _reference(pool, loc, k_nope, k_rope, q_nope, q_rope):
    set_mla_kv_buffer(pool, loc, k_nope, k_rope)
    return concat_mla_absorb_q(q_nope, q_rope)


@pytest.mark.parametrize("loc_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(
    "bs,heads",
    [(1, 8), (12, 8), (64, 8), (300, 16), (1024, 8)],  # spans nw=4 and nw=8 tiers
)
def test_bitexact_vs_two_kernels(bs, heads, loc_dtype):
    _require_supported()
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(bs, heads, loc_dtype, bs)
    pool_ref = pool.clone()

    assert covered(pool, loc, k_nope, k_rope, q_nope, q_rope)
    query = set_mla_kv_concat_q(pool, loc, k_nope, k_rope, q_nope, q_rope)
    query_ref = _reference(pool_ref, loc, k_nope, k_rope, q_nope, q_rope)
    torch.cuda.synchronize()

    assert torch.equal(pool, pool_ref), "KV pool rows diverge from two-kernel path"
    assert torch.equal(query, query_ref), "concat query diverges from two-kernel path"


def test_graph_capture_replay_bitexact():
    """Serving decode runs the fused call inside a CUDA graph: the loc buffer
    and activations are refilled between replays while the captured kernel
    keeps the captured pointers. Replay after refill must match the eager
    two-kernel result on the new values."""
    _require_supported()
    bs, heads = 64, 8
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(bs, heads, torch.int64, 7)

    # Warmup (JIT + allocator) outside the graph.
    _ = set_mla_kv_concat_q(pool, loc, k_nope, k_rope, q_nope, q_rope)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query = set_mla_kv_concat_q(pool, loc, k_nope, k_rope, q_nope, q_rope)

    for round_seed in (100, 101):
        (
            pool_new,
            loc_new,
            k_nope_new,
            k_rope_new,
            q_nope_new,
            q_rope_new,
        ) = _make_inputs(bs, heads, torch.int64, round_seed)
        # Refill the captured buffers in place (what replay-prep does).
        pool.copy_(pool_new)
        loc.copy_(loc_new)
        k_nope.copy_(k_nope_new)
        k_rope.copy_(k_rope_new)
        q_nope.copy_(q_nope_new)
        q_rope.copy_(q_rope_new)
        pool_ref = pool.clone()

        graph.replay()
        query_ref = _reference(pool_ref, loc, k_nope, k_rope, q_nope, q_rope)
        torch.cuda.synchronize()

        assert torch.equal(pool, pool_ref)
        assert torch.equal(query, query_ref)


def test_zero_batch():
    _require_supported()
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(0, 8, torch.int64, 3)
    pool_ref = pool.clone()
    query = set_mla_kv_concat_q(pool, loc, k_nope, k_rope, q_nope, q_rope)
    torch.cuda.synchronize()
    assert query.shape == (0, 8, TOTAL_DIM)
    assert torch.equal(pool, pool_ref)


def test_covered_rejects_unsupported_layouts():
    """Alignment/layout gates: each rejected layout would make the device
    code's vector accesses fault (16B on kv/nope/q rows, 4B on rope rows) or
    read the wrong elements, so ``covered()`` must send it down the
    two-kernel fallback instead."""
    _require_supported()
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(12, 8, torch.int64, 9)
    assert covered(pool, loc, k_nope, k_rope, q_nope, q_rope)

    # kv pool with a 577-element row pitch: row stride 1154B % 16 != 0.
    raw = torch.zeros(PAGES * 577, device="cuda", dtype=torch.bfloat16)
    pool_odd = torch.as_strided(raw, (PAGES, TOTAL_DIM), (577, 1))
    assert not covered(pool_odd, loc, k_nope, k_rope, q_nope, q_rope)

    # kv pool base offset by one element: base % 16 != 0.
    pool_off = torch.as_strided(raw[1:], (PAGES, TOTAL_DIM), (577, 1))
    assert not covered(pool_off, loc, k_nope, k_rope, q_nope, q_rope)

    # q_nope base offset by one element: base % 16 != 0 for int4 loads.
    q_flat = torch.zeros(12 * 8 * NOPE_DIM + 1, device="cuda", dtype=torch.bfloat16)
    q_nope_off = torch.as_strided(
        q_flat[1:], (12, 8, NOPE_DIM), (8 * NOPE_DIM, NOPE_DIM, 1)
    )
    assert not covered(pool, loc, k_nope, k_rope, q_nope_off, q_rope)

    # Odd q_rope row stride: rope rows land 2B-aligned, int loads need 4B.
    r_flat = torch.zeros(12 * 8 * (ROPE_DIM + 1), device="cuda", dtype=torch.bfloat16)
    q_rope_odd = torch.as_strided(
        r_flat, (12, 8, ROPE_DIM), (8 * (ROPE_DIM + 1), ROPE_DIM + 1, 1)
    )
    assert not covered(pool, loc, k_nope, k_rope, q_nope, q_rope_odd)

    # Batch mismatch between loc and activations.
    assert not covered(pool, loc[:6], k_nope, k_rope, q_nope, q_rope)

    # Non-contiguous last dim.
    q_strided_last = torch.zeros(
        12, 8, NOPE_DIM * 2, device="cuda", dtype=torch.bfloat16
    )[..., ::2]
    assert not covered(pool, loc, k_nope, k_rope, q_strided_last, q_rope)

    # Unsupported loc dtype.
    assert not covered(pool, loc.to(torch.int16), k_nope, k_rope, q_nope, q_rope)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
