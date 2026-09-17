"""Correctness tests for the fused fp8 quantize + KV-scatter + q-concat kernel.

The fused kernel replaces the fp8 decode path's aten chain
(concat_mla_absorb_q -> .to(float8_e4m3fn) x3 -> KV-row write), so it must be
BIT-equal to that chain: the conversion is round-to-nearest bf16 -> e4m3 with
satfinite semantics (finite |x|>448 -> +-448, 0x7E/0xFE), bit-equal to
clamp(+-448).to(float8_e4m3fn), and everything else is byte movement.
Covers production-like strided views (k halves sliced from one latent row,
q halves sliced from one [B, H, 576] projection), int32/int64 locs, CUDA
graph capture/replay with refilled buffers, and the covered_fp8() alignment
guards (base + stride, the PR #208 failure class).
"""

import pytest
import torch

from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    can_use_set_mla_kv_concat_q_fp8,
    covered_fp8,
    set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general
from sglang.kernels.ops.quantization.fp8_utils import to_fp8_satfinite
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

NOPE, ROPE, TOTAL, PAGES = 512, 64, 576, 2048


def _require_supported():
    if not can_use_set_mla_kv_concat_q_fp8():
        pytest.skip("fused fp8 set_mla_kv_concat_q requires SM90+")


def _make_inputs(bs, heads, loc_dtype, seed, scale=100.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, scale=1.0):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(scale)
            .to(torch.bfloat16)
        )

    pool = torch.zeros(PAGES, TOTAL, device="cuda", dtype=torch.float8_e4m3fn)
    # Mix magnitudes incl. values beyond the e4m3 range to exercise saturation.
    latent = rand(bs, TOTAL, scale=scale)
    q_all = rand(bs, heads, TOTAL, scale=scale)
    loc = torch.randperm(PAGES, generator=gen, device="cuda")[:bs].to(loc_dtype)
    return (
        pool,
        loc,
        latent[:, :NOPE],
        latent[:, NOPE:],
        q_all[..., :NOPE],
        q_all[..., NOPE:],
    )


def _reference(pool, loc, k_nope, k_rope, q_nope, q_rope):
    """The exact aten chain the fusion replaces."""
    q_fp8 = to_fp8_satfinite(
        concat_mla_absorb_q_general(q_nope, q_rope), torch.float8_e4m3fn
    )
    row = torch.cat(
        [
            to_fp8_satfinite(k_nope, torch.float8_e4m3fn),
            to_fp8_satfinite(k_rope, torch.float8_e4m3fn),
        ],
        dim=-1,
    )
    pool[loc.long()] = row
    return q_fp8


@pytest.mark.parametrize("loc_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("bs,heads", [(1, 8), (12, 8), (64, 8), (300, 16), (1024, 8)])
def test_bitexact_vs_aten_chain(bs, heads, loc_dtype):
    _require_supported()
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(bs, heads, loc_dtype, bs)
    pool_ref = pool.clone()

    assert covered_fp8(pool, loc, k_nope, k_rope, q_nope, q_rope)
    query = set_mla_kv_concat_q_fp8(pool, loc, k_nope, k_rope, q_nope, q_rope)
    query_ref = _reference(pool_ref, loc, k_nope, k_rope, q_nope, q_rope)
    torch.cuda.synchronize()

    assert torch.equal(pool.view(torch.uint8), pool_ref.view(torch.uint8))
    assert torch.equal(query.view(torch.uint8), query_ref.view(torch.uint8))


def test_graph_capture_replay_bitexact():
    _require_supported()
    bs, heads = 64, 8
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(bs, heads, torch.int64, 7)

    _ = set_mla_kv_concat_q_fp8(pool, loc, k_nope, k_rope, q_nope, q_rope)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query = set_mla_kv_concat_q_fp8(pool, loc, k_nope, k_rope, q_nope, q_rope)

    for seed in (100, 101):
        p2, l2, kn2, kr2, qn2, qr2 = _make_inputs(bs, heads, torch.int64, seed)
        pool.copy_(p2)
        loc.copy_(l2)
        k_nope.copy_(kn2)
        k_rope.copy_(kr2)
        q_nope.copy_(qn2)
        q_rope.copy_(qr2)
        pool_ref = pool.clone()

        graph.replay()
        query_ref = _reference(pool_ref, loc, k_nope, k_rope, q_nope, q_rope)
        torch.cuda.synchronize()

        assert torch.equal(pool.view(torch.uint8), pool_ref.view(torch.uint8))
        assert torch.equal(query.view(torch.uint8), query_ref.view(torch.uint8))


@pytest.mark.parametrize("world,rank", [(8, 0), (8, 3), (8, 7), (4, 1)])
def test_dcp_virtual_loc_semantics(world, rank):
    """Under DCP the loc is VIRTUAL: only the owner rank
    (loc % world == rank) writes, at physical row loc // world — mirroring
    the triton writer's is_valid mask. Bug regression: the first fused
    version treated virtual locs as physical rows, writing up to world-x
    past the pool (illegal memory access at bs=64 DCP8 serving)."""
    _require_supported()
    bs, heads = 64, 8
    gen = torch.Generator(device="cuda").manual_seed(world * 10 + rank)
    pool = torch.zeros(PAGES, TOTAL, device="cuda", dtype=torch.float8_e4m3fn)
    latent = (
        torch.randn(bs, TOTAL, generator=gen, device="cuda", dtype=torch.float32)
        .mul(2.0)
        .to(torch.bfloat16)
    )
    q_all = (
        torch.randn(bs, heads, TOTAL, generator=gen, device="cuda", dtype=torch.float32)
        .mul(2.0)
        .to(torch.bfloat16)
    )
    # Virtual locs spread across the virtual space [0, PAGES * world).
    loc = torch.randperm(PAGES * world, generator=gen, device="cuda")[:bs]
    k_nope, k_rope = latent[:, :NOPE], latent[:, NOPE:]
    q_nope, q_rope = q_all[..., :NOPE], q_all[..., NOPE:]
    pool_ref = pool.clone()

    query = set_mla_kv_concat_q_fp8(
        pool,
        loc,
        k_nope,
        k_rope,
        q_nope,
        q_rope,
        dcp_world_size=world,
        dcp_rank=rank,
    )
    # Reference: owner-masked scatter at loc // world; query for every token.
    owned = (loc % world) == rank
    row = torch.cat(
        [
            to_fp8_satfinite(k_nope, torch.float8_e4m3fn),
            to_fp8_satfinite(k_rope, torch.float8_e4m3fn),
        ],
        dim=-1,
    )
    pool_ref[(loc[owned] // world).long()] = row[owned]
    query_ref = to_fp8_satfinite(
        concat_mla_absorb_q_general(q_nope, q_rope), torch.float8_e4m3fn
    )
    torch.cuda.synchronize()

    assert torch.equal(pool.view(torch.uint8), pool_ref.view(torch.uint8))
    assert torch.equal(query.view(torch.uint8), query_ref.view(torch.uint8))


def test_covered_rejects_unsupported_layouts():
    _require_supported()
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(12, 8, torch.int64, 9)
    assert covered_fp8(pool, loc, k_nope, k_rope, q_nope, q_rope)

    # Odd pool row pitch: 577 % 16 != 0 for the TMA store.
    raw = torch.zeros(PAGES * 577, device="cuda", dtype=torch.float8_e4m3fn)
    pool_odd = torch.as_strided(raw, (PAGES, TOTAL), (577, 1))
    assert not covered_fp8(pool_odd, loc, k_nope, k_rope, q_nope, q_rope)

    # q_nope base offset by one element: 16B int4 loads fault.
    q_flat = torch.zeros(12 * 8 * NOPE + 1, device="cuda", dtype=torch.bfloat16)
    q_off = torch.as_strided(q_flat[1:], (12, 8, NOPE), (8 * NOPE, NOPE, 1))
    assert not covered_fp8(pool, loc, k_nope, k_rope, q_off, q_rope)

    # bf16 pool (wrong dtype for the fp8 kernel).
    pool_bf16 = torch.zeros(PAGES, TOTAL, device="cuda", dtype=torch.bfloat16)
    assert not covered_fp8(pool_bf16, loc, k_nope, k_rope, q_nope, q_rope)

    # Batch mismatch between loc and activations.
    assert not covered_fp8(pool, loc[:6], k_nope, k_rope, q_nope, q_rope)


def test_overflow_saturates_no_nan_bytes():
    _require_supported()
    bs, heads = 4, 8
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(
        bs, heads, torch.int64, 11, scale=1.0
    )
    bf16_max = torch.finfo(torch.bfloat16).max
    k_nope[0, 0] = 464.0
    k_nope[0, -1] = 500.0
    k_rope[0, 0] = -1000.0
    k_rope[0, -1] = bf16_max
    q_nope[0, 0, 0] = -bf16_max
    q_nope[0, 0, -1] = 464.0
    q_rope[0, 0, 0] = -464.0
    q_rope[0, 0, -1] = -1000.0

    query = set_mla_kv_concat_q_fp8(pool, loc, k_nope, k_rope, q_nope, q_rope)
    pool_rows = pool[loc.long()]
    pool_row = pool_rows[0]
    expected_pool = {
        0: 0x7E,
        NOPE - 1: 0x7E,
        NOPE: 0xFE,
        TOTAL - 1: 0x7E,
    }
    expected_query = {
        0: 0xFE,
        NOPE - 1: 0x7E,
        NOPE: 0xFE,
        TOTAL - 1: 0xFE,
    }
    for lane, byte in expected_pool.items():
        assert int(pool_row[lane].view(torch.uint8)) == byte
    for lane, byte in expected_query.items():
        assert int(query[0, 0, lane].view(torch.uint8)) == byte

    assert not ((pool_rows.view(torch.uint8) & 0x7F) == 0x7F).any()
    assert not ((query.view(torch.uint8) & 0x7F) == 0x7F).any()
    assert torch.equal(
        pool_row[[0, NOPE - 1, NOPE, TOTAL - 1]].to(torch.float32),
        torch.tensor([448.0, 448.0, -448.0, 448.0], device="cuda"),
    )
    assert torch.equal(
        query[0, 0, [0, NOPE - 1, NOPE, TOTAL - 1]].to(torch.float32),
        torch.tensor([-448.0, 448.0, -448.0, -448.0], device="cuda"),
    )


def test_mixed_row_overflow_lane_leaves_others_unchanged():
    _require_supported()
    pool, loc, k_nope, k_rope, q_nope, q_rope = _make_inputs(
        1, 8, torch.int64, 12, scale=1.0
    )
    k_nope[0, 0] = 500.0
    k_rope[0, -1] = -1000.0
    q_nope[0, 0, 0] = 500.0
    q_rope[0, 0, -1] = -1000.0

    query = set_mla_kv_concat_q_fp8(pool, loc, k_nope, k_rope, q_nope, q_rope)
    pool_row = pool[loc[0].long()]
    pool_ref = torch.cat(
        [
            k_nope.to(torch.float8_e4m3fn),
            k_rope.to(torch.float8_e4m3fn),
        ],
        dim=-1,
    )[0]
    query_ref = concat_mla_absorb_q_general(q_nope, q_rope).to(torch.float8_e4m3fn)
    pool_mask = torch.ones(TOTAL, dtype=torch.bool, device="cuda")
    pool_mask[[0, TOTAL - 1]] = False
    query_mask = torch.ones((8, TOTAL), dtype=torch.bool, device="cuda")
    query_mask[0, [0, TOTAL - 1]] = False
    assert int(pool_row[0].view(torch.uint8)) == 0x7E
    assert int(pool_row[-1].view(torch.uint8)) == 0xFE
    assert int(query[0, 0, 0].view(torch.uint8)) == 0x7E
    assert int(query[0, 0, -1].view(torch.uint8)) == 0xFE
    assert torch.equal(
        pool_row[pool_mask].view(torch.uint8), pool_ref[pool_mask].view(torch.uint8)
    )
    assert torch.equal(
        query[0][query_mask].view(torch.uint8),
        query_ref[0][query_mask].view(torch.uint8),
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
