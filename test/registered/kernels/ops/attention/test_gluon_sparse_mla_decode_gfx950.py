"""gfx950 Gluon sparse-MLA decode vs an fp32 reference and the Triton kernel.

Covers every producer route (generic, and the exact-shape ov / sb3 / mt rows) at
8 and 16 local heads, -1 padded index rows, slots past the int32 row-offset
limit, and CUDA-graph replay.
"""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_ON_GFX950 = is_hip() and is_gfx95_supported()
pytestmark = pytest.mark.skipif(not _ON_GFX950, reason="gfx950 Gluon kernels")

if _ON_GFX950:
    from sglang.kernels.ops.attention.dsa import (
        gluon_sparse_mla_decode_gfx950 as kernels,
    )
    from sglang.kernels.ops.attention.dsa.gluon_sparse_mla_decode_gfx950 import (
        _TUNED_ROUTES,
        gluon_sparse_mla_decode,
        gluon_sparse_mla_decode_supported,
    )
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
        triton_sparse_mla_fwd,
    )

FP8 = torch.float8_e4m3fn
TOPK, D_V, D = 2048, 512, 576
SM_SCALE = 256**-0.5
# Every exact-shape row plus generic M on both sides of each route boundary.
SHAPES = [(8, m) for m in (1, 3, 4, 5, 6, 7, 12, 18, 24, 25, 30, 36, 42, 48, 49)]
SHAPES += [(8, m) for m in (60, 64, 97, 128)]
SHAPES += [(16, m) for m in (1, 6, 9, 30, 36, 42, 48, 72, 73, 120, 144, 193, 240)]


def _inputs(h, m, num_slots=1 << 16, high_slots=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(m, h, D, device="cuda", generator=g) * 0.5).to(FP8)
    kv = (torch.randn(num_slots, 1, D, device="cuda", generator=g) * 0.5).to(FP8)
    low = num_slots - (1 << 15) if high_slots else 0
    idx = torch.randint(low, num_slots, (m, TOPK), device="cuda", generator=g)
    idx = idx.to(torch.int32)
    # Short rows (-1 padded tail) and, when there are several rows, one empty row.
    idx[1::3, 1500:] = -1
    if m > 2:
        idx[2] = -1
    return q, kv, idx


def _reference(q, kv, idx):
    rows = kv[idx.clamp(min=0).long(), 0].float()  # [M, TOPK, D]
    s = torch.einsum("mhd,mkd->mhk", q.float(), rows) * SM_SCALE
    s = s.masked_fill((idx < 0)[:, None, :], float("-inf"))
    p = torch.nan_to_num(torch.softmax(s, dim=-1))
    return torch.einsum("mhk,mkd->mhd", p, rows[..., :D_V])


def _rel_rms(x, ref):
    return ((x.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("h,m", SHAPES)
def test_matches_reference_and_triton(h, m):
    q, kv, idx = _inputs(h, m)
    assert gluon_sparse_mla_decode_supported(q, kv, idx)
    out = gluon_sparse_mla_decode(q, kv, idx, SM_SCALE)
    assert out.shape == (1, m, h, D_V) and out.dtype == torch.bfloat16
    out = out[0]
    stock = triton_sparse_mla_fwd(
        q_nope=q[:, :, :D_V],
        q_rope=q[:, :, D_V:],
        kv=kv,
        indices=idx.unsqueeze(1),
        sm_scale=SM_SCALE,
        d_v=D_V,
    )[0]
    ref = _reference(q, kv, idx)
    # Both kernels round P to fp8; hold the new one to the Triton kernel's error.
    rows = (idx >= 0).any(dim=1)
    err = _rel_rms(out[rows], ref[rows])
    stock_err = _rel_rms(stock[rows], ref[rows])
    assert err <= max(0.05, 1.25 * stock_err), (err, stock_err)
    assert _rel_rms(out[rows], stock[rows].float()) <= 0.05
    assert torch.all(out[~rows] == 0)


@pytest.mark.parametrize("h,m", [(8, 6), (8, 60), (16, 36)])
def test_slots_past_int32_row_offsets(h, m):
    """slot * 576 overflows int32 above 3.73M slots; the kernels address in uint32."""
    q, kv, idx = _inputs(h, m, num_slots=4_000_000, high_slots=True)
    out = gluon_sparse_mla_decode(q, kv, idx, SM_SCALE)[0]
    assert _rel_rms(out, _reference(q, kv, idx)) <= 0.05


class _Recorder:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            ints = [a for a in args if isinstance(a, int)]
            self.log.append((self.name, tuple(grid), ints, kwargs))

        return launch


# 16-head target-verify launches of the measured routes: (producer, grid, runtime
# ints, BK) then (reduce, grid, SP). M = 6 * batch for batch 1, 4 and 16.
_H16_VERIFY_ROUTES = {
    6: [("ov", (192,), [], 32), ("reduce_tuned", (1536,), 32)],
    24: [("gen", (24, 1, 16), [16, 2], 64), ("reduce_gen", (24, 16, 4), 32)],
    96: [("gen", (96, 1, 4), [4, 8], 64), ("reduce_gen", (96, 16, 4), 32)],
}


@pytest.mark.parametrize("m", sorted(_H16_VERIFY_ROUTES))
def test_h16_verify_routes_match_the_measured_chain(m, monkeypatch):
    log = []
    for name, attr in (
        ("ov", "_sparse_mla_split_k_ov_kernel"),
        ("sb3", "_sparse_mla_split_k_sb3_kernel"),
        ("mt", "_sparse_mla_split_k_mt_kernel"),
        ("gen", "_sparse_mla_split_k_gen_kernel"),
        ("reduce_gen", "_sparse_mla_reduce_gen_kernel"),
        ("reduce_tuned", "_sparse_mla_reduce_tuned_kernel"),
    ):
        monkeypatch.setattr(kernels, attr, _Recorder(name, log))
    q = torch.empty(m, 16, D, dtype=FP8, device="meta")
    kv = torch.empty(1 << 20, 1, D, dtype=FP8, device="meta")
    idx = torch.empty(m, TOPK, dtype=torch.int32, device="meta")
    assert gluon_sparse_mla_decode_supported(q, kv, idx)
    gluon_sparse_mla_decode(q, kv, idx, SM_SCALE)
    (p_name, p_grid, p_ints, bk), (r_name, r_grid, sp) = _H16_VERIFY_ROUTES[m]
    assert [(n, g, i) for n, g, i, _ in log] == [
        (p_name, p_grid, p_ints),
        (r_name, r_grid, [p_ints[0]] if r_name == "reduce_gen" else []),
    ]
    assert log[0][3]["BK"] == bk and log[1][3]["SP"] == sp


# Row offsets are uint32 (slot * 576); the first slot that wraps is 2**32 // 576.
_LAST_SLOTS = (1 << 32) // D


def test_pool_past_uint32_row_offsets_falls_back():
    """A pool whose row offsets wrap uint32 must take the Triton path.

    Without this, every sequence with a slot >= 7,456,540 reads the wrong KV rows
    (rms error 0.2-0.7 on an 8M-slot pool), while the KV writes stay correct.
    """
    q = torch.empty(6, 8, D, dtype=FP8, device="meta")
    idx = torch.empty(6, TOPK, dtype=torch.int32, device="meta")
    for slots, ok in (
        (_LAST_SLOTS, True),
        (_LAST_SLOTS + 1, False),
        (8_000_000, False),
    ):
        kv = torch.empty(slots, 1, D, dtype=FP8, device="meta")
        assert gluon_sparse_mla_decode_supported(q, kv, idx) == ok, slots


@pytest.mark.parametrize("h,m", [(8, 6), (16, 36)])
def test_largest_supported_pool(h, m):
    q, kv, idx = _inputs(h, m, num_slots=_LAST_SLOTS, high_slots=True)
    idx[:, :8] = _LAST_SLOTS - 1
    assert gluon_sparse_mla_decode_supported(q, kv, idx)
    out = gluon_sparse_mla_decode(q, kv, idx, SM_SCALE)[0]
    rows = (idx >= 0).any(dim=1)
    assert _rel_rms(out[rows], _reference(q, kv, idx)[rows]) <= 0.05


@pytest.mark.parametrize(
    "h,m", [(8, 6), (8, 30), (8, 36), (16, 36), (8, 60), (16, 120)]
)
def test_cuda_graph_replay_matches_eager(h, m):
    q, kv, idx = _inputs(h, m)
    eager = gluon_sparse_mla_decode(q, kv, idx, SM_SCALE).clone()
    assert torch.equal(gluon_sparse_mla_decode(q, kv, idx, SM_SCALE), eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = gluon_sparse_mla_decode(q, kv, idx, SM_SCALE)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize(
    "h,m", sorted(_TUNED_ROUTES) + [(8, 1), (8, 64), (16, 1), (16, 120)]
)
def test_repeated_runs_are_bitwise_equal(h, m):
    """A shared-memory race inside a producer shows up as run-to-run differences."""
    q, kv, idx = _inputs(h, m)
    first = gluon_sparse_mla_decode(q, kv, idx, SM_SCALE).clone()
    for _ in range(8):
        assert torch.equal(gluon_sparse_mla_decode(q, kv, idx, SM_SCALE), first)


def test_supported_rejects_what_the_kernels_do_not_implement():
    q, kv, idx = _inputs(8, 6)
    assert gluon_sparse_mla_decode_supported(q, kv, idx)
    bad = (
        (q.to(torch.bfloat16), kv, idx),
        (q[:, :, :D_V], kv, idx),
        (q.transpose(0, 1).contiguous().transpose(0, 1), kv, idx),
        (q, kv.to(torch.bfloat16), idx),
        (q, kv, idx.long()),
        (q, kv, idx[:, :1024]),
        (q[:5], kv, idx),
    )
    for args in bad:
        assert not gluon_sparse_mla_decode_supported(*args)
    for h, m in ((12, 6), (8, 129), (16, 241), (32, 6)):
        q = torch.empty(m, h, D, dtype=FP8, device="cuda")
        idx = torch.zeros(m, TOPK, dtype=torch.int32, device="cuda")
        assert not gluon_sparse_mla_decode_supported(q, kv, idx), (h, m)
    for h, m in _TUNED_ROUTES:
        assert (h, m) in SHAPES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
