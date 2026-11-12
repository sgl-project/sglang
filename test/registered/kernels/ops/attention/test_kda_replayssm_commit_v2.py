# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""KDA ReplaySSM commit v2 (full-K tiles, request grid-stride) vs the stock exact-fold commit and a float64 fold.

K3.1 TP8 shapes: 12 heads, K = V = 128, FP32 state and rings (L=16; test_commit_v2_bf16_state: BF16 state and raw
k / v rings, its tolerances in its docstring), 3 layers in one launch (the all-layers entry).
Cases cover full commits (P = B), sparse commits (small P, most commit_lens 0), the extra_buffer track snapshot, a
null slot (-1) and commit_len 0 rows.

Tolerances, fixed before the first run:
- v2 vs stock: max|dS| / max|S| <= 1e-5 over every touched slot; with one warp per program (the stock tile and
  reduction tree) the whole pool must be bit-identical.
- v2 vs float64 fold of the same ring records (independent torch, definition order): <= 1e-5 relative.
- rows with commit_len 0, the null slot and slots no request names: bit-identical to the input.
- track slot: equals the float64 state after step `steps_to_track` (<= 1e-5) for crossing rows, untouched otherwise.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from sglang.kernels.ops.attention.linear.kda_replayssm_commit_v2 import (  # noqa: E402
    commit_kda_replayssm_v2_all_layers,
)
from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (  # noqa: E402
    _sparse_commit_block_v,
    commit_kda_replayssm_spec_all_layers,
)

DEV = "cuda"
H, K, L, LAYERS = 12, 128, 16, 3
TOL = 1e-5


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def _pool(B, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    slots = B + 3
    S = torch.randn(LAYERS, slots, H, K, K, device=DEV, generator=g) * 0.05
    rv = torch.randn(LAYERS, slots, H, L, K, device=DEV, generator=g)
    rk = torch.randn(LAYERS, slots, H, L, K, device=DEV, generator=g)
    gk = -torch.rand(LAYERS, slots, H, L, K, device=DEV, generator=g)
    beta = torch.rand(LAYERS, slots, H, L, device=DEV, generator=g)
    return S, rv, rk, gk, beta


def _fold64(S, rv, rk, gk, beta, layer, slot, n, snap_at=-1):
    h = S[layer, slot].double().clone()
    snap = None
    for t in range(n):
        k = rk[layer, slot, :, t].double()
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        h = h * torch.exp(gk[layer, slot, :, t].double()).unsqueeze(1)
        v = (rv[layer, slot, :, t].double() - torch.einsum("hvk,hk->hv", h, k)) * beta[
            layer, slot, :, t
        ].double().unsqueeze(1)
        h = h + v.unsqueeze(2) * k.unsqueeze(1)
        if t == snap_at:
            snap = h.clone()
    return h, snap


@pytest.mark.parametrize(
    "B,P,block_v,num_warps,sparse",
    [
        (16, None, 16, 1, False),
        (16, None, 8, 1, False),
        (16, None, 32, 4, False),
        (40, 4, 16, 1, True),
        (40, None, 16, 1, True),
    ],
)
@pytest.mark.parametrize("track", [False, True])
def test_commit_v2(B, P, block_v, num_warps, sparse, track):
    S, rv, rk, gk, beta = _pool(B, seed=B + (P or 0) + block_v)
    gen = torch.Generator().manual_seed(5)
    idx = torch.arange(1, B + 1, dtype=torch.int32)
    idx[B // 2] = -1  # null slot (CUDA-graph padding)
    lens = torch.randint(1, 9, (B,), generator=gen, dtype=torch.int32)
    lens[0], lens[1] = 1, 8
    if sparse:
        keep = torch.zeros(B, dtype=torch.bool)
        # rows on different residues mod P (and mod P + 1) so a wrong stride skips one
        keep[[0, 1, 9, 19, B - 3]] = True
        lens = torch.where(keep, lens, torch.zeros_like(lens))
    lens[2] = 0
    track_idx = steps = None
    if track:
        track_idx = torch.full((B,), -1, dtype=torch.int32)
        steps = torch.full((B,), -1, dtype=torch.int32)
        # row 1 crosses at step 5 into track slot B+2; row 0 crosses at step 0 into slot B+1
        track_idx[0], steps[0] = B + 1, 0
        track_idx[1], steps[1] = B + 2, 5
        track_idx, steps = track_idx.to(DEV), steps.to(DEV)
    idx_d, lens_d = idx.to(DEV), lens.to(DEV)

    S0 = S.clone()
    S_v2 = S.clone()
    commit_kda_replayssm_v2_all_layers(
        S_v2, rv, rk, gk, beta, idx_d, lens_d, max_cache_len=L, num_k_heads=H,
        mamba_track_indices=track_idx, mamba_steps_to_track=steps, null_block_id=-1,
        req_programs=P, block_v=block_v, num_warps=num_warps,
    )  # fmt: skip
    S_st = S.clone()
    commit_kda_replayssm_spec_all_layers(
        S_st, rv, rk, gk, beta, idx_d, lens_d, max_cache_len=L, num_k_heads=H,
        mamba_track_indices=track_idx, mamba_steps_to_track=steps, null_block_id=-1,
    )  # fmt: skip
    torch.cuda.synchronize()

    touched = [int(idx[i]) for i in range(B) if idx[i] >= 0 and lens[i] > 0]
    tracked = [B + 1, B + 2] if track else []
    assert touched, "test must fold something"
    for s in touched + tracked:
        e = _rel(S_v2[:, s], S_st[:, s])
        assert e <= TOL, f"slot {s}: v2 vs stock rel {e:.3e}"
    if num_warps == 1:
        # Same tile and one-warp reduction tree as the stock kernel: bit-identical.
        assert torch.equal(S_v2, S_st), "one-warp v2 must be bit-identical to stock"
    for i in range(B):
        s, n = int(idx[i]), int(lens[i])
        if s < 0 or n <= 0:
            continue
        for layer in range(LAYERS):
            snap_at = int(steps[i]) if track and i in (0, 1) else -1
            ref, snap = _fold64(S0, rv, rk, gk, beta, layer, s, n, snap_at)
            e = _rel(S_v2[layer, s], ref)
            assert e <= TOL, f"slot {s} layer {layer}: v2 vs float64 rel {e:.3e}"
            if snap is not None and snap_at < n:
                t_slot = int(track_idx[i])
                e = _rel(S_v2[layer, t_slot], snap)
                assert e <= TOL, f"track slot {t_slot} layer {layer}: rel {e:.3e}"
    untouched = torch.ones(S.shape[1], dtype=torch.bool)
    untouched[touched + tracked] = False
    if track and int(lens[1]) <= 5:
        untouched[B + 2] = True  # row 1 never reaches step 5: its track slot must stay untouched
    assert torch.equal(S_v2[:, untouched], S0[:, untouched]), "v2 wrote a slot it must not touch"


def _ulp_bf16(x):
    return torch.ldexp(torch.ones_like(x), torch.frexp(x).exponent - 8)


@pytest.mark.parametrize("track", [False, True])
def test_commit_v2_bf16_state(track):
    """BF16 state and BF16 raw k / v rings (the SGLANG_OPT_KDA_CUTE_VERIFY_BF16_STATE path, which fold-mode flushes
    commit through v2). Tolerances, fixed before the first run: one-warp v2 equals the stock commit bit for bit (same
    FP32 replay, one RNE rounding per commit); every committed element (and track snapshot) is within one BF16 ulp
    + 1e-6 absolute of the float64 fold of the same BF16 inputs. v2 runs at the V tile the sparse commit uses for
    BF16 state: on BF16 state the stock commit's bits depend on the tile (1-2 BF16 ulps in a few elements of these
    inputs at tile 16 vs 8). Catches v2 refusing BF16 state (its former FP32-only assert), any extra rounding or other
    arithmetic change against the stock commit, and a sparse commit tile that differs from the stock one."""
    B = 16
    S, rv, rk, gk, beta = _pool(B, seed=77)
    S, rv, rk = S.bfloat16(), rv.bfloat16(), rk.bfloat16()
    gen = torch.Generator().manual_seed(6)
    idx = torch.arange(1, B + 1, dtype=torch.int32)
    idx[B // 2] = -1
    lens = torch.randint(1, 9, (B,), generator=gen, dtype=torch.int32)
    lens[0], lens[1], lens[2] = 1, 8, 0
    track_idx = steps = None
    if track:
        track_idx = torch.full((B,), -1, dtype=torch.int32)
        steps = torch.full((B,), -1, dtype=torch.int32)
        track_idx[0], steps[0] = B + 1, 0
        track_idx[1], steps[1] = B + 2, 5
        track_idx, steps = track_idx.to(DEV), steps.to(DEV)
    idx_d, lens_d = idx.to(DEV), lens.to(DEV)

    S0 = S.clone()
    S_v2 = S.clone()
    commit_kda_replayssm_v2_all_layers(
        S_v2, rv, rk, gk, beta, idx_d, lens_d, max_cache_len=L, num_k_heads=H,
        mamba_track_indices=track_idx, mamba_steps_to_track=steps, null_block_id=-1,
        block_v=_sparse_commit_block_v(S.dtype, S.shape[-2]), num_warps=1,
    )  # fmt: skip
    S_st = S.clone()
    commit_kda_replayssm_spec_all_layers(
        S_st, rv, rk, gk, beta, idx_d, lens_d, max_cache_len=L, num_k_heads=H,
        mamba_track_indices=track_idx, mamba_steps_to_track=steps, null_block_id=-1,
    )  # fmt: skip
    torch.cuda.synchronize()

    assert not torch.equal(S_v2, S0), "test must fold something"
    assert torch.equal(S_v2, S_st), "one-warp v2 must be bit-identical to the stock commit on BF16 state"
    for i in range(B):
        s, n = int(idx[i]), int(lens[i])
        if s < 0 or n <= 0:
            continue
        for layer in range(LAYERS):
            snap_at = int(steps[i]) if track and i in (0, 1) else -1
            ref, snap = _fold64(S0, rv, rk, gk, beta, layer, s, n, snap_at)
            off = (S_v2[layer, s].double() - ref).abs() - _ulp_bf16(ref) - 1e-6
            assert off.max().item() <= 0, f"slot {s} layer {layer}: more than one BF16 ulp from float64"
            if snap is not None and snap_at < n:
                t_slot = int(track_idx[i])
                off = (S_v2[layer, t_slot].double() - snap).abs() - _ulp_bf16(snap) - 1e-6
                assert off.max().item() <= 0, f"track slot {t_slot} layer {layer}: more than one BF16 ulp"


@pytest.mark.parametrize("half", [0, 1])
def test_fold_post_verify_and_flush_match_stock_commit(half):
    """Fold-mode bookkeeping: post-verify commits only track-crossing rows (sparse, compacted) and records the accept
    length (with the ring half the verify wrote) for the rest; a flush then commits the pending rows. Final states and
    track slots must match the stock commit on the same window; before the flush, non-crossing rows must be untouched
    and pending must be exact. `half` = the ring half holding the window just verified (the stock reference reads a
    copy of that half at ring row 0)."""
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
        commit_kda_replayssm_fold_after_verify,
        kda_fold_flush,
    )

    B = 37
    S, rv, rk, gk, beta = _pool(B, seed=99)
    gen = torch.Generator().manual_seed(11)
    idx = torch.randperm(B, generator=gen).to(torch.int32) + 1  # slots 1..B in random row order
    idx[7] = -1  # padding row
    lens = torch.randint(1, 9, (B,), generator=gen, dtype=torch.int32)
    track_idx = torch.full((B,), -1, dtype=torch.int32)
    steps = torch.full((B,), -1, dtype=torch.int32)
    track_idx[3], steps[3] = B + 1, 0
    track_idx[24], steps[24] = B + 2, 2
    lens[24] = max(int(lens[24]), 3)
    steps[34] = 1  # crossing row whose track slot is null: commit, no snapshot
    lens[34] = max(int(lens[34]), 2)
    crossing_rows = [3, 24, 34]
    idx_d, lens_d = idx.to(DEV), lens.to(DEV)
    tr_d, st_d = track_idx.to(DEV), steps.to(DEV)

    S_st = S.clone()
    h0r = half * (L // 2)

    def _at0(t):  # the window's half moved to ring row 0 (what the stock commit reads)
        u = torch.zeros_like(t)
        u[:, :, :, : L // 2] = t[:, :, :, h0r : h0r + L // 2]
        return u

    commit_kda_replayssm_spec_all_layers(
        S_st, _at0(rv), _at0(rk), _at0(gk), _at0(beta), idx_d, lens_d, max_cache_len=L,
        num_k_heads=H, mamba_track_indices=tr_d, mamba_steps_to_track=st_d, null_block_id=-1,
    )  # fmt: skip

    S_f = S.clone()
    spec = SimpleNamespace(
        temporal=S_f, replayssm_rawv=rv, replayssm_rawk=rk, replayssm_g=gk,
        replayssm_beta=beta, conv=[], intermediate_conv_window=[],
    )  # fmt: skip
    # The verify wrote half `half`, so before it the slots' windows were in half 1 - half.
    pending = torch.full((S.shape[1],), (1 - half) << 8, dtype=torch.int32, device=DEV)
    commit_kda_replayssm_fold_after_verify(
        spec_state=spec, pending=pending, state_batch_indices=idx_d, accept_lens=lens_d,
        last_correct_step_indices=(lens_d - 1).to(torch.int64), mamba_track_indices=tr_d,
        mamba_steps_to_track=st_d, null_block_id=-1,
    )  # fmt: skip
    torch.cuda.synchronize()
    want = torch.full((S.shape[1],), (1 - half) << 8, dtype=torch.int32)
    for r in range(B):
        if idx[r] >= 0:
            want[int(idx[r])] = (0 if r in crossing_rows else int(lens[r])) | (half << 8)
    assert torch.equal(pending.cpu(), want), "pending after fold post-verify"
    for r in range(B):
        s = int(idx[r])
        if s < 0:
            continue
        if r in crossing_rows:
            e = _rel(S_f[:, s], S_st[:, s])
            assert e <= TOL, f"crossing row {r} committed now: rel {e:.3e}"
        else:
            assert torch.equal(S_f[:, s], S[:, s]), f"non-crossing row {r} touched before flush"
    for t_slot in (B + 1, B + 2):
        e = _rel(S_f[:, t_slot], S_st[:, t_slot])
        assert e <= TOL, f"track slot {t_slot}: rel {e:.3e}"

    kda_fold_flush(spec_state=spec, pending=pending, state_batch_indices=idx_d, null_block_id=-1)
    torch.cuda.synchronize()
    live_t = torch.tensor([int(s) for s in idx if s >= 0])
    assert torch.equal(pending.cpu()[live_t], torch.full_like(live_t, 1 << 8).int()), (
        "flush must leave 0 records in half 1"
    )
    live = [int(s) for s in idx if s >= 0]
    e = _rel(S_f[:, live], S_st[:, live])
    assert e <= TOL, f"after flush vs stock: rel {e:.3e}"
    untouched = torch.ones(S.shape[1], dtype=torch.bool)
    untouched[live + [B + 1, B + 2]] = False
    assert torch.equal(S_f[:, untouched], S[:, untouched]), "flush touched a slot it must not"


@pytest.mark.parametrize("block_v", [None, 16])
def test_fold_bf16_commits_follow_block_v(block_v):
    """BF16 state in fold mode: the post-verify sparse commit of track-crossing rows (entered through
    commit_kda_replayssm_after_verify, as the backends call it) and the flush of the other rows take the V tile the
    stock commit takes for the same ``block_v`` (SGLANG_OPT_KDA_REPLAYSSM_FOLD_BLOCK_V; None = the default tile).
    Tolerance, fixed before the first run: the whole pool (committed slots, track snapshots, untouched slots) is
    bit-identical to the stock commit at that ``block_v``. Control at block_v 16: the stock commit's BF16 bits at
    tile 16 differ from tile 8 on these inputs in at least one crossing-row slot and one flushed slot, so a sparse
    commit that kept tile 8 on either path would fail the bitwise check."""
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
        commit_kda_replayssm_after_verify,
        kda_fold_flush,
    )

    B = 64
    S, rv, rk, gk, beta = _pool(B, seed=131)
    S, rv, rk = S.bfloat16(), rv.bfloat16(), rk.bfloat16()
    gen = torch.Generator().manual_seed(13)
    idx = torch.randperm(B, generator=gen).to(torch.int32) + 1  # slots 1..B in random row order
    idx[7] = -1  # padding row (not crossing)
    lens = torch.randint(1, 9, (B,), generator=gen, dtype=torch.int32)
    track_idx = torch.full((B,), -1, dtype=torch.int32)
    steps = torch.full((B,), -1, dtype=torch.int32)
    crossing_rows = list(range(0, B, 2))  # committed by the post-verify sparse commit; odd rows by the flush
    for j, r in enumerate(crossing_rows):
        steps[r] = int(torch.randint(0, int(lens[r]), (1,), generator=gen))
        if j < 2:
            track_idx[r] = B + 1 + j  # two snapshots; the other crossing rows have a null track slot
    idx_d, lens_d = idx.to(DEV), lens.to(DEV)
    tr_d, st_d = track_idx.to(DEV), steps.to(DEV)

    def stock(bv):
        out = S.clone()
        commit_kda_replayssm_spec_all_layers(
            out, rv, rk, gk, beta, idx_d, lens_d, max_cache_len=L, num_k_heads=H,
            mamba_track_indices=tr_d, mamba_steps_to_track=st_d, null_block_id=-1, block_v=bv,
        )  # fmt: skip
        return out

    S_st = stock(block_v)
    S_f = S.clone()
    spec = SimpleNamespace(
        temporal=S_f, replayssm_rawv=rv, replayssm_rawk=rk, replayssm_g=gk,
        replayssm_beta=beta, conv=[], intermediate_conv_window=[],
    )  # fmt: skip
    pending = torch.full((S.shape[1],), 1 << 8, dtype=torch.int32, device=DEV)  # the verify writes half 0
    commit_kda_replayssm_after_verify(
        spec_state=spec, state_batch_indices=idx_d, accept_lens=lens_d,
        last_correct_step_indices=(lens_d - 1).to(torch.int64), mamba_track_indices=tr_d,
        mamba_steps_to_track=st_d, null_block_id=-1, block_v=block_v, fold_pending=pending,
    )  # fmt: skip
    kda_fold_flush(spec_state=spec, pending=pending, state_batch_indices=idx_d, null_block_id=-1, block_v=block_v)
    torch.cuda.synchronize()

    assert not torch.equal(S_st, S), "test must fold something"
    if block_v == 16:
        S_8 = stock(None)
        torch.cuda.synchronize()
        moved = (S_8 != S_st).flatten(2).any(-1).any(0).cpu()  # [slots]: the tile changed some bit of this slot
        cross = {int(idx[r]) for r in crossing_rows if idx[r] >= 0}
        flushed = {int(s) for s in idx.tolist() if s >= 0} - cross
        assert any(moved[s] for s in cross), "control: tile 16 vs 8 moves no crossing-row slot (test blind there)"
        assert any(moved[s] for s in flushed), "control: tile 16 vs 8 moves no flushed slot (test blind there)"
    assert torch.equal(S_f, S_st), "fold-mode commits on BF16 state must equal the stock commit at the same block_v"
