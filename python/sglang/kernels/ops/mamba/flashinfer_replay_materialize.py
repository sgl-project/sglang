"""Eager endpoint materialization for FlashInfer 0.6.18 checkpointing SSU.

Consumes its slot-indexed old_x/old_B/processed-dt/cumulative-decay layout,
not the newer ring-buffer API. Used by the opt-in Mamba2 speculative replay
path after acceptance; persistent checkpoints never depend on these records.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import convert_rs_fp16x2


@triton.jit
def _endpoint(
    BASE,
    X,
    B,
    DT,
    CUM,
    END,
    SLOT,
    BANK,
    LAYER,
    HEAD,
    M,
    NN,
    TT,
    K: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
):
    scalar_base = ((LAYER * K + SLOT) * 2 + BANK) * H * W + HEAD * W
    total = tl.load(CUM + scalar_base + END)
    cumulative = tl.load(CUM + scalar_base + TT, mask=TT <= END, other=0)
    dt = tl.load(DT + scalar_base + TT, mask=TT <= END, other=0)
    coeff = tl.exp(total - cumulative) * dt
    b = tl.load(
        B
        + (((LAYER * K + SLOT) * 2 + BANK) * W + TT[:, None]) * G * N
        + (HEAD // (H // G)) * N
        + NN[None, :],
        mask=(TT[:, None] <= END) & (NN[None, :] < N),
        other=0,
    )
    # Match checkpointing SSU's BF16 scaled-B MMA operands, not a sequential
    # unrounded FP32 transition. Pad the dot reduction to a valid MMA tile.
    b_scaled = (b.to(tl.float32) * coeff[:, None]).to(tl.bfloat16)
    x = tl.load(
        X + ((LAYER * K + SLOT) * W + TT[None, :]) * H * P + HEAD * P + M[:, None],
        mask=(TT[None, :] <= END) & (M[:, None] < P),
        other=0,
    )
    return tl.dot(x, b_scaled, BASE * tl.exp(total))


@triton.jit
def _materialize(
    STATE,
    X,
    B,
    DT,
    CUM,
    BANKS,
    SLOTS,
    LAST,
    TRACKS,
    TRACK_STEPS,
    SEEDS,
    K: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
):
    tile, row, layer_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer, head = layer_head // H, layer_head % H
    slot = tl.load(SLOTS + row).to(tl.int64)
    last = tl.load(LAST + row)
    if slot < 0 or slot >= K or last < 0 or last >= W:
        return
    bank = tl.load(BANKS + layer * K + slot)
    m = tile * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    t = tl.arange(0, BT)
    offsets = ((layer * K + slot) * H + head) * P * N + m[:, None] * N + n[None, :]
    mask = (m[:, None] < P) & (n[None, :] < N)
    base = tl.load(STATE + offsets, mask=mask, other=0).to(tl.float32)
    final = _endpoint(
        base, X, B, DT, CUM, last, slot, bank, layer, head, m, n, t, K, W, H, P, G, N
    )
    if PHILOX_ROUNDS > 0:
        seed = tl.load(SEEDS + layer)
        # One deterministic independent counter per physical FP16 pair.
        # This preserves unbiased SR, not FlashInfer's CTA-layout-specific RNG
        # bit mapping. Live/track at the same endpoint use identical randomness.
        random = tl.randint(seed, (offsets // 2 * 2).to(tl.uint64), PHILOX_ROUNDS)
    if HAS_TRACK:
        track = tl.load(TRACKS + row).to(tl.int64)
        step = tl.load(TRACK_STEPS + row)
        if track >= 0 and track < K and step >= 0 and step <= last:
            tracked = _endpoint(
                base,
                X,
                B,
                DT,
                CUM,
                step,
                slot,
                bank,
                layer,
                head,
                m,
                n,
                t,
                K,
                W,
                H,
                P,
                G,
                N,
            )
            if PHILOX_ROUNDS > 0:
                tracked = convert_rs_fp16x2(tracked, random)
            target = (
                ((layer * K + track) * H + head) * P * N + m[:, None] * N + n[None, :]
            )
            tl.store(STATE + target, tracked, mask=mask)
    if PHILOX_ROUNDS > 0:
        final = convert_rs_fp16x2(final, random)
    tl.store(STATE + offsets, final, mask=mask)


def materialize_flashinfer_mamba2(
    state,
    old_x,
    old_B,
    old_dt,
    old_cumAdt,
    banks,
    slots,
    last,
    tracks=None,
    track_steps=None,
    *,
    seeds=None,
    philox_rounds=0,
):
    """Write accepted and optional tracked endpoints across all local layers.

    All tensors are layer-major. last/track_steps are inclusive verify positions.
    Call only with zero pending history at verify entry. No buffer is allocated
    here, no host device-data inspection occurs, and compact records are read-only.
    Destinations must not alias other active rows' source or destination slots.
    """
    layers, capacity, heads, dim, dstate = state.shape
    width, groups = old_B.shape[3:5]
    assert state.dtype == torch.float16
    assert old_x.dtype == old_B.dtype == torch.bfloat16
    assert old_dt.dtype == old_cumAdt.dtype == torch.float32
    assert 1 <= width <= 16 and heads % groups == 0
    assert dim == 64 and dstate == 128
    assert old_x.shape == (layers, capacity, width, heads, dim)
    assert old_B.shape == (layers, capacity, 2, width, groups, dstate)
    assert old_dt.shape == old_cumAdt.shape == (layers, capacity, 2, heads, width)
    assert banks.shape == (layers, capacity) and banks.dtype == torch.int32
    assert all(
        x.is_contiguous() for x in (state, old_x, old_B, old_dt, old_cumAdt, banks)
    )
    assert (tracks is None) == (track_steps is None)
    batch = last.numel()
    assert slots.numel() >= batch
    if tracks is not None:
        assert tracks.numel() >= batch and track_steps.numel() >= batch
    assert philox_rounds >= 0
    if philox_rounds:
        assert (
            seeds is not None
            and seeds.shape == (layers,)
            and seeds.dtype == torch.int64
        )
    if not layers or not batch:
        return
    _materialize[(triton.cdiv(dim, 16), batch, layers * heads)](
        state,
        old_x,
        old_B,
        old_dt,
        old_cumAdt,
        banks,
        slots,
        last,
        tracks,
        track_steps,
        seeds,
        capacity,
        width,
        heads,
        dim,
        groups,
        dstate,
        tracks is not None,
        philox_rounds,
        16,
        128,
        32,
        num_warps=4,
    )
