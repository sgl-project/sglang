"""Device-side metadata, schedule and cold-hint operations for exact Few prefill."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld


@triton.jit
def add_up(x, y):
    return ld.add_ru(x, y)

@triton.jit(do_not_specialize=["N"])
def key_finish(PS, PB, S, BAD, N, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(S, tl.max(tl.load(PS + i, i < N, 0.0), 0))
    tl.store(BAD, tl.max(tl.load(PB + i, i < N, 0), 0))

@triton.jit
def query_check(
    W,
    H14,
    H16,
    H20,
    BUDGET,
    S,
    KBAD,
    SB,
    TAU,
    ERR,
    SAFE,
    GROUP: tl.constexpr,
    DOT: tl.constexpr,
    G: tl.constexpr,
    SEED_DIAG: tl.constexpr,
    HEADS: tl.constexpr,
    HIGH: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.load(BUDGET + row // GROUP)
    valid_family = (b == 14) | (b == 16) | (b == HIGH)
    hp = tl.where(b == 14, H14, tl.where(b == 16, H16, H20))
    hint = tl.load(hp + row)
    h = tl.arange(0, HEADS)
    w = tl.load(W + row * HEADS + h)
    mag = w.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    domain = (mag == 0) | ((mag >= 0x15800000) & (mag <= 0x6E800000))
    sw = tl.reduce(tl.abs(w).to(tl.float64), 0, add_up)
    one = tl.full((), 1.0, tl.float64)
    u = tl.full((), 2.0**-24, tl.float64)
    d = ld.mul_ru(
        ld.add_ru(one, tl.full((), DOT, tl.float64)),
        tl.full((), 25690112.0, tl.float64),
    )
    pre = ld.mul_ru(ld.mul_ru(d, sw), ld.add_ru(one, tl.full((), G, tl.float64)))
    double_pre = ld.mul_ru(tl.full((), 2.0, tl.float64), pre)
    smax = ld.add_ru(
        ld.mul_ru(ld.mul_ru(ld.add_ru(one, u), tl.load(S).to(tl.float64)), pre),
        tl.full((), 2.0**-126, tl.float64),
    )
    limit = tl.full((), 3.4028234663852886e38, tl.float64)
    tau = tl.load(TAU + row)
    bad = (
        ~valid_family
        | (hint != 2)
        | (tl.load(KBAD) != 0)
        | (tl.load(SB + row * 4 + 3) == 3 if SEED_DIAG else tl.load(SB + row) != 0)
        | (tl.sum((~domain).to(tl.int32)) != 0)
        | (sw > tl.full((), 2.0**100, tl.float64))
        | ~(double_pre < limit)
        | ~(smax < limit)
        | ~(tl.abs(tau) < float("inf"))
    )
    if bad:
        tl.atomic_or(ERR + row, 1024)
    tl.store(SAFE + row, tl.where(bad | (tl.load(ERR + row) != 0), -float("inf"), tau))

@triton.jit
def clear_live_table(TABLE, ERROR, STATUS, LIVE, Q: tl.constexpr,
                     STRIDE: tl.constexpr, CAP: tl.constexpr,
                     SEED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    nk = tl.load(LIVE)
    valid = (nk >= SEED) & (nk <= CAP)
    if valid:
        columns = (nk - SEED + 127) // 128
        for begin in range(0, columns, BLOCK):
            c = begin + tl.arange(0, BLOCK)
            tl.store(TABLE + row * STRIDE + c, -1, c < columns)
    else:
        if row < Q:
            tl.atomic_or(ERROR + row, 1)
        if row == 0:
            tl.atomic_or(STATUS, 2)

@triton.jit
def key_parts_live(KS, RHO, PS, PB, LIVE, CAP: tl.constexpr,
                   SEED: tl.constexpr, WORKERS: tl.constexpr, BLOCK: tl.constexpr):
    worker = tl.program_id(0)
    nk = tl.load(LIVE)
    valid = (nk >= SEED) & (nk <= CAP)
    maximum = tl.full((BLOCK,), 0.0, tl.float32)
    bad = tl.full((BLOCK,), 0, tl.int32)
    if valid:
        parts = (nk + BLOCK - 1) // BLOCK
        for part in range(worker, parts, WORKERS):
            i = part * BLOCK + tl.arange(0, BLOCK)
            m = i < nk
            scale = tl.load(KS + i, m, 1.0)
            bits = scale.to(tl.int32, bitcast=True)
            rho = tl.load(RHO + i, m, 0)
            ok = (bits >= 0x00800000) & (bits < 0x7F800000) & (rho >= 0) & (rho < 127)
            maximum = tl.maximum(maximum, tl.where(m & ok, scale, 0.0))
            bad = bad | (m & ~ok).to(tl.int32)
    else:
        bad = tl.full((BLOCK,), 1, tl.int32)
    tl.store(PS + worker, tl.max(maximum, 0))
    tl.store(PB + worker, tl.max(bad, 0))

@triton.jit
def compact_groups(BUDGET, IDS, INV, N, PAID: tl.constexpr, B: tl.constexpr):
    g = tl.arange(0, B)
    ok = g < N
    b0 = tl.load(BUDGET + 2 * g, ok, 0)
    b1 = tl.load(BUDGET + 2 * g + 1, ok, 0)
    active = ok & ((b0 == PAID) | (b1 == PAID))
    pos = tl.cumsum(active.to(tl.int32), 0) - 1
    tl.store(IDS + pos, g, active)
    tl.store(INV + g, tl.where(active, pos, -1), ok)
    return tl.sum(active.to(tl.int32), 0)

@triton.jit
def exact_parts(count, tiles, sms: tl.constexpr, grid):
    # The callers bound tiles <= 8192 before using the exact256 breakpoints.
    groups = tl.maximum(count, 1)
    t = tl.maximum(tiles, 1)
    cap = tl.maximum(1, tl.minimum(t, grid // groups))
    chosen = tl.full((), 1, tl.int32)
    if (count > 0) & (tiles > 0):
        j = tl.arange(0, 256)
        d = j % 128 + 1
        p = tl.where(j < 128, d, (t - 1) // d + 1)
        valid = p <= cap
        pe = tl.minimum(p, cap)
        cost = ((groups * pe + sms - 1) // sms) * ((t + pe - 1) // pe)
        best = tl.min(tl.where(valid, cost, 2147483647), 0)
        chosen = tl.min(tl.where(valid & (cost == best), p, 2147483647), 0)
    return chosen

@triton.jit
def eight_wave_parts(count, tiles, sms: tl.constexpr, grid):
    groups = tl.maximum(count, 1)
    minwaves = (groups + sms - 1) // sms
    maxparts = tl.maximum(1, tl.minimum(tl.maximum(tiles, 1), grid // groups))
    chosen = tl.full((), 1, tl.int32)
    best = tl.full((), 2147483647, tl.int32)
    for delta in tl.static_range(8):
        wave = minwaves + delta
        p = tl.maximum(1, tl.minimum(maxparts, (wave * sms) // groups))
        cost = ((groups * p + sms - 1) // sms) * ((tiles + p - 1) // p)
        better = (cost < best) | ((cost == best) & (p < chosen))
        chosen = tl.where(better, p, chosen)
        best = tl.minimum(best, cost)
    return chosen

@triton.jit(do_not_specialize=["N", "GRID"])
def prepare_live(BUDGET, LIVE_NK, IDS, INV, CONFIG, N, GRID,
                 SMS: tl.constexpr, MAX_K: tl.constexpr, SEED: tl.constexpr,
                 TILE: tl.constexpr, PAID: tl.constexpr, B: tl.constexpr):
    tl.static_assert(SMS > 0)
    tl.static_assert((PAID == 14 and TILE == 256) or (PAID == 16 and TILE == 128))
    tl.static_assert(SEED >= 0)
    tl.static_assert(SEED <= MAX_K)
    tl.static_assert(MAX_K <= 1048576)
    tl.static_assert(SEED % 256 == 0)
    tl.static_assert((MAX_K - SEED + TILE - 1) // TILE <= 8192)
    count = compact_groups(BUDGET, IDS, INV, N, PAID, B)
    nk = tl.load(LIVE_NK)
    valid = (nk >= SEED) & (nk <= MAX_K)
    # Clip before subtraction to avoid integer overflow. Invalid input still
    # publishes count=-1; clipping does not admit an invalid logical extent.
    safe_nk = tl.minimum(tl.maximum(nk, SEED), MAX_K)
    tiles = (safe_nk - SEED + TILE - 1) // TILE
    chosen = tl.full((), 1, tl.int32)
    if valid:
        if PAID == 14:
            chosen = exact_parts(count, tiles, SMS, GRID)
        else:
            chosen = eight_wave_parts(count, tiles, SMS, GRID)
    tl.store(CONFIG, tl.where(valid, count, -1))
    tl.store(CONFIG + 1, chosen)

@triton.jit(do_not_specialize=["NK", "NPAGES"])
def _identity_gather(CACHE, SCALES, TABLE, K, S, NK, NPAGES=0):
    page = tl.program_id(0)
    i = tl.arange(0, 64)
    d = tl.arange(0, 128)
    physical = tl.load(TABLE + page).to(tl.int64)
    valid = page * 64 + i < NK
    if NPAGES > 0:
        # This additional cold-hint read must not go outside the physical pool.
        # Masked pages overwrite the scratch with zeros, never stale contents.
        valid = valid & (physical >= 0) & (physical < NPAGES)
    raw = tl.load(
        CACHE + physical * 8448 + i[:, None] * 128 + d[None, :],
        valid[:, None],
        0,
    )
    scale = tl.load(SCALES + physical * 2112 + 2048 + i, valid, 0.0)
    output_valid = page * 64 + i < NK
    tl.store(K + (page * 64 + i[:, None]) * 128 + d[None, :], raw, output_valid[:, None])
    tl.store(S + page * 64 + i, scale, output_valid)

@triton.jit(do_not_specialize=["NV"])
def _add_votes(IDS, VOTES, N: tl.constexpr, NV, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    key = tl.load(IDS + i, i < N, -1)
    tl.atomic_add(VOTES + key, 1, (i < N) & (key >= 0) & (key < NV), sem="relaxed")

@triton.jit
def _spread_queries(Q, W, OQ, OW, NQ: tl.constexpr, HEADS: tl.constexpr):
    anchor = tl.program_id(0)
    row = anchor * (NQ - 1) // 3
    i = tl.arange(0, HEADS * 128)
    tl.store(OQ + anchor * HEADS * 128 + i, tl.load(Q + row * HEADS * 128 + i))
    weight = tl.load(W + row * HEADS + i, i < HEADS, 0.0)
    tl.store(OW + anchor * HEADS + i, weight, i < HEADS)
