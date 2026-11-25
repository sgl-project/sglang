# SPDX-License-Identifier: Apache-2.0
"""Bole spec-verify kernels for GDN (gated delta rule) linear attention.

Target verify over a linear draft chain of T tokens keeps, per (request, head),
only the low-rank factors of every step instead of the full per-step state:

    S_a = p_a * S_0 + sum_{j<=a} (p_a / p_j) * k_j u_j^T

where p is the cumulative decay, k_j the L2-normalized key and u_j the
correction vector solved from a TxT unit lower-triangular system.

* ``bole_factor_kernel``  -- per (batch, head): log p, beta, normalized k, and
  the TxT matrices G (k-k interaction) and C (q-k interaction).
* ``bole_solve_kernel``   -- per (V tile, batch, head): solves for u and writes
  the verify output, reading S_0 from the slot.
* ``bole_commit_kernel``  -- per (V tile, K tile, layer*batch*head): rebuilds
  the accepted step's state from S_0 and the factors, in place.

Grouped Q/K heads are kept; state layout matches SGLang [slot, HV, V, K].
Factors and accumulation are FP32 (TF32x3 dots).
"""

import triton
import triton.language as tl


@triton.jit
def bole_factor_kernel(
    Q,
    Kp,
    A,
    Bg,
    A_log,
    Dt,
    Mask,
    G,
    C,
    P,
    Beta,
    Knorm,
    T: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    kh = head // (HV // HK)
    rows = tl.arange(0, BT)
    cols = tl.arange(0, BK)
    offsets = (batch * T + rows[:, None]) * QS + kh * DK + cols[None, :]
    koff = (batch * T + rows[:, None]) * KS + kh * DK + cols[None, :]
    query = tl.load(Q + offsets, (rows[:, None] < T) & (cols[None, :] < DK), 0).to(
        tl.float32
    )
    key = tl.load(Kp + koff, (rows[:, None] < T) & (cols[None, :] < DK), 0).to(
        tl.float32
    )
    qnorm = tl.rsqrt(tl.sum(query * query, 1) + 1.0e-6) * (DK**-0.5)
    knorm = tl.rsqrt(tl.sum(key * key, 1) + 1.0e-6)
    key = key * knorm[:, None]
    if head % (HV // HK) == 0:
        tl.store(
            Knorm + ((batch * HK + kh) * T + rows[:, None]) * DK + cols[None, :],
            key,
            (rows[:, None] < T) & (cols[None, :] < DK),
        )
    aval = tl.load(A + (batch * T + rows) * AS + head, rows < T, 0).to(tl.float32)
    bval = tl.load(Bg + (batch * T + rows) * BS + head, rows < T, 0).to(tl.float32)
    a_log = tl.load(A_log + head).to(tl.float32)
    bias = tl.load(Dt + head).to(tl.float32)
    x = aval + bias
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    log_gamma = -tl.exp(a_log) * softplus
    beta = 1.0 / (1.0 + tl.exp(-bval))
    mask = tl.load(
        Mask + rows[:, None] * T + rows[None, :],
        (rows[:, None] < T) & (rows[None, :] < T),
        0,
    )
    log_p = tl.sum(tl.where(mask != 0, log_gamma[None, :], 0.0), 1)
    p = tl.exp(log_p)
    tl.store(P + (batch * HV + head) * T + rows, log_p, rows < T)
    tl.store(Beta + (batch * HV + head) * T + rows, beta, rows < T)
    ancestor = (mask != 0) & (rows[:, None] != rows[None, :])
    ratio = tl.exp(tl.where(mask != 0, log_p[:, None] - log_p[None, :], 0.0))
    kk = tl.full((BT, BT), 0.0, tl.float32)
    qk = tl.full((BT, BT), 0.0, tl.float32)
    for start in range(0, DK, 32):
        small = start + tl.arange(0, 32)
        idx = (batch * T + rows[:, None]) * QS + kh * DK + small[None, :]
        kidx = (batch * T + rows[:, None]) * KS + kh * DK + small[None, :]
        qsmall = (
            tl.load(Q + idx, (rows[:, None] < T) & (small[None, :] < DK), 0).to(
                tl.float32
            )
            * qnorm[:, None]
        )
        ksmall = (
            tl.load(Kp + kidx, (rows[:, None] < T) & (small[None, :] < DK), 0).to(
                tl.float32
            )
            * knorm[:, None]
        )
        kk += tl.dot(ksmall, tl.trans(ksmall), input_precision="tf32x3")
        qk += tl.dot(qsmall, tl.trans(ksmall), input_precision="tf32x3")
    g = tl.where(ancestor, beta[:, None] * ratio * kk, 0.0)
    c = tl.where(mask != 0, ratio * qk, 0.0)
    dst = ((batch * HV + head) * BT + rows[:, None]) * BT + rows[None, :]
    tl.store(G + dst, g)
    tl.store(C + dst, c)


@triton.jit
def bole_solve_kernel(
    Q,
    Kraw,
    Vraw,
    S,
    Slots,
    G,
    C,
    P,
    Beta,
    U,
    O,
    T: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DEPTH: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kh = head // (HV // HK)
    r = tl.arange(0, BT)
    k = tl.arange(0, BK)
    v = tile * BV + tl.arange(0, BV)
    slot = tl.load(Slots + batch).to(tl.int64)
    offs = (batch * T + r[:, None]) * QS + kh * DK + k[None, :]
    koffs = (batch * T + r[:, None]) * KS + kh * DK + k[None, :]
    query = tl.load(Q + offs, (r[:, None] < T) & (k[None, :] < DK), 0).to(tl.float32)
    key = tl.load(Kraw + koffs, (r[:, None] < T) & (k[None, :] < DK), 0).to(tl.float32)
    qnorm = tl.rsqrt(tl.sum(query * query, 1) + 1.0e-6) * (DK**-0.5)
    knorm = tl.rsqrt(tl.sum(key * key, 1) + 1.0e-6)
    values = tl.load(
        Vraw + (batch * T + r[:, None]) * VS + head * DV + v[None, :],
        (r[:, None] < T) & (v[None, :] < DV),
        0,
    ).to(tl.float32)
    p = tl.exp(tl.load(P + (batch * HV + head) * T + r, r < T, 0.0))
    beta = tl.load(Beta + (batch * HV + head) * T + r, r < T, 0.0)
    qs = tl.full((BT, BV), 0.0, tl.float32)
    ks = tl.full((BT, BV), 0.0, tl.float32)
    for start in range(0, DK, 32):
        small = start + tl.arange(0, 32)
        idx = (batch * T + r[:, None]) * QS + kh * DK + small[None, :]
        kidx = (batch * T + r[:, None]) * KS + kh * DK + small[None, :]
        qsmall = (
            tl.load(Q + idx, (r[:, None] < T) & (small[None, :] < DK), 0).to(tl.float32)
            * qnorm[:, None]
        )
        ksmall = (
            tl.load(Kraw + kidx, (r[:, None] < T) & (small[None, :] < DK), 0).to(
                tl.float32
            )
            * knorm[:, None]
        )
        state = tl.load(
            S + slot * STATE_STRIDE + (head * DV + v[None, :]) * DK + small[:, None],
            (small[:, None] < DK) & (v[None, :] < DV) & (slot >= 0),
            0,
        ).to(tl.float32)
        qs += tl.dot(qsmall, state, input_precision="tf32x3")
        ks += tl.dot(ksmall, state, input_precision="tf32x3")
    base = p[:, None] * qs
    residual = beta[:, None] * (values - p[:, None] * ks)
    g = tl.load(G + ((batch * HV + head) * BT + r[:, None]) * BT + r[None, :])
    z = residual
    correction = residual
    for _ in range(DEPTH):
        z = -tl.dot(g, z, input_precision="tf32x3")
        correction += z
    c = tl.load(C + ((batch * HV + head) * BT + r[:, None]) * BT + r[None, :])
    output = base + tl.dot(c, correction, input_precision="tf32x3")
    tl.store(
        U + ((batch * HV + head) * T + r[:, None]) * DV + v[None, :],
        correction,
        (r[:, None] < T) & (v[None, :] < DV),
    )
    tl.store(
        O + ((batch * T + r[:, None]) * HV + head) * DV + v[None, :],
        output,
        (r[:, None] < T) & (v[None, :] < DV),
    )


@triton.jit
def bole_commit_kernel(
    S,
    Knorm,
    P,
    U,
    SrcSlots,
    Slots,
    Terminal,
    Mask,
    T: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BUFFER_BS: tl.constexpr,
    ACTIVE_BS: tl.constexpr,
    LAYER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
):
    vt, kt, lb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer = lb // (BUFFER_BS * HV)
    bh = lb % (BUFFER_BS * HV)
    batch, head = bh // HV, bh % HV
    kh = head // (HV // HK)
    terminal = tl.load(Terminal + batch, batch < ACTIVE_BS, -1)
    slot = tl.load(Slots + batch, batch < ACTIVE_BS, -1).to(tl.int64)
    # Prefix state is read from SrcSlots and the rebuilt state written to Slots. They are the same for the
    # request commit; radix tracking rebuilds an intermediate step from the request slot into its track slot.
    src = tl.load(SrcSlots + batch, batch < ACTIVE_BS, -1).to(tl.int64)
    valid = (terminal >= 0) & (slot >= 0) & (src >= 0) & (batch < ACTIVE_BS)
    r = tl.arange(0, BT)
    k = kt * BK + tl.arange(0, BK)
    v = vt * BV + tl.arange(0, BV)
    fp = ((layer * BUFFER_BS + batch) * HV + head) * T
    log_pa = tl.load(P + fp + terminal, valid, 0.0)
    log_pj = tl.load(P + fp + r, (r < T) & valid, 0.0)
    pa = tl.exp(log_pa)
    accepted = tl.load(Mask + terminal * T + r, (r < T) & valid, 0) != 0
    key = tl.load(
        Knorm
        + (((layer * BUFFER_BS + batch) * HK + kh) * T + r[:, None]) * DK
        + k[None, :],
        (r[:, None] < T) & (k[None, :] < DK) & valid,
        0.0,
    )
    ratio = tl.exp(tl.where(accepted, log_pa - log_pj, 0.0))
    weighted = tl.where(accepted[:, None], key * ratio[:, None], 0.0)
    correction = tl.load(
        U
        + (((layer * BUFFER_BS + batch) * HV + head) * T + r[:, None]) * DV
        + v[None, :],
        (r[:, None] < T) & (v[None, :] < DV) & valid,
        0.0,
    )
    dst = (
        S
        + layer.to(tl.int64) * LAYER_STRIDE
        + slot * SLOT_STRIDE
        + (head * DV + v[None, :]) * DK
        + k[:, None]
    )
    prefix = (
        S
        + layer.to(tl.int64) * LAYER_STRIDE
        + src * SLOT_STRIDE
        + (head * DV + v[None, :]) * DK
        + k[:, None]
    )
    state = tl.load(prefix, (k[:, None] < DK) & (v[None, :] < DV) & valid, 0.0).to(
        tl.float32
    )
    updated = pa * state + tl.dot(
        tl.trans(weighted), correction, input_precision="tf32x3"
    )
    tl.store(
        dst,
        updated.to(S.dtype.element_ty),
        (k[:, None] < DK) & (v[None, :] < DV) & valid,
    )
