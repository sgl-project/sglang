"""Coral experiment kernels for Qwen3.6-35B-A3B on H100 (SGLang v0.5.22)."""
import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
GROUP = 128


@triton.jit
def _add_gemma_rmsnorm_quant_kernel(
    x_ptr, res_ptr, w_ptr, q_ptr, s_ptr, M, M_pad, eps,
    K: tl.constexpr, G: tl.constexpr, FP8_MAX: tl.constexpr,
):
    # One program per token row. In place: res = x + res ; x = gemma_rmsnorm(res) (bf16).
    # Also writes q = fp8(x / s) per 128-group and s (fp32, column-major [K/G, M_pad]) for DeepGEMM.
    row = tl.program_id(0)
    offs = tl.arange(0, K)
    if row >= M:  # DeepGEMM pad rows (M_pad = round_up(M, 4)): zero q, unit scale — no separate fill kernels
        tl.store(q_ptr + row * K + offs, tl.zeros((K,), dtype=tl.float32).to(q_ptr.dtype.element_ty))
        tl.store(s_ptr + tl.arange(0, K // G) * M_pad + row, tl.full((K // G,), 1.0, tl.float32))
        return
    x = tl.load(x_ptr + row * K + offs).to(tl.float32)
    r = tl.load(res_ptr + row * K + offs).to(tl.float32)
    h = x + r
    tl.store(res_ptr + row * K + offs, h.to(tl.bfloat16))
    var = tl.sum(h * h, axis=0) / K
    w = tl.load(w_ptr + offs).to(tl.float32)
    y = (h * tl.rsqrt(var + eps) * (1.0 + w)).to(tl.bfloat16)
    tl.store(x_ptr + row * K + offs, y)
    # Quantize the bf16-rounded output, exactly what a separate per-token-group quant would read.
    yg = tl.reshape(y.to(tl.float32), (K // G, G))
    # Same arithmetic as SGLang's JIT per_token_group_quant (fp32 scales): scale = amax * (1/448),
    # q = y * (448 / amax). Multiply-by-reciprocal (not divide) keeps the fp8 codes bit-identical.
    amax = tl.maximum(tl.max(tl.abs(yg), axis=1), 1e-10)
    scale = amax * (1.0 / FP8_MAX)
    q = tl.clamp(yg * (FP8_MAX / amax)[:, None], -FP8_MAX, FP8_MAX)
    tl.store(q_ptr + row * K + tl.reshape(offs, (K // G, G)), q.to(q_ptr.dtype.element_ty))
    tl.store(s_ptr + tl.arange(0, K // G) * M_pad + row, scale)


def add_gemma_rmsnorm_quant(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """In-place fused residual-add + Gemma RMSNorm (+1 weight) + FP8 per-token-group(128) quant.

    Returns (q, scale): q [M_pad, K] float8_e4m3fn (pad rows zero), scale [M_pad, K/128] fp32 with
    stride (1, M_pad) — the column-major TMA-aligned layout SGLang's DeepGEMM pre-quantized path expects.
    """
    M, K = x.shape
    assert K % GROUP == 0 and x.is_contiguous() and residual.is_contiguous()
    M_pad = (M + 3) // 4 * 4
    q = torch.empty(M_pad, K, dtype=torch.float8_e4m3fn, device=x.device)
    s_cm = torch.empty(K // GROUP, M_pad, dtype=torch.float32, device=x.device)
    _add_gemma_rmsnorm_quant_kernel[(M_pad,)](x, residual, weight, q, s_cm, M, M_pad, eps,
                                          K=K, G=GROUP, FP8_MAX=FP8_MAX, num_warps=4)
    return q, s_cm.t()


@triton.jit
def _skinny_gemm_kernel(x_ptr, w_ptr, y_ptr, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        BLOCK_M: tl.constexpr):
    # y[M, N] = x[M, K] @ w[N, K]^T for tiny M (<= BLOCK_M): each program streams a BLOCK_N x K slab of w once
    # (cold HBM read) and multiplies all M tokens with one tensor-core dot per K step.
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=rn[:, None] < N, other=0.0)
        x = tl.load(x_ptr + rm[None, :] * K + rk[:, None], mask=rm[None, :] < M, other=0.0)
        acc = tl.dot(w, x, acc)
    tl.store(y_ptr + rm[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=(rn[:, None] < N) & (rm[None, :] < M))


def skinny_gemm(x: torch.Tensor, w: torch.Tensor, block_n: int = 16) -> torch.Tensor:
    """BF16 x[M, K] @ w[N, K]^T for M <= 16 (decode). fp32 accumulate, bf16 out."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and x.is_contiguous() and w.is_contiguous()
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _skinny_gemm_kernel[(triton.cdiv(N, block_n),)](x, w, y, M, N, K=K, BLOCK_N=block_n, BLOCK_K=256, BLOCK_M=16,
                                                    num_warps=4)
    return y


@triton.jit
def _rows_gemv_kernel(x_ptr, w_ptr, y_ptr, M, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
    # y[M, N] = x[M, K] @ w[N, K]^T for tiny M: one program per BLOCK_N weight rows (many CTAs -> many cold HBM
    # reads in flight); the rows are loaded ONCE into registers, then reused for every token (x is L2-hot).
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, K)
    w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=rn[:, None] < N, other=0.0).to(tl.float32)
    for m in range(M):
        x = tl.load(x_ptr + m * K + rk).to(tl.float32)
        tl.store(y_ptr + m * N + rn, tl.sum(w * x[None, :], axis=1).to(tl.bfloat16), mask=rn < N)


def rows_gemv(x: torch.Tensor, w: torch.Tensor, block_n: int = 1) -> torch.Tensor:
    """BF16 x[M, K] @ w[N, K]^T for decode-sized M; fp32 accumulate, bf16 out."""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _rows_gemv_kernel[(triton.cdiv(N, block_n),)](x, w, y, M, N, K=K, BLOCK_N=block_n, num_warps=4)
    return y


@triton.jit
def _fp8w_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, S_STRIDE: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, GROUPS_PER_ITER: tl.constexpr):
    # y[M, N] = x[M, K] (bf16) @ dequant(w[N, K] fp8, s[N/128, K/128] fp32)^T for decode-sized M (<= BLOCK_M).
    # Weight-only FP8 (W8A16): no activation quantization. Per 128-wide K group: one tensor-core dot of the fp8
    # weights (exact in bf16) with bf16 x, then the fp32 block scale — the same per-block accumulation DeepGEMM does.
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    nmask = rn < N
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    sb = s_ptr + (pid * BLOCK_N // 128) * S_STRIDE          # BLOCK_N <= 128 and divides 128: one scale row
    for g0 in range(0, K // 128, GROUPS_PER_ITER):
        for gi in tl.static_range(GROUPS_PER_ITER):
            rk = (g0 + gi) * 128 + tl.arange(0, 128)
            w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=nmask[:, None], other=0.0).to(tl.bfloat16)
            xt = tl.load(x_ptr + rm[None, :] * K + rk[:, None], mask=rm[None, :] < M, other=0.0)
            part = tl.dot(w, xt)
            acc += part * tl.load(sb + g0 + gi)
    tl.store(y_ptr + rm[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=nmask[:, None] & (rm[None, :] < M))


def fp8w_gemv(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, block_n: int = 32, groups_per_iter: int = 2):
    """W8A16 GEMV for M <= 16: x bf16 [M, K], w fp8 [N, K], s fp32 [ceil(N/128), K/128] (128x128 block scales)."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and K % (128 * groups_per_iter) == 0 and 128 % block_n == 0
    assert x.is_contiguous() and w.is_contiguous() and s.is_contiguous() and s.dtype == torch.float32
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _fp8w_gemv_kernel[(triton.cdiv(N, block_n),)](x, w, s, y, M, N, K=K, S_STRIDE=s.shape[1], BLOCK_N=block_n,
                                                  BLOCK_M=16, GROUPS_PER_ITER=groups_per_iter, num_warps=4)
    return y


@triton.jit
def _fp8w_gemv_splitk_kernel(x_ptr, w_ptr, s_ptr, y_ptr, ws_ptr, cnt_ptr, M, N,
                             K: tl.constexpr, S_STRIDE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                             SPLIT_K: tl.constexpr):
    # W8A16 GEMV, split-K. Program (pid_n, pid_k) covers BLOCK_N rows x K/SPLIT_K columns; its whole K-slice is a
    # static unroll so every weight load is issued up front (bytes in flight is what a cold-HBM GEMV needs).
    # Reduction happens in-kernel: each split writes fp32 partials, bumps a per-N-block counter (acq_rel), and the
    # last split to arrive sums the partials, writes bf16 y, and re-arms the counter -> one launch, graph-safe.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    G: tl.constexpr = K // 128 // SPLIT_K
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    nmask = rn < N
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    sb = s_ptr + (pid_n * BLOCK_N // 128) * S_STRIDE + pid_k * G
    for gi in tl.static_range(G):
        rk = (pid_k * G + gi) * 128 + tl.arange(0, 128)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=nmask[:, None], other=0.0).to(tl.bfloat16)
        xt = tl.load(x_ptr + rm[None, :] * K + rk[:, None], mask=rm[None, :] < M, other=0.0)
        acc += tl.dot(w, xt) * tl.load(sb + gi)
    omask = nmask[:, None] & (rm[None, :] < M)
    if SPLIT_K == 1:
        tl.store(y_ptr + rm[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=omask)
    else:
        tl.store(ws_ptr + pid_k * (BLOCK_M * N) + rm[None, :] * N + rn[:, None], acc, mask=omask)
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for k in tl.static_range(SPLIT_K):
                tot += tl.load(ws_ptr + k * (BLOCK_M * N) + rm[None, :] * N + rn[:, None], mask=omask, other=0.0,
                               cache_modifier=".cg")
            tl.store(y_ptr + rm[None, :] * N + rn[:, None], tot.to(tl.bfloat16), mask=omask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0)


_SPLITK_STATE = {}  # (device, N) -> zeroed per-N-block arrival counters (persistent; last arriver re-arms them)


def fp8w_gemv_splitk(x, w, s, block_n: int = 32, split_k: int = 4, num_warps: int = 4):
    """W8A16 GEMV for M <= 16 with in-kernel split-K reduction. Not safe to run concurrently with itself on the
    same N on two streams (shared counters)."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and K % (128 * split_k) == 0 and 128 % block_n == 0 and s.dtype == torch.float32
    nb = triton.cdiv(N, block_n)
    key = (x.device, N, block_n)
    cnt = _SPLITK_STATE.get(key)
    if cnt is None:
        cnt = _SPLITK_STATE[key] = torch.zeros(nb, dtype=torch.int32, device=x.device)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    ws = torch.empty(split_k, 16, N, dtype=torch.float32, device=x.device) if split_k > 1 else y
    _fp8w_gemv_splitk_kernel[(nb, split_k)](x, w, s, y, ws, cnt, M, N, K=K, S_STRIDE=s.shape[1], BLOCK_N=block_n,
                                            BLOCK_M=16, SPLIT_K=split_k, num_warps=num_warps)
    return y


# ---------------------------------------------------------------------------------------------------------------
# Decode MoE (M <= 4 tokens), W8A16, Qwen3.5/3.6 shape: 256 experts top-8 (softmax, renorm) + 1 sigmoid-gated
# shared expert treated as slot 8.  Two launches:
#   moe_up  : grid (M*9, I/BN, SK)  per-CTA routing -> gate/up rows of its slot's expert -> split-K partials ->
#             last arriver: silu(g)*u -> act[m, slot, :] (fp32); slot weights -> wts[m, slot]
#   moe_down: grid (M*9, H/BN, SK2) down rows -> partials * slot weight -> last arriver of (m, n-block) sums all
#             9*SK2 partials in fixed order (deterministic) -> y[m, n] bf16
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_route_slot(logits_ptr, m, j, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr):
    # expert id and routing weight of the j-th top-k slot of token m (softmax over E, top-k, optional renorm)
    re = tl.arange(0, E)
    lg = tl.load(logits_ptr + m * E + re).to(tl.float32)
    p = tl.exp(lg - tl.max(lg, axis=0))
    p = p / tl.sum(p, axis=0)
    tot = 0.0
    eid = 0
    wj = 0.0
    for t in tl.static_range(TOPK):
        v = tl.max(p, axis=0)
        i = tl.argmax(p, axis=0)
        tot += v
        eid = tl.where(t == j, i, eid)
        wj = tl.where(t == j, v, wj)
        p = tl.where(re == i, -1.0, p)
    if RENORM:
        wj = wj / tot
    return eid, wj


@triton.jit
def _moe_up_kernel(x_ptr, xg_ptr, logits_ptr, sg_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, ws_ptr, cnt_ptr, act_ptr, wts_ptr,
                   H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr,
                   BN: tl.constexpr, SK: tl.constexpr):
    pid_s = tl.program_id(0)            # m * (TOPK+1) + slot
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = I // BN
    m = pid_s // NS
    j = pid_s % NS
    G: tl.constexpr = H // 128 // SK
    SROW: tl.constexpr = H // 128                       # scale row stride
    if j < TOPK:
        eid, wj = _moe_route_slot(logits_ptr, m, j, E, TOPK, RENORM)
        wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
        sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
    else:
        wbase = sh13_ptr
        sbase = shs13_ptr
        rh = tl.arange(0, H)
        xs = tl.load(xg_ptr + m * H + rh).to(tl.float32)  # shared gate always sees unquantized bf16 x
        wj = tl.sigmoid(tl.sum(xs * tl.load(sg_ptr + rh).to(tl.float32), axis=0))
    rg = pid_n * BN + tl.arange(0, BN)                  # gate rows; up rows are rg + I
    accg = tl.zeros((BN,), dtype=tl.float32)
    accu = tl.zeros((BN,), dtype=tl.float32)
    srow_g = (pid_n * BN) // 128
    srow_u = (I + pid_n * BN) // 128
    for gi in tl.static_range(G):
        kg = pid_k * G + gi
        rk = kg * 128 + tl.arange(0, 128)
        xk = tl.load(x_ptr + m * H + rk).to(tl.float32)
        wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
        wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
        accg += tl.sum(wg * xk[None, :], axis=1) * tl.load(sbase + srow_g * SROW + kg)
        accu += tl.sum(wu * xk[None, :], axis=1) * tl.load(sbase + srow_u * SROW + kg)
    if SK == 1:
        tl.store(act_ptr + pid_s * I + rg, accg / (1.0 + tl.exp(-accg)) * accu)
    else:
        wsb = ws_ptr + ((pid_s * NB + pid_n) * SK) * (2 * BN)
        tl.store(wsb + pid_k * (2 * BN) + tl.arange(0, BN), accg)
        tl.store(wsb + pid_k * (2 * BN) + BN + tl.arange(0, BN), accu)
        prev = tl.atomic_add(cnt_ptr + pid_s * NB + pid_n, 1, sem="acq_rel")
        if prev == SK - 1:
            g = tl.zeros((BN,), dtype=tl.float32)
            u = tl.zeros((BN,), dtype=tl.float32)
            for k in tl.static_range(SK):
                g += tl.load(wsb + k * (2 * BN) + tl.arange(0, BN), cache_modifier=".cg")
                u += tl.load(wsb + k * (2 * BN) + BN + tl.arange(0, BN), cache_modifier=".cg")
            tl.store(act_ptr + pid_s * I + rg, g / (1.0 + tl.exp(-g)) * u)
            tl.atomic_xchg(cnt_ptr + pid_s * NB + pid_n, 0)
    if pid_n == 0 and pid_k == 0:
        tl.store(wts_ptr + pid_s, wj)


@triton.jit
def _moe_down_kernel(act_ptr, wts_ptr, logits_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr,
                     H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr,
                     BN: tl.constexpr, SK: tl.constexpr, FAKEQ: tl.constexpr = False):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = H // BN
    m = pid_s // NS
    j = pid_s % NS
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    if j < TOPK:
        eid, _w = _moe_route_slot(logits_ptr, m, j, E, TOPK, RENORM)
        wbase = w2_ptr + eid.to(tl.int64) * (H * I)
        sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
    else:
        wbase = sh2_ptr
        sbase = shs2_ptr
    rn = pid_n * BN + tl.arange(0, BN)
    srow = (pid_n * BN) // 128
    acc = tl.zeros((BN,), dtype=tl.float32)
    for gi in tl.static_range(G):
        kg = pid_k * G + gi
        rk = kg * 128 + tl.arange(0, 128)
        a = tl.load(act_ptr + pid_s * I + rk)
        if FAKEQ:  # ablation: emulate stock per-token-group(128) FP8 activation quantization (SGLang arithmetic)
            am = tl.maximum(tl.max(tl.abs(a), axis=0), 1e-10)
            a = (tl.clamp(a * (448.0 / am), -448.0, 448.0).to(tl.float8e4nv)).to(tl.float32) * (am * (1.0 / 448.0))
        w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1) * tl.load(sbase + srow * SROW + kg)
    acc = acc * tl.load(wts_ptr + pid_s)
    # partial slot index within token m: (slot j, split k) -> j * SK + k ; ws laid out [M, NB, NS*SK, BN]
    wsb = ws_ptr + ((m * NB + pid_n) * (NS * SK)) * BN
    tl.store(wsb + (j * SK + pid_k) * BN + tl.arange(0, BN), acc)
    prev = tl.atomic_add(cnt_ptr + m * NB + pid_n, 1, sem="acq_rel")
    if prev == NS * SK - 1:
        tot = tl.zeros((BN,), dtype=tl.float32)
        for q in tl.static_range(NS * SK):
            tot += tl.load(wsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
        tl.store(y_ptr + m * H + rn, tot.to(tl.bfloat16))
        tl.atomic_xchg(cnt_ptr + m * NB + pid_n, 0)


_MOE_STATE = {}


def fp8_group_fakequant(x, group=128):
    """bf16 -> fp8 per-token-group(128) quantize -> dequantize, SGLang's arithmetic (ablation helper)."""
    xf = x.float().view(x.shape[0], -1, group)
    am = xf.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    q = (xf * (448.0 / am)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * (am * (1.0 / 448.0))
    return q.view_as(x).to(x.dtype)


def coral_moe_decode(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                     bn_up=32, sk_up=4, bn_dn=64, sk_dn=2, max_m=4, fakeq=False):
    """Fused decode MoE (routed top-k + gated shared expert) for M <= max_m tokens. All weights block-FP8 128x128
    with fp32 scales [rows/128, cols/128] (per expert for w13/w2). Returns bf16 [M, H]."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M <= max_m and logits.shape == (M, E)
    nb_up, nb_dn = I // bn_up, H // bn_dn
    key = (x.device, bn_up, sk_up, bn_dn, sk_dn, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = (torch.zeros(max_m * NS * nb_up, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device))
    cnt_up, cnt_dn = st
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    wts = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    ws_up = torch.empty(M * NS * nb_up * sk_up * 2 * bn_up, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sk_dn * bn_dn, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    # ablation: stock quantizes only the expert GEMM inputs; router logits and the shared gate use bf16 x
    xin = fp8_group_fakequant(x) if fakeq else x
    _moe_up_kernel[(M * NS, nb_up, sk_up)](xin, x, logits, shared_gate_w, w13, s13, sh13, shs13, ws_up, cnt_up, act, wts,
                                           H=H, I=I, E=E, TOPK=top_k, RENORM=renorm, BN=bn_up, SK=sk_up, num_warps=4)
    _moe_down_kernel[(M * NS, nb_dn, sk_dn)](act, wts, logits, w2, s2, sh2, shs2, ws_dn, cnt_dn, y,
                                             H=H, I=I, E=E, TOPK=top_k, RENORM=renorm, BN=bn_dn, SK=sk_dn,
                                             FAKEQ=fakeq, num_warps=4)
    return y


# ---------------------------------------------------------------------------------------------------------------
# v2 decode MoE: the shared expert is ONE slot for all M tokens (its weights are read from HBM once per step, not
# once per token). Grid dim0 = M*TOPK routed slots + 1 shared slot. The shared CTA keeps each weight tile in
# registers and loops over the (<= MAXM) tokens. act/wts rows: routed (m, j) -> m*TOPK + j ; shared m -> M*TOPK + m.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_up_kernel_v2(x_ptr, logits_ptr, sg_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, ws_ptr, wsh_ptr, cnt_ptr,
                      act_ptr, wts_ptr, M, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                      RENORM: tl.constexpr, BN: tl.constexpr, SK: tl.constexpr, MAXM: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NB: tl.constexpr = I // BN
    G: tl.constexpr = H // 128 // SK
    SROW: tl.constexpr = H // 128
    rg = pid_n * BN + tl.arange(0, BN)
    srow_g = (pid_n * BN) // 128
    srow_u = (I + pid_n * BN) // 128
    if pid_s < M * TOPK:
        m = pid_s // TOPK
        j = pid_s % TOPK
        eid, wj = _moe_route_slot(logits_ptr, m, j, E, TOPK, RENORM)
        wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
        sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
        accg = tl.zeros((BN,), dtype=tl.float32)
        accu = tl.zeros((BN,), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            xk = tl.load(x_ptr + m * H + rk).to(tl.float32)
            wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
            wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
            accg += tl.sum(wg * xk[None, :], axis=1) * tl.load(sbase + srow_g * SROW + kg)
            accu += tl.sum(wu * xk[None, :], axis=1) * tl.load(sbase + srow_u * SROW + kg)
        if SK == 1:
            tl.store(act_ptr + pid_s * I + rg, accg / (1.0 + tl.exp(-accg)) * accu)
        else:
            wsb = ws_ptr + ((pid_s * NB + pid_n) * SK) * (2 * BN)
            tl.store(wsb + pid_k * (2 * BN) + tl.arange(0, BN), accg)
            tl.store(wsb + pid_k * (2 * BN) + BN + tl.arange(0, BN), accu)
            prev = tl.atomic_add(cnt_ptr + pid_s * NB + pid_n, 1, sem="acq_rel")
            if prev == SK - 1:
                g = tl.zeros((BN,), dtype=tl.float32)
                u = tl.zeros((BN,), dtype=tl.float32)
                for k in tl.static_range(SK):
                    g += tl.load(wsb + k * (2 * BN) + tl.arange(0, BN), cache_modifier=".cg")
                    u += tl.load(wsb + k * (2 * BN) + BN + tl.arange(0, BN), cache_modifier=".cg")
                tl.store(act_ptr + pid_s * I + rg, g / (1.0 + tl.exp(-g)) * u)
                tl.atomic_xchg(cnt_ptr + pid_s * NB + pid_n, 0)
        if pid_n == 0 and pid_k == 0:
            tl.store(wts_ptr + pid_s, wj)
    else:
        # shared expert, all M tokens: one tensor-core dot per 128-wide K group against all tokens at once
        # (fp8 -> bf16 is exact; x is bf16; fp32 accumulate), so the weight tile is read once and no per-token loop
        rm = tl.arange(0, 16)
        sh_g = tl.zeros((BN, 16), dtype=tl.float32)
        sh_u = tl.zeros((BN, 16), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            swg = tl.load(sh13_ptr + rg[:, None] * H + rk[None, :]).to(tl.bfloat16)
            swu = tl.load(sh13_ptr + (rg[:, None] + I) * H + rk[None, :]).to(tl.bfloat16)
            sxt = tl.load(x_ptr + rm[None, :] * H + rk[:, None], mask=rm[None, :] < M, other=0.0)
            sh_g += tl.dot(swg, sxt) * tl.load(shs13_ptr + srow_g * SROW + kg)
            sh_u += tl.dot(swu, sxt) * tl.load(shs13_ptr + srow_u * SROW + kg)
        row0 = M * TOPK
        omask = (rm[None, :] < M) & (rg[:, None] < I)
        if SK == 1:
            tl.store(act_ptr + (row0 + rm[None, :]) * I + rg[:, None], sh_g / (1.0 + tl.exp(-sh_g)) * sh_u, mask=omask)
        else:
            tile: tl.constexpr = 2 * BN * 16
            swsb = wsh_ptr + (pid_n * SK) * tile
            offs = tl.arange(0, BN)[:, None] * 16 + rm[None, :]
            tl.store(swsb + pid_k * tile + offs, sh_g)
            tl.store(swsb + pid_k * tile + BN * 16 + offs, sh_u)
            sprev = tl.atomic_add(cnt_ptr + row0 * NB + pid_n, 1, sem="acq_rel")
            if sprev == SK - 1:
                sg2 = tl.zeros((BN, 16), dtype=tl.float32)
                su2 = tl.zeros((BN, 16), dtype=tl.float32)
                for k in tl.static_range(SK):
                    sg2 += tl.load(swsb + k * tile + offs, cache_modifier=".cg")
                    su2 += tl.load(swsb + k * tile + BN * 16 + offs, cache_modifier=".cg")
                tl.store(act_ptr + (row0 + rm[None, :]) * I + rg[:, None], sg2 / (1.0 + tl.exp(-sg2)) * su2, mask=omask)
                tl.atomic_xchg(cnt_ptr + row0 * NB + pid_n, 0)
        if pid_n == 0 and pid_k == 0:
            rh = tl.arange(0, H)
            gw = tl.load(sg_ptr + rh).to(tl.float32)
            for mm in tl.static_range(MAXM):
                if mm < M:
                    xs = tl.load(x_ptr + mm * H + rh).to(tl.float32)
                    tl.store(wts_ptr + row0 + mm, tl.sigmoid(tl.sum(xs * gw, axis=0)))


@triton.jit
def _moe_down_kernel_v2(act_ptr, wts_ptr, logits_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr, M,
                        H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr,
                        BN: tl.constexpr, SK: tl.constexpr, MAXM: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = H // BN
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    rn = pid_n * BN + tl.arange(0, BN)
    srow = (pid_n * BN) // 128
    if pid_s < M * TOPK:
        m = pid_s // TOPK
        j = pid_s % TOPK
        eid, _w = _moe_route_slot(logits_ptr, m, j, E, TOPK, RENORM)
        wbase = w2_ptr + eid.to(tl.int64) * (H * I)
        sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            a = tl.load(act_ptr + pid_s * I + rk)
            w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
            acc += tl.sum(w * a[None, :], axis=1) * tl.load(sbase + srow * SROW + kg)
        acc = acc * tl.load(wts_ptr + pid_s)
        wsb = ws_ptr + ((m * NB + pid_n) * (NS * SK)) * BN
        tl.store(wsb + (j * SK + pid_k) * BN + tl.arange(0, BN), acc)
        prev = tl.atomic_add(cnt_ptr + m * NB + pid_n, 1, sem="acq_rel")
        if prev == NS * SK - 1:
            tot = tl.zeros((BN,), dtype=tl.float32)
            for q in tl.static_range(NS * SK):
                tot += tl.load(wsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
            tl.store(y_ptr + m * H + rn, tot.to(tl.bfloat16))
            tl.atomic_xchg(cnt_ptr + m * NB + pid_n, 0)
    else:
        rm = tl.arange(0, 16)
        row0 = M * TOPK
        sacc = tl.zeros((BN, 16), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            sw = tl.load(sh2_ptr + rn[:, None] * I + rk[None, :]).to(tl.bfloat16)
            sat = tl.load(act_ptr + (row0 + rm[None, :]) * I + rk[:, None], mask=rm[None, :] < M, other=0.0).to(tl.bfloat16)
            sacc += tl.dot(sw, sat) * tl.load(shs2_ptr + srow * SROW + kg)
        for mm in tl.static_range(MAXM):
            if mm < M:
                part = tl.sum(tl.where(rm[None, :] == mm, sacc, 0.0), axis=1) * tl.load(wts_ptr + row0 + mm)
                swsb = ws_ptr + ((mm * NB + pid_n) * (NS * SK)) * BN
                tl.store(swsb + (TOPK * SK + pid_k) * BN + tl.arange(0, BN), part)
                sprev = tl.atomic_add(cnt_ptr + mm * NB + pid_n, 1, sem="acq_rel")
                if sprev == NS * SK - 1:
                    stot = tl.zeros((BN,), dtype=tl.float32)
                    for q in tl.static_range(NS * SK):
                        stot += tl.load(swsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
                    tl.store(y_ptr + mm * H + rn, stot.to(tl.bfloat16))
                    tl.atomic_xchg(cnt_ptr + mm * NB + pid_n, 0)


def coral_moe_decode_v2(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                        bn_up=64, sk_up=8, bn_dn=64, sk_dn=2, max_m=4):
    """As coral_moe_decode, but the shared expert is loaded once per step for all M tokens."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M <= max_m and logits.shape == (M, E) and max_m <= 4
    nb_up, nb_dn = I // bn_up, H // bn_dn
    key = ("v2", x.device, bn_up, sk_up, bn_dn, sk_dn, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = (torch.zeros(max_m * NS * nb_up, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device))
    cnt_up, cnt_dn = st
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    wts = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    ws_up = torch.empty(M * top_k * nb_up * sk_up * 2 * bn_up, dtype=torch.float32, device=x.device)
    wsh_up = torch.empty(nb_up * sk_up * 2 * bn_up * 16, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sk_dn * bn_dn, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    grid0 = M * top_k + 1
    _moe_up_kernel_v2[(grid0, nb_up, sk_up)](x, logits, shared_gate_w, w13, s13, sh13, shs13, ws_up, wsh_up, cnt_up,
                                             act, wts, M, H=H, I=I, E=E, TOPK=top_k, RENORM=renorm, BN=bn_up,
                                             SK=sk_up, MAXM=4, num_warps=4)
    _moe_down_kernel_v2[(grid0, nb_dn, sk_dn)](act, wts, logits, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, M, H=H, I=I,
                                               E=E, TOPK=top_k, RENORM=renorm, BN=bn_dn, SK=sk_dn, MAXM=4,
                                               num_warps=4)
    return y


# ---------------------------------------------------------------------------------------------------------------
# v3 decode MoE: shared expert read ONCE per step, without a straggler. The shared expert gets SHR=4 grid rows;
# shared row r owns a BN/SHR slice of the output rows of every n-block and applies it to all M (<= 4) tokens, so
# per-CTA work matches a routed slot and the kernel keeps v1's fp32 register path (same occupancy).
# Grid dim0 = M*TOPK + SHR.  act/wts rows: routed (m, j) -> m*TOPK + j ; shared token m -> M*TOPK + m.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_up_kernel_v3(x_ptr, logits_ptr, sg_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, ws_ptr, wsh_ptr, cnt_ptr,
                      act_ptr, wts_ptr, M, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                      RENORM: tl.constexpr, BN: tl.constexpr, SK: tl.constexpr, SHR: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NB: tl.constexpr = I // BN
    G: tl.constexpr = H // 128 // SK
    SROW: tl.constexpr = H // 128
    srow_g = (pid_n * BN) // 128
    srow_u = (I + pid_n * BN) // 128
    if pid_s < M * TOPK:
        rg = pid_n * BN + tl.arange(0, BN)
        m = pid_s // TOPK
        j = pid_s % TOPK
        eid, wj = _moe_route_slot(logits_ptr, m, j, E, TOPK, RENORM)
        wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
        sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
        accg = tl.zeros((BN,), dtype=tl.float32)
        accu = tl.zeros((BN,), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            xk = tl.load(x_ptr + m * H + rk).to(tl.float32)
            wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
            wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
            accg += tl.sum(wg * xk[None, :], axis=1) * tl.load(sbase + srow_g * SROW + kg)
            accu += tl.sum(wu * xk[None, :], axis=1) * tl.load(sbase + srow_u * SROW + kg)
        wsb = ws_ptr + ((pid_s * NB + pid_n) * SK) * (2 * BN)
        tl.store(wsb + pid_k * (2 * BN) + tl.arange(0, BN), accg)
        tl.store(wsb + pid_k * (2 * BN) + BN + tl.arange(0, BN), accu)
        prev = tl.atomic_add(cnt_ptr + pid_s * NB + pid_n, 1, sem="acq_rel")
        if prev == SK - 1:
            g = tl.zeros((BN,), dtype=tl.float32)
            u = tl.zeros((BN,), dtype=tl.float32)
            for k in tl.static_range(SK):
                g += tl.load(wsb + k * (2 * BN) + tl.arange(0, BN), cache_modifier=".cg")
                u += tl.load(wsb + k * (2 * BN) + BN + tl.arange(0, BN), cache_modifier=".cg")
            tl.store(act_ptr + pid_s * I + rg, g / (1.0 + tl.exp(-g)) * u)
            tl.atomic_xchg(cnt_ptr + pid_s * NB + pid_n, 0)
        if pid_n == 0 and pid_k == 0:
            tl.store(wts_ptr + pid_s, wj)
    else:
        r = pid_s - M * TOPK                                   # shared row 0..SHR-1
        SB: tl.constexpr = BN // SHR
        rs = pid_n * BN + r * SB + tl.arange(0, SB)            # this row's slice of gate rows (up = rs + I)
        rm = tl.arange(0, 4)
        sh_g = tl.zeros((SB, 4), dtype=tl.float32)
        sh_u = tl.zeros((SB, 4), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            sgs = tl.load(shs13_ptr + srow_g * SROW + kg)
            sus = tl.load(shs13_ptr + srow_u * SROW + kg)
            swg = tl.load(sh13_ptr + rs[:, None] * H + rk[None, :]).to(tl.float32)
            swu = tl.load(sh13_ptr + (rs[:, None] + I) * H + rk[None, :]).to(tl.float32)
            for mm in tl.static_range(4):
                if mm < M:
                    sxk = tl.load(x_ptr + mm * H + rk).to(tl.float32)
                    sh_g += tl.where(rm[None, :] == mm, (tl.sum(swg * sxk[None, :], axis=1) * sgs)[:, None], 0.0)
                    sh_u += tl.where(rm[None, :] == mm, (tl.sum(swu * sxk[None, :], axis=1) * sus)[:, None], 0.0)
        tile: tl.constexpr = 2 * SB * 4
        offs = tl.arange(0, SB)[:, None] * 4 + rm[None, :]
        swsb = wsh_ptr + ((r * NB + pid_n) * SK) * tile
        tl.store(swsb + pid_k * tile + offs, sh_g)
        tl.store(swsb + pid_k * tile + SB * 4 + offs, sh_u)
        cidx = (M * TOPK + r) * NB + pid_n
        sprev = tl.atomic_add(cnt_ptr + cidx, 1, sem="acq_rel")
        if sprev == SK - 1:
            sg2 = tl.zeros((SB, 4), dtype=tl.float32)
            su2 = tl.zeros((SB, 4), dtype=tl.float32)
            for k in tl.static_range(SK):
                sg2 += tl.load(swsb + k * tile + offs, cache_modifier=".cg")
                su2 += tl.load(swsb + k * tile + SB * 4 + offs, cache_modifier=".cg")
            tl.store(act_ptr + (M * TOPK + rm[None, :]) * I + rs[:, None], sg2 / (1.0 + tl.exp(-sg2)) * su2,
                     mask=rm[None, :] < M)
            tl.atomic_xchg(cnt_ptr + cidx, 0)
        if r == 0 and pid_n == 0 and pid_k == 0:
            rh = tl.arange(0, H)
            gw = tl.load(sg_ptr + rh).to(tl.float32)
            for mm in tl.static_range(4):
                if mm < M:
                    xs = tl.load(x_ptr + mm * H + rh).to(tl.float32)
                    tl.store(wts_ptr + M * TOPK + mm, tl.sigmoid(tl.sum(xs * gw, axis=0)))


@triton.jit
def _moe_down_kernel_v3(act_ptr, wts_ptr, logits_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr, M,
                        H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr,
                        BN: tl.constexpr, SK: tl.constexpr, SHR: tl.constexpr):
    # partial vectors per (m, n-block): TOPK*SK routed + SK shared (the shared one assembled from SHR row slices)
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NB: tl.constexpr = H // BN
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    NP: tl.constexpr = (TOPK + 1) * SK                       # partial vectors summed per (m, n-block)
    ARR: tl.constexpr = TOPK * SK + SHR * SK                 # arrivals per (m, n-block)
    srow = (pid_n * BN) // 128
    if pid_s < M * TOPK:
        rn = pid_n * BN + tl.arange(0, BN)
        m = pid_s // TOPK
        j = pid_s % TOPK
        eid, _w = _moe_route_slot(logits_ptr, m, j, E, TOPK, RENORM)
        wbase = w2_ptr + eid.to(tl.int64) * (H * I)
        sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            a = tl.load(act_ptr + pid_s * I + rk)
            w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
            acc += tl.sum(w * a[None, :], axis=1) * tl.load(sbase + srow * SROW + kg)
        acc = acc * tl.load(wts_ptr + pid_s)
        wsb = ws_ptr + ((m * NB + pid_n) * NP) * BN
        tl.store(wsb + (j * SK + pid_k) * BN + tl.arange(0, BN), acc)
        prev = tl.atomic_add(cnt_ptr + m * NB + pid_n, 1, sem="acq_rel")
        if prev == ARR - 1:
            tot = tl.zeros((BN,), dtype=tl.float32)
            for q in tl.static_range(NP):
                tot += tl.load(wsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
            tl.store(y_ptr + m * H + rn, tot.to(tl.bfloat16))
            tl.atomic_xchg(cnt_ptr + m * NB + pid_n, 0)
    else:
        r = pid_s - M * TOPK
        SB: tl.constexpr = BN // SHR
        rsn = pid_n * BN + r * SB + tl.arange(0, SB)
        rm = tl.arange(0, 4)
        sacc = tl.zeros((SB, 4), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            sc = tl.load(shs2_ptr + srow * SROW + kg)
            sw = tl.load(sh2_ptr + rsn[:, None] * I + rk[None, :]).to(tl.float32)
            for mm in tl.static_range(4):
                if mm < M:
                    sa = tl.load(act_ptr + (M * TOPK + mm) * I + rk)
                    sacc += tl.where(rm[None, :] == mm, (tl.sum(sw * sa[None, :], axis=1) * sc)[:, None], 0.0)
        for mm in tl.static_range(4):
            if mm < M:
                part = tl.sum(tl.where(rm[None, :] == mm, sacc, 0.0), axis=1) * tl.load(wts_ptr + M * TOPK + mm)
                swsb = ws_ptr + ((mm * NB + pid_n) * NP) * BN
                tl.store(swsb + (TOPK * SK + pid_k) * BN + r * SB + tl.arange(0, SB), part)
                sprev = tl.atomic_add(cnt_ptr + mm * NB + pid_n, 1, sem="acq_rel")
                if sprev == ARR - 1:
                    stot = tl.zeros((BN,), dtype=tl.float32)
                    for q in tl.static_range(NP):
                        stot += tl.load(swsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
                    tl.store(y_ptr + mm * H + pid_n * BN + tl.arange(0, BN), stot.to(tl.bfloat16))
                    tl.atomic_xchg(cnt_ptr + mm * NB + pid_n, 0)


def coral_moe_decode_v3(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                        bn_up=64, sk_up=8, bn_dn=64, sk_dn=2, max_m=4, shr=4):
    """Decode MoE, shared expert read once per step (split over SHR grid rows by output rows). M <= 4."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M <= max_m <= 4 and logits.shape == (M, E) and bn_up % shr == 0 and bn_dn % shr == 0 and sk_up >= 2
    nb_up, nb_dn = I // bn_up, H // bn_dn
    key = ("v3", x.device, bn_up, sk_up, bn_dn, sk_dn, max_m, shr)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = (torch.zeros((max_m * top_k + shr) * nb_up, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device))
    cnt_up, cnt_dn = st
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    wts = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    ws_up = torch.empty(M * top_k * nb_up * sk_up * 2 * bn_up, dtype=torch.float32, device=x.device)
    wsh_up = torch.empty(shr * nb_up * sk_up * 2 * (bn_up // shr) * 4, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sk_dn * bn_dn, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    grid0 = M * top_k + shr
    _moe_up_kernel_v3[(grid0, nb_up, sk_up)](x, logits, shared_gate_w, w13, s13, sh13, shs13, ws_up, wsh_up, cnt_up,
                                             act, wts, M, H=H, I=I, E=E, TOPK=top_k, RENORM=renorm, BN=bn_up,
                                             SK=sk_up, SHR=shr, num_warps=4)
    _moe_down_kernel_v3[(grid0, nb_dn, sk_dn)](act, wts, logits, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, M, H=H, I=I,
                                               E=E, TOPK=top_k, RENORM=renorm, BN=bn_dn, SK=sk_dn, SHR=shr,
                                               num_warps=4)
    return y


# ---------------------------------------------------------------------------------------------------------------
# v4 decode MoE, expert-grouped for any M (<= 4): slot 0 = shared expert (all tokens, sigmoid gate); slot s >= 1 =
# s-th distinct routed expert across the M tokens (ascending id). Each expert's weights are read once per step and
# applied to every token routed to it. Every CTA recomputes the (tiny) routing; CTAs whose slot >= U+1 exit.
# act rows: slot * MAXM + m.  Down partials per (m, n-block): (U+1)*SK vectors summed in fixed slot order.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_route_all(logits_ptr, M, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr, MAXM: tl.constexpr):
    # returns union mask over experts [E] (int1) and per-token routing weights W [MAXM, E] (0 where not routed)
    re = tl.arange(0, E)
    rm = tl.arange(0, MAXM)
    union = re < 0
    W = tl.zeros((MAXM, E), dtype=tl.float32)
    for mm in tl.static_range(MAXM):
        if mm < M:
            lg = tl.load(logits_ptr + mm * E + re).to(tl.float32)
            p = tl.exp(lg - tl.max(lg, axis=0))
            p = p / tl.sum(p, axis=0)
            q = p
            sel = re < 0
            tot = 0.0
            for t in tl.static_range(TOPK):
                i = tl.argmax(q, axis=0)
                tot += tl.max(q, axis=0)
                sel = sel | (re == i)
                q = tl.where(re == i, -1.0, q)
            wrow = tl.where(sel, p, 0.0)
            if RENORM:
                wrow = wrow / tot
            W += tl.where(rm[:, None] == mm, wrow[None, :], 0.0)
            union = union | sel
    return union, W


@triton.jit
def _moe_slot_expert(union, E: tl.constexpr, slot):
    # slot >= 1 -> id of the (slot-1)-th set bit of union; also returns U (number of distinct routed experts)
    re = tl.arange(0, E)
    u = union.to(tl.int32)
    pos = tl.cumsum(u, axis=0) - 1
    U = tl.sum(u, axis=0)
    eid = tl.sum(tl.where((u == 1) & (pos == slot - 1), re, 0), axis=0)
    return eid, U


@triton.jit
def _moe_up_kernel_v4(x_ptr, logits_ptr, sg_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, ws_ptr, cnt_ptr, act_ptr,
                      wts_ptr, M, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                      RENORM: tl.constexpr, BN: tl.constexpr, SK: tl.constexpr, MAXM: tl.constexpr):
    slot = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NB: tl.constexpr = I // BN
    G: tl.constexpr = H // 128 // SK
    SROW: tl.constexpr = H // 128
    rm = tl.arange(0, MAXM)
    re = tl.arange(0, E)
    union, W = _moe_route_all(logits_ptr, M, E, TOPK, RENORM, MAXM)
    eid, U = _moe_slot_expert(union, E, slot)
    if slot <= U:
        if slot == 0:
            wbase = sh13_ptr
            sbase = shs13_ptr
            # shared gate weights for all tokens
            if pid_n == 0 and pid_k == 0:
                rh = tl.arange(0, H)
                gw = tl.load(sg_ptr + rh).to(tl.float32)
                for mm in tl.static_range(MAXM):
                    if mm < M:
                        xs = tl.load(x_ptr + mm * H + rh).to(tl.float32)
                        tl.store(wts_ptr + mm, tl.sigmoid(tl.sum(xs * gw, axis=0)))
            tw = (rm < M).to(tl.float32)                       # every token uses the shared expert
        else:
            wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
            sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
            tw = tl.sum(tl.where(re[None, :] == eid, W, 0.0), axis=1)   # [MAXM] routing weight of this expert
            if pid_n == 0 and pid_k == 0:
                tl.store(wts_ptr + slot * MAXM + rm, tw)
        rg = pid_n * BN + tl.arange(0, BN)
        srow_g = (pid_n * BN) // 128
        srow_u = (I + pid_n * BN) // 128
        accg = tl.zeros((BN, MAXM), dtype=tl.float32)
        accu = tl.zeros((BN, MAXM), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            sgs = tl.load(sbase + srow_g * SROW + kg)
            sus = tl.load(sbase + srow_u * SROW + kg)
            wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
            wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
            for mm in tl.static_range(MAXM):
                twm = tl.sum(tl.where(rm == mm, tw, 0.0), axis=0)
                if twm != 0.0:
                    xk = tl.load(x_ptr + mm * H + rk).to(tl.float32)
                    accg += tl.where(rm[None, :] == mm, (tl.sum(wg * xk[None, :], axis=1) * sgs)[:, None], 0.0)
                    accu += tl.where(rm[None, :] == mm, (tl.sum(wu * xk[None, :], axis=1) * sus)[:, None], 0.0)
        tile: tl.constexpr = 2 * BN * MAXM
        offs = tl.arange(0, BN)[:, None] * MAXM + rm[None, :]
        wsb = ws_ptr + ((slot * NB + pid_n) * SK) * tile
        tl.store(wsb + pid_k * tile + offs, accg)
        tl.store(wsb + pid_k * tile + BN * MAXM + offs, accu)
        prev = tl.atomic_add(cnt_ptr + slot * NB + pid_n, 1, sem="acq_rel")
        if prev == SK - 1:
            g = tl.zeros((BN, MAXM), dtype=tl.float32)
            u = tl.zeros((BN, MAXM), dtype=tl.float32)
            for k in tl.static_range(SK):
                g += tl.load(wsb + k * tile + offs, cache_modifier=".cg")
                u += tl.load(wsb + k * tile + BN * MAXM + offs, cache_modifier=".cg")
            tl.store(act_ptr + (slot * MAXM + rm[None, :]) * I + rg[:, None], g / (1.0 + tl.exp(-g)) * u,
                     mask=rm[None, :] < M)
            tl.atomic_xchg(cnt_ptr + slot * NB + pid_n, 0)


@triton.jit
def _moe_down_kernel_v4(act_ptr, wts_ptr, logits_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr, M,
                        H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr,
                        BN: tl.constexpr, SK: tl.constexpr, MAXM: tl.constexpr, MAXS: tl.constexpr):
    slot = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NB: tl.constexpr = H // BN
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    rm = tl.arange(0, MAXM)
    union, W = _moe_route_all(logits_ptr, M, E, TOPK, RENORM, MAXM)
    eid, U = _moe_slot_expert(union, E, slot)
    if slot <= U:
        if slot == 0:
            wbase = sh2_ptr
            sbase = shs2_ptr
        else:
            wbase = w2_ptr + eid.to(tl.int64) * (H * I)
            sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
        tw = tl.load(wts_ptr + slot * MAXM + rm, mask=rm < M, other=0.0)
        if slot == 0:
            tw = tl.load(wts_ptr + rm, mask=rm < M, other=0.0)
        rn = pid_n * BN + tl.arange(0, BN)
        srow = (pid_n * BN) // 128
        acc = tl.zeros((BN, MAXM), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            sc = tl.load(sbase + srow * SROW + kg)
            w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
            for mm in tl.static_range(MAXM):
                twm = tl.sum(tl.where(rm == mm, tw, 0.0), axis=0)
                if twm != 0.0:
                    a = tl.load(act_ptr + (slot * MAXM + mm) * I + rk)
                    acc += tl.where(rm[None, :] == mm, (tl.sum(w * a[None, :], axis=1) * sc * twm)[:, None], 0.0)
        NP = (U + 1) * SK
        for mm in tl.static_range(MAXM):
            if mm < M:
                part = tl.sum(tl.where(rm[None, :] == mm, acc, 0.0), axis=1)
                wsb = ws_ptr + ((mm * NB + pid_n) * (MAXS * SK)) * BN
                tl.store(wsb + (slot * SK + pid_k) * BN + tl.arange(0, BN), part)
                prev = tl.atomic_add(cnt_ptr + mm * NB + pid_n, 1, sem="acq_rel")
                if prev == NP - 1:
                    tot = tl.zeros((BN,), dtype=tl.float32)
                    for q in range(0, NP):
                        tot += tl.load(wsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
                    tl.store(y_ptr + mm * H + rn, tot.to(tl.bfloat16))
                    tl.atomic_xchg(cnt_ptr + mm * NB + pid_n, 0)


def coral_moe_decode_v4(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                        bn_up=64, sk_up=8, bn_dn=64, sk_dn=2, max_m=4):
    """Expert-grouped decode MoE for M <= 4: each distinct expert (and the shared expert) read once per step."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    MAXS = max_m * top_k + 1
    assert M <= max_m == 4 and logits.shape == (M, E)
    nb_up, nb_dn = I // bn_up, H // bn_dn
    key = ("v4", x.device, bn_up, sk_up, bn_dn, sk_dn, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = (torch.zeros(MAXS * nb_up, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device))
    cnt_up, cnt_dn = st
    S = M * top_k + 1                                    # upper bound on live slots for this M
    act = torch.empty(S * max_m, I, dtype=torch.float32, device=x.device)
    wts = torch.empty(S * max_m, dtype=torch.float32, device=x.device)
    ws_up = torch.empty(S * nb_up * sk_up * 2 * bn_up * max_m, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * MAXS * sk_dn * bn_dn, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _moe_up_kernel_v4[(S, nb_up, sk_up)](x, logits, shared_gate_w, w13, s13, sh13, shs13, ws_up, cnt_up, act, wts, M,
                                         H=H, I=I, E=E, TOPK=top_k, RENORM=renorm, BN=bn_up, SK=sk_up, MAXM=max_m,
                                         num_warps=4)
    _moe_down_kernel_v4[(S, nb_dn, sk_dn)](act, wts, logits, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, M, H=H, I=I, E=E,
                                           TOPK=top_k, RENORM=renorm, BN=bn_dn, SK=sk_dn, MAXM=max_m, MAXS=MAXS,
                                           num_warps=4)
    return y


# ---------------------------------------------------------------------------------------------------------------
# v5 decode MoE = v4 (expert-grouped, any M <= 4) with routing hoisted into one tiny launch. Tables:
#   tab_i32[0] = U (distinct routed experts), tab_i32[1 + s] = expert id of slot s >= 1 (slot 0 = shared)
#   tab_w[s * MAXM + m] = weight of slot s for token m (shared slot: sigmoid gate; 0 if token not routed)
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_route_kernel(x_ptr, logits_ptr, sg_ptr, tab_i32_ptr, tab_w_ptr, M, H: tl.constexpr, E: tl.constexpr,
                      TOPK: tl.constexpr, RENORM: tl.constexpr, MAXM: tl.constexpr, MAXS: tl.constexpr):
    re = tl.arange(0, E)
    rm = tl.arange(0, MAXM)
    union, W = _moe_route_all(logits_ptr, M, E, TOPK, RENORM, MAXM)
    u = union.to(tl.int32)
    pos = tl.cumsum(u, axis=0) - 1
    U = tl.sum(u, axis=0)
    tl.store(tab_i32_ptr, U)
    # scatter: expert e (if routed) -> slot pos[e] + 1
    tl.store(tab_i32_ptr + 2 + pos, re, mask=u == 1)
    for mm in tl.static_range(MAXM):
        wrow = tl.sum(tl.where(rm[:, None] == mm, W, 0.0), axis=0)          # [E] weights of token mm
        tl.store(tab_w_ptr + (pos + 1) * MAXM + mm, wrow, mask=u == 1)
    rh = tl.arange(0, H)
    gw = tl.load(sg_ptr + rh).to(tl.float32)
    for mm in tl.static_range(MAXM):
        if mm < M:
            xs = tl.load(x_ptr + mm * H + rh).to(tl.float32)
            tl.store(tab_w_ptr + mm, tl.sigmoid(tl.sum(xs * gw, axis=0)))
        else:
            tl.store(tab_w_ptr + mm, 0.0)


@triton.jit
def _moe_up_kernel_v5(x_ptr, tab_i32_ptr, tab_w_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, ws_ptr, cnt_ptr, act_ptr,
                      M, H: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, SK: tl.constexpr, MAXM: tl.constexpr):
    slot = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NB: tl.constexpr = I // BN
    G: tl.constexpr = H // 128 // SK
    SROW: tl.constexpr = H // 128
    U = tl.load(tab_i32_ptr)
    if slot <= U:
        rm = tl.arange(0, MAXM)
        if slot == 0:
            wbase = sh13_ptr
            sbase = shs13_ptr
        else:
            eid = tl.load(tab_i32_ptr + 1 + slot)
            wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
            sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
        tw = tl.load(tab_w_ptr + slot * MAXM + rm, mask=rm < M, other=0.0)
        rg = pid_n * BN + tl.arange(0, BN)
        srow_g = (pid_n * BN) // 128
        srow_u = (I + pid_n * BN) // 128
        accg = tl.zeros((BN, MAXM), dtype=tl.float32)
        accu = tl.zeros((BN, MAXM), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            sgs = tl.load(sbase + srow_g * SROW + kg)
            sus = tl.load(sbase + srow_u * SROW + kg)
            wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
            wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
            for mm in tl.static_range(MAXM):
                twm = tl.sum(tl.where(rm == mm, tw, 0.0), axis=0)
                if twm != 0.0:
                    xk = tl.load(x_ptr + mm * H + rk).to(tl.float32)
                    accg += tl.where(rm[None, :] == mm, (tl.sum(wg * xk[None, :], axis=1) * sgs)[:, None], 0.0)
                    accu += tl.where(rm[None, :] == mm, (tl.sum(wu * xk[None, :], axis=1) * sus)[:, None], 0.0)
        tile: tl.constexpr = 2 * BN * MAXM
        offs = tl.arange(0, BN)[:, None] * MAXM + rm[None, :]
        wsb = ws_ptr + ((slot * NB + pid_n) * SK) * tile
        tl.store(wsb + pid_k * tile + offs, accg)
        tl.store(wsb + pid_k * tile + BN * MAXM + offs, accu)
        prev = tl.atomic_add(cnt_ptr + slot * NB + pid_n, 1, sem="acq_rel")
        if prev == SK - 1:
            g = tl.zeros((BN, MAXM), dtype=tl.float32)
            u = tl.zeros((BN, MAXM), dtype=tl.float32)
            for k in tl.static_range(SK):
                g += tl.load(wsb + k * tile + offs, cache_modifier=".cg")
                u += tl.load(wsb + k * tile + BN * MAXM + offs, cache_modifier=".cg")
            tl.store(act_ptr + (slot * MAXM + rm[None, :]) * I + rg[:, None], g / (1.0 + tl.exp(-g)) * u,
                     mask=rm[None, :] < M)
            tl.atomic_xchg(cnt_ptr + slot * NB + pid_n, 0)


@triton.jit
def _moe_down_kernel_v5(act_ptr, tab_i32_ptr, tab_w_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr, M,
                        H: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, SK: tl.constexpr, MAXM: tl.constexpr,
                        MAXS: tl.constexpr):
    slot = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NB: tl.constexpr = H // BN
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    U = tl.load(tab_i32_ptr)
    if slot <= U:
        rm = tl.arange(0, MAXM)
        if slot == 0:
            wbase = sh2_ptr
            sbase = shs2_ptr
        else:
            eid = tl.load(tab_i32_ptr + 1 + slot)
            wbase = w2_ptr + eid.to(tl.int64) * (H * I)
            sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
        tw = tl.load(tab_w_ptr + slot * MAXM + rm, mask=rm < M, other=0.0)
        rn = pid_n * BN + tl.arange(0, BN)
        srow = (pid_n * BN) // 128
        acc = tl.zeros((BN, MAXM), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            sc = tl.load(sbase + srow * SROW + kg)
            w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
            for mm in tl.static_range(MAXM):
                twm = tl.sum(tl.where(rm == mm, tw, 0.0), axis=0)
                if twm != 0.0:
                    a = tl.load(act_ptr + (slot * MAXM + mm) * I + rk)
                    acc += tl.where(rm[None, :] == mm, (tl.sum(w * a[None, :], axis=1) * sc * twm)[:, None], 0.0)
        NP = (U + 1) * SK
        for mm in tl.static_range(MAXM):
            if mm < M:
                part = tl.sum(tl.where(rm[None, :] == mm, acc, 0.0), axis=1)
                wsb = ws_ptr + ((mm * NB + pid_n) * (MAXS * SK)) * BN
                tl.store(wsb + (slot * SK + pid_k) * BN + tl.arange(0, BN), part)
                prev = tl.atomic_add(cnt_ptr + mm * NB + pid_n, 1, sem="acq_rel")
                if prev == NP - 1:
                    tot = tl.zeros((BN,), dtype=tl.float32)
                    for q in range(0, NP):
                        tot += tl.load(wsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
                    tl.store(y_ptr + mm * H + rn, tot.to(tl.bfloat16))
                    tl.atomic_xchg(cnt_ptr + mm * NB + pid_n, 0)


def coral_moe_decode_v5(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                        bn_up=64, sk_up=8, bn_dn=64, sk_dn=2, max_m=4):
    """Expert-grouped decode MoE (any M <= 4): routing launch + up + down. Each distinct expert read once."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    MAXS = max_m * top_k + 1
    assert M <= max_m == 4 and logits.shape == (M, E)
    nb_up, nb_dn = I // bn_up, H // bn_dn
    key = ("v5", x.device, bn_up, sk_up, bn_dn, sk_dn, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = (torch.zeros(MAXS * nb_up, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device))
    cnt_up, cnt_dn = st
    S = M * top_k + 1
    tab_i32 = torch.empty(2 + MAXS, dtype=torch.int32, device=x.device)
    tab_w = torch.empty(MAXS * max_m, dtype=torch.float32, device=x.device)
    act = torch.empty(S * max_m, I, dtype=torch.float32, device=x.device)
    ws_up = torch.empty(S * nb_up * sk_up * 2 * bn_up * max_m, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * MAXS * sk_dn * bn_dn, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _moe_route_kernel[(1,)](x, logits, shared_gate_w, tab_i32, tab_w, M, H=H, E=E, TOPK=top_k, RENORM=renorm,
                            MAXM=max_m, MAXS=MAXS, num_warps=4)
    _moe_up_kernel_v5[(S, nb_up, sk_up)](x, tab_i32, tab_w, w13, s13, sh13, shs13, ws_up, cnt_up, act, M, H=H, I=I,
                                         BN=bn_up, SK=sk_up, MAXM=max_m, num_warps=4)
    _moe_down_kernel_v5[(S, nb_dn, sk_dn)](act, tab_i32, tab_w, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, M, H=H, I=I,
                                           BN=bn_dn, SK=sk_dn, MAXM=max_m, MAXS=MAXS, num_warps=4)
    return y


# ---------------------------------------------------------------------------------------------------------------
# P1: persistent decode MoE — ONE launch (grid = NPROG resident CTAs, <= #SMs so all are co-resident).
# Same math as v1 (token-major, shared expert = slot TOPK, W8A16, fixed-order reductions), but stage boundaries
# are memory flags instead of kernel boundaries:
#   up items   (m*NS+slot, n-block, k-split): per-CTA routing -> partials -> last arriver of (slot, n-block) writes
#              silu(g)*u to act and bumps act_rdy[slot] (release)
#   down items (slot, n-block, k-split): route + load the FP8 down tile and its scales FIRST (independent of act),
#              then spin on act_rdy[slot] == NBU (acquire), then dot -> partial; last arriver of (m, n-block) sums
#              NS*SKD partials in fixed order -> y
#   teardown   : last CTA to finish resets act_rdy and the done counter (up/down counters reset by last arrivers)
# Every CTA finishes all its up items before any down item, so spinning can never block an unfinished up item.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_persist_kernel(x_ptr, logits_ptr, sg_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, w2_ptr, s2_ptr, sh2_ptr,
                        shs2_ptr, ws_up_ptr, ws_dn_ptr, cnt_up_ptr, cnt_dn_ptr, rdy_ptr, done_ptr, act_ptr, y_ptr, M,
                        H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr,
                        BNU: tl.constexpr, SKU: tl.constexpr, BND: tl.constexpr, SKD: tl.constexpr,
                        NPROG: tl.constexpr, MAXM: tl.constexpr, DO_UP: tl.constexpr = True,
                        DO_DN: tl.constexpr = True, PREFETCH: tl.constexpr = True):
    pid = tl.program_id(0)
    NS: tl.constexpr = TOPK + 1
    NBU: tl.constexpr = I // BNU
    NBD: tl.constexpr = H // BND
    GU: tl.constexpr = H // 128 // SKU
    SROWU: tl.constexpr = H // 128
    KD: tl.constexpr = I // SKD
    SROWD: tl.constexpr = I // 128
    rh = tl.arange(0, H)
    # ---------------- up / gate ----------------
    n_up = M * NS * NBU * SKU * DO_UP
    for it in range(pid, n_up, NPROG):
        pid_k = it % SKU
        t = it // SKU
        pid_n = t % NBU
        pid_s = t // NBU
        m = pid_s // NS
        j = pid_s % NS
        if j < TOPK:
            eid, _wj = _moe_route_slot(logits_ptr, m, j, E, TOPK, RENORM)
            wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
            sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROWU)
        else:
            wbase = sh13_ptr
            sbase = shs13_ptr
        rg = pid_n * BNU + tl.arange(0, BNU)
        accg = tl.zeros((BNU,), dtype=tl.float32)
        accu = tl.zeros((BNU,), dtype=tl.float32)
        srow_g = (pid_n * BNU) // 128
        srow_u = (I + pid_n * BNU) // 128
        for gi in tl.static_range(GU):
            kg = pid_k * GU + gi
            rk = kg * 128 + tl.arange(0, 128)
            xk = tl.load(x_ptr + m * H + rk).to(tl.float32)
            wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
            wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
            accg += tl.sum(wg * xk[None, :], axis=1) * tl.load(sbase + srow_g * SROWU + kg)
            accu += tl.sum(wu * xk[None, :], axis=1) * tl.load(sbase + srow_u * SROWU + kg)
        wsb = ws_up_ptr + ((pid_s * NBU + pid_n) * SKU) * (2 * BNU)
        tl.store(wsb + pid_k * (2 * BNU) + tl.arange(0, BNU), accg)
        tl.store(wsb + pid_k * (2 * BNU) + BNU + tl.arange(0, BNU), accu)
        prev = tl.atomic_add(cnt_up_ptr + pid_s * NBU + pid_n, 1, sem="acq_rel")
        if prev == SKU - 1:
            g = tl.zeros((BNU,), dtype=tl.float32)
            u = tl.zeros((BNU,), dtype=tl.float32)
            for k in tl.static_range(SKU):
                g += tl.load(wsb + k * (2 * BNU) + tl.arange(0, BNU), cache_modifier=".cg")
                u += tl.load(wsb + k * (2 * BNU) + BNU + tl.arange(0, BNU), cache_modifier=".cg")
            tl.store(act_ptr + pid_s * I + rg, g / (1.0 + tl.exp(-g)) * u)
            tl.atomic_xchg(cnt_up_ptr + pid_s * NBU + pid_n, 0)
            tl.atomic_add(rdy_ptr + pid_s, 1, sem="release")
    # ---------------- down ----------------
    n_dn = M * NS * NBD * SKD * DO_DN
    for it in range(pid, n_dn, NPROG):
        pid_k = it % SKD
        t = it // SKD
        pid_n = t % NBD
        pid_s = t // NBD
        m = pid_s // NS
        j = pid_s % NS
        if j < TOPK:
            eid, wj = _moe_route_slot(logits_ptr, m, j, E, TOPK, RENORM)
            wbase = w2_ptr + eid.to(tl.int64) * (H * I)
            sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROWD)
        else:
            wbase = sh2_ptr
            sbase = shs2_ptr
            xs = tl.load(x_ptr + m * H + rh).to(tl.float32)
            wj = tl.sigmoid(tl.sum(xs * tl.load(sg_ptr + rh).to(tl.float32), axis=0))
        rn = pid_n * BND + tl.arange(0, BND)
        rk = pid_k * KD + tl.arange(0, KD)
        # weight tile + scales issued before the wait: overlaps the HBM fetch with the up stage's tail
        if PREFETCH:
            w = tl.load(wbase + rn[:, None] * I + rk[None, :])
            sc = tl.load(sbase + ((pid_n * BND) // 128) * SROWD + rk // 128)
        if DO_UP:
            r = tl.atomic_add(rdy_ptr + pid_s, 0, sem="acquire")
            while r < NBU:
                r = tl.atomic_add(rdy_ptr + pid_s, 0, sem="acquire")
        if not PREFETCH:
            w = tl.load(wbase + rn[:, None] * I + rk[None, :])
            sc = tl.load(sbase + ((pid_n * BND) // 128) * SROWD + rk // 128)
        a = tl.load(act_ptr + pid_s * I + rk, cache_modifier=".cg")
        acc = tl.sum(w.to(tl.float32) * (a * sc)[None, :], axis=1) * wj
        wsb = ws_dn_ptr + ((m * NBD + pid_n) * (NS * SKD)) * BND
        tl.store(wsb + (j * SKD + pid_k) * BND + tl.arange(0, BND), acc)
        prev = tl.atomic_add(cnt_dn_ptr + m * NBD + pid_n, 1, sem="acq_rel")
        if prev == NS * SKD - 1:
            tot = tl.zeros((BND,), dtype=tl.float32)
            for q in tl.static_range(NS * SKD):
                tot += tl.load(wsb + q * BND + tl.arange(0, BND), cache_modifier=".cg")
            tl.store(y_ptr + m * H + rn, tot.to(tl.bfloat16))
            tl.atomic_xchg(cnt_dn_ptr + m * NBD + pid_n, 0)
    # ---------------- teardown ----------------
    prev = tl.atomic_add(done_ptr, 1, sem="acq_rel")
    if prev == NPROG - 1:
        RR: tl.constexpr = triton.next_power_of_2(MAXM * NS)
        rr = tl.arange(0, RR)
        tl.atomic_xchg(rdy_ptr + rr, tl.zeros((RR,), dtype=tl.int32), mask=rr < MAXM * NS)
        tl.atomic_xchg(done_ptr, 0)


_PERSIST_OCC = {}


def _persist_max_ctas(ck, num_warps, dev):
    """Max co-resident CTAs on the device for a compiled kernel, from the CUDA occupancy API."""
    import cuda.bindings.driver as cu
    err, per_sm = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(cu.CUfunction(ck.function), 32 * num_warps,
                                                                 ck.metadata.shared)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return torch.cuda.get_device_properties(dev).multi_processor_count * max(1, per_sm)


def coral_moe_persist(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                      bn_up=64, sk_up=8, bn_dn=64, sk_dn=2, nprog=None, max_m=4, num_warps=4, do_up=True, do_dn=True,
                      prefetch=True):
    """Persistent decode MoE: routed top-k + gated shared expert in ONE launch. Returns bf16 [M, H].
    nprog=None -> max co-resident CTAs; any request above that is clamped (spin-waits need co-residency)."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M <= max_m and logits.shape == (M, E)
    assert bn_up <= 128 and bn_dn <= 128, "a tile must sit inside one 128-row FP8 scale block"
    nb_up, nb_dn = I // bn_up, H // bn_dn
    key = ("p1", x.device, bn_up, sk_up, bn_dn, sk_dn, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = (torch.zeros(max_m * NS * nb_up, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * NS, dtype=torch.int32, device=x.device),
                                torch.zeros(1, dtype=torch.int32, device=x.device))
    cnt_up, cnt_dn, rdy, done = st
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    ws_up = torch.empty(M * NS * nb_up * sk_up * 2 * bn_up, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sk_dn * bn_dn, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    okey = (x.device, bn_up, sk_up, bn_dn, sk_dn, max_m, num_warps, do_up, do_dn, prefetch)
    cap = _PERSIST_OCC.get(okey)
    n = sms if cap is None else (cap if nprog is None else min(nprog, cap))
    kw = dict(H=H, I=I, E=E, TOPK=top_k, RENORM=renorm, BNU=bn_up, SKU=sk_up, BND=bn_dn, SKD=sk_dn, MAXM=max_m,
              DO_UP=do_up, DO_DN=do_dn, PREFETCH=prefetch, num_warps=num_warps)
    ck = _moe_persist_kernel[(n,)](x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, ws_up, ws_dn,
                                   cnt_up, cnt_dn, rdy, done, act, y, M, NPROG=n, **kw)
    if cap is None and ck is not None:
        _PERSIST_OCC[okey] = _persist_max_ctas(ck, num_warps, x.device)
    return y


def persist_ctas(bn_up=64, sk_up=8, bn_dn=64, sk_dn=2, max_m=4, num_warps=4, device="cuda"):
    return _PERSIST_OCC.get((torch.device(device, torch.cuda.current_device()) if isinstance(device, str) else device,
                             bn_up, sk_up, bn_dn, sk_dn, max_m, num_warps))


# ---------------------------------------------------------------------------------------------------------------
# v6 decode MoE = v1 math with routing computed ONCE per layer (3 launches: route -> up -> down).
# v1 recomputed softmax + top-k in every CTA before its first weight load; with a table, each CTA's first HBM
# request goes out after one dependent 4-byte load. Measured (M=1/2/4, us): up+down from a ready table
# 17.3/27.4/44.0; with the 1-warp route launch 21.3/30.8/47.7; v1 22.1/34.9/57.9.
#   route: grid (M,)  eid[m*NS + j] (int32), wt[m*NS + j] (fp32); slot TOPK = shared expert (wt = sigmoid gate)
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _topk_rank_table(lg, eid_ptr, wt_ptr, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr):
    # top-k of E fp32 logits -> eid/wt at slots 0..TOPK-1 (descending, ties to lower index like torch.topk); weights =
    # softmax over all E, renormalized over the top-k. Serial TOPK max/argmax rounds: run it with num_warps=1 so
    # each round is a warp-shuffle reduction (no cross-warp barrier). A parallel [E, E] rank compare measured
    # 8-21 us vs 3.4 us for this (route kernel incl. launch, 2026-10-11).
    re = tl.arange(0, E)
    rt = tl.arange(0, 16)
    p = tl.exp(lg - tl.max(lg, axis=0))
    p = p / tl.sum(p, axis=0)
    ids = tl.zeros((16,), dtype=tl.int32)
    ws = tl.zeros((16,), dtype=tl.float32)
    for t in tl.static_range(TOPK):
        v = tl.max(p, axis=0)
        i = tl.argmax(p, axis=0)
        ids = tl.where(rt == t, i, ids)
        ws = tl.where(rt == t, v, ws)
        p = tl.where(re == i, -1.0, p)
    if RENORM:
        ws = ws / tl.sum(ws, axis=0)
    tl.store(eid_ptr + rt, ids, mask=rt < TOPK)
    tl.store(wt_ptr + rt, ws, mask=rt < TOPK)


@triton.jit
def _moe_route_tab_kernel(x_ptr, logits_ptr, sg_ptr, eid_ptr, wt_ptr, H: tl.constexpr, E: tl.constexpr,
                          TOPK: tl.constexpr, RENORM: tl.constexpr):
    m = tl.program_id(0)
    NS: tl.constexpr = TOPK + 1
    re = tl.arange(0, E)
    lg = tl.load(logits_ptr + m * E + re).to(tl.float32)
    _topk_rank_table(lg, eid_ptr + m * NS, wt_ptr + m * NS, E, TOPK, RENORM)
    rh = tl.arange(0, H)
    xs = tl.load(x_ptr + m * H + rh).to(tl.float32)
    tl.store(wt_ptr + m * NS + TOPK, tl.sigmoid(tl.sum(xs * tl.load(sg_ptr + rh).to(tl.float32), axis=0)))
    tl.store(eid_ptr + m * NS + TOPK, 0)


@triton.jit
def _moe_up_tab_kernel(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, ws_ptr, cnt_ptr, act_ptr,
            H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr, SK: tl.constexpr):
    pid_s = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = I // BN
    m = pid_s // NS; j = pid_s % NS
    G: tl.constexpr = H // 128 // SK
    SROW: tl.constexpr = H // 128
    if j < TOPK:
        eid = tl.load(eid_ptr + pid_s)
        wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
        sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
    else:
        wbase = sh13_ptr; sbase = shs13_ptr
    rg = pid_n * BN + tl.arange(0, BN)
    accg = tl.zeros((BN,), dtype=tl.float32); accu = tl.zeros((BN,), dtype=tl.float32)
    srow_g = (pid_n * BN) // 128; srow_u = (I + pid_n * BN) // 128
    for gi in tl.static_range(G):
        kg = pid_k * G + gi
        rk = kg * 128 + tl.arange(0, 128)
        xk = tl.load(x_ptr + m * H + rk).to(tl.float32)
        wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
        wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
        accg += tl.sum(wg * xk[None, :], axis=1) * tl.load(sbase + srow_g * SROW + kg)
        accu += tl.sum(wu * xk[None, :], axis=1) * tl.load(sbase + srow_u * SROW + kg)
    wsb = ws_ptr + ((pid_s * NB + pid_n) * SK) * (2 * BN)
    tl.store(wsb + pid_k * (2 * BN) + tl.arange(0, BN), accg)
    tl.store(wsb + pid_k * (2 * BN) + BN + tl.arange(0, BN), accu)
    prev = tl.atomic_add(cnt_ptr + pid_s * NB + pid_n, 1, sem="acq_rel")
    if prev == SK - 1:
        g = tl.zeros((BN,), dtype=tl.float32); u = tl.zeros((BN,), dtype=tl.float32)
        for k in tl.static_range(SK):
            g += tl.load(wsb + k * (2 * BN) + tl.arange(0, BN), cache_modifier=".cg")
            u += tl.load(wsb + k * (2 * BN) + BN + tl.arange(0, BN), cache_modifier=".cg")
        tl.store(act_ptr + pid_s * I + rg, g / (1.0 + tl.exp(-g)) * u)
        tl.atomic_xchg(cnt_ptr + pid_s * NB + pid_n, 0)

@triton.jit
def _moe_down_tab_kernel(act_ptr, eid_ptr, wt_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr,
            H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr, SK: tl.constexpr):
    pid_s = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = H // BN
    m = pid_s // NS; j = pid_s % NS
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    if j < TOPK:
        eid = tl.load(eid_ptr + pid_s)
        wbase = w2_ptr + eid.to(tl.int64) * (H * I)
        sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
    else:
        wbase = sh2_ptr; sbase = shs2_ptr
    rn = pid_n * BN + tl.arange(0, BN)
    srow = (pid_n * BN) // 128
    acc = tl.zeros((BN,), dtype=tl.float32)
    for gi in tl.static_range(G):
        kg = pid_k * G + gi
        rk = kg * 128 + tl.arange(0, 128)
        a = tl.load(act_ptr + pid_s * I + rk)
        w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1) * tl.load(sbase + srow * SROW + kg)
    acc = acc * tl.load(wt_ptr + pid_s)
    wsb = ws_ptr + ((m * NB + pid_n) * (NS * SK)) * BN
    tl.store(wsb + (j * SK + pid_k) * BN + tl.arange(0, BN), acc)
    prev = tl.atomic_add(cnt_ptr + m * NB + pid_n, 1, sem="acq_rel")
    if prev == NS * SK - 1:
        tot = tl.zeros((BN,), dtype=tl.float32)
        for q in tl.static_range(NS * SK):
            tot += tl.load(wsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
        tl.store(y_ptr + m * H + rn, tot.to(tl.bfloat16))
        tl.atomic_xchg(cnt_ptr + m * NB + pid_n, 0)


def coral_moe_decode_tab(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                         bn_up=64, sk_up=8, bn_dn=64, sk_dn=2, max_m=4, route_warps=1):
    """v6: route once (1 tiny launch) + up + down = 3 launches. Same math/numerics as coral_moe_decode."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M <= max_m and logits.shape == (M, E) and top_k <= 8
    nb_up, nb_dn = I // bn_up, H // bn_dn
    key = ("v6", x.device, bn_up, sk_up, bn_dn, sk_dn, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = (torch.zeros(max_m * NS * nb_up, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device))
    cnt_up, cnt_dn = st
    eid = torch.empty(M * NS, dtype=torch.int32, device=x.device)
    wt = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    ws_up = torch.empty(M * NS * nb_up * sk_up * 2 * bn_up, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sk_dn * bn_dn, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _moe_route_tab_kernel[(M,)](x, logits, shared_gate_w, eid, wt, H=H, E=E, TOPK=top_k, RENORM=renorm,
                                num_warps=route_warps)
    _moe_up_tab_kernel[(M * NS, nb_up, sk_up)](x, eid, w13, s13, sh13, shs13, ws_up, cnt_up, act, H=H, I=I, TOPK=top_k,
                                               BN=bn_up, SK=sk_up, num_warps=4)
    _moe_down_tab_kernel[(M * NS, nb_dn, sk_dn)](act, eid, wt, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, H=H, I=I,
                                                 TOPK=top_k, BN=bn_dn, SK=sk_dn, num_warps=4)
    return y


# ---------------------------------------------------------------------------------------------------------------
# v7 decode MoE: router GEMV + top-k fused (the last-arriving router CTA does softmax/top-k/renorm + shared gate and
# writes the eid/wt table) -> v6 up -> v6 down. 3 launches for the whole MoE block INCLUDING the router (v1 path was
# also router + 2), but no CTA recomputes routing and there is no separate route launch.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _router_route_kernel(x_ptr, w_ptr, sg_ptr, lg_ptr, cnt_ptr, eid_ptr, wt_ptr, M, K: tl.constexpr, E: tl.constexpr,
                         TOPK: tl.constexpr, RENORM: tl.constexpr, MAXM: tl.constexpr):
    pid = tl.program_id(0)                      # one router row (expert) per CTA, like rows_gemv(block_n=1)
    NS: tl.constexpr = TOPK + 1
    rk = tl.arange(0, K)
    w = tl.load(w_ptr + pid * K + rk).to(tl.float32)
    for m in range(M):
        x = tl.load(x_ptr + m * K + rk).to(tl.float32)
        tl.store(lg_ptr + m * E + pid, tl.sum(w * x, axis=0).to(tl.bfloat16))   # bf16 logits, as the stock router
    prev = tl.atomic_add(cnt_ptr, 1, sem="acq_rel")
    if prev == E - 1:
        re = tl.arange(0, E)
        for m in range(M):
            lg = tl.load(lg_ptr + m * E + re, cache_modifier=".cg").to(tl.float32)
            _topk_rank_table(lg, eid_ptr + m * NS, wt_ptr + m * NS, E, TOPK, RENORM)
            xs = tl.load(x_ptr + m * K + rk).to(tl.float32)
            gs = tl.sigmoid(tl.sum(xs * tl.load(sg_ptr + rk).to(tl.float32), axis=0))
            tl.store(wt_ptr + m * NS + TOPK, gs)
            tl.store(eid_ptr + m * NS + TOPK, 0)
        tl.atomic_xchg(cnt_ptr, 0)


def coral_moe_decode_fused(x, router_w, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                           bn_up=64, sk_up=8, bn_dn=64, sk_dn=2, max_m=4, router_warps=4):
    """v7: whole MoE block from hidden states: fused router+top-k (1 launch) + up + down = 3 launches."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M <= max_m and top_k <= 8 and router_w.shape == (E, H)
    nb_up, nb_dn = I // bn_up, H // bn_dn
    key = ("v7", x.device, bn_up, sk_up, bn_dn, sk_dn, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = (torch.zeros(max_m * NS * nb_up, dtype=torch.int32, device=x.device),
                                torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device),
                                torch.zeros(1, dtype=torch.int32, device=x.device))
    cnt_up, cnt_dn, cnt_r = st
    lg = torch.empty(M, E, dtype=torch.bfloat16, device=x.device)
    eid = torch.empty(M * NS, dtype=torch.int32, device=x.device)
    wt = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    ws_up = torch.empty(M * NS * nb_up * sk_up * 2 * bn_up, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sk_dn * bn_dn, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _router_route_kernel[(E,)](x, router_w, shared_gate_w, lg, cnt_r, eid, wt, M, K=H, E=E, TOPK=top_k, RENORM=renorm,
                               MAXM=max_m, num_warps=router_warps)
    _moe_up_tab_kernel[(M * NS, nb_up, sk_up)](x, eid, w13, s13, sh13, shs13, ws_up, cnt_up, act, H=H, I=I, TOPK=top_k,
                                               BN=bn_up, SK=sk_up, num_warps=4)
    _moe_down_tab_kernel[(M * NS, nb_dn, sk_dn)](act, eid, wt, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, H=H, I=I,
                                                 TOPK=top_k, BN=bn_dn, SK=sk_dn, num_warps=4)
    return y


# ---------------------------------------------------------------------------------------------------------------
# v9 decode MoE: route (1-warp launch) -> up WITHOUT split-K (each CTA: BN gate + BN up rows of one slot, full K,
# writes silu(g)*u directly: no atomics, no partial round trip) -> v6 split-K down. 3 launches.
# Why: the split-K last-arriver epilogue was most of the gap to the pure-read ceiling (up 10.06 -> 8.97 us without
# it at M=1; ceiling 7.95). The down stage keeps split-K: an all-slots-per-CTA down was slower (6.0 vs 5.6 us).
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_up_ns_body(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, act_ptr, H: tl.constexpr, I: tl.constexpr,
                    TOPK: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    NS: tl.constexpr = TOPK + 1
    m = pid_s // NS
    j = pid_s % NS
    SROW: tl.constexpr = H // 128
    if j < TOPK:
        eid = tl.load(eid_ptr + pid_s)
        wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
        sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
    else:
        wbase = sh13_ptr
        sbase = shs13_ptr
    rg = pid_n * BN + tl.arange(0, BN)
    srow_g = (pid_n * BN) // 128
    srow_u = (I + pid_n * BN) // 128
    accg = tl.zeros((BN,), dtype=tl.float32)
    accu = tl.zeros((BN,), dtype=tl.float32)
    for k0 in tl.static_range(0, H, KB):
        rk = k0 + tl.arange(0, KB)
        xk = tl.load(x_ptr + m * H + rk).to(tl.float32)
        sg = tl.load(sbase + srow_g * SROW + rk // 128)
        su = tl.load(sbase + srow_u * SROW + rk // 128)
        wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
        wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
        accg += tl.sum(wg * (xk * sg)[None, :], axis=1)
        accu += tl.sum(wu * (xk * su)[None, :], axis=1)
    tl.store(act_ptr + pid_s * I + rg, accg / (1.0 + tl.exp(-accg)) * accu)


@triton.jit
def _moe_up_ns_kernel(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, act_ptr, H: tl.constexpr, I: tl.constexpr,
                      TOPK: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    _moe_up_ns_body(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, act_ptr, H, I, TOPK, BN, KB)


# per-M (up BN, up KB, up warps, down BN, down SK, down warps), tuned H100 cold-L2 2026-10-11 (moe_nosplit_test.py,
# moe_tab_sweep.py)
_V9_CFG = {1: (8, 512, 8, 64, 2, 4), 2: (16, 256, 4, 64, 2, 4), 3: (16, 512, 8, 32, 1, 2), 4: (16, 512, 8, 32, 1, 2)}
# DFlash2 verify (bench/moe_v9_sweep.py, H100 2026-10-15, forced overlap 33/64 and 55/128 distinct experts):
# M=8 55.92 -> 52.88 us, M=16 94.06 -> 85.76 us vs the M=4 config. M=6..7 and 9..15 reuse the nearest tuned M.
_V9_CFG.update({m: (8, 512, 2, 64, 1, 4) for m in range(6, 9)})
_V9_CFG.update({m: (8, 512, 2, 128, 1, 4) for m in range(9, 17)})
_V9_CFG[5] = (16, 512, 4, 64, 1, 2)  # M=5 (4 draft steps verify), tuned at 24 distinct experts: 40.0 vs 42.5 us (moe_v9_m5_sweep.py)


def coral_moe_decode_v9(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                        max_m=4, cfg=None):
    """v9: route (1 launch) + no-split up + split-K down = 3 launches. Same math as coral_moe_decode."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M <= max_m and logits.shape == (M, E) and top_k <= 8
    bu, kb, wu, bd, sd, wd = cfg or _V9_CFG[M]
    nb_dn = H // bd
    key = ("v9", x.device, bd, sd, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device)
    cnt_dn = st
    eid = torch.empty(M * NS, dtype=torch.int32, device=x.device)
    wt = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sd * bd, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _moe_route_tab_kernel[(M,)](x, logits, shared_gate_w, eid, wt, H=H, E=E, TOPK=top_k, RENORM=renorm, num_warps=1)
    _moe_up_ns_kernel[(M * NS, I // bu)](x, eid, w13, s13, sh13, shs13, act, H=H, I=I, TOPK=top_k, BN=bu, KB=kb,
                                         num_warps=wu)
    _moe_down_tab_kernel[(M * NS, nb_dn, sd)](act, eid, wt, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, H=H, I=I, TOPK=top_k,
                                              BN=bd, SK=sd, num_warps=wd)
    return y


# ---------------------------------------------------------------------------------------------------------------
# W8A16 GEMV WITHOUT split-K for one token: each CTA owns BN rows over the full K (no atomics, no partial round
# trip). At M=1 this reaches the pure-read ceiling on every dense FP8 shape of the model (fp8gemv_nosplit.py,
# 2026-10-11): 2048x4096 4.92 us (ceil 4.54, split-K 8.15), 9216x2048 7.95 (7.80, 12.02), 12288x2048 9.97 (9.77,
# split-K 14.62, DeepGEMM 12.7 in-server). At M=4 it is slower than split-K, so callers use it for M=1 only.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _fp8w_gemv_ns_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K: tl.constexpr, SROW: tl.constexpr, BN: tl.constexpr,
                         KB: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    srow = (pid * BN) // 128
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in tl.static_range(0, K, KB):
        rk = k0 + tl.arange(0, KB)
        xs = tl.load(x_ptr + rk).to(tl.float32) * tl.load(s_ptr + srow * SROW + rk // 128)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :]).to(tl.float32)
        acc += tl.sum(w * xs[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(tl.bfloat16))


def fp8w_gemv_ns(x, w, s, block_n=8, kb=1024, num_warps=4):
    """One-token W8A16 GEMV: x bf16 [1, K], w fp8 [N, K], s fp32 [ceil(N/128), K/128]. N % block_n == 0, block_n | 128."""
    M, K = x.shape
    N = w.shape[0]
    assert M == 1 and N % block_n == 0 and 128 % block_n == 0 and K % kb == 0
    y = torch.empty(1, N, dtype=torch.bfloat16, device=x.device)
    _fp8w_gemv_ns_kernel[(N // block_n,)](x, w, s, y, N, K=K, SROW=K // 128, BN=block_n, KB=kb, num_warps=num_warps)
    return y


# ---------------------------------------------------------------------------------------------------------------
# v10: expert-GROUPED MoE for M=2..4 (MTP verify). Same 3 launches as v9 (route -> up -> split-K down), but each
# expert's weights are read once per step instead of once per token that chose it. Grid stays one CTA per (token,
# slot); a CTA whose expert was already picked by an earlier token exits, the first-picker CTA applies the tile to
# every token that chose that expert (<= MAXM dots per tile). The shared expert is owned by token 0's slot for all
# tokens. MT-Bench under MTP: 4 verify tokens hit 20.8 distinct of 32 routed slots (+ shared 4 -> 1) => ~39% fewer
# MoE bytes at M=4.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _grp_members(eid_ptr, pid_s, M, TOPK: tl.constexpr, MAXM: tl.constexpr):
    # -> (active, hit[MAXM] 0/1 per token, jpos[MAXM] slot of this expert in each token's table)
    NS: tl.constexpr = TOPK + 1
    m = pid_s // NS
    j = pid_s % NS
    rm = tl.arange(0, MAXM)
    rt = tl.arange(0, 16)
    e = tl.load(eid_ptr + pid_s)
    tile = tl.load(eid_ptr + rm[:, None] * NS + rt[None, :], mask=(rm[:, None] < M) & (rt[None, :] < TOPK), other=-1)
    match = (tile == e).to(tl.int32)
    hit_r = tl.sum(match, axis=1)
    jpos_r = tl.sum(match * rt[None, :], axis=1)
    act_r = tl.sum(tl.where(rm < m, hit_r, 0), axis=0) == 0
    is_sh = j == TOPK
    hit = tl.where(is_sh, (rm < M).to(tl.int32), hit_r)
    jpos = tl.where(is_sh, TOPK, jpos_r)
    active = tl.where(is_sh, m == 0, act_r)
    return active, hit, jpos


@triton.jit
def _pick(v, rm, mm):
    return tl.sum(tl.where(rm == mm, v, 0), axis=0)


@triton.jit
def _grp_up_dot(accg, accu, f, x_ptr, mm, H: tl.constexpr, rk, wg, wu, sg, su):
    if f != 0:
        xk = tl.load(x_ptr + mm * H + rk).to(tl.float32)
        accg += tl.sum(wg * (xk * sg)[None, :], axis=1)
        accu += tl.sum(wu * (xk * su)[None, :], axis=1)
    return accg, accu


@triton.jit
def _grp_up_store(accg, accu, f, jp, act_ptr, mm, NS: tl.constexpr, I: tl.constexpr, rg):
    if f != 0:
        tl.store(act_ptr + (mm * NS + jp) * I + rg, accg / (1.0 + tl.exp(-accg)) * accu)


@triton.jit
def _moe_up_grp_kernel(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, act_ptr, M, H: tl.constexpr,
                       I: tl.constexpr, TOPK: tl.constexpr, MAXM: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    NS: tl.constexpr = TOPK + 1
    SROW: tl.constexpr = H // 128
    active, hit, jpos = _grp_members(eid_ptr, pid_s, M, TOPK, MAXM)
    if active:
        rm = tl.arange(0, MAXM)
        f0 = _pick(hit, rm, 0); f1 = _pick(hit, rm, 1); f2 = _pick(hit, rm, 2); f3 = _pick(hit, rm, 3)
        j = pid_s % NS
        if j < TOPK:
            eid = tl.load(eid_ptr + pid_s)
            wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
            sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
        else:
            wbase = sh13_ptr
            sbase = shs13_ptr
        rg = pid_n * BN + tl.arange(0, BN)
        srow_g = (pid_n * BN) // 128
        srow_u = (I + pid_n * BN) // 128
        g0 = tl.zeros((BN,), dtype=tl.float32); u0 = tl.zeros((BN,), dtype=tl.float32)
        g1 = tl.zeros((BN,), dtype=tl.float32); u1 = tl.zeros((BN,), dtype=tl.float32)
        g2 = tl.zeros((BN,), dtype=tl.float32); u2 = tl.zeros((BN,), dtype=tl.float32)
        g3 = tl.zeros((BN,), dtype=tl.float32); u3 = tl.zeros((BN,), dtype=tl.float32)
        for k0 in tl.static_range(0, H, KB):
            rk = k0 + tl.arange(0, KB)
            sg = tl.load(sbase + srow_g * SROW + rk // 128)
            su = tl.load(sbase + srow_u * SROW + rk // 128)
            wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
            wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
            g0, u0 = _grp_up_dot(g0, u0, f0, x_ptr, 0, H, rk, wg, wu, sg, su)
            g1, u1 = _grp_up_dot(g1, u1, f1, x_ptr, 1, H, rk, wg, wu, sg, su)
            g2, u2 = _grp_up_dot(g2, u2, f2, x_ptr, 2, H, rk, wg, wu, sg, su)
            g3, u3 = _grp_up_dot(g3, u3, f3, x_ptr, 3, H, rk, wg, wu, sg, su)
        _grp_up_store(g0, u0, f0, _pick(jpos, rm, 0), act_ptr, 0, NS, I, rg)
        _grp_up_store(g1, u1, f1, _pick(jpos, rm, 1), act_ptr, 1, NS, I, rg)
        _grp_up_store(g2, u2, f2, _pick(jpos, rm, 2), act_ptr, 2, NS, I, rg)
        _grp_up_store(g3, u3, f3, _pick(jpos, rm, 3), act_ptr, 3, NS, I, rg)


@triton.jit
def _grp_dn_dot(acc, f, jp, act_ptr, mm, NS: tl.constexpr, I: tl.constexpr, rk, w, sc):
    if f != 0:
        a = tl.load(act_ptr + (mm * NS + jp) * I + rk)
        acc += tl.sum(w * a[None, :], axis=1) * sc
    return acc


@triton.jit
def _grp_dn_epi(acc, f, jp, mm, wt_ptr, ws_ptr, cnt_ptr, y_ptr, pid_n, pid_k, rn, H: tl.constexpr, NS: tl.constexpr,
                SK: tl.constexpr, NB: tl.constexpr, BN: tl.constexpr):
    if f != 0:
        acc = acc * tl.load(wt_ptr + mm * NS + jp)
        wsb = ws_ptr + ((mm * NB + pid_n) * (NS * SK)) * BN
        tl.store(wsb + (jp * SK + pid_k) * BN + tl.arange(0, BN), acc)
        prev = tl.atomic_add(cnt_ptr + mm * NB + pid_n, 1, sem="acq_rel")
        if prev == NS * SK - 1:
            tot = tl.zeros((BN,), dtype=tl.float32)
            for q in tl.static_range(NS * SK):
                tot += tl.load(wsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
            tl.store(y_ptr + mm * H + rn, tot.to(tl.bfloat16))
            tl.atomic_xchg(cnt_ptr + mm * NB + pid_n, 0)


@triton.jit
def _moe_down_grp_kernel(act_ptr, eid_ptr, wt_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr, M,
                         H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, MAXM: tl.constexpr, BN: tl.constexpr,
                         SK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = H // BN
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    active, hit, jpos = _grp_members(eid_ptr, pid_s, M, TOPK, MAXM)
    if active:
        rm = tl.arange(0, MAXM)
        f0 = _pick(hit, rm, 0); f1 = _pick(hit, rm, 1); f2 = _pick(hit, rm, 2); f3 = _pick(hit, rm, 3)
        j0 = _pick(jpos, rm, 0); j1 = _pick(jpos, rm, 1); j2 = _pick(jpos, rm, 2); j3 = _pick(jpos, rm, 3)
        j = pid_s % NS
        if j < TOPK:
            eid = tl.load(eid_ptr + pid_s)
            wbase = w2_ptr + eid.to(tl.int64) * (H * I)
            sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
        else:
            wbase = sh2_ptr
            sbase = shs2_ptr
        rn = pid_n * BN + tl.arange(0, BN)
        srow = (pid_n * BN) // 128
        a0 = tl.zeros((BN,), dtype=tl.float32); a1 = tl.zeros((BN,), dtype=tl.float32)
        a2 = tl.zeros((BN,), dtype=tl.float32); a3 = tl.zeros((BN,), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
            sc = tl.load(sbase + srow * SROW + kg)
            a0 = _grp_dn_dot(a0, f0, j0, act_ptr, 0, NS, I, rk, w, sc)
            a1 = _grp_dn_dot(a1, f1, j1, act_ptr, 1, NS, I, rk, w, sc)
            a2 = _grp_dn_dot(a2, f2, j2, act_ptr, 2, NS, I, rk, w, sc)
            a3 = _grp_dn_dot(a3, f3, j3, act_ptr, 3, NS, I, rk, w, sc)
        _grp_dn_epi(a0, f0, j0, 0, wt_ptr, ws_ptr, cnt_ptr, y_ptr, pid_n, pid_k, rn, H, NS, SK, NB, BN)
        _grp_dn_epi(a1, f1, j1, 1, wt_ptr, ws_ptr, cnt_ptr, y_ptr, pid_n, pid_k, rn, H, NS, SK, NB, BN)
        _grp_dn_epi(a2, f2, j2, 2, wt_ptr, ws_ptr, cnt_ptr, y_ptr, pid_n, pid_k, rn, H, NS, SK, NB, BN)
        _grp_dn_epi(a3, f3, j3, 3, wt_ptr, ws_ptr, cnt_ptr, y_ptr, pid_n, pid_k, rn, H, NS, SK, NB, BN)


# per-M (up BN, up KB, up warps, down BN, down SK, down warps); M=1 is never grouped (falls to v9)
_V10_CFG = {2: (16, 512, 8, 32, 1, 2), 3: (16, 512, 8, 32, 1, 2), 4: (16, 512, 8, 32, 1, 2)}


def coral_moe_decode_v10(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                         max_m=4, cfg=None):
    """v10: route (1 launch) + expert-grouped no-split up + expert-grouped split-K down = 3 launches. M=1 -> v9."""
    M, H = x.shape
    if M == 1:
        return coral_moe_decode_v9(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k, renorm,
                                   max_m, cfg)
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert 1 < M <= 4 and M <= max_m and logits.shape == (M, E) and top_k <= 8
    bu, kb, wu, bd, sd, wd = cfg or _V10_CFG[M]
    nb_dn = H // bd
    key = ("v10", x.device, bd, sd)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = torch.zeros(4 * nb_dn, dtype=torch.int32, device=x.device)
    cnt_dn = st
    eid = torch.empty(M * NS, dtype=torch.int32, device=x.device)
    wt = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sd * bd, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _moe_route_tab_kernel[(M,)](x, logits, shared_gate_w, eid, wt, H=H, E=E, TOPK=top_k, RENORM=renorm, num_warps=1)
    _moe_up_grp_kernel[(M * NS, I // bu)](x, eid, w13, s13, sh13, shs13, act, M, H=H, I=I, TOPK=top_k, MAXM=4, BN=bu,
                                          KB=kb, num_warps=wu)
    _moe_down_grp_kernel[(M * NS, nb_dn, sd)](act, eid, wt, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, M, H=H, I=I,
                                              TOPK=top_k, MAXM=4, BN=bd, SK=sd, num_warps=wd)
    return y


# v10b: branch-free grouped kernels. The per-token `if` blocks above kept the dequantized weight tile live across 4
# branches (138 / 123 regs vs v9's 78 / 56 -> half the resident CTAs, 2x slower). Here all MAXM tokens go through one
# tensor op (non-member tokens masked to 0; up to MAXM dots per tile regardless of membership) and the down epilogue
# is vectorized over tokens (masked vector atomics), so live state stays bounded.
@triton.jit
def _moe_up_grp3_kernel(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, act_ptr, M, H: tl.constexpr,
                        I: tl.constexpr, TOPK: tl.constexpr, MAXM: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr,
                        DOT: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    NS: tl.constexpr = TOPK + 1
    SROW: tl.constexpr = H // 128
    active, hit, jpos = _grp_members(eid_ptr, pid_s, M, TOPK, MAXM)
    if active:
        j = pid_s % NS
        if j < TOPK:
            eid = tl.load(eid_ptr + pid_s)
            wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
            sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
        else:
            wbase = sh13_ptr
            sbase = shs13_ptr
        rm = tl.arange(0, MAXM)
        rg = pid_n * BN + tl.arange(0, BN)
        srow_g = (pid_n * BN) // 128
        srow_u = (I + pid_n * BN) // 128
        accg = tl.zeros((BN, MAXM), dtype=tl.float32)
        accu = tl.zeros((BN, MAXM), dtype=tl.float32)
        for k0 in tl.static_range(0, H, KB):
            rk = k0 + tl.arange(0, KB)
            sg = tl.load(sbase + srow_g * SROW + rk // 128)
            su = tl.load(sbase + srow_u * SROW + rk // 128)
            wg = tl.load(wbase + rg[:, None] * H + rk[None, :])
            wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :])
            xk = tl.load(x_ptr + rm[:, None] * H + rk[None, :], mask=hit[:, None] != 0, other=0.0).to(tl.float32)
            if DOT:
                accg += tl.dot(wg.to(tl.float16), tl.trans((xk * sg[None, :]).to(tl.float16)))
                accu += tl.dot(wu.to(tl.float16), tl.trans((xk * su[None, :]).to(tl.float16)))
            else:
                accg += tl.sum(wg.to(tl.float32)[:, None, :] * (xk * sg[None, :])[None, :, :], axis=2)
                accu += tl.sum(wu.to(tl.float32)[:, None, :] * (xk * su[None, :])[None, :, :], axis=2)
        a = accg / (1.0 + tl.exp(-accg)) * accu
        tl.store(act_ptr + (rm[None, :] * NS + jpos[None, :]) * I + rg[:, None], a, mask=hit[None, :] != 0)


@triton.jit
def _moe_down_grp3_kernel(act_ptr, eid_ptr, wt_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr, M,
                          H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, MAXM: tl.constexpr, BN: tl.constexpr,
                          SK: tl.constexpr, DOT: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = H // BN
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    active, hit, jpos = _grp_members(eid_ptr, pid_s, M, TOPK, MAXM)
    if active:
        j = pid_s % NS
        if j < TOPK:
            eid = tl.load(eid_ptr + pid_s)
            wbase = w2_ptr + eid.to(tl.int64) * (H * I)
            sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
        else:
            wbase = sh2_ptr
            sbase = shs2_ptr
        rm = tl.arange(0, MAXM)
        hm = hit != 0
        slot = rm * NS + jpos
        rn = pid_n * BN + tl.arange(0, BN)
        srow = (pid_n * BN) // 128
        acc = tl.zeros((BN, MAXM), dtype=tl.float32)
        for gi in tl.static_range(G):
            kg = pid_k * G + gi
            rk = kg * 128 + tl.arange(0, 128)
            w = tl.load(wbase + rn[:, None] * I + rk[None, :])
            a = tl.load(act_ptr + slot[:, None] * I + rk[None, :], mask=hm[:, None], other=0.0)
            sc = tl.load(sbase + srow * SROW + kg)
            if DOT:
                acc += tl.dot(w.to(tl.float16), tl.trans(a.to(tl.float16))) * sc
            else:
                acc += tl.sum(w.to(tl.float32)[:, None, :] * a[None, :, :], axis=2) * sc
        acc = acc * tl.load(wt_ptr + slot, mask=hm, other=0.0)[None, :]
        wsb = ((rm * NB + pid_n) * (NS * SK)) * BN                      # [MAXM] per-token workspace base
        rb = tl.arange(0, BN)
        tl.store(ws_ptr + wsb[None, :] + (jpos[None, :] * SK + pid_k) * BN + rb[:, None], acc, mask=hm[None, :])
        prev = tl.atomic_add(cnt_ptr + rm * NB + pid_n, 1, mask=hm, sem="acq_rel")
        last = hm & (prev == NS * SK - 1)
        if tl.sum(last.to(tl.int32), axis=0) > 0:
            tot = tl.zeros((BN, MAXM), dtype=tl.float32)
            for q in tl.static_range(NS * SK):
                tot += tl.load(ws_ptr + wsb[None, :] + q * BN + rb[:, None], mask=last[None, :], other=0.0,
                               cache_modifier=".cg")
            tl.store(y_ptr + rm[None, :] * H + rn[:, None], tot.to(tl.bfloat16), mask=last[None, :])
            tl.atomic_xchg(cnt_ptr + rm * NB + pid_n, 0, mask=last)


def coral_moe_decode_v10b(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                          max_m=4, cfg=None, dot_up=False, dot_dn=False):
    """v10b: route + branch-free grouped up + grouped split-K down = 3 launches. M=1 -> v9."""
    M, H = x.shape
    if M == 1:
        return coral_moe_decode_v9(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k, renorm,
                                   max_m, cfg)
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert 1 < M <= 4 and M <= max_m and logits.shape == (M, E) and top_k <= 8
    bu, kb, wu, bd, sd, wd = cfg or _V10_CFG[M]
    nb_dn = H // bd
    key = ("v10", x.device, bd, sd)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = torch.zeros(4 * nb_dn, dtype=torch.int32, device=x.device)
    cnt_dn = st
    eid = torch.empty(M * NS, dtype=torch.int32, device=x.device)
    wt = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(4 * nb_dn * NS * sd * bd, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _moe_route_tab_kernel[(M,)](x, logits, shared_gate_w, eid, wt, H=H, E=E, TOPK=top_k, RENORM=renorm, num_warps=1)
    _moe_up_grp3_kernel[(M * NS, I // bu)](x, eid, w13, s13, sh13, shs13, act, M, H=H, I=I, TOPK=top_k, MAXM=4,
                                           BN=bu, KB=kb, DOT=dot_up, num_warps=wu)
    _moe_down_grp3_kernel[(M * NS, nb_dn, sd)](act, eid, wt, w2, s2, sh2, shs2, ws_dn, cnt_dn, y, M, H=H, I=I,
                                               TOPK=top_k, MAXM=4, BN=bd, SK=sd, DOT=dot_dn, num_warps=wd)
    return y


@triton.jit
def _moe_route_tab16_kernel(x_ptr, logits_ptr, sg_ptr, eid_ptr, wt_ptr, x16_ptr, H: tl.constexpr, E: tl.constexpr,
                            TOPK: tl.constexpr, RENORM: tl.constexpr):
    """_moe_route_tab_kernel + writes an f16 copy of x (the B operand of the CUDA mma MoE, coral_moe_mma.py)."""
    m = tl.program_id(0)
    NS: tl.constexpr = TOPK + 1
    re = tl.arange(0, E)
    lg = tl.load(logits_ptr + m * E + re).to(tl.float32)
    _topk_rank_table(lg, eid_ptr + m * NS, wt_ptr + m * NS, E, TOPK, RENORM)
    rh = tl.arange(0, H)
    xs = tl.load(x_ptr + m * H + rh).to(tl.float32)
    tl.store(x16_ptr + m * H + rh, xs.to(tl.float16))
    tl.store(wt_ptr + m * NS + TOPK, tl.sigmoid(tl.sum(xs * tl.load(sg_ptr + rh).to(tl.float32), axis=0)))
    tl.store(eid_ptr + m * NS + TOPK, 0)


@triton.jit
def _moe_up_ns_pdl_kernel(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, act_ptr, H: tl.constexpr,
                          I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    """_moe_up_ns_kernel that lets its PDL dependent (the CUDA expert-loop down, coral_moe_mma.moe_down1) launch at
    once, so the down kernel's weight loads overlap this kernel."""
    tl.extra.cuda.gdc_launch_dependents()
    _moe_up_ns_body(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, act_ptr, H, I, TOPK, BN, KB)


# ---------------------------------------------------------------------------------------------------------------
# Fused residual-add + Gemma RMSNorm + MoE router GEMV (decode, M <= 8): removes the router's own launch (~3.4 us,
# latency-bound on 1 MB) per MoE layer. Every CTA recomputes add+norm for the M tokens (2 x 2048 elements per token,
# deterministic -> identical in every CTA) and computes BN router rows from the bf16-rounded norm output, exactly the
# operand the separate router GEMV reads. CTA 0 alone writes h (new residual) and y (norm output) to NEW tensors, so
# no CTA reads a buffer another CTA is writing.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _add_norm_router_kernel(x_ptr, res_ptr, w_ptr, wr_ptr, y_ptr, h_ptr, lg_ptr, M, eps, K: tl.constexpr,
                            N: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, K)
    rn = pid * BN + tl.arange(0, BN)
    wn = tl.load(w_ptr + offs).to(tl.float32)
    wr = tl.load(wr_ptr + rn[:, None] * K + offs[None, :]).to(tl.float32)
    for m in range(M):
        h = tl.load(x_ptr + m * K + offs).to(tl.float32) + tl.load(res_ptr + m * K + offs).to(tl.float32)
        var = tl.sum(h * h, axis=0) / K
        y = (h * tl.rsqrt(var + eps) * (1.0 + wn)).to(tl.bfloat16)
        if pid == 0:
            tl.store(h_ptr + m * K + offs, h.to(tl.bfloat16))
            tl.store(y_ptr + m * K + offs, y)
        tl.store(lg_ptr + m * N + rn, tl.sum(wr * y.to(tl.float32)[None, :], axis=1).to(tl.bfloat16))


def add_norm_router(x, residual, weight, eps, router_w, block_n=4, num_warps=4):
    """-> (y = gemma_rmsnorm(x + residual) bf16, h = x + residual bf16, logits = y @ router_w^T bf16). New tensors."""
    M, K = x.shape
    N = router_w.shape[0]
    assert N % block_n == 0 and x.is_contiguous() and residual.is_contiguous() and router_w.is_contiguous()
    y = torch.empty_like(x)
    h = torch.empty_like(residual)
    lg = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _add_norm_router_kernel[(N // block_n,)](x, residual, weight, router_w, y, h, lg, M, eps, K=K, N=N, BN=block_n,
                                             num_warps=num_warps)
    return y, h, lg


# ---------------------------------------------------------------------------------------------------------------
# GDN in_proj_qkvz (M=1) with the causal depthwise conv1d update folded into the GEMV epilogue (CORAL_GDN_CONV_FOLD):
# each CTA owns BN output channels, so each conv channel's state is read and rewritten by exactly one CTA (no race).
# Channels < NCONV (q|k|v) get out = silu(bias + w0*s0 + w1*s1 + w2*s2 + w3*y) with y = the bf16-rounded projection
# (SGLang's conv update order); the state becomes [s1, s2, y]. z channels (>= NCONV) pass through.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _fp8w_gemv_ns_conv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, cs_ptr, cidx_ptr, cw_ptr, cb_ptr, cs_slot, cs_c, cs_t,
                              K: tl.constexpr, SROW: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr,
                              NCONV: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    srow = (pid * BN) // 128
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in tl.static_range(0, K, KB):
        rk = k0 + tl.arange(0, KB)
        xs = tl.load(x_ptr + rk).to(tl.float32) * tl.load(s_ptr + srow * SROW + rk // 128)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :]).to(tl.float32)
        acc += tl.sum(w * xs[None, :], axis=1)
    yb = acc.to(tl.bfloat16)
    if pid * BN < NCONV:
        slot = tl.load(cidx_ptr).to(tl.int64)
        base = cs_ptr + slot * cs_slot + rn.to(tl.int64) * cs_c
        s0 = tl.load(base)
        s1 = tl.load(base + cs_t)
        s2 = tl.load(base + 2 * cs_t)
        if HAS_BIAS:
            out = tl.load(cb_ptr + rn).to(tl.float32)
        else:
            out = tl.zeros((BN,), dtype=tl.float32)
        out += tl.load(cw_ptr + rn * 4 + 0).to(tl.float32) * s0.to(tl.float32)
        out += tl.load(cw_ptr + rn * 4 + 1).to(tl.float32) * s1.to(tl.float32)
        out += tl.load(cw_ptr + rn * 4 + 2).to(tl.float32) * s2.to(tl.float32)
        out += tl.load(cw_ptr + rn * 4 + 3).to(tl.float32) * yb.to(tl.float32)
        if SILU:
            out = out / (1.0 + tl.exp(-out))
        tl.store(base, s1)
        tl.store(base + cs_t, s2)
        tl.store(base + 2 * cs_t, yb.to(s0.dtype))
        tl.store(y_ptr + rn, out.to(tl.bfloat16))
    else:
        tl.store(y_ptr + rn, yb)


def fp8w_gemv_ns_conv(x, w, s, conv_state, cache_idx, conv_w, conv_b, nconv, silu=True, block_n=16, kb=512,
                      num_warps=4):
    """x bf16 [1, K]; w fp8 [N, K]; s fp32 [N/128, K/128]; conv_state [slots, >=nconv, 3] (any strides);
    cache_idx int [1]; conv_w [nconv, 4]; conv_b [nconv] or None. Updates conv_state[cache_idx] in place."""
    M, K = x.shape
    N = w.shape[0]
    assert M == 1 and N % block_n == 0 and nconv % block_n == 0 and 128 % block_n == 0 and conv_state.shape[-1] == 3
    cw = conv_w.reshape(conv_w.shape[0], -1)
    assert cw.shape[1] == 4 and cw.is_contiguous()
    y = torch.empty(1, N, dtype=torch.bfloat16, device=x.device)
    _fp8w_gemv_ns_conv_kernel[(N // block_n,)](
        x, w, s, y, conv_state, cache_idx, cw, conv_b if conv_b is not None else cw,
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        K=K, SROW=K // 128, BN=block_n, KB=kb, NCONV=nconv, HAS_BIAS=conv_b is not None, SILU=silu, num_warps=num_warps)
    return y



@triton.jit
def _add_gemma_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, eps, K: tl.constexpr):
    # _add_gemma_rmsnorm_quant_kernel without the FP8 copy: decode-sized consumers are W8A16 (read the bf16 output).
    row = tl.program_id(0)
    offs = tl.arange(0, K)
    h = tl.load(x_ptr + row * K + offs).to(tl.float32) + tl.load(res_ptr + row * K + offs).to(tl.float32)
    tl.store(res_ptr + row * K + offs, h.to(tl.bfloat16))
    var = tl.sum(h * h, axis=0) / K
    w = tl.load(w_ptr + offs).to(tl.float32)
    tl.store(x_ptr + row * K + offs, (h * tl.rsqrt(var + eps) * (1.0 + w)).to(tl.bfloat16))


def add_gemma_rmsnorm(x, residual, weight, eps, num_warps=4):
    """In place: residual += x; x = gemma_rmsnorm(residual). Same math/rounding as add_gemma_rmsnorm_quant."""
    M, K = x.shape
    assert x.is_contiguous() and residual.is_contiguous()
    _add_gemma_rmsnorm_kernel[(M,)](x, residual, weight, eps, K=K, num_warps=num_warps)


# ---------------------------------------------------------------------------------------------------------------
# add + Gemma RMSNorm + router GEMV + ROUTE (top-k table + shared-expert gate) in one launch, M = 1.
# Every CTA writes its router-logit rows (bf16, exactly what _moe_route_tab_kernel reads), then bumps a counter;
# the last CTA to arrive reloads all E logits and builds the [TOPK + 1] (expert id, weight) table with the same
# serial top-k (_topk_rank_table) plus the shared-expert gate sigmoid(y . sg) from the y it computed itself.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _add_norm_router_route_kernel(x_ptr, res_ptr, w_ptr, wr_ptr, sg_ptr, y_ptr, h_ptr, lg_ptr, cnt_ptr, eid_ptr, wt_ptr,
                                  eps, K: tl.constexpr, E: tl.constexpr, BN: tl.constexpr, TOPK: tl.constexpr,
                                  RENORM: tl.constexpr):
    pid = tl.program_id(0)
    NPROG: tl.constexpr = E // BN
    offs = tl.arange(0, K)
    rn = pid * BN + tl.arange(0, BN)
    wn = tl.load(w_ptr + offs).to(tl.float32)
    wr = tl.load(wr_ptr + rn[:, None] * K + offs[None, :]).to(tl.float32)
    h = tl.load(x_ptr + offs).to(tl.float32) + tl.load(res_ptr + offs).to(tl.float32)
    var = tl.sum(h * h, axis=0) / K
    y = (h * tl.rsqrt(var + eps) * (1.0 + wn)).to(tl.bfloat16)
    if pid == 0:
        tl.store(h_ptr + offs, h.to(tl.bfloat16))
        tl.store(y_ptr + offs, y)
    tl.store(lg_ptr + rn, tl.sum(wr * y.to(tl.float32)[None, :], axis=1).to(tl.bfloat16))
    prev = tl.atomic_add(cnt_ptr, 1, sem="acq_rel")
    if prev == NPROG - 1:
        re = tl.arange(0, E)
        lg = tl.load(lg_ptr + re, cache_modifier=".cg").to(tl.float32)
        _topk_rank_table(lg, eid_ptr, wt_ptr, E, TOPK, RENORM)
        sgv = tl.sigmoid(tl.sum(y.to(tl.float32) * tl.load(sg_ptr + offs).to(tl.float32), axis=0))
        tl.store(wt_ptr + TOPK, sgv)
        tl.store(eid_ptr + TOPK, 0)
        tl.atomic_xchg(cnt_ptr, 0)


_NRR_CNT = {}


def add_norm_router_route(x, residual, weight, eps, router_w, shared_gate_w, top_k=8, renorm=True, block_n=4,
                          num_warps=4):
    """M=1 -> (y, h, logits, eid int32 [top_k+1], wt fp32 [top_k+1]); eid/wt in _moe_route_tab_kernel's layout."""
    M, K = x.shape
    E = router_w.shape[0]
    assert M == 1 and E % block_n == 0 and top_k <= 8
    cnt = _NRR_CNT.get(x.device)
    if cnt is None:
        cnt = _NRR_CNT[x.device] = torch.zeros(1, dtype=torch.int32, device=x.device)
    y = torch.empty_like(x)
    h = torch.empty_like(residual)
    lg = torch.empty(1, E, dtype=torch.bfloat16, device=x.device)
    eid = torch.empty(top_k + 1, dtype=torch.int32, device=x.device)
    wt = torch.empty(top_k + 1, dtype=torch.float32, device=x.device)
    _add_norm_router_route_kernel[(E // block_n,)](x, residual, weight, router_w, shared_gate_w, y, h, lg, cnt, eid, wt,
                                                   eps, K=K, E=E, BN=block_n, TOPK=top_k, RENORM=renorm,
                                                   num_warps=num_warps)
    return y, h, lg, eid, wt


def coral_moe_decode_v9_pre(x, eid, wt, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, max_m=4, cfg=None):
    """v9 MoE (up + split-K down = 2 launches) from a precomputed routing table (add_norm_router_route)."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M == 1
    bu, kb, wu, bd, sd, wd = cfg or _V9_CFG[1]
    nb_dn = H // bd
    key = ("v9", x.device, bd, sd, max_m)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device)
    act = torch.empty(NS, I, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(nb_dn * NS * sd * bd, dtype=torch.float32, device=x.device)
    y = torch.empty(1, H, dtype=torch.bfloat16, device=x.device)
    _moe_up_ns_kernel[(NS, I // bu)](x, eid, w13, s13, sh13, shs13, act, H=H, I=I, TOPK=top_k, BN=bu, KB=kb, num_warps=wu)
    _moe_down_tab_kernel[(NS, nb_dn, sd)](act, eid, wt, w2, s2, sh2, shs2, ws_dn, st, y, H=H, I=I, TOPK=top_k, BN=bd,
                                          SK=sd, num_warps=wd)
    return y


@triton.jit
def _add_norm_router_mt_kernel(x_ptr, res_ptr, w_ptr, wr_ptr, y_ptr, h_ptr, lg_ptr, M, eps, K: tl.constexpr,
                               N: tl.constexpr, BN: tl.constexpr, MP: tl.constexpr, KC: tl.constexpr):
    """_add_norm_router_kernel with all M (<= MP) tokens in parallel: pass 1 computes every token's rstd from the
    full row; pass 2 walks K in KC chunks, recomputing h/y per chunk (same per-element math -> same bf16 y) and
    accumulating the router dots as [BN, MP]. CTA 0 writes h and y."""
    pid = tl.program_id(0)
    rm = tl.arange(0, MP)
    rn = pid * BN + tl.arange(0, BN)
    mm = rm[:, None] < M
    offs = tl.arange(0, K)
    h = (tl.load(x_ptr + rm[:, None] * K + offs[None, :], mask=mm, other=0.0).to(tl.float32)
         + tl.load(res_ptr + rm[:, None] * K + offs[None, :], mask=mm, other=0.0).to(tl.float32))
    var = tl.sum(h * h, axis=1) / K
    rstd = tl.rsqrt(var + eps)
    acc = tl.zeros((BN, MP), dtype=tl.float32)
    for k0 in tl.static_range(0, K, KC):
        rk = k0 + tl.arange(0, KC)
        hc = (tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm, other=0.0).to(tl.float32)
              + tl.load(res_ptr + rm[:, None] * K + rk[None, :], mask=mm, other=0.0).to(tl.float32))
        yc = (hc * rstd[:, None] * (1.0 + tl.load(w_ptr + rk).to(tl.float32))[None, :]).to(tl.bfloat16)
        if pid == 0:
            tl.store(h_ptr + rm[:, None] * K + rk[None, :], hc.to(tl.bfloat16), mask=mm)
            tl.store(y_ptr + rm[:, None] * K + rk[None, :], yc, mask=mm)
        wr = tl.load(wr_ptr + rn[:, None] * K + rk[None, :]).to(tl.float32)
        acc += tl.sum(wr[:, None, :] * yc.to(tl.float32)[None, :, :], axis=2)
    tl.store(lg_ptr + rm[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=rm[None, :] < M)


def add_norm_router_mt(x, residual, weight, eps, router_w, block_n=2, kc=256, num_warps=4):
    """Multi-token (M <= 16) add_norm_router: tokens in parallel instead of a serial loop. y and h use the same
    per-element math as add_norm_router; the router logits accumulate over K in KC chunks, so they can differ from
    the serial kernel's in the last bf16 bit. MP (padded token tile) is 8 up to 8 tokens, as on the v0.5.18 stack,
    and 16 for 9..16 (DFlash2 block 16, or 2 requests at block 8). Pass 1 holds [MP, K] fp32 per CTA, so MP=16
    wants more warps; bench/norm_router_m16_test.py measures it."""
    M, K = x.shape
    N = router_w.shape[0]
    assert M <= 16 and N % block_n == 0
    y = torch.empty_like(x)
    h = torch.empty_like(residual)
    lg = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _add_norm_router_mt_kernel[(N // block_n,)](x, residual, weight, router_w, y, h, lg, M, eps, K=K, N=N, BN=block_n,
                                                MP=8 if M <= 8 else 16, KC=kc, num_warps=num_warps)
    return y, h, lg


# ---------------------------------------------------------------------------------------------------------------
# v10c: expert-grouped like v10, but with v9's single-token register footprint: the owner CTA of a distinct expert
# walks its member tokens one at a time and RE-LOADS the weight tile inside each token's branch (L1/L2 hits after the
# first), so nothing stays live across tokens (v10's 138-register problem). Duplicate-slot CTAs exit immediately,
# so each distinct expert's HBM bytes are pulled by one CTA instead of by M separate CTAs through L2.
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_up_grp_c_kernel(x_ptr, eid_ptr, w13_ptr, s13_ptr, sh13_ptr, shs13_ptr, act_ptr, M, H: tl.constexpr,
                         I: tl.constexpr, TOPK: tl.constexpr, MAXM: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    NS: tl.constexpr = TOPK + 1
    SROW: tl.constexpr = H // 128
    active, hit, jpos = _grp_members(eid_ptr, pid_s, M, TOPK, MAXM)
    if active:
        j = pid_s % NS
        if j < TOPK:
            eid = tl.load(eid_ptr + pid_s)
            wbase = w13_ptr + eid.to(tl.int64) * (2 * I * H)
            sbase = s13_ptr + eid.to(tl.int64) * ((2 * I // 128) * SROW)
        else:
            wbase = sh13_ptr
            sbase = shs13_ptr
        rm = tl.arange(0, MAXM)
        rg = pid_n * BN + tl.arange(0, BN)
        srow_g = (pid_n * BN) // 128
        srow_u = (I + pid_n * BN) // 128
        for mm in tl.static_range(MAXM):
            if _pick(hit, rm, mm) != 0:
                accg = tl.zeros((BN,), dtype=tl.float32)
                accu = tl.zeros((BN,), dtype=tl.float32)
                for k0 in tl.static_range(0, H, KB):
                    rk = k0 + tl.arange(0, KB)
                    xk = tl.load(x_ptr + mm * H + rk).to(tl.float32)
                    sg = tl.load(sbase + srow_g * SROW + rk // 128)
                    su = tl.load(sbase + srow_u * SROW + rk // 128)
                    wg = tl.load(wbase + rg[:, None] * H + rk[None, :]).to(tl.float32)
                    wu = tl.load(wbase + (rg[:, None] + I) * H + rk[None, :]).to(tl.float32)
                    accg += tl.sum(wg * (xk * sg)[None, :], axis=1)
                    accu += tl.sum(wu * (xk * su)[None, :], axis=1)
                tl.store(act_ptr + (mm * NS + _pick(jpos, rm, mm)) * I + rg, accg / (1.0 + tl.exp(-accg)) * accu)


@triton.jit
def _moe_down_grp_c_kernel(act_ptr, eid_ptr, wt_ptr, w2_ptr, s2_ptr, sh2_ptr, shs2_ptr, ws_ptr, cnt_ptr, y_ptr, M,
                           H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, MAXM: tl.constexpr, BN: tl.constexpr,
                           SK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = H // BN
    G: tl.constexpr = I // 128 // SK
    SROW: tl.constexpr = I // 128
    active, hit, jpos = _grp_members(eid_ptr, pid_s, M, TOPK, MAXM)
    if active:
        j = pid_s % NS
        if j < TOPK:
            eid = tl.load(eid_ptr + pid_s)
            wbase = w2_ptr + eid.to(tl.int64) * (H * I)
            sbase = s2_ptr + eid.to(tl.int64) * ((H // 128) * SROW)
        else:
            wbase = sh2_ptr
            sbase = shs2_ptr
        rm = tl.arange(0, MAXM)
        rn = pid_n * BN + tl.arange(0, BN)
        srow = (pid_n * BN) // 128
        for mm in tl.static_range(MAXM):
            if _pick(hit, rm, mm) != 0:
                jp = _pick(jpos, rm, mm)
                acc = tl.zeros((BN,), dtype=tl.float32)
                for gi in tl.static_range(G):
                    kg = pid_k * G + gi
                    rk = kg * 128 + tl.arange(0, 128)
                    a = tl.load(act_ptr + (mm * NS + jp) * I + rk)
                    w = tl.load(wbase + rn[:, None] * I + rk[None, :]).to(tl.float32)
                    acc += tl.sum(w * a[None, :], axis=1) * tl.load(sbase + srow * SROW + kg)
                _grp_dn_epi(acc, 1, jp, mm, wt_ptr, ws_ptr, cnt_ptr, y_ptr, pid_n, pid_k, rn, H, NS, SK, NB, BN)


def coral_moe_decode_v10c(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                          max_m=8, cfg=None):
    """route + grouped up (serial member tokens) + grouped split-K down = 3 launches, M = 2..16 (MAXM 8, or 16 for
    M > 8: DFlash2 block-16 verify; the M <= 8 launch is unchanged)."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert 1 < M <= 16
    maxm = 8 if M <= 8 else 16
    bu, kb, wu, bd, sd, wd = cfg or _V9_CFG[min(M, 5)]
    nb_dn = H // bd
    key = ("v10c", x.device, bd, sd, maxm)
    st = _MOE_STATE.get(key)
    if st is None:
        st = _MOE_STATE[key] = torch.zeros(maxm * nb_dn, dtype=torch.int32, device=x.device)
    eid = torch.empty(M * NS, dtype=torch.int32, device=x.device)
    wt = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(maxm * nb_dn * NS * sd * bd, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _moe_route_tab_kernel[(M,)](x, logits, shared_gate_w, eid, wt, H=H, E=E, TOPK=top_k, RENORM=renorm, num_warps=1)
    _moe_up_grp_c_kernel[(M * NS, I // bu)](x, eid, w13, s13, sh13, shs13, act, M, H=H, I=I, TOPK=top_k, MAXM=maxm,
                                            BN=bu, KB=kb, num_warps=wu)
    _moe_down_grp_c_kernel[(M * NS, nb_dn, sd)](act, eid, wt, w2, s2, sh2, shs2, ws_dn, st, y, M, H=H, I=I, TOPK=top_k,
                                                MAXM=maxm, BN=bd, SK=sd, num_warps=wd)
    return y
