"""Pinned Gated Residual (hyper-connection) program shared by rollout and trainer.

One program at every token count. Triton kernels with fixed tiles own every reduction
and the low-rank up and inject projections; the down projection uses the shared BF16
batch-invariant GEMM.
Rounding boundaries: normalized branches, both projection outputs and the
low-rank activation are BF16; gates, the branch sum and the residual update are
FP32 with one final BF16 cast.

Rounding follows SGLang's BF16 module: projection outputs are rounded to BF16 before
their sigmoid. The kernel structure matches the trainer's kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _grouped_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    normed_ptr,  # BF16 [T, N*C]
    rstd_ptr,  # FP32 [T, N]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + EPS)
    w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * (1.0 + w)
    tl.store(normed_ptr + base + offs, y.to(normed_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + t * N + c, rstd)


# Fixed tiles at every token count keep each row's arithmetic batch-invariant.
# Swept on GB300 with HBM-cold weights; re-tune when the hyper-connection shapes change.
_UP_BM, _UP_BC, _UP_BR, _UP_WARPS, _UP_STAGES = 64, 32, 64, 4, 3
_INJECT_BLOCK = 1024


# Swept with HBM-cold weights at T=16 (decode) and T=8192 (prefill).
_DOWN_BM, _DOWN_BN, _DOWN_BK, _DOWN_SPLITS, _DOWN_WARPS, _DOWN_STAGES = (
    64,
    64,
    128,
    16,
    8,
    3,
)


@triton.jit(do_not_specialize=["T"])
def _down_partial_kernel(
    x_ptr,  # BF16 [T, K], normalized branches
    w_ptr,  # BF16 [R, K], down-projection weight
    part_ptr,  # FP32 [SPLITS, T, R]
    T,
    K: tl.constexpr,
    R: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Partial x @ W^T over K-split s (fixed split count and tiles, independent of T)."""
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    m_mask = offs_m < T
    n_mask = offs_n < R
    rows = offs_m.to(tl.int64)
    KS: tl.constexpr = K // SPLITS
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in tl.range(0, KS, BK):
        offs_k = split * KS + k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rows[:, None] * K + offs_k[None, :], mask=m_mask[:, None], other=0.0
        )
        w = tl.load(
            w_ptr + offs_n[None, :].to(tl.int64) * K + offs_k[:, None],
            mask=n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(x, w, acc)
    tl.store(
        part_ptr + (split * T + rows[:, None]) * R + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit(do_not_specialize=["T"])
def _down_act_kernel(
    part_ptr,
    act_ptr,
    T,
    R: tl.constexpr,
    SPLITS: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """act = BF16(silu(BF16(sum_s partial_s) / N)), splits summed in order."""
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < R
    z = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        z += tl.load(part_ptr + (s * T + t) * R + offs, mask=mask, other=0.0)
    x = z.to(tl.bfloat16).to(tl.float32) / N
    tl.store(act_ptr + t * R + offs, (x * tl.sigmoid(x)).to(tl.bfloat16), mask=mask)


@triton.jit
def _branch_gate(
    z, normed_ptr, rows, offs_c, mask, c: tl.constexpr, C: tl.constexpr, N: tl.constexpr
):
    """sigmoid(BF16 logit) * normed for branch c."""
    z = z.to(tl.bfloat16).to(tl.float32)
    nrm = tl.load(
        normed_ptr + rows[:, None] * (N * C) + c * C + offs_c[None, :],
        mask=mask,
        other=0.0,
    )
    return tl.sigmoid(z) * nrm.to(tl.float32)


@triton.jit(do_not_specialize=["T"])
def _up_mix_kernel(
    act_ptr,  # BF16 [T, R], silu(down-projection / 4)
    w_ptr,  # BF16 [4*C, R], up-projection weight
    normed_ptr,  # BF16 [T, 4*C]
    out_ptr,  # BF16 [T, C]
    T,
    R: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
    BR: tl.constexpr,
):
    """mixed = mean_c(sigmoid(act @ W_up_c^T) * normed_c) for 4 branches; logits rounded to BF16."""
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_c = tl.program_id(1) * BC + tl.arange(0, BC)
    m_mask = offs_m < T
    c_mask = offs_c < C
    rows = offs_m.to(tl.int64)
    cols = offs_c.to(tl.int64)
    z0 = tl.zeros([BM, BC], dtype=tl.float32)
    z1 = tl.zeros([BM, BC], dtype=tl.float32)
    z2 = tl.zeros([BM, BC], dtype=tl.float32)
    z3 = tl.zeros([BM, BC], dtype=tl.float32)
    for r0 in tl.range(0, R, BR):
        offs_r = r0 + tl.arange(0, BR)
        act = tl.load(
            act_ptr + rows[:, None] * R + offs_r[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        w = w_ptr + offs_r[:, None]
        z0 = tl.dot(
            act,
            tl.load(w + (0 * C + cols)[None, :] * R, mask=c_mask[None, :], other=0.0),
            z0,
        )
        z1 = tl.dot(
            act,
            tl.load(w + (1 * C + cols)[None, :] * R, mask=c_mask[None, :], other=0.0),
            z1,
        )
        z2 = tl.dot(
            act,
            tl.load(w + (2 * C + cols)[None, :] * R, mask=c_mask[None, :], other=0.0),
            z2,
        )
        z3 = tl.dot(
            act,
            tl.load(w + (3 * C + cols)[None, :] * R, mask=c_mask[None, :], other=0.0),
            z3,
        )
    mask = m_mask[:, None] & c_mask[None, :]
    acc = _branch_gate(z0, normed_ptr, rows, offs_c, mask, 0, C, 4)
    acc += _branch_gate(z1, normed_ptr, rows, offs_c, mask, 1, C, 4)
    acc += _branch_gate(z2, normed_ptr, rows, offs_c, mask, 2, C, 4)
    acc += _branch_gate(z3, normed_ptr, rows, offs_c, mask, 3, C, 4)
    acc = acc / 4
    tl.store(
        out_ptr + rows[:, None] * C + offs_c[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _inject_gate(
    normed_ptr, w_ptr, t, c, N: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr
):
    """2 * sigmoid(BF16(normed_t . W_inject_c) / N), FP32 fixed-block dot."""
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in tl.static_range(0, W, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < W
        x = tl.load(normed_ptr + t * W + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c * W + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * w
    u = tl.sum(acc, axis=0).to(tl.bfloat16).to(tl.float32)
    return 2.0 * tl.sigmoid(u / N)


@triton.jit(do_not_specialize=["T"])
def _inject_kernel(
    normed_ptr,  # BF16 [T, W]
    w_ptr,  # BF16 [N, W]
    h_ptr,  # FP32 [T, N]
    T,
    N: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    tl.store(h_ptr + t * N + c, _inject_gate(normed_ptr, w_ptr, t, c, N, W, BLOCK))


@triton.jit
def _combine_row(
    res_ptr,
    y_ptr,
    out_ptr,
    t,
    c,
    a,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + offs, (r + a * y).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _combine_kernel(
    res_ptr,  # BF16 [T, N*C]
    y_ptr,  # BF16 [T, C]
    h_ptr,  # FP32 [T, N]
    out_ptr,  # BF16 [T, N*C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    _combine_row(
        res_ptr, y_ptr, out_ptr, t, c, tl.load(h_ptr + t * N + c), N, C, BLOCK_C
    )


@triton.jit(do_not_specialize=["T"])
def _inject_combine_kernel(
    res_ptr,  # BF16 [T, N*C]
    y_ptr,  # BF16 [T, C]
    normed_ptr,  # BF16 [T, N*C]
    w_ptr,  # BF16 [N, N*C], inject weight
    out_ptr,  # BF16 [T, N*C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """hc_combine(residual, y, hc_inject(normed)) without materializing the write gates."""
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    a = _inject_gate(normed_ptr, w_ptr, t, c, N, N * C, BLOCK)
    _combine_row(res_ptr, y_ptr, out_ptr, t, c, a, N, C, BLOCK_C)


def _check_bf16_2d(name, x, width):
    if x.ndim != 2 or x.shape[1] != width or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError(
            f"TOP HC requires CUDA BF16 {name} [T, {width}], got {tuple(x.shape)} {x.dtype}"
        )


def _down_act(normed, w_down, n):
    """BF16 silu(BF16(normed @ W_down^T) / n) via fixed split-K partials."""
    T, K = normed.shape
    R = w_down.shape[0]
    part = torch.empty(_DOWN_SPLITS, T, R, dtype=torch.float32, device=normed.device)
    _down_partial_kernel[
        (triton.cdiv(T, _DOWN_BM), triton.cdiv(R, _DOWN_BN), _DOWN_SPLITS)
    ](
        normed,
        w_down,
        part,
        T,
        K=K,
        R=R,
        SPLITS=_DOWN_SPLITS,
        BM=_DOWN_BM,
        BN=_DOWN_BN,
        BK=_DOWN_BK,
        num_warps=_DOWN_WARPS,
        num_stages=_DOWN_STAGES,
    )
    act = torch.empty(T, R, dtype=torch.bfloat16, device=normed.device)
    _down_act_kernel[(T,)](
        part, act, T, R=R, SPLITS=_DOWN_SPLITS, N=n, BLOCK=triton.next_power_of_2(R)
    )
    return act


def grouped_gemma_rmsnorm(x2d, weight, n, eps):
    """Per-branch offset RMSNorm of BF16 or FP32 input: BF16 normed [T, n*C], FP32 rstd [T, n]."""
    T, W = x2d.shape
    if W % n:
        raise ValueError(
            f"TOP grouped RMSNorm width {W} is not divisible by {n} groups"
        )
    if x2d.dtype not in (torch.bfloat16, torch.float32) or not x2d.is_cuda:
        raise ValueError(
            f"TOP grouped RMSNorm requires CUDA BF16/FP32 input, got {x2d.dtype}"
        )
    if weight.shape != (W,):
        raise ValueError(
            f"TOP grouped RMSNorm weight must be [{W}], got {tuple(weight.shape)}"
        )
    C = W // n
    x2d = x2d.contiguous()
    normed = torch.empty(T, W, dtype=torch.bfloat16, device=x2d.device)
    rstd = torch.empty(T, n, dtype=torch.float32, device=x2d.device)
    if T:
        _grouped_rmsnorm_kernel[(T * n,)](
            x2d,
            weight.contiguous(),
            normed,
            rstd,
            T,
            N=n,
            C=C,
            EPS=eps,
            BLOCK_C=triton.next_power_of_2(C),
        )
    return normed, rstd


def hc_mix(x2d, norm_weight, w_down, w_up, n, eps):
    """Stands in for the stock GatedResidual mix, which picks its kernel by row count and whose fused_hc_mix
    accumulates the down projection with device-scope atomics, so the summation order varies across replays.

    Read the branch stream: (mixed [T, C], normed [T, n*C], rstd [T, n]).

    mixed = mean_c(sigmoid(W_up silu(W_down normed / n))_c * normed_c)
    """
    T, W = x2d.shape
    C = W // n
    _check_bf16_2d("input", x2d, W)
    if w_down.shape[1] != W or w_up.shape != (W, w_down.shape[0]):
        raise ValueError("TOP HC requires W_down [r, n*C] and W_up [n*C, r]")
    normed, rstd = grouped_gemma_rmsnorm(x2d, norm_weight, n, eps)
    mixed = torch.empty(T, C, dtype=torch.bfloat16, device=x2d.device)
    if T == 0:
        return mixed, normed, rstd
    R = w_down.shape[0]
    if R % _UP_BR or n != 4 or W % (_DOWN_SPLITS * _DOWN_BK):
        raise ValueError(
            f"TOP HC requires 4 branches, low rank % {_UP_BR}, width % {_DOWN_SPLITS * _DOWN_BK}"
        )
    act = _down_act(normed, w_down.contiguous(), n)
    _up_mix_kernel[(triton.cdiv(T, _UP_BM), triton.cdiv(C, _UP_BC))](
        act,
        w_up.contiguous(),
        normed,
        mixed,
        T,
        R=R,
        C=C,
        BM=_UP_BM,
        BC=_UP_BC,
        BR=_UP_BR,
        num_warps=_UP_WARPS,
        num_stages=_UP_STAGES,
    )
    return mixed, normed, rstd


def hc_inject(normed, w_inject, n):
    """FP32 write gates [T, n] = 2 * sigmoid(W_inject normed / n)."""
    _check_bf16_2d("normed", normed, w_inject.shape[1])
    if w_inject.shape[0] != n:
        raise ValueError(f"TOP HC inject weight must have {n} rows")
    T, W = normed.shape
    h = torch.empty(T, n, dtype=torch.float32, device=normed.device)
    if T:
        _inject_kernel[(T, n)](
            normed, w_inject.contiguous(), h, T, N=n, W=W, BLOCK=_INJECT_BLOCK
        )
    return h


def hc_combine(residual2d, y2d, h_post, n):
    """Write the block output back: out_c = residual_c + h_post_c * y, BF16 [T, n*C]."""
    T, W = residual2d.shape
    C = W // n
    _check_bf16_2d("residual", residual2d, W)
    _check_bf16_2d("block output", y2d, C)
    if h_post.shape != (T, n) or h_post.dtype != torch.float32:
        raise ValueError(f"TOP HC requires FP32 write gates [{T}, {n}]")
    residual2d = residual2d.contiguous()
    out = torch.empty_like(residual2d)
    if T:
        _combine_kernel[(T * n,)](
            residual2d,
            y2d.contiguous(),
            h_post.contiguous(),
            out,
            T,
            N=n,
            C=C,
            BLOCK_C=triton.next_power_of_2(C),
        )
    return out


def hc_inject_combine(residual2d, y2d, normed, w_inject, n):
    """Stands in for the stock hc_combine (kernels/ops/elementwise/hc_combine.py), which sums the inject dot in a
    different order; switching would also need the trainer's combine, which receives only the write gates, to
    receive ``normed`` and ``w_inject``.

    hc_combine(residual2d, y2d, hc_inject(normed, w_inject, n), n) in one kernel (bitwise equal)."""
    T, W = residual2d.shape
    C = W // n
    _check_bf16_2d("residual", residual2d, W)
    _check_bf16_2d("block output", y2d, C)
    _check_bf16_2d("normed", normed, W)
    if w_inject.shape != (n, W):
        raise ValueError(f"TOP HC inject weight must be [{n}, {W}]")
    residual2d = residual2d.contiguous()
    out = torch.empty_like(residual2d)
    if T:
        _inject_combine_kernel[(T * n,)](
            residual2d,
            y2d.contiguous(),
            normed.contiguous(),
            w_inject.contiguous(),
            out,
            T,
            N=n,
            C=C,
            BLOCK_C=triton.next_power_of_2(C),
            BLOCK=_INJECT_BLOCK,
        )
    return out
