import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

PDL = is_arch_support_pdl()


@triton.jit
def _linear_k_groups(
    X,
    W,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    total = tl.full((BM, BN), 0, tl.float32)
    for group in range(tl.cdiv(K, 1024)):
        partial = tl.full((BM, BN), 0, tl.float32)
        for tile in range(1024 // BK):
            k = group * 1024 + tile * BK + ks
            x = tl.load(
                X + rows[:, None] * SX + k[None, :],
                (rows[:, None] < M) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                W + cols[None, :] * SW + k[:, None],
                (cols[None, :] < N) & (k[:, None] < K),
                other=0,
            )
            partial = tl.dot(x, w, partial)
        if group == 0:
            total = partial
        else:
            total = total + partial
    if HAS_BIAS:
        total = total + tl.load(B + cols, cols < N, other=0)[None, :]
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        total,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _linear_k_group_partials(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr = 16,
    X2=None,
    W2=None,
):
    """With X2 and W2, groups past the first cdiv(K, 1024) compute X2 @ W2.T
    into the next (groups, M, N) block of P."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    group = tl.program_id(2)
    if X2 is not None:
        GROUPS: tl.constexpr = (K + 1023) // 1024
        if group >= GROUPS:
            X = X2
            W = W2
            P += GROUPS * M * N
            group -= GROUPS
    ks = tl.arange(0, BLOCK_K)
    partial = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for tile in range(1024 // BLOCK_K):
        k = group * 1024 + tile * BLOCK_K + ks
        x = tl.load(
            X + rows[:, None] * SX + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + cols[None, :] * SW + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        partial = tl.dot(x, w, partial)
    tl.store(
        P + (group * M + rows[:, None]) * N + cols[None, :],
        partial,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _linear_k_group_sum(
    P, B, Y, M, N: tl.constexpr, GROUPS: tl.constexpr, HAS_BIAS: tl.constexpr
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < M * N
    total = tl.load(P + offsets, mask, other=0)
    for group in tl.static_range(1, GROUPS):
        total = total + tl.load(P + group * M * N + offsets, mask, other=0)
    if HAS_BIAS:
        total = total + tl.load(B + offsets % N, mask, other=0)
    tl.store(Y + offsets, total, mask)


def _k_group_blocks(m, n):
    return (
        (16, 16)
        if n < 32 or m <= 16
        else (16, 32)
        if m <= 32
        else (32, 32)
        if m <= 64
        else (64, 32)
        if m <= 128
        else (64, 64)
        if m <= 256 or n < 256
        else (64, 128)
    )


def linear_k_group_partials(flat, weight, block_k=256, num_stages=3):
    """Per-1024-wide-K-group partial products of _linear_k_groups, shape (groups, m, n)."""
    m, n, k = flat.shape[0], weight.shape[0], weight.shape[1]
    block_m, block_n = _k_group_blocks(m, n)
    wide = block_n == 128
    groups = triton.cdiv(k, 1024)
    partials = torch.empty((groups, m, n), device=flat.device, dtype=torch.float32)
    _linear_k_group_partials[
        (triton.cdiv(m, block_m), triton.cdiv(n, block_n), groups)
    ](
        flat,
        weight,
        partials,
        m,
        n,
        k,
        flat.stride(0),
        weight.stride(0),
        block_n,
        block_k,
        block_m,
        num_warps=8 if wide else 4,
        num_stages=min(num_stages, 2) if wide else num_stages,
        enable_fp_fusion=False,
        launch_pdl=PDL,
    )
    return partials


def linear_k_group_partials_pair(x, weight, x2, weight2):
    """linear_k_group_partials of (x, weight) and (x2, weight2) in one launch,
    shape (2, groups, m, n); bitwise equal to the two separate calls."""
    m, n, k = x.shape[0], weight.shape[0], weight.shape[1]
    assert x2.shape == x.shape and x2.stride() == x.stride()
    assert weight2.shape == weight.shape and weight2.stride() == weight.stride()
    block_m, block_n = _k_group_blocks(m, n)
    wide = block_n == 128
    groups = triton.cdiv(k, 1024)
    partials = torch.empty((2, groups, m, n), device=x.device, dtype=torch.float32)
    _linear_k_group_partials[
        (triton.cdiv(m, block_m), triton.cdiv(n, block_n), 2 * groups)
    ](
        x,
        weight,
        partials,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        block_n,
        256,
        block_m,
        x2,
        weight2,
        num_warps=8 if wide else 4,
        num_stages=2 if wide else 3,
        enable_fp_fusion=False,
        launch_pdl=PDL,
    )
    return partials


def _linear_k_groups_split(flat, weight, bias, out, block_k=256, num_stages=3):
    m, n = flat.shape[0], weight.shape[0]
    partials = linear_k_group_partials(flat, weight, block_k, num_stages)
    groups = partials.shape[0]
    _linear_k_group_sum[(triton.cdiv(m * n, 256),)](
        partials,
        bias,
        out,
        m,
        n,
        groups,
        bias is not None,
        enable_fp_fusion=False,
        launch_pdl=PDL,
    )


def uses_k_groups(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and torch.cuda.get_device_capability(x.device)[0] in (9, 10)
        and x.dtype == weight.dtype == torch.bfloat16
        and x.stride(-1) == weight.stride(-1) == 1
        and weight.shape[1] >= 2560
        and weight.shape[0] not in (1, 2, 4)
    )


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out_dtype = out_dtype or x.dtype
    if uses_k_groups(x, weight):
        flat = x.reshape(-1, x.shape[-1])
        out = torch.empty(
            (flat.shape[0], weight.shape[0]), device=x.device, dtype=out_dtype
        )
        tiles = triton.cdiv(flat.shape[0], 16) * triton.cdiv(weight.shape[0], 32)
        if flat.shape[0] and tiles < 256:
            _linear_k_groups_split(flat, weight, bias, out)
        elif flat.shape[0]:
            block_n, block_k = (16, 128) if weight.shape[0] <= 16 else (128, 64)
            _linear_k_groups[
                (triton.cdiv(flat.shape[0], 64), triton.cdiv(weight.shape[0], block_n))
            ](
                flat,
                weight,
                bias,
                out,
                flat.shape[0],
                weight.shape[0],
                weight.shape[1],
                flat.stride(0),
                weight.stride(0),
                bias is not None,
                64,
                block_n,
                block_k,
                num_warps=4,
                num_stages=4,
                enable_fp_fusion=False,
            )
        return out.view(*x.shape[:-1], weight.shape[0])
    if out_dtype == torch.float32 and x.is_cuda and x.dtype != torch.float32:
        flat = x.reshape(-1, x.shape[-1])
        out = (
            torch.mm(flat, weight.T, out_dtype=torch.float32)
            if bias is None
            else torch.addmm(bias, flat, weight.T, out_dtype=torch.float32)
        )
        return out.view(*x.shape[:-1], weight.shape[0])
    return F.linear(x, weight, bias).to(out_dtype)
