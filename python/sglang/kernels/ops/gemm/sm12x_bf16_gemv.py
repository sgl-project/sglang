"""Low-M BF16 GEMVs for NVIDIA SM120/SM121 (RTX PRO 6000, RTX 5090, DGX Spark GB10).

b12x_bf16_gemv runs FlashInfer's b12x SIMT GEMV with one row per tile and no bias.
sm12x_bf16_gemv streams the weights through a Triton kernel, one CTA per 32 output
columns, for at most MAX_M rows and at least MIN_N columns. At one row cuBLAS
streams wide BF16 projections at ~175 GB/s on GB10 against ~240 GB/s for it; on
RTX PRO 6000 cuBLAS takes up to 1.4x as long at two to sixteen rows (3072 -> 2304).
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, device_guard

# RTX PRO 6000, RTX 5090: at 17-32 rows this kernel takes 0.63-1.00x cuBLAS's time
# on inkling2's projections and LM head (2048 to 201024 columns).
MAX_M = 32
# RTX PRO 6000: at 1024 columns cuBLAS is up to 1.25x faster (K 2048-14336);
# from 2048 columns this kernel takes 0.56-1.03x cuBLAS's time.
MIN_N = 2048
_BN, _BK = 32, 256


@cache_once
def _b12x_projection(device: int, n: int, k: int, out_dtype: torch.dtype):
    from flashinfer.experimental.b12x.gemm.bf16_gemv._kernel import (
        compile_projection,
    )

    # One row per tile: fastest at 1-3 rows on inkling2's 2304 -> 32 and 2304 -> 128
    # projections, and at one row on the 2304 -> 201024 LM head.
    out_name = str(out_dtype).removeprefix("torch.")
    return compile_projection(
        device, "simt", 1, n, k, "bfloat16", "bfloat16", out_name, None
    )


def b12x_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """out[M, N] = x[M, K] @ weight[N, K]^T with FP32 accumulation: BF16 operands, a
    contiguous BF16 or FP32 out, no bias. Compiles once per (N, K, out dtype) on first
    use, so run it outside CUDA graph capture first."""
    _b12x_projection(x.device.index, *weight.shape, out.dtype)(x, weight, out)
    return out


@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def use_sm12x_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    """Whether sm12x_bf16_gemv takes these operands: BF16 x[..., K] of at most MAX_M
    rows with unit last stride, a BF16 weight[N, K] with unit column stride and
    N >= MIN_N on x's device, and an optional unit-stride BF16 bias[N]."""
    return (
        x.is_cuda
        and x.ndim >= 2
        and weight.dim() == 2
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.stride(-1) == weight.stride(1) == 1
        and weight.shape[0] >= MIN_N
        and 0 < x.shape[-1] == weight.shape[1]
        and x.numel() // x.shape[-1] <= MAX_M
        and (
            bias is None
            or (
                bias.shape == (weight.shape[0],)
                and bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
    )


def sm12x_bf16_gemv(x, weight, bias=None, out_dtype=None, out=None):
    """x[..., K] @ weight[N, K]^T (+ bias) with FP32 accumulation, in out_dtype (BF16
    by default) or into out, an [M, N] view with unit column stride."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    assert m <= MAX_M, f"one program covers at most {MAX_M} rows, got {m}"
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    if m:
        _gemv_kernel[(triton.cdiv(n, _BN),)](
            x2,
            weight,
            bias,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            HAS_BIAS=bias is not None,
            BN=_BN,
            BK=_BK,
            MP=16 if m <= 16 else MAX_M,
            num_warps=4,
        )
    return out.view(*lead, n)


@triton.jit
def _gemv_3src_kernel(
    Xq,
    Xk,
    Xv,
    W,
    B,
    Y,
    M,
    N,
    K,
    SXq,
    SXk,
    SXv,
    SW,
    SY,
    K_LO,
    V_LO,
    R_LO,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0).to(tl.int64)
    col0 = pid * BN
    cols = col0 + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    rows_i64 = rows[:, None].to(tl.int64)
    use_k = (col0 >= K_LO) & (col0 < V_LO)
    use_v = (col0 >= V_LO) & (col0 < R_LO)
    X = tl.where(use_k, Xk, tl.where(use_v, Xv, Xq))
    SX = tl.where(use_k, SXk, tl.where(use_v, SXv, SXq))
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :] * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows_i64 * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows_i64 * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def _operands_require_grad(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    return torch.is_grad_enabled() and (
        xq.requires_grad
        or xk.requires_grad
        or xv.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )


def use_sm12x_bf16_gemv_3src(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    weight: torch.Tensor,
    k_lo: int,
    v_lo: int,
    r_lo: int,
    bias: torch.Tensor | None = None,
) -> bool:
    return (
        current_arch().is_sm(12)
        and xq.ndim == 2
        and use_sm12x_bf16_gemv(xq, weight, bias)
        and xk.shape == xq.shape
        and xv.shape == xq.shape
        and xk.dtype == xv.dtype == xq.dtype
        and xk.device == xv.device == xq.device
        and xk.stride(-1) == xv.stride(-1) == 1
        and not _operands_require_grad(xq, xk, xv, weight, bias)
        and 0 <= k_lo <= v_lo <= r_lo <= weight.shape[0]
        and k_lo % _BN == 0
        and v_lo % _BN == 0
        and r_lo % _BN == 0
    )


def sm12x_bf16_gemv_3src(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    weight: torch.Tensor,
    k_lo: int,
    v_lo: int,
    r_lo: int,
    bias: torch.Tensor | None = None,
    *,
    use_pdl: bool = False,
) -> torch.Tensor:
    """Project three 2D BF16 sources into one [M, N] output by column band:
    [k_lo, v_lo) reads xk, [v_lo, r_lo) reads xv, and other columns read xq.
    Bounds must satisfy 0 <= k_lo <= v_lo <= r_lo <= N and align to _BN.
    With use_pdl=True, PDL successors must call gdc_wait / griddepcontrol.wait before reading."""
    arch = current_arch()
    if not arch.is_sm(12):
        raise RuntimeError("sm12x_bf16_gemv_3src requires NVIDIA SM12x")
    if _operands_require_grad(xq, xk, xv, weight, bias):
        raise ValueError("sm12x_bf16_gemv_3src does not support autograd")
    if xq.ndim != 2 or xk.shape != xq.shape or xv.shape != xq.shape:
        raise ValueError(
            f"expected matching 2D sources, got xq={tuple(xq.shape)}, "
            f"xk={tuple(xk.shape)}, xv={tuple(xv.shape)}"
        )
    if weight.dim() != 2 or weight.shape[1] != xq.shape[1] or xq.shape[1] <= 0:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match input K={xq.shape[1]}"
        )
    if (
        not xq.is_cuda
        or xk.device != xq.device
        or xv.device != xq.device
        or weight.device != xq.device
        or xq.dtype != torch.bfloat16
        or xk.dtype != torch.bfloat16
        or xv.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            "xq, xk, xv, and weight must be BF16 tensors on the same CUDA device"
        )
    m, k = xq.shape
    if m > MAX_M:
        raise ValueError(f"one program covers at most {MAX_M} rows, got {m}")
    n = weight.shape[0]
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype != torch.bfloat16
        or bias.device != xq.device
        or bias.stride(0) != 1
    ):
        raise ValueError(f"bias must be contiguous BF16 of shape ({n},) on {xq.device}")
    if (
        not (0 <= k_lo <= v_lo <= r_lo <= n)
        or k_lo % _BN != 0
        or v_lo % _BN != 0
        or r_lo % _BN != 0
    ):
        raise ValueError(
            f"boundaries must satisfy 0 <= k_lo <= v_lo <= r_lo <= {n} and be "
            f"multiples of {_BN}, got ({k_lo}, {v_lo}, {r_lo})"
        )
    if (
        xq.stride(-1) != 1
        or xk.stride(-1) != 1
        or xv.stride(-1) != 1
        or weight.stride(1) != 1
    ):
        raise ValueError("xq, xk, xv, and weight must have unit last stride")
    out = torch.empty((m, n), device=xq.device, dtype=xq.dtype)
    if m and n:
        pdl = use_pdl and arch.supports_pdl
        with device_guard(xq):
            _gemv_3src_kernel[(triton.cdiv(n, _BN),)](
                xq,
                xk,
                xv,
                weight,
                bias,
                out,
                m,
                n,
                k,
                xq.stride(0),
                xk.stride(0),
                xv.stride(0),
                weight.stride(0),
                out.stride(0),
                k_lo,
                v_lo,
                r_lo,
                HAS_BIAS=bias is not None,
                BN=_BN,
                BK=_BK,
                MP=16 if m <= 16 else MAX_M,
                USE_PDL=pdl,
                num_warps=4,
                **({"launch_pdl": True} if pdl else {}),
            )
    return out
