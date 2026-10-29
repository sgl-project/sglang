"""Cake SVDQuant NVFP4 GEMM (``mm_nvfp4_svdquant(backend="cake")``) via FlashInfer.

FlashInfer entry: ``flashinfer.gemm.mm_nvfp4_svdquant(a, b, a_sf, b_sf, alpha, d,
l1, bias=None, out=None, backend="cake", enable_pdl=None)``
(``flashinfer/gemm/gemm_svdquant.py``; admission ``_cake_nvfp4_svdquant_requirement``;
routes from ``flashinfer.jit.cake_nvfp4_svdquant`` with per-arch catalogs
``cake_nvfp4_svdquant_sm10{0,3}a_catalog``). ``"cake"`` is explicit-only; the
heuristic strips it from the ``"auto"`` candidates.

Contract at FlashInfer ``46340689a5ab`` (SM100a / SM103a, CUDA >= 13.0):
``out = alpha * (a @ b^T + d @ l1^T) [+ bias]`` with ``a`` ``(m, k/2)`` uint8 packed
E2M1 row-major, ``b`` ``(n, k/2)`` uint8, ``a_sf`` / ``b_sf`` uint8 128x4-swizzled
UE4M3 block scales (``numel >= pad128(rows) * pad4(k/16)``), ``alpha`` FP32 (first
element used), ``d`` ``(m, r)`` BF16 contiguous (16-byte aligned, TMA), ``l1``
``(n, r)`` BF16 pre-divided by alpha, ``bias`` ``(n,)`` BF16, ``out`` ``(m, n)``
BF16 contiguous; ``n % 32 == 0``, ``k % 32 == 0``, rank ``r`` a positive multiple
of 32. Only catalogued ``(M, N, K, rank, bias)`` routes are served (N = 3072 with
K in {3072, 12288}, M in {129, 4096, 6889, 6912, 9216, 16384}, rank in {32, 64,
96, 128}; plus (M, N, K) = (4096 | 16384, 12288, 3072) rank 32 with bias);
``is_cake_nvfp4_svdquant_problem_supported`` must find a unique route. The TMA
descriptor workspace is a FlashInfer-cached buffer (128-byte aligned);
``enable_pdl=None`` keeps the route's compiled PDL mode.

Not supported here (keep the existing SGLang path): uncatalogued shapes, CUDA
12.x, SM12x (``cute-dsl`` backend) and ``nvfp4_quantize_smooth`` /
``svdquant_linear`` (no Cake branch at this commit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.gemm.gemm_svdquant"
FI_JIT_MODULE = "flashinfer.jit.cake_nvfp4_svdquant"
ARCHS = BLACKWELL_DATACENTER
RANK_GRANULARITY = 32
MIN_CUDA_MAJOR = 13


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _swizzled_sf_size(rows: int, sf_cols: int) -> int:
    return _pad_up(rows, 128) * _pad_up(sf_cols, 4)


def _cuda_13_or_newer() -> bool:
    import torch

    version = torch.version.cuda
    if not version:
        return False
    try:
        return int(version.split(".")[0]) >= MIN_CUDA_MAJOR
    except ValueError:
        return False


def _route_supported(
    device: torch.device, m: int, n: int, k: int, rank: int, has_bias: bool
) -> bool:
    try:
        from flashinfer.jit.cake_nvfp4_svdquant import (
            is_cake_nvfp4_svdquant_problem_supported,
        )

        return bool(
            is_cake_nvfp4_svdquant_problem_supported(
                m=m, n=n, k=k, rank=rank, has_bias=has_bias, device=device
            )
        )
    except Exception:
        return False


def supports_mm_nvfp4_svdquant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer common + Cake requirements; never raises."""
    import torch

    if not (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and _cuda_13_or_newer()
        and cuda_tensor_on(a, ARCHS)
        and a.ndim == 2
        and b.ndim == 2
        and a.dtype == torch.uint8
        and b.dtype == torch.uint8
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        return False
    m, k_packed = (int(v) for v in a.shape)
    n = int(b.shape[0])
    if int(b.shape[1]) != k_packed:
        return False
    k = 2 * k_packed
    if n % 32 or k % 32:
        return False
    if d.ndim != 2 or int(d.shape[0]) != m:
        return False
    rank = int(d.shape[1])
    if rank < RANK_GRANULARITY or rank % RANK_GRANULARITY:
        return False
    if l1.ndim != 2 or tuple(l1.shape) != (n, rank):
        return False
    if (
        d.dtype != torch.bfloat16
        or l1.dtype != torch.bfloat16
        or not d.is_contiguous()
        or not l1.is_contiguous()
        or d.data_ptr() % 16
    ):
        return False
    if (
        a_sf.dtype != torch.uint8
        or b_sf.dtype != torch.uint8
        or a_sf.numel() < _swizzled_sf_size(m, k // 16)
        or b_sf.numel() < _swizzled_sf_size(n, k // 16)
        or not a_sf.is_contiguous()
        or not b_sf.is_contiguous()
    ):
        return False
    if alpha.dtype != torch.float32 or alpha.numel() < 1:
        return False
    if bias is not None and (tuple(bias.shape) != (n,) or bias.dtype != torch.bfloat16):
        return False
    if out is not None and (
        tuple(out.shape) != (m, n)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        return False
    tensors = (b, a_sf, b_sf, alpha, d, l1, bias, out)
    if any(t is not None and t.device != a.device for t in tensors):
        return False
    return _route_supported(a.device, m, n, k, rank, bias is not None)


def mm_nvfp4_svdquant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.gemm.mm_nvfp4_svdquant(..., backend="cake")``; returns ``out``."""
    from flashinfer.gemm import mm_nvfp4_svdquant as fi_mm_nvfp4_svdquant

    return fi_mm_nvfp4_svdquant(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
        d,
        l1,
        bias=bias,
        out=out,
        backend="cake",
        enable_pdl=enable_pdl,
    )
