"""Cake per-token NVFP4 GEMM (``mm_fp4(backend="cake")``) and its prepared runners.

FlashInfer entries (all SM100a / SM103a, compute capability 10.0 / 10.3):

* ``flashinfer.gemm.mm_fp4(a, b, a_descale, b_descale, alpha, out_dtype, out,
  block_size=16, use_8x4_sf_layout=False, backend="cake", use_nvfp4=True,
  enable_pdl=True)`` -> ``flashinfer.experimental.cake_nvfp4_per_token.cake_backend:
  mm_fp4_per_token``; admission ``...cake_nvfp4_per_token.support:cake_mm_fp4_requirement``.
  ``"cake"`` is explicit-only (never chosen by ``backend="auto"``).
* ``cake_backend.prepare_mm_fp4_per_token(a_fp4, a_sf, b_fp4, b_sf, alpha, out,
  *, tactic=None) -> NVFP4PerTokenGemmRunner`` (``launch()`` allocation-free,
  CUDA-graph capturable; prepare outside capture).
* ``cake_backend.prepare_nvfp4_per_token_chain(x, global_scale_inv, b_fp4, b_sf,
  out, workspace, *, out_scale=None) -> NVFP4PerTokenChainRunner`` (per-token
  quantizer + GEMM; ``workspace`` from ``allocate_nvfp4_per_token_quantize_outputs``).

Contract at FlashInfer ``46340689a5ab``: per-token-alpha path only. ``alpha``
FP32 ``[M]`` (scalar alpha rejected); ``a`` contiguous packed E2M1 uint8
``[M, K/2]``; ``b`` the column-major ``[K/2, N]`` view of a contiguous ``[N, K/2]``
weight (``b_fp4.T``, stride ``(1, K/2)``); 128x4 swizzled E4M3 block scales on
both sides (``b_descale = b_sf.T``), ``block_size == 16`` (NVFP4 only),
``N % 8 == 0``, ``K % 256 == 0``; contiguous BF16 / FP16 output; ``enable_pdl``
must stay ``True``. The host dispatch (``default_tactic``) depends on the SM
count (148 on B200, 152 on GB300) and only kernel keys of the validated matrix
are registered: ``(K, N)`` in {(7168,2112), (7168,1536), (16384,7168),
(7168,18432), (18432,7168), (8192,8192), (8192,28672), (28672,8192)} x
``M`` in {1, 8, 17, 32, 128, 130, 257, 512, 2048, 8192} plus ragged
``M`` in {3, 1000, 4097} on two families; other shapes raise
``NotImplementedError`` naming the missing kernel. ``supports_*`` resolves the
plan and reports ``False`` for unregistered kernels.

Not supported here (keep the existing SGLang path): scalar alpha, MXFP4
(``block_size 32``), 8x4 scale layout, ``enable_pdl=False``, SM90 / SM12x, FP32
output, weights that are not the transposed view of a contiguous ``[N, K/2]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.experimental.cake_nvfp4_per_token.cake_backend"
FI_SUPPORT_MODULE = "flashinfer.experimental.cake_nvfp4_per_token.support"
FI_JIT_MODULE = "flashinfer.experimental.cake_nvfp4_per_token.cake_jit"
FI_GEMM_MODULE = "flashinfer.gemm.gemm_base"
ARCHS = BLACKWELL_DATACENTER
K_TILE = 256
SF_VEC = 16
ROW_TILE = 128
ARCH_NAMES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


def _padded_rows(m: int) -> int:
    return (m + ROW_TILE - 1) // ROW_TILE * ROW_TILE


def _padded_sf_cols(k: int) -> int:
    # Swizzled 128x4 layout: K/16 scale columns rounded up to a multiple of 4.
    return (k // SF_VEC + 3) // 4 * 4


def _gemm_kernel_registered(
    device: torch.device, m: int, n: int, k: int, out_f16: bool
) -> bool:
    """Resolve FlashInfer's plan and check its kernel key is registered; never raises."""
    try:
        import torch
        from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cb
        from flashinfer.experimental.cake_nvfp4_per_token.cake_jit import (
            kernel_module_name,
        )

        arch = ARCH_NAMES[tuple(torch.cuda.get_device_capability(device))]
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
        kernel_module_name(arch, plan.kernel_key)
        return True
    except Exception:
        return False


def _quant_kernel_registered(
    device: torch.device, m: int, k: int, x_bf16: bool, fold: bool
) -> bool:
    try:
        import torch
        from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cb
        from flashinfer.experimental.cake_nvfp4_per_token.cake_jit import (
            kernel_module_name,
        )

        arch = ARCH_NAMES[tuple(torch.cuda.get_device_capability(device))]
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        plan = cb.quant_plan(m, k, x_bf16, fold, arch, sm_count)
        kernel_module_name(arch, plan.kernel_key)
        return True
    except Exception:
        return False


def _is_fp4_bytes(t: torch.Tensor) -> bool:
    import torch

    if t.dtype == torch.uint8:
        return True
    native = getattr(torch, "float4_e2m1fn_x2", None)
    return native is not None and t.dtype == native


def _scales_ok(sf: torch.Tensor, rows: int, k: int) -> bool:
    import torch

    return (
        sf.dtype in (torch.uint8, torch.float8_e4m3fn)
        and sf.is_contiguous()
        and sf.numel() == _padded_rows(rows) * _padded_sf_cols(k)
    )


def supports_mm_fp4_per_token(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    *,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    use_nvfp4: bool = True,
    enable_pdl: bool = True,
) -> bool:
    """Admission check mirroring ``cake_mm_fp4_requirement``; never raises."""
    import torch

    if not (
        flashinfer_module_available(
            FI_MODULE, FI_SUPPORT_MODULE, FI_JIT_MODULE, FI_GEMM_MODULE
        )
        and cuda_tensor_on(a, ARCHS)
        and a.ndim == 2
        and b.ndim == 2
        and _is_fp4_bytes(a)
        and _is_fp4_bytes(b)
        and a.is_contiguous()
        and int(a.shape[1]) == int(b.shape[0])
        and b.device == a.device
        and tuple(b.stride()) == (1, int(b.shape[0]))
    ):
        return False
    m, kh = (int(v) for v in a.shape)
    n = int(b.shape[1])
    k = 2 * kh
    if (
        alpha is None
        or alpha.device != a.device
        or alpha.dtype != torch.float32
        or alpha.ndim != 1
        or alpha.numel() != m
    ):
        return False
    if not (use_nvfp4 and block_size == 16 and not use_8x4_sf_layout and enable_pdl):
        return False
    if out_dtype not in (torch.bfloat16, torch.float16):
        return False
    if n % 8 or k % K_TILE or m < 1 or n < 1:
        return False
    # a_descale is the swizzled [padded M, K/16] scale tile; b_descale = b_sf.T.
    if a_descale.device != a.device or not _scales_ok(a_descale, m, k):
        return False
    b_sf = b_descale.t() if b_descale.ndim == 2 else b_descale
    if b_descale.device != a.device or not _scales_ok(b_sf, n, k):
        return False
    if out is not None and not (
        out.device == a.device
        and out.dtype == out_dtype
        and tuple(out.shape) == (m, n)
        and out.is_contiguous()
    ):
        return False
    return _gemm_kernel_registered(a.device, m, n, k, out_dtype == torch.float16)


def supports_prepare_mm_fp4_per_token(
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Admission check for the prepared GEMM runner (``[N, K/2]`` weight); never raises."""
    import torch

    if not (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(a_fp4, ARCHS)
        and a_fp4.ndim == 2
        and a_fp4.dtype == torch.uint8
        and a_fp4.is_contiguous()
        and b_fp4.ndim == 2
        and b_fp4.dtype == torch.uint8
        and b_fp4.is_contiguous()
        and int(a_fp4.shape[1]) == int(b_fp4.shape[1])
    ):
        return False
    m, kh = (int(v) for v in a_fp4.shape)
    n = int(b_fp4.shape[0])
    k = 2 * kh
    device = a_fp4.device
    return (
        all(t.device == device for t in (a_sf, b_fp4, b_sf, alpha, out))
        and m >= 1
        and n >= 1
        and k % K_TILE == 0
        and out.dtype in (torch.bfloat16, torch.float16)
        and tuple(out.shape) == (m, n)
        and out.is_contiguous()
        and alpha.dtype == torch.float32
        and alpha.numel() == m
        and alpha.is_contiguous()
        and _scales_ok(a_sf, m, k)
        and _scales_ok(b_sf, n, k)
        and _gemm_kernel_registered(device, m, n, k, out.dtype == torch.float16)
    )


def supports_prepare_nvfp4_per_token_chain(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    out: torch.Tensor,
    workspace: Any,
    *,
    out_scale: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check for the quantizer + GEMM chain; never raises.

    ``workspace`` is a ``PerTokenQuantizeOutputs`` (``fp4``, ``sf``, ``scale``)
    from :func:`allocate_nvfp4_per_token_quantize_outputs` for ``x``.
    """
    import torch

    try:
        fp4, sf, scale = workspace
    except (TypeError, ValueError):
        return False
    if not (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(x, ARCHS)
        and x.ndim == 2
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float16)
    ):
        return False
    m, k = (int(v) for v in x.shape)
    if k % SF_VEC:
        return False
    device = x.device
    if not (
        all(t.device == device for t in (fp4, sf, scale, global_scale_inv))
        and tuple(fp4.shape) == (m, k // 2)
        and fp4.dtype == torch.uint8
        and fp4.is_contiguous()
        and _scales_ok(sf, m, k)
        and tuple(scale.shape) == (m,)
        and scale.dtype == torch.float32
        and scale.is_contiguous()
        and global_scale_inv.dtype == torch.float32
        and global_scale_inv.numel() == 1
    ):
        return False
    if out_scale is not None and not (
        out_scale.device == device
        and out_scale.dtype == torch.float32
        and out_scale.numel() == 1
    ):
        return False
    if not _quant_kernel_registered(
        device, m, k, x.dtype == torch.bfloat16, out_scale is not None
    ):
        return False
    return supports_prepare_mm_fp4_per_token(fp4, sf, b_fp4, b_sf, scale, out)


def mm_fp4_per_token(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    use_nvfp4: bool = True,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Forward to ``flashinfer.gemm.mm_fp4(..., backend="cake")``; returns ``out``.

    Prepares and launches on every call (the generated program is loaded once).
    Use :func:`prepare_mm_fp4_per_token` for repeated launches / CUDA graphs.
    """
    import torch
    from flashinfer.gemm import mm_fp4

    if out_dtype is None:
        out_dtype = torch.bfloat16
    return mm_fp4(
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_8x4_sf_layout,
        backend="cake",
        use_nvfp4=use_nvfp4,
        enable_pdl=enable_pdl,
    )


def prepare_mm_fp4_per_token(
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    *,
    tactic: Optional[dict] = None,
) -> Any:
    """Forward to FlashInfer; returns an ``NVFP4PerTokenGemmRunner``.

    ``b_fp4`` is the contiguous ``[N, K/2]`` weight (not its transpose).
    ``launch()`` is allocation-free and CUDA-graph capturable; prepare outside
    capture (the JIT module is built and loaded here).
    """
    from flashinfer.experimental.cake_nvfp4_per_token.cake_backend import (
        prepare_mm_fp4_per_token as prepare,
    )

    return prepare(a_fp4, a_sf, b_fp4, b_sf, alpha, out, tactic=tactic)


def allocate_nvfp4_per_token_quantize_outputs(
    m: int, k: int, device: torch.device
) -> Any:
    """Forward to FlashInfer; returns ``PerTokenQuantizeOutputs(fp4, sf, scale)``.

    Workspace of the chain runner for an ``x[M, K]`` activation (no launch).
    """
    from flashinfer.experimental.cake_nvfp4_per_token.cake_backend import (
        allocate_nvfp4_per_token_quantize_outputs as allocate,
    )

    return allocate(m, k, device)


def prepare_nvfp4_per_token_chain(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    out: torch.Tensor,
    workspace: Any,
    *,
    out_scale: Optional[torch.Tensor] = None,
) -> Any:
    """Forward to FlashInfer; returns an ``NVFP4PerTokenChainRunner``.

    One ``launch()`` runs the per-token quantizer and the GEMM into the
    caller-owned workspace and ``out`` (allocation-free, CUDA-graph capturable;
    ``x`` is re-read on device at every launch). Prepare outside capture.
    """
    from flashinfer.experimental.cake_nvfp4_per_token.cake_backend import (
        prepare_nvfp4_per_token_chain as prepare,
    )

    return prepare(
        x, global_scale_inv, b_fp4, b_sf, out, workspace, out_scale=out_scale
    )


def get_nvfp4_per_token_gemm_runner_class() -> type:
    from flashinfer.experimental.cake_nvfp4_per_token.cake_backend import (
        NVFP4PerTokenGemmRunner,
    )

    return NVFP4PerTokenGemmRunner


def get_nvfp4_per_token_chain_runner_class() -> type:
    from flashinfer.experimental.cake_nvfp4_per_token.cake_backend import (
        NVFP4PerTokenChainRunner,
    )

    return NVFP4PerTokenChainRunner


def get_per_token_quantize_outputs_class() -> type:
    from flashinfer.experimental.cake_nvfp4_per_token.cake_backend import (
        PerTokenQuantizeOutputs,
    )

    return PerTokenQuantizeOutputs
