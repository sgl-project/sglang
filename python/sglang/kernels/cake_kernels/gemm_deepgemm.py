"""DeepGEMM-family Cake GEMMs (prepared plans, exported per-route) via FlashInfer.

Five prepared-plan factories, each returning a plan whose ``run()`` submits the
exported program on the current PyTorch stream with no allocation (CUDA-graph
replay tested upstream; preparation happens outside capture):

* ``flashinfer.fp8_batched_gemm.prepare_fp8_batched_gemm(a, b, *, output_fp8=True,
  alpha=None, out=None, output_scales=None, descriptor_workspace=None)`` ->
  ``BatchedGemmPlan`` (``experimental.deepgemm_batched_gemm.batched_gemm``):
  ``A[T,H,K] @ B[H,N,K] -> [T,H,N]`` with ``a = (E4M3 [T,H,K], FP32 scales
  [T,H,K/128])``, ``b = (E4M3 [H,N,K], FP32 scales [H,N/128,K/128])``; scales are
  positive powers of two and are *packed at prepare time* (re-prepare when scale
  values change). Exported only for ``H=8, K=4096, N=1024, num_stages=5``: FP8
  output for ``T in {1,4,16,128,512,4096}``, BF16 and alpha for ``T in {4,128}``.
  FP8 output = (E4M3 ``[T,H,N]``, int32 UE8M0 scale words ``[T, H*N/128]``
  column-major with ``T`` padded to 4). ``plan.run()`` returns ``plan.output``.
* ``flashinfer.fp4_gemm.prepare_fp4_gemm(a, b, a_scales, b_scales, *, m, alpha=1.0,
  out=None, num_stages=None, block_n=128, epilogue_store_n=32,
  descriptor_workspace=None)`` -> ``Fp4GemmPlan``: packed E2M1 ``A [storage_M, K/2]``
  / ``B [N, K/2]`` (int8 / uint8), UE8M0 scales ``int32/uint32 [K/128, storage_MN]``
  (four granularity-32 bytes per word), BF16 out ``[physical_M, N]`` with
  ``plan.output`` the ``[m, N]`` view. Route key ``M:N:K:num_sms:num_stages:
  block_n:epilogue_store_n``; exported ``M in {16,128,512,4096}`` x ``(N,K) in
  {(4608,5120),(5120,2304)}`` plus smoke ``256x128x256`` and ``256x128x2048``
  (``num_stages=7``).
* ``flashinfer.experimental.deepgemm_fp8_gemm.prepare_fp8_gemm_1d1d(a, b, sfa, sfb,
  out, *, accumulate=False, cache_dir)`` -> ``Fp8GemmPlan``: single
  specialization ``M4096 / N7168 / K4096``; ``a`` uint8 ``[4096,4096]``, ``b`` uint8
  ``[7168,4096]`` (E4M3 bytes), ``sfa`` uint32 ``[8,4096]``, ``sfb`` uint32
  ``[8,7168]`` (four K128 UE8M0 bytes per word, MN-major); ``forward`` writes BF16
  ``out``; ``accumulate=True`` does ``D += A @ B^T`` into FP32 ``out`` in place.
  PTX source export: needs ``ptxas`` and TVM-FFI ``embed_cubin``; assembled at
  first use into ``cache_dir`` (no top-level ``flashinfer.*`` wrapper).
* ``flashinfer.fp4_k_grouped_gemm.prepare_fp4_k_grouped_gemm(a, b, a_scales,
  b_scales, *, m, group_ks, k_alignment=256, use_psum_layout=True,
  output_dtype="bf16", accumulate=False, num_stages=7, out=None,
  grouped_layout=None, descriptor_workspace=None)`` -> ``GroupedFP4Plan``:
  independently reduced ``A_g @ B_g.T`` per K group; ``A [ceil(m/256)*256,
  sum(padded_K)/2]``, ``B [N, sum(padded_K)/2]`` packed E2M1 (zero padding),
  scales ``[sum(padded_K)/128, physical_M]`` / ``[.., N]``; ``N % 128 == 0``;
  ``out [groups, physical_M, N]`` bf16 / fp32 (``accumulate`` requires fp32 and an
  initialized caller-owned ``out``; each ``run()`` adds once). Empty groups allowed.
* ``flashinfer.fp8_fp4_gemm.prepare_fp8_fp4_gemm(a, b, a_scales, b_scales, *, m,
  out=None, block_n=128, gran_k_a=32, variant=None, descriptor_workspace=None)``
  -> ``MixedGemmPlan``: ``A [storage_M, K]`` E4M3 / uint8 x ``B [N, K/2]`` packed
  E2M1 -> BF16 ``[physical_M, N]`` (``plan.output`` is ``[m, N]``); route key
  ``M:N:K:num_sms:variant:block_n:gran_k_a``; exported ``M in {1,16,128,512,4096}``
  x ``(N,K) in {(4608,5120),(5120,2304)}``, ``256x256x256`` smoke,
  ``4096x7168x4096`` (``block_n=224, gran_k_a=128``), ``variant="bk128_s6"`` at
  ``256x224x128`` / ``4096x7168x4096`` and ``variant="bk256_s4"`` at ``256x128x256``.

All five are exact-program exports for SM100a with **148 SMs** (B200) and
SM103a with **152 SMs** (GB300): the device compute capability selects the
catalog section and the ``multi_processor_count`` is part of the route key, so
other SKUs of the same architecture are refused (``NotImplementedError`` /
``ValueError``). The ``supports_*`` functions below resolve the route key against
FlashInfer's catalog and report ``False`` when no exported route exists. The
Cake provenance of this family is inferred from the exporter template and the
``deepgemm_fp8_gemm`` README ("exported as Cake-generated PTX source").

Not supported here (keep the existing SGLang path): any shape / option outside
the exported catalog, devices with other SM counts, SM90 / SM12x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    device_capability,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_BATCHED_MODULE = "flashinfer.fp8_batched_gemm"
FI_BATCHED_RUNTIME = "flashinfer.experimental.deepgemm_batched_gemm.batched_gemm"
FI_FP4_MODULE = "flashinfer.fp4_gemm"
FI_FP4_RUNTIME = "flashinfer.experimental.deepgemm_fp4_gemm.fp4_gemm"
FI_FP8_1D1D_MODULE = "flashinfer.experimental.deepgemm_fp8_gemm"
FI_FP8_1D1D_RUNTIME = "flashinfer.experimental.deepgemm_fp8_gemm.runtime"
FI_KGROUP_MODULE = "flashinfer.fp4_k_grouped_gemm"
FI_KGROUP_RUNTIME = "flashinfer.experimental.deepgemm_kgroup_gemm.kgroup_gemm"
FI_MIXED_MODULE = "flashinfer.fp8_fp4_gemm"
FI_MIXED_RUNTIME = "flashinfer.experimental.deepgemm_mixed_gemm.mixed_gemm"
ARCHS = BLACKWELL_DATACENTER
# Exported SM counts per architecture (route keys carry ``num_sms``).
EXPORTED_SM_COUNTS = {(10, 0): 148, (10, 3): 152}


def _sm_count(device: torch.device) -> int:
    import torch

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _device_exported(device: torch.device) -> bool:
    cc = device_capability(device.index)
    return EXPORTED_SM_COUNTS.get(cc) == _sm_count(device)


def _route_exists(runtime_module: str, device: torch.device, options: dict) -> bool:
    """Resolve ``options`` against FlashInfer's exported catalog; never raises."""
    import importlib

    try:
        runtime = importlib.import_module(runtime_module)
        arch = runtime.device_arch(device)
        routes = runtime._catalog()["arches"][arch]["routes"]
        return runtime.route_key(options) in routes
    except Exception:
        return False


# ---------------------------------------------------------------------------
# prepare_fp8_batched_gemm
# ---------------------------------------------------------------------------


def supports_fp8_batched_gemm(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    *,
    output_fp8: bool = True,
    alpha: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    output_scales: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring ``BatchedGemmPlan.__init__``; never raises."""
    import torch

    try:
        aq, asf = a
        bq, bsf = b
    except (TypeError, ValueError):
        return False
    if not (
        flashinfer_module_available(FI_BATCHED_MODULE, FI_BATCHED_RUNTIME)
        and cuda_tensor_on(aq, ARCHS)
        and _device_exported(aq.device)
        and aq.ndim == 3
        and bq.ndim == 3
        and aq.dtype == torch.float8_e4m3fn
        and bq.dtype == torch.float8_e4m3fn
    ):
        return False
    tokens, heads, inner = (int(v) for v in aq.shape)
    width = int(bq.shape[1])
    if (
        int(bq.shape[0]) != heads
        or int(bq.shape[2]) != inner
        or tokens < 1
        or width % 128
        or inner % 512
    ):
        return False
    if (
        asf.dtype != torch.float32
        or bsf.dtype != torch.float32
        or tuple(asf.shape) != (tokens, heads, inner // 128)
        or tuple(bsf.shape) != (heads, width // 128, inner // 128)
    ):
        return False
    if (
        any(t.device != aq.device for t in (bq, asf, bsf))
        or not aq.is_contiguous()
        or not bq.is_contiguous()
    ):
        return False
    if output_fp8 and alpha is not None:
        return False
    if not output_fp8 and output_scales is not None:
        return False
    dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
    if out is not None and not (
        out.dtype == dtype
        and tuple(out.shape) == (tokens, heads, width)
        and out.is_contiguous()
        and out.device == aq.device
    ):
        return False
    if output_fp8 and output_scales is not None:
        shape = (tokens, heads * width // 128)
        stride = (1, (tokens + 3) // 4 * 4)
        if not (
            output_scales.dtype == torch.int32
            and output_scales.device == aq.device
            and tuple(output_scales.shape) == shape
            and tuple(output_scales.stride()) == stride
        ):
            return False
    epilogue = "fp8" if output_fp8 else "alpha" if alpha is not None else "bf16"
    options = dict(
        tokens=tokens,
        num_heads=heads,
        inner=inner,
        width=width,
        num_sms=_sm_count(aq.device),
        num_stages=5,
        epilogue=epilogue,
    )
    return _route_exists(FI_BATCHED_RUNTIME, aq.device, options)


def prepare_fp8_batched_gemm(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    *,
    output_fp8: bool = True,
    alpha: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    output_scales: Optional[torch.Tensor] = None,
    descriptor_workspace: Optional[torch.Tensor] = None,
) -> Any:
    """Forward to FlashInfer; returns a ``BatchedGemmPlan`` (``run()`` -> output)."""
    from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm as prepare

    return prepare(
        a,
        b,
        output_fp8=output_fp8,
        alpha=alpha,
        out=out,
        output_scales=output_scales,
        descriptor_workspace=descriptor_workspace,
    )


# ---------------------------------------------------------------------------
# prepare_fp4_gemm
# ---------------------------------------------------------------------------


def supports_fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: int,
    num_stages: Optional[int] = None,
    block_n: int = 128,
    epilogue_store_n: int = 32,
) -> bool:
    """Admission check mirroring ``Fp4GemmPlan.__init__`` (route existence); never raises.

    The exported route also pins ``storage_M`` (``a.shape[0]``) and a declared
    alpha for some routes; those are validated by FlashInfer at prepare time.
    """
    import torch

    if not (
        flashinfer_module_available(FI_FP4_MODULE, FI_FP4_RUNTIME)
        and cuda_tensor_on(a, ARCHS)
        and _device_exported(a.device)
        and a.ndim == 2
        and b.ndim == 2
        and a.dtype in (torch.int8, torch.uint8)
        and b.dtype in (torch.int8, torch.uint8)
    ):
        return False
    n, k = int(b.shape[0]), int(b.shape[1]) * 2
    if int(a.shape[1]) * 2 != k:
        return False
    if a_scales.dtype not in (torch.int32, torch.uint32) or b_scales.dtype not in (
        torch.int32,
        torch.uint32,
    ):
        return False
    if tuple(a_scales.shape) != (k // 128, int(a.shape[0])) or tuple(
        b_scales.shape
    ) != (k // 128, n):
        return False
    tensors = (b, a_scales, b_scales)
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        return False
    options = dict(
        M=int(m),
        N=n,
        K=k,
        num_sms=_sm_count(a.device),
        num_stages=num_stages,
        block_n=block_n,
        epilogue_store_n=epilogue_store_n,
    )
    return a.is_contiguous() and _route_exists(FI_FP4_RUNTIME, a.device, options)


def prepare_fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: int,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
    num_stages: Optional[int] = None,
    block_n: int = 128,
    epilogue_store_n: int = 32,
    descriptor_workspace: Optional[torch.Tensor] = None,
) -> Any:
    """Forward to FlashInfer; returns an ``Fp4GemmPlan`` (``run()`` -> ``[m, N]`` view)."""
    from flashinfer.fp4_gemm import prepare_fp4_gemm as prepare

    return prepare(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        alpha=alpha,
        out=out,
        num_stages=num_stages,
        block_n=block_n,
        epilogue_store_n=epilogue_store_n,
        descriptor_workspace=descriptor_workspace,
    )


# ---------------------------------------------------------------------------
# prepare_fp8_gemm_1d1d
# ---------------------------------------------------------------------------

FP8_1D1D_M, FP8_1D1D_N, FP8_1D1D_K = 4096, 7168, 4096


def supports_fp8_gemm_1d1d(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    *,
    accumulate: bool = False,
) -> bool:
    """Admission check mirroring ``prepare_fp8_gemm_1d1d``; never raises.

    The availability of ``ptxas`` / TVM-FFI ``embed_cubin`` (needed to assemble
    the exported PTX at first use) is not probed here.
    """
    import torch

    if not (
        flashinfer_module_available(FI_FP8_1D1D_MODULE, FI_FP8_1D1D_RUNTIME)
        and cuda_tensor_on(out, ARCHS)
        and _device_exported(out.device)
    ):
        return False
    m, n, k = FP8_1D1D_M, FP8_1D1D_N, FP8_1D1D_K
    if (
        tuple(a.shape) != (m, k)
        or tuple(b.shape) != (n, k)
        or tuple(out.shape) != (m, n)
        or a.dtype != torch.uint8
        or b.dtype != torch.uint8
        or sfa.dtype != torch.uint32
        or sfb.dtype != torch.uint32
        or tuple(sfa.shape) != (k // 512, m)
        or tuple(sfb.shape) != (k // 512, n)
        or out.dtype != (torch.float32 if accumulate else torch.bfloat16)
        or any(t.device != out.device for t in (a, b, sfa, sfb))
    ):
        return False
    try:
        from flashinfer.experimental.deepgemm_fp8_gemm import runtime

        arch = runtime.device_arch(out.device)
        route = runtime._catalog()["arches"][arch]["routes"].get(
            "wgrad" if accumulate else "forward"
        )
        return route is not None and int(route["config"]["num_sms"]) == _sm_count(
            out.device
        )
    except Exception:
        return False


def prepare_fp8_gemm_1d1d(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    *,
    accumulate: bool = False,
    cache_dir: str,
) -> Any:
    """Forward to FlashInfer; returns an ``Fp8GemmPlan`` (``run()`` writes ``out``).

    ``cache_dir`` receives the assembled cubin and build receipt (first use runs
    ``ptxas``). Restore the FP32 initializer before each accumulated evaluation.
    """
    from flashinfer.experimental.deepgemm_fp8_gemm import (
        prepare_fp8_gemm_1d1d as prepare,
    )

    return prepare(a, b, sfa, sfb, out, accumulate=accumulate, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# prepare_fp4_k_grouped_gemm
# ---------------------------------------------------------------------------


def supports_fp4_k_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: int,
    group_ks: Sequence[int],
    k_alignment: int = 256,
    use_psum_layout: bool = True,
    output_dtype: str = "bf16",
    accumulate: bool = False,
    num_stages: int = 7,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring ``GroupedFP4Plan.__init__``; never raises.

    All-empty groups (``sum(group_ks) == 0``) need no exported route and are
    admitted when the device is exported.
    """
    import torch

    try:
        ks = tuple(int(k) for k in group_ks)
    except (TypeError, ValueError):
        return False
    if not (
        flashinfer_module_available(FI_KGROUP_MODULE, FI_KGROUP_RUNTIME)
        and cuda_tensor_on(a, ARCHS)
        and _device_exported(a.device)
    ):
        return False
    if m < 1 or not ks or any(k < 0 for k in ks):
        return False
    if k_alignment < 256 or k_alignment % 256:
        return False
    if output_dtype not in ("bf16", "fp32") or (accumulate and output_dtype != "fp32"):
        return False
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.dtype not in (torch.int8, torch.uint8)
        or b.dtype not in (torch.int8, torch.uint8)
    ):
        return False
    n = int(b.shape[0])
    if n < 1 or n % 128:
        return False
    physical_m = (int(m) + 255) // 256 * 256
    padded = [(k + k_alignment - 1) // k_alignment * k_alignment for k in ks]
    total_k = sum(padded)
    if tuple(a.shape) != (physical_m, total_k // 2) or tuple(b.shape) != (
        n,
        total_k // 2,
    ):
        return False
    for tensor, shape in (
        (a_scales, ((total_k + 127) // 128, physical_m)),
        (b_scales, ((total_k + 127) // 128, n)),
    ):
        if (
            tensor.dtype not in (torch.int32, torch.uint32)
            or tuple(tensor.shape) != shape
        ):
            return False
    dtype = torch.bfloat16 if output_dtype == "bf16" else torch.float32
    if out is None:
        if accumulate:
            return False
    elif out.dtype != dtype or tuple(out.shape) != (len(ks), physical_m, n):
        return False
    tensors = [a, b, a_scales, b_scales] + ([out] if out is not None else [])
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        return False
    if total_k == 0:
        return True
    options = dict(
        M=int(m),
        N=n,
        group_ks=ks,
        num_sms=_sm_count(a.device),
        k_alignment=k_alignment,
        use_psum_layout=use_psum_layout,
        output_dtype=output_dtype,
        accumulate=accumulate,
        num_stages=num_stages,
    )
    return _route_exists(FI_KGROUP_RUNTIME, a.device, options)


def prepare_fp4_k_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: int,
    group_ks: Sequence[int],
    k_alignment: int = 256,
    use_psum_layout: bool = True,
    output_dtype: str = "bf16",
    accumulate: bool = False,
    num_stages: int = 7,
    out: Optional[torch.Tensor] = None,
    grouped_layout: Optional[torch.Tensor] = None,
    descriptor_workspace: Optional[torch.Tensor] = None,
) -> Any:
    """Forward to FlashInfer; returns a ``GroupedFP4Plan`` (``run()`` -> ``plan.output``)."""
    from flashinfer.fp4_k_grouped_gemm import prepare_fp4_k_grouped_gemm as prepare

    return prepare(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        group_ks=group_ks,
        k_alignment=k_alignment,
        use_psum_layout=use_psum_layout,
        output_dtype=output_dtype,
        accumulate=accumulate,
        num_stages=num_stages,
        out=out,
        grouped_layout=grouped_layout,
        descriptor_workspace=descriptor_workspace,
    )


# ---------------------------------------------------------------------------
# prepare_fp8_fp4_gemm
# ---------------------------------------------------------------------------


def supports_fp8_fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: int,
    block_n: int = 128,
    gran_k_a: int = 32,
    variant: Optional[str] = None,
) -> bool:
    """Admission check mirroring ``MixedGemmPlan.__init__`` (route existence); never raises.

    The exported route also pins ``storage_M`` (``a.shape[0]``) and the packed
    scale geometry; FlashInfer validates those against the catalog at prepare.
    """
    import torch

    if not (
        flashinfer_module_available(FI_MIXED_MODULE, FI_MIXED_RUNTIME)
        and cuda_tensor_on(a, ARCHS)
        and _device_exported(a.device)
        and a.ndim == 2
        and b.ndim == 2
        and a.dtype in (torch.float8_e4m3fn, torch.uint8)
        and b.dtype in (torch.int8, torch.uint8)
    ):
        return False
    n, k = int(b.shape[0]), int(b.shape[1]) * 2
    if int(a.shape[1]) != k:
        return False
    if a_scales.dtype not in (torch.int32, torch.uint32) or b_scales.dtype not in (
        torch.int32,
        torch.uint32,
    ):
        return False
    tensors = (a, b, a_scales, b_scales)
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        return False
    options = dict(
        M=int(m),
        N=n,
        K=k,
        num_sms=_sm_count(a.device),
        block_n=block_n,
        gran_k_a=gran_k_a,
        variant=variant,
    )
    return _route_exists(FI_MIXED_RUNTIME, a.device, options)


def prepare_fp8_fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: int,
    out: Optional[torch.Tensor] = None,
    block_n: int = 128,
    gran_k_a: int = 32,
    variant: Optional[str] = None,
    descriptor_workspace: Optional[torch.Tensor] = None,
) -> Any:
    """Forward to FlashInfer; returns a ``MixedGemmPlan`` (``run()`` -> ``plan.output``)."""
    from flashinfer.fp8_fp4_gemm import prepare_fp8_fp4_gemm as prepare

    return prepare(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        out=out,
        block_n=block_n,
        gran_k_a=gran_k_a,
        variant=variant,
        descriptor_workspace=descriptor_workspace,
    )


def get_batched_gemm_plan_class() -> type:
    from flashinfer.experimental.deepgemm_batched_gemm.batched_gemm import (
        BatchedGemmPlan,
    )

    return BatchedGemmPlan


def get_fp4_gemm_plan_class() -> type:
    from flashinfer.experimental.deepgemm_fp4_gemm.fp4_gemm import Fp4GemmPlan

    return Fp4GemmPlan


def get_fp8_gemm_plan_class() -> type:
    from flashinfer.experimental.deepgemm_fp8_gemm.runtime import Fp8GemmPlan

    return Fp8GemmPlan


def get_grouped_fp4_plan_class() -> type:
    from flashinfer.experimental.deepgemm_kgroup_gemm.kgroup_gemm import (
        GroupedFP4Plan,
    )

    return GroupedFP4Plan


def get_mixed_gemm_plan_class() -> type:
    from flashinfer.experimental.deepgemm_mixed_gemm.mixed_gemm import MixedGemmPlan

    return MixedGemmPlan
