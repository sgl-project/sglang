"""Phase-3b IPM solver: cuBLASDx Matmul + cuSolverDx, all-dynamic-shmem layout.

Extends Phase 3 to arbitrary NC (up to Hopper's 223 KB dynamic-shmem cap) by
putting **all** Matmul tiles (A, B, C) and the pre-scaled / original A buffers
into dynamic shared memory via ``nvmath.device.make_tensor`` + layouts from
``Matmul.get_layout_smem_{a,b,c}()``.

Technique:
  1. Pre-scale A -> As where As[i,k] = A[i,k] * x[k].
  2. Alias the same As buffer as Matmul's A (row-major) and B (col-major) so
     Matmul computes ``ata = As @ As.T`` in-place.
  3. C output lands in col-major NC x NC directly compatible with
     cuSolverDx's CholeskySolver (same buffer passed to chol.factorize).
  4. Wrap each dynamic-shmem slice with ``make_tensor(slice, layout)`` using
     the layouts cuBLASDx reports.

Shmem layout::

  DYNAMIC (opt-in via cuKernelSetAttribute):
    buf_AB    Matmul A/B aliased   NC*NV elems
    buf_C     Matmul C = chol ata  NC*NC elems  (col-major)
    A_orig    unscaled A           NC*NV elems  (for GEMM 3)
    c_vec     objective            NV elems
    x_vec     state                NV elems
    d_vec     direction            NV elems

  STATIC (<= 48 KB, easy):
    rhs_s     cuSolverDx RHS       chol.b_size() elems
    info_s    1
    red_s     block_dim

Max NC is governed by ``shmem_budget.shmem_bytes`` (covered by the per-GPU
budget table).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from numba import cuda, float32, int32

from .shmem_budget import assert_fits, report, shmem_bytes

try:
    from nvmath.device import CholeskySolver, Matmul, make_tensor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "nvmath-python[cu12-dx] 0.9.0+ is required for ipm_cublasdx. "
        "Install via: pip install 'nvmath-python[cu12-dx]==0.9.0'"
    ) from e


_kernel_cache: dict[tuple[int, int, int, int], object] = {}
_spec_cache: dict[tuple[int, int, int, int], tuple[object, int, bool]] = {}


def _optin_dynamic_shmem(spec_dispatcher, need_bytes: int, device_ordinal: int) -> None:
    from cuda.bindings import driver as drv

    kern = next(iter(spec_dispatcher.overloads.values()))
    kern.bind()
    cufunc = kern._codelibrary.get_cufunc()
    attr = drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    (status,) = drv.cuKernelSetAttribute(attr, need_bytes, cufunc.handle, device_ordinal)
    if status != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"cuKernelSetAttribute failed: {status!r} (need={need_bytes} B)"
        )


def _choose_block_dim(nc: int) -> int:
    if nc <= 32:
        return 64
    if nc <= 64:
        return 128
    return 128


def _build_kernel(nc: int, nv: int, block_dim: int, num_iters: int):
    key = (nc, nv, block_dim, num_iters)
    if key in _kernel_cache:
        return _kernel_cache[key]

    mm = Matmul(
        size=(nc, nc, nv),
        precision=np.float32,
        data_type="real",
        arrangement=("row_major", "col_major", "col_major"),
        execution="Block",
        block_dim=(block_dim, 1, 1),
    )
    chol = CholeskySolver(
        size=(nc, nc, 1),
        precision=np.float32,
        data_type="real",
        execution="Block",
        fill_mode="lower",
        block_dim=(block_dim, 1, 1),
    )

    la_smem = mm.get_layout_smem_a()
    lb_smem = mm.get_layout_smem_b()
    lc_smem = mm.get_layout_smem_c()

    # Matmul's C layout must match cuSolverDx's ata layout (col-major, ldc=NC).
    assert lc_smem.cosize == chol.a_size(), (
        f"Matmul C cosize ({lc_smem.cosize}) != chol.a_size ({chol.a_size()}); "
        f"cannot alias C <- ata."
    )

    # Dynamic-shmem layout (fp32 element offsets)
    N_AB = la_smem.cosize   # aliased Matmul A/B (As buffer)
    N_C = lc_smem.cosize    # Matmul C = cuSolverDx ata
    N_ORIGA = nc * nv       # original A (for GEMM 3)
    N_CVEC = nv
    N_X = nv
    N_D = nv
    OFF_AB = 0
    OFF_C = OFF_AB + N_AB
    OFF_ORIGA = OFF_C + N_C
    OFF_CVEC = OFF_ORIGA + N_ORIGA
    OFF_X = OFF_CVEC + N_CVEC
    OFF_D = OFF_X + N_X
    TOTAL_DYN = OFF_D + N_D

    N_RHS = chol.b_size()
    chol_value_type = chol.value_type
    chol_info_type = chol.info_type
    mm_value_type = mm.a_value_type  # same as chol_value_type for real fp32

    @cuda.jit
    def ipm_kernel(A_g, b_g, c_g, x_out):
        tid = cuda.threadIdx.x

        # One dynamic blob — partition into typed views.
        smem = cuda.shared.array(shape=0, dtype=float32)
        buf_AB = smem[OFF_AB:OFF_AB + N_AB]
        buf_C = smem[OFF_C:OFF_C + N_C]
        A_orig = smem[OFF_ORIGA:OFF_ORIGA + N_ORIGA]
        c_s = smem[OFF_CVEC:OFF_CVEC + N_CVEC]
        x_s = smem[OFF_X:OFF_X + N_X]
        d_s = smem[OFF_D:OFF_D + N_D]

        # Small scratch stays static.
        rhs_s = cuda.shared.array(shape=N_RHS, dtype=chol_value_type)
        info_s = cuda.shared.array(shape=1, dtype=chol_info_type)
        red_s = cuda.shared.array(shape=block_dim, dtype=float32)

        # Load A, c; init x = 1.
        for idx in range(tid, N_ORIGA, block_dim):
            A_orig[idx] = A_g[idx // nv, idx % nv]
        for j in range(tid, nv, block_dim):
            c_s[j] = c_g[j]
            x_s[j] = float32(1.0)
        if tid == 0:
            info_s[0] = chol_info_type(0)
        cuda.syncthreads()

        d_max = float32(0.0)
        failed = False

        for _it in range(num_iters):
            # 1a: buf_AB[i,k] = A[i,k] * x[k]  (row-major, ld=NV, span=N_ORIGA)
            for idx in range(tid, N_ORIGA, block_dim):
                k = idx % nv
                buf_AB[idx] = A_orig[idx] * x_s[k]
            cuda.syncthreads()

            # 1b: ata = As @ As.T via cuBLASDx. Aliased A = B.
            ta = make_tensor(buf_AB, la_smem)
            tb = make_tensor(buf_AB, lb_smem)
            tc = make_tensor(buf_C, lc_smem)
            mm.execute(float32(1.0), ta, tb, float32(0.0), tc)
            cuda.syncthreads()

            # 1c: regularize ata diagonal by 1e-6 (col-major NC×NC, [i,i] = i*NC+i).
            # Mirrors the torch IPM reference; keeps Cholesky stable when x[k]
            # values shrink and `ata` becomes near-singular in float32 noise.
            for i in range(tid, nc, block_dim):
                buf_C[i * nc + i] = buf_C[i * nc + i] + float32(1e-6)
            cuda.syncthreads()

            # 2: rhs[i] = sum_k A[i,k] * x[k]^2 * c[k]  (Numba, small)
            for i in range(tid, nc, block_dim):
                s = float32(0.0)
                for k in range(nv):
                    xk = x_s[k]
                    s += A_orig[i * nv + k] * xk * xk * c_s[k]
                rhs_s[i] = s
            cuda.syncthreads()

            # 3: Cholesky on buf_C (col-major, ldc=NC).
            chol.factorize(buf_C, info_s)
            cuda.syncthreads()

            if info_s[0] != chol_info_type(0):
                failed = True
                break

            chol.solve(buf_C, rhs_s)
            cuda.syncthreads()

            # 4: r = A.T @ delta;  d = x * (c - r)  (Numba, small)
            for j in range(tid, nv, block_dim):
                s = float32(0.0)
                for i in range(nc):
                    s += A_orig[i * nv + j] * float32(rhs_s[i])
                d_s[j] = x_s[j] * (c_s[j] - s)
            cuda.syncthreads()

            # 5: d_max via block reduction
            local_max = float32(-1e30)
            for j in range(tid, nv, block_dim):
                v = d_s[j]
                if v > local_max:
                    local_max = v
            red_s[tid] = local_max
            cuda.syncthreads()
            stride = block_dim // 2
            while stride > 0:
                if tid < stride:
                    v = red_s[tid + stride]
                    if v > red_s[tid]:
                        red_s[tid] = v
                cuda.syncthreads()
                stride //= 2
            d_max = red_s[0]

            if d_max <= float32(0.0):
                alpha = float32(0.0)
            else:
                alpha = float32(0.999) / d_max
            for j in range(tid, nv, block_dim):
                x_s[j] = x_s[j] * (float32(1.0) - alpha * d_s[j])
            cuda.syncthreads()

        # Convergence: max|Ax - b|_inf (reuse rhs_s to hold |residual|)
        max_res = float32(1e30)
        if not failed:
            for i in range(tid, nc, block_dim):
                s = float32(0.0)
                for k in range(nv):
                    s += A_orig[i * nv + k] * x_s[k]
                r_i = s - b_g[i]
                rhs_s[i] = r_i if r_i >= float32(0.0) else -r_i
            cuda.syncthreads()

            local_max = float32(0.0)
            for i in range(tid, nc, block_dim):
                v = float32(rhs_s[i])
                if v > local_max:
                    local_max = v
            red_s[tid] = local_max
            cuda.syncthreads()
            stride = block_dim // 2
            while stride > 0:
                if tid < stride:
                    v = red_s[tid + stride]
                    if v > red_s[tid]:
                        red_s[tid] = v
                cuda.syncthreads()
                stride //= 2
            max_res = red_s[0]

        x_last = x_s[nv - 1]
        ok = (
            (not failed)
            and (d_max < float32(0.1))
            and (x_last >= float32(0.0))
            and (x_last < float32(1e-4))
            and (max_res < float32(0.05))
        )

        # Per-element output sanitization. Each thread independently checks its
        # x_s[j] for NaN / Inf / out-of-range and writes 0.5 if bad — even when
        # `ok` is True, individual elements can still be NaN because the
        # d_max / max_res reductions silently drop NaN inputs. Using libdevice
        # math.isnan/math.isinf (verified in a unit test) avoids any compiler
        # fold-away of the bare ``v != v`` idiom.
        for j in range(tid, nv, block_dim):
            v = x_s[j]
            if (
                (not ok)
                or math.isnan(v)
                or math.isinf(v)
                or (v < float32(-1e-3))
                or (v > float32(1e6))
            ):
                x_out[j] = float32(0.5)
            else:
                x_out[j] = v

    _kernel_cache[key] = (ipm_kernel, mm, chol, TOTAL_DYN)
    return ipm_kernel, mm, chol, TOTAL_DYN


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:
    """Drop-in replacement for ``torch_solver.solve_ipm``.

    Uses cuBLASDx for the dominant SYRK and cuSolverDx for the KKT Cholesky.
    Tiles live in dynamic shared memory — no static-shmem ceiling on NC.
    """
    assert A.is_cuda and b.is_cuda and c.is_cuda
    assert A.dtype == torch.float32
    nc, nv = A.shape
    assert b.shape == (nc,)
    assert c.shape == (nv,)

    assert_fits(nc, nv, gpu="h100")

    A_cu = cuda.as_cuda_array(A.contiguous())
    b_cu = cuda.as_cuda_array(b.contiguous())
    c_cu = cuda.as_cuda_array(c.contiguous())
    x_out = torch.empty(nv, dtype=torch.float32, device=A.device)
    x_cu = cuda.as_cuda_array(x_out)

    block_dim = _choose_block_dim(nc)
    key = (nc, nv, block_dim, num_iters)
    entry = _spec_cache.get(key)
    if entry is None:
        kernel, _mm, _chol, total_dyn_elems = _build_kernel(nc, nv, block_dim, num_iters)
        spec = kernel.specialize(A_cu, b_cu, c_cu, x_cu)
        dyn_shmem_bytes = total_dyn_elems * 4
        entry = (spec, dyn_shmem_bytes, False)
        _spec_cache[key] = entry

    spec, dyn_shmem_bytes, optin_done = entry
    if not optin_done and dyn_shmem_bytes > 48 * 1024:
        _optin_dynamic_shmem(spec, dyn_shmem_bytes, A.device.index or 0)
        _spec_cache[key] = (spec, dyn_shmem_bytes, True)

    # Launch on PyTorch's current stream so subsequent torch ops on x_out are
    # ordered after the kernel write. PyTorch (sglang) uses per-thread default
    # streams; Numba's bare `0` arg means CUDA's legacy default stream — the
    # two are NOT auto-synchronized, so without this the caller can read x_out
    # before the kernel finishes (manifesting as NaN/Inf garbage).
    torch_stream_handle = torch.cuda.current_stream(A.device).cuda_stream
    numba_stream = cuda.external_stream(torch_stream_handle)
    spec[1, block_dim, numba_stream, dyn_shmem_bytes](A_cu, b_cu, c_cu, x_cu)
    return x_out


def warmup(nc: int, nv: int, num_iters: int = 5, device: str = "cuda") -> None:
    A = torch.zeros(nc, nv, dtype=torch.float32, device=device)
    b = torch.zeros(nc, dtype=torch.float32, device=device)
    c = torch.zeros(nv, dtype=torch.float32, device=device)
    solve_ipm(A, b, c, num_iters=num_iters)
