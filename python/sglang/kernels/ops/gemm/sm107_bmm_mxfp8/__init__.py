"""CC 10.7 batched MXFP8 GEMM ``out[l] = a[l] @ b[l]`` on FlashInfer's SM107 CuTe-DSL kernel, with an
MXFP8 epilogue (FP8 values + 128x4-swizzled UE8M0 scales) from flashinfer-ai/flashinfer#9003.

Same contract as ``flashinfer.bmm_mxfp8(backend="cute-dsl", out_scale=...)`` restricted to K-major A/B
and row-major out: ``a`` [l, m, k] FP8 (rows may be strided), ``b`` [l, k, n] K-major FP8,
``a_scale``/``b_scale`` 128x4-swizzled UE8M0 bytes per batch, ``out`` [l, m, n] bf16 or FP8 (strided).
For FP8 ``out``, ``out_scale`` is the 128x4 swizzle of the 2-D row-major matrix that holds ``out``,
whose batches must sit side by side in its rows (``out[l] = y[:, l * n:(l + 1) * n]``).
"""

from typing import Optional

import torch

_MMA_INST_SHAPE_K = 64
_MMA_TILER_K = 128
_KERNELS = {}
_ALPHA = {}


def _tactic(m: int):
    if m % 256 != 0:
        return (128, 256), (1, 1), 128
    return (256, 256), ((2, 4) if m >= 4096 else (4, 1)), 256


def _alpha(device: torch.device):
    from cutlass.cute.runtime import from_dlpack

    if device.index not in _ALPHA:
        alpha = torch.ones(1, device=device, dtype=torch.float32)
        _ALPHA[device.index] = (alpha, from_dlpack(alpha))
    return _ALPHA[device.index][1]


def supports(m: int, n: int, k: int) -> bool:
    return m > 0 and m % 128 == 0 and n % 256 == 0 and k % 128 == 0


def bmm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    out_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import make_ptr
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    from .dense_blockscaled_gemm_sm107 import Sm107BlockScaledPersistentDenseGemmKernel

    batch, m, k = a.shape
    n = b.shape[2]
    fp8_out = out.dtype == torch.float8_e4m3fn
    assert supports(m, n, k), (m, n, k)
    assert a.stride(2) == 1 and b.stride(1) == 1 and out.stride(2) == 1
    assert out.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert fp8_out == (out_scale is not None)
    assert a_scale.data_ptr() % 16 == 0 and b_scale.data_ptr() % 16 == 0
    pitch = sfc_offset = 0
    if fp8_out:
        pitch, col0 = out.stride(1), out.stride(0)
        assert pitch % 128 == 0 and col0 % 128 == 0 and (batch - 1) * col0 + n <= pitch
        sfc_offset = (col0 // 128) * 512
        assert out_scale.numel() >= m * pitch // 32

    tile, cluster, inst_m = _tactic(m)
    c_dtype = cutlass.Float8E4M3FN if fp8_out else cutlass.BFloat16
    key = (tile, cluster, inst_m, out.dtype, batch, a.device.index)
    ptr = lambda dt, t, al: make_ptr(
        dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=al
    )
    f8, e8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    args = (
        ptr(f8, a, 16),
        ptr(f8, b, 16),
        ptr(e8, a_scale, 16),
        ptr(e8, b_scale, 16),
        ptr(c_dtype, out, 16),
        _alpha(a.device),
    )
    problem = (cutlass.Int32(m), cutlass.Int32(n), cutlass.Int32(k), batch)
    sfc = ptr(cutlass.Uint8, out_scale, 4) if fp8_out else None
    strides = (
        cutlass.Int64(out.stride(1)),
        sfc,
        cutlass.Int64(pitch),
        cutlass.Int64(sfc_offset),
        cutlass.Int64(a.stride(1)),
        cutlass.Int64(a.stride(0)),
        cutlass.Int64(out.stride(0)),
    )
    with torch.cuda.device(a.device):
        stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        if key not in _KERNELS:
            gemm = Sm107BlockScaledPersistentDenseGemmKernel(
                32,
                (inst_m, tile[1], _MMA_INST_SHAPE_K),
                (tile[0], tile[1], _MMA_TILER_K),
                cluster,
            )
            layouts = (
                OperandMajorMode.K,
                OperandMajorMode.K,
                cutlass_utils.LayoutEnum.ROW_MAJOR,
            )
            _KERNELS[key] = cute.compile(
                gemm,
                *args,
                layouts,
                problem,
                get_max_active_clusters(cluster[0] * cluster[1]),
                stream,
                lambda x: x,
                *strides,
                True,
            )
        _KERNELS[key](
            *args, problem, stream, *(strides if fp8_out else strides[:1] + strides[2:])
        )
    return out
