"""Whether a Triton ``tl.dot`` may consume operands in the fp8 KV-cache dtype.

The attention kernels that read a KV pool feed the matrix core in the pool's
dtype, casting the wide operand down (``tl.dot(q.to(k.dtype), k)``,
``tl.dot(p.to(v.dtype), v)``) so an fp8 pool also buys fp8 matrix-core
throughput. On gfx950 that only works for reductions of 128 or more.

Triton's AMD backend promotes an fp8 x fp8 ``tl.dot`` on gfx950 to the scaled
``V_MFMA_*_F8F6F4`` family, which is 16x16x128 under the
``matrix_instr_nonkdim=16`` launch hint these kernels pass. A narrower
reduction -- ``BLOCK_N`` for P @ V, ``BLOCK_DPE`` for the RoPE half of Q @ K,
``BLOCK_DMODEL`` at head_dim <= 64 -- does not tile that instruction, and
``AMDMfmaEncodingAttr::getRepForOperand`` clamps the resulting zero
K-repetitions up to one. The LLVM conversion then reads 32 elements per lane
from a register range holding only ``K / 4``, tripping an ``llvm::SmallVector``
bounds assertion inside ``ConvertTritonAMDGPUToLLVM``::

    Assertion `idx < size()' failed.
    error: Failures have been detected while processing an MLIR pass pipeline
    RuntimeError: PassManager::run failed

triton-lang/triton#8278 made Triton reject the intrinsic instead, by dropping
the ``enforcedNonKDim == 0 &&`` escape hatch that had let an explicit
``matrix_instr_nonkdim`` skip its "K is not a multiple of the intrinsic's kDim"
check; the Triton 3.4 wheels in the ROCm images predate that fix. Keeping the
narrow dots in the query dtype avoids the promotion on every Triton version and
gives up no throughput: the non-scaled ``V_MFMA_F32_16X16X32_FP8_FP8`` a fixed
Triton falls back to reduces the same 32 elements per instruction as
``V_MFMA_F32_16X16X32_BF16`` does on CDNA4.

Only dots that cast one operand down reach the promotion, so this covers every
one of those: extend (both kernels), decode's grouped stage 1, and the two
verify stage-1 kernels. Two other ROCm Triton kernels dot against a tile that
could in principle be fp8 but need no verdict:

* ``prefill_attention.py`` never reads a KV pool, its ``Q @ K`` is uncast (so
  operands already share a dtype), and its gfx950 ``BLOCK_N`` is 128, which
  tiles the scaled instruction even when ``P @ V`` runs in fp8.
* ``rocm_mla_decode_rope.py`` does read the pool, but its ``Q @ K`` is uncast
  too, and a bf16 query against an fp8 pool fails Triton's frontend
  (``Unsupported rhs dtype fp8e4nv``) long before the AMD backend -- a separate
  pre-existing gap in that kernel, not this crash.
"""

import torch

from sglang.srt.utils import is_gfx95_supported

_is_gfx95 = is_gfx95_supported()

# K reduced per V_MFMA_SCALE_F32_16X16X128_F8F6F4, the scaled instruction Triton
# selects for an fp8 x fp8 dot once matrix_instr_nonkdim=16 pins the 16x16 tile.
_GFX950_SCALED_MFMA_K = 128

_FP8_DTYPES = frozenset(
    {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
)


def dot_in_kv_dtype(kv_dtype: torch.dtype, reduction_width: int) -> bool:
    """Whether a ``tl.dot`` reducing ``reduction_width`` elements may run in ``kv_dtype``.

    ``True`` (keep today's cast-down) for every non-fp8 pool and every GPU
    other than gfx950; on gfx950 an fp8 pool is only allowed the reduction
    widths that tile the scaled MFMA instruction. Callers that get ``False``
    must upcast the KV tile to the query dtype instead of casting the query
    tile down.
    """
    if not _is_gfx95 or kv_dtype not in _FP8_DTYPES:
        return True
    return reduction_width % _GFX950_SCALED_MFMA_K == 0
