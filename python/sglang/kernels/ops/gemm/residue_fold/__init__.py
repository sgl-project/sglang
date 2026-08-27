"""Residue NVFP4 mext_r1 fold GEMM dispatch for Blackwell."""

from sglang.kernels.ops.gemm.residue_fold.fold import (
    compile_stats,
    observe_compiles,
    run_fold,
    warmup,
)

__all__ = [
    "compile_stats",
    "observe_compiles",
    "run_fold",
    "warmup",
]
