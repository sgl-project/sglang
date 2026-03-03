"""
Benchmark: JIT compile time before vs after head_dim/interleave template specialisation.

OLD: no JIT_HEAD_DIM define  -> compiles all 6 instantiations (3 head_dims x 2 bool)
NEW: JIT_HEAD_DIM + JIT_INTERLEAVE defines -> compiles exactly 1 instantiation

Run:
    python python/sglang/jit_kernel/benchmark/bench_compile_time.py
"""

import tempfile
import time

from sglang.jit_kernel.utils import load_jit

CUDA_FILE = ["elementwise/fused_qknorm_rope.cuh"]
WRAPPER = [("fused_qk_norm_rope", "fused_qk_norm_rope")]
BASE_CFLAGS = ["--use_fast_math"]

CASES = [
    # (label, extra_cflags)
    ("OLD  – 6 instantiations (no template specialisation)", []),
    (
        "NEW  – 1 instantiation  (head_dim=64,  neox=False)",
        ["-DJIT_HEAD_DIM=64", "-DJIT_INTERLEAVE=1"],
    ),
    (
        "NEW  – 1 instantiation  (head_dim=128, neox=True)",
        ["-DJIT_HEAD_DIM=128", "-DJIT_INTERLEAVE=0"],
    ),
    (
        "NEW  – 1 instantiation  (head_dim=256, neox=True)",
        ["-DJIT_HEAD_DIM=256", "-DJIT_INTERLEAVE=0"],
    ),
]


def measure(label: str, extra_cflags: list) -> float:
    with tempfile.TemporaryDirectory() as build_dir:
        t0 = time.perf_counter()
        load_jit(
            "bench_compile_time",
            cuda_files=CUDA_FILE,
            cuda_wrappers=WRAPPER,
            extra_cuda_cflags=BASE_CFLAGS + extra_cflags,
            build_directory=build_dir,
        )
        elapsed = time.perf_counter() - t0
    print(f"  {elapsed:5.1f}s  {label}")
    return elapsed


def main():
    print("Measuring JIT compile time (each run uses a fresh temp build dir)\n")
    results = [(label, measure(label, cflags)) for label, cflags in CASES]

    old_time = results[0][1]
    print()
    for label, t in results[1:]:
        speedup = old_time / t if t > 0 else float("inf")
        print(f"  speedup vs OLD: {speedup:.2f}x  ({label})")


if __name__ == "__main__":
    main()
