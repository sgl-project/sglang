import os
import platform
import statistics
import time

import torch

from sglang.kernels.ops.layernorm import (
    _FUSED_DUAL_RESIDUAL_RMSNORM,
    _FUSED_RMSNORM,
)

HIDDEN_SIZE = 6144  # Grok-1 hidden size
EPS = 1e-5  # Grok-1 rms_norm_eps
ROWS = [1, 4096]
DTYPES = [torch.bfloat16, torch.float16]
NOISE_ALLOWANCE = 0.05


def make_inputs(rows, dtype):
    return (
        torch.randn([rows, HIDDEN_SIZE], dtype=dtype),  # single-op activation
        torch.randn([rows, HIDDEN_SIZE], dtype=dtype),  # dual-op activation
        torch.randn([rows, HIDDEN_SIZE], dtype=dtype),  # dual-op residual
        torch.randn(HIDDEN_SIZE, dtype=dtype),
        torch.randn(HIDDEN_SIZE, dtype=dtype),
    )


def bench(fn, warmup=5, target_seconds=0.3):
    """Median wall time per call, in microseconds, over >= 7 repetitions."""
    for _ in range(warmup):
        fn()
    reps = 0
    elapsed = 0.0
    samples = []
    while elapsed < target_seconds or len(samples) < 7:
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
        elapsed += samples[-1]
        reps += 1
        if reps > 5000:  # 1-row shapes are microseconds; keep the run bounded
            break
    return statistics.median(samples) * 1e6


def compile_or_none(fn):
    try:
        return torch.compile(fn, dynamic=False)
    except Exception as exc:  # pragma: no cover - report-only provider
        print(f"  torch.compile unavailable: {type(exc).__name__}: {exc}")
        return None


def row(rows, label, single_us, dual_us, eager_single_us, eager_dual_us):
    def ratio(kernel_us, eager_us):
        return kernel_us / eager_us if eager_us else float("nan")

    print(
        f"{rows:>5} {label:<20} "
        f"{single_us:>10.2f} {eager_single_us:>10.2f} {ratio(single_us, eager_single_us):>9.2f} "
        f"{dual_us:>10.2f} {eager_dual_us:>10.2f} {ratio(dual_us, eager_dual_us):>9.2f}"
    )


def main():
    print(f"host            : {platform.processor() or platform.machine()}")
    print(f"torch           : {torch.__version__}")
    print(f"threads         : {torch.get_num_threads()} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")
    print(f"hidden size     : {HIDDEN_SIZE}, eps {EPS}, providers warmup=5")
    print()
    header = (
        f"{'rows':>5} {'provider':<20} "
        f"{'single_us':>10} {'eager_us':>10} {'ratio':>9} "
        f"{'dual_us':>10} {'eager_us':>10} {'ratio':>9}"
    )
    print(header)
    print("-" * len(header))

    verdicts = []
    for dtype in DTYPES:
        for rows in ROWS:
            single_x, dual_x, residual, w1, w2 = make_inputs(rows, dtype)

            eager_single = lambda: _FUSED_RMSNORM.forward_native(
                single_x, w1, EPS
            )
            eager_dual = lambda: _FUSED_DUAL_RESIDUAL_RMSNORM.forward_native(
                dual_x, residual, w1, w2, EPS
            )
            kernel_single = lambda: torch.ops.sgl_kernel.fused_rmsnorm_cpu(
                single_x, w1, EPS
            )
            kernel_dual = lambda: torch.ops.sgl_kernel.fused_dual_residual_rmsnorm_cpu(
                dual_x, residual, w1, w2, EPS
            )

            eager_single_us = bench(eager_single)
            eager_dual_us = bench(eager_dual)
            single_us = bench(kernel_single)
            dual_us = bench(kernel_dual)
            row(rows, f"kernel {str(dtype).split('.')[1]}", single_us, dual_us, eager_single_us, eager_dual_us)

            compiled_single = compile_or_none(eager_single)
            compiled_dual = compile_or_none(eager_dual)
            if compiled_single is not None:
                row(
                    rows,
                    f"torch.compile {str(dtype).split('.')[1]}",
                    bench(compiled_single),
                    bench(compiled_dual) if compiled_dual else float("nan"),
                    eager_single_us,
                    eager_dual_us,
                )

            verdicts.append(
                (
                    dtype,
                    rows,
                    single_us / eager_single_us,
                    dual_us / eager_dual_us,
                )
            )

    print()
    print("requirement (kernel not slower than the eager fallback at 6144 rows):")
    for dtype, rows, single_ratio, dual_ratio in verdicts:
        if rows != max(ROWS):
            continue
        for name, ratio in (("single", single_ratio), ("dual", dual_ratio)):
            status = "PASS" if ratio <= 1.0 - NOISE_ALLOWANCE else (
                "NOISE" if ratio < 1.0 else "FAIL"
            )
            print(
                f"  {name:>6} {str(dtype).split('.')[1]:>8} rows={rows}: "
                f"kernel/eager = {ratio:.3f} -> {status}"
            )


if __name__ == "__main__":
    main()
