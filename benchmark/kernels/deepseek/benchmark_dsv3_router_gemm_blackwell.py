import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flashinfer.gemm import mm_M1_16_K6144_N256, mm_M1_16_K7168_N256
from flashinfer.testing import bench_gpu_time_with_cupti
from sgl_kernel import dsv3_router_gemm

# K=7168 covers DeepSeek-V3/V3.2; K=6144 covers GLM-MoE-DSA. The router gate is
# replicated across TP ranks, so M*N*K is the full per-rank problem.
KN_PAIRS = [(7168, 256), (6144, 256)]
BATCH_SIZES = list(range(1, 17))

# Custom router-gemm kernels (sgl_kernel + flashinfer mm_M1_16_*) only support M<=16.
CUSTOM_KERNEL_M_MAX = 16


def create_benchmark_configs():
    return [(m, n, k) for k, n in KN_PAIRS for m in BATCH_SIZES]


def _flashinfer_op_for(k: int, n: int):
    if k == 7168 and n == 256:
        return mm_M1_16_K7168_N256
    if k == 6144 and n == 256:
        return mm_M1_16_K6144_N256
    return None


def dsv3_router_gemm_flashinfer(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
):
    k = hidden_states.shape[1]
    n = router_weights.shape[0]
    op = _flashinfer_op_for(k, n)
    if op is None:
        raise RuntimeError(f"No flashinfer router-gemm op for (K={k}, N={n})")
    output = torch.empty(
        hidden_states.shape[0],
        router_weights.shape[0],
        device="cuda",
        dtype=torch.float32,
    )
    op(hidden_states, router_weights.t(), output, launch_with_pdl=args.use_pdl)
    return output


def dsv3_router_gemm_sgl(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
):
    return dsv3_router_gemm(hidden_states, router_weights, out_dtype=torch.float32)


def _close_enough(a: torch.Tensor, b: torch.Tensor, atol=1e-2, rtol=1e-2, percent=0.99):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return False
    if a.shape != b.shape:
        return False
    return torch.isclose(a, b, atol=atol, rtol=rtol).float().mean().item() >= percent


def calculate_diff(m: int, n: int, k: int):
    if m > CUSTOM_KERNEL_M_MAX:
        return

    hidden_states = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    router_weights = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    ref = (hidden_states.float() @ router_weights.float().t()).to(torch.float32)

    out_flashinfer = dsv3_router_gemm_flashinfer(hidden_states, router_weights)
    assert _close_enough(
        out_flashinfer, ref
    ), f"flashinfer mismatch at m={m}, n={n}, k={k}"

    if k == 7168:
        out_sgl = dsv3_router_gemm_sgl(hidden_states, router_weights)
        assert _close_enough(
            out_sgl, ref
        ), f"sgl_kernel mismatch at m={m}, n={n}, k={k}"


def _benchmark(m, n, k, provider):
    hidden_states = torch.randn(
        (m, k), device="cuda", dtype=torch.bfloat16
    ).contiguous()
    router_weights = torch.randn(
        (n, k), device="cuda", dtype=torch.bfloat16
    ).contiguous()

    if provider == "sglang":
        if k != 7168 or m > CUSTOM_KERNEL_M_MAX:
            # sgl_kernel.dsv3_router_gemm is specialized for K=7168, M<=16.
            return float("nan"), float("nan"), float("nan")
        out = torch.empty((m, n), device="cuda", dtype=torch.float32)
        fn = lambda: torch.ops.sgl_kernel.dsv3_router_gemm(
            out, hidden_states, router_weights
        )
    elif provider == "flashinfer":
        op = _flashinfer_op_for(k, n)
        if op is None or m > CUSTOM_KERNEL_M_MAX:
            return float("nan"), float("nan"), float("nan")
        out = torch.empty((m, n), device="cuda", dtype=torch.float32)
        weights_t = router_weights.t()  # column-major view
        fn = lambda: op(hidden_states, weights_t, out, launch_with_pdl=args.use_pdl)
    elif provider == "torch":
        # Same call shape as sglang's MoEGate fallback: F.linear(hidden_states, weight, None).
        fn = lambda: F.linear(hidden_states, router_weights, None)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    times_ms = bench_gpu_time_with_cupti(
        fn=fn,
        dry_run_time_ms=25,
        repeat_time_ms=100,
        cold_l2_cache=True,
    )
    ms = float(np.median(times_ms))
    min_ms = float(np.quantile(times_ms, 0.2))
    max_ms = float(np.quantile(times_ms, 0.8))
    return ms, max_ms, min_ms


# Provider id -> human-readable label used in the result table.
PROVIDERS = [
    ("torch", "torch (F.linear)"),
    ("sglang", "SGLang"),
    ("flashinfer", "Flashinfer (custom)"),
]


def run_benchmark(save_path=None) -> pd.DataFrame:
    rows = []
    for m, n, k in create_benchmark_configs():
        row = {"m": m, "n": n, "k": k}
        for prov_id, prov_name in PROVIDERS:
            ms, _, _ = _benchmark(m, n, k, prov_id)
            row[prov_name] = ms * 1000  # ms -> us
        # Winner = provider with the minimum (non-NaN) median time.
        valid = {p[1]: row[p[1]] for p in PROVIDERS if not np.isnan(row[p[1]])}
        row["Winner"] = min(valid, key=valid.get) if valid else "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    fmt = {col: "{:.3f}".format for _, col in PROVIDERS}
    print(df.to_string(index=False, formatters=fmt, na_rep="NaN"))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, "dsv3-router-gemm-bf16.csv"), index=False)

    return df


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        print("Skipping benchmark because the device is not supported")
        exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/dsv3_router_gemm/",
        help="Path to save dsv3 router gemm benchmark results",
    )
    parser.add_argument(
        "--run-correctness",
        action="store_true",
        default=True,
        help="Whether to run correctness test",
    )
    parser.add_argument(
        "--use-pdl",
        action="store_true",
        default=False,
        help="Use PDL if true.",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if args.use_pdl:
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

    # Silent correctness check; raises on mismatch.
    if args.run_correctness:
        configs = create_benchmark_configs()
        for m, n, k in configs:
            calculate_diff(m, n, k)
        print(f"Correctness OK ({len(configs)} configs).")

    print("Running performance benchmark...")
    run_benchmark(save_path=args.save_path)
