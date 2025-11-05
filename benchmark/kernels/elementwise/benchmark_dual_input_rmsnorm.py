import torch
import triton
from flashinfer.norm import rmsnorm
from triton.testing import do_bench
from vllm import _custom_ops as vllm_ops

from sglang.srt.layers.elementwise import fused_dual_input_rmsnorm


def compute_dual_rms_baseline(x1, x2, w1, w2, eps):
    o1 = torch.nn.functional.rms_norm(x1, (x1.shape[-1],), w1, eps)
    o2 = torch.nn.functional.rms_norm(x2, (x2.shape[-1],), w2, eps)
    return o1, o2


def compute_dual_rmsnorm_flashinfer(
    x1: torch.Tensor,
    x2: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps: float = 1e-6,
):
    out1 = torch.empty_like(x1)
    out2 = torch.empty_like(x2)
    vllm_ops.rms_norm(out1, x1, weight1, eps)
    vllm_ops.rms_norm(out2, x2, weight2, eps)
    return out1, out2


def compute_dual_rmsnorm_vllm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps: float = 1e-6,
):
    return rmsnorm(x1, weight1, eps), rmsnorm(x2, weight2, eps)


def compute_dual_rms_triton(x1, x2, w1, w2, eps):
    return fused_dual_input_rmsnorm(
        x1,
        w1,
        eps,
        x2,
        w2,
        eps,
    )


def get_benchmark():
    num_tokens_range = [2**i for i in range(0, 13)]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=num_tokens_range,
            line_arg="version",
            line_vals=["baseline", "triton", "flashinfer", "vllm"],
            line_names=["Original", "Triton", "Flashinfer", "vLLM"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("yellow", "-")],
            ylabel="us",
            plot_name="dual_rmsnorm_performance",
            args={},
        )
    )
    def benchmark(num_tokens, version):
        hidden_size1 = 1536
        hidden_size2 = 64
        eps = 1e-6

        dtype = torch.bfloat16
        device = "cuda"
        x1 = torch.randn([num_tokens, hidden_size1], dtype=dtype, device=device)
        w1 = torch.ones(hidden_size1, dtype=dtype, device=device)
        x2 = torch.randn([num_tokens, hidden_size2], dtype=dtype, device=device)
        w2 = torch.ones(hidden_size2, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            if version == "baseline":
                compute_dual_rms_baseline(x1, x2, w1, w2, eps)
            elif version == "triton":
                compute_dual_rms_triton(x1, x2, w1, w2, eps)
            elif version == "flashinfer":
                compute_dual_rmsnorm_flashinfer(x1, x2, w1, w2, eps)
            elif version == "vllm":
                compute_dual_rmsnorm_vllm(x1, x2, w1, w2, eps)
            else:
                raise ValueError(f"benchmark not support {version=}.")

        # Benchmark
        quantiles = [0.5, 0.2, 0.8]
        if version == "baseline":
            ms, min_ms, max_ms = do_bench(
                lambda: compute_dual_rms_baseline(x1, x2, w1, w2, eps),
                quantiles=quantiles,
            )
        elif version == "triton":
            ms, min_ms, max_ms = do_bench(
                lambda: compute_dual_rms_triton(x1, x2, w1, w2, eps),
                quantiles=quantiles,
            )
        elif version == "flashinfer":
            ms, min_ms, max_ms = do_bench(
                lambda: compute_dual_rmsnorm_flashinfer(x1, x2, w1, w2, eps),
                quantiles=quantiles,
            )
        elif version == "vllm":
            ms, min_ms, max_ms = do_bench(
                lambda: compute_dual_rmsnorm_vllm(x1, x2, w1, w2, eps),
                quantiles=quantiles,
            )
        else:
            raise ValueError(f"benchmark not support {version=}.")

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def verify_correctness(num_tokens=1024, check_no_contiguous_layout=False):
    hidden_size1 = 1536
    hidden_size2 = 64
    eps = 1e-6
    dtype = torch.bfloat16
    device = "cuda"

    if check_no_contiguous_layout:
        buffer_size = hidden_size1 + hidden_size2
        x1 = torch.randn(num_tokens * buffer_size, dtype=dtype, device=device)
        x1 = torch.as_strided(
            x1, size=(num_tokens, hidden_size1), stride=(buffer_size, 1)
        )
        x2 = torch.randn(num_tokens * buffer_size, dtype=dtype, device=device)
        x2 = torch.as_strided(
            x2, size=(num_tokens, hidden_size2), stride=(buffer_size, 1)
        )
    else:
        x1 = torch.randn([num_tokens, hidden_size1], dtype=dtype, device=device)
        x2 = torch.randn([num_tokens, hidden_size2], dtype=dtype, device=device)
    w1 = torch.randn(hidden_size1, dtype=dtype, device=device)
    w2 = torch.randn(hidden_size2, dtype=dtype, device=device)

    out_baseline = compute_dual_rms_baseline(x1, x2, w1, w2, eps)
    out_triton = compute_dual_rms_triton(x1, x2, w1, w2, eps)

    if torch.allclose(
        out_baseline[0], out_triton[0], atol=1e-2, rtol=1e-2
    ) and torch.allclose(out_baseline[1], out_triton[1], atol=1e-2, rtol=1e-2):
        print(f"✅ All implementations match")
    else:
        print("❌ Implementations differ")
        print(
            f"Baseline vs Triton 1: {(out_baseline[0] - out_triton[0]).abs().max().item()}"
        )
        print(
            f"Baseline vs Triton 2: {(out_baseline[1] - out_triton[1]).abs().max().item()}"
        )


if __name__ == "__main__":
    print("Running correctness verification...")
    verify_correctness()
    verify_correctness(check_no_contiguous_layout=True)

    print("\nRunning performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(
        print_data=True,
        # save_path="./configs/benchmark_ops/dual_input_rmsnorm/"
    )
