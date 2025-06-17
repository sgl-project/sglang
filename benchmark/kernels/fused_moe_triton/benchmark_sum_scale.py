import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
            )
            accumulator += tmp
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


def moe_sum_reduce(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return


def compute_sum_scaled_baseline(
    x: torch.Tensor, out: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)
    return out


@torch.compile
def compute_sum_scaled_compiled(
    x: torch.Tensor, out: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    torch.sum(x * routed_scaling_factor, dim=1, out=out)
    return out


def get_benchmark():
    num_tokens_range = [2**i for i in range(0, 13)]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=num_tokens_range,
            line_arg="version",
            line_vals=["baseline", "compiled", "triton"],
            line_names=["Original", "TorchCompile", "TritonKernel"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="us",
            plot_name="sum_scaled_performance",
            args={},
        )
    )
    def benchmark(num_tokens, version):
        topk = 9
        hidden_size = 4096
        dtype = torch.bfloat16
        scaling_factor = 0.3

        x = torch.randn(num_tokens, topk, hidden_size, dtype=dtype, device="cuda")
        out = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")

        # Warmup
        for _ in range(3):
            if version == "baseline":
                compute_sum_scaled_baseline(x, out, scaling_factor)
            elif version == "compiled":
                compute_sum_scaled_compiled(x, out, scaling_factor)
            else:
                moe_sum_reduce(x, out, scaling_factor)

        # Benchmark
        quantiles = [0.5, 0.2, 0.8]
        if version == "baseline":
            ms, min_ms, max_ms = do_bench(
                lambda: compute_sum_scaled_baseline(x, out, scaling_factor),
                quantiles=quantiles,
            )
        elif version == "compiled":
            ms, min_ms, max_ms = do_bench(
                lambda: compute_sum_scaled_compiled(x, out, scaling_factor),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = do_bench(
                lambda: moe_sum_reduce(x, out, scaling_factor), quantiles=quantiles
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def verify_correctness(num_tokens=1024):
    x = torch.randn(num_tokens, 9, 4096, device="cuda", dtype=torch.bfloat16)
    scaling_factor = 0.3

    out_baseline = torch.empty_like(x[:, 0])
    compute_sum_scaled_baseline(x, out_baseline, scaling_factor)

    out_compiled = torch.empty_like(out_baseline)
    compute_sum_scaled_compiled(x, out_compiled, scaling_factor)

    out_triton = torch.empty_like(out_baseline)
    moe_sum_reduce(x, out_triton, scaling_factor)

    if torch.allclose(
        out_baseline, out_compiled, atol=1e-2, rtol=1e-2
    ) and torch.allclose(out_baseline, out_triton, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")
        print(
            f"Baseline vs Compiled: {(out_baseline - out_compiled).abs().max().item()}"
        )
        print(f"Baseline vs Triton: {(out_baseline - out_triton).abs().max().item()}")


if __name__ == "__main__":
    print("Running correctness verification...")
    verify_correctness()

    print("\nRunning performance benchmark...")
    benchmark = get_benchmark()
    benchmark.run(
        print_data=True,
        # save_path="./configs/benchmark_ops/sum_scaled/"
    )
