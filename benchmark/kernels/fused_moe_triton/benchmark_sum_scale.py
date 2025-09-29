import torch
import triton
import triton.language as tl
from sgl_kernel import moe_sum_reduce as moe_sum_reduce_cuda
from triton.testing import do_bench


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    scale_ptr,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    in_ty = input_ptr.dtype.element_ty
    acc_ty = tl.float64 if in_ty == tl.float64 else tl.float32
    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=acc_ty)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0,
        )
        accumulator += tile.to(tl.float32)
    sf = tl.load(scale_ptr).to(acc_ty)
    accumulator *= sf

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
def moe_sum_reduce_triton(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    scale_dev = torch.tensor(
        routed_scaling_factor, dtype=input.dtype, device=input.device
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        scale_dev,
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
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


def get_benchmark(dtype=torch.bfloat16):
    num_tokens_range = [2**i for i in range(0, 13)]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=num_tokens_range,
            line_arg="version",
            line_vals=["baseline", "compiled", "triton", "cuda"],
            line_names=["Original", "TorchCompile", "TritonKernel", "CudaKernel"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("yellow", "-")],
            ylabel="us",
            plot_name=f"sum_scaled_performance_{str(dtype).split('.')[-1]}",
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
            elif version == "triton":
                moe_sum_reduce_triton(x, out, scaling_factor)
            else:
                moe_sum_reduce_cuda(x, out, scaling_factor)

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
        elif version == "triton":
            ms, min_ms, max_ms = do_bench(
                lambda: moe_sum_reduce_triton(x, out, scaling_factor),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = do_bench(
                lambda: moe_sum_reduce_cuda(x, out, scaling_factor),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def verify_correctness(num_tokens=1024, dtype=torch.bfloat16):
    x = torch.randn(num_tokens, 9, 4096, device="cuda", dtype=dtype)
    scaling_factor = 0.3

    out_baseline = torch.empty_like(x[:, 0])
    compute_sum_scaled_baseline(x, out_baseline, scaling_factor)

    out_compiled = torch.empty_like(out_baseline)
    compute_sum_scaled_compiled(x, out_compiled, scaling_factor)

    out_cuda = torch.empty_like(out_baseline)
    moe_sum_reduce_cuda(x, out_cuda, scaling_factor)

    triton_skipped = dtype == torch.float64
    if not triton_skipped:
        out_triton = torch.empty_like(out_baseline)
        moe_sum_reduce_triton(x, out_triton, scaling_factor)

    if dtype == torch.float64:
        atol, rtol = 1e-12, 1e-12
    elif dtype == torch.float32:
        atol, rtol = 1e-6, 1e-6
    else:  # bfloat16 / float16
        atol, rtol = 1e-2, 1e-2

    ok_compiled = torch.allclose(out_baseline, out_compiled, atol=atol, rtol=rtol)
    ok_cuda = torch.allclose(out_baseline, out_cuda, atol=atol, rtol=rtol)
    ok_triton = (
        True
        if triton_skipped
        else torch.allclose(out_baseline, out_triton, atol=atol, rtol=rtol)
    )

    if ok_compiled and ok_triton and ok_cuda:
        msg = "✅ All implementations match"
        if triton_skipped:
            msg += " (Triton skipped for float64)"
        print(msg)
    else:
        print("❌ Implementations differ")
        print(
            f"Baseline vs Compiled: {(out_baseline - out_compiled).abs().max().item()}"
        )
        if not triton_skipped:
            print(
                f"Baseline vs Triton: {(out_baseline - out_triton).abs().max().item()}"
            )
        print(f"Baseline vs Cuda: {(out_baseline - out_cuda).abs().max().item()}")


if __name__ == "__main__":
    print("Running correctness verification for bfloat16...")
    verify_correctness(dtype=torch.bfloat16)
    print("Running correctness verification for float64...")
    verify_correctness(dtype=torch.float64)

    print("\nRunning performance benchmark for bfloat16...")
    benchmark = get_benchmark(dtype=torch.bfloat16)
    benchmark.run(
        print_data=True,
        # save_path="./configs/benchmark_ops/sum_scaled/"
    )
