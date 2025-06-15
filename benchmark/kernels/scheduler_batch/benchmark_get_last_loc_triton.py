import os

import torch
import triton
import triton.language as tl


@torch.compile(dynamic=True)
def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


def test_get_last_loc():
    max_batch = 4097
    max_context_len = 6148
    batch_size = 20

    # Initialize input tensors
    req_to_token = torch.zeros(
        (max_batch, max_context_len), dtype=torch.int32, device="cuda"
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    pre_lens = torch.randint(
        -max_context_len // 2,
        max_context_len,
        (batch_size,),
        dtype=torch.int64,
        device="cuda",
    )

    last_loc_res = get_last_loc_triton(req_to_token, req_pool_indices, pre_lens)
    last_loc_ref = get_last_loc_torch(req_to_token, req_pool_indices, pre_lens)

    # Compare results
    torch.testing.assert_close(last_loc_res, last_loc_ref)


def get_benchmark():
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=batch_sizes,
            line_arg="provider",
            line_vals=["reference", "triton"],
            line_names=["PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="get-last-loc-performance",
            args={},
        )
    )
    def benchmark(batch_size, provider):
        max_batch = 2048
        max_context_len = 16384

        req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device="cuda"
        )
        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device="cuda")
        pre_lens = torch.randint(
            -max_context_len // 2,
            max_context_len,
            (batch_size,),
            dtype=torch.int64,
            device="cuda",
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "reference":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: get_last_loc_torch(req_to_token, req_pool_indices, pre_lens),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: get_last_loc_triton(req_to_token, req_pool_indices, pre_lens),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def run_benchmark(save_path: str = "./configs/benchmark_ops/get_last_loc/"):
    """Run benchmark and save results"""

    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)

    # Run correctness test
    test_get_last_loc()
    print("Correctness test passed!")

    # Run performance test
    benchmark = get_benchmark()
    benchmark.run(print_data=True, save_path=save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/get_last_loc/",
        help="Path to save benchmark results",
    )
    args = parser.parse_args()

    run_benchmark(args.save_path)
