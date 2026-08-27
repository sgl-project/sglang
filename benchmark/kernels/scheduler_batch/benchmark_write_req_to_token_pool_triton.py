import itertools
import os

import torch
import triton
import triton.language as tl


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)

    # TODO: optimize this?
    cumsum_start = 0
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


@triton.jit
def write_req_to_token_pool_triton_optimize(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_token = tl.program_id(1)

    req_pool_index = tl.load(req_pool_indices + pid_batch)
    pre_len = tl.load(pre_lens + pid_batch)
    seq_len = tl.load(seq_lens + pid_batch)
    extend_len = seq_len - pre_len

    cumsum_start = 0
    for i in range(pid_batch):
        cumsum_start += tl.load(extend_lens + i)

    token_start = pid_token * BLOCK_SIZE

    offset = tl.arange(0, BLOCK_SIZE)
    actual_offset = token_start + offset
    mask = actual_offset < extend_len

    src_ptr = out_cache_loc + cumsum_start + actual_offset
    src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
    value = tl.load(src_ptr, mask=mask)
    dst_ptr = (
        req_to_token_ptr
        + req_pool_index * req_to_token_ptr_stride
        + actual_offset
        + pre_len
    )
    dst_ptr = tl.max_contiguous(tl.multiple_of(dst_ptr, BLOCK_SIZE), BLOCK_SIZE)

    tl.store(dst_ptr, value, mask=mask)


def write_req_to_token_pool_reference(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    pre_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    """Reference implementation using PyTorch"""
    for i in range(len(req_pool_indices)):
        req_pool_idx = req_pool_indices[i].item()
        pre_len = pre_lens[i].item()
        seq_len = seq_lens[i].item()
        extend_len = extend_lens[i].item()

        cumsum_start = sum(extend_lens[:i].tolist())

        # Copy values from out_cache_loc to req_to_token
        req_to_token[req_pool_idx, pre_len:seq_len] = out_cache_loc[
            cumsum_start : cumsum_start + extend_len
        ]


def test_write_req_to_token_pool():
    max_batch = 4097
    max_context_len = 6148
    batch_size = 1
    extend_len = 14

    # Initialize input tensors
    req_to_token = torch.zeros(
        (max_batch, max_context_len), dtype=torch.int32, device="cuda"
    )
    req_pool_indices = torch.tensor([42], dtype=torch.int32, device="cuda")
    pre_lens = torch.tensor([8], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([22], dtype=torch.int32, device="cuda")
    extend_lens = torch.tensor([extend_len], dtype=torch.int32, device="cuda")
    out_cache_loc = torch.arange(extend_len, dtype=torch.int32, device="cuda")

    # Create copies for reference implementation
    req_to_token_ref = req_to_token.clone()
    req_to_token_opt = req_to_token.clone()

    # Run original triton kernel
    write_req_to_token_pool_triton[(batch_size,)](
        req_to_token,
        req_pool_indices,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
        max_context_len,
    )

    # Run optimized triton kernel
    def grid(batch_size, extend_len):
        num_token_blocks = triton.cdiv(extend_len, 512)
        return (batch_size, num_token_blocks)

    write_req_to_token_pool_triton_optimize[grid(batch_size, extend_len)](
        req_to_token_opt,
        req_pool_indices,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
        max_context_len,
        BLOCK_SIZE=512,
    )

    # Run reference implementation
    write_req_to_token_pool_reference(
        req_to_token_ref,
        req_pool_indices,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
    )

    # Compare results
    torch.testing.assert_close(req_to_token, req_to_token_ref)
    torch.testing.assert_close(req_to_token_opt, req_to_token_ref)

    # Test case 2: batch size > 1
    batch_size = 3
    extend_lens_list = [14, 20, 30]
    total_extend_len = sum(extend_lens_list)

    req_to_token = torch.zeros(
        (max_batch, max_context_len), dtype=torch.int32, device="cuda"
    )
    req_pool_indices = torch.tensor([42, 100, 200], dtype=torch.int32, device="cuda")
    pre_lens = torch.tensor([8, 10, 15], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([22, 30, 45], dtype=torch.int32, device="cuda")
    extend_lens = torch.tensor(extend_lens_list, dtype=torch.int32, device="cuda")
    out_cache_loc = torch.arange(total_extend_len, dtype=torch.int32, device="cuda")

    req_to_token_ref = req_to_token.clone()
    req_to_token_opt = req_to_token.clone()

    # Run original triton kernel
    write_req_to_token_pool_triton[(batch_size,)](
        req_to_token,
        req_pool_indices,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
        max_context_len,
    )

    # Run optimized triton kernel
    max_extend_len = max(extend_lens_list)
    write_req_to_token_pool_triton_optimize[grid(batch_size, max_extend_len)](
        req_to_token_opt,
        req_pool_indices,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
        max_context_len,
        BLOCK_SIZE=512,
    )

    # Run reference implementation
    write_req_to_token_pool_reference(
        req_to_token_ref,
        req_pool_indices,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
    )

    # Compare results
    torch.testing.assert_close(req_to_token, req_to_token_ref)
    torch.testing.assert_close(req_to_token_opt, req_to_token_ref)


def get_benchmark():
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    extend_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    configs = list(itertools.product(batch_sizes, extend_lens))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "extend_len"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["reference", "triton", "triton_optimize"],
            line_names=["PyTorch", "Triton", "Triton Optimized"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="us",
            plot_name="write-req-to-token-pool-performance",
            args={},
        )
    )
    def benchmark(batch_size, extend_len, provider):
        max_batch = 256
        max_context_len = 16384

        extend_lens_list = [extend_len] * batch_size
        total_extend_len = sum(extend_lens_list)

        req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device="cuda"
        )
        req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
        pre_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda") * 8
        seq_lens = pre_lens + extend_len
        extend_lens = torch.tensor(extend_lens_list, dtype=torch.int32, device="cuda")
        out_cache_loc = torch.arange(total_extend_len, dtype=torch.int32, device="cuda")

        quantiles = [0.5, 0.2, 0.8]

        if provider == "reference":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: write_req_to_token_pool_reference(
                    req_to_token.clone(),
                    req_pool_indices,
                    pre_lens,
                    seq_lens,
                    extend_lens,
                    out_cache_loc,
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: write_req_to_token_pool_triton[(batch_size,)](
                    req_to_token.clone(),
                    req_pool_indices,
                    pre_lens,
                    seq_lens,
                    extend_lens,
                    out_cache_loc,
                    max_context_len,
                ),
                quantiles=quantiles,
            )
        else:

            def run_optimized():
                block_size = 128 if extend_len <= 1024 else 512
                grid_config = (batch_size, triton.cdiv(extend_len, block_size))
                write_req_to_token_pool_triton_optimize[grid_config](
                    req_to_token.clone(),
                    req_pool_indices,
                    pre_lens,
                    seq_lens,
                    extend_lens,
                    out_cache_loc,
                    max_context_len,
                    BLOCK_SIZE=block_size,
                )

            ms, min_ms, max_ms = triton.testing.do_bench(
                run_optimized, quantiles=quantiles
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def run_benchmark(save_path: str = "./configs/benchmark_ops/write_req_to_token_pool/"):
    """Run benchmark and save results"""

    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)

    # Run correctness test
    test_write_req_to_token_pool()
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
        default="./configs/benchmark_ops/write_req_to_token_pool/",
        help="Path to save benchmark results",
    )
    args = parser.parse_args()

    run_benchmark(args.save_path)
