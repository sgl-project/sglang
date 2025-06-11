import argparse
import itertools

import pandas as pd
import torch
import triton

from sglang.srt.layers.moe.ep_moe.kernels import pre_reorder_triton_kernel


def benchmark_pre_reorder(batch_size, topk, model_config):
    hidden_size = model_config["hidden_size"]
    block_size = model_config["block_size"]
    expert_range = model_config["expert_range"]

    input_ptr = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="cuda")
    gateup_input_ptr = torch.zeros(
        batch_size * topk, hidden_size, dtype=torch.float16, device="cuda"
    )
    src2dst_ptr = torch.randint(
        0, batch_size * topk, (batch_size, topk), dtype=torch.int32, device="cuda"
    )
    topk_ids_ptr = torch.randint(
        expert_range[0],
        expert_range[1] + 1,
        (batch_size, topk),
        dtype=torch.int32,
        device="cuda",
    )
    a1_scales_ptr = torch.rand(
        expert_range[1] - expert_range[0] + 1, dtype=torch.float32, device="cuda"
    )

    input_ptr = input_ptr.view(-1)
    gateup_input_ptr = gateup_input_ptr.view(-1)
    src2dst_ptr = src2dst_ptr.view(-1)
    topk_ids_ptr = topk_ids_ptr.view(-1)

    def run_kernel():
        pre_reorder_triton_kernel[(batch_size,)](
            input_ptr,
            gateup_input_ptr,
            src2dst_ptr,
            topk_ids_ptr,
            a1_scales_ptr,
            expert_range[0],
            expert_range[1],
            topk,
            hidden_size,
            block_size,
            use_per_token_if_dynamic=True,
        )

    for _ in range(10):
        run_kernel()
    torch.cuda.synchronize()

    ms, _, _ = triton.testing.do_bench(run_kernel, quantiles=[0.5, 0.2, 0.8])
    return ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()

    model_config = {
        "hidden_size": args.hidden_size,
        "block_size": args.block_size,
        "expert_range": (0, 255),
    }

    batch_sizes = [64, 128, 256, 512, 640, 768, 1024]
    topks = [2, 4, 8]
    configs = list(itertools.product(batch_sizes, topks))

    # Prepare results dict: keys = topk, each row is indexed by batch_size
    results_dict = {topk: {} for topk in topks}

    for batch_size, topk in configs:
        ms = benchmark_pre_reorder(batch_size, topk, model_config)
        results_dict[topk][batch_size] = ms

    # Build dataframe
    df = pd.DataFrame(
        {
            "batch_size": batch_sizes,
            **{
                f"TopK={topk}": [results_dict[topk].get(bs, None) for bs in batch_sizes]
                for topk in topks
            },
        }
    )

    print("\npre-reorder-performance:")
    print(df.to_string(index=False, float_format="%.6f"))


if __name__ == "__main__":
    main()
