import argparse
import copy
import itertools

import torch
import triton
from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size

bs_range = [1, 8, 32, 64, 128, 256]
qlen_range = [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

configs = list(itertools.product(bs_range, qlen_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=configs,
        x_log=False,
        line_arg="provider",
        line_vals=[
            "128 heads",
            "64 heads",
            "32 heads",
            "16 heads",
        ],
        line_names=[
            "128 heads",
            "64 heads",
            "32 heads",
            "16 heads",
        ],
        styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--")],
        ylabel="GB/s",
        plot_name="cutlass mla",
        args={},
    )
)
def benchmark(batch_size, seq_len, provider, block_size, num_kv_splits):
    d = 576
    dn = 64
    dv = 512

    h_q_map = {
        "128": 128,
        "64": 64,
        "32": 32,
        "16": 16,
    }
    parsed_h_q = next(
        (value for key, value in h_q_map.items() if key in provider), None
    )

    if parsed_h_q is None:
        raise ValueError(f"Unknown head configuration in provider: {provider}")
    h_q = parsed_h_q

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    max_seq_len = seq_lens.max().item()
    block_num = (max_seq_len + block_size - 1) // block_size

    # Pad block_num so that small blocks can be packed into full 128-sized CUTLASS tiles.
    # One 128-wide tile can hold (128 // block_size) small blocks.
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    qn = (
        torch.randn(h_q, batch_size, d - dn, dtype=torch.bfloat16, device="cuda")
        * 100.0
    )
    qr = torch.randn(batch_size, h_q, dn, dtype=torch.bfloat16, device="cuda") * 100.0
    block_table = torch.randint(
        0,
        batch_size * block_num,
        (batch_size, block_num),
        dtype=torch.int32,
        device="cuda",
    )

    kv_cache = torch.randn(
        block_table.numel(), block_size, d, dtype=torch.bfloat16, device="cuda"
    )

    workspace_size = cutlass_mla_get_workspace_size(
        block_num * block_size, batch_size, num_kv_splits=num_kv_splits
    )
    workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: cutlass_mla_decode(
            qn.transpose(0, 1),
            qr,
            kv_cache,
            seq_lens,
            block_table,
            workspace,
            1.44,
            num_kv_splits,
        ),
        quantiles=quantiles,
    )

    q_size = qn.numel() * qn.element_size() + qr.numel() * qr.element_size()

    gbps = (
        lambda ms: (
            q_size + q_size * dv / d + kv_cache.numel() * kv_cache.element_size()
        )
        * 1e-9
        / (ms * 1e-3)
    )
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=[1, 32, 64, 128],
        help="List of batch sizes",
    )
    parser.add_argument(
        "--num-kv-splits",
        nargs="+",
        type=int,
        default=[-1],
        help="List of batch sizes",
    )
    args = parser.parse_args()

    for block_size in args.block_sizes:
        for kv_split in args.num_kv_splits:
            print(f"block_size={block_size}, num_kv_splits={kv_split}: ")
            benchmark.run(
                print_data=True,
                show_plots=True,
                save_path="bench_blackwell_mla_res",
                block_size=block_size,
                num_kv_splits=kv_split,
            )

    print("Benchmark finished!")
