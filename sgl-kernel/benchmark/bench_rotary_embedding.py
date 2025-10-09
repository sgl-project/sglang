import itertools
import os

import torch
import triton
from sgl_kernel import FusedSetKVBufferArg
from sgl_kernel.testing.rotary_embedding import (
    FlashInferRotaryEmbedding,
    MHATokenToKVPool,
    RotaryEmbedding,
    create_inputs,
)

from sglang.srt.utils.bench_utils import bench_kineto

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# CI environment uses simplified parameters
if IS_CI:
    batch_seq_configs = [(1, 1)]  # Single config for CI
    save_kv_configs = [False]  # Single option for CI
else:
    batch_seq_configs = [
        (1, 1),
        (32, 1),
        (128, 1),
        (512, 1),
        (2, 512),
        (4, 4096),
    ]
    save_kv_configs = [False, True]

configs = [
    (batch_size, seq_len, save_kv_cache)
    for batch_size, seq_len in batch_seq_configs
    for save_kv_cache in save_kv_configs
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "save_kv_cache"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="bench_rotary_embedding",
        args={},
    )
)
def benchmark(batch_size, seq_len, save_kv_cache, provider):
    device = torch.device("cuda")

    num_q_heads = 32
    num_kv_heads = 8
    head_size = 64
    dtype = torch.bfloat16

    config = dict(
        head_size=head_size,
        rotary_dim=64,
        max_position_embeddings=4096,
        base=8000,
        is_neox_style=True,
        dtype=dtype,
    )
    rope_flashinfer = FlashInferRotaryEmbedding(**config).to(device)
    pool_flashinfer = MHATokenToKVPool(head_num=num_kv_heads, head_dim=head_size)

    inputs = create_inputs(
        head_size=head_size,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        dtype=dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )

    query_flashinfer, key_flashinfer = inputs["query"].clone(), inputs["key"].clone()

    bench_fn = lambda: rope_flashinfer.forward_cuda(
        inputs["pos_ids"],
        query_flashinfer,
        key_flashinfer,
        fused_set_kv_buffer_arg=(
            FusedSetKVBufferArg(
                value=inputs["value"],
                k_buffer=pool_flashinfer.k_buffer[0].view(-1, num_kv_heads * head_size),
                v_buffer=pool_flashinfer.v_buffer[0].view(-1, num_kv_heads * head_size),
                k_scale=None,
                v_scale=None,
                cache_loc=inputs["out_cache_loc"],
            )
            if save_kv_cache
            else None
        ),
    )

    time_s = bench_kineto(bench_fn, kernel_names="BatchQKApplyRotaryPosIds")
    return time_s * 1e6


if __name__ == "__main__":
    benchmark.run(print_data=True)
