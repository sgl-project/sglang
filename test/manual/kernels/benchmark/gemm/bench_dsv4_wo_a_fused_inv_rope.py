"""Fused inverse-RoPE + WO-A against the unfused SM90 decode chain.

The unfused chain is fused_rope_inplace(inverse=True) + torch.einsum, which is
what DeepseekV4 runs on Hopper. Both sides replay from a CUDA graph and cycle
through COPIES weight buffers so every call reads its 8 MB per group from HBM.

    python test/manual/kernels/benchmark/gemm/bench_dsv4_wo_a_fused_inv_rope.py
"""

import torch

from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace
from sglang.kernels.ops.gemm.dsv4_wo_a import (
    WO_A_FUSED_INV_ROPE_MAX_TOKENS,
    wo_a_bf16_fused_inv_rope,
)

# 24 x 8 MB is well past the 50 MB L2 on H100/H200.
COPIES = 24
MAX_POS = 8192


def _graph_time_us(fn) -> float:
    fn(0)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        for i in range(COPIES):
            fn(i)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(25):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / COPIES)
    return sorted(times)[len(times) // 2]


def main():
    angles = torch.outer(
        torch.arange(MAX_POS, device="cuda", dtype=torch.float32),
        1.0 / 10000.0 ** (torch.arange(32, device="cuda", dtype=torch.float32) / 32),
    )
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    freqs = torch.view_as_real(freqs_cis).flatten(1)
    print(f"{torch.cuda.get_device_name()}")
    print("groups tokens  unfused_us  fused_us  speedup")
    for groups in (1, 2, 4, 8):
        weights = [
            torch.randn(groups, 1024, 4096, dtype=torch.bfloat16, device="cuda")
            for _ in range(COPIES)
        ]
        for tokens in (1, 2, 4, 8, 12, 16):
            outs = [
                torch.randn(tokens, 64, 512, dtype=torch.bfloat16, device="cuda")[
                    :, : 8 * groups
                ]
                for _ in range(COPIES)
            ]
            positions = torch.randint(0, MAX_POS, (tokens,), device="cuda")

            def unfused(i):
                fused_rope_inplace(
                    outs[i][..., -64:], None, freqs_cis, positions, inverse=True
                )
                return torch.einsum(
                    "tgd,grd->tgr", outs[i].view(tokens, groups, 4096), weights[i]
                )

            def fused(i):
                return wo_a_bf16_fused_inv_rope(
                    outs[i].view(tokens, groups, 4096), weights[i], freqs, positions
                )

            assert tokens <= WO_A_FUSED_INV_ROPE_MAX_TOKENS
            base = _graph_time_us(unfused)
            new = _graph_time_us(fused)
            print(f"{groups:6d} {tokens:6d} {base:11.2f} {new:9.2f} {base / new:7.2f}x")


if __name__ == "__main__":
    main()
