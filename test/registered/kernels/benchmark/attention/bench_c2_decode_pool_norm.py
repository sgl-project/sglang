"""Ratio-2 decode pair pooling and RMSNorm, excluding projections and KV stores."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.attention.dsv4.c2_decode_pool import c2_decode_pool
from sglang.kernels.ops.layernorm.rmsnorm_fp32 import rmsnorm_fp32
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HEAD_DIM = 512
RING_SIZE = 8
EPS = 1e-6


def pool_norm(kv, score, pos, raw_out_loc, out_loc, req, state, weight, *, fused):
    # Slice after graph input cloning to preserve the interleaved state layout.
    pooled, group_pos, slots = c2_decode_pool(
        kv,
        score,
        pos,
        raw_out_loc,
        out_loc,
        req,
        state[:, :HEAD_DIM],
        state[:, HEAD_DIM:],
        state.shape[0] - 1,
        ring_size=RING_SIZE,
        norm_weight=weight if fused else None,
        norm_eps=EPS,
    )
    if not fused:
        pooled = rmsnorm_fp32(pooled.to(torch.bfloat16), weight, EPS)
    return pooled, group_pos, slots


@marker.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64], [1, 16, 64])
@marker.benchmark("impl", ["separate", "fused"])
def benchmark(batch_size: int, impl: str):
    torch.manual_seed(42)
    kv = create_random(batch_size, HEAD_DIM, dtype=torch.float32)
    score = create_random(batch_size, HEAD_DIM, dtype=torch.float32)
    state = create_random(batch_size * RING_SIZE + 1, 2 * HEAD_DIM, dtype=torch.float32)
    weight = create_random(HEAD_DIM) * 0.1 + 1.0
    req = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    # Alternate completing and pending pairs; batch size 1 completes a pair.
    pos = 2 * RING_SIZE + 1 + req % 2
    raw_out_loc = 2 * (req + 1) + pos % 2
    out_loc = torch.where(pos % 2 == 1, raw_out_loc // 2, -1)
    # Replays write pos % RING_SIZE and read its predecessor, so inputs stay fixed.
    return marker.do_bench(
        pool_norm,
        input_args=(kv, score, pos, raw_out_loc, out_loc, req, state, weight),
        input_kwargs={"fused": impl == "fused"},
        graph_clone_args="all",
        # Only two rows per request are touched; the full ring size overcounts bytes.
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
