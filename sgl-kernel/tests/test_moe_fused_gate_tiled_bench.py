import os
import time
from statistics import median

import pytest
import torch

from sgl_kernel import moe_fused_gate


def _measure_time_ms(fn, warmup: int, repeat: int) -> list[float]:
    times_ms: list[float] = []
    # Warmup
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    # Timed runs
    for _ in range(max(1, repeat)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "num_experts,num_expert_group,topk_group,topk",
    [
        # Tiled path (VPT > 32)
        (64, 1, 1, 6),      # kimi-vl: VPT=64
        (384, 1, 1, 8),     # kimi-K2: VPT=384
        (1024, 8, 4, 8),    # VPT=128
        (2048, 8, 4, 8),    # VPT=256
    ],
)
def test_moe_fused_gate_tiled_perf(num_experts, num_expert_group, topk_group, topk):
    # Keep the problem size moderate; allow override by env
    seq_length = int(os.getenv("SGL_BENCH_SEQ", "2048"))
    warmup = int(os.getenv("SGL_BENCH_WARMUP", "5"))
    repeat = int(os.getenv("SGL_BENCH_REPEAT", "10"))

    dtype = torch.float32
    device = "cuda"

    # Inputs
    torch.manual_seed(0)
    scores = torch.rand((seq_length, num_experts), dtype=dtype, device=device)
    bias = torch.rand((num_experts,), dtype=dtype, device=device)

    def run_once():
        # No shared experts here to focus on core selection cost
        moe_fused_gate(
            scores,
            bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            topk=topk,
            num_fused_shared_experts=0,
            routed_scaling_factor=1.0,
        )

    # Measure
    times_ms = _measure_time_ms(run_once, warmup=warmup, repeat=repeat)
    avg_ms = sum(times_ms) / len(times_ms)
    p50_ms = median(times_ms)
    p90_ms = sorted(times_ms)[int(0.9 * (len(times_ms) - 1))]

    # Basic sanity: returns correct shapes and dtype
    out, idx = moe_fused_gate(
        scores,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    assert out.shape == (seq_length, topk)
    assert idx.shape == (seq_length, topk)
    assert out.dtype == torch.float32
    assert idx.dtype in (torch.int32, torch.int64)

    # Emit a concise perf line (useful in local runs; harmless in CI)
    print(
        f"moe_fused_gate tiled perf | experts={num_experts} groups={num_expert_group} "
        f"topk_group={topk_group} topk={topk} | avg={avg_ms:.3f}ms p50={p50_ms:.3f}ms p90={p90_ms:.3f}ms"
    )


