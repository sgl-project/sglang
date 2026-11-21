# SPDX-License-Identifier: Apache-2.0
"""Prepared attention microbenchmark, same estimator with fusion off/on.

Run with H3_BENCH_RESULT=result.json and torchrun --nproc_per_node=4 or 8.
Requires SM120 and FlashInfer 0.7.4. This does not compare BF16 routing or full models.
"""

import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


@torch.inference_mode()
def main():
    from sglang.multimodal_gen.runtime.distributed import (
        sage_ulysses,
    )
    from sglang.multimodal_gen.runtime.distributed import sage_ulysses_route as route
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
        prepared_sage_attention as adapter,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
        SubBlockSparseSchedule,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world in (4, 8) and torch.cuda.get_device_capability() == (12, 0)
    used, heads, dim = 37765, 56, 128
    total = (used + world - 1) // world * world
    torch.manual_seed(1101 + rank)
    packed = torch.randn(
        total // world, heads, 3, dim, device="cuda", dtype=torch.bfloat16
    )
    q, k, v = packed.unbind(2)
    mask = torch.arange((used + 63) // 64, device="cuda") % 3 != 0
    schedule = SubBlockSparseSchedule(
        compute_mode="sage_fp8",
        quantize_before_a2a=True,
        sparsity=0.75,
        n_q=4,
        n_k=4,
        min_seq_len=0,
        skip_first_steps=10,
        skip_first_layers=0,
    )
    impl = SimpleNamespace(
        schedule=schedule,
        layer_enabled=True,
        _step_enabled=lambda: True,
        softmax_scale=dim**-0.5,
    )
    transport = sage_ulysses._PreparedSageTransport(dist.group.WORLD, q, used)
    try:
        with (
            patch.object(
                route,
                "get_sp_group",
                lambda: SimpleNamespace(ulysses_group=dist.group.WORLD),
            ),
            patch.object(route, "get_ulysses_parallel_world_size", lambda: world),
        ):

            def run(fused):
                os.environ["SGLANG_DIFFUSION_H3_SAGE_FUSED_PREPOST"] = str(fused)
                return adapter.prepared_subblock_sage_attention_forward(
                    q,
                    k,
                    v,
                    impl=impl,
                    cu_seqlens_host=(0, used, total),
                    max_seqlen=used,
                    ulysses_active=True,
                    ring_active=False,
                    query_block_mask=mask,
                    transport=transport,
                )

            reference = run(0).clone()
            actual = run(1).clone()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert torch.isfinite(actual).all()
            for fused in (0, 1):
                for _ in range(3):
                    run(fused)
            timings = {0: [], 1: []}
            iterations, trials = 10, 6
            for trial in range(trials):
                for fused in (0, 1) if trial % 2 == 0 else (1, 0):
                    os.environ["SGLANG_DIFFUSION_H3_SAGE_FUSED_PREPOST"] = str(fused)
                    torch.cuda.synchronize()
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(iterations):
                        run(fused)
                    end.record()
                    end.synchronize()
                    elapsed = torch.tensor(
                        start.elapsed_time(end) / iterations, device="cuda"
                    )
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    timings[fused].append(elapsed.item())
            if rank == 0:
                result = {
                    "scope": "NCCL prepared attention, fused vs unfused same quantized-operand estimator",
                    "world": world,
                    "used_sequence": used,
                    "padded_sequence": total,
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "heads": heads,
                    "head_dim": dim,
                    "sparsity": 0.75,
                    "n_q": 4,
                    "n_k": 4,
                    "dtype": "bfloat16",
                    "qkv_strides": q.stride(),
                    "mixed_query_mask": True,
                    "fused_unfused_exact": True,
                    "warmup_per_arm": 3,
                    "trials": trials,
                    "iterations_per_trial": iterations,
                    "timing": "CUDA events; maximum rank time per trial; alternating arm order",
                    "unfused_ms": timings[0],
                    "fused_ms": timings[1],
                    "unfused_median_ms": statistics.median(timings[0]),
                    "fused_median_ms": statistics.median(timings[1]),
                    "speedup": statistics.median(timings[0])
                    / statistics.median(timings[1]),
                    "full_model_quality_or_speed_claim": False,
                }
                out = Path(os.environ["H3_BENCH_RESULT"])
                out.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result), flush=True)
    finally:
        transport.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
