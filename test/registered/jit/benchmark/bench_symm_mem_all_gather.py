"""Benchmark the symmetric-memory multimem all-gather vs NCCL.

Providers:
- ``nccl``        : ``all_gather_into_tensor`` + concat-along-hidden reshape
                    (what ``tensor_model_parallel_all_gather(dim=-1)`` does)
- ``mm_safe``     : multimem kernel, ``safe=True`` (clones the buffer view)
- ``mm``          : multimem kernel, ``safe=False`` (fc gather config)
- ``mm_skipsync`` : multimem kernel, ``safe=False, skip_entry_sync=True``
                    (logits gather config)

Usage::

    # Benchmark on the default world sizes (2, 4, 8 GPUs):
    python test/registered/jit/benchmark/bench_symm_mem_all_gather.py
    # Pick a specific world size (or comma-separated list):
    python test/registered/jit/benchmark/bench_symm_mem_all_gather.py --num-gpu 8
"""

from __future__ import annotations

import atexit
import logging
import os

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import get_benchmark_range, multigpu_bench_main
from sglang.kernels.jit import cache_once
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    all_gather_inner,
    create_state,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)
register_amd_ci(est_time=120, stage="jit-kernel-benchmark", runner_config="amd")

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

DTYPE = torch.bfloat16
PROVIDERS = ["nccl", "mm_safe", "mm", "mm_skipsync"]
# Full gathered hidden width H (per-rank shard is H / world_size).
HIDDENS = [7168, 16384, 163840]
NUM_TOKENS = [1, 8, 16, 32, 64, 128]
WORLD_SIZES = list(range(2, 9))

HIDDENS = get_benchmark_range(HIDDENS, [7168, 163840])
NUM_TOKENS = get_benchmark_range(NUM_TOKENS, [16, 64])
WORLD_SIZES = get_benchmark_range(WORLD_SIZES, [2, 4, 8])

MAX_HIDDEN = max(HIDDENS)
MAX_TOKENS = max(NUM_TOKENS)

# ---------------------------------------------------------------------------
# Per-rank distributed init (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return ps._WORLD.cpu_group


@cache_once
def _init_nccl_group() -> dist.ProcessGroup:
    _init_cpu_group()
    coord = ps._WORLD
    assert coord is not None and coord.device_group is not None
    return coord.device_group


@cache_once
def _init_state():
    _init_cpu_group()
    coord = ps._WORLD
    return create_state(
        group=coord.device_group,
        rank_in_group=coord.rank_in_group,
        max_tokens=MAX_TOKENS,
        hidden_size=MAX_HIDDEN,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@marker.parametrize("hidden", HIDDENS)
@marker.parametrize("num_tokens", NUM_TOKENS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(num_tokens: int, hidden: int, provider: str):
    gpu_group = _init_nccl_group()
    state = _init_state()
    world_size = state.world_size
    local_hidden = hidden // world_size
    if hidden % world_size != 0 or local_hidden % 8 != 0:
        marker.skip(f"hidden={hidden} incompatible with world_size={world_size}")
    if provider != "nccl" and state.symm_mem_hdl.multicast_ptr == 0:
        marker.skip(f"multimem multicast unavailable for world_size={world_size}")

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    x = torch.randn(num_tokens, local_hidden, dtype=DTYPE, device=device)

    if provider == "nccl":
        out_buf = torch.empty(
            world_size * num_tokens, local_hidden, dtype=DTYPE, device=device
        )

        def fn(inp: torch.Tensor) -> torch.Tensor:
            dist.all_gather_into_tensor(out_buf, inp, group=gpu_group)
            return (
                out_buf.reshape(world_size, num_tokens, local_hidden)
                .movedim(0, 1)
                .reshape(num_tokens, hidden)
            )

    else:
        safe = provider == "mm_safe"
        skip_entry_sync = provider == "mm_skipsync"

        def fn(inp: torch.Tensor) -> torch.Tensor:
            return all_gather_inner(
                state,
                inp,
                tp_hidden_dim=hidden,
                skip_entry_sync=skip_entry_sync,
                safe=safe,
            )

    return marker.do_bench(
        fn,
        input_args=(x,),
        graph_clone_args=(0,),
        sync_multigpu_fn=lambda: dist.barrier(gpu_group),
        # Footprint = the gathered output every rank ends up with.
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=num_tokens * hidden * x.element_size(),
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=WORLD_SIZES,
        main_fn=benchmark.run,
    )
