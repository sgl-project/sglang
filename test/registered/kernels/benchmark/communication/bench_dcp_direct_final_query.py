"""Benchmark Kimi-K3 DCP Query AllGather against direct-final NVLS.

The benchmark isolates Query publication after local Q production. It covers:

- ``source``: six local heads on DCP4, matching the prior vLLM Kimi-K3 study;
- ``tp4``: 24 local heads on DCP4, matching this four-GPU SGLang target.

Usage::

    python test/registered/kernels/benchmark/communication/bench_dcp_direct_final_query.py \
      --num-gpu 4
"""

from __future__ import annotations

import atexit
import functools
import logging
import os

import sglang.srt.distributed.parallel_state as ps
import torch
import torch.distributed as dist
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import (
    get_benchmark_range,
    multigpu_bench_main,
)
from sglang.srt.layers.dcp.query import DCPDirectFinalQueryGatherer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires four GPUs and self-skips in CI",
)

NOPE_DIM = 512
ROPE_DIM = 64
LOCAL_HEADS = get_benchmark_range([6, 24], [6, 24])
NUM_TOKENS = get_benchmark_range([1, 2, 4, 8, 16, 17, 32, 128], [1, 17, 128])
PROVIDERS = ["allgather", "direct_final"]
MAX_TOKENS = max(NUM_TOKENS)


@functools.cache
def _init_world() -> ps.GroupCoordinator:
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
    assert ps._WORLD is not None
    return ps._WORLD


@functools.cache
def _gatherer(local_heads: int) -> DCPDirectFinalQueryGatherer:
    world = _init_world()
    return DCPDirectFinalQueryGatherer(
        group=world,
        max_tokens=MAX_TOKENS,
        local_heads=local_heads,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        device=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"),
    )


def _allgather(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce all_gather_q_for_mla_decode, including its local pack."""
    world = _init_world()
    combined = torch.cat((q_rope.transpose(0, 1), q_nope.transpose(0, 1)), dim=-1)
    gathered = world.all_gather(combined, dim=0)
    final_rope, final_nope = gathered.split((ROPE_DIM, NOPE_DIM), dim=-1)
    return final_nope.transpose(0, 1), final_rope.transpose(0, 1)


@marker.parametrize("local_heads", LOCAL_HEADS)
@marker.parametrize("num_tokens", NUM_TOKENS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(num_tokens: int, local_heads: int, provider: str):
    world = _init_world()
    if world.world_size != 4:
        marker.skip("Kimi-K3 source and TP4/DCP4 geometries require four ranks.")

    gatherer = _gatherer(local_heads)
    if provider == "direct_final" and gatherer.state.symm_mem_hdl.multicast_ptr == 0:
        marker.skip("NVLS multicast is unavailable.")

    device = gatherer.state.device
    q_nope_storage = torch.randn(
        local_heads,
        num_tokens,
        NOPE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    q_nope = q_nope_storage.transpose(0, 1)
    q_rope_storage = torch.randn(
        num_tokens,
        local_heads,
        128 + ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    q_rope = q_rope_storage[..., -ROPE_DIM:]

    if provider == "allgather":

        def fn(nope: torch.Tensor, rope: torch.Tensor):
            return _allgather(nope, rope)

    else:

        def fn(nope: torch.Tensor, rope: torch.Tensor):
            return gatherer(nope, rope)

    return marker.do_bench(
        fn,
        input_args=(q_nope, q_rope),
        graph_clone_args=(0, 1),
        sync_multigpu_fn=lambda: dist.barrier(world.device_group),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=(
            num_tokens
            * local_heads
            * world.world_size
            * (NOPE_DIM + ROPE_DIM)
            * q_nope.element_size()
        ),
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=(4,),
        main_fn=benchmark.run,
    )
