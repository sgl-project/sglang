"""Latency sweep for tuning the custom all-reduce (v2) crossovers.

Times every algo our own kernel can run, plus NCCL as the fallback reference,
at each message size. The winner per size is what
``config/custom_all_reduce_v2_tuning.py`` encodes, so a row of this table maps
straight onto a tuning entry. Latency only -- a crossover is a latency
question, and bandwidth is the same number rescaled by a constant.

Columns are the forced dispatch, not the tuned one: ``force_algo`` bypasses the
band table entirely, so every algo is measured at every size, including sizes
where the current table would hand back to NCCL.

``.g`` runs the pull kernel against the CUDA-graph pointer table instead of the
eager staging buffer; ``.mc`` reduces over the multicast VA. 1shot_push writes
into the symmetric push slots and multicast reads the mc VA, so neither
consumes a graph row and neither gets a ``.g`` column.

Run locally (needs >= 2 GPUs):

    python test/manual/kernels/tune_custom_all_reduce.py
    python test/manual/kernels/tune_custom_all_reduce.py --num-gpu 4,8
"""

from __future__ import annotations

import atexit
import os
from typing import NamedTuple

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import multigpu_bench_main
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

DTYPE = torch.bfloat16
MESSAGE_SIZES_KB = [2**x for x in range(2, 20)]
MESSAGE_SIZES_KB += [192, 384, 640, 768, 896, 1536, 3072]
MESSAGE_SIZES_KB.sort()
MAX_BYTES = max(MESSAGE_SIZES_KB) * 1024
MAX_1_SHOT_BYTES = min(MAX_BYTES, 64 * 1024 * 1024)  # clip to 64MB
WORLD_SIZES = list(range(2, 9)) + [16]


class Variant(NamedTuple):
    algo: AllReduceAlgo
    use_multicast: bool
    in_graph: bool


VARIANTS = {
    "1shot_push": Variant(AllReduceAlgo.ONE_SHOT_PUSH, False, False),
    "1shot_pull": Variant(AllReduceAlgo.ONE_SHOT_PULL, False, False),
    "1shot_pull.g": Variant(AllReduceAlgo.ONE_SHOT_PULL, False, True),
    "2shot_pull": Variant(AllReduceAlgo.TWO_SHOT_PULL, False, False),
    "2shot_pull.g": Variant(AllReduceAlgo.TWO_SHOT_PULL, False, True),
    "2shot_pull.mc": Variant(AllReduceAlgo.TWO_SHOT_PULL, True, False),
}
PROVIDERS = ["nccl", *VARIANTS]


@cache_once
def _init_cpu_group() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_nccl_group() -> dist.ProcessGroup:
    _init_cpu_group()
    local_rank = int(os.environ["LOCAL_RANK"])
    group = dist.new_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert isinstance(group, dist.ProcessGroup)
    return group


@cache_once
def _init_comm() -> CustomAllReduceV2:
    """A communicator every algo can run at every sweep size.

    Both workspaces are sized to the sweep maximum rather than to the tuned
    crossovers, since the point is to measure past them. The push half costs
    ``2 * world_size`` slots of that size, so a large sweep on a large world is
    the memory-hungry case; shrink ``MESSAGE_SIZES_KB`` if that is a problem.
    """
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = CustomAllReduceV2(
        _init_cpu_group(),
        device,
        max_push_size=MAX_1_SHOT_BYTES,
        max_pull_size=MAX_BYTES,
    )
    if comm.disabled:
        raise RuntimeError("CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


@marker.parametrize("message_KB", MESSAGE_SIZES_KB)
@marker.benchmark("provider", PROVIDERS)
def benchmark(message_KB: int, provider: str):
    gpu_group = _init_nccl_group()
    comm = _init_comm()  # built even for nccl, so setup never lands in a cell
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    x = torch.randn(message_KB * 1024 // DTYPE.itemsize, dtype=DTYPE, device=device)

    if message_KB * 1024 > MAX_1_SHOT_BYTES and provider.startswith("1shot"):
        marker.skip("1shot is too slow for large messages, skip")

    if provider == "nccl":
        fn = lambda t: dist.all_reduce(t, group=gpu_group)
        ctx_fn = None
    else:
        variant = VARIANTS[provider]
        if variant.use_multicast and not comm.has_multicast:
            marker.skip("symmetric memory on this group has no multicast VA")
        comm.force_algo(variant.algo, variant.use_multicast)
        fn = comm.custom_all_reduce
        ctx_fn = comm.capture if variant.in_graph else None

    return marker.do_bench(
        fn,
        input_args=(x,),
        graph_context_fn=ctx_fn,
        sync_multigpu_fn=lambda: dist.barrier(gpu_group),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=WORLD_SIZES,
        main_fn=benchmark.run,
    )
