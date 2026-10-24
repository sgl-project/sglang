"""The deterministic custom all-reduce on ROCm must sum every element over the ranks in
the order 0..world_size-1, whatever the input size: it switches from the 1-stage to the
2-stage kernel above a size threshold, and both must give exactly that sum. Checked bit
for bit against the same sum computed with torch, at 4 and 8 ranks, in eager mode and in
CUDA graphs, with data whose float32 sum depends on the order of the ranks.
"""

import multiprocessing as mp
import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd-mi35x")

HIDDEN = 4096
# Rows of HIDDEN bf16 values; 32 rows (256 KiB) and 64 rows (512 KiB) are the 2-stage
# thresholds at 8 and 4 ranks. The largest size stays under the 16 MiB buffer.
EAGER_ROWS = (1, 7, 31, 32, 33, 63, 64, 65, 128, 500, 1024, 2047)
GRAPH_ROWS = (1, 16, 32, 64, 256)


def _open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _order_sensitive(rows, dtype, rank):
    # Values of magnitude 2^12, 1 and 2^-12 with random signs: their float32 sum over the
    # ranks rounds differently in a different order (randn data would hide that).
    g = torch.Generator().manual_seed(1000 * rank + rows)
    exp = torch.tensor([12.0, 0.0, -12.0])[
        torch.randint(0, 3, (rows, HIDDEN), generator=g)
    ]
    sign = torch.randint(0, 2, (rows, HIDDEN), generator=g) * 2 - 1
    return (sign * exp.exp2() * (1 + torch.rand(rows, HIDDEN, generator=g))).to(dtype)


def _fixed_order_sum(x, world_size):
    parts = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(parts, x)
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(x.dtype)


def _worker(world_size, rank, port, failures):
    try:
        os.environ["SGLANG_USE_1STAGE_ALLREDUCE"] = (
            "1"  # selects the deterministic path
        )
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        car = CustomAllreduce(
            dist.new_group(backend="gloo"), torch.device("cuda", rank)
        )
        assert not car.disabled and car.use_amd_deterministic_impl
        for dtype in (torch.bfloat16, torch.float16):
            for rows in EAGER_ROWS:
                x = _order_sensitive(rows, dtype, rank).cuda()
                out = car.custom_all_reduce(x)
                ref = _fixed_order_sum(x, world_size)
                if not torch.equal(out, ref):
                    failures.put(
                        f"{world_size} ranks, rank {rank}, {dtype}, eager, {rows} rows"
                    )
            with car.capture():
                graphs = {}
                for rows in GRAPH_ROWS:
                    static_in = torch.zeros(rows, HIDDEN, dtype=dtype, device="cuda")
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_out = car.custom_all_reduce(static_in)
                    graphs[rows] = (graph, static_in, static_out)
            for rows, (graph, static_in, static_out) in graphs.items():
                x = _order_sensitive(rows, dtype, rank).cuda()
                static_in.copy_(x)
                graph.replay()
                torch.cuda.synchronize()
                if not torch.equal(static_out, _fixed_order_sum(x, world_size)):
                    failures.put(
                        f"{world_size} ranks, rank {rank}, {dtype}, graph, {rows} rows"
                    )
        car.close()
        dist.destroy_process_group()
    except Exception:
        failures.put(f"rank {rank}: {traceback.format_exc()}")


@pytest.mark.skipif(not is_hip(), reason="ROCm deterministic all-reduce")
@pytest.mark.parametrize("world_size", [4, 8])
def test_fixed_rank_order_at_every_size(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    ctx = mp.get_context("spawn")
    failures = ctx.Queue()
    port = _open_port()
    procs = [
        ctx.Process(target=_worker, args=(world_size, rank, port, failures))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=600)
    found = []
    while not failures.empty():
        found.append(failures.get())
    assert all(p.exitcode == 0 for p in procs), found
    assert not found, "\n".join(found)
