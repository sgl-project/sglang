"""CA v2 per-algorithm sweep: every algo forced, plus the tuned picker and NCCL.

torchrun --standalone --nproc-per-node=N bench.py  (RESULT=<json path>, MODE=graph|eager|both)
Both modes time one CUDA-graph replay of ITERS back-to-back calls. Graph mode
captures under comm.capture() (graph heuristics + pointer table); eager
captures without it, so the eager dispatch path (staging copies) is timed on
the GPU without Python launch overhead. Max over ranks, median of REPEATS.
"""

import contextlib
import json
import logging
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

KB, MB = 1024, 1024 * 1024
SIZES_KB = [int(s) for s in os.environ.get("SIZES_KB", "").split(",") if s] or sorted(
    {
        64,
        128,
        192,
        256,
        384,
        512,
        640,
        768,
        896,
        1024,
        1280,
        1536,
        1792,
        2048,
        2304,
        2560,
        2816,
        3072,
        3584,
        4096,
        4608,
        5120,
        6144,
        7168,
        8192,
    }
)
MODES = {"both": ("graph", "eager")}.get(
    os.environ.get("MODE", "both"), (os.environ.get("MODE"),)
)
ITERS, REPEATS = 100, 9
DTYPE = torch.bfloat16


def init():
    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.runtime_context import get_parallel

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    device = torch.device("cuda", local_rank)
    cap = max(SIZES_KB) * KB
    auto = CustomAllReduceV2(coord.cpu_group, device)
    forced = CustomAllReduceV2(
        coord.cpu_group, device, max_pull_size=cap, max_push_size=cap
    )
    assert not auto.disabled and not forced.disabled
    nccl = dist.new_group(backend="nccl", device_id=device)
    return auto, forced, nccl, device


def main():
    from sglang.kernels.ops.communication.all_reduce import (
        AllReduceAlgo,
        custom_all_reduce,
    )

    auto, forced, nccl, device = init()
    rank, world = dist.get_rank(), dist.get_world_size()
    results = []
    out_path = Path(os.environ.get("RESULT", f"sweep_tp{world}.json"))

    def emit(record):
        results.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
            out_path.write_text(json.dumps(results, indent=1))

    def ranks_max(value):
        vals = [None] * world
        dist.all_gather_object(vals, value)
        return max(vals)

    def measure(mode, comm, fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with comm.capture() if mode == "graph" else contextlib.nullcontext():
            with torch.cuda.graph(graph):
                for _ in range(ITERS):
                    fn()
        run = graph.replay
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        samples = []
        for _ in range(REPEATS):
            dist.barrier()
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record()
            run()
            end.record()
            end.synchronize()
            samples.append(ranks_max(begin.elapsed_time(end) * 1000 / ITERS))
        return statistics.median(samples)

    def forced_algo(algo):
        def fn(x):
            forced.override_algo = algo
            try:
                return forced.custom_all_reduce(x)
            finally:
                forced.override_algo = None

        return fn

    def mc_2shot(x):
        return custom_all_reduce(
            forced.obj, x, AllReduceAlgo.TWO_SHOT_PULL, use_multicast=True
        )

    def nccl_ar(x):
        dist.all_reduce(x, group=nccl)
        return x

    arms = {
        "nccl": (None, nccl_ar),
        "v2_auto": (auto, auto.custom_all_reduce),
        "1shot_push": (forced, forced_algo(AllReduceAlgo.ONE_SHOT_PUSH)),
        "2shot_push": (forced, forced_algo(AllReduceAlgo.TWO_SHOT_PUSH)),
        "1shot_pull": (forced, forced_algo(AllReduceAlgo.ONE_SHOT_PULL)),
        "2shot_pull": (forced, forced_algo(AllReduceAlgo.TWO_SHOT_PULL)),
    }
    if forced.has_multicast:
        # tuned tables leave mc off at small world sizes; measure it anyway
        forced.obj.set_pull_multicast_blocks(
            forced.config.num_mc_blocks or int(os.environ.get("MC_BLOCKS", 64))
        )
        arms["2shot_pull_mc"] = (forced, mc_2shot)

    for size_kb in SIZES_KB:
        numel = size_kb * KB // DTYPE.itemsize
        g = torch.Generator(device=device).manual_seed(1000 + size_kb * 7 + rank)
        x = torch.randn(numel, dtype=DTYPE, device=device, generator=g)
        ref = x.clone()
        dist.all_reduce(ref, group=nccl)
        outs = {}
        for name, (comm, fn) in arms.items():
            if name != "nccl":
                outs[name] = fn(x).clone()
        torch.cuda.synchronize()
        base = outs["1shot_push"]
        for name, out in outs.items():
            stats = torch.tensor(
                [
                    (out.float() - ref.float()).abs().max().item(),
                    float(not torch.equal(out, base)),
                ],
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=nccl)
            emit(
                dict(
                    check=name,
                    size_kb=size_kb,
                    max_abs_vs_nccl=stats[0].item(),
                    differs_from_1shot_push=bool(stats[1].item()),
                )
            )
        auto_algo = auto._pick_config(numel * DTYPE.itemsize, can_use_graph=False)
        for mode in MODES:
            graph_algo = auto._pick_config(numel * DTYPE.itemsize, can_use_graph=True)
            for name, (comm, fn) in arms.items():
                work = x.clone()
                try:
                    us = measure(mode, comm or auto, lambda fn=fn, work=work: fn(work))
                    rec = dict(
                        world=world,
                        mode=mode,
                        size_kb=size_kb,
                        name=name,
                        us=round(us, 3),
                    )
                    if name == "v2_auto":
                        picked = graph_algo if mode == "graph" else auto_algo
                        rec["picked"] = (
                            None
                            if picked is None
                            else picked.algo.algo_name
                            + ("+mc" if picked.use_multicast else "")
                        )
                    emit(rec)
                except Exception as e:
                    torch.cuda.synchronize()
                    emit(
                        dict(
                            world=world,
                            mode=mode,
                            size_kb=size_kb,
                            name=name,
                            error=repr(e)[:300],
                        )
                    )
    dist.barrier()


if __name__ == "__main__":
    main()
