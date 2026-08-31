"""Sweep the K3 SP-MoE collective strategies on the local device.

Produces the per-device tuning table consumed by
``sglang.kernels.ops.kimi_k3.sp_collective.get_dispatch`` (the
``configs/sp_collective/world=W,H=H,device_name=D.json`` schema): for every
global-token bucket, times NCCL against the multicast push/pull kernels over
the kernel-supported launch-shape grid and records the winner. Fusion sections
(``reduce_scatter_attn_res`` / ``attn_res_all_gather``) are deliberately left
out — absent sections fall back to the separate path, which is the safe
default until a fusion sweep is run.

Every candidate is captured into a CUDA graph under ``comm.capture()`` and
timed via graph replays. That mirrors how production launches these kernels
(decode CUDA graphs) and keeps the push-workspace flag lifecycle bounded —
free-running eager loops exhaust the staging ring and spin-deadlock.

Launch (one node, all GPUs of the serving TP group):

    torchrun --nproc-per-node 8 scripts/bench_sp_collective_sweep.py \
        --output-dir /tmp/sp-sweep

Correctness is asserted against NCCL per (strategy, bucket, shape); a
mismatching or crashing configuration is disqualified rather than tabulated.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
from datetime import datetime, timezone

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.kimi_k3 import all_reduce, sp_collective
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

_HIDDEN = 7168
_MB = 1024 * 1024
_BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
_WARMUP = 10
_ITERS = 50
_PUSH_SLOT_BYTES = 64 * _MB

# Launch-shape grids per (kind, strategy), constrained to the shapes the
# kernels are known to support (superset of every entry in the shipped GB300
# table). Free combinations outside these shapes can spin-deadlock.
_GRIDS = {
    ("rs", "push"): [(nb, 512) for nb in (4, 8, 16, 32, 64, 96, 128)],
    # NVLS pull RS deadlocks when driven from a bare _SymmetricMemory input in
    # this standalone harness (its input contract is tied to the serving-side
    # named symm-buffer pool); left unswept — absent entries fall back to
    # push/nccl, which is still strictly better than no table at all.
    ("rs", "pull"): [],
    ("ag", "push"): [(nb, bs) for nb in (16, 32, 64, 128) for bs in (128, 256, 512)],
    ("ag", "direct"): [(nb, 1024) for nb in (1, 2, 4, 8)],
}


def _device() -> torch.device:
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


def _init_world():
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
    # Graph capture requires a non-default stream.
    torch.cuda.set_stream(torch.cuda.Stream())
    nccl_group = dist.new_group(backend="nccl", device_id=_device())
    return coord.cpu_group, nccl_group


def _init_comm(cpu_group) -> CustomAllReduceV2:
    comm = CustomAllReduceV2(
        cpu_group,
        _device(),
        max_pull_size=_PUSH_SLOT_BYTES,
        max_push_size=_PUSH_SLOT_BYTES,
    )
    if comm.disabled or not comm.has_multicast:
        raise RuntimeError("SP collective sweep requires multicast symmetric memory")
    all_reduce.register_comm(comm.obj)
    sp_collective.register_comm(comm.obj)
    register_comm_cleanup(comm)
    return comm


def _symmetric_tensor(shape, cpu_group):
    from torch._C._distributed_c10d import _SymmetricMemory

    tensor = _SymmetricMemory.empty_strided_p2p(
        shape,
        torch.empty(shape).stride(),
        torch.bfloat16,
        _device(),
        cpu_group.group_name,
    )
    handle = _SymmetricMemory.rendezvous(tensor)
    rank = dist.get_rank()
    multicast_ptr = (
        int(handle.multicast_ptr) + tensor.data_ptr() - int(handle.buffer_ptrs[rank])
    )
    if multicast_ptr == 0:
        raise RuntimeError("symmetric tensor has no multicast mapping")
    return tensor, handle, multicast_ptr


def _assert_close(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=3e-2)


class _Bencher:
    def __init__(self, comm, nccl_group):
        self.comm = comm
        self.nccl_group = nccl_group

    def run(self, fn, *, use_comm_capture: bool, check=None) -> float:
        """Capture fn into a CUDA graph, verify once, time replays (µs).

        check: optional (actual, expected) verified after one replay.
        Capture/verify failures are synchronized across ranks so a
        single-rank failure cannot desynchronize the collective sequence.
        """
        failure = None
        try:
            ctx = self.comm.capture() if use_comm_capture else contextlib.nullcontext()
            with ctx:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
            graph.replay()
            torch.cuda.synchronize()
            if check is not None:
                _assert_close(check[0], check[1])
        except Exception as exc:
            failure = exc
        ok = torch.tensor([0.0 if failure is not None else 1.0], device=_device())
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=self.nccl_group)
        if ok.item() == 0.0:
            raise failure or RuntimeError("config failed on a peer rank")
        samples = []
        for _ in range(3):
            for _ in range(_WARMUP):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_ITERS):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            local_us = start.elapsed_time(end) * 1000.0 / _ITERS
            t = torch.tensor([local_us], device=_device())
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.nccl_group)
            samples.append(t.item())
        return sorted(samples)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cpu_group, nccl_group = _init_world()
    comm = _init_comm(cpu_group)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = _device()
    device_name = torch.cuda.get_device_name(device)
    bench = _Bencher(comm, nccl_group)

    max_t = max(_BUCKETS)
    gen = torch.Generator(device="cuda").manual_seed(100 + rank)
    # Persistent symmetric buffer: pull-RS input / direct-AG output.
    symm_full, _handle, symm_mc = _symmetric_tensor((max_t, _HIDDEN), cpu_group)
    plain_full = torch.randn(
        max_t, _HIDDEN, generator=gen, device=device, dtype=torch.bfloat16
    )
    symm_full.copy_(plain_full)

    results = {"reduce_scatter": {}, "all_gather": {}}
    raw: dict = {"device": device_name, "world": world, "buckets": {}}

    def _log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    for tokens in _BUCKETS:
        local = tokens // world
        if local == 0:
            continue
        rows = raw["buckets"][str(tokens)] = {}
        rs_in = plain_full[:tokens]
        rs_symm_in = symm_full[:tokens]
        residual = plain_full[:local].clone()
        rs_out = torch.empty(local, _HIDDEN, device=device, dtype=torch.bfloat16)

        # NCCL baseline: reduce-scatter plus the residual add the fused
        # kernels fold in.
        ref = torch.empty_like(rs_out)

        def nccl_rs():
            dist.reduce_scatter_tensor(ref, rs_in, group=nccl_group)
            ref.add_(residual)

        _log(f"try rs nccl T={tokens}")
        rows["rs_nccl"] = bench.run(nccl_rs, use_comm_capture=False)
        rs_ref = ref.clone()

        best_rs = ("nccl", None, rows["rs_nccl"])
        for strategy in ("push", "pull"):
            if strategy == "push" and tokens * _HIDDEN * 2 > _PUSH_SLOT_BYTES:
                continue
            for nb, bs in _GRIDS[("rs", strategy)]:
                _log(f"try rs {strategy} T={tokens} nb={nb} bs={bs}")
                tuning = sp_collective.Tuning(num_blocks=nb, block_size=bs)
                if strategy == "push":
                    fn = lambda: sp_collective.reduce_scatter_res(
                        world, rs_in, rs_out, residual, tuning=tuning
                    )
                else:
                    fn = lambda: sp_collective.reduce_scatter_pull(
                        world,
                        rs_symm_in,
                        rs_out,
                        residual,
                        input_mc_ptr=symm_mc,
                        tuning=tuning,
                    )
                try:
                    us = bench.run(fn, use_comm_capture=True, check=(rs_out, rs_ref))
                except Exception as exc:  # disqualify, keep sweeping
                    rows[f"rs_{strategy}_{nb}x{bs}"] = f"failed: {exc}"[:120]
                    continue
                rows[f"rs_{strategy}_{nb}x{bs}"] = us
                if us < best_rs[2]:
                    best_rs = (strategy, tuning, us)

        results["reduce_scatter"][str(tokens)] = (
            {"strategy": "nccl"}
            if best_rs[0] == "nccl"
            else {
                "strategy": best_rs[0],
                "num_blocks": best_rs[1].num_blocks,
                "block_size": best_rs[1].block_size,
            }
        )

        ag_in = plain_full[:local]
        ag_out = torch.empty(tokens, _HIDDEN, device=device, dtype=torch.bfloat16)
        ag_ref = torch.empty_like(ag_out)

        def nccl_ag():
            dist.all_gather_into_tensor(ag_ref, ag_in, group=nccl_group)

        _log(f"try ag nccl T={tokens}")
        rows["ag_nccl"] = bench.run(nccl_ag, use_comm_capture=False)
        ag_expected = ag_ref.clone()

        best_ag = ("nccl", None, rows["ag_nccl"])
        ag_symm_out = symm_full[:tokens]
        for strategy in ("push", "direct"):
            if strategy == "push" and tokens * _HIDDEN * 2 > _PUSH_SLOT_BYTES:
                continue
            for nb, bs in _GRIDS[("ag", strategy)]:
                _log(f"try ag {strategy} T={tokens} nb={nb} bs={bs}")
                tuning = sp_collective.Tuning(num_blocks=nb, block_size=bs)
                if strategy == "push":
                    fn = lambda: sp_collective.all_gather(
                        world, ag_in, ag_out, tuning=tuning
                    )
                    out_ref = ag_out
                else:
                    fn = lambda: sp_collective.all_gather_direct(
                        world,
                        ag_in,
                        ag_symm_out,
                        output_mc_ptr=symm_mc,
                        tuning=tuning,
                    )
                    out_ref = ag_symm_out
                try:
                    us = bench.run(
                        fn, use_comm_capture=True, check=(out_ref, ag_expected)
                    )
                except Exception as exc:
                    rows[f"ag_{strategy}_{nb}x{bs}"] = f"failed: {exc}"[:120]
                    continue
                rows[f"ag_{strategy}_{nb}x{bs}"] = us
                if us < best_ag[2]:
                    best_ag = (strategy, tuning, us)
        # Restore the shared symmetric buffer for the next bucket's pull input.
        symm_full.copy_(plain_full)

        results["all_gather"][str(tokens)] = (
            {"strategy": "nccl"}
            if best_ag[0] == "nccl"
            else {
                "strategy": best_ag[0],
                "num_blocks": best_ag[1].num_blocks,
                "block_size": best_ag[1].block_size,
            }
        )
        _log(
            f"T={tokens}: rs={results['reduce_scatter'][str(tokens)]} "
            f"ag={results['all_gather'][str(tokens)]}"
        )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        table = {
            "source": {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "device": device_name,
                "nodes": 1,
                "gpus_per_node": world,
                "push_slot_bytes": _PUSH_SLOT_BYTES,
                "raw_results": ["sp-collective-sweep-raw.json"],
                "method": "cuda-graph replay under comm.capture()",
            },
            "selection": {
                "rule": "nearest global-token bucket at or below the workload",
                "fallback": "nccl",
                "note": (
                    "Base reduce_scatter/all_gather sweep only; the fused "
                    "attn-res sections are intentionally absent and fall "
                    "back to the separate path."
                ),
            },
            "configs": results,
        }
        name = device_name.replace(" ", "_").replace("/", "_")
        table_path = os.path.join(
            args.output_dir, f"world={world},H={_HIDDEN},device_name={name}.json"
        )
        with open(table_path, "w") as f:
            json.dump(table, f, indent=2)
        with open(
            os.path.join(args.output_dir, "sp-collective-sweep-raw.json"), "w"
        ) as f:
            json.dump(raw, f, indent=2)
        print(f"TABLE {table_path}", flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
