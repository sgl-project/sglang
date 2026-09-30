"""Alternate MORI EPv2 CUDA-graph tiers with changing routing and lifecycle."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from mori.cco import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp


def _sync(comm):
    torch.cuda.synchronize()
    comm.barrier()


def _make_op(comm, rank, max_tokens):
    return EpDispatchCombineOp(
        EpDispatchCombineConfig(
            rank=rank,
            world_size=8,
            hidden_dim=7168,
            max_num_inp_token_per_rank=max_tokens,
            num_experts_per_rank=48,
            num_experts_per_token=6,
            data_type=torch.bfloat16,
        ),
        comm,
    )


def _capture_tier(op, comm, count, rank):
    generator = torch.Generator(device="cpu").manual_seed(20260803 + rank + count)
    inp = torch.randn(count, 7168, generator=generator, dtype=torch.bfloat16).cuda()
    indices = torch.randint(
        0, 384, (count, 6), generator=generator, dtype=torch.int32
    ).cuda()
    weights = torch.rand(count, 6, generator=generator, dtype=torch.float32).cuda()

    for _ in range(2):
        recv, _, _, recv_indices, _, routing = op.dispatch(
            inp, weights, None, indices, return_routing=True
        )
        op.combine(recv, None, recv_indices, routing=routing)
        _sync(comm)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recv, _, _, recv_indices, _, routing = op.dispatch(
            inp, weights, None, indices, return_routing=True
        )
        output, _ = op.combine(recv, None, recv_indices, routing=routing)
    _sync(comm)
    return {
        "count": count,
        "input": inp,
        "indices": indices,
        "weights": weights,
        "graph": graph,
        "output": output,
        "routing": routing,
    }


def _mutate_and_replay(tier, iteration, comm):
    count = tier["count"]
    inp = tier["input"]
    indices = tier["indices"]
    weights = tier["weights"]
    value = ((iteration % 13) + 1) / 16
    weight_value = ((iteration % 7) + 1) / 8
    inp.fill_(value)
    slots = torch.arange(6, device=indices.device, dtype=torch.int32)
    peers = (slots + iteration) % 8
    experts = peers * 48 + ((slots + iteration) % 48)
    indices.copy_(experts.view(1, 6).expand(count, 6))
    weights.fill_(weight_value)
    tier["graph"].replay()
    _sync(comm)

    expected = torch.full(
        (count, 7168),
        6 * value,
        dtype=torch.bfloat16,
        device=inp.device,
    )
    hidden_ok = torch.allclose(
        tier["output"].float(), expected.float(), atol=2e-2, rtol=2e-2
    )
    return hidden_ok


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    uid = Communicator.get_unique_id() if rank == 0 else None
    values = [uid]
    dist.broadcast_object_list(values, src=0)
    comm = Communicator.init(8, rank, values[0], per_rank_vmm=4 << 30)

    tier_counts = (8, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
    op = _make_op(comm, rank, max(tier_counts))
    _sync(comm)
    tiers = {count: _capture_tier(op, comm, count, rank) for count in tier_counts}

    order = (8, 8192, 128, 4096, 64, 2048, 256, 1024, 512)
    local_failures = 0
    first_failure = None
    iteration = 0
    for _ in range(100):
        for count in order:
            iteration += 1
            if not _mutate_and_replay(tiers[count], iteration, comm):
                local_failures += 1
                if first_failure is None:
                    first_failure = (iteration, count)

    summaries = [None] * 8
    dist.all_gather_object(
        summaries,
        {
            "rank": rank,
            "failures": local_failures,
            "first_failure": first_failure,
        },
    )
    if rank == 0:
        print(f"# EPV2-GRAPH-TIERS {summaries}", flush=True)
    ok = all(item["failures"] == 0 for item in summaries)

    # Graph and routing objects must be released before their shared op.
    tiers.clear()
    torch.cuda.synchronize()
    op.close()
    _sync(comm)

    lifecycle_failures = 0
    for lifecycle in range(10):
        small_op = _make_op(comm, rank, 128)
        _sync(comm)
        tier = _capture_tier(small_op, comm, 128, rank)
        for replay in range(10):
            if not _mutate_and_replay(tier, lifecycle * 10 + replay + 1, comm):
                lifecycle_failures += 1
        del tier
        torch.cuda.synchronize()
        small_op.close()
        _sync(comm)

    lifecycle_summaries = [None] * 8
    dist.all_gather_object(
        lifecycle_summaries,
        {"rank": rank, "failures": lifecycle_failures},
    )
    if rank == 0:
        print(f"# EPV2-GRAPH-LIFECYCLE {lifecycle_summaries}", flush=True)
    ok &= all(item["failures"] == 0 for item in lifecycle_summaries)

    comm.destroy()
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
