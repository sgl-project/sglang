# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 Ulysses exchange: the field-major row->head all-to-all and its
inverse plus the head merge, checked against plain slicing on 2 GPUs."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world: int, port: int, result_path: str) -> None:
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _vdn_a2a_heads_to_rows,
        _vdn_a2a_rows_to_heads,
        _vdn_merge_heads,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    device = torch.device("cuda", rank)
    heads, head_dim, local_rows = 6, 32, 24
    local_heads = heads // world
    seq = local_rows * world
    # every rank builds the same global tensors and exchanges its row shard, so
    # the result can be checked against slicing the global tensor
    g = torch.Generator(device="cpu").manual_seed(3)
    qkv = torch.randn(seq, 3 * heads * head_dim, generator=g).to(device, torch.bfloat16)
    q = qkv.view(seq, 3, heads, head_dim)[:, 0]  # strided, as the split qkv is
    rows = slice(rank * local_rows, (rank + 1) * local_rows)
    ok = True
    with torch.inference_mode():
        work, recv = _vdn_a2a_rows_to_heads(
            q[rows], ulysses_ws=world, role="test_q", process_group=dist.group.WORLD
        )
        work.wait()
        mine = slice(rank * local_heads, (rank + 1) * local_heads)
        ok &= recv.is_contiguous() and torch.equal(recv, q[:, mine])
        # inverse: this rank's heads for every row -> row shard, every head
        work, back = _vdn_a2a_heads_to_rows(
            q[:, mine].contiguous() * 2,
            ulysses_ws=world,
            role="test_out",
            process_group=dist.group.WORLD,
        )
        work.wait()
        ok &= torch.equal(_vdn_merge_heads(back), q[rows] * 2)
    torch.cuda.synchronize()
    with open(f"{result_path}.{rank}", "w") as f:
        f.write("ok" if ok else "fail")
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_exchange_round_trip_two_ranks(tmp_path) -> None:
    result = str(tmp_path / "result")
    port = 29000 + (os.getpid() % 1000)
    mp.spawn(_worker, args=(2, port, result), nprocs=2, join=True)
    for rank in range(2):
        with open(f"{result}.{rank}") as f:
            assert f.read() == "ok", f"rank {rank} exchange mismatch"
