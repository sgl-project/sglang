# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 Ulysses exchange: the field-major row->head all-to-all and its
inverse plus the head merge, checked against plain slicing on 2 GPUs."""

import torch
import torch.distributed as dist

from sglang.test.test_utils import run_distributed_test


def _check(rank: int) -> None:
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn_attention import (
        _vdn_a2a_heads_to_rows,
        _vdn_a2a_rows_to_heads,
        _vdn_merge_heads,
    )

    world = dist.get_world_size()
    device = torch.device("cuda", rank)
    heads, head_dim, local_rows = 6, 32, 24
    local_heads = heads // world
    seq = local_rows * world
    # every rank builds the same global tensors, so a shard is checked by slicing
    g = torch.Generator(device="cpu").manual_seed(3)
    qkv = torch.randn(seq, 3 * heads * head_dim, generator=g).to(device, torch.bfloat16)
    q = qkv.view(seq, 3, heads, head_dim)[:, 0]  # strided, as the split qkv is
    scalars = torch.randn(seq, heads, 2, generator=g).to(device, torch.bfloat16)
    rows = slice(rank * local_rows, (rank + 1) * local_rows)
    mine = slice(rank * local_heads, (rank + 1) * local_heads)
    with torch.inference_mode():
        for name, field in (("q", q), ("scalars", scalars)):
            work, recv = _vdn_a2a_rows_to_heads(
                field[rows], ulysses_ws=world, role=name, process_group=dist.group.WORLD
            )
            work.wait()
            assert recv.is_contiguous()
            assert torch.equal(recv, field[:, mine])
        # inverse: this rank's heads for every row -> row shard, every head
        work, back = _vdn_a2a_heads_to_rows(
            q[:, mine].contiguous() * 2,
            ulysses_ws=world,
            role="out",
            process_group=dist.group.WORLD,
        )
        work.wait()
        assert torch.equal(_vdn_merge_heads(back), q[rows] * 2)
    torch.cuda.synchronize()


def test_exchange_round_trip_two_ranks() -> None:
    run_distributed_test(_check, world_size=2)


if __name__ == "__main__":
    test_exchange_round_trip_two_ranks()
