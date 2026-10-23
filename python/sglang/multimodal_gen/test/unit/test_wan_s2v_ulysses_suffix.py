# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V's minimal-copy Ulysses exchange, on two real ranks (gloo, CPU)."""

import os
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.models.dits import wan_s2v

WS, S_LOCAL, NUM_REP, HEADS, DIM = 2, 6, 3, 4, 8


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _round_trip_worker(rank, port, compiled):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WS)
    try:
        group = SimpleNamespace(ulysses_group=dist.group.WORLD)
        with (
            patch.object(wan_s2v, "get_ulysses_parallel_world_size", return_value=WS),
            patch.object(wan_s2v, "get_ulysses_parallel_rank", return_value=rank),
            patch.object(wan_s2v, "get_sp_group", return_value=group),
        ):
            torch.manual_seed(rank)
            shard = torch.randn(1, S_LOCAL, HEADS, DIM)
            torch.manual_seed(100)  # the suffix is replicated: identical on every rank
            suffix = torch.randn(1, NUM_REP, HEADS, DIM)
            x = torch.cat([shard, suffix], dim=1)
            y = wan_s2v.ulysses_input_a2a_with_suffix(x, NUM_REP)
            out_fn = wan_s2v.ulysses_output_a2a_with_suffix
            if compiled:
                out_fn = torch.compile(out_fn, backend="eager")
            z = out_fn(y, NUM_REP)
        assert torch.equal(z, x), f"rank {rank}: round trip changed the tokens"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_ulysses_suffix_exchange_round_trips_on_two_ranks(compiled):
    """The input exchange followed by the output exchange returns each rank's own tokens,
    replicated suffix included. A wrong-rank all-gather buffer passes XPU's eager collective,
    which checks only its size; gloo rejects it eagerly and dynamo when compiled."""
    mp.spawn(_round_trip_worker, args=(_free_port(), compiled), nprocs=WS, join=True)
