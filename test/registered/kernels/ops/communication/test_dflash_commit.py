"""The DFLASH verify commit kernel (csrc/distributed/dflash_commit.cuh).

The local commit must match the stock _commit_accept bit for bit. The push-plane
exchange must also hand every rank rank 0's accept lengths and bonus tokens, and
leave the plane usable for the push all-reduces that share it.

Run with ``python <this file>``; it relaunches under torchrun on 2 and 4 GPUs.
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import dflash_commit
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_worker_v2 import _commit_accept
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

VOCAB = 262144
# Odd sizes leave the last 16-byte exchange vector half filled; 1024 is the kernel's cap.
BATCH_SIZES = [1, 3, 64, 255, 1024]


def _decision(bs: int, block: int, seed: int):
    """Candidates and rank 0's selector decision; every rank draws the same values."""
    g = torch.Generator().manual_seed(seed)
    candidates = torch.randint(0, VOCAB, (bs, block), generator=g)
    accept_len = torch.randint(0, block, (bs,), generator=g, dtype=torch.int32)
    bonus = torch.randint(0, VOCAB, (bs,), generator=g)
    # Both ends of the block: no draft accepted, and every draft accepted.
    accept_len[0] = 0
    accept_len[-1] = block - 1
    return candidates.cuda(), accept_len.cuda(), bonus.cuda()


@pytest.mark.parametrize("bs", BATCH_SIZES)
@pytest.mark.parametrize("block", [2, 7, 16])
def test_local_commit_matches_commit_accept(bs: int, block: int):
    candidates, accept_len, bonus = _decision(bs, block, seed=bs * 31 + block)
    want_tokens, want_lens = _commit_accept(candidates, accept_len, bonus)
    want_accept, want_bonus = accept_len.clone(), bonus.clone()
    tokens, lens = dflash_commit.commit(candidates, accept_len, bonus)
    assert torch.equal(tokens, want_tokens)
    assert torch.equal(lens, want_lens)
    # The kernel writes accept_len and bonus back; locally they must not change.
    assert torch.equal(accept_len, want_accept)
    assert torch.equal(bonus, want_bonus)


@cache_once
def _init() -> CustomAllReduceV2:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{local_rank}"),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=1 << 20,
    )
    assert not comm.disabled
    comm.override_algo = AllReduceAlgo.ONE_SHOT_PUSH
    register_comm_cleanup(comm)
    return comm


@pytest.mark.parametrize("bs", BATCH_SIZES)
@pytest.mark.parametrize("block", [2, 16])
def test_exchange_hands_every_rank_rank0_decision(bs: int, block: int):
    comm = _init()
    rank, world = comm.rank, comm.world_size
    partial = torch.full((256,), float(rank + 1), device="cuda")
    want_sum = torch.full_like(partial, world * (world + 1) / 2)
    mismatches = []
    for step in range(4):
        candidates, accept_len, bonus = _decision(
            bs, block, seed=step * 4099 + bs * 31 + block
        )
        want_tokens, want_lens = _commit_accept(candidates, accept_len, bonus)
        want_accept, want_bonus = accept_len.clone(), bonus.clone()
        if rank != 0:
            # This rank's own draw diverged; the exchange must replace it.
            accept_len = block - 1 - accept_len
            bonus = bonus + VOCAB * rank
        tokens, lens = dflash_commit.commit(candidates, accept_len, bonus, comm)
        checks = [
            ("accept_len", accept_len, want_accept),
            ("bonus", bonus, want_bonus),
            ("out_tokens", tokens, want_tokens),
            ("commit_lens", lens, want_lens),
        ]
        if step % 2:
            # Every call flips the phase, so one call on even steps and two on odd
            # ones walk the exchange and the push all-reduce through both halves.
            checks.append(("all_reduce", comm.custom_all_reduce(partial), want_sum))
        mismatches += [
            f"step {step}: {name}"
            for name, got, want in checks
            if not torch.equal(got, want)
        ]
    # Asserted after the loop so a failing rank still makes every collective call.
    assert not mismatches, f"rank {rank}: {mismatches}"


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
