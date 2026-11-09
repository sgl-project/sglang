"""Needs 8 GPUs with NVLink multicast.

Usage::

    python test/registered/kernels/ops/attention/test_dsa_dcp_topk_merge.py
    python test/registered/kernels/ops/attention/test_dsa_dcp_topk_merge.py -k periodic
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Optional

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_candidates_v2
from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import DcpTopKMerge
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=138, stage="extra-b", runner_config="8-gpu-h200")

TOPK = 2048
PAGE_SIZE = 64
MAX_ROWS = 16
SEEDS = (1234, 1235, 1236)  # inputs the graph alternates between
GRAPH_CHECKS, REPLAYS_PER_CHECK = 5, 20  # 3 * 100 calls per row slot > 255 tags


@cache_once
def _init_once() -> Optional[DcpTopKMerge]:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.cuda.set_stream(torch.cuda.Stream())
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=ps._WORLD)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    return DcpTopKMerge.create(ps._WORLD, MAX_ROWS, TOPK, ps._WORLD.rank_in_group)


def _rows(kind: str, seed: int) -> list[torch.Tensor]:
    """The same global score rows on every rank."""
    g = torch.Generator().manual_seed(seed)
    if kind == "random":
        return [torch.randn(131072, generator=g)]
    if kind == "ties":
        return [
            torch.randint(0, 200, (32768 - 100 * i,), generator=g).float() / 8
            for i in range(4)
        ]
    if kind == "periodic":
        # one rank owns the whole top-k: its candidates cannot certify it
        rows = []
        for n in (65536, 20000):
            base = torch.randn(8, generator=g)
            base[3] += 10
            # distinct scores keep the local top-k exact, so only the merge is tested
            rows.append(
                base.repeat(n // 8 + 1)[:n] + 4.0 * torch.randperm(n, generator=g) / n
            )
        return rows
    if kind == "signed_zeros":
        # ties at the threshold between -0 (even positions) and +0 (odd ones)
        x = torch.zeros(131072)
        x[0::2] = -0.0
        x[torch.randperm(x.numel(), generator=g)[:1000]] = 1.0
        return [x]
    if kind == "short":
        lengths = [1, 3, 7, 9, 300, 2047, 2048, 2049, 2055, 3000, 5000]
        return [torch.randn(n, generator=g) for n in lengths]
    raise ValueError(kind)


def _all_ranks_ok(ok: bool) -> bool:
    """Agree on a check across ranks, so that a failure on one rank fails the
    test everywhere instead of leaving the others waiting on its pushes."""
    flag = torch.tensor([int(ok)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _keys(scores: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    scores = torch.where(scores == 0, 0.0, scores.float())  # -0 ties with +0
    b = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ordered = torch.where(b >= 0x80000000, 0xFFFFFFFF - b, b + 0x80000000)
    return ((ordered - 0x80000000) << 32) | (0xFFFFFFFF - pos.to(torch.int64))


class _Case:
    def __init__(self, kind: str, seed: int, rank: int, world: int, span: int):
        device = torch.device("cuda", torch.cuda.current_device())
        glob = _rows(kind, seed)
        self.rows = len(glob)
        lens = [x.numel() for x in glob]
        self.local_lens = torch.tensor(
            [max((n - rank + world - 1) // world, 0) for n in lens],
            dtype=torch.int32,
            device=device,
        )
        width = (max(int(self.local_lens.max()), 1) + 11) // 4 * 4
        self.logits = torch.full((self.rows, width), float("-inf"), device=device)
        for i, x in enumerate(glob):
            local = x[rank::world]
            self.logits[i, : local.numel()] = local.to(device)
        self.global_lens = torch.tensor(lens, dtype=torch.int32, device=device)
        pages = (max(lens) + span - 1) // span
        g = torch.Generator().manual_seed(seed + 99)
        self.page_table = (
            torch.stack(
                [
                    torch.randperm(4 * pages, generator=g)[:pages]
                    for _ in range(self.rows)
                ]
            )
            .to(torch.int32)
            .to(device)
        )

        self.expect, self.expect_local = [], []
        for i, x in enumerate(glob):
            local_pos = torch.arange(x.numel())[rank::world]
            local_keys = _keys(x[rank::world], local_pos)
            local_top = local_pos[
                torch.topk(local_keys, min(TOPK, local_pos.numel())).indices
            ]
            self.expect_local.append(sorted(local_top.tolist()))
            pos = torch.arange(x.numel())
            top = pos[torch.topk(_keys(x, pos), min(TOPK, x.numel())).indices]
            mine = top[top % world == rank]
            pt = self.page_table[i].cpu().long()
            slots = pt[mine // span] * span + mine % span
            self.expect.append(sorted(slots.tolist()))

        self.candidates = torch.empty(
            (self.rows, TOPK, 2), dtype=torch.int32, device=device
        )
        self.out = torch.empty((self.rows, TOPK), dtype=torch.int32, device=device)

    def run(self, merge: DcpTopKMerge, world: int, rank: int, span: int) -> None:
        topk_transform_candidates_v2(
            self.logits.clone(),
            self.local_lens,
            out_candidates=self.candidates,
            dcp_size=world,
            dcp_rank=rank,
            deterministic=True,
        )
        merge(self.candidates, self.global_lens, self.page_table, self.out, span)

    def failures(self, rank: int) -> list[str]:
        bad = []
        for i in range(self.rows):
            got_local = self.candidates[i, :, 1]
            if sorted(got_local[got_local >= 0].tolist()) != self.expect_local[i]:
                bad.append(f"rank {rank} row {i}: local candidates are not exact")
            n = len(self.expect[i])
            if sorted(self.out[i, :n].tolist()) != self.expect[i]:
                bad.append(f"rank {rank} row {i}: merged slots differ")
            if not bool((self.out[i, n:] == -1).all()):
                bad.append(f"rank {rank} row {i}: padding is not -1")
        return bad


@pytest.mark.parametrize(
    "kind", ["random", "ties", "periodic", "signed_zeros", "short"]
)
@torch.inference_mode()
def test_dcp_topk_merge(kind: str) -> None:
    merge = _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 8:
        pytest.skip("the merge needs 8 ranks")
    if merge is None:
        pytest.skip("multimem multicast unavailable")
    span = PAGE_SIZE * world
    cases = [_Case(kind, seed, rank, world, span) for seed in SEEDS]

    for case in cases:
        case.run(merge, world, rank, span)
        torch.cuda.synchronize()
        bad = case.failures(rank)
        assert _all_ranks_ok(not bad), bad or "failed on another rank"
        fell_back = merge.fallback[: case.rows].tolist()
        if kind == "periodic":
            assert all(fell_back), fell_back
        elif kind in ("random", "short"):
            assert not any(fell_back), fell_back

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            case.run(merge, world, rank, span)
    for _ in range(GRAPH_CHECKS):
        for _ in range(REPLAYS_PER_CHECK):
            graph.replay()
        torch.cuda.synchronize()
        bad = [msg for case in cases for msg in case.failures(rank)]
        assert _all_ranks_ok(not bad), bad or "failed on another rank"


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
