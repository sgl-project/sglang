"""Multi-GPU test of the DSA DCP fused top-k (``DcpFusedTopK``).

Every rank holds the token-interleaved shard of the same global score rows and
selects, exchanges and merges them in one kernel. The union of the ranks'
results must be exactly a global top-k: no duplicates, min(k, L) keys whose
values (NaN as -inf, -0 below +0) equal the top-k value multiset. Each rank's
local slots, widened slots, counts and -1 padding must agree with each other,
and its local slots and counts must equal what ``dcp_localize_compact``
computes from the widened ones, since the backend takes them in its place. It
needs 8 SM100 GPUs with NVLink multicast.

Usage::

    python test/registered/kernels/ops/attention/test_dsa_dcp_fused_topk.py
    python test/registered/kernels/ops/attention/test_dsa_dcp_fused_topk.py -k graph
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
from sglang.kernels.ops.attention.dsa.dcp_fused_topk import dcp_fused_topk
from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import (
    DcpFusedTopK,
    _localize_compact_kernel,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="extra-b", runner_config="8-gpu-b300")

TOPK = 2048
PAGE_SIZE = 64
WORLD = 8
MAX_ROWS = 16
SEEDS = (1234, 1235, 1236)  # inputs the graph alternates between


@cache_once
def _init_once() -> Optional[DcpFusedTopK]:
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
    return DcpFusedTopK.create(
        ps._WORLD, MAX_ROWS, TOPK, PAGE_SIZE, ps._WORLD.rank_in_group
    )


def _rows(kind: str, seed: int) -> list[torch.Tensor]:
    """The same global score rows on every rank."""
    g = torch.Generator().manual_seed(seed)
    if kind == "real_like":
        # indexer-like rows: a negative bulk, a positive band and a few sinks,
        # so the window range crosses zero and several binades; one narrow row;
        # 393216 has more local keys than one register batch (32768); 1M has
        # local positions above 16 bits and more than 1024 pages
        rows = []
        for n in (1048576, 393216, 262144, 110592, 32768, 5003):
            x = torch.randn(n, generator=g) * 1.5 - 4.0
            band = torch.rand(n, generator=g) < 0.02
            x[band] = torch.randn(int(band.sum()), generator=g) * 0.5 + 2.0
            x[torch.randint(0, n, (16,), generator=g)] = 60.0
            rows.append(x)
        rows.append(0.05 * torch.randn(65536, generator=g))
        return rows
    if kind == "ties":
        # many equal scores at the threshold, and a row of one value only
        return [
            torch.randint(0, 200, (32768 - 100 * i,), generator=g).float() / 8
            for i in range(3)
        ] + [torch.full((40000,), 0.5)]
    if kind == "zero_nan":
        rows = []
        for n in (65536, 20000):
            x = torch.randn(n, generator=g)
            r = torch.rand(n, generator=g)
            x[r < 0.3] = 0.0
            x[(r >= 0.3) & (r < 0.6)] = -0.0
            x[(r >= 0.6) & (r < 0.75)] = float("nan")
            rows.append(x)
        # fewer finite keys than k: NaN keys fill the top-k as -inf
        x = torch.full((30000,), float("nan"))
        x[torch.randperm(30000, generator=g)[:1000]] = torch.randn(1000, generator=g)
        rows.append(x)
        return rows
    if kind == "short":
        lengths = [1, 3, 7, 9, 300, 2047, 2048, 2049, 2055, 2056, 3000, 5000, 1000]
        return [torch.randn(n, generator=g) for n in lengths]
    raise ValueError(kind)


def _all_ranks_ok(ok: bool) -> bool:
    """Agree on a check across ranks, so that a failure on one rank fails the
    test everywhere instead of leaving the others waiting on its pushes."""
    flag = torch.tensor([int(ok)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _keys(scores: torch.Tensor) -> torch.Tensor:
    """The kernel's total order: NaN as -inf, -0 below +0."""
    x = torch.where(scores.isnan(), float("-inf"), scores.float())
    b = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(b >= 0x80000000, 0xFFFFFFFF - b, b + 0x80000000)


class _Case:
    """One input set: this rank's shard of the rows and the expected top-k
    value multisets."""

    def __init__(self, rows: list[torch.Tensor], seed: int, rank: int):
        device = torch.device("cuda", torch.cuda.current_device())
        self.glob = rows
        self.rank = rank
        self.rows = len(rows)
        lens = [x.numel() for x in rows]
        self.local_lens = torch.tensor(
            [max((n - rank + WORLD - 1) // WORLD, 0) for n in lens],
            dtype=torch.int32,
            device=device,
        )
        width = (max(int(self.local_lens.max()), 1) + 11) // 4 * 4
        self.logits = torch.full((self.rows, width), float("-inf"), device=device)
        for i, x in enumerate(rows):
            local = x[rank::WORLD]
            self.logits[i, : local.numel()] = local.to(device)
        self.global_lens = torch.tensor(lens, dtype=torch.int32, device=device)
        pages = (max(max(lens) // WORLD, 1) + PAGE_SIZE - 1) // PAGE_SIZE + 1
        g = torch.Generator().manual_seed(seed + 99)
        self.page_table = torch.stack(
            [torch.randperm(4 * pages, generator=g)[:pages] for _ in range(self.rows)]
        ).to(torch.int32)
        self.expect = [
            sorted(torch.topk(_keys(x), min(TOPK, x.numel())).values.tolist())
            for x in rows
        ]
        self.page_table = self.page_table.to(device)
        self.out = torch.empty((self.rows, TOPK), dtype=torch.int32, device=device)
        self.local = self.lengths = None

    def run(self, fused: DcpFusedTopK, window=None) -> None:
        if window is None:
            self.lengths = self.out.new_empty(self.rows)
            self.local = fused(
                self.logits.clone(),
                self.local_lens,
                self.global_lens,
                self.page_table,
                self.out,
                self.lengths,
            )
            return
        self.local = torch.empty_like(self.out)
        self.lengths = torch.empty(self.rows, dtype=torch.int32, device=self.out.device)
        dcp_fused_topk(
            self.logits.clone(),
            self.local_lens,
            self.global_lens,
            self.page_table,
            out_local=self.local,
            out_slots=self.out,
            out_len=self.lengths,
            buffer=fused.buffer,
            multicast_ptr=fused.multicast_ptr,
            counters=fused.counters,
            fallbacks=fused.fallbacks,
            dcp_rank=fused.dcp_rank,
            window=window,
        )

    def failures(self) -> list[str]:
        """This rank's checks; also all-gathers every rank's global positions
        and checks their union (collective)."""
        rank, bad = self.rank, []
        local, slots, lengths = self.local.cpu(), self.out.cpu(), self.lengths.cpu()
        local_lens = self.local_lens.cpu()
        positions = []
        for i in range(self.rows):
            n = int(lengths[i])
            got = local[i, :n].long()
            if not bool((local[i, n:] == -1).all()):
                bad.append(f"rank {rank} row {i}: padding is not -1")
            want = torch.where(local[i] >= 0, local[i] * WORLD + rank, -1)
            if not torch.equal(slots[i], want):
                bad.append(f"rank {rank} row {i}: widened slots differ")
            inv = {int(p): j for j, p in enumerate(self.page_table[i].tolist())}
            pos = []
            for s in got.tolist():
                j = inv.get(s // PAGE_SIZE)
                loc = -1 if j is None else j * PAGE_SIZE + s % PAGE_SIZE
                if not 0 <= loc < int(local_lens[i]):
                    bad.append(f"rank {rank} row {i}: slot {s} is not a local key")
                    break
                pos.append(loc * WORLD + rank)
            positions.append(pos)
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, positions)
        for i, x in enumerate(self.glob):
            union = [p for r in range(WORLD) for p in gathered[r][i]]
            if len(set(union)) != len(union):
                bad.append(f"row {i}: duplicate positions")
                continue
            got_keys = _keys(x[torch.tensor(union, dtype=torch.long)]).tolist()
            if sorted(got_keys) != self.expect[i]:
                bad.append(f"row {i}: not the global top-k values")

        # the backend's localization of the widened slots, which the kernel's replaces
        ref_local = torch.empty_like(self.out)
        ref_lengths = torch.empty_like(self.lengths)
        _localize_compact_kernel[(self.rows,)](
            self.out.clone(),
            self.out.stride(0),
            ref_local,
            ref_local.stride(0),
            ref_lengths,
            WORLD,
            rank,
            TOPK=TOPK,
        )
        same = torch.equal(ref_local, self.local)
        if not (same and torch.equal(ref_lengths, self.lengths)):
            bad.append(f"rank {rank}: differs from dcp_localize_compact")
        return bad


def _check(case: _Case) -> None:
    torch.cuda.synchronize()
    bad = case.failures()
    assert _all_ranks_ok(not bad), bad or "failed on another rank"


def _fused() -> DcpFusedTopK:
    fused = _init_once()
    if ps._WORLD.world_size != WORLD:
        pytest.skip("the fused top-k needs 8 ranks")
    if fused is None:
        pytest.skip("needs SM100 and multimem multicast")
    return fused


@pytest.mark.parametrize("kind", ["real_like", "ties", "zero_nan", "short"])
@torch.inference_mode()
def test_dcp_fused_topk(kind: str) -> None:
    fused = _fused()
    rank = ps._WORLD.rank_in_group
    case = _Case(_rows(kind, SEEDS[0]), SEEDS[0], rank)
    case.run(fused)
    _check(case)


@torch.inference_mode()
def test_dcp_fused_topk_forced_fallback() -> None:
    """A one-rank window fails the certificate: the exact fallback runs."""
    fused = _fused()
    rank = ps._WORLD.rank_in_group
    case = _Case(_rows("real_like", SEEDS[1]), SEEDS[1], rank)
    before = fused.fallbacks[: case.rows].clone()
    case.run(fused, window=(256, 256))
    _check(case)
    assert bool((fused.fallbacks[: case.rows] > before).all())


@pytest.mark.parametrize("lagging", [False, True])
@torch.inference_mode()
def test_dcp_fused_topk_short_call_between_exchanges(lagging: bool) -> None:
    """A call without exchange (L <= k) between two exchanges of a row slot
    waits on no peer, so it must leave the exchange buffers alone: with rank 0
    lagging behind it, the other ranks' next exchange messages already sit in
    its buffer, and a clear there would leave its next exchange polling
    forever."""
    fused = _fused()
    rank = ps._WORLD.rank_in_group
    first = _Case(_rows("real_like", SEEDS[0])[3:4], SEEDS[0], rank)
    short = _Case(_rows("short", SEEDS[1])[-1:], SEEDS[1], rank)
    third = _Case(_rows("real_like", SEEDS[2])[3:4], SEEDS[2], rank)
    if not lagging:
        for case in (first, short, third):
            case.run(fused)
            _check(case)
        return
    first.run(fused)
    if rank == 0:
        torch.cuda._sleep(200_000_000)  # the other ranks run ahead into `third`
    short.run(fused)
    third.run(fused)
    for case in (first, short, third):
        _check(case)


@torch.inference_mode()
def test_dcp_fused_topk_graph() -> None:
    """Graph replays on three alternating inputs, across both buffer parities."""
    fused = _fused()
    rank = ps._WORLD.rank_in_group
    cases = [
        _Case(_rows("real_like", seed)[1:4] + _rows("short", seed)[6:9], seed, rank)
        for seed in SEEDS
    ]
    for case in cases:
        case.run(fused)
        _check(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            case.run(fused)
    for _ in range(3):
        for _ in range(7):
            graph.replay()
        for case in cases:
            _check(case)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
