"""Multi-GPU test of the DSA prefill ShardQ exchange (shard_q_push +
ShardQExchange): every rank holds its heads of all q rows and gets back every
rank's heads of its own row block, then pushes its attention output back so
each rank holds its heads of all rows. Both directions must be byte-identical
to a reference built from all-gathers, for even and KV-balanced (uneven) row
bounds, batches with fewer rows than ranks, and buffer growth.

Usage::

    python test/registered/kernels/ops/communication/test_shard_q_exchange.py   # relaunches under torchrun
    python test/registered/kernels/ops/communication/test_shard_q_exchange.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

from sglang.kernels.jit.utils import cache_once
from sglang.srt.layers.attention.dsa.utils import kv_balanced_row_bounds
from sglang.srt.layers.communication.shard_q_exchange import (
    ShardQExchange,
    get_shard_q_exchange,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=90, stage="extra-b", runner_config="8-gpu-h200")

HEADS = 12  # Kimi-K3.1 MLA heads per rank at TP8
Q_DIM = 576  # fp8 nope + rope
V_DIM = 512


@cache_once
def _setup() -> ShardQExchange:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    atexit.register(dist.destroy_process_group)
    exchange = get_shard_q_exchange(dist.group.WORLD)
    assert exchange is not None, "symmetric memory unavailable"
    return exchange


def _all_gather(x: torch.Tensor) -> torch.Tensor:
    world = dist.get_world_size()
    out = x.new_empty((world, *x.shape))
    dist.all_gather_into_tensor(
        out.view(torch.uint8).view(world, -1), x.view(torch.uint8).view(-1)
    )
    return out


def _check(exchange: ShardQExchange, bounds: list[int], seed: int) -> None:
    world, rank = exchange.world, exchange.rank
    rows = bounds[-1]
    lo, hi = bounds[rank], bounds[rank + 1]
    counts = [b - a for a, b in zip(bounds, bounds[1:])]
    g = torch.Generator(device="cuda").manual_seed(seed * 131 + rank)
    bounds_gpu = torch.tensor(bounds, dtype=torch.int32, device="cuda")

    q = torch.randn(rows, HEADS, Q_DIM, device="cuda", generator=g)
    q = q.to(torch.float8_e4m3fn)
    got_q = exchange.exchange_q(q, bounds_gpu, max_count=max(counts), count=hi - lo)
    want_q = (
        _all_gather(q)[:, lo:hi].transpose(0, 1).reshape(hi - lo, world * HEADS, Q_DIM)
    )
    assert torch.equal(got_q.view(torch.uint8), want_q.view(torch.uint8))

    out = torch.randn(hi - lo, world * HEADS, V_DIM, device="cuda", generator=g)
    out = out.to(torch.bfloat16)
    padded = out.new_zeros(max(counts), world * HEADS, V_DIM)
    padded[: hi - lo] = out
    owners = _all_gather(padded)
    want_o = torch.cat(
        [owners[s, :c, rank * HEADS : (rank + 1) * HEADS] for s, c in enumerate(counts)]
    )
    got_o = exchange.exchange_out(out, row0=lo, rows=rows)
    assert torch.equal(got_o.view(torch.uint8), want_o.view(torch.uint8))


@pytest.mark.parametrize("rows", [3, 7, 2048, 12345])
def test_even_bounds(rows):
    exchange = _setup()
    per = -(-rows // exchange.world)
    bounds = [min(r * per, rows) for r in range(exchange.world + 1)]
    _check(exchange, bounds, seed=rows)


@pytest.mark.parametrize("rows", [2048, 12345])
def test_kv_balanced_bounds(rows):
    """A long prefix extend next to a fresh request, as the indexer and the
    backend cut it: row blocks differ in size by a lot."""
    exchange = _setup()
    ext = [rows // 3, rows - rows // 3]
    seq = torch.tensor([ext[0] + 60000, ext[1]])
    bounds = kv_balanced_row_bounds(seq, ext, rows, exchange.world)
    assert len(set(b - a for a, b in zip(bounds, bounds[1:]))) > 1
    _check(exchange, bounds, seed=rows + 1)


def test_repeated_calls_after_growth():
    """Back-to-back exchanges reuse the buffers; a larger batch grows them."""
    exchange = _setup()
    for rows in (512, 16384, 1024, 16384):
        per = -(-rows // exchange.world)
        bounds = [min(r * per, rows) for r in range(exchange.world + 1)]
        _check(exchange, bounds, seed=rows + 2)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[2, 4, 8])
