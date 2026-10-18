"""The T-token TP4 verify logits megakernel: the vocab-parallel lm head, the logit
scale and softcap and the gather of the shards, as close to FP64 as the stock
path, replicated on every rank.

Run with ``python <this file>``; it relaunches itself under torchrun on 4 GPUs.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import os

import pytest
import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

try:
    import helion
except ModuleNotFoundError as error:
    if error.name != "helion":
        raise
    HELION_MEGAKERNELS = False
else:
    HELION_MEGAKERNELS = (
        "cross_loop_pipeline" in inspect.signature(helion.Config).parameters
    )
    from sglang.kernels.ops.muse_spark.helion_megakernels.lm_head_verify_megakernel import (
        LogitScalars,
        MuseLmHeadVerifyMegakernel,
    )

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.skipif(
        not HELION_MEGAKERNELS, reason="requires Helion with cross-loop pipelining"
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="the megakernel is specialized for SM100",
    ),
]

# The served shard, which is not a whole number of G blocks.
HIDDEN, VOCAB_SHARD = 8192, 50512
SCALARS = LogitScalars(scale=0.75, cap=20.0) if HELION_MEGAKERNELS else None


@functools.cache
def _init_group_once() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    atexit.register(dist.destroy_process_group)
    return dist.group.WORLD


def _inputs(seed: int, rank: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Replicated hidden states, per-rank weights; logits reach the softcap knee.
    x = torch.Generator(device="cuda").manual_seed(seed)
    g = torch.Generator(device="cuda").manual_seed(seed * 16 + rank)
    hidden = torch.randn(tokens, HIDDEN, device="cuda", generator=x).bfloat16()
    weight = 0.15 * torch.randn(VOCAB_SHARD, HIDDEN, device="cuda", generator=g)
    return hidden, weight.bfloat16()


def _stock(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logits = torch.mm(hidden, weight.T, out_dtype=torch.float32)
    logits.mul_(SCALARS.scale)
    shards = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
    dist.all_gather(shards, logits)
    logits = torch.cat(shards, dim=-1)
    return SCALARS.cap * torch.tanh(logits / SCALARS.cap)


def _fp64_shard(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logits = (hidden.double() @ weight.double().T) * SCALARS.scale
    return SCALARS.cap * torch.tanh(logits / SCALARS.cap)


@pytest.mark.parametrize("tokens", [8, 16], ids=lambda t: f"t{t}")
@torch.inference_mode()
def test_lm_head_verify_matches_stock_logits(tokens: int) -> None:
    group = _init_group_once()
    rank, world = group.rank(), group.size()
    mk = MuseLmHeadVerifyMegakernel(
        torch.device("cuda", torch.cuda.current_device()),
        group,
        SCALARS,
        tokens=tokens,
        vocab_shard=VOCAB_SHARD,
    )
    failures = []
    # Fresh inputs per call, so a stale cross-launch shard block shows up.
    for seed in range(21, 26):
        hidden, weight = _inputs(seed, rank, tokens)
        out = torch.full((tokens, world * VOCAB_SHARD), float("nan"), device="cuda")
        logits = mk(hidden_states=hidden, weight=weight, out=out)
        stock = _stock(hidden, weight)
        ref = _fp64_shard(hidden, weight)
        mine = logits.view(tokens, world, VOCAB_SHARD)[:, rank].double()
        theirs = stock.view(tokens, world, VOCAB_SHARD)[:, rank].double()
        err = (mine - ref).abs().max().item()
        stock_err = (theirs - ref).abs().max().item()
        peers = [torch.empty_like(logits) for _ in range(world)]
        dist.all_gather(peers, logits)
        replicated = all(torch.equal(p, logits) for p in peers)
        if not (
            logits.data_ptr() == out.data_ptr()
            and replicated
            and err <= 2 * stock_err + 1e-4
        ):
            failures.append(
                f"seed {seed}: err {err:.3e} (stock {stock_err:.3e}) "
                f"replicated {replicated}"
            )
    ok = torch.tensor([int(not failures)], device="cuda")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    assert ok.item(), f"rank {rank}/{world}: {failures}"


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
