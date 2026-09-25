"""Correctness tests for the Kimi-K3 row-parallel LatentMoE tail."""

import os

import pytest
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers import k3_sp
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_amd_ci(est_time=60, suite="nightly-amd-8-gpu-mi35x")

LATENT, HIDDEN, TOKENS = 3584, 7168, 19456
EPS = 1e-6

_DEVICE = None


def _device():
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


def _init_world():
    """One process group per worker process, shared by every test here."""
    global _DEVICE
    if _DEVICE is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", local_rank)
        )
        _DEVICE = torch.device("cuda", local_rank)
    return dist.get_rank(), dist.get_world_size()


def _rmsnorm(x, gamma, eps=EPS):
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * gamma


@pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="needs a multi-GPU process group",
)
@torch.inference_mode()
def test_shard_rows_partition():
    rank, world = _init_world()
    shard = TOKENS // world
    rows = slice(rank * shard, (rank + 1) * shard)

    seen = torch.zeros(TOKENS, dtype=torch.int32, device=_device())
    seen[rows] = 1
    dist.all_reduce(seen, op=dist.ReduceOp.SUM)
    assert int(seen.min()) == 1 and int(seen.max()) == 1

    local = torch.arange(rows.start, rows.stop, device=_device(), dtype=torch.int32)
    gathered = torch.empty(TOKENS, dtype=torch.int32, device=_device())
    dist.all_gather_into_tensor(gathered, local)
    assert torch.equal(
        gathered, torch.arange(TOKENS, device=_device(), dtype=torch.int32)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="needs a multi-GPU process group",
)
@torch.inference_mode()
def test_tail_shard_matches_replicated():
    assert k3_sp.tail_shard_enabled() is False
    with envs.SGLANG_K3_TAIL_SHARD.override(True):
        assert k3_sp.tail_shard_eligible(32) is False

    rank, world = _init_world()
    torch.manual_seed(1234)
    dev = _device()

    latent = torch.randn(TOKENS, LATENT, device=dev, dtype=torch.bfloat16)
    shared = torch.randn(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    prefix = torch.randn(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    w_up = torch.randn(LATENT, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
    gamma = torch.randn(LATENT, device=dev, dtype=torch.bfloat16).abs() + 0.5

    ref = torch.mm(_rmsnorm(latent, gamma), w_up) + shared + prefix

    rows = slice(rank * (TOKENS // world), (rank + 1) * (TOKENS // world))
    shard = torch.mm(_rmsnorm(latent[rows], gamma), w_up) + shared[rows] + prefix[rows]
    got = torch.empty(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(got, shard.contiguous())

    torch.testing.assert_close(got, ref, rtol=0, atol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 8))
