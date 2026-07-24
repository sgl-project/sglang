"""Correctness tests for the K3 fused o_proj GEMM + all-reduce kernel.

Checks ``out = sum_r bf16(x_r @ W_r^T)`` against an NCCL reference for a
sweep of token counts covering every dispatch cell (member-1 persistent,
member-3 one-shot, two-shot) plus cell-change ring resets and back-to-back
same-cell launches (ring reuse).

Usage::

    python test/registered/jit/kimi_k3/test_k3_gemm_ar.py   # relaunches under torchrun

Requires >= 2 SM100+ GPUs with full NVLink P2P on one node (perf-tuned on
GB300 / sm_103a).
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.kimi_k3 import gemm_ar
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.kernels.jit.utils import cache_once, get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240,
    stage="extra-b",
    runner_config="8-gpu-h200",
    disabled="requires SM100+ (B200/GB300) GPUs with full NVLink P2P",
)

N = gemm_ar.N
K = 12288 // int(os.environ.get("WORLD_SIZE", "8"))

BATCH = get_ci_test_range(
    [1, 5, 8, 16, 24, 32, 64, 100, 128, 300, 512], [1, 8, 64, 512]
)


def _precompile(num_gpus):
    for n in num_gpus:
        gemm_ar._jit_module(12288 // n, n)


@cache_once
def _init_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    return coord.cpu_group


@cache_once
def _init_state():
    cpu_group = _init_world()
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("gemm_ar requires SM100+")
    gemm_ar.init(
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["LOCAL_RANK"]),
        group=cpu_group,
        k=K,
    )


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


@cache_once
def _weight() -> torch.Tensor:
    rank = int(os.environ["LOCAL_RANK"])
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    return torch.randn(N, K, generator=g).to(torch.bfloat16).to(_device())


def _make_x(m: int) -> torch.Tensor:
    rank = int(os.environ["LOCAL_RANK"])
    g = torch.Generator(device="cpu").manual_seed(7 * m + rank)
    return torch.randn(m, K, generator=g).to(torch.bfloat16).to(_device())


def _ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # fused semantics: fp32-accumulated partial, rounded to bf16 PRE-sum,
    # summed in fp32 across ranks
    partial = (x.float() @ weight.float().t()).to(torch.bfloat16).float()
    dist.all_reduce(partial, group=ps._WORLD.device_group)
    return partial


def _check(out: torch.Tensor, ref: torch.Tensor, m: int):
    got = out.float()
    bad = ((got - ref).abs() > 0.05 + 0.02 * ref.abs()).sum().item()
    assert bad <= out.numel() / 1000, f"M={m}: {bad}/{out.numel()} out of tolerance"


@pytest.mark.parametrize("bs", BATCH)
@torch.inference_mode()
def test_gemm_ar_correctness(bs: int):
    _init_state()
    weight = _weight()
    x = _make_x(bs)
    ref = _ref(x, weight)
    # back-to-back same-cell launches exercise the epoch ring reuse
    for _ in range(3):
        out = gemm_ar.o_proj_gemm_ar(x, weight)
    torch.cuda.synchronize()
    dist.barrier()
    _check(out, ref, bs)


@torch.inference_mode()
def test_gemm_ar_cell_flip():
    """Alternating cells forces the collective ring reset each call."""
    _init_state()
    weight = _weight()
    for bs in [1, 64, 1, 300, 8]:
        x = _make_x(bs)
        ref = _ref(x, weight)
        out = gemm_ar.o_proj_gemm_ar(x, weight)
        torch.cuda.synchronize()
        dist.barrier()
        _check(out, ref, bs)


multigpu_pytest_main(
    __name__,
    __file__,
    num_gpus=(4, 8),
    pre_launch_fn=_precompile,
)
