"""Usage::

python test/registered/kernels/ops/communication/test_moe_front_ag.py   # relaunches under torchrun
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
from sglang.kernels.ops.communication import all_reduce_residual as all_reduce
from sglang.kernels.ops.communication import moe_front_ag
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

# Where test_collectives.py runs: the K3 fused collectives are SM100-only.
register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-b200")
register_cuda_ci(est_time=120, stage="nightly", runner_config="8-gpu-b200")

N = 896 + 3584  # K3 [router | latent] rows
H = 7168
MB = 1024 * 1024
TOKENS = [18, 96, 256]


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
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_nccl_group():
    _init_world()
    local_rank = int(os.environ["LOCAL_RANK"])
    group = dist.new_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert isinstance(group, dist.ProcessGroup)
    return group


@cache_once
def _build_comm() -> Optional[CustomAllReduceV2]:
    cpu_group = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    # 4 MB slots: the largest block here is TP2 at 256 tokens (2.3 MB)
    comm = CustomAllReduceV2(
        cpu_group, device, max_pull_size=1 * MB, max_push_size=4 * MB
    )
    if comm.disabled or not comm.has_multicast:
        return None
    all_reduce.register_comm(comm.obj)
    register_comm_cleanup(comm)
    return comm


def _init_comm() -> CustomAllReduceV2:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("the K3 MoE front gather is used on SM100+ only")
    comm = _build_comm()
    if comm is None:
        pytest.skip("moe_front_ag requires CustomAllReduceV2 with multicast")
    return comm


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def _block(num_tokens: int, n_local: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed * 1009 + dist.get_rank())
    x = torch.randn(num_tokens, n_local, generator=g)
    x[0, :4] = 0.0  # one +0.0 vector: the marker swap must not hang the spin
    return x.to(_device())


def _nccl_ref(block: torch.Tensor) -> torch.Tensor:
    world = dist.get_world_size()
    parts = [torch.empty_like(block) for _ in range(world)]
    dist.all_gather(parts, block, group=_init_nccl_group())
    return torch.cat(parts, dim=1)


def _assert_gathered(out: torch.Tensor, ref: torch.Tensor) -> None:
    # bit-exact up to the sign of zero
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    zeros = ref == 0
    assert torch.equal(out[~zeros].view(torch.int32), ref[~zeros].view(torch.int32))


@pytest.mark.parametrize("num_tokens", TOKENS)
@torch.inference_mode()
def test_moe_front_all_gather(num_tokens: int):
    comm = _init_comm()
    world = comm.world_size
    n_local = N // world
    # production gathers a column slice of the [block | gate_up] GEMM output
    gate_up = 12288 // world
    wide = torch.randn(num_tokens, n_local + gate_up, device=_device())
    wide[:, :n_local] = _block(num_tokens, n_local, 300 + num_tokens)
    block = wide[:, :n_local]
    out = torch.empty(num_tokens, N, dtype=torch.float32, device=_device())
    moe_front_ag.moe_front_all_gather(world, block, out)
    torch.cuda.synchronize()
    _assert_gathered(out, _nccl_ref(block.contiguous()))


@torch.inference_mode()
def test_moe_front_all_gather_stress_with_push_all_reduce():
    # both use the one push workspace's phase double-buffering
    comm = _init_comm()
    world = comm.world_size
    n_local = N // world
    for it in range(24):
        num_tokens = TOKENS[it % len(TOKENS)]
        block = _block(num_tokens, n_local, 100 + it)
        out = torch.empty(num_tokens, N, dtype=torch.float32, device=_device())
        moe_front_ag.moe_front_all_gather(world, block, out)
        g = torch.Generator(device="cpu").manual_seed(it * 7 + dist.get_rank())
        x = torch.randint(0, 16, (8 * H,), generator=g).to(torch.bfloat16)
        x = x.to(_device())
        ref_x = x.clone()
        dist.all_reduce(ref_x, group=_init_nccl_group())
        all_reduce.all_reduce_push_res(world, x, None)
        _assert_gathered(out, _nccl_ref(block))
        torch.testing.assert_close(x, ref_x, atol=0, rtol=0)


@torch.inference_mode()
def test_moe_front_all_gather_graph_capture():
    comm = _init_comm()
    world = comm.world_size
    cpu_group = _init_world()
    n_local = N // world
    num_tokens = 24
    block = torch.zeros(num_tokens, n_local, dtype=torch.float32, device=_device())
    out = torch.empty(num_tokens, N, dtype=torch.float32, device=_device())

    def _run():
        moe_front_ag.moe_front_all_gather(world, block, out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run()

    for it in range(4):
        block.copy_(_block(num_tokens, n_local, 500 + it))
        ref = _nccl_ref(block)
        dist.barrier(group=cpu_group)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        _assert_gathered(out, ref)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
