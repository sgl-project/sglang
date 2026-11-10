from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.muse_spark_all_gather_norm_residual import (
    all_gather_norm_residual,
    all_gather_norm_residual_supported,
    all_reduce_norm_residual,
)
from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import (
    fused_norm_residual,
)
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.communication.muse_spark_gemm_ag import (
    reserve_projection_workspace,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN = 8192
EPS = 1e-6


@cache_once
def _init_comm() -> CustomAllReduceV2:
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
    comm = CustomAllReduceV2(coord.cpu_group, torch.device(f"cuda:{local_rank}"))
    if comm.disabled or not comm.has_multicast:
        pytest.skip("requires CustomAllReduceV2 with multicast")
    reserve_projection_workspace(comm, HIDDEN)
    register_comm_cleanup(comm)
    return comm


@pytest.mark.parametrize("num_tokens", [1, 3, 8, 16])
@pytest.mark.parametrize("want_router", [True, False])
def test_all_reduce_norm_residual(num_tokens, want_router):
    comm = _init_comm()
    assert all_gather_norm_residual_supported(comm, num_tokens, HIDDEN)
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    stream = torch.randn(num_tokens, HIDDEN, generator=gen, device="cuda") * 3
    pre_weight = (
        torch.randn(HIDDEN, generator=gen, device="cuda") * 0.1 + 1
    ).bfloat16()
    alpha = torch.rand(HIDDEN, generator=gen, device="cuda") * 0.5 + 0.5
    beta = torch.rand(HIDDEN, generator=gen, device="cuda") * 0.5 + 0.25
    rank_gen = torch.Generator(device="cuda").manual_seed(
        1000 * (dist.get_rank() + 1) + num_tokens
    )
    for _ in range(2):
        partial = torch.randn(num_tokens, HIDDEN, generator=rank_gen, device="cuda")
        partial = partial.bfloat16()
        partial[0, :64] = 0  # +0 is the sentinel
        reduced = comm.custom_all_reduce(partial.clone())
        expected = fused_norm_residual(
            reduced,
            stream,
            EPS,
            EPS,
            pre_weight=pre_weight,
            alpha=alpha,
            beta=beta,
            want_router=want_router,
        )
        actual = all_reduce_norm_residual(
            comm,
            partial,
            stream,
            EPS,
            EPS,
            want_router=want_router,
            pre_weight=pre_weight,
            alpha=alpha,
            beta=beta,
        )
        for got, ref in zip(actual, expected):
            if ref is None:
                assert got is None
            else:
                torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [3, 128])
def test_all_gather_norm_residual(num_tokens):
    comm = _init_comm()
    assert all_gather_norm_residual_supported(comm, num_tokens, HIDDEN)
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    stream = torch.randn(num_tokens, HIDDEN, generator=gen, device="cuda") * 3
    weights = dict(
        post_weight=(
            torch.randn(HIDDEN, generator=gen, device="cuda") * 0.1 + 1
        ).bfloat16(),
        pre_weight=(
            torch.randn(HIDDEN, generator=gen, device="cuda") * 0.1 + 1
        ).bfloat16(),
        alpha=torch.rand(HIDDEN, generator=gen, device="cuda") * 0.5 + 0.5,
        beta=torch.rand(HIDDEN, generator=gen, device="cuda") * 0.5 + 0.25,
    )
    for _ in range(2):
        branch = torch.randn(
            num_tokens, HIDDEN, generator=gen, device="cuda"
        ).bfloat16()
        branch[0, :64] = 0
        local = branch.chunk(comm.world_size, dim=1)[dist.get_rank()].contiguous()
        expected = fused_norm_residual(
            branch, stream, EPS, EPS, post_norm_is_input_scale=True, **weights
        )
        actual = all_gather_norm_residual(
            comm, local, stream, EPS, EPS, post_norm_is_input_scale=True, **weights
        )
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[4])
