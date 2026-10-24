"""Muse Spark column-split projection (slice GEMM + multicast all-gather).

The gathered output must equal every rank's slice GEMM placed side by side,
up to the per-arch row bound, with the push space the model reserves.

Usage::

    python test/registered/kernels/ops/communication/test_muse_spark_gemm_ag.py
    python test/registered/kernels/ops/communication/test_muse_spark_gemm_ag.py --num-gpu 2
"""

from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import muse_spark_all_gather
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.communication import muse_spark_gemm_ag as gemm_ag
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="extra-b", runner_config="8-gpu-h200")

# (K, N) of the pooler and the unpooler.
_SHAPES = [(8192, 4096), (4096, 8192)]


def _device():
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


@cache_once
def _init_comm():
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
    comm = CustomAllReduceV2(coord.cpu_group, _device())
    for _, n in _SHAPES:
        gemm_ag.reserve_projection_workspace(comm, n)
    register_comm_cleanup(comm)
    return comm


@pytest.mark.parametrize("k,n", _SHAPES)
@torch.inference_mode()
def test_gemm_ag_matches_slice_gemms(k, n):
    comm = _init_comm()
    if comm.disabled or not comm.has_multicast:
        pytest.skip("needs a multicast-capable NVLink group")
    max_tokens = muse_spark_all_gather.projection_max_tokens(
        comm.world_size, n, _device()
    )
    n_local = n // comm.world_size
    generator = torch.Generator().manual_seed(n)
    weight = (torch.randn(n, k, generator=generator) * 0.02).to(
        _device(), torch.bfloat16
    )
    for m in sorted({1, max_tokens}):
        x = torch.randn(m, k, generator=generator).to(_device(), torch.bfloat16)
        assert muse_spark_all_gather.gemm_ag_supported(comm, x, weight)
        output = gemm_ag.gemm_ag(comm, x, weight)
        expected = torch.cat(
            [
                F.linear(x, weight[rank * n_local : (rank + 1) * n_local])
                for rank in range(comm.world_size)
            ],
            dim=1,
        )
        torch.cuda.synchronize()
        assert torch.equal(output, expected), f"{m=}"
    past_bound = torch.empty(max_tokens + 1, k, device=_device(), dtype=torch.bfloat16)
    assert not muse_spark_all_gather.gemm_ag_supported(comm, past_bound, weight)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
