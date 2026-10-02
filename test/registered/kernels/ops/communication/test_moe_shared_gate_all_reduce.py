"""Compare the gated collective with the existing rank-local MoE chain."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.all_reduce_fusion import (
    moe_finalize_all_reduce,
    moe_finalize_shared_gate_all_reduce,
    register_comm,
)
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.kernels.ops.moe.moe_finalize_fuse_shared import moe_finalize_fuse_shared
from sglang.kernels.ops.moe.shared_expert_gate import shared_expert_gate
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.fixture(scope="module")
def communicators():
    if "LOCAL_RANK" not in os.environ:
        pytest.skip("Run this file directly to launch the four-rank test")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("The serving specialization is enabled on SM100/SM103")
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=300))
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(4)), local_rank=rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    torch.cuda.set_stream(torch.cuda.Stream())
    device = torch.device("cuda", rank)
    plain = CustomAllReduceV2(coord.cpu_group, device)
    fused = CustomAllReduceV2(
        coord.cpu_group,
        device,
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * 1024 * 1024,
        max_push_blocks=512,
    )
    try:
        assert not plain.disabled and not fused.disabled
        register_comm(fused.obj)
        yield plain, fused
    finally:
        plain.close()
        fused.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("cluster", [1, 2, 5])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_changed_graph_inputs(communicators, cluster, weight_dtype):
    plain, fused = communicators
    torch.manual_seed(42 + dist.get_rank())
    hidden = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.randn_like(hidden) * 0.02
    shared = torch.randn_like(hidden)
    gemm2 = torch.randn(80, 2560, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(1, 10, device="cuda", dtype=weight_dtype)
    indices = torch.arange(10, device="cuda", dtype=torch.int32)

    def reference():
        routed = moe_finalize_fuse_shared(gemm2, indices, weights, None, 10, True)
        fused_gate_sigmoid_mul_add(hidden, gate_weight.squeeze(0), shared, routed)
        return plain.custom_all_reduce(routed)

    def candidate():
        gate = shared_expert_gate(hidden, gate_weight)
        return moe_finalize_shared_gate_all_reduce(
            gemm2, indices, weights, shared, gate, fused.obj, cluster_size=cluster
        )

    expected, actual = reference(), candidate()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The new gate parameter must also leave the existing ungated API intact.
    local = moe_finalize_fuse_shared(gemm2, indices, weights, None, 10, True) + shared
    expected_ungated = plain.custom_all_reduce(local)
    actual_ungated = moe_finalize_all_reduce(
        gemm2,
        indices,
        weights,
        10,
        shared,
        world_size=4,
        hidden_dim=2560,
        cluster_size=cluster,
    )
    torch.testing.assert_close(actual_ungated, expected_ungated, rtol=0, atol=0)
    with plain.capture():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            expected, actual = reference(), candidate()
    for step in range(5):
        hidden.normal_()
        shared.normal_()
        gemm2.normal_()
        weights.uniform_()
        indices.copy_(torch.randperm(80, device="cuda", dtype=torch.int32)[:10])
        if step % 2:
            indices[0] = -1
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
