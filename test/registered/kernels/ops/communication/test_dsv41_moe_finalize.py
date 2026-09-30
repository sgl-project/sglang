import os
from types import SimpleNamespace

import flashinfer
import pytest
import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.all_reduce_fusion import moe_finalize_all_reduce
from sglang.kernels.ops.communication.all_reduce_mhc import (
    moe_finalize_all_reduce_mhc,
    moe_finalize_all_reduce_mhc_quant,
)
from sglang.kernels.ops.communication.all_reduce_mhc_combine import (
    moe_finalize_all_reduce_mhc_combine,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.distributed.parallel_state import init_world_group
from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as moe
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=400, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

ROUTING_CASES = [(6, torch.bfloat16), (6, torch.float32), (8, torch.float32)]


@pytest.fixture(scope="module")
def communicator():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    world = init_world_group(ranks=list(range(4)), local_rank=rank, backend="nccl")
    get_parallel().override_permanently(world_group=world)
    nccl = dist.new_group(backend="nccl")
    base = CustomAllReduceV2(
        world.cpu_group,
        torch.device("cuda", rank),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * 1024**2,
        max_push_blocks=512,
    )
    assert not base.disabled
    get_parallel().override_permanently(
        tp_group=SimpleNamespace(ca_comm=base, world_size=4)
    )
    assert moe._fused_finalize_all_reduce_comm_world_size() == 4
    comm = moe._fused_finalize_all_reduce_comm
    assert comm.max_push_size == 12 * 1024**2
    assert comm.config.num_push_blocks == 1024
    yield rank, nccl
    dist.barrier()
    comm.close()
    base.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("rows", [9, 96, 384, 385, 512, 1024])
@pytest.mark.parametrize("epilogue", ["plain", "post", "combine"])
@pytest.mark.parametrize("topk,weight_dtype", ROUTING_CASES)
def test_extended_finalize_replay(communicator, rows, epilogue, topk, weight_dtype):
    check_finalize(communicator, rows, epilogue, topk, weight_dtype)


@pytest.mark.parametrize("topk,weight_dtype", ROUTING_CASES)
def test_existing_quantized_epilogue_replay(communicator, topk, weight_dtype):
    check_finalize(communicator, 8, "quant", topk, weight_dtype)


def finalize_reference(expert_output, indices, weights, shared, group):
    rows, topk = weights.shape
    indices = indices.view(rows, topk)
    local = (
        expert_output[indices.clamp_min(0)].float()
        * (indices >= 0).unsqueeze(-1)
        * weights.unsqueeze(-1)
    ).sum(1)
    reduced = (local + shared.float()).bfloat16()
    dist.all_reduce(reduced, group=group)
    return reduced


def post_reference(reduced, residual, post_mix, residual_mix):
    updated = reduced.float().unsqueeze(1) * post_mix.unsqueeze(-1)
    for stream in range(4):
        updated = updated + residual[:, stream].float().unsqueeze(1) * residual_mix[
            :, stream
        ].unsqueeze(-1)
    return updated.bfloat16()


def check_finalize(communicator, rows, epilogue, topk, weight_dtype):
    rank, nccl = communicator
    hidden = 5120
    torch.manual_seed(123)
    expert_output = torch.full(
        (rows * topk, hidden), rank + 1, device="cuda", dtype=torch.bfloat16
    )
    indices = torch.arange(rows * topk, device="cuda", dtype=torch.int32)
    indices[::17] = -1
    weights = torch.full((rows, topk), 0.125, device="cuda", dtype=weight_dtype)
    shared = torch.full((rows, hidden), 0.25, device="cuda", dtype=torch.bfloat16)
    residual = (torch.randint(-4, 5, (rows, 4, hidden), device="cuda") / 4).bfloat16()
    post_mix = torch.tensor([0.125, 0.25, 0.5, 1.0], device="cuda").repeat(rows, 1)
    residual_mix = torch.tensor(
        [
            [0.5, 0.25, 0.0, 0.0],
            [0.0, 0.5, 0.25, 0.0],
            [0.0, 0.0, 0.5, 0.25],
            [0.25, 0.0, 0.0, 0.5],
        ],
        device="cuda",
    ).repeat(rows, 1, 1)
    pre_mix = torch.tensor([0.125, 0.25, 0.5, 0.125], device="cuda").repeat(rows, 1)
    norm_weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)

    def run():
        args = (expert_output, indices, weights, topk, shared)
        if epilogue == "plain":
            return (moe_finalize_all_reduce(*args, world_size=4, hidden_dim=hidden),)
        args += (residual, post_mix, residual_mix)
        if epilogue == "post":
            return moe_finalize_all_reduce_mhc(*args, world_size=4)
        if epilogue == "combine":
            return moe_finalize_all_reduce_mhc_combine(*args, pre_mix, world_size=4)
        assert epilogue == "quant"
        return moe_finalize_all_reduce_mhc_quant(
            *args, pre_mix, norm_weight, 1e-6, world_size=4
        )

    def check(got):
        reduced = finalize_reference(expert_output, indices, weights, shared, nccl)
        torch.testing.assert_close(got[0], reduced, rtol=0, atol=0)
        if epilogue == "plain":
            return
        updated = post_reference(reduced, residual, post_mix, residual_mix)
        torch.testing.assert_close(got[1], updated, rtol=0, atol=0)
        combined = (updated.float() * pre_mix.unsqueeze(-1)).sum(1).bfloat16()
        if epilogue == "combine":
            torch.testing.assert_close(got[2], combined, rtol=0, atol=0)
        elif epilogue == "quant":
            expected = flashinfer.norm.rmsnorm(combined, norm_weight, 1e-6)
            torch.testing.assert_close(got[2], expected, rtol=0.008, atol=1e-6)
            q, sf = flashinfer.mxfp8_quantize(got[2], is_sf_swizzled_layout=True)
            assert torch.equal(got[3].view(torch.uint8), q.view(torch.uint8))
            groups = torch.arange(hidden // 32, device="cuda")
            r = torch.arange(rows, device="cuda")[:, None]
            offsets = (
                (groups // 4) * 512 + ((r % 32) * 4 + (r // 32) % 4) * 4 + groups % 4
            )
            assert torch.equal(got[4].flatten()[offsets], sf.flatten()[offsets])

    for _ in range(3):
        check(run())
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        got = run()
    for iteration in range(6):
        expert_output.fill_(rank + 1 + iteration)
        shared.fill_(0.25 * (iteration + 1))
        residual.mul_(-1)
        graph.replay()
        check(got)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[4], timeout=600)
