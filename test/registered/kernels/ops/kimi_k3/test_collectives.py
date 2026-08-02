from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.kimi_k3 import (
    all_reduce,
    attn_res,
    gemm_ag,
    gemm_ar,
    sp_collective,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=480, suite="nightly-8-gpu-b200", nightly=True)

_HIDDEN_SIZE = 7168
_GEMM_AR_K_TOTAL = 12288
_GEMM_AG_WORLD_SIZE = 8
_MB = 1024 * 1024
_SP_TUNING = sp_collective.Tuning(num_blocks=1, block_size=256)


def _device():
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("Kimi K3 collectives require SM100+")


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
    cpu_group = coord.cpu_group
    assert isinstance(cpu_group, dist.ProcessGroup)
    nccl_group = dist.new_group(backend="nccl", device_id=_device())
    return cpu_group, nccl_group


@cache_once
def _init_comm():
    cpu_group, _ = _init_world()
    comm = CustomAllReduceV2(
        cpu_group,
        _device(),
        max_pull_size=4 * _MB,
        max_push_size=4 * _MB,
    )
    if comm.disabled or comm.mc_base_ptr == 0:
        raise RuntimeError("Kimi K3 collectives require multicast symmetric memory")
    all_reduce.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    sp_collective.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    attn_res.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    register_comm_cleanup(comm)
    return comm


@cache_once
def _init_gemm_ar():
    cpu_group, _ = _init_world()
    world_size = dist.get_world_size()
    gemm_ar.init(
        world_size=world_size,
        rank=dist.get_rank(),
        group=cpu_group,
        k=_GEMM_AR_K_TOTAL // world_size,
    )


def _symmetric_tensor(shape):
    from torch._C._distributed_c10d import _SymmetricMemory

    cpu_group, _ = _init_world()
    tensor = _SymmetricMemory.empty_strided_p2p(
        shape,
        torch.empty(shape).stride(),
        torch.bfloat16,
        _device(),
        cpu_group.group_name,
    )
    handle = _SymmetricMemory.rendezvous(tensor)
    rank = dist.get_rank()
    multicast_ptr = (
        int(handle.multicast_ptr) + tensor.data_ptr() - int(handle.buffer_ptrs[rank])
    )
    if multicast_ptr == 0:
        raise RuntimeError("symmetric tensor has no multicast mapping")
    return tensor, handle, multicast_ptr


@torch.inference_mode()
def test_all_reduce_push():
    _require_sm100()
    comm = _init_comm()
    rank = dist.get_rank()
    generator = torch.Generator().manual_seed(10 + rank)
    x = torch.randint(
        0,
        16,
        (_HIDDEN_SIZE,),
        generator=generator,
        dtype=torch.bfloat16,
    ).to(_device())
    residual = (
        torch.arange(_HIDDEN_SIZE, dtype=torch.int32, device=_device())
        .remainder_(7)
        .to(torch.bfloat16)
    )
    expected = x.clone()
    _, nccl_group = _init_world()
    dist.all_reduce(expected, group=nccl_group)
    expected += residual

    all_reduce.all_reduce_push_res(
        comm.world_size,
        x,
        residual,
        ws_mc_base=comm.mc_base_ptr,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_sequence_parallel_collectives():
    _require_sm100()
    comm = _init_comm()
    rank, world_size = dist.get_rank(), comm.world_size
    local_tokens = 2
    generator = torch.Generator(device="cuda").manual_seed(20 + rank)
    reduce_input = torch.randn(
        world_size * local_tokens,
        _HIDDEN_SIZE,
        generator=generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    residual = torch.randn(
        local_tokens,
        _HIDDEN_SIZE,
        generator=torch.Generator(device="cuda").manual_seed(21),
        device=_device(),
        dtype=torch.bfloat16,
    )
    expected_reduce = reduce_input.float()
    _, nccl_group = _init_world()
    dist.all_reduce(expected_reduce, group=nccl_group)
    lo = rank * local_tokens
    expected_reduce = (expected_reduce[lo : lo + local_tokens] + residual.float()).to(
        torch.bfloat16
    )
    reduce_output = torch.empty_like(expected_reduce)
    sp_collective.reduce_scatter_res(
        world_size,
        reduce_input,
        reduce_output,
        residual,
        tuning=_SP_TUNING,
    )

    gather_input = torch.randn(
        local_tokens,
        _HIDDEN_SIZE,
        generator=generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    expected_gather = torch.empty(
        world_size * local_tokens,
        _HIDDEN_SIZE,
        device=_device(),
        dtype=torch.bfloat16,
    )
    dist.all_gather_into_tensor(
        expected_gather,
        gather_input,
        group=nccl_group,
    )
    gather_output = torch.empty_like(expected_gather)
    sp_collective.all_gather(
        world_size,
        gather_input,
        gather_output,
        ws_mc_base=comm.mc_base_ptr,
        tuning=_SP_TUNING,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(reduce_output, expected_reduce, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(gather_output, expected_gather, rtol=0, atol=0)


@torch.inference_mode()
def test_gemm_all_gather():
    _require_sm100()
    if int(os.environ["WORLD_SIZE"]) != _GEMM_AG_WORLD_SIZE:
        pytest.skip("Kimi K3 gemm_ag is compiled for TP8")
    comm = _init_comm()
    generator = torch.Generator().manual_seed(30)
    x = (
        (torch.randn(1, gemm_ag.K, generator=generator) * 0.05)
        .to(torch.bfloat16)
        .to(_device())
    )
    weight = (
        (torch.randn(gemm_ag.N, gemm_ag.K, generator=generator) * 0.05)
        .to(torch.bfloat16)
        .to(_device())
    )
    bias = torch.randn(1, gemm_ag.N, generator=generator).to(
        device=_device(), dtype=torch.bfloat16
    )
    output = torch.empty(1, gemm_ag.N, device=_device(), dtype=torch.bfloat16)
    expected = (x.float() @ weight.float().t() + bias.float()).to(torch.bfloat16)

    gemm_ag.gemm_ag_up_proj(
        comm.world_size,
        x,
        weight,
        bias,
        None,
        output,
        ws_mc_base=comm.mc_base_ptr,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)


@torch.inference_mode()
def test_gemm_all_reduce():
    _require_sm100()
    _init_gemm_ar()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_k = _GEMM_AR_K_TOTAL // world_size
    generator = torch.Generator().manual_seed(40 + rank)
    x = torch.randn(1, local_k, generator=generator).to(
        device=_device(), dtype=torch.bfloat16
    )
    weight = torch.randn(gemm_ar.N, local_k, generator=generator).to(
        device=_device(), dtype=torch.bfloat16
    )
    expected = (x.float() @ weight.float().t()).to(torch.bfloat16).float()
    _, nccl_group = _init_world()
    dist.all_reduce(expected, group=nccl_group)

    output = gemm_ar.o_proj_gemm_ar(x, weight)
    torch.cuda.synchronize()
    bad = ((output.float() - expected).abs() > 0.05 + 0.02 * expected.abs()).sum()
    assert bad.item() <= output.numel() / 1000


@torch.inference_mode()
def test_attention_residual_direct_all_gather():
    _require_sm100()
    comm = _init_comm()
    rank, local_tokens, num_bank_rows = dist.get_rank(), 2, 3
    generator = torch.Generator(device="cuda").manual_seed(50 + rank)
    prefix = torch.randn(
        local_tokens,
        _HIDDEN_SIZE,
        generator=generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    bank = torch.randn(
        local_tokens,
        num_bank_rows + 1,
        _HIDDEN_SIZE,
        generator=generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    combine_weight = torch.linspace(
        -0.01, 0.01, _HIDDEN_SIZE, device=_device(), dtype=torch.bfloat16
    )
    output_weight = torch.linspace(
        1.25, 0.75, _HIDDEN_SIZE, device=_device(), dtype=torch.bfloat16
    )
    local_reference = torch.empty_like(prefix)
    attn_res.attn_res_fused_tma(
        prefix,
        bank.clone(),
        combine_weight,
        output_weight,
        local_reference,
        num_bank_rows,
        1e-6,
    )
    full_reference = torch.empty(
        comm.world_size * local_tokens,
        _HIDDEN_SIZE,
        device=_device(),
        dtype=torch.bfloat16,
    )
    _, nccl_group = _init_world()
    dist.all_gather_into_tensor(
        full_reference,
        local_reference,
        group=nccl_group,
    )

    output, handle, multicast_ptr = _symmetric_tensor(tuple(full_reference.shape))
    attn_res.attn_res_fused_direct_ag(
        comm.world_size,
        prefix,
        bank,
        combine_weight,
        output_weight,
        output,
        num_bank_rows,
        1e-6,
        output_mc_ptr=multicast_ptr,
        max_blocks=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, full_reference, rtol=2e-2, atol=3e-2)
    assert handle is not None


def _precompile(num_gpus):
    for world_size in num_gpus:
        all_reduce._jit_module(world_size)
        sp_collective._jit_module(world_size)
        gemm_ar._jit_module(_GEMM_AR_K_TOTAL // world_size, world_size)
    if _GEMM_AG_WORLD_SIZE in num_gpus:
        gemm_ag._jit_module()
    attn_res._jit_fused_tma_module(4, 1, 200)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(4, 8),
        pre_launch_fn=_precompile,
    )
