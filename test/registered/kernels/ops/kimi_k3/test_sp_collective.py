"""Direct correctness tests for the Kimi-K3 SP collectives.

The test keeps only the reference checks from the original tuning benchmark:
every kernel entry point is compared with NCCL/PyTorch, without timing,
candidate sweeps, serving-layer imports, or generated configuration output.
"""

from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.kimi_k3 import attn_res, sp_collective
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 7168
_TUNING = sp_collective.Tuning(num_blocks=1, block_size=256)


def _device() -> torch.device:
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


@cache_once
def _init_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    atexit.register(dist.destroy_process_group)
    nccl_group = dist.new_group(backend="nccl", device_id=_device())
    return dist.group.WORLD, nccl_group


@cache_once
def _init_comm() -> CustomAllReduceV2:
    cpu_group, _ = _init_world()
    comm = CustomAllReduceV2(
        cpu_group,
        _device(),
        max_pull_size=4 * 1024 * 1024,
        max_push_size=4 * 1024 * 1024,
    )
    if comm.disabled or comm.mc_base_ptr == 0:
        raise RuntimeError("SP collectives require multicast symmetric memory")
    sp_collective.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    attn_res.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    register_comm_cleanup(comm)
    return comm


def _symmetric_tensor(shape: tuple[int, ...]) -> tuple[torch.Tensor, object, int]:
    from torch._C._distributed_c10d import _SymmetricMemory

    cpu_group, _ = _init_world()
    stride = torch.empty(shape).stride()
    tensor = _SymmetricMemory.empty_strided_p2p(
        shape,
        stride,
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
        raise RuntimeError("symmetric tensor does not have a multicast mapping")
    return tensor, handle, multicast_ptr


def _make_inputs():
    comm = _init_comm()
    rank = dist.get_rank()
    world_size = comm.world_size
    local_tokens = 8
    num_tokens = world_size * local_tokens

    generator = torch.Generator(device="cuda").manual_seed(100 + rank)
    reduce_scatter_input = torch.randn(
        num_tokens,
        _HIDDEN_SIZE,
        generator=generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    residual_generator = torch.Generator(device="cuda").manual_seed(777)
    full_residual = torch.randn(
        num_tokens,
        _HIDDEN_SIZE,
        generator=residual_generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    lo = rank * local_tokens
    local_residual = full_residual[lo : lo + local_tokens].contiguous()

    reduce_scatter_sum = reduce_scatter_input.float()
    _, nccl_group = _init_world()
    dist.all_reduce(reduce_scatter_sum, group=nccl_group)
    reduce_scatter_reference = (
        reduce_scatter_sum[lo : lo + local_tokens] + local_residual.float()
    ).to(torch.bfloat16)

    gather_generator = torch.Generator(device="cuda").manual_seed(900 + rank)
    all_gather_input = torch.randn(
        local_tokens,
        _HIDDEN_SIZE,
        generator=gather_generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    all_gather_reference = torch.empty(
        num_tokens,
        _HIDDEN_SIZE,
        device=_device(),
        dtype=torch.bfloat16,
    )
    dist.all_gather_into_tensor(
        all_gather_reference, all_gather_input, group=nccl_group
    )
    return (
        reduce_scatter_input,
        local_residual,
        reduce_scatter_reference,
        all_gather_input,
        all_gather_reference,
    )


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("Kimi-K3 SP collectives require SM100+")


@torch.inference_mode()
def test_push_reduce_scatter_and_all_gather():
    _require_sm100()
    comm = _init_comm()
    rs_input, residual, rs_ref, ag_input, ag_ref = _make_inputs()

    rs_output = torch.empty_like(rs_ref)
    sp_collective.reduce_scatter_res(
        comm.world_size,
        rs_input,
        rs_output,
        residual,
        tuning=_TUNING,
    )
    ag_output = torch.empty_like(ag_ref)
    sp_collective.all_gather(
        comm.world_size,
        ag_input,
        ag_output,
        ws_mc_base=comm.mc_base_ptr,
        tuning=_TUNING,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(rs_output, rs_ref, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(ag_output, ag_ref, rtol=0, atol=0)


@torch.inference_mode()
def test_pull_reduce_scatter_and_direct_all_gather():
    _require_sm100()
    comm = _init_comm()
    rs_input, residual, rs_ref, ag_input, ag_ref = _make_inputs()

    symmetric_input, input_handle, input_mc_ptr = _symmetric_tensor(
        tuple(rs_input.shape)
    )
    symmetric_input.copy_(rs_input)
    rs_output = torch.empty_like(rs_ref)
    sp_collective.reduce_scatter_pull(
        comm.world_size,
        symmetric_input,
        rs_output,
        residual,
        input_mc_ptr=input_mc_ptr,
        tuning=_TUNING,
    )

    ag_output, output_handle, output_mc_ptr = _symmetric_tensor(tuple(ag_ref.shape))
    sp_collective.all_gather_direct(
        comm.world_size,
        ag_input,
        ag_output,
        output_mc_ptr=output_mc_ptr,
        tuning=_TUNING,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(rs_output, rs_ref, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(ag_output, ag_ref, rtol=0, atol=0)
    assert input_handle is not None
    assert output_handle is not None


def _attn_res_inputs(num_tokens: int, num_bank_rows: int):
    rank = dist.get_rank()
    generator = torch.Generator(device="cuda").manual_seed(1200 + rank)
    prefix = torch.randn(
        (num_tokens, _HIDDEN_SIZE),
        generator=generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    bank = torch.randn(
        (num_tokens, num_bank_rows, _HIDDEN_SIZE),
        generator=generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    cw = torch.linspace(
        -0.01, 0.01, _HIDDEN_SIZE, device=_device(), dtype=torch.bfloat16
    )
    ow = torch.linspace(
        1.25, 0.75, _HIDDEN_SIZE, device=_device(), dtype=torch.bfloat16
    )
    return prefix, bank, cw, ow


@torch.inference_mode()
def test_attention_residual_direct_all_gather():
    _require_sm100()
    comm = _init_comm()
    local_tokens, nvb, eps = 8, 3, 1e-6
    prefix, bank, cw, ow = _attn_res_inputs(local_tokens, nvb + 1)

    local_reference = torch.empty_like(prefix)
    attn_res.attn_res_fused_tma(prefix, bank.clone(), cw, ow, local_reference, nvb, eps)
    full_reference = torch.empty(
        (comm.world_size * local_tokens, _HIDDEN_SIZE),
        device=_device(),
        dtype=torch.bfloat16,
    )
    _, nccl_group = _init_world()
    dist.all_gather_into_tensor(full_reference, local_reference, group=nccl_group)

    output, output_handle, output_mc_ptr = _symmetric_tensor(
        tuple(full_reference.shape)
    )
    attn_res.attn_res_fused_direct_ag(
        comm.world_size,
        prefix,
        bank,
        cw,
        ow,
        output,
        nvb,
        eps,
        output_mc_ptr=output_mc_ptr,
        max_blocks=4,
        write_prefix=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, full_reference, rtol=2e-2, atol=3e-2)
    assert torch.equal(bank[:, nvb], prefix)
    assert output_handle is not None


@torch.inference_mode()
def test_attention_residual_pull_reduce_scatter():
    _require_sm100()
    comm = _init_comm()
    rs_input, residual, rs_ref, _, _ = _make_inputs()
    local_tokens, nvb, eps = rs_ref.shape[0], 3, 1e-6
    _, bank, cw, ow = _attn_res_inputs(local_tokens, nvb + 1)

    expected = torch.empty_like(rs_ref)
    attn_res.attn_res_fused_tma(rs_ref, bank.clone(), cw, ow, expected, nvb, eps)

    symmetric_input, input_handle, input_mc_ptr = _symmetric_tensor(
        tuple(rs_input.shape)
    )
    symmetric_input.copy_(rs_input)
    output = torch.empty_like(rs_ref)
    prefix_output = torch.empty_like(rs_ref)
    attn_res.attn_res_fused_pull_rs(
        comm.world_size,
        symmetric_input,
        residual,
        bank,
        cw,
        ow,
        output,
        prefix_output,
        nvb,
        eps,
        input_mc_ptr=input_mc_ptr,
        max_blocks=4,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(prefix_output, rs_ref, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=3e-2)
    assert input_handle is not None


def _precompile(num_gpus):
    for world_size in num_gpus:
        sp_collective._jit_module(world_size)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(4,),
        pre_launch_fn=_precompile,
    )
