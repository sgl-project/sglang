from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention import attn_res
from sglang.kernels.ops.communication import all_reduce_residual as all_reduce
from sglang.kernels.ops.communication import (
    attn_res_collective,
    gemm_ag,
    sp_collective,
)
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=21, stage="base-c", runner_config="4-gpu-b200")
register_cuda_ci(est_time=480, stage="nightly", runner_config="8-gpu-b200")

_HIDDEN_SIZE = 7168
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
    get_parallel().override_permanently(world_group=coord)
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
    if comm.disabled or not comm.has_multicast:
        raise RuntimeError("Kimi K3 collectives require multicast symmetric memory")
    all_reduce.register_comm(comm.obj)
    sp_collective.register_comm(comm.obj)
    attn_res_collective.register_comm(comm.obj)
    register_comm_cleanup(comm)
    return comm


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
        tuning=_SP_TUNING,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(reduce_output, expected_reduce, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(gather_output, expected_gather, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("with_prefix", [False, True])
@torch.inference_mode()
def test_gemm_all_gather(num_tokens, with_prefix):
    _require_sm100()
    if int(os.environ["WORLD_SIZE"]) != _GEMM_AG_WORLD_SIZE:
        pytest.skip("Kimi K3 gemm_ag is compiled for TP8")
    comm = _init_comm()
    generator = torch.Generator(device=_device()).manual_seed(30)
    x = (
        torch.randn(
            num_tokens,
            gemm_ag.K,
            generator=generator,
            device=_device(),
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    weight = (
        torch.randn(
            gemm_ag.N,
            gemm_ag.K,
            generator=generator,
            device=_device(),
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    bias = torch.randn(
        num_tokens,
        gemm_ag.N,
        generator=generator,
        device=_device(),
        dtype=torch.bfloat16,
    )
    prefix = torch.randn_like(bias) if with_prefix else None
    local_weight = weight.chunk(comm.world_size)[dist.get_rank()].clone()
    output = torch.empty_like(bias)
    local_output = torch.empty_like(bias)

    def run():
        gemm_ag.gemm_ag_up_proj(comm.world_size, x, weight, bias, prefix, output)
        gemm_ag.gemm_ag_up_proj(
            comm.world_size, x, local_weight, bias, prefix, local_output
        )

    def check():
        expected = (x.float() @ weight.float().t()).to(torch.bfloat16) + bias
        if prefix is not None:
            expected = expected + prefix
        torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(local_output, output, rtol=0, atol=0)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        run()
    for _ in range(5):
        x.normal_(std=0.05, generator=generator)
        bias.normal_(generator=generator)
        graph.replay()
        torch.cuda.synchronize()
        check()


@torch.inference_mode()
def test_kimi_latent_up_projection():
    _require_sm100()
    if int(os.environ["WORLD_SIZE"]) != _GEMM_AG_WORLD_SIZE:
        pytest.skip("Kimi K3 latent-up sharding is enabled for TP8")
    _init_world()
    from sglang.srt.models.kimi_k3 import KimiK3LatentUpProjection

    group = get_parallel().world_group
    generator = torch.Generator(device=_device()).manual_seed(31)
    weight = (
        torch.randn(
            gemm_ag.N,
            gemm_ag.K,
            generator=generator,
            device=_device(),
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    with get_parallel().override(tp_group=group, tp_size=8, nnodes=1), _device():
        layer = KimiK3LatentUpProjection(
            gemm_ag.K,
            gemm_ag.N,
            bias=False,
            params_dtype=torch.bfloat16,
            parallel_group="tp",
        )
        layer.weight.weight_loader(layer.weight, weight)
        torch.testing.assert_close(
            layer.weight, weight.chunk(8)[group.rank_in_group], rtol=0, atol=0
        )
        assert layer.weight.untyped_storage().nbytes() == weight.numel() * 2 // 8
        # Include zero rows, the GEMV boundary, odd/strided input, the multicast
        # limit and the larger-prefill fallback. Consume scratch before reuse.
        for rows in (0, 1, 12, 13, 219, 267, 512, 513):
            storage = torch.randn(
                rows,
                gemm_ag.K + 8,
                generator=generator,
                device=_device(),
                dtype=torch.bfloat16,
            )
            x = storage[:, : gemm_ag.K]

            def run():
                return layer(x)[0].clone()

            for _ in range(3):
                output = run()
            expected = torch.nn.functional.linear(x, weight)
            torch.testing.assert_close(output, expected, rtol=1e-2, atol=2e-2)
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                output = run()
            for _ in range(5):
                storage.normal_(generator=generator)
                graph.replay()
                torch.cuda.synchronize()
                expected = torch.nn.functional.linear(x, weight)
                torch.testing.assert_close(output, expected, rtol=1e-2, atol=2e-2)


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
    attn_res_collective.attn_res_fused_direct_ag(
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
