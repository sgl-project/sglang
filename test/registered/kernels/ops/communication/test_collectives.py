from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention import attn_res
from sglang.kernels.ops.communication import (
    attn_res_collective,
    gemm_ag,
    kimi_k3_comm,
    nvlink_comm,
    sp_collective,
)
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.communication.k3_ar_fusion import _SHARDED_UP_MAX_TOKENS
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=45, stage="base-c", runner_config="4-gpu-b200")
register_cuda_ci(est_time=480, stage="nightly", runner_config="8-gpu-b200")


def _rows(t):
    """nvlink_comm's [tokens, hidden] view: K3 rows where they divide, else 16-byte rows."""
    if t is None:
        return None
    return t.view(-1, 3584) if t.numel() % 3584 == 0 else t.view(-1, 8)


def _push_res(comm, x, residual):
    """In place: ``x = allreduce(x) [+ residual]``, 1-shot push (K3's small-message path)."""
    nvlink_comm.all_reduce_push(comm.obj, _rows(x), _rows(x), _rows(residual))


_HIDDEN_SIZE = 7168
# The GEMV table's two ends, the first batch past it (slice GEMM + gather_add3 only),
# and the model's gather cap, where the consumer reaches its widest block and strides.
_GEMM_AG_NUM_TOKENS = (
    1,
    gemm_ag.GEMV_MAX_TOKENS,
    gemm_ag.GEMV_MAX_TOKENS + 1,
    _SHARDED_UP_MAX_TOKENS,
)
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
    kimi_k3_comm.register_comm(lambda: comm.obj, world_size=comm.world_size)
    sp_collective.register_comm(lambda: comm.obj, world_size=comm.world_size)
    attn_res_collective.register_comm(lambda: comm.obj, world_size=comm.world_size)
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

    _push_res(comm, x, residual)
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


@cache_once
def _gemm_ag_operands():
    generator = torch.Generator().manual_seed(30)
    max_tokens = max(_GEMM_AG_NUM_TOKENS)
    x = (torch.randn(max_tokens, gemm_ag.K, generator=generator) * 0.05).to(
        torch.bfloat16
    )
    weight = (torch.randn(gemm_ag.N, gemm_ag.K, generator=generator) * 0.05).to(
        torch.bfloat16
    )
    b = torch.randn(max_tokens, gemm_ag.N, generator=generator).to(torch.bfloat16)
    c = torch.randn(max_tokens, gemm_ag.N, generator=generator).to(torch.bfloat16)
    return tuple(t.to(_device()) for t in (x, weight, b, c))


@pytest.mark.parametrize("with_c", [False, True])
@pytest.mark.parametrize("num_tokens", _GEMM_AG_NUM_TOKENS)
@torch.inference_mode()
def test_gemm_all_gather(num_tokens, with_c):
    _require_sm100()
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in gemm_ag.WORLD_SIZES:
        pytest.skip(f"Kimi K3 gemm_ag is built for TP{gemm_ag.WORLD_SIZES}")
    _init_comm()
    _, nccl_group = _init_world()
    x, weight, b, c = _gemm_ag_operands()
    x, b = x[:num_tokens], b[:num_tokens]
    c = c[:num_tokens] if with_c else None
    output = torch.empty(num_tokens, gemm_ag.N, device=_device(), dtype=torch.bfloat16)

    if num_tokens <= gemm_ag.GEMV_MAX_TOKENS:
        expected = x.float() @ weight.float().t() + b.float()
        if c is not None:
            expected += c.float()
        gemm_ag.gemm_ag_up_proj(world_size, x, weight, b, c, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, expected.to(torch.bfloat16), rtol=3e-2, atol=3e-2
        )

    # Slice GEMM + gather: exact against the slices gathered by NCCL, with the
    # replicated tail's bf16(bf16(up_proj + b) + c) rounding.
    rows = gemm_ag.N // world_size
    rank = dist.get_rank()
    local = torch.nn.functional.linear(x, weight[rank * rows : (rank + 1) * rows])
    slices = torch.empty(
        world_size, num_tokens, rows, device=_device(), dtype=torch.bfloat16
    )
    dist.all_gather_into_tensor(slices, local, group=nccl_group)
    expected = slices.permute(1, 0, 2).reshape(num_tokens, gemm_ag.N) + b
    if c is not None:
        expected = expected + c
    for _ in range(3):  # each round flips the push-plane phase
        output.fill_(float("nan"))
        gemm_ag.gather_add3(world_size, local, b, c, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


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


# One row; fewer rows than some ranks' share of an even split; an even split; a
# ragged split with several rows a block, so the chunk slots are reused.
_STREAM_NUM_TOKENS = (1, 5, 64, 1027)
# (chunk_rows, occupancy, consumer_regs), bank rows: one chunk that ends with the
# prefix row, the prefix row in a chunk of its own, bank rows in both chunks, and
# the two-blocks-an-SM config of one bank row.
_STREAM_CONFIGS = (
    ((4, 1, 200), 3),
    ((4, 1, 200), 4),
    ((4, 1, 200), 6),
    ((2, 2, 0), 1),
)


@cache_once
def _stream_operands():
    """What every rank holds a copy of, and this rank's two symmetric buffers."""
    max_tokens, max_bank_rows = max(_STREAM_NUM_TOKENS), 6
    generator = torch.Generator().manual_seed(60)
    bank = torch.randn(max_tokens, max_bank_rows, _HIDDEN_SIZE, generator=generator)
    residual = torch.randn(max_tokens, _HIDDEN_SIZE, generator=generator)
    replicated = tuple(
        t.to(_device(), torch.bfloat16)
        for t in (
            bank,
            residual,
            torch.linspace(-0.01, 0.01, _HIDDEN_SIZE),
            torch.linspace(1.25, 0.75, _HIDDEN_SIZE),
        )
    )
    x, x_handle, x_mc = _symmetric_tensor((max_tokens, _HIDDEN_SIZE))
    stream, stream_handle, stream_mc = _symmetric_tensor((max_tokens, _HIDDEN_SIZE))
    _stream_words(stream).copy_(_STREAM_EMPTY)
    return replicated, (x, x_mc, stream, stream_mc), (x_handle, stream_handle)


# The stream marker: 0x80000000 in the first 32-bit word of every 16 bytes.
_STREAM_EMPTY = torch.tensor([-(2**31), 0, 0, 0], dtype=torch.int32)


def _stream_words(stream):
    return stream.reshape(-1).view(torch.int32).view(-1, 4)


@pytest.mark.parametrize("with_residual", (False, True))
@pytest.mark.parametrize("config,num_bank_rows", _STREAM_CONFIGS)
@pytest.mark.parametrize("num_tokens", _STREAM_NUM_TOKENS)
@torch.inference_mode()
def test_attention_residual_stream_all_reduce(
    num_tokens, config, num_bank_rows, with_residual
):
    """The aggregation beside the streamed pull all-reduce, against the in-place
    pull all-reduce followed by the aggregation."""
    _require_sm100()
    comm = _init_comm()
    (bank, residual, cw, ow), (x, x_mc, stream, stream_mc), _ = _stream_operands()
    bank, x, stream = bank[:num_tokens], x[:num_tokens], stream[:num_tokens]
    residual = residual[:num_tokens] if with_residual else None
    local = attn_res._jit_fused_tma_module(*config)
    streamed = attn_res._jit_fused_stream_module(*config)
    partial = torch.randn(
        num_tokens,
        _HIDDEN_SIZE,
        generator=torch.Generator().manual_seed(61 + dist.get_rank()),
    ).to(_device(), torch.bfloat16)
    if not with_residual:
        # (+0.0, -0.0) pairs sum to the marker's own bits: a delivered row must
        # not read as one that has not arrived.
        partial[0].view(-1, 2).copy_(torch.tensor([0.0, -0.0]))
        partial[-1].view(-1, 2).copy_(torch.tensor([0.0, -0.0]))

    x.copy_(partial)
    nvlink_comm.all_reduce_pull(
        comm.obj, x, x, residual, in_mc_ptr=x_mc, out_mc_ptr=x_mc
    )
    expected_prefix = x.clone()
    expected = torch.empty_like(x)
    local.run(expected_prefix, bank, cw, ow, expected, num_bank_rows, 1e-6, False)

    for _ in range(3):  # every round finds the stream as the round before left it
        x.copy_(partial)
        output = torch.full_like(x, float("nan"))
        blocks = nvlink_comm.all_reduce_pull_stream(
            comm.obj, x, stream, residual, in_mc_ptr=x_mc, out_mc_ptr=stream_mc
        )
        streamed.run(
            stream,
            x,
            bank,
            cw,
            ow,
            output,
            num_bank_rows,
            1e-6,
            False,
            comm.world_size,
            blocks,
        )
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(x, expected_prefix)  # as values: a -0.0 may come as +0.0
        assert (_stream_words(stream) == _STREAM_EMPTY.to(stream.device)).all()


def _precompile(num_gpus):
    for world_size in num_gpus:
        nvlink_comm._jit_push_module(torch.bfloat16, world_size)
        sp_collective._jit_module(world_size)
        if world_size in gemm_ag.WORLD_SIZES:
            gemm_ag._jit_module(world_size)
    nvlink_comm._jit_pull_module(torch.bfloat16, 4)
    for config in {config for config, _ in _STREAM_CONFIGS}:
        attn_res._jit_fused_tma_module(*config)
        attn_res._jit_fused_stream_module(*config)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(4, 8),
        pre_launch_fn=_precompile,
    )
