"""Correctness and CUDA-graph tests for direct-final DCP Query publication.

The source-reproduction geometry matches the vLLM Kimi-K3 Query experiment
(six local heads on DCP4). The TP4/DCP4 geometry matches this repository's
four-GPU Kimi-K3 development target (24 local heads, 96 gathered heads).

Usage::

    python test/registered/kernels/ops/communication/test_dcp_direct_final_query.py \
      --num-gpu 4
"""

from __future__ import annotations

import atexit
import functools
import logging
import os

import pytest
import sglang.srt.distributed.parallel_state as ps
import torch
import torch.distributed as dist
from sglang.kernels.jit.utils import get_ci_test_range
from sglang.srt.layers.dcp.query import DCPDirectFinalQueryGatherer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=180, stage="extra-b", runner_config="4-gpu-h200")

NOPE_DIM = 512
ROPE_DIM = 64
MAX_TOKENS = 128
TEST_TOKENS = get_ci_test_range([1, 8, 17, 32, 128], [1, 17, 128])
TEST_LOCAL_HEADS = get_ci_test_range([6, 24], [6, 24])


@functools.cache
def _init_world() -> ps.GroupCoordinator:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    assert ps._WORLD is not None
    return ps._WORLD


@functools.cache
def _gatherer(local_heads: int) -> DCPDirectFinalQueryGatherer:
    world = _init_world()
    return DCPDirectFinalQueryGatherer(
        group=world,
        max_tokens=MAX_TOKENS,
        local_heads=local_heads,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        device=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"),
    )


def _reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, local_heads, _ = q_nope.shape
    local = torch.cat((q_nope, q_rope), dim=-1).reshape(num_tokens, -1)
    gathered = torch.empty(
        world_size * num_tokens,
        local.shape[-1],
        dtype=local.dtype,
        device=local.device,
    )
    dist.all_gather_into_tensor(gathered, local, group=group)
    final = (
        gathered.view(world_size, num_tokens, local.shape[-1])
        .movedim(0, 1)
        .reshape(num_tokens, world_size * local_heads, NOPE_DIM + ROPE_DIM)
    )
    return final.split((NOPE_DIM, ROPE_DIM), dim=-1)


def _assert_query_equal(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def _make_production_layout_inputs(
    *,
    num_tokens: int,
    local_heads: int,
    nope_value: int,
    rope_value: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The absorb BMM produces [H,T,512] and forward_mla transposes it without
    # materializing. RoPE is a tail view of a packed [T,H,192] Query.
    nope_storage = torch.full(
        (local_heads, num_tokens, NOPE_DIM),
        fill_value=nope_value,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_storage = torch.full(
        (num_tokens, local_heads, 128 + ROPE_DIM),
        fill_value=rope_value,
        dtype=torch.bfloat16,
        device=device,
    )
    return nope_storage.transpose(0, 1), rope_storage[..., -ROPE_DIM:]


@pytest.mark.parametrize("local_heads", TEST_LOCAL_HEADS)
@pytest.mark.parametrize("num_tokens", TEST_TOKENS)
@torch.inference_mode()
def test_dcp_direct_final_query_eager_and_reuse(
    local_heads: int,
    num_tokens: int,
) -> None:
    world = _init_world()
    if world.world_size != 4:
        pytest.skip("Kimi-K3 source and TP4/DCP4 geometries require four ranks.")
    gatherer = _gatherer(local_heads)
    if gatherer.state.symm_mem_hdl.multicast_ptr == 0:
        pytest.skip("NVLS multicast is unavailable.")

    device = gatherer.state.device
    for step in range(8):
        dist.barrier(world.device_group)
        q_nope, q_rope = _make_production_layout_inputs(
            num_tokens=num_tokens,
            local_heads=local_heads,
            nope_value=step * world.world_size + world.rank_in_group,
            rope_value=100 + step * world.world_size + world.rank_in_group,
            device=device,
        )
        expected = _reference(q_nope, q_rope, world.device_group, world.world_size)
        actual = gatherer(q_nope, q_rope)
        _assert_query_equal(actual, expected)


@pytest.mark.parametrize("local_heads", TEST_LOCAL_HEADS)
@pytest.mark.parametrize("num_tokens", [1, 17])
@torch.inference_mode()
def test_dcp_direct_final_query_changing_input_cuda_graph(
    local_heads: int,
    num_tokens: int,
) -> None:
    world = _init_world()
    if world.world_size != 4:
        pytest.skip("Kimi-K3 source and TP4/DCP4 geometries require four ranks.")
    gatherer = _gatherer(local_heads)
    if gatherer.state.symm_mem_hdl.multicast_ptr == 0:
        pytest.skip("NVLS multicast is unavailable.")

    device = gatherer.state.device
    static_nope_storage = torch.empty(
        (local_heads, num_tokens, NOPE_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    static_rope_storage = torch.empty(
        (num_tokens, local_heads, 128 + ROPE_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    static_nope = static_nope_storage.transpose(0, 1)
    static_rope = static_rope_storage[..., -ROPE_DIM:]

    # Compile Triton and initialize all lazy CUDA state outside capture.
    static_nope.fill_(world.rank_in_group)
    static_rope_storage.fill_(100 + world.rank_in_group)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(world.device_group)
    with torch.cuda.stream(warmup_stream):
        gatherer(static_nope, static_rope)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    dist.barrier(world.device_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = gatherer(static_nope, static_rope)

    for step in range(8):
        static_nope.fill_(step * world.world_size + world.rank_in_group)
        static_rope_storage.fill_(100 + step * world.world_size + world.rank_in_group)
        dist.barrier(world.device_group)
        graph.replay()
        expected = _reference(
            static_nope, static_rope, world.device_group, world.world_size
        )
        _assert_query_equal(graph_output, expected)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
