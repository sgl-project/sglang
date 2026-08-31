"""Distributed exactness for producer-integrated DSV4 DCP A2A output.

CI launches DCP2. The same worker is intended for manual DCP4/DCP8 MI355X
validation with ``torchrun --nproc_per_node={4,8} <this file>``.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("aiter")

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.kernels.ops.attention.dcp_a2a import (
    DCPA2AOutputWorkspace,
    DCPA2APackedOutput,
    DCPDestinationPushOutput,
    DCPDestinationPushWorkspace,
    can_use_dsv4_dcp_destination_push_output,
)
from sglang.kernels.ops.attention.dcp_kernels import dcp_pack_a2a_send
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)
from sglang.srt.layers.dcp import (
    dcp_a2a_lse_reduce,
    dcp_registered_destination_push_lse_reduce,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=180, stage="sgl-kernel-unit", runner_config="2-gpu-amd")

HEADS = 128
HEAD_DIM = 512


def _launch_workers(nproc: int) -> None:
    result = subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", __file__],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"torchrun (nproc={nproc}) failed with rc={result.returncode}\n"
        f"{result.stdout}"
    )


def test_dsv4_dcp_direct_a2a_output_collective() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2 GPUs")
    _launch_workers(2)


def _init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size in (2, 4, 8)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")
    coordinator = parallel_state.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    return rank, world_size, device, coordinator


def _workspace(world_size: int, device: torch.device) -> DCPA2AOutputWorkspace:
    return DCPA2AOutputWorkspace.allocate(
        world_size=world_size,
        max_batch_size=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        device=device,
    )


def _decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    sink: torch.Tensor,
    *,
    output_workspace: DCPA2AOutputWorkspace | None = None,
    destination_push_workspace: DCPDestinationPushWorkspace | None = None,
):
    return _sparse_attn_v4_paged_decode_triton(
        q,
        kv,
        indices,
        indptr,
        sink,
        HEAD_DIM**-0.5,
        block_h=16,
        kv_splits=64,
        block_k=16,
        return_lse=True,
        attn_sink_logit_shift=-math.log(float(dist.get_world_size())),
        output_workspace=output_workspace,
        destination_push_workspace=destination_push_workspace,
    )


@torch.inference_mode()
def _worker_test(
    rank: int,
    world_size: int,
    device: torch.device,
    coordinator,
) -> None:
    generator = torch.Generator(device=device).manual_seed(20260829)
    q = torch.randn(
        (1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    local_kv_len = 80 + rank
    rank_generator = torch.Generator(device=device).manual_seed(3100 + rank)
    kv = torch.randn(
        (local_kv_len, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=rank_generator,
    )
    indices = torch.arange(local_kv_len, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, local_kv_len], dtype=torch.int32, device=device)
    sink = torch.randn(
        (HEADS,), dtype=torch.float32, device=device, generator=generator
    )

    old_out, old_lse = _decode(q, kv, indices, indptr, sink)
    expected_send = torch.empty(
        (world_size, 1, HEADS // world_size, HEAD_DIM + 2),
        dtype=torch.bfloat16,
        device=device,
    )
    dcp_pack_a2a_send(
        old_out,
        old_lse,
        expected_send[..., :HEAD_DIM],
        expected_send.view(torch.float32)[..., HEAD_DIM // 2],
    )

    direct_workspace = _workspace(world_size, device)
    packed = _decode(
        q,
        kv,
        indices,
        indptr,
        sink,
        output_workspace=direct_workspace,
    )
    assert isinstance(packed, DCPA2APackedOutput)
    assert torch.equal(
        direct_workspace.send_combined.view(torch.int16),
        expected_send.view(torch.int16),
    )

    old_workspace = _workspace(world_size, device)
    old_result = dcp_a2a_lse_reduce(
        old_out,
        old_lse,
        coordinator,
        cuda_graph_buffers={
            "send_combined": old_workspace.send_combined,
            "recv_combined": old_workspace.recv_combined,
        },
    )
    direct_result = dcp_a2a_lse_reduce(packed, None, coordinator)
    assert torch.equal(old_result.view(torch.int16), direct_result.view(torch.int16))

    push_workspace = None
    if hasattr(coordinator.ca_comm, "prepare_dsv4_destination_push_workspace"):
        registered_push = coordinator.prepare_dsv4_dcp_destination_push_workspace()
        assert registered_push is not None
        push_workspace = DCPDestinationPushWorkspace(
            recv_planes=registered_push[0],
            peer_recv_ptrs=registered_push[1],
            epoch=registered_push[2],
            source_rank=coordinator.rank_in_group,
        )
        assert can_use_dsv4_dcp_destination_push_output(
            q=q,
            unified_kv=kv,
            workspace=push_workspace,
            communicator=coordinator,
            feature_enabled=True,
            is_gfx950=True,
            dcp_size=world_size,
            comm_backend="a2a",
            is_plain_decode=True,
            batch_size=1,
            speculative=False,
            tbo=False,
            memory_saver=False,
            hisparse=False,
            state_capture=False,
            piecewise_graph=False,
            breakable_graph=False,
        )
        pushed = _decode(
            q,
            kv,
            indices,
            indptr,
            sink,
            destination_push_workspace=push_workspace,
        )
        assert isinstance(pushed, DCPDestinationPushOutput)
        push_result = dcp_registered_destination_push_lse_reduce(pushed, coordinator)
        assert torch.equal(old_result.view(torch.int16), push_result.view(torch.int16))

    graph_q = q.clone()
    graph_kv = kv.clone()
    graph_sink = sink.clone()
    baseline_workspace = _workspace(world_size, device)
    baseline_graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with coordinator.graph_capture() as graph_context:
        with torch.cuda.graph(baseline_graph, stream=graph_context.stream):
            graph_old_out, graph_old_lse = _decode(
                graph_q, graph_kv, indices, indptr, graph_sink
            )
            graph_old_result = dcp_a2a_lse_reduce(
                graph_old_out,
                graph_old_lse,
                coordinator,
                cuda_graph_buffers={
                    "send_combined": baseline_workspace.send_combined,
                    "recv_combined": baseline_workspace.recv_combined,
                },
            )

    graph_direct_workspace = _workspace(world_size, device)
    direct_graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with coordinator.graph_capture() as graph_context:
        with torch.cuda.graph(direct_graph, stream=graph_context.stream):
            graph_packed = _decode(
                graph_q,
                graph_kv,
                indices,
                indptr,
                graph_sink,
                output_workspace=graph_direct_workspace,
            )
            graph_direct_result = dcp_a2a_lse_reduce(graph_packed, None, coordinator)
    assert isinstance(graph_packed, DCPA2APackedOutput)

    push_graph = None
    if push_workspace is not None:
        push_graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with coordinator.graph_capture() as graph_context:
            with torch.cuda.graph(push_graph, stream=graph_context.stream):
                graph_pushed = _decode(
                    graph_q,
                    graph_kv,
                    indices,
                    indptr,
                    graph_sink,
                    destination_push_workspace=push_workspace,
                )
                graph_push_result = dcp_registered_destination_push_lse_reduce(
                    graph_pushed, coordinator
                )
        assert isinstance(graph_pushed, DCPDestinationPushOutput)

    for replay in range(3):
        replay_generator = torch.Generator(device=device).manual_seed(
            9000 + replay * world_size + rank
        )
        graph_q.copy_(
            torch.randn(
                graph_q.shape,
                dtype=graph_q.dtype,
                device=device,
                generator=replay_generator,
            )
        )
        graph_kv.copy_(
            torch.randn(
                graph_kv.shape,
                dtype=graph_kv.dtype,
                device=device,
                generator=replay_generator,
            )
        )
        graph_sink.copy_(
            torch.randn(
                graph_sink.shape,
                dtype=graph_sink.dtype,
                device=device,
                generator=replay_generator,
            )
        )
        baseline_graph.replay()
        direct_graph.replay()
        if push_graph is not None:
            push_graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            graph_old_result.view(torch.int16),
            graph_direct_result.view(torch.int16),
        )
        if push_graph is not None:
            assert torch.equal(
                graph_old_result.view(torch.int16),
                graph_push_result.view(torch.int16),
            )


def _worker_main() -> None:
    rank, world_size, device, coordinator = _init_distributed()
    try:
        _worker_test(rank, world_size, device, coordinator)
        dist.barrier(group=coordinator.cpu_group)
    finally:
        coordinator.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker_main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
