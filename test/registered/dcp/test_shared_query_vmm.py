"""Correctness and graph replay for consumer-direct Shared-DCP Query."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=150,
    stage="base-b",
    runner_config="4-gpu-b200",
)

WORLD_SIZE = int(os.environ.get("DCP_TEST_WORLD_SIZE", "4"))
ROWS = (1, 8, 32, 64, 128)
LOCAL_HEADS = 16
NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM


def _launch_worker() -> None:
    result = subprocess.run(
        ["torchrun", f"--nproc_per_node={WORLD_SIZE}", __file__],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_shared_query_vmm() -> None:
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"Requires {WORLD_SIZE} GPUs")
    _launch_worker()


def _make_inputs(
    rows: int,
    rank: int,
    step: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(10_000 + rank * 97 + step)
    q_nope = torch.randn(
        rows,
        LOCAL_HEADS,
        NOPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    q_rope = torch.randn(
        rows,
        LOCAL_HEADS,
        ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_nope = torch.randn(
        rows,
        NOPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_rope = torch.randn(
        rows,
        ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(rows, dtype=torch.int64, device="cuda") * 3 + 1

    frequency = torch.arange(ROPE_DIM // 2, device="cuda", dtype=torch.float32)
    frequency = 1.0 / (10_000 ** (frequency / (ROPE_DIM // 2)))
    angles = torch.outer(
        torch.arange(ROWS[-1] * 3 + 2, device="cuda", dtype=torch.float32),
        frequency,
    )
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return q_nope, q_rope, k_nope, k_rope, positions, cos_sin_cache


def _gather_query_shard(
    local: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    rows, local_heads, dim = local.shape
    gathered = torch.empty(
        WORLD_SIZE * rows,
        local_heads,
        dim,
        dtype=local.dtype,
        device=local.device,
    )
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    return (
        gathered.view(WORLD_SIZE, rows, local_heads, dim)
        .permute(1, 0, 2, 3)
        .reshape(rows, WORLD_SIZE * local_heads, dim)
    )


def _reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8

    gathered_q_nope = _gather_query_shard(q_nope, group)
    gathered_q_rope = _gather_query_shard(q_rope, group)
    return mla_quantize_and_rope_for_fp8(
        gathered_q_nope,
        gathered_q_rope,
        k_nope,
        k_rope,
        positions,
        cos_sin_cache,
        is_neox,
        NOPE_DIM,
        ROPE_DIM,
    )


def _assert_fp8_byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        actual.view(torch.uint8),
        expected.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def _run_sparse_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    result = trtllm_batch_decode_with_kv_cache_mla(
        query=query.unsqueeze(1),
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        block_tables=topk_indices,
        seq_lens=seq_lens,
        max_seq_len=topk_indices.shape[-1],
        sparse_mla_top_k=topk_indices.shape[-1],
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        return_lse=True,
        backend="trtllm-gen",
    )
    assert isinstance(result, tuple)
    return result


def _worker() -> None:
    from sglang.srt.layers.dcp.shared_query_direct import (
        create_dcp_query_direct_vmm_workspace,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(WORLD_SIZE)),
        local_rank=rank,
        backend="nccl",
    )
    direct_workspace = create_dcp_query_direct_vmm_workspace(
        max_rows=max(ROWS),
        local_heads=LOCAL_HEADS,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        group=coordinator,
    )

    try:
        assert direct_workspace.local_query.shape == (
            max(ROWS),
            LOCAL_HEADS,
            QUERY_DIM,
        )
        assert direct_workspace.local_query.is_contiguous()
        assert direct_workspace.peer_queries.shape == (
            WORLD_SIZE,
            max(ROWS),
            LOCAL_HEADS,
            QUERY_DIM,
        )

        for step, rows in enumerate(ROWS):
            inputs = _make_inputs(rows, rank, step)
            is_neox = bool(step % 2)
            expected_q, expected_k_nope, expected_k_rope = _reference(
                *inputs,
                is_neox,
                coordinator.device_group,
            )
            direct_q, direct_k_nope, direct_k_rope = direct_workspace.quantize_remote(
                *inputs, is_neox=is_neox
            )
            _assert_fp8_byte_equal(direct_q, expected_q)
            _assert_fp8_byte_equal(direct_k_nope, expected_k_nope)
            _assert_fp8_byte_equal(direct_k_rope, expected_k_rope)

        # Verify rank-major producer ownership independently of random inputs.
        rows = 8
        q_nope = torch.empty(
            rows,
            LOCAL_HEADS,
            NOPE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_rope = torch.empty(
            rows,
            LOCAL_HEADS,
            ROPE_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for local_head in range(LOCAL_HEADS):
            value = rank * LOCAL_HEADS + local_head
            q_nope[:, local_head].fill_(value)
            q_rope[:, local_head].fill_(value)
        k_nope = torch.zeros(rows, NOPE_DIM, dtype=torch.bfloat16, device="cuda")
        k_rope = torch.zeros(rows, ROPE_DIM, dtype=torch.bfloat16, device="cuda")
        positions = torch.zeros(rows, dtype=torch.int64, device="cuda")
        cos_sin_cache = torch.cat(
            (
                torch.ones(1, ROPE_DIM // 2, device="cuda"),
                torch.zeros(1, ROPE_DIM // 2, device="cuda"),
            ),
            dim=-1,
        )
        sentinel_query, _, _ = direct_workspace.quantize_remote(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            positions,
            cos_sin_cache,
            is_neox=True,
        )
        torch.cuda.synchronize()
        expected_heads = torch.arange(
            WORLD_SIZE * LOCAL_HEADS,
            dtype=torch.bfloat16,
            device="cuda",
        ).to(torch.float8_e4m3fn)
        torch.testing.assert_close(
            sentinel_query[:, :, 0].view(torch.uint8),
            expected_heads.expand(rows, -1).view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sentinel_query[:, :, NOPE_DIM].view(torch.uint8),
            expected_heads.expand(rows, -1).view(torch.uint8),
            rtol=0,
            atol=0,
        )
        # The unchanged TRTLLM-GEN sparse MLA binding must consume the
        # consumer-direct output identically to the AllGather reference.
        rows = 8
        inputs = _make_inputs(rows, rank, 77)
        dense_q, _, _ = _reference(
            *inputs,
            True,
            coordinator.device_group,
        )
        direct_q, _, _ = direct_workspace.quantize_remote(*inputs, is_neox=True)
        topk = 128
        page_size = 64
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20_000 + rank)
        kv_cache = torch.randn(
            topk // page_size,
            1,
            page_size,
            QUERY_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ).to(torch.float8_e4m3fn)
        topk_indices = (
            torch.arange(topk, dtype=torch.int32, device="cuda")
            .view(1, 1, topk)
            .expand(rows, 1, topk)
            .clone()
        )
        seq_lens = torch.full((rows,), topk, dtype=torch.int32, device="cuda")
        direct_attention_workspace = torch.zeros(
            512 * 1024 * 1024, dtype=torch.int8, device="cuda"
        )
        dense_attention_workspace = torch.zeros_like(direct_attention_workspace)
        direct_output, direct_lse = _run_sparse_mla(
            direct_q,
            kv_cache,
            topk_indices,
            seq_lens,
            direct_attention_workspace,
        )
        dense_output, dense_lse = _run_sparse_mla(
            dense_q,
            kv_cache,
            topk_indices,
            seq_lens,
            dense_attention_workspace,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(direct_output, dense_output, rtol=0, atol=0)
        torch.testing.assert_close(direct_lse, dense_lse, rtol=0, atol=0)
        del direct_attention_workspace, dense_attention_workspace

        # Capture once, then replay with changing producer-local BF16 inputs.
        rows = 32
        graph_inputs = list(_make_inputs(rows, rank, 99))
        dist.barrier(group=coordinator.cpu_group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_q, graph_k_nope, graph_k_rope = direct_workspace.quantize_remote(
                *graph_inputs, is_neox=True
            )

        for replay in range(4):
            if replay:
                graph_inputs[0].add_(0.125)
                graph_inputs[1].sub_(0.0625)
                graph_inputs[2].add_(0.25)
                graph_inputs[3].sub_(0.125)
            expected_q, expected_k_nope, expected_k_rope = _reference(
                *graph_inputs,
                True,
                coordinator.device_group,
            )
            dist.barrier(group=coordinator.cpu_group)
            graph.replay()
            torch.cuda.synchronize()
            _assert_fp8_byte_equal(graph_q, expected_q)
            _assert_fp8_byte_equal(graph_k_nope, expected_k_nope)
            _assert_fp8_byte_equal(graph_k_rope, expected_k_rope)

        pipelined_workspaces = [
            create_dcp_query_direct_vmm_workspace(
                max_rows=max(ROWS),
                local_heads=LOCAL_HEADS,
                nope_dim=NOPE_DIM,
                rope_dim=ROPE_DIM,
                group=coordinator,
            )
            for _ in range(2)
        ]
        pipelined_inputs = []
        for step in range(6):
            inputs = _make_inputs(rows, rank, 400 + step)
            pipelined_inputs.append(
                (
                    inputs,
                    _reference(
                        *inputs,
                        True,
                        coordinator.device_group,
                    ),
                )
            )
        pipelined_outputs = []
        for step, (inputs, _) in enumerate(pipelined_inputs):
            output = pipelined_workspaces[step % 2].quantize_remote(
                *inputs,
                is_neox=True,
                pipelined=True,
            )
            pipelined_outputs.append(tuple(tensor.clone() for tensor in output))
        torch.cuda.synchronize()
        for actual, (_, expected) in zip(
            pipelined_outputs, pipelined_inputs, strict=True
        ):
            for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
                _assert_fp8_byte_equal(actual_tensor, expected_tensor)
        for pipelined_workspace in pipelined_workspaces:
            pipelined_workspace.close()
    finally:
        direct_workspace.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
