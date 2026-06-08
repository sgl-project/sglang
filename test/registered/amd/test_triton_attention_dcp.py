import math
import os
import socket
import unittest

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_dcp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.utils import cp_lse_ag_out_rs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

DCP_SIZE = 2
WORLD_SIZE = 8
register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd-mi35x")


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_tp_dcp_worker(world_size: int, rank: int, port: int) -> None:
    # Match the existing distributed unit-test style: each worker binds itself
    # to its global rank so TP and DCP groups are deterministic.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    torch.cuda.set_device(rank)
    init_distributed_environment(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        decode_context_parallel_size=DCP_SIZE,
    )


def _destroy_worker() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


def _make_q(rank: int, shape: tuple[int, int, int]) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(3000 + rank)
    return torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)


def _expected_lse_merge(
    step: int,
    dcp_size: int,
    dcp_rank: int,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    batch, heads, head_dim = shape
    h = torch.arange(heads, device="cuda", dtype=torch.float32).view(1, heads)
    d = torch.arange(head_dim, device="cuda", dtype=torch.float32).view(1, 1, head_dim)
    lses = []
    outs = []
    for rank_in_group in range(dcp_size):
        lse = 0.13 * step + 0.37 * rank_in_group + 0.01 * h
        out = 0.07 * step + 0.11 * rank_in_group + 0.001 * d
        lses.append(lse.expand(batch, heads))
        outs.append(out.expand(batch, heads, head_dim))

    stacked_lse = torch.stack(lses, dim=0)
    stacked_out = torch.stack(outs, dim=0)
    global_lse = torch.logsumexp(stacked_lse, dim=0)
    merged = torch.sum(
        stacked_out * torch.exp(stacked_lse - global_lse).unsqueeze(-1), dim=0
    )

    local_heads = heads // dcp_size
    start = dcp_rank * local_heads
    end = start + local_heads
    return merged[:, start:end, :].contiguous()


def _fill_lse_merge_inputs(
    out: torch.Tensor,
    lse: torch.Tensor,
    step: int,
    dcp_rank: int,
) -> None:
    heads = out.shape[1]
    head_dim = out.shape[2]
    h = torch.arange(heads, device=out.device, dtype=out.dtype).view(1, heads)
    d = torch.arange(head_dim, device=out.device, dtype=out.dtype).view(1, 1, head_dim)
    lse.copy_(0.13 * step + 0.37 * dcp_rank + 0.01 * h)
    out.copy_(0.07 * step + 0.11 * dcp_rank + 0.001 * d)


def _dcp_attention_worker(rank: int, world_size: int, port: int) -> None:
    _init_tp_dcp_worker(world_size, rank, port)
    try:
        group = get_dcp_group()
        assert group.world_size == DCP_SIZE
        assert group.rank_in_group == rank % DCP_SIZE

        batch = 3
        local_heads = 2
        head_dim = 16
        seq_len = 9
        group_base_rank = rank - group.rank_in_group

        q_local = _make_q(rank, (batch, local_heads, head_dim))
        q_all = group.all_gather(q_local, dim=1).contiguous()

        expected_q_chunks = [
            _make_q(r, (batch, local_heads, head_dim))
            for r in range(group_base_rank, group_base_rank + DCP_SIZE)
        ]
        torch.testing.assert_close(q_all, torch.cat(expected_q_chunks, dim=1))

        gen = torch.Generator(device="cuda")
        gen.manual_seed(1000 + group_base_rank)
        k_full = torch.randn(
            (batch, seq_len, head_dim),
            generator=gen,
            device="cuda",
            dtype=torch.float32,
        )
        gen.manual_seed(2000 + group_base_rank)
        v_full = torch.randn(
            (batch, seq_len, head_dim),
            generator=gen,
            device="cuda",
            dtype=torch.float32,
        )

        owner_mask = (
            torch.arange(seq_len, device="cuda") % DCP_SIZE == group.rank_in_group
        )
        k_local = k_full[:, owner_mask, :].contiguous()
        v_local = v_full[:, owner_mask, :].contiguous()

        scale = 1.0 / math.sqrt(head_dim)
        local_logits = torch.einsum("bhd,bsd->bhs", q_all, k_local) * scale
        local_lse = torch.logsumexp(local_logits, dim=-1)
        local_probs = torch.exp(local_logits - local_lse.unsqueeze(-1))
        local_out = torch.einsum("bhs,bsd->bhd", local_probs, v_local).contiguous()

        out = cp_lse_ag_out_rs(local_out, local_lse, group)

        ref_logits = torch.einsum("bhd,bsd->bhs", q_all, k_full) * scale
        ref_probs = torch.softmax(ref_logits, dim=-1)
        ref_out = torch.einsum("bhs,bsd->bhd", ref_probs, v_full)
        head_start = group.rank_in_group * local_heads
        head_end = head_start + local_heads
        torch.testing.assert_close(
            out,
            ref_out[:, head_start:head_end, :].contiguous(),
            rtol=2e-4,
            atol=2e-4,
        )

        # CUDA graph replay should consume the updated LSE/output buffers each time.
        total_heads = local_heads * DCP_SIZE
        cp_out = torch.empty(
            (batch, total_heads, head_dim), device="cuda", dtype=torch.float32
        )
        cp_lse = torch.empty((batch, total_heads), device="cuda", dtype=torch.float32)

        _fill_lse_merge_inputs(cp_out, cp_lse, step=0, dcp_rank=group.rank_in_group)
        eager = cp_lse_ag_out_rs(cp_out, cp_lse, group)
        torch.testing.assert_close(
            eager,
            _expected_lse_merge(0, DCP_SIZE, group.rank_in_group, cp_out.shape),
            rtol=1e-5,
            atol=1e-5,
        )

        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        set_graph_pool_id(pool)
        _fill_lse_merge_inputs(cp_out, cp_lse, step=10, dcp_rank=group.rank_in_group)
        with graph_capture() as graph_capture_context:
            with torch.cuda.graph(
                graph, pool=pool, stream=graph_capture_context.stream
            ):
                graph_out = cp_lse_ag_out_rs(cp_out, cp_lse, group)

        for step in range(11, 14):
            _fill_lse_merge_inputs(
                cp_out, cp_lse, step=step, dcp_rank=group.rank_in_group
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                graph_out,
                _expected_lse_merge(step, DCP_SIZE, group.rank_in_group, cp_out.shape),
                rtol=1e-5,
                atol=1e-5,
            )
    finally:
        _destroy_worker()


class TestDCPAttention(CustomTestCase):
    def test_tp8_dcp2_attention_merge(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
            self.skipTest("TP+DCP attention test requires 8 CUDA devices")

        port = _get_open_port()
        mp.spawn(
            _dcp_attention_worker,
            args=(WORLD_SIZE, port),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
