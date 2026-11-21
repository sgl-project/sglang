import unittest

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase, run_distributed_test

# backend-specific: a gfx950 HIP kernel over AITER's IPC buffers, DCP8 only
register_amd_ci(est_time=45, suite="base-c-test-large-8-gpu-amd-mi35x")

WORLD = 8
HEADS = 12 * WORLD  # Kimi-K3.1
HEAD_DIM = 128


def _inputs(rows, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.randn(rows, HEADS, HEAD_DIM, generator=g).to(torch.bfloat16).cuda()
    lse = (torch.randn(rows, HEADS, generator=g) * 4).cuda()
    lse[::7, 3] = float("-inf")  # empty partials
    return out, lse


def _reference(out, lse):
    from sglang.kernels.ops.attention.dcp_kernels import (
        dcp_lse_combine_triton,
        dcp_pack_a2a_send,
    )

    rows, heads, dim = out.shape
    send = torch.empty(
        WORLD, rows, heads // WORLD, dim + 2, dtype=out.dtype, device=out.device
    )
    recv = torch.empty_like(send)
    dcp_pack_a2a_send(
        out, lse, send[:, :, :, :dim], send.view(torch.float32)[:, :, :, dim // 2]
    )
    dist.all_to_all_single(
        recv.reshape(-1).view(torch.uint8), send.reshape(-1).view(torch.uint8)
    )
    combined, _ = dcp_lse_combine_triton(
        recv[:, :, :, :dim], recv.view(torch.float32)[:, :, :, dim // 2]
    )
    return combined


def _exchanged(comm, out, lse):
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    from sglang.kernels.ops.communication import dcp_ipc_a2a_hip as ipc

    recv_out, recv_lse = ipc.ipc_exchange(comm, out, lse)
    combined, _ = dcp_lse_combine_triton(recv_out, recv_lse)
    return combined


def _same(got, want):
    return torch.equal(got.view(torch.int16), want.view(torch.int16))


def _check_rank(rank):
    from sglang.kernels.ops.communication import dcp_ipc_a2a_hip as ipc

    comm = ipc.create_ipc_comm(
        dist.new_group(backend="gloo"), torch.device(f"cuda:{rank}")
    )
    assert comm is not None, "create_ipc_comm returned None"
    failures = []
    # blocks without rows, and the end of the scratch
    for rows in (1, ipc.MAX_ROWS):
        out, lse = _inputs(rows, rows + rank)
        assert ipc.ipc_exchange_covers(out, lse)
        if not _same(_exchanged(comm, out, lse), _reference(out, lse)):
            failures.append(f"eager: {rows} rows")

    # a ragged second pass over the blocks, replayed on new data
    rows = 435
    static_out, static_lse = _inputs(rows, 7 + rank)
    result = {}

    def step():
        result["out"] = _exchanged(comm, static_out, static_lse)

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for replay in range(2):
        out, lse = _inputs(rows, 50 + 100 * replay + rank)
        static_out.copy_(out)
        static_lse.copy_(lse)
        graph.replay()
        if not _same(result["out"], _reference(out, lse)):
            failures.append(f"graph replay {replay}")
    assert not failures, failures


@unittest.skipUnless(
    is_gfx95_supported() and torch.cuda.device_count() >= WORLD, "8 gfx950 GPUs"
)
class TestDcpIpcAllToAll(CustomTestCase):
    def test_matches_the_rccl_exchange_bitwise(self):
        run_distributed_test(_check_rank, world_size=WORLD)


if __name__ == "__main__":
    unittest.main()
