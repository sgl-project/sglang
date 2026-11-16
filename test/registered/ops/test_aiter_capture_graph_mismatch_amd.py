import multiprocessing
import socket
import time
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-c-test-4-gpu-amd")

_WORLD_SIZE = 4
# bf16 elements per rank; all-reduce only reads peer inputs above ~128 KiB.
_NUMEL = 1 << 19
_HIDDEN = 1024


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _make_buffers(device: torch.device) -> dict:
    rows = _NUMEL // _HIDDEN
    return dict(
        ar_in=torch.zeros(_NUMEL, dtype=torch.bfloat16, device=device),
        ag_in=torch.zeros(_NUMEL, dtype=torch.bfloat16, device=device),
        ag_out=torch.zeros(_NUMEL * _WORLD_SIZE, dtype=torch.bfloat16, device=device),
        rs_in=torch.zeros(
            _WORLD_SIZE * rows, _HIDDEN, dtype=torch.bfloat16, device=device
        ),
        rs_out=torch.zeros(rows, _HIDDEN, dtype=torch.bfloat16, device=device),
    )


def _run_graph_mismatch(rank: int, world_size: int, port: int) -> None:
    from sglang.srt.distributed.parallel_state import (
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.test.test_utils import publish_build_topology

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    publish_build_topology(
        tp_size=world_size,
        dp_size=2,
        enable_dp_attention=True,
        attention_backend="dsa",
        world_rank=rank,
    )
    initialize_model_parallel()
    tp_group = get_tp_group()
    assert tp_group.ca_comm is not None and not tp_group.ca_comm.disabled

    buffers = {label: _make_buffers(device) for label in ("A", "B")}
    graphs = {}
    with graph_capture() as capture_context:
        for label, buf in buffers.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_context.stream):
                buf["ar_out"] = tp_group.all_reduce(buf["ar_in"])
                tp_group.all_gather_into_tensor(buf["ag_out"], buf["ag_in"])
                tp_group.reduce_scatter_tensor(buf["rs_out"], buf["rs_in"])
            graphs[label] = graph

    # The two DP lanes replay different graphs, as per-lane DSA variants do.
    mine, other = ("A", "B") if rank < world_size // 2 else ("B", "A")
    for name in ("ar_in", "ag_in", "rs_in"):
        buffers[mine][name].fill_(rank + 1)
        buffers[other][name].zero_()
    torch.cuda.synchronize()
    tp_group.barrier()
    graphs[mine].replay()
    torch.cuda.synchronize()
    tp_group.barrier()

    buf = buffers[mine]
    rank_sum = world_size * (world_size + 1) // 2
    torch.testing.assert_close(
        buf["ar_out"], torch.full_like(buf["ar_out"], rank_sum), rtol=0, atol=0
    )
    expected_gather = torch.arange(
        1, world_size + 1, dtype=torch.bfloat16, device=device
    ).repeat_interleave(_NUMEL)
    torch.testing.assert_close(buf["ag_out"], expected_gather, rtol=0, atol=0)
    torch.testing.assert_close(
        buf["rs_out"], torch.full_like(buf["rs_out"], rank_sum), rtol=0, atol=0
    )
    tp_group.barrier()


def _spawn(test_case, target, world_size, name):
    port = _get_open_port()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=target, args=(rank, world_size, port))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 300
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    if any(process.is_alive() for process in processes):
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        test_case.fail(f"{name} timed out")
    for rank, process in enumerate(processes):
        test_case.assertEqual(process.exitcode, 0, f"{name} failed on rank {rank}")


class TestAiterCaptureGraphMismatch(CustomTestCase):
    @unittest.skipUnless(
        torch.version.hip is not None and torch.cuda.device_count() >= _WORLD_SIZE,
        "requires four ROCm GPUs",
    )
    def test_dp_lanes_replaying_different_graphs(self):
        """With DSA and DP attention, DP lanes that replay different captured graphs
        must still get full all-reduce, all-gather and reduce-scatter results."""
        _spawn(self, _run_graph_mismatch, _WORLD_SIZE, "DSA DP graph mismatch test")


if __name__ == "__main__":
    unittest.main()
