"""Two-NPU regression for asynchronous EPLB on CANN with HIXL.

Run with two free NPUs and the repository on PYTHONPATH:
    torchrun --standalone --nproc-per-node=2 test/manual/test_npu_async_eplb.py

Checks physical NZ padding, scales, shared-expert suffixes, captured graph
addresses over three layout epochs, and a collective snapshot-error abort.
"""

import json
import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu

from sglang.srt.eplb.async_rebalance import AsyncEplbCoordinator, RebalanceAborted
from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.hardware_backend.npu.eplb import _ExpertWeightAdapter
from sglang.srt.hardware_backend.npu.eplb_async import (
    EplbControl,
    NpuEplbRuntime,
    PlanningSpec,
)
from sglang.srt.hardware_backend.npu.hixl_transport import HixlExpertTransport
from sglang.srt.runtime_context import get_context


def main():
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.config.allow_internal_format = True
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    control_group, plan_group, transport_group = [
        dist.new_group(backend="gloo", timeout=timedelta(seconds=90)) for _ in range(3)
    ]

    def inverse(physical):
        result = torch.full((2, 6, 8), -1, dtype=torch.long)
        for layer in range(2):
            for expert in range(6):
                slots = (physical[layer] == expert).nonzero().flatten()
                result[layer, expert, : len(slots)] = slots
        return result

    old = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]] * 2)
    with get_context().override_server_args(
        ep_dispatch_algorithm="static", ep_size=2, nnodes=1
    ):
        metadata = ExpertLocationMetadata._init_raw(2, old.npu(), inverse(old).npu())
        weights = {}
        full = []
        for layer in range(2):
            matrix = torch_npu.npu_format_cast(
                torch.zeros((5, 17, 33), dtype=torch.int8, device="npu"), 29
            )
            assert torch_npu.get_npu_format(matrix) in (29, 50, 51), (
                "NZ format was silently disabled"
            )
            scale = torch.zeros((5, 17), dtype=torch.float32, device="npu")
            full.append((matrix, scale))
            weights[layer] = [matrix[:4], scale[:4]]
        adapter = _ExpertWeightAdapter(weights, [0, 1], rank)
        for layer in range(2):
            views = adapter.make_weight_views(weights[layer])
            for slot, expert in enumerate(
                old[layer, rank * 4 : (rank + 1) * 4].tolist()
            ):
                views[0][slot].fill_(expert + 1 + layer * 10)
                weights[layer][1][slot].fill_(expert + 1 + layer * 10)
        shared = [[t[4].cpu().clone() for t in tensors] for tensors in full]
        pointers = [[t.data_ptr() for t in tensors] for tensors in weights.values()]
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        endpoint = f"{socket.gethostbyname(socket.gethostname())}:{port}"
        runtime = NpuEplbRuntime(
            weights,
            metadata,
            PlanningSpec(8, 4, None, 1, EplbAlgorithm.deepseek),
            rank,
            2,
            2,
            plan_group,
            lambda w, s: HixlExpertTransport(w, s, transport_group, endpoint),
        )
        coordinator = AsyncEplbCoordinator(EplbControl(control_group), runtime)
        # The graph holds live weight pointers across all in-place installations.
        for _ in range(3):
            graph_output = runtime.weights[0][0].sum(dim=1)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output = runtime.weights[0][0].sum(dim=1)
        steps, elapsed, migrated = 0, [], []
        for epoch in range(3):
            if epoch == 0:
                counts = torch.tensor(
                    [[[50, 30, 20, 10, 5, 1]] * 2], dtype=torch.int64, device="npu"
                )
                coordinator.start(runtime.planner(counts))
            else:
                physical = metadata.physical_to_logical_map_cpu.roll(3, dims=1).clone()
                placement = (physical, inverse(physical))
                coordinator.start(lambda: placement)
            begin = time.perf_counter()
            while coordinator.active:
                expected = metadata.physical_to_logical_map_cpu[
                    0, rank * 4 : (rank + 1) * 4
                ].clone()
                graph.replay()
                coordinator.poll()
                torch.npu.synchronize()
                assert torch.equal(
                    graph_output.cpu(), (expected + 1) * runtime.weights[0][0].shape[1]
                ), (epoch, steps, graph_output.cpu(), expected)
                steps += 1
                if time.perf_counter() - begin > 30:
                    raise TimeoutError("runtime protocol did not finish")
            runtime.synchronize()
            elapsed.append(time.perf_counter() - begin)
            migrated.append(runtime.bytes_read)
            for layer in range(2):
                for slot, expert in enumerate(
                    metadata.physical_to_logical_map_cpu[
                        layer, rank * 4 : (rank + 1) * 4
                    ].tolist()
                ):
                    code = expert + 1 + layer * 10
                    assert torch.all(runtime.weights[layer][0][slot].cpu() == code)
                    assert torch.all(weights[layer][0].cpu()[slot] == code)
                    assert torch.all(weights[layer][1][slot].cpu() == code)
                for tensor, before in zip(full[layer], shared[layer]):
                    assert torch.equal(tensor[4].cpu(), before), (
                        "shared-expert suffix overwritten"
                    )
            assert pointers == [
                [t.data_ptr() for t in tensors] for tensors in weights.values()
            ]
        # A snapshot failure on just one rank must become a drained abort everywhere.
        coordinator.start(runtime.planner(None if rank == 1 else counts))
        aborted = False
        deadline = time.monotonic() + 10
        while coordinator.active and time.monotonic() < deadline:
            try:
                coordinator.poll()
            except RebalanceAborted:
                aborted = True
        assert aborted
        coordinator.reset()
        coordinator.close()
        print(
            json.dumps(
                {
                    "status": "NPU_RUNTIME_STATIC_GRAPH_NZ_PASS",
                    "rank": rank,
                    "epochs": 3,
                    "snapshot_failure_drained": aborted,
                    "forward_steps": steps,
                    "epoch_seconds": elapsed,
                    "cumulative_read_bytes": migrated,
                    "staging_bytes": sum(t.numel() for t in runtime.staging),
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
