"""Measure full-visible Q1 LIM + Copy-SFA with fixed inputs and real host KV.

Run the same script against each source/wheel pair for an A/B comparison.
Timings cover the indexer/manager/attention subpath, not a checkpoint server.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_lim_framework import FrameworkCase, inputs

from sglang.srt.environ import envs


def run(args):
    torch.npu.set_device(0)
    options = SimpleNamespace(
        batch=args.batch,
        context=args.context,
        tail=1,
        capacity=2048,
        index_heads=32,
        query_heads=8,
        layers=2,
    )
    fixture = inputs(args.batch, args.context, 1024, 32, 8, 20261023)
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_COPY_SFA.override(True),
    ):
        before = torch.npu.memory_allocated()
        case = FrameworkCase("lim", fixture, options)
        torch.npu.synchronize()
        manager_allocated = torch.npu.memory_allocated() - before
        for route in (0, 1, 0, 1):
            output = case.step(route)
            torch.npu.synchronize()
            torch.testing.assert_close(
                case.latest_ids.sort(dim=1).values,
                fixture.expected[route],
                rtol=0,
                atol=0,
            )
            plan = case.backend.forward_metadata.sparse_kv_lim_plan
            for row in range(args.batch):
                ids = plan.source_ids[row].cpu().long()
                slots = plan.device_slots[row].cpu().long()
                selected = (
                    fixture.host[row + 1, ids, 0] + (options.layers - 1)
                ).float()
                actual_kv = case.manager.device_kv_buffer[-1][row + 1, :, 0].cpu()[
                    slots
                ]
                torch.testing.assert_close(
                    actual_kv, selected.to(torch.bfloat16), rtol=0, atol=0
                )
                scores = (
                    fixture.q[row].cpu().float() @ selected[:, :512].T
                    + fixture.rope[row].cpu().float() @ selected[:, 512:].T
                ) * case.layer.scaling
                expected = torch.softmax(scores, dim=-1) @ selected[:, :512]
                torch.testing.assert_close(
                    output[row].cpu().float().view(8, 512),
                    expected,
                    rtol=0.02,
                    atol=0.002,
                )
        case.batch.sparse_kv_copy_sfa_capture = True
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(args.pairs):
                case.step(0)
                case.step(1)
        for _ in range(5):
            graph.replay()
        torch.npu.synchronize()
        timings = []
        for _ in range(5):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            timings.append(
                start.elapsed_time(end) * 1000 / (args.iterations * args.pairs * 2)
            )
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            case.step(0)
            torch.npu.synchronize()
        keys = {"aten::copy_", "aten::where", "aten::masked_fill_"}
        counts = {
            event.key: event.count
            for event in prof.key_averages()
            if event.key in keys or event.key.startswith("npu::fused_")
        }
        record = {
            "scope": "fixed-input full-visible Q1 indexer/manager/attention; no model throughput claim",
            "batch": args.batch,
            "context": args.context,
            "layers": options.layers,
            "graph_us_per_step_samples": timings,
            "graph_us_per_step_median": statistics.median(timings),
            "manager_allocated_bytes": manager_allocated,
            "manager_budget_bytes": case.manager.budget.managed_hbm_bytes,
            "topk_and_resident_payload": "pass",
            "attention_cpu_oracle": "pass",
            "selected_ids_sha256": hashlib.sha256(
                case.latest_ids.cpu().numpy().tobytes()
            ).hexdigest(),
            "dispatcher_counts_per_step": counts,
        }
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
