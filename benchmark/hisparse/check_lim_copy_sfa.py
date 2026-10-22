"""Validate the real LIM adapter, registered host allocator and Copy-SFA dispatch."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_lim_framework import FrameworkCase, inputs
from sglang.srt.environ import envs


def verify(case, output, *, resident=True):
    plan = case.backend.forward_metadata.sparse_kv_lim_plan
    ids = plan.source_ids.cpu().long()
    slots = plan.device_slots.cpu().long()
    actual = output.cpu().float().view(case.args.batch, case.args.query_heads, 512)
    for row in range(case.args.batch):
        if not plan.valid_rows[row].item():
            torch.testing.assert_close(
                actual[row], torch.zeros_like(actual[row]), rtol=0, atol=0
            )
            continue
        valid = (ids[row] >= 0) & (ids[row] < int(case.batch.seq_lens[row]))
        selected_ids = ids[row, valid]
        selected = (
            case.fixture.host[row + 1, selected_ids, 0] + case.args.layers - 1
        ).float()
        q = case.fixture.q[row].cpu().float()
        rope = case.fixture.rope[row].cpu().float()
        scores = (
            q @ selected[:, :512].T + rope @ selected[:, 512:].T
        ) * case.layer.scaling
        expected = torch.softmax(scores, dim=-1) @ selected[:, :512]
        torch.testing.assert_close(actual[row], expected, rtol=0.02, atol=0.002)
        if resident and plan.sparse_rows[row].item():
            for layer in range(case.args.layers):
                cached = case.manager.device_kv_buffer[layer][
                    row + 1, slots[row, valid], 0
                ].cpu()
                expected_kv = case.fixture.host[row + 1, selected_ids, 0] + layer
                torch.testing.assert_close(cached, expected_kv, rtol=0, atol=0)


def run(output_path):
    torch.npu.set_device(0)
    args = SimpleNamespace(
        batch=2,
        context=8192,
        tail=1,
        capacity=2048,
        index_heads=32,
        query_heads=8,
        layers=2,
    )
    fixture = inputs(2, args.context, 1024, 32, 8, 20261022)
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_COPY_SFA.override(True),
    ):
        case = FrameworkCase("lim", fixture, args)
        for route in (0, 1, 0):
            result = case.step(route)
            torch.npu.synchronize()
            verify(case, result)
            torch.testing.assert_close(
                case.latest_ids.sort(dim=1).values,
                fixture.expected[route],
                rtol=0,
                atol=0,
            )
        graph = torch.npu.NPUGraph()
        case.batch.sparse_kv_copy_sfa_capture = True
        with torch.npu.graph(graph):
            result = case.step(1)
        case.reset()
        graph.replay()
        torch.npu.synchronize()
        verify(case, result)
        case.batch.req_pool_indices[0] = 0
        case.batch.seq_lens[0] = 0
        graph.replay()
        torch.npu.synchronize()
        verify(case, result)
        case.batch.req_pool_indices[0] = 1
        case.batch.seq_lens[0] = args.context + args.tail
        case.reset()
        graph.replay()
        torch.npu.synchronize()
        verify(case, result)
        # A mixed short/long eager batch must preserve the fused resident data
        # and force a cold fill before graph replay is eligible again.
        case.batch.sparse_kv_copy_sfa_capture = False
        case.batch.seq_lens[0] = 128
        case.batch.seq_lens_cpu[0] = 128
        case.backend.forward_metadata.seq_lens_cpu_int[0] = 128
        eager_result = case.step(0)
        torch.npu.synchronize()
        verify(case, eager_result, resident=False)
        # The fallback consumes host KV directly, without publishing residency.
        plan = case.backend.forward_metadata.sparse_kv_lim_plan
        assert torch.all(
            case.manager.lim_adapter.last_prefix[0, plan.request_slots.long()] == -1
        )
        case.batch.seq_lens[0] = args.context + args.tail
        case.batch.seq_lens_cpu[0] = args.context + args.tail
        case.backend.forward_metadata.seq_lens_cpu_int[0] = args.context + args.tail
        case.batch.sparse_kv_copy_sfa_capture = True
        graph.replay()
        torch.npu.synchronize()
        verify(case, result)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            case.step(0)
            torch.npu.synchronize()
        counts = {
            event.key: event.count
            for event in prof.key_averages()
            if "fused_lightning_indexer_manage" in event.key
            or "fused_scatter_copy_sparse_flash_attention" in event.key
        }
        assert (
            sum(v for k, v in counts.items() if "fused_lightning_indexer_manage" in k)
            > 0
        ), counts
        assert (
            sum(
                v
                for k, v in counts.items()
                if "fused_scatter_copy_sparse_flash_attention" in k
            )
            > 0
        ), counts
        record = {
            "scope": "fixed projections, real manager/allocator/attention; no model score or timing claim",
            "eager_topk": "pass",
            "resident_payload_bitwise": "pass",
            "attention_cpu_oracle": "pass",
            "graph_reset_padding_and_reuse": "pass",
            "mixed_fallback_to_graph": "pass",
            "operator_calls": counts,
        }
        output_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
