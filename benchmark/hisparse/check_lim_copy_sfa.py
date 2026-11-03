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
            case.fixture.host[row + 1, selected_ids, 0] + (case.args.layers - 1)
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


def run(output_path, padded_batch=False, capacity=2048):
    torch.npu.set_device(0)
    args = SimpleNamespace(
        batch=16 if padded_batch else 2,
        context=max(8192, capacity + 1024),
        tail=1,
        capacity=capacity,
        index_heads=32,
        query_heads=8,
        layers=2,
    )
    fixture = inputs(args.batch, args.context, 1024, 32, 8, 20261022)
    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_COPY_SFA.override(True),
    ):
        case = FrameworkCase(
            "lim",
            fixture,
            args,
            request_capacity=2,
            graph_batch_capacity=args.batch if padded_batch else 0,
        )
        if padded_batch:
            case.batch.req_pool_indices[2:].zero_()
            case.batch.seq_lens[2:].zero_()
            case.batch.seq_lens_cpu[2:].zero_()
            case.backend.forward_metadata.seq_lens_cpu_int[2:] = [0] * (args.batch - 2)
            assert case.manager.batch_capacity == 16 and case.manager.size == 3
        prefill = SimpleNamespace(
            req_pool_indices=torch.tensor([2, 0, 1], dtype=torch.int64, device="npu"),
            seq_lens=torch.tensor([3, 0, 5], dtype=torch.int64, device="npu"),
            req_pool_indices_cpu=torch.tensor([2, 0, 1], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3, 0, 5], dtype=torch.int64),
        )
        for layer_id in range(args.layers):
            nope, rope = case.manager.get_forward_kv(layer_id, prefill)
            actual = torch.cat((nope, rope), dim=-1).cpu()
            expected = torch.cat((fixture.host[2, :3], fixture.host[1, :5])) + layer_id
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for route in (0, 1, 0):
            result = case.step(route)
            torch.npu.synchronize()
            verify(case, result)
            torch.testing.assert_close(
                case.latest_ids[:2].sort(dim=1).values,
                fixture.expected[route][:2],
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
        plan = case.backend.forward_metadata.sparse_kv_lim_plan
        metadata = case.manager.lim_adapter.copy_sfa_metadata(case.manager, plan)
        for actual, native in zip(
            metadata[:6],
            (
                plan.source_ids,
                plan.device_slots,
                plan.topk_misses,
                plan.miss_sources,
                plan.miss_slots,
                plan.miss_counts,
            ),
        ):
            assert actual.is_contiguous() and actual.data_ptr() == native.data_ptr()
        record = {
            "scope": "fixed projections, real manager/allocator/attention; no model score or timing claim",
            "eager_topk": "pass",
            "resident_payload_bitwise": "pass",
            "attention_cpu_oracle": "pass",
            "graph_reset_padding_and_reuse": "pass",
            "mixed_fallback_to_graph": "pass",
            "prefill_ragged_gather": "pass",
            "operator_calls": counts,
            "request_rows": case.manager.size,
            "batch_capacity": case.manager.batch_capacity,
            "resident_capacity": case.manager.device_cache_size,
            "decode_path": case.manager.last_decode_path,
            "native_plan_shared": True,
        }
        output_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padded-batch", action="store_true")
    parser.add_argument(
        "--capacity", type=int, choices=(2048, 4096, 8192), default=2048
    )
    args = parser.parse_args()
    run(args.output, args.padded_batch, args.capacity)
