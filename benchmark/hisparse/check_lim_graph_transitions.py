"""NPU correctness smoke for dynamic length/padding/reset on a captured graph.

Uses fixed projections and a registered-host boundary, without timing claims.
The newly visible page has high index scores, forcing real new-prefix misses.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from bench_lim_framework import FrameworkCase, inputs


def run(args):
    torch.npu.set_device(0)
    sys.path.insert(0, str(args.registered_host_helper))
    from hisparse_registered_host_view import allocate_registered_host, make_npu_view

    owners = []

    def allocate_host(*, shape, dtype, **kwargs):
        owner = allocate_registered_host(torch.zeros(shape, dtype=dtype))
        view, host_ptr, _, mapped_ptr = make_npu_view(owner)
        owners.append((owner, view))
        return owner, int(host_ptr), int(mapped_ptr)

    config = SimpleNamespace(
        batch=2,
        context=8192,
        tail=1,
        capacity=4096,
        index_heads=32,
        query_heads=8,
        layers=1,
    )
    f = inputs(2, config.context, 1024, 32, 8, 20261020)
    keys = f.key.view(2, f.context, 1, 128)
    grown = []
    expected_cpu = f.expected.cpu().tolist()
    for row in range(2):
        a, b = set(expected_cpu[0][row]), set(expected_cpu[1][row])
        losers_a, losers_b = sorted(a - b)[:128], sorted(b - a)[:128]
        keys[row, losers_a, 0, 0] = 2
        keys[row, losers_b, 0, 1] = 2
        new = set(range(config.context, config.context + 128))
        grown.append((sorted(a - set(losers_a) | new), sorted(b - set(losers_b) | new)))
    keys[:, config.context :, 0, :2] = 20
    keys = torch.cat(
        (keys, torch.zeros(2, 128, 1, 128, device="npu", dtype=torch.bfloat16)), dim=1
    )
    f.context += 128
    f.key = keys.reshape(-1, 128, 1, 128)
    f.table = torch.arange(2 * f.context // 128, device="npu", dtype=torch.int32).view(
        2, -1
    )
    f.host = torch.cat(
        (f.host, torch.randn(3, 128, 1, 576, dtype=torch.bfloat16)), dim=1
    )
    outcomes = {}
    with patch(
        "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager.create_shm_tensor",
        allocate_host,
    ):
        for mode in ("li_reference", "lim"):
            case = FrameworkCase(mode, f, config)
            case.step(0)
            case.step(1)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                case.step(1)
                case.step(0)
            graph.replay()
            torch.npu.synchronize()
            case.verify_payload()
            case.batch.seq_lens.fill_(config.context + 129)
            graph.replay()
            torch.npu.synchronize()
            case.verify_payload()
            expected = torch.tensor(
                [row[0] for row in grown], dtype=torch.int32, device="npu"
            )
            torch.testing.assert_close(
                case.manager.lim_adapter.source_ids[:2, 0].sort(dim=1).values,
                expected,
                rtol=0,
                atol=0,
            )
            case.batch.req_pool_indices[0] = 0
            case.batch.seq_lens[0] = 0
            graph.replay()
            torch.npu.synchronize()
            case.verify_payload()
            torch.testing.assert_close(
                case.manager.lim_adapter.source_ids[0],
                torch.full_like(case.manager.lim_adapter.source_ids[0], -1),
            )
            case.batch.req_pool_indices.copy_(
                torch.tensor([1, 2], dtype=torch.int32, device="npu")
            )
            case.batch.seq_lens.fill_(config.context + 129)
            case.reset()
            graph.replay()
            torch.npu.synchronize()
            case.verify_payload()
            torch.testing.assert_close(
                case.manager.lim_adapter.source_ids[:2, 0].sort(dim=1).values,
                expected,
                rtol=0,
                atol=0,
            )
            outcomes[mode] = {
                "new_visible_page_topk": "pass",
                "kv_payload_bitwise": "pass",
                "mixed_padding": "pass",
                "request_reset_in_graph": "pass",
            }
    args.output.write_text(json.dumps(outcomes, indent=2) + "\n")
    print(json.dumps(outcomes, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-host-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
