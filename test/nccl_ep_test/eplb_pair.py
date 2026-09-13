"""Native two-rank EPLB + LL Graph gate, with real Triton expert compute.

Run with torchrun --nproc-per-node=2 --module nccl_ep_test.eplb_pair.
This requires the pinned SM90+ pair environment used by the other EP gates.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from .dispatcher import initialize
from .environment import Unavailable
from .ep_audit import EpAudit
from .eplb import metadata, received_counts, tensor_addresses
from .moe_model import make_moe
from .oracle import RoutingBatch
from .pair_followups import expected_output
from .sglang_graph import backend_for, close_runtime
from .shared_compute import make_shared_mlp
from .triton_compute import configure_compute, make_compute_fixture


def exercise(*, replays=100, generations=2, sbo=False):
    from sglang.srt.eplb.expert_distribution import ExpertDistributionRecorder
    from sglang.srt.eplb.expert_location_dispatch import (
        ExpertLocationDispatchInfo,
        topk_ids_logical_to_physical,
    )
    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
    from sglang.srt.layers.moe.moe_runner.nccl_ep_triton import run_nccl_ep_triton
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpDispatcher
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        nccl_ep_eager_session,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.model_executor.runner.shape_key import ShapeKey
    from sglang.srt.runtime_context import get_flags, get_resources

    capacity = 8
    rank, coordinator, bindings = initialize(
        capacity, graph_enabled=True, weight_transfer=True
    )
    configure_compute(graph_enabled=True, dispatch_algorithm="static")
    get_flags().moe.sbo_enabled = sbo
    layouts = [[0, 1, 2, 3, 0, 1], [3, 0, 0, 1, 2, 3], [1, 2, 3, 0, 1, 2]]
    old = metadata([layouts[0]] * 2, ep_size=2, rank=rank)
    get_resources().expert_location_metadata = old
    recorder = ExpertDistributionRecorder.init_new(
        SimpleNamespace(
            moe_a2a_backend="nccl_ep",
            expert_distribution_recorder_mode="stat",
            expert_distribution_recorder_buffer_size=8,
            enable_expert_distribution_metrics=False,
            device="cuda",
        ),
        old,
        rank,
    )
    recorder.start_record()
    get_resources().expert_distribution_recorder = recorder
    _, global_quant, config = make_compute_fixture(hidden=2048, experts=4)
    config = replace(config, num_experts=6, num_local_experts=3)
    quant_fields = ("w13_weight", "w2_weight", "w13_scale", "w2_scale")
    local_ids = old.physical_to_logical_map[0, rank * 3 : (rank + 1) * 3]
    quants = [
        replace(
            global_quant,
            **{
                name: getattr(global_quant, name).index_select(0, local_ids).clone()
                for name in quant_fields
            },
        )
        for _ in range(2)
    ]
    weights = {
        layer: [getattr(quant, name) for name in quant_fields]
        for layer, quant in enumerate(quants)
    }
    addresses = tensor_addresses(old, weights)
    infos = [
        ExpertLocationDispatchInfo(
            "static",
            old.logical_to_rank_dispatch_physical_map[layer],
            old.logical_to_all_physical_map[layer],
            old.logical_to_all_physical_map_num_valid[layer],
            6,
        )
        for layer in range(2)
    ]
    x = torch.ones(capacity, 2048, dtype=torch.bfloat16, device="cuda") * 0.125
    logical_ids = torch.tensor([[0, 3]] * capacity, device="cuda")
    factors = torch.tensor([[0.25, 0.75]] * capacity, device="cuda")
    active = torch.full((), capacity, device="cuda", dtype=torch.int64)
    updater = ExpertLocationUpdater()
    checked = 0
    cases = ((8, 8), (0, 4), (8, 0), (1, 7))

    with patch(
        "sglang.srt.layers.linear.get_tp_group", return_value=coordinator
    ), EpAudit() as audit:
        dispatchers = [
            NcclEpDispatcher(replace(config, layer_id=layer), coordinator)
            for layer in range(2)
        ]
        shared = make_shared_mlp(hidden=2048) if sbo else None

        class Router:
            def __call__(
                self,
                tokens,
                logits,
                *,
                expert_location_dispatch_info,
                num_token_non_padded,
            ):
                ids = topk_ids_logical_to_physical(
                    logical_ids[: len(tokens)], expert_location_dispatch_info
                )
                ids = torch.where(
                    torch.arange(len(tokens), device="cuda")[:, None]
                    < num_token_non_padded,
                    ids,
                    -1,
                )
                return StandardTopKOutput(factors[: len(tokens)], ids, None)

            def empty_topk_output(self, device, **kwargs):
                return StandardTopKOutput(factors[:0], logical_ids[:0], None)

        models = (
            [
                make_moe(
                    dispatcher,
                    quants[layer],
                    replace(config, layer_id=layer),
                    shared,
                    Router(),
                    scale=1.0,
                    sbo=True,
                )
                for layer, dispatcher in enumerate(dispatchers)
            ]
            if sbo
            else []
        )

        def forward(bucket):
            outputs = []
            for layer, dispatcher in enumerate(dispatchers):
                if sbo:
                    # Native harness compares complete padded buckets. Zero the
                    # shared-MLP input on inactive rows, whose results serving
                    # would normally slice away after replay.
                    tokens = torch.where(
                        torch.arange(bucket, device="cuda")[:, None] < active,
                        x[:bucket],
                        0,
                    )
                    with recorder.with_current_layer(layer):
                        outputs.append(
                            models[layer].forward_deepep(
                                tokens, SimpleNamespace(num_token_non_padded=active)
                            )
                        )
                    continue
                ids = topk_ids_logical_to_physical(logical_ids[:bucket], infos[layer])
                ids = torch.where(
                    torch.arange(bucket, device="cuda")[:, None] < active, ids, -1
                )
                with recorder.with_current_layer(layer):
                    dispatched = dispatcher.dispatch(
                        x[:bucket], StandardTopKOutput(factors[:bucket], ids, None)
                    )
                outputs.append(
                    dispatcher.combine(
                        run_nccl_ep_triton(dispatched, quants[layer], config)
                    )
                )
            return outputs

        for generation in range(generations):
            backend = backend_for(coordinator, capacity)
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream), backend.capture_session(
                capture_stream
            ):
                for bucket in (8, 4):
                    backend.capture_one(
                        ShapeKey(bucket), lambda bucket=bucket: forward(bucket)
                    )
            torch.cuda.current_stream().wait_stream(capture_stream)
            native_updates = sum(
                event["operation"] == "handle_update" for event in audit.events
            )
            for step in range(replays):
                native_handles = len(audit.handles)
                target = metadata(
                    [layouts[(step + layer + 1) % len(layouts)] for layer in range(2)],
                    ep_size=2,
                    rank=rank,
                )
                for layer in range(2):
                    assert (
                        updater.update(weights, target, [layer], nnodes=1, rank=rank)
                        == {}
                    )
                    slots = old.physical_to_logical_map[
                        layer, rank * 3 : (rank + 1) * 3
                    ]
                    for name in quant_fields:
                        actual = getattr(quants[layer], name)
                        expected = getattr(global_quant, name).index_select(0, slots)
                        # Compare FP8 bytes directly, without a dtype-dependent tolerance.
                        torch.testing.assert_close(
                            actual.view(torch.uint8),
                            expected.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                assert tensor_addresses(old, weights) == addresses
                counts = cases[step % len(cases)]
                bucket = 4 if max(counts) <= 4 else 8
                ids_cpu = (
                    torch.arange(bucket)[:, None] + torch.tensor([step, step + 1])
                ) % 4
                x_cpu = torch.full(
                    (bucket, 2048), 0.125 * (step % 3 + 1), dtype=torch.bfloat16
                )
                with backend.replay_session():
                    logical_ids[:bucket].copy_(ids_cpu)
                    x[:bucket].copy_(x_cpu)
                    active.fill_(counts[rank])
                    for gatherer in recorder._single_pass_gatherers.values():
                        gatherer.reset()
                    outputs = backend.replay(ShapeKey(bucket), None)
                logical_routes = tuple(
                    torch.where(torch.arange(bucket)[:, None] < count, ids_cpu, -1)
                    for count in counts
                )
                batch = RoutingBatch(
                    (x_cpu, x_cpu),
                    logical_routes,
                    (factors[:bucket].cpu(), factors[:bucket].cpu()),
                    4,
                )
                expected = expected_output(
                    batch,
                    rank,
                    global_quant,
                    shared if sbo else lambda tokens: torch.zeros_like(tokens),
                    1.0,
                    compute_backend="triton",
                )
                for output in outputs:
                    torch.testing.assert_close(
                        output.cpu().float(), expected, rtol=0.02, atol=0.02
                    )
                expected_counts = received_counts(
                    old.physical_to_logical_map_cpu.tolist(), logical_routes
                )
                # Check each physical slot, including work received by idle
                # senders, against source routing and per-rank replica selection.
                for gatherer in recorder._single_pass_gatherers.values():
                    received = gatherer.collect()["global_physical_count"]
                    dist.all_reduce(received)
                    torch.testing.assert_close(
                        received.cpu(), expected_counts, rtol=0, atol=0
                    )
                assert len(audit.handles) == native_handles
                assert (
                    sum(event["operation"] == "handle_update" for event in audit.events)
                    == native_updates
                )
                # Eager uses true input lengths, including a zero-token rank,
                # before the next replay returns to the persistent Graph group.
                with nccl_ep_eager_session():
                    eager = forward(counts[rank])
                for output in eager:
                    torch.testing.assert_close(
                        output.cpu().float(),
                        expected[: counts[rank]],
                        rtol=0.02,
                        atol=0.02,
                    )
                checked += 1
            backend.cleanup()
        dist.barrier()
        closure = close_runtime(coordinator, audit)
    return dict(
        rank=rank,
        checked=checked,
        generations=generations,
        sbo=sbo,
        bindings=bindings,
        closure=closure,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument(
        "--sbo",
        action="store_true",
        help="Also run real DeepSeek shared-expert/dispatch overlap",
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.replays < 4 or args.generations < 1:
        parser.error("use at least four replays and one generation")
    try:
        report = exercise(
            replays=args.replays, generations=args.generations, sbo=args.sbo
        )
    except Unavailable as error:
        print(f"SKIP: {error}")
        raise SystemExit(77)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    mode = "eplb-sbo" if args.sbo else "eplb"
    path = args.report_dir / f"{mode}-rank{report['rank']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
