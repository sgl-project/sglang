"""Matched two-layer MoE timing with real shared/Triton experts and native EP.

Run each configuration in a fresh torchrun process. Attention is an identity
fixture; these measurements do not establish serving throughput or model quality.
No EP audit wrappers or EPLB weight migration run inside the timed steps.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from .benchmark import benchmark_steps, statistics_ms
from .dispatcher import initialize
from .environment import Unavailable
from .eplb import metadata
from .followup_server import source_head
from .moe_model import make_moe
from .oracle import make_fixture
from .pair_followups import expected_output
from .shared_compute import make_shared_mlp
from .tbo_model import forward_tbo
from .triton_compute import configure_compute, make_compute_fixture


def fixture(bucket, **kwargs):
    batch = make_fixture(bucket, **kwargs)
    return replace(batch, tokens=tuple(x / 64 for x in batch.tokens))


def run(coordinator, *, sbo=False, tbo=False, fixture_fn=fixture, **options):
    from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
        NcclEpBuffer,
        NcclEpDispatcher,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_flags, get_resources

    rank = coordinator.rank
    experts = coordinator.world_size * 2
    configure_compute(graph_enabled=True)
    placement = metadata(
        [list(range(experts))] * 2, ep_size=coordinator.world_size, rank=rank
    )
    _, global_quant, config = make_compute_fixture(hidden=2048, experts=experts)
    config = replace(
        config, num_experts=experts, num_local_experts=2, routed_scaling_factor=1.0
    )
    local_quant = replace(
        global_quant,
        **{
            name: getattr(global_quant, name)[rank * 2 : rank * 2 + 2]
            for name in ("w13_weight", "w2_weight", "w13_scale", "w2_scale")
        },
    )

    class Router:
        start = 0

        def set_subbatch(self, index, bounds):
            self.start = bounds[0]

        def __call__(self, hidden_states, router_logits, **kwargs):
            rows = slice(self.start, self.start + len(hidden_states))
            return StandardTopKOutput(self.weights[rows], self.ids[rows], None)

    with get_flags().moe.override(
        sbo_enabled=sbo, tbo_enabled=tbo, runner_backend=MoeRunnerBackend.TRITON
    ), patch.object(get_resources(), "expert_location_metadata", placement), patch(
        "sglang.srt.layers.linear.get_tp_group", return_value=coordinator
    ):
        shared = make_shared_mlp(hidden=2048)
        dispatcher_type = MaybeTboDeepEPDispatcher if tbo else NcclEpDispatcher
        models = [
            make_moe(
                dispatcher_type(
                    moe_runner_config=replace(config, layer_id=layer),
                    ep_group=coordinator,
                ),
                local_quant,
                replace(config, layer_id=layer),
                shared,
                Router(),
                scale=1.0,
                sbo=sbo,
            )
            for layer in range(2)
        ]

        def layer(model, x, ids, weights, rank):
            model.topk.ids, model.topk.weights = ids, weights
            if tbo:
                split = len(x) // 2
                combined = forward_tbo(
                    [model],
                    x,
                    split=split,
                    padded=(split, len(x) - split),
                    counts=(None, None),
                    mode=ForwardMode.DECODE,
                )
            else:
                combined = model.forward_deepep(
                    x, SimpleNamespace(num_token_non_padded=None)
                )
            return None, None, combined

        def verify(batch, rank, received, counters, combined):
            wanted = expected_output(
                batch, rank, global_quant, shared, 1.0, compute_backend="triton"
            )
            torch.testing.assert_close(
                combined.cpu().float(), wanted, rtol=0.02, atol=0.02
            )

        try:
            result = benchmark_steps(
                rank,
                coordinator,
                models,
                fixture=fixture_fn,
                run_layer=layer,
                verify=verify,
                retain_eager_output=False,
                **options,
            )
        finally:
            torch.cuda.synchronize()
            NcclEpBuffer.destroy()
    result.update(
        implementation="nccl_ep_shared_triton_overlap",
        tolerance={"rtol": 0.02, "atol": 0.02},
        measurement_scope="Two independent MoE layers with shared MLP, Triton routed GEMMs, GPU input/routing/weight copies and native EP. TBO uses identity attention stages. Excludes audit wrappers, oracle, warmup, capture, JIT, barriers and EPLB migration.",
        output_ownership="Production model outputs; no receive snapshots or extra driver clones.",
    )
    result["config"].update(
        sbo=sbo,
        tbo=tbo,
        routed_scaling_factor=1.0,
        shared_experts=2,
        world_size=coordinator.world_size,
        routed_experts=experts,
        routed_intermediate=global_quant.w2_weight.shape[-1],
        shared_intermediate=shared.down_proj.weight.shape[-1],
        weight_dtype="block128_fp8",
        activation_dtype="bf16",
    )
    return result


def summarize(reports):
    if len(reports) != 2 or [r.get("rank") for r in reports] != [0, 1]:
        raise ValueError("Both rank reports are required in rank order")
    for report in reports:
        if (
            not report.get("passed")
            or not report.get("native_ep_tested")
            or report.get("result", {}).get("implementation")
            != "nccl_ep_shared_triton_overlap"
        ):
            raise ValueError("Both native benchmark ranks must pass")
    if (
        reports[0]["source_head"] != reports[1]["source_head"]
        or reports[0]["result"]["config"] != reports[1]["result"]["config"]
    ):
        raise ValueError("Source commits and benchmark configurations must match")
    config = reports[0]["result"]["config"]
    expected = {
        (b, c, r, m)
        for b in config["buckets"]
        for c in config["cases"]
        for r in range(config["rounds"])
        for m in ("eager", "graph")
    }
    indexed = []
    for report in reports:
        records = report["result"]["records"]
        mapping = {
            (r["bucket"], r["case"], r["round"], r["mode"]): r["samples"]
            for r in records
        }
        if set(mapping) != expected or len(mapping) != len(records):
            raise ValueError("Missing or duplicate benchmark block")
        indexed.append(mapping)
    rows = []
    for bucket in config["buckets"]:
        for case in config["cases"]:
            for mode in ("eager", "graph"):
                metrics = {}
                for metric in ("cuda_step_ms", "host_submit_ms", "host_step_ms"):
                    values = []
                    for rnd in range(config["rounds"]):
                        pair = [
                            rank[bucket, case, rnd, mode][metric] for rank in indexed
                        ]
                        if any(len(v) != config["samples_per_round"] for v in pair):
                            raise ValueError("Incomplete timing sample block")
                        for samples in pair:
                            statistics_ms(samples)
                        values += [max(a, b) for a, b in zip(*pair)]
                    metrics[metric] = statistics_ms(values)
                rows.append(
                    dict(
                        bucket=bucket,
                        case=case,
                        mode=mode,
                        max_rank_per_sample_ms=metrics,
                    )
                )
    return dict(
        source_head=reports[0]["source_head"],
        config=config,
        rows=rows,
        model_loaded=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbo", action="store_true")
    parser.add_argument("--tbo", action="store_true")
    parser.add_argument("--buckets", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--summarize", nargs=2, type=Path)
    args = parser.parse_args()
    if args.summarize:
        print(
            json.dumps(
                summarize([json.loads(p.read_text()) for p in args.summarize]), indent=2
            )
        )
        return
    if args.report_dir is None:
        parser.error("--report-dir is required for measurement")
    if any(b < 2 or b > 1024 or b % 2 for b in args.buckets):
        parser.error("use even buckets in [2, 1024] for matched TBO comparisons")
    if args.samples < 2 or args.warmups < 2 or args.rounds < 2 or args.rounds % 2:
        parser.error("use >=2 samples/warmups and even rounds >=2")
    try:
        rank, coordinator, bindings = initialize(max(args.buckets), graph_enabled=True)
        result = run(
            coordinator,
            sbo=args.sbo,
            tbo=args.tbo,
            buckets=args.buckets,
            samples=args.samples,
            warmups=args.warmups,
            rounds=args.rounds,
        )
        import nccl.core as core

        core.Communicator(ptr=coordinator.pynccl_comm.comm.value).destroy()
        coordinator.pynccl_comm.available = False
        coordinator.pynccl_comm.disabled = True
        dist.destroy_process_group()
    except Unavailable as error:
        print(f"SKIP: {error}")
        raise SystemExit(77)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / f"overlap-rank{rank}.json"
    path.write_text(
        json.dumps(
            dict(
                rank=rank,
                source_head=source_head(),
                passed=True,
                native_ep_tested=True,
                bindings=bindings,
                hardware=dict(
                    name=torch.cuda.get_device_name(),
                    sm=list(torch.cuda.get_device_capability()),
                    torch=torch.__version__,
                    torch_cuda=torch.version.cuda,
                ),
                result=result,
            ),
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
