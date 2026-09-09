"""Matched SGLang eager/full-Graph synthetic steps, without EP audit wrappers.

Both modes copy GPU-resident inputs, run two serial MoE layers, and retain the
same receive/counter/combine observations. CUDA events measure the whole step,
including host launch gaps, not the sum of EP kernel durations. Each sample
waits for its end event: this is serial latency, not pipelined throughput.
"""

import math
import statistics
import time

import torch
import torch.distributed as dist

from .comparison import compare
from .dispatcher import forward_layer, initialize
from .oracle import make_fixture, validate_capacity
from .sglang_graph import backend_for, dispatchers_for, gpu_inputs

METRICS = ("cuda_step_ms", "host_submit_ms", "host_step_ms")


def statistics_ms(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timing samples must be finite and positive")
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"count": len(values), "median": statistics.median(values), "p95": p95}


def benchmark_steps(
    rank,
    coordinator,
    dispatchers,
    *,
    buckets=(8, 16, 32),
    cases=("balanced", "hotspot"),
    samples=200,
    warmups=20,
    rounds=4,
    fixture=make_fixture,
):
    """Shared real driver; local tests substitute only the EP library/fixtures."""
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        nccl_ep_eager_session,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    buckets = sorted(set(buckets))
    if (
        not buckets
        or min(buckets) < 1
        or max(buckets) > 1024
        or not cases
        or len(set(cases)) != len(cases)
        or samples < 2
        or warmups < 2
        or rounds < 2
        or rounds % 2
        or len(dispatchers) != 2
    ):
        raise ValueError(
            "Use two layers, positive buckets, >=2 samples/warmups, even rounds >=2"
        )
    capacity = max(buckets)
    prepared = {}
    started = time.perf_counter()
    for bucket in buckets:
        for case in cases:
            pairs = []
            for step in (0, 1):
                batches = [
                    fixture(bucket, case=case, step=step + layer, change="all")
                    for layer in range(2)
                ]
                for batch in batches:
                    validate_capacity(batch, capacity)
                pairs.append((batches, [gpu_inputs(batch, rank) for batch in batches]))
            prepared[bucket, case] = pairs
    static = [
        tuple(x.clone() for x in inputs)
        for inputs in prepared[capacity, cases[0]][0][1]
    ]
    torch.cuda.synchronize()
    input_preparation_seconds = time.perf_counter() - started
    backend = backend_for(coordinator, capacity)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    records, checked = [], 0

    def forward(bucket, *, graph):
        outputs = []
        for dispatcher, inputs in zip(dispatchers, static):
            received, counters, combined = forward_layer(
                dispatcher, *(x[:bucket] for x in inputs), rank
            )
            # Graph combine already makes its ownership clone. Eager must also
            # retain each layer's output before the next layer reuses scratch.
            if not graph:
                combined = combined.clone()
            outputs.append((received, counters, combined))
        return outputs

    def step(mode, bucket, live):
        session = (
            backend.replay_session() if mode == "graph" else nccl_ep_eager_session()
        )
        with session:
            for inputs, sources in zip(static, live):
                for target, source in zip(inputs, sources):
                    target[:bucket].copy_(source)
            return (
                backend.replay(ShapeKey(bucket), None)
                if mode == "graph"
                else forward(bucket, graph=False)
            )

    try:
        with torch.cuda.stream(stream):
            coordinator.barrier()
            started = time.perf_counter()
            # Pay eager group creation and first-use/JIT before any sampling.
            for bucket in buckets:
                step("eager", bucket, prepared[bucket, cases[0]][0][1])
            stream.synchronize()
            eager_first_use_seconds = time.perf_counter() - started

            coordinator.barrier()
            started = time.perf_counter()
            with backend.capture_session(stream):
                for bucket in reversed(buckets):
                    backend.capture_one(
                        ShapeKey(bucket),
                        lambda bucket=bucket: forward(bucket, graph=True),
                    )
            stream.synchronize()
            warmup_capture_seconds = time.perf_counter() - started

            # Materialize reusable events outside samples, including their
            # first record. Neither allocation nor first-use enters timings.
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(samples)
            ]
            for start, end in events:
                start.record()
                end.record()
            stream.synchronize()
            for bucket in buckets:
                for case in cases:
                    pairs = prepared[bucket, case]
                    for round_index in range(rounds):
                        order = (
                            ("eager", "graph")
                            if round_index % 2 == 0
                            else ("graph", "eager")
                        )
                        for mode in order:
                            # Exact receive/combine checks of both data variants
                            # precede warmup; no CPU oracle enters the sample loop.
                            for batches, live in pairs:
                                actual = step(mode, bucket, live)
                                stream.synchronize()
                                for batch, output in zip(batches, actual):
                                    compare(batch, rank, *output)
                                    checked += 1
                            warmup_started = time.perf_counter()
                            for iteration in range(warmups):
                                step(mode, bucket, pairs[iteration % 2][1])
                            stream.synchronize()
                            warmup_seconds = time.perf_counter() - warmup_started
                            coordinator.barrier()
                            measured = {metric: [] for metric in METRICS}
                            for iteration, (start, end) in enumerate(events):
                                live = pairs[iteration % 2][1]
                                host_started = time.perf_counter()
                                start.record()
                                actual = step(mode, bucket, live)
                                end.record()
                                submitted = time.perf_counter()
                                end.synchronize()
                                completed = time.perf_counter()
                                measured["cuda_step_ms"].append(start.elapsed_time(end))
                                measured["host_submit_ms"].append(
                                    (submitted - host_started) * 1000
                                )
                                measured["host_step_ms"].append(
                                    (completed - host_started) * 1000
                                )
                            for batch, output in zip(
                                pairs[(samples - 1) % 2][0], actual
                            ):
                                compare(batch, rank, *output)
                                checked += 1
                            records.append(
                                {
                                    "bucket": bucket,
                                    "case": case,
                                    "round": round_index,
                                    "mode": mode,
                                    "warmup_seconds": warmup_seconds,
                                    "samples": measured,
                                    "statistics_ms": {
                                        key: statistics_ms(values)
                                        for key, values in measured.items()
                                    },
                                }
                            )
    finally:
        backend.cleanup()
    return {
        "implementation": "sglang_dispatcher_eager_vs_full_graph",
        "config": {
            "buckets": buckets,
            "cases": list(cases),
            "layers": 2,
            "samples_per_round": samples,
            "warmups_per_block": warmups,
            "rounds": rounds,
            "seed": 32774,
            "hidden": 2048,
            "top_k": 2,
            "order": "alternate eager/graph and graph/eager by round",
            "input_sequence": "preloaded step 0/1, alternating every sample",
        },
        "measurement_scope": "Serial two-layer synthetic step: GPU input copies, dispatch, FP8 dequant, expert arithmetic, combine and retained observations; excludes fixture/oracle, initial JIT, warmup, capture and barriers",
        "host_timing_scope": "Includes event recording; host_step_ms also includes end-event synchronization; host_submit_ms is not pure Python overhead",
        "output_ownership": "One combine clone per layer in both modes; Graph inside dispatcher, eager in driver",
        "input_preparation_seconds": input_preparation_seconds,
        "eager_first_use_seconds": eager_first_use_seconds,
        "warmup_capture_seconds": warmup_capture_seconds,
        "checked": checked,
        "tolerance": {"rtol": 0, "atol": 0},
        "records": records,
        "model_loaded": False,
    }


def exercise_benchmark(**kwargs):
    started = time.perf_counter()
    rank, coordinator, bindings = initialize(max(kwargs["buckets"]), graph_enabled=True)
    initialization_seconds = time.perf_counter() - started
    try:
        result = benchmark_steps(
            rank, coordinator, dispatchers_for(coordinator, 2), **kwargs
        )
    finally:
        import nccl.core as core

        from sglang.srt.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        core.Communicator(ptr=coordinator.pynccl_comm.comm.value).destroy()
        coordinator.pynccl_comm.available = False
        coordinator.pynccl_comm.disabled = True
        dist.destroy_process_group()
    result.update(
        bindings=bindings,
        initialization_seconds=initialization_seconds,
        native_ep_tested=True,
        resource_cleanup_returned=True,
    )
    return result


def summarize_pair(reports):
    """Align rounds/samples across ranks before computing critical-rank latency."""
    if len(reports) != 2 or [x.get("rank") for x in reports] != [0, 1]:
        raise ValueError("Both rank 0 and rank 1 reports are required")
    if any(x.get("status") != "PASS" for x in reports):
        raise ValueError("Both benchmark ranks must pass")
    results = [x["result"] for x in reports]
    if any(
        x.get("native_ep_tested") is not True
        or x.get("resource_cleanup_returned") is not True
        or x.get("tolerance") != {"rtol": 0, "atol": 0}
        for x in results
    ):
        raise ValueError(
            "Real EP benchmark, exact checks and completed cleanup required"
        )
    if results[0]["config"] != results[1]["config"]:
        raise ValueError("Rank benchmark configurations differ")
    config = results[0]["config"]
    expected = {
        (bucket, case, rnd, mode)
        for bucket in config["buckets"]
        for case in config["cases"]
        for rnd in range(config["rounds"])
        for mode in ("eager", "graph")
    }
    indexed = []
    for result in results:
        mapping = {
            (r["bucket"], r["case"], r["round"], r["mode"]): r["samples"]
            for r in result["records"]
        }
        if len(mapping) != len(result["records"]) or set(mapping) != expected:
            raise ValueError("Missing or duplicate benchmark block")
        for metrics in mapping.values():
            if set(metrics) != set(METRICS):
                raise ValueError("Missing timing metric")
            for values in metrics.values():
                if len(values) != config["samples_per_round"]:
                    raise ValueError("Incomplete sample block")
                statistics_ms(values)
        indexed.append(mapping)
    rows = []
    for bucket in config["buckets"]:
        for case in config["cases"]:
            modes = {}
            for mode in ("eager", "graph"):
                modes[mode] = {}
                for metric in METRICS:
                    values = [
                        [
                            value
                            for rnd in range(config["rounds"])
                            for value in rank[bucket, case, rnd, mode][metric]
                        ]
                        for rank in indexed
                    ]
                    modes[mode][metric] = {
                        "rank0": statistics_ms(values[0]),
                        "rank1": statistics_ms(values[1]),
                        "max_rank_per_sample": statistics_ms(
                            [max(a, b) for a, b in zip(*values)]
                        ),
                    }
            rows.append(
                {
                    "bucket": bucket,
                    "case": case,
                    "modes": modes,
                    "median_speedup": {
                        metric: modes["eager"][metric]["max_rank_per_sample"]["median"]
                        / modes["graph"][metric]["max_rank_per_sample"]["median"]
                        for metric in ("cuda_step_ms", "host_step_ms")
                    },
                }
            )
    return {
        "config": config,
        "rows": rows,
        "units": "ms",
        "full_model_benchmark": False,
    }
