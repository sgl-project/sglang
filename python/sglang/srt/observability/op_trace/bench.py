"""Fixed-shape e2e op benchmark (experimental): one process, one model, several
workloads; writes traces, per-workload op summaries and benchmark records.

Decode steps replay full CUDA graphs and are traced GPU-only (kineto CUDA
activity). Prefill runs eager: one CPU+GPU step joins its kernels to the
``op::`` module spans, then a GPU-only pass of the same call is timed and
joined to it by kernel ordinal. Prefill CUDA graphs are disabled for that, a
deliberate divergence from the default serving config. TP=1 only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import msgspec
import torch

from sglang.srt.environ import envs


class Workload(msgspec.Struct, frozen=True):
    mode: str  # "decode" | "prefill"
    size: int  # decode: batch size; prefill: prompt tokens (bs=1)

    @property
    def tag(self) -> str:
        return f"{self.mode}_{'bs' if self.mode == 'decode' else 'len'}{self.size}"


def parse_workloads(text: str) -> list[Workload]:
    out = []
    for item in text.split(","):
        mode, size = item.split(":")
        if mode not in ("decode", "prefill"):
            raise ValueError(f"unknown workload mode {mode!r} in {item!r}")
        out.append(Workload(mode=mode, size=int(size)))
    return out


def add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--workloads", default="decode:1,decode:8,decode:64,prefill:4096"
    )
    parser.add_argument(
        "--steps", type=int, default=40, help="decode steps per workload"
    )
    parser.add_argument("--prefill-steps", type=int, default=5)
    parser.add_argument(
        "--decode-context", type=int, default=512, help="prompt tokens before decode"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "server_args", nargs=argparse.REMAINDER, help="extra ServerArgs after `--`"
    )


def run_bench(args: argparse.Namespace) -> str:
    """Returns the directory the records were appended to."""
    from sglang.kernels.jit.benchmark.marker.record import sample_gpu_state

    workloads = parse_workloads(args.workloads)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    # Both must be set before the model loads: the op map is written at capture.
    envs.SGLANG_ENABLE_OP_TRACE_MAP.set(True)
    envs.SGLANG_TORCH_PROFILER_DIR.set(out_dir)
    runner = _load_runner(args=args, workloads=workloads)
    for workload in workloads:
        gpu_before = sample_gpu_state()
        trace, reference = _profile_workload(
            runner=runner, workload=workload, args=args, out_dir=out_dir
        )
        gpu_after = sample_gpu_state()
        _summarize_and_export(
            trace=trace,
            reference=reference,
            workload=workload,
            args=args,
            out_dir=out_dir,
            gpu_state=(gpu_before, gpu_after),
        )
    return out_dir


def _server_argv(*, args: argparse.Namespace, workloads: list[Workload]) -> list[str]:
    extra = [a for a in args.server_args if a != "--"]
    argv = ["--model-path", args.model, *extra]
    decode_bs = sorted({w.size for w in workloads if w.mode == "decode"})
    if decode_bs and "--cuda-graph-bs-decode" not in extra:
        argv += ["--cuda-graph-bs-decode", *map(str, decode_bs)]
    if "--disable-prefill-cuda-graph" not in extra:
        argv.append("--disable-prefill-cuda-graph")
    return argv


def _load_runner(*, args: argparse.Namespace, workloads: list[Workload]):
    from sglang.benchmark import one_batch
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import configure_logger

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server_args = ServerArgs.from_cli_args(
        parser.parse_args(_server_argv(args=args, workloads=workloads))
    )
    if server_args.tp_size != 1:
        raise ValueError("op_trace bench supports tp_size=1 only")
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=0, pp_rank=0), gpu_id=0
        ),
    )
    one_batch._init_process_global_configs()
    configure_logger(server_args, prefix=" TP0")
    runner, _ = one_batch.load_model(server_args, port_args, 0, 0)
    return runner


def _profile_workload(*, runner, workload: Workload, args, out_dir: str) -> tuple:
    """Returns (timing trace, eager reference or None)."""
    from sglang.srt.observability.op_trace import capture
    from sglang.srt.observability.op_trace.analyze import eager_reference_from_trace

    stem = os.path.join(out_dir, workload.tag)
    if workload.mode == "decode":
        capture.set_eager_spans(False)
        trace = _trace_decode(
            runner=runner, workload=workload, args=args, path=f"{stem}.trace.json.gz"
        )
        reference = None
    else:
        # CPU tracing and the op spans make eager prefill partly CPU-bound,
        # which changes kernel durations, so one CPU+GPU step is used only for
        # attribution and a GPU-only pass of the same call for timing.
        capture.set_eager_spans(True)
        attr_trace = _trace_prefill(
            runner=runner,
            workload=workload,
            path=f"{stem}.attr.trace.json.gz",
            steps=1,
            cpu=True,
        )
        reference = eager_reference_from_trace(attr_trace)
        capture.set_eager_spans(False)
        trace = _trace_prefill(
            runner=runner,
            workload=workload,
            path=f"{stem}.trace.json.gz",
            steps=args.prefill_steps,
            cpu=False,
        )
    capture.set_eager_spans(None)
    runner.clear()
    return trace, reference


def _trace_decode(*, runner, workload: Workload, args, path: str) -> str:
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test

    runner.clear()
    reqs = prepare_synthetic_inputs_for_latency_test(workload.size, args.decode_context)
    next_ids, _, batch = runner.extend(reqs)
    for _ in range(10):
        next_ids, _ = runner.decode(next_ids, batch)
    runner.synchronize()
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
    prof.start()
    for _ in range(args.steps):
        next_ids, _ = runner.decode(next_ids, batch)
    runner.synchronize()
    prof.stop()
    prof.export_chrome_trace(path)
    return path


def _trace_prefill(
    *, runner, workload: Workload, path: str, steps: int, cpu: bool
) -> str:
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test

    activities = [torch.profiler.ProfilerActivity.CUDA]
    if cpu:
        activities.insert(0, torch.profiler.ProfilerActivity.CPU)
    prof = torch.profiler.profile(activities=activities)
    # The first extend is a warmup outside the profiler.
    for i in range(steps + 1):
        runner.clear()
        reqs = prepare_synthetic_inputs_for_latency_test(1, workload.size)
        runner.synchronize()
        if i == 1:
            prof.start()
        runner.extend(reqs)
        runner.synchronize()
    prof.stop()
    prof.export_chrome_trace(path)
    return path


def _summarize_and_export(
    *, trace: str, reference, workload: Workload, args, out_dir: str, gpu_state
) -> None:
    from sglang.srt.observability.op_trace.analyze import (
        analyze_trace,
        discover_op_maps,
    )
    from sglang.srt.observability.op_trace.export import (
        summary_to_records,
        write_records,
    )
    from sglang.srt.observability.op_trace.report import render_summary

    summary = analyze_trace(
        trace_path=trace,
        op_map_paths=discover_op_maps(trace),
        eager_reference=reference,
        eager_steps=args.prefill_steps if reference is not None else None,
    )
    stem = os.path.join(out_dir, workload.tag)
    Path(f"{stem}.summary.json").write_bytes(msgspec.json.encode(summary))
    Path(f"{stem}.md").write_text(render_summary(summary, top=args.top))
    params: dict[str, Any] = {
        "mode": workload.mode,
        "bs": workload.size if workload.mode == "decode" else 1,
        "input_len": args.decode_context
        if workload.mode == "decode"
        else workload.size,
    }
    bench_id = f"e2e-op::{os.path.basename(args.model.rstrip('/'))}::{workload.mode}"
    gpu_before, gpu_after = gpu_state
    records = summary_to_records(
        summary,
        bench_id=bench_id,
        params=params,
        world_size=1,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
    )
    write_records(records, output_dir=out_dir)
