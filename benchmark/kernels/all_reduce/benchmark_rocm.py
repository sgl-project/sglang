"""Benchmark ROCm all-reduce implementations and generate a Markdown report.

Benchmark examples:
    python benchmark_rocm.py --sweep-output-dir results --dtype bf16

    torchrun --nproc_per_node=8 benchmark_rocm.py --mode eager --dtype bf16 \
        --csv-out results/tp8_eager_bf16.csv
    torchrun --nproc_per_node=8 benchmark_rocm.py --mode graph --dtype bf16 \
        --csv-out results/tp8_graph_bf16.csv

Report example (no torchrun or GPU required):
    python benchmark_rocm.py --report-input-dir results \
        --report-out results/allreduce_report.md
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import signal
import subprocess
import sys
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

IMPLEMENTATIONS = (
    "rccl",
    "pynccl",
    "sgl-v1",
    "sgl-v2",
    "aiter",
    "torch-symm-mem",
    "qr-fp",
    "qr-fp8",
    "qr-int6",
    "qr-int4",
    "qr-int3",
)
LOSSLESS_IMPLEMENTATIONS = (
    "rccl",
    "pynccl",
    "sgl-v1",
    "sgl-v2",
    "aiter",
    "torch-symm-mem",
)
QUICK_REDUCE_IMPLEMENTATIONS = (
    "qr-fp",
    "qr-fp8",
    "qr-int6",
    "qr-int4",
    "qr-int3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ROCm all-reduce backends, run the full matrix, "
            "or render prior CSVs."
        )
    )
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters-small", type=int, default=101)
    parser.add_argument("--iters-large", type=int, default=101)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num-inputs", type=int, default=16)
    parser.add_argument("--mode", choices=["eager", "graph"], default="eager")
    parser.add_argument(
        "--impls",
        default=",".join(IMPLEMENTATIONS),
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--min-size-kb", type=int, default=1)
    parser.add_argument("--max-size-mb", type=int, default=512)
    parser.add_argument(
        "--size-bytes",
        type=int,
        help="Benchmark one exact message size; used by isolated sweep workers.",
    )
    parser.add_argument(
        "--comm-buffer-mb",
        type=int,
        default=64,
        help=(
            "Minimum communicator staging-buffer size. AITER CAR can produce "
            "invalid output with very small exact-size buffers."
        ),
    )
    parser.add_argument("--input-pool-mb", type=int, default=512)
    parser.add_argument("--csv-out")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--sgl-v2-shot", type=int, choices=[1, 2])
    parser.add_argument("--sgl-v2-pull-blocks", type=int)
    parser.add_argument("--sgl-v2-pull-threads", type=int)
    parser.add_argument(
        "--report-input-dir",
        help="Read all benchmark CSVs in this directory instead of benchmarking.",
    )
    parser.add_argument(
        "--report-out",
        help="Markdown report path for report-only or sweep mode.",
    )
    parser.add_argument(
        "--sweep-output-dir",
        help=(
            "Run the complete TP2/TP4/TP8 eager+graph matrix and write CSVs, "
            "logs, and allreduce_report.md here."
        ),
    )
    parser.add_argument(
        "--sweep-tp",
        default="2,4,8",
        help="Comma-separated tensor-parallel sizes for --sweep-output-dir.",
    )
    parser.add_argument(
        "--sweep-modes",
        default="eager,graph",
        help="Comma-separated modes for --sweep-output-dir.",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=2700,
        help="Per-torchrun timeout in sweep mode.",
    )
    args = parser.parse_args()
    if args.report_input_dir and not args.report_out:
        parser.error("--report-input-dir and --report-out must be used together")
    if args.report_out and not (args.report_input_dir or args.sweep_output_dir):
        parser.error("--report-out requires --report-input-dir or --sweep-output-dir")
    if args.report_input_dir and args.sweep_output_dir:
        parser.error("--report-input-dir and --sweep-output-dir are mutually exclusive")
    if args.size_bytes is not None and args.size_bytes <= 0:
        parser.error("--size-bytes must be positive")
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    return args


def get_env_rank_world() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def human_size(num_bytes: int) -> str:
    for suffix, base in (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if num_bytes >= base and num_bytes % base == 0:
            return f"{num_bytes // base} {suffix}"
    return f"{num_bytes} B"


def get_message_sizes(min_size_kb: int, max_size_mb: int) -> List[int]:
    min_bytes = min_size_kb * 1024
    max_bytes = max_size_mb * 1024 * 1024
    if min_bytes <= 0 or max_bytes < min_bytes:
        raise ValueError("Require 0 < --min-size-kb <= --max-size-mb")
    size = 1024
    while size < min_bytes:
        size *= 2
    result = []
    while size <= max_bytes:
        result.append(size)
        size *= 2
    return result


def parse_impls(raw: str) -> List[str]:
    impls = [value.strip().lower() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(impls) - set(IMPLEMENTATIONS))
    if unknown:
        raise ValueError(f"Unknown implementations: {unknown}")
    if not impls:
        raise ValueError("--impls cannot be empty")
    return impls


class RcclAllReduce:
    graph_supported = False

    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.outputs: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}

    def benchmark_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        key = (inp.numel(), inp.dtype)
        out = self.outputs.get(key)
        if out is None:
            out = torch.empty_like(inp)
            self.outputs[key] = out
        out.copy_(inp)
        dist.all_reduce(out, group=self.group)
        return out


class PyNcclAllReduce:
    def __init__(self, group: dist.ProcessGroup, device: torch.device):
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.comm = PyNcclCommunicator(group=group, device=device)
        self.comm.disabled = False

    def benchmark_all_reduce(self, inp: torch.Tensor) -> Optional[torch.Tensor]:
        return self.comm.outplace_all_reduce(inp)


def construct_impl(
    name: str,
    pg: dist.ProcessGroup,
    reference_pg: Optional[dist.ProcessGroup],
    device: torch.device,
    max_size: int,
    args: argparse.Namespace,
):
    try:
        if name == "rccl":
            if reference_pg is None:
                raise RuntimeError("RCCL process group is unavailable")
            return RcclAllReduce(reference_pg)
        if name == "pynccl":
            return PyNcclAllReduce(pg, device)
        if name == "sgl-v1":
            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            return CustomAllreduce(group=pg, device=device, max_size=max_size)
        if name == "sgl-v2":
            from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
                CustomAllReduceV2,
            )

            comm = CustomAllReduceV2(
                group=pg,
                device=device,
                max_size=max_size,
                max_pull_blocks=args.sgl_v2_pull_blocks,
            )
            if args.sgl_v2_shot is not None:
                comm.override_shot(args.sgl_v2_shot)
            if args.sgl_v2_pull_threads is not None:
                config = comm.obj.config_pull()
                comm.obj.config_pull(
                    num_blocks=args.sgl_v2_pull_blocks or config.num_blocks,
                    num_threads=args.sgl_v2_pull_threads,
                )
            return comm
        if name == "aiter":
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            return CustomAllreduce(group=pg, device=device, max_size=max_size)
        if name == "torch-symm-mem":
            from sglang.srt.distributed.device_communicators.torch_symm_mem import (
                TorchSymmMemCommunicator,
            )

            return TorchSymmMemCommunicator(group=pg, device=device)
        if name.startswith("qr-"):
            os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = name.removeprefix(
                "qr-"
            ).upper()
            os.environ["AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] = str(
                max_size // (1024 * 1024)
            )
            from aiter.dist.device_communicators.quick_all_reduce import (
                QuickAllReduce,
            )

            return QuickAllReduce(group=pg, device=device)
    except Exception as exc:
        if dist.get_rank() == 0:
            print(f"Failed to construct {name}: {exc}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
    return None


@torch.inference_mode()
def run_once(comm, inp: torch.Tensor) -> Optional[torch.Tensor]:
    if getattr(comm, "disabled", False):
        return None
    if hasattr(comm, "should_quick_allreduce"):
        if not comm.should_quick_allreduce(inp):
            return None
        return comm.quick_all_reduce(inp)
    if hasattr(comm, "benchmark_all_reduce"):
        return comm.benchmark_all_reduce(inp)
    if hasattr(comm, "all_reduce_unreg"):
        return comm.all_reduce_unreg(inp)
    if hasattr(comm, "custom_all_reduce"):
        return comm.custom_all_reduce(inp)
    if hasattr(comm, "all_reduce"):
        return comm.all_reduce(inp)
    raise RuntimeError("No known all-reduce method on communicator")


def correctness_tolerances(
    name: str, dtype: torch.dtype, world_size: int
) -> Tuple[float, float]:
    if name == "qr-fp":
        base = 5e-3 if dtype == torch.float16 else 4e-2
        return base * world_size, base * world_size
    if name.startswith("qr-"):
        return 0.5 * world_size, 1.25 * world_size
    if dtype == torch.bfloat16:
        return 2e-2, 1.25e-1
    return 1e-2, 1e-2


def align_ranks(
    reference_pg: Optional[dist.ProcessGroup],
    control_pg: dist.ProcessGroup,
    device: torch.device,
) -> None:
    dist.barrier(group=control_pg)
    if reference_pg is not None:
        token = torch.zeros(1, device=device)
        dist.all_reduce(token, group=reference_pg)
    torch.cuda.synchronize()
    dist.barrier(group=control_pg)


@torch.inference_mode()
def check_correctness(
    name: str,
    comm,
    inp: torch.Tensor,
    control_pg: dist.ProcessGroup,
    reference_pg: Optional[dist.ProcessGroup],
) -> bool:
    if reference_pg is None:
        return True
    candidate = inp.clone()
    reference = inp.clone()
    out = run_once(comm, candidate)
    if out is None:
        return False
    dist.all_reduce(reference, group=reference_pg)
    torch.cuda.synchronize()
    local_ok = torch.ones(1, dtype=torch.int32)
    try:
        torch.testing.assert_close(
            out,
            reference,
            rtol=correctness_tolerances(name, inp.dtype, dist.get_world_size())[0],
            atol=correctness_tolerances(name, inp.dtype, dist.get_world_size())[1],
        )
    except AssertionError as exc:
        local_ok.zero_()
        if dist.get_rank() == 0:
            print(f"[{name}] correctness failed: {exc}", file=sys.stderr)
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN, group=control_pg)
    return bool(local_ok.item())


def aggregate_times(times_ms: List[float], pg: dist.ProcessGroup) -> Optional[float]:
    stats = torch.tensor(
        [sum(times_ms) / len(times_ms), min(times_ms), max(times_ms)],
        dtype=torch.float64,
    )
    gathered = [torch.zeros_like(stats) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stats, group=pg)
    if dist.get_rank() != 0:
        return None
    all_stats = torch.stack(gathered)
    slowest_rank_avg = float(all_stats[:, 0].max().item())
    print(
        f"{slowest_rank_avg:.3f} ms (slowest-rank avg; "
        f"rank-mean={all_stats[:, 0].mean().item():.3f}, "
        f"sample-min={all_stats[:, 1].min().item():.3f}, "
        f"sample-max={all_stats[:, 2].max().item():.3f})"
    )
    return slowest_rank_avg


@torch.inference_mode()
def bench_eager(
    name: str,
    comm,
    sizes: List[int],
    device: torch.device,
    args: argparse.Namespace,
    dtype: torch.dtype,
    pg: dist.ProcessGroup,
    reference_pg: Optional[dist.ProcessGroup],
) -> List[Tuple[int, Optional[float]]]:
    results = []
    align_ranks(reference_pg, pg, device)
    for size_bytes in sizes:
        elems = size_bytes // torch.empty((), dtype=dtype).element_size()
        base = torch.empty(elems, dtype=dtype, device=device).uniform_(0, 1)
        if not args.skip_correctness and not check_correctness(
            name, comm, base, pg, reference_pg
        ):
            results.append((size_bytes, None))
            continue
        disabled = False
        dist.barrier(group=pg)
        for _ in range(args.warmup):
            out = run_once(comm, base.clone())
            torch.cuda.synchronize()
            if out is None:
                disabled = True
                break
        dist.barrier(group=pg)
        if disabled:
            results.append((size_bytes, None))
            continue
        num_iters = args.iters_small if size_bytes <= 1024 * 1024 else args.iters_large
        rotate_count = max(
            1,
            min(
                args.num_inputs,
                num_iters,
                args.input_pool_mb * 1024 * 1024 // max(1, size_bytes),
            ),
        )
        inputs = [torch.empty_like(base).uniform_(0, 1) for _ in range(rotate_count)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        dist.barrier(group=pg)
        torch.cuda.synchronize()
        for iteration in range(num_iters):
            start.record()
            out = run_once(comm, inputs[iteration % len(inputs)])
            end.record()
            end.synchronize()
            if out is None:
                disabled = True
                break
            elapsed = start.elapsed_time(end)
            times.append(elapsed)
            if args.verbose and dist.get_rank() == 0:
                print(f"[{name}] {human_size(size_bytes)} #{iteration}: {elapsed}")
        dist.barrier(group=pg)
        if disabled or not times:
            results.append((size_bytes, None))
            continue
        if dist.get_rank() == 0:
            print(f"[{name}] {human_size(size_bytes)}: ", end="")
        results.append((size_bytes, aggregate_times(times, pg)))
    return results


@torch.inference_mode()
def bench_graph(
    name: str,
    comm,
    sizes: List[int],
    device: torch.device,
    args: argparse.Namespace,
    dtype: torch.dtype,
    pg: dist.ProcessGroup,
    reference_pg: Optional[dist.ProcessGroup],
) -> List[Tuple[int, Optional[float]]]:
    if not getattr(comm, "graph_supported", True):
        return [(size, None) for size in sizes]
    results = []
    align_ranks(reference_pg, pg, device)
    for size_bytes in sizes:
        elems = size_bytes // torch.empty((), dtype=dtype).element_size()
        graph_inp = torch.empty(elems, dtype=dtype, device=device).uniform_(0, 1)
        reference = graph_inp.clone()
        try:
            dist.barrier(group=pg)
            capture = comm.capture() if hasattr(comm, "capture") else nullcontext()
            with capture:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_out = run_once(comm, graph_inp)
            if graph_out is not None:
                graph.replay()
                torch.cuda.synchronize()
            dist.barrier(group=pg)
        except Exception as exc:
            if dist.get_rank() == 0:
                print(f"[{name}] {human_size(size_bytes)} capture failed: {exc}")
            results.append((size_bytes, None))
            continue
        if graph_out is None:
            results.append((size_bytes, None))
            continue
        if not args.skip_correctness and reference_pg is not None:
            dist.all_reduce(reference, group=reference_pg)
            torch.cuda.synchronize()
            local_ok = torch.ones(1, dtype=torch.int32)
            try:
                rtol, atol = correctness_tolerances(name, dtype, dist.get_world_size())
                torch.testing.assert_close(graph_out, reference, rtol=rtol, atol=atol)
            except AssertionError:
                local_ok.zero_()
            dist.all_reduce(local_ok, op=dist.ReduceOp.MIN, group=pg)
            if not bool(local_ok.item()):
                results.append((size_bytes, None))
                continue
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        num_iters = args.iters_small if size_bytes <= 1024 * 1024 else args.iters_large
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        dist.barrier(group=pg)
        for iteration in range(num_iters):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            times.append(elapsed)
            if args.verbose and dist.get_rank() == 0:
                print(
                    f"[{name}] graph {human_size(size_bytes)} #{iteration}: {elapsed}"
                )
        dist.barrier(group=pg)
        if dist.get_rank() == 0:
            print(f"[{name}] graph {human_size(size_bytes)}: ", end="")
        results.append((size_bytes, aggregate_times(times, pg)))
    return results


def close_comm(comm, pg: dist.ProcessGroup) -> None:
    if hasattr(comm, "close"):
        try:
            comm.close()
        except Exception:
            pass
    dist.barrier(group=pg)
    del comm
    gc.collect()
    torch.cuda.empty_cache()


def write_csv(
    path: str,
    results: Dict[str, List[Tuple[int, Optional[float]]]],
    sizes: List[int],
    impls: List[str],
    args: argparse.Namespace,
    world_size: int,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "world_size",
        "mode",
        "dtype",
        "implementation",
        "size_bytes",
        "size",
        "latency_ms",
        "available",
        "aggregation",
        "warmup",
        "iters_small",
        "iters_large",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in impls:
            values = dict(results.get(name, []))
            for size in sizes:
                latency = values.get(size)
                writer.writerow(
                    {
                        "world_size": world_size,
                        "mode": args.mode,
                        "dtype": args.dtype,
                        "implementation": name,
                        "size_bytes": size,
                        "size": human_size(size),
                        "latency_ms": latency,
                        "available": latency is not None,
                        "aggregation": "slowest_rank_average",
                        "warmup": args.warmup,
                        "iters_small": args.iters_small,
                        "iters_large": args.iters_large,
                    }
                )


def load_report_data(input_dir: str) -> dict:
    required = {
        "world_size",
        "mode",
        "dtype",
        "implementation",
        "size_bytes",
        "latency_ms",
        "available",
    }
    configurations: dict = {}
    for path in sorted(Path(input_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        config_key = f"{rows[0]['world_size']}-{rows[0]['mode']}-{rows[0]['dtype']}"
        config = configurations.setdefault(
            config_key,
            {
                "world_size": int(rows[0]["world_size"]),
                "mode": rows[0]["mode"],
                "dtype": rows[0]["dtype"],
                "sizes": set(),
                "series": defaultdict(dict),
                "sources": [],
                "metadata": {
                    key: rows[0].get(key)
                    for key in (
                        "aggregation",
                        "warmup",
                        "iters_small",
                        "iters_large",
                    )
                    if rows[0].get(key)
                },
            },
        )
        if path.name not in config["sources"]:
            config["sources"].append(path.name)
        for row in rows:
            row_key = f"{row['world_size']}-{row['mode']}-{row['dtype']}"
            if row_key != config_key:
                raise ValueError(f"{path} contains multiple configurations")
            size = int(row["size_bytes"])
            config["sizes"].add(size)
            available = row["available"].lower() == "true"
            implementation = row["implementation"]
            latency = float(row["latency_ms"]) if available else None
            existing = config["series"][implementation].get(size)
            if existing is not None and latency is not None:
                raise ValueError(
                    f"Duplicate measurement for {config_key}, {implementation}, {size}"
                )
            if size not in config["series"][implementation] or latency is not None:
                config["series"][implementation][size] = latency
    if not configurations:
        raise ValueError(f"No non-empty CSV files found in {input_dir}")
    for config in configurations.values():
        sizes = sorted(config["sizes"])
        config["sizes"] = sizes
        config["labels"] = [human_size(size) for size in sizes]
        config["series"] = {
            name: [values.get(size) for size in sizes]
            for name, values in config["series"].items()
        }
    return configurations


def markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    def cell(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return [
        "| " + " | ".join(cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(cell(value) for value in row) + " |" for row in rows],
    ]


def format_latency(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.4f}"


def best_measurement(
    config: dict, names: Tuple[str, ...], index: int
) -> Optional[Tuple[str, float]]:
    candidates = [
        (name, config["series"].get(name, [None] * len(config["sizes"]))[index])
        for name in names
    ]
    available = [(name, value) for name, value in candidates if value is not None]
    return min(available, key=lambda item: item[1]) if available else None


def write_markdown_report(input_dir: str, output_path: str) -> None:
    data = load_report_data(input_dir)
    configs = sorted(
        data.values(), key=lambda config: (config["world_size"], config["mode"])
    )
    lines = [
        "# ROCm unfused all-reduce benchmark",
        "",
        (
            "Per-call GPU-event latency in milliseconds. The primary metric is the "
            "slowest rank's mean. `—` means unavailable or ineligible."
        ),
        "",
        "## Summary",
        "",
    ]
    summary_rows = []
    all_names = LOSSLESS_IMPLEMENTATIONS + QUICK_REDUCE_IMPLEMENTATIONS
    for config in configs:
        lossless_wins: Dict[str, int] = defaultdict(int)
        quick_wins: Dict[str, int] = defaultdict(int)
        quick_beats_lossless = 0
        for index in range(len(config["sizes"])):
            lossless = best_measurement(config, LOSSLESS_IMPLEMENTATIONS, index)
            quick = best_measurement(config, QUICK_REDUCE_IMPLEMENTATIONS, index)
            if lossless:
                lossless_wins[lossless[0]] += 1
            if quick:
                quick_wins[quick[0]] += 1
            if lossless and quick and quick[1] < lossless[1]:
                quick_beats_lossless += 1
        top_lossless = (
            max(lossless_wins.items(), key=lambda item: item[1])
            if lossless_wins
            else ("—", 0)
        )
        top_quick = (
            max(quick_wins.items(), key=lambda item: item[1])
            if quick_wins
            else ("—", 0)
        )
        available = sum(
            any(value is not None for value in config["series"].get(name, []))
            for name in all_names
        )
        summary_rows.append(
            [
                f"TP{config['world_size']}",
                config["mode"],
                config["dtype"],
                f"{available}/{len(all_names)}",
                f"{top_lossless[0]} ({top_lossless[1]})",
                f"{top_quick[0]} ({top_quick[1]})",
                f"{quick_beats_lossless}/{len(config['sizes'])}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "TP",
                "Mode",
                "Dtype",
                "Available",
                "Most lossless wins",
                "Most QR wins",
                "QR faster sizes",
            ],
            summary_rows,
        )
    )
    lines.extend(
        [
            "",
            "> QR speedup is `best lossless latency / best QuickReduce latency`; "
            "values above 1.0x favor QuickReduce.",
            "",
        ]
    )

    for config in configs:
        metadata = config["metadata"]
        title = (
            f"TP{config['world_size']} · {config['mode'].capitalize()} · "
            f"{config['dtype'].upper()}"
        )
        lines.extend(
            [
                f"## {title}",
                "",
                (
                    f"Aggregation: `{metadata.get('aggregation', 'unknown')}` · "
                    f"warmups: {metadata.get('warmup', '?')} · timed iterations: "
                    f"{metadata.get('iters_small', '?')} small / "
                    f"{metadata.get('iters_large', '?')} large · "
                    f"CSV files: {len(config['sources'])}"
                ),
                "",
                "### Availability",
                "",
            ]
        )
        availability_rows = []
        for name in all_names:
            values = config["series"].get(name, [None] * len(config["sizes"]))
            measured = [
                config["labels"][index]
                for index, value in enumerate(values)
                if value is not None
            ]
            availability_rows.append(
                [
                    name,
                    "lossless" if name in LOSSLESS_IMPLEMENTATIONS else "QuickReduce",
                    f"{len(measured)}/{len(config['sizes'])}",
                    f"{measured[0]} – {measured[-1]}" if measured else "—",
                ]
            )
        lines.extend(
            markdown_table(
                ["Implementation", "Family", "Measured rows", "Measured range"],
                availability_rows,
            )
        )

        for heading, names in (
            ("Lossless latency (ms)", LOSSLESS_IMPLEMENTATIONS),
            ("QuickReduce latency (ms)", QUICK_REDUCE_IMPLEMENTATIONS),
        ):
            lines.extend(["", f"### {heading}", ""])
            rows = []
            for index, label in enumerate(config["labels"]):
                rows.append(
                    [
                        label,
                        *[
                            format_latency(
                                config["series"].get(
                                    name, [None] * len(config["sizes"])
                                )[index]
                            )
                            for name in names
                        ],
                    ]
                )
            lines.extend(markdown_table(["Size", *names], rows))

        lines.extend(["", "### Best implementation by size", ""])
        winner_rows = []
        for index, label in enumerate(config["labels"]):
            lossless = best_measurement(config, LOSSLESS_IMPLEMENTATIONS, index)
            quick = best_measurement(config, QUICK_REDUCE_IMPLEMENTATIONS, index)
            speedup = lossless[1] / quick[1] if lossless and quick else None
            if lossless and quick:
                overall = quick if quick[1] < lossless[1] else lossless
            else:
                overall = lossless or quick
            winner_rows.append(
                [
                    label,
                    lossless[0] if lossless else "—",
                    format_latency(lossless[1] if lossless else None),
                    quick[0] if quick else "—",
                    format_latency(quick[1] if quick else None),
                    f"{speedup:.3f}x" if speedup is not None else "—",
                    overall[0] if overall else "—",
                ]
            )
        lines.extend(
            markdown_table(
                [
                    "Size",
                    "Best lossless",
                    "Lossless ms",
                    "Best QR",
                    "QR ms",
                    "QR speedup",
                    "Overall",
                ],
                winner_rows,
            )
        )
        lines.append("")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(
        f"Wrote {output} with {len(configs)} configuration(s) from "
        f"{sum(len(v['sizes']) * len(v['series']) for v in configs)} rows"
    )


def parse_sweep_values(raw: str, allowed: Tuple[str, ...], label: str) -> List[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid or not values:
        raise ValueError(
            f"Invalid {label}: {invalid or values}; expected values from {allowed}"
        )
    return values


def build_worker_command(
    args: argparse.Namespace,
    tp: int,
    mode: str,
    impls: List[str],
    csv_path: Path,
    size_bytes: Optional[int] = None,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={tp}",
        str(Path(__file__).resolve()),
        "--backend",
        args.backend,
        "--mode",
        mode,
        "--dtype",
        args.dtype,
        "--impls",
        ",".join(impls),
        "--warmup",
        str(args.warmup),
        "--iters-small",
        str(args.iters_small),
        "--iters-large",
        str(args.iters_large),
        "--num-inputs",
        str(args.num_inputs),
        "--comm-buffer-mb",
        str(args.comm_buffer_mb),
        "--input-pool-mb",
        str(args.input_pool_mb),
        "--csv-out",
        str(csv_path),
    ]
    if size_bytes is None:
        command.extend(
            [
                "--min-size-kb",
                str(args.min_size_kb),
                "--max-size-mb",
                str(args.max_size_mb),
            ]
        )
    else:
        command.extend(["--size-bytes", str(size_bytes)])
    if args.skip_correctness:
        command.append("--skip-correctness")
    if args.verbose:
        command.append("--verbose")
    if args.sgl_v2_shot is not None:
        command.extend(["--sgl-v2-shot", str(args.sgl_v2_shot)])
    if args.sgl_v2_pull_blocks is not None:
        command.extend(["--sgl-v2-pull-blocks", str(args.sgl_v2_pull_blocks)])
    if args.sgl_v2_pull_threads is not None:
        command.extend(["--sgl-v2-pull-threads", str(args.sgl_v2_pull_threads)])
    return command


def build_sweep_jobs(
    args: argparse.Namespace,
    tp_values: List[int],
    modes: List[str],
    impls: List[str],
) -> List[Tuple[int, str, List[str], str, Optional[int]]]:
    jobs = []
    for tp in tp_values:
        for mode in modes:
            if mode != "graph" or "aiter" not in impls:
                jobs.append((tp, mode, impls, f"tp{tp}_{mode}_{args.dtype}", None))
                continue
            non_aiter = [name for name in impls if name != "aiter"]
            if non_aiter:
                jobs.append(
                    (
                        tp,
                        mode,
                        non_aiter,
                        f"tp{tp}_{mode}_{args.dtype}_non_aiter",
                        None,
                    )
                )
            for size_bytes in get_message_sizes(args.min_size_kb, args.max_size_mb):
                jobs.append(
                    (
                        tp,
                        mode,
                        ["aiter"],
                        f"tp{tp}_{mode}_{args.dtype}_aiter_{size_bytes}b",
                        size_bytes,
                    )
                )
    return jobs


def run_sweep(args: argparse.Namespace) -> None:
    tp_values = [
        int(value)
        for value in parse_sweep_values(args.sweep_tp, ("2", "4", "8"), "TP sizes")
    ]
    modes = parse_sweep_values(args.sweep_modes, ("eager", "graph"), "sweep modes")
    impls = parse_impls(args.impls)
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < max(tp_values):
        raise RuntimeError(
            f"Sweep requires {max(tp_values)} visible GPUs, found {visible_gpus}"
        )

    output_dir = Path(args.sweep_output_dir)
    csv_dir = output_dir / "csv"
    log_dir = output_dir / "logs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for path in csv_dir.glob("*.csv"):
        path.unlink()
    for path in log_dir.glob("*.log"):
        path.unlink()

    jobs = build_sweep_jobs(args, tp_values, modes, impls)
    total_workers = len(jobs)
    completed = 0

    def launch(
        tp: int,
        mode: str,
        names: List[str],
        stem: str,
        size_bytes: Optional[int] = None,
    ) -> None:
        nonlocal completed
        csv_path = csv_dir / f"{stem}.csv"
        log_path = log_dir / f"{stem}.log"
        command = build_worker_command(
            args, tp, mode, names, csv_path, size_bytes=size_bytes
        )
        print(
            f"[{completed + 1}/{total_workers}] {stem}: {','.join(names)}",
            flush=True,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.worker_timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(
                    f"{stem} timed out after {args.worker_timeout_seconds}s; "
                    f"see {log_path}"
                )
        if returncode != 0:
            raise RuntimeError(
                f"{stem} failed with exit code {returncode}; see {log_path}"
            )
        completed += 1

    for tp, mode, names, stem, size_bytes in jobs:
        launch(tp, mode, names, stem, size_bytes=size_bytes)

    report_path = (
        Path(args.report_out) if args.report_out else output_dir / "allreduce_report.md"
    )
    write_markdown_report(str(csv_dir), str(report_path))
    print(f"Sweep complete: {report_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = get_env_rank_world()
    impls = parse_impls(args.impls)
    if world_size not in (2, 4, 8):
        print(f"WARNING: world_size={world_size} is outside 2/4/8", file=sys.stderr)
    if not dist.is_initialized():
        dist.init_process_group(args.backend, init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    reference_pg = (
        dist.new_group(backend="nccl")
        if torch.cuda.is_available() and dist.is_nccl_available()
        else None
    )
    pg = dist.group.WORLD
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    sizes = (
        [args.size_bytes]
        if args.size_bytes is not None
        else get_message_sizes(args.min_size_kb, args.max_size_mb)
    )
    max_size = max(max(sizes), args.comm_buffer_mb * 1024 * 1024)
    results: Dict[str, List[Tuple[int, Optional[float]]]] = {}
    for name in impls:
        if args.mode == "graph" and name == "aiter":
            isolated = []
            for size in sizes:
                comm = construct_impl(name, pg, reference_pg, device, max_size, args)
                if comm is None:
                    isolated.append((size, None))
                    continue
                isolated.extend(
                    bench_graph(
                        name, comm, [size], device, args, dtype, pg, reference_pg
                    )
                )
                close_comm(comm, pg)
            results[name] = isolated
            continue
        comm = construct_impl(name, pg, reference_pg, device, max_size, args)
        if comm is None:
            results[name] = [(size, None) for size in sizes]
            continue
        bench = bench_graph if args.mode == "graph" else bench_eager
        results[name] = bench(name, comm, sizes, device, args, dtype, pg, reference_pg)
        close_comm(comm, pg)
    if rank == 0:
        print(f"\nResults ({args.mode}; slowest-rank average ms; None = unavailable)")
        for size in sizes:
            values = [f"{name}={dict(results[name]).get(size)}" for name in impls]
            print(f"{human_size(size):>9}: " + ", ".join(values))
        if args.csv_out:
            write_csv(args.csv_out, results, sizes, impls, args, world_size)
            print(f"Saved CSV to {args.csv_out}")
    dist.barrier()
    if reference_pg is not None:
        dist.destroy_process_group(reference_pg)
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.sweep_output_dir:
        run_sweep(args)
        return
    if args.report_input_dir:
        write_markdown_report(args.report_input_dir, args.report_out)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
