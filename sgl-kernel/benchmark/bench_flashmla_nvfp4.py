"""Profile SM90 sparse FlashMLA decode with FP8 or NVFP4 KV caches.

The default profile shape models GLM-5 NSA decode: every request owns a
disjoint 32K-token physical cache range and selects 2048 tokens from it.  Use
``--legacy-hot-33-pages`` to reproduce the old, shared-cache microbenchmark.

NCU example (the Python process launches the FlashMLA operation exactly once):

    ncu --replay-mode kernel --cache-control all --nvtx \
      --nvtx-include 'flashmla_profile_target/' \
      python bench_flashmla_nvfp4.py --mode ncu-single \
      --provider nvfp4 --batch-size 12
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import triton
from sgl_kernel.flash_mla import (
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_nvfp4,
    get_mla_metadata,
)

from sglang.srt.layers.attention.dsa.nvfp4_k_cache import (
    NVFP4_BYTES_PER_TOKEN,
    quantize_nvfp4_k_cache_into,
)
from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache

MIB = 1024 * 1024
DEFAULT_SEED = 20260708
STAGE_RECORD_NAMES = (
    "consumer_local",
    "consumer_remote",
    "producer_warp0",
    "producer_warp1",
    "producer_warp2",
    "producer_warp3",
)
STAGE_METRIC_NAMES = (
    "timed_tiles",
    "load",
    "dequant",
    "handoff",
    "consumer",
    "consumer_ready_wait",
    "producer_available_wait",
    "consumer_sync_wait",
)
RESULT_FIELDS = [
    "record_type",
    "mode",
    "provider",
    "batch_size",
    "round",
    "latency_us",
    "median_us",
    "p90_us",
    "cv",
    "num_rounds",
    "speedup",
    "fp8_us",
    "nvfp4_us",
    "cache_layout",
    "context_length",
    "pages_per_request",
    "total_pages",
    "physical_tokens_per_request",
    "topk",
    "num_heads",
    "head_dim",
    "head_dim_v",
    "num_index_sets",
    "index_sampling",
    "cache_seqlen",
    "cache_bytes_per_token",
    "flush_bytes",
    "output_shape",
    "cta",
    "timing_record",
    "metric",
    "timed_tiles",
    "total_cycles",
    "cycles_per_tile",
    "median_cycles_per_tile",
    "p90_cycles_per_tile",
    "num_samples",
    "aggregation",
    "out_max_abs_diff",
    "lse_max_abs_diff",
]


class ResultWriter:
    """Write records incrementally so partial profile runs remain usable."""

    def __init__(
        self,
        jsonl_path: str | None,
        csv_path: str | None,
        print_json: bool,
    ) -> None:
        self.print_json = print_json
        self._jsonl = self._open(jsonl_path) if jsonl_path else None
        self._csv_file = self._open(csv_path) if csv_path else None
        self._csv = (
            csv.DictWriter(self._csv_file, fieldnames=RESULT_FIELDS)
            if self._csv_file
            else None
        )
        if self._csv:
            self._csv.writeheader()
            self._csv_file.flush()

    @staticmethod
    def _open(path: str):
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        return output.open("w", encoding="utf-8", newline="")

    def emit(self, record: dict[str, Any]) -> None:
        if self._jsonl:
            self._jsonl.write(json.dumps(record, sort_keys=True) + "\n")
            self._jsonl.flush()
        if self._csv and record.get("record_type") != "config":
            row = {field: record.get(field, "") for field in RESULT_FIELDS}
            if isinstance(row["output_shape"], (tuple, list)):
                row["output_shape"] = json.dumps(row["output_shape"])
            self._csv.writerow(row)
            self._csv_file.flush()
        if self.print_json:
            print(json.dumps(record, sort_keys=True), flush=True)

    def close(self) -> None:
        if self._jsonl:
            self._jsonl.close()
        if self._csv_file:
            self._csv_file.close()


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_manifest(repo_root: Path) -> dict[str, Any]:
    commit = _command_output(["git", "rev-parse", "HEAD"], repo_root)
    status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], repo_root
    )
    diff: bytes | None = None
    try:
        result = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            diff = result.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    digest = None
    if diff is not None:
        hasher = hashlib.sha256()
        hasher.update(diff)
        hasher.update((status or "").encode("utf-8"))
        # ``git diff HEAD`` excludes untracked files.  Hash their paths and
        # contents as well; the benchmark itself may still be untracked while
        # iterating on a profile build.
        try:
            result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                for raw_path in result.stdout.split(b"\0"):
                    if not raw_path:
                        continue
                    hasher.update(b"\0untracked\0" + raw_path + b"\0")
                    path = repo_root / os.fsdecode(raw_path)
                    if path.is_symlink():
                        hasher.update(os.fsencode(os.readlink(path)))
                    elif path.is_file():
                        with path.open("rb") as file:
                            chunk = file.read(1024 * 1024)
                            while chunk:
                                hasher.update(chunk)
                                chunk = file.read(1024 * 1024)
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            pass
        digest = hasher.hexdigest()
    return {
        "commit": commit,
        "dirty": bool(status),
        "worktree_diff_sha256": digest,
        "git_status": status.splitlines() if status else [],
    }


def _flashmla_tag(repo_root: Path) -> str | None:
    cmake_file = repo_root / "sgl-kernel" / "cmake" / "flashmla.cmake"
    try:
        contents = cmake_file.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(
        r"FetchContent_Declare\(\s*repo-flashmla.*?GIT_TAG\s+([^\s\)]+)",
        contents,
        flags=re.DOTALL,
    )
    return match.group(1) if match else None


def _l2_bytes(properties: Any) -> int | None:
    for name in ("l2_cache_size", "L2_cache_size"):
        value = getattr(properties, name, None)
        if value:
            return int(value)
    return None


def _environment_manifest(
    args: argparse.Namespace,
    properties: Any,
    flush_bytes: int,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    nvidia_smi = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,clocks.current.sm,"
            "clocks.current.memory,clocks.max.sm,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "record_type": "config",
        "command": shlex.join([sys.executable, *sys.argv]),
        "arguments": vars(args),
        "git": _git_manifest(repo_root),
        "flashmla_git_tag": _flashmla_tag(repo_root),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": getattr(triton, "__version__", None),
        "ncu_version": _command_output(["ncu", "--version"]),
        "gpu": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "clock_rate_khz": getattr(properties, "clock_rate", None),
            "l2_bytes": _l2_bytes(properties),
            "nvidia_smi": nvidia_smi.splitlines() if nvidia_smi else [],
        },
        "flush_bytes": flush_bytes,
        "build_environment": {
            name: os.environ.get(name)
            for name in (
                "CMAKE_ARGS",
                "CUDA_HOME",
                "NVCC_FLAGS",
                "TORCH_CUDA_ARCH_LIST",
            )
        },
    }


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _adaptive_rounds(
    measure: Callable[[], float],
    min_rounds: int,
    max_rounds: int,
    cv_threshold: float,
) -> list[float]:
    values: list[float] = []
    while len(values) < min_rounds:
        values.append(measure())
    while len(values) < max_rounds and _coefficient_of_variation(values) > cv_threshold:
        values.append(measure())
    return values


def _make_index_sets(
    batch_size: int,
    topk: int,
    num_index_sets: int,
    cache_layout: str,
    context_length: int,
    physical_tokens_per_request: int,
    shared_capacity: int,
    index_sampling: str,
    device: torch.device,
    seed: int,
) -> list[torch.Tensor]:
    if cache_layout == "disjoint":
        offsets = (
            torch.arange(batch_size, dtype=torch.int32, device=device)
            .view(batch_size, 1, 1)
            .mul_(physical_tokens_per_request)
        )
        index_limit = context_length
    else:
        offsets = None
        index_limit = shared_capacity

    index_sets = []
    for set_idx in range(num_index_sets):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 104729 * set_idx + batch_size)
        if index_sampling == "unique":
            # NSA top-k contains unique token positions.  Random scores avoid
            # materializing one 32K randperm at a time for every request while
            # preserving independent, without-replacement samples per row.
            scores = torch.rand(
                (batch_size, index_limit), device=device, generator=generator
            )
            indices = (
                torch.topk(scores, topk, dim=1, largest=True, sorted=False)
                .indices.to(torch.int32)
                .unsqueeze(1)
            )
        else:
            # Retain the old benchmark's sampling behavior for exact hot-cache
            # reproduction.
            indices = torch.randint(
                0,
                index_limit,
                (batch_size, 1, topk),
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
        if offsets is not None:
            indices.add_(offsets)
        index_sets.append(indices)
    return index_sets


def _make_cache(
    providers: list[str],
    total_pages: int,
    page_size: int,
    head_dim: int,
    head_dim_v: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    kv = torch.randn(
        (total_pages, page_size, 1, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    kv.div_(10)
    caches: dict[str, torch.Tensor] = {}
    if "fp8" in providers:
        caches["fp8"] = quantize_k_cache(kv)
    global_scale = torch.ones(1, dtype=torch.float32, device=device)
    if "nvfp4" in providers:
        nvfp4_cache = torch.zeros(
            (total_pages, page_size, 1, NVFP4_BYTES_PER_TOKEN),
            dtype=torch.uint8,
            device=device,
        )
        capacity = total_pages * page_size
        loc = torch.arange(capacity, dtype=torch.int32, device=device)
        quantize_nvfp4_k_cache_into(
            kv[..., :head_dim_v].reshape(capacity, head_dim_v),
            kv[..., head_dim_v:].reshape(capacity, head_dim - head_dim_v),
            nvfp4_cache,
            loc,
            global_scale,
        )
        caches["nvfp4"] = nvfp4_cache
    del kv
    return caches, global_scale


def _record_base(
    args: argparse.Namespace,
    provider: str,
    batch_size: int,
    cache_layout: str,
    pages_per_request: int,
    total_pages: int,
    physical_tokens_per_request: int,
    cache_bytes_per_token: int,
    flush_bytes: int,
    cache_seqlen: int,
) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "provider": provider,
        "batch_size": batch_size,
        "cache_layout": cache_layout,
        "context_length": args.context_length,
        "pages_per_request": pages_per_request,
        "total_pages": total_pages,
        "physical_tokens_per_request": physical_tokens_per_request,
        "topk": args.topk,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "head_dim_v": args.head_dim_v,
        "num_index_sets": args.num_index_sets,
        "index_sampling": args.index_sampling,
        "cache_seqlen": cache_seqlen,
        "cache_bytes_per_token": cache_bytes_per_token,
        "flush_bytes": flush_bytes,
    }


def _nearest_rank_p90(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.9 * len(values)) - 1]


def _emit_stage_timing(
    writer: ResultWriter,
    base: dict[str, Any],
    timing: torch.Tensor,
    output_shape: list[int],
) -> None:
    timing_cpu = timing.detach().cpu()
    expected_tail = (len(STAGE_RECORD_NAMES), len(STAGE_METRIC_NAMES))
    if timing_cpu.dtype != torch.int64 or tuple(timing_cpu.shape[1:]) != expected_tail:
        raise AssertionError(
            "stage timing must be int64 [num_ctas, 6, 8], got "
            f"dtype={timing_cpu.dtype} shape={tuple(timing_cpu.shape)}"
        )

    values_by_record_metric: dict[tuple[str, str], list[float]] = {}
    raw_values: dict[tuple[int, int], dict[str, float]] = {}
    for cta_idx in range(timing_cpu.shape[0]):
        for record_idx, record_name in enumerate(STAGE_RECORD_NAMES):
            timed_tiles = int(timing_cpu[cta_idx, record_idx, 0].item())
            if timed_tiles <= 0:
                continue
            per_record = {}
            for metric_idx, metric_name in enumerate(STAGE_METRIC_NAMES[1:], start=1):
                total_cycles = int(timing_cpu[cta_idx, record_idx, metric_idx].item())
                cycles_per_tile = total_cycles / timed_tiles
                per_record[metric_name] = cycles_per_tile
                values_by_record_metric.setdefault(
                    (record_name, metric_name), []
                ).append(cycles_per_tile)
                writer.emit(
                    {
                        **base,
                        "record_type": "stage_cycle",
                        "cta": cta_idx,
                        "timing_record": record_name,
                        "metric": metric_name,
                        "timed_tiles": timed_tiles,
                        "total_cycles": total_cycles,
                        "cycles_per_tile": cycles_per_tile,
                        "output_shape": output_shape,
                    }
                )
            raw_values[(cta_idx, record_idx)] = per_record

    for (record_name, metric_name), values in values_by_record_metric.items():
        writer.emit(
            {
                **base,
                "record_type": "stage_summary",
                "timing_record": record_name,
                "metric": metric_name,
                "median_cycles_per_tile": statistics.median(values),
                "p90_cycles_per_tile": _nearest_rank_p90(values),
                "num_samples": len(values),
                "output_shape": output_shape,
            }
        )

    # These group summaries are the directly comparable critical-path service
    # and wait buckets.  The records inside each group execute in parallel, so
    # first take the slowest record per CTA and only then aggregate across
    # CTAs.  Pooling all records would understate an imbalanced critical path.
    groups = {
        "consumer_all": (
            range(0, 2),
            ("consumer", "consumer_ready_wait", "consumer_sync_wait"),
        ),
        "producer_all": (
            range(2, 6),
            ("load", "dequant", "handoff", "producer_available_wait"),
        ),
    }
    for group_name, (record_indices, metrics) in groups.items():
        for metric_name in metrics:
            values = []
            for cta_idx in range(timing_cpu.shape[0]):
                parallel_values = [
                    raw_values[(cta_idx, record_idx)][metric_name]
                    for record_idx in record_indices
                    if (cta_idx, record_idx) in raw_values
                    and metric_name in raw_values[(cta_idx, record_idx)]
                ]
                if parallel_values:
                    values.append(max(parallel_values))
            if not values:
                continue
            writer.emit(
                {
                    **base,
                    "record_type": "stage_group_summary",
                    "timing_record": group_name,
                    "metric": metric_name,
                    "median_cycles_per_tile": statistics.median(values),
                    "p90_cycles_per_tile": _nearest_rank_p90(values),
                    "num_samples": len(values),
                    "aggregation": "per_cta_max_parallel_records",
                    "output_shape": output_shape,
                }
            )


def _human_summary(record: dict[str, Any]) -> None:
    record_type = record["record_type"]
    if record_type == "timing_summary":
        print(
            f"mode={record['mode']} provider={record['provider']} "
            f"B={record['batch_size']:2d} H={record['num_heads']} "
            f"topk={record['topk']} median_us={record['median_us']:.2f} "
            f"p90_us={record['p90_us']:.2f} cv={record['cv']:.4f} "
            f"rounds={record['num_rounds']}",
            flush=True,
        )
    elif record_type == "comparison":
        print(
            f"mode={record['mode']} B={record['batch_size']:2d} "
            f"fp8_us={record['fp8_us']:.2f} "
            f"nvfp4_us={record['nvfp4_us']:.2f} "
            f"speedup={record['speedup']:.3f}x",
            flush=True,
        )
    elif record_type == "ncu_single":
        print(
            f"NCU target complete: provider={record['provider']} "
            f"B={record['batch_size']} output_shape={record['output_shape']}",
            flush=True,
        )


def benchmark(args: argparse.Namespace, writer: ResultWriter) -> None:
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)

    l2_bytes = _l2_bytes(properties) or 0
    requested_flush_bytes = args.flush_mib * MIB
    automatic_flush_bytes = max(4 * l2_bytes, 256 * MIB)
    flush_bytes = (
        max(requested_flush_bytes, automatic_flush_bytes)
        if args.mode in ("cold", "ncu-single", "stage-timing")
        else 0
    )
    flush_buffer = (
        torch.zeros(flush_bytes, dtype=torch.uint8, device=device)
        if flush_bytes
        else None
    )
    writer.emit(_environment_manifest(args, properties, flush_bytes))

    cache_layout = args.cache_layout
    if cache_layout == "auto":
        cache_layout = "shared" if args.num_pages is not None else "disjoint"

    providers = ["fp8", "nvfp4"] if args.provider == "both" else [args.provider]
    softmax_scale = 1.0 / math.sqrt(args.head_dim)
    page_size = args.page_size

    for batch_size in args.batch_sizes:
        if cache_layout == "disjoint":
            pages_per_request = args.num_pages or math.ceil(
                args.context_length / page_size
            )
            physical_tokens_per_request = pages_per_request * page_size
            if args.context_length > physical_tokens_per_request:
                raise ValueError(
                    "context length exceeds the --num-pages allocation per request"
                )
            total_pages = pages_per_request * batch_size
            shared_capacity = 0
            cache_seqlen = args.context_length
        else:
            total_pages = args.num_pages or math.ceil(args.context_length / page_size)
            pages_per_request = total_pages
            physical_tokens_per_request = total_pages * page_size
            shared_capacity = physical_tokens_per_request
            cache_seqlen = shared_capacity

        selection_capacity = (
            args.context_length if cache_layout == "disjoint" else shared_capacity
        )
        if args.topk > selection_capacity:
            raise ValueError(
                f"top-k {args.topk} exceeds the request selection range "
                f"{selection_capacity}"
            )

        caches, global_scale = _make_cache(
            providers,
            total_pages,
            page_size,
            args.head_dim,
            args.head_dim_v,
            device,
        )
        q = torch.randn(
            (batch_size, 1, args.num_heads, args.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        q.div_(10)
        index_sets = _make_index_sets(
            batch_size,
            args.topk,
            args.num_index_sets,
            cache_layout,
            args.context_length,
            physical_tokens_per_request,
            shared_capacity,
            args.index_sampling,
            device,
            args.seed,
        )
        seqlens = torch.full(
            (batch_size,), cache_seqlen, dtype=torch.int32, device=device
        )
        metadata, num_splits = get_mla_metadata(
            seqlens,
            args.num_heads,
            1,
            args.num_heads,
            is_fp8_kvcache=True,
            topk=args.topk,
        )
        block_table = torch.empty((batch_size, 0), dtype=torch.int32, device=device)
        steps = {provider: 0 for provider in providers}

        def run(provider: str):
            indices = index_sets[steps[provider] % args.num_index_sets]
            steps[provider] += 1
            if provider == "fp8":
                return flash_mla_with_kvcache(
                    q=q,
                    k_cache=caches[provider],
                    block_table=block_table,
                    cache_seqlens=seqlens,
                    head_dim_v=args.head_dim_v,
                    tile_scheduler_metadata=metadata,
                    num_splits=num_splits,
                    softmax_scale=softmax_scale,
                    is_fp8_kvcache=True,
                    indices=indices,
                )
            return flash_mla_with_kvcache_nvfp4(
                q=q,
                k_cache=caches[provider],
                kv_global_scale=global_scale,
                cache_seqlens=seqlens,
                tile_scheduler_metadata=metadata,
                num_splits=num_splits,
                indices=indices,
                head_dim_v=args.head_dim_v,
                softmax_scale=softmax_scale,
            )

        if args.mode == "stage-timing":
            provider = "nvfp4"
            # Compare the diagnostic build against the unchanged production
            # entry point using exactly the same cache, metadata, and indices.
            reference_out, reference_lse = run(provider)
            torch.cuda.synchronize()
            if flush_buffer is not None:
                flush_buffer.add_(1)
            torch.cuda.synchronize()
            try:
                stage_op = torch.ops.sgl_kernel._fwd_kvcache_mla_nvfp4_stage_timing
                stage_default = stage_op.default
            except (AttributeError, RuntimeError) as exc:
                raise RuntimeError(
                    "stage-timing mode requires a profile build configured with "
                    "-DSGLANG_FLASHMLA_NVFP4_STAGE_TIMING=ON"
                ) from exc
            stage_out, stage_lse, timing = stage_default(
                q,
                caches[provider],
                global_scale,
                args.head_dim_v,
                seqlens,
                softmax_scale,
                metadata,
                num_splits,
                index_sets[0],
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(
                stage_out, reference_out, atol=8e-4, rtol=2.01 / 128
            )
            torch.testing.assert_close(
                stage_lse, reference_lse, atol=2e-4, rtol=8.01 / 65536
            )
            out_max_abs_diff = float(
                (stage_out.float() - reference_out.float()).abs().max().item()
            )
            lse_max_abs_diff = float(
                (stage_lse.float() - reference_lse.float()).abs().max().item()
            )
            base = _record_base(
                args,
                provider,
                batch_size,
                cache_layout,
                pages_per_request,
                total_pages,
                physical_tokens_per_request,
                caches[provider].shape[-1],
                flush_bytes,
                cache_seqlen,
            )
            output_shape = list(stage_out.shape)
            correctness = {
                **base,
                "record_type": "stage_correctness",
                "out_max_abs_diff": out_max_abs_diff,
                "lse_max_abs_diff": lse_max_abs_diff,
                "output_shape": output_shape,
            }
            writer.emit(correctness)
            _emit_stage_timing(writer, base, timing, output_shape)
            if not args.print_json:
                print(
                    f"stage timing complete: B={batch_size} "
                    f"ctas={timing.shape[0]} out_max_abs_diff={out_max_abs_diff:.6g} "
                    f"lse_max_abs_diff={lse_max_abs_diff:.6g}",
                    flush=True,
                )
            return

        if args.mode == "ncu-single":
            provider = providers[0]
            if flush_buffer is not None:
                flush_buffer.add_(1)
            torch.cuda.synchronize()
            if not args.print_json:
                print(
                    f"NCU_TARGET_BEGIN range={args.nvtx_range} "
                    f"provider={provider} B={batch_size}",
                    flush=True,
                )
            torch.cuda.nvtx.range_push(args.nvtx_range)
            output = run(provider)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            record = {
                **_record_base(
                    args,
                    provider,
                    batch_size,
                    cache_layout,
                    pages_per_request,
                    total_pages,
                    physical_tokens_per_request,
                    caches[provider].shape[-1],
                    flush_bytes,
                    cache_seqlen,
                ),
                "record_type": "ncu_single",
                "output_shape": list(output[0].shape),
            }
            writer.emit(record)
            if not args.print_json:
                _human_summary(record)
            return

        # A non-timed launch validates dispatch and ensures lazy CUDA state is
        # initialized before either timing mode.
        output_shapes = {}
        for provider in providers:
            output = run(provider)
            output_shapes[provider] = list(output[0].shape)
        if len(set(map(tuple, output_shapes.values()))) != 1:
            raise AssertionError(f"provider output shapes differ: {output_shapes}")
        torch.cuda.synchronize()

        summaries: dict[str, dict[str, Any]] = {}
        for provider in providers:
            if args.mode == "warm":

                def measure() -> float:
                    return (
                        triton.testing.do_bench(
                            lambda: run(provider),
                            warmup=args.warmup,
                            rep=args.rep,
                        )
                        * 1000
                    )

            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                def measure() -> float:
                    samples = []
                    for _ in range(args.cold_rep):
                        assert flush_buffer is not None
                        flush_buffer.add_(1)
                        start.record()
                        run(provider)
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000)
                    return statistics.median(samples)

            round_values = _adaptive_rounds(
                measure,
                args.rounds,
                args.max_rounds,
                args.cv_threshold,
            )
            base = _record_base(
                args,
                provider,
                batch_size,
                cache_layout,
                pages_per_request,
                total_pages,
                physical_tokens_per_request,
                caches[provider].shape[-1],
                flush_bytes,
                cache_seqlen,
            )
            for round_idx, latency_us in enumerate(round_values, start=1):
                writer.emit(
                    {
                        **base,
                        "record_type": "timing_round",
                        "round": round_idx,
                        "latency_us": latency_us,
                        "output_shape": output_shapes[provider],
                    }
                )
            summary = {
                **base,
                "record_type": "timing_summary",
                "median_us": statistics.median(round_values),
                "p90_us": sorted(round_values)[math.ceil(0.9 * len(round_values)) - 1],
                "cv": _coefficient_of_variation(round_values),
                "num_rounds": len(round_values),
                "output_shape": output_shapes[provider],
            }
            writer.emit(summary)
            summaries[provider] = summary
            if not args.print_json:
                _human_summary(summary)

            if args.mode == "warm" and args.cudagraph:
                graph_us = (
                    triton.testing.do_bench_cudagraph(lambda: run(provider)) * 1000
                )
                graph_record = {
                    **base,
                    "record_type": "cudagraph",
                    "latency_us": graph_us,
                    "output_shape": output_shapes[provider],
                }
                writer.emit(graph_record)
                if not args.print_json:
                    print(
                        f"mode=warm-cudagraph provider={provider} "
                        f"B={batch_size:2d} latency_us={graph_us:.2f}",
                        flush=True,
                    )

        if set(summaries) == {"fp8", "nvfp4"}:
            comparison = {
                **_record_base(
                    args,
                    "both",
                    batch_size,
                    cache_layout,
                    pages_per_request,
                    total_pages,
                    physical_tokens_per_request,
                    0,
                    flush_bytes,
                    cache_seqlen,
                ),
                "record_type": "comparison",
                "fp8_us": summaries["fp8"]["median_us"],
                "nvfp4_us": summaries["nvfp4"]["median_us"],
                "speedup": summaries["fp8"]["median_us"]
                / summaries["nvfp4"]["median_us"],
            }
            writer.emit(comparison)
            if not args.print_json:
                _human_summary(comparison)

        del caches, q, index_sets, seqlens, metadata, block_table
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("warm", "cold", "ncu-single", "stage-timing"),
        default="warm",
        help=(
            "timing/profile mode; ncu-single invokes FlashMLA exactly once; "
            "stage-timing requires the profile-only diagnostic build"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("fp8", "nvfp4", "both"),
        help="profile one provider; 'both' is retained for the legacy comparison",
    )
    batches = parser.add_mutually_exclusive_group()
    batches.add_argument("--batch-size", type=int, help="single batch size")
    batches.add_argument("--batch-sizes", type=int, nargs="+", help="batch matrix")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--num-pages",
        type=int,
        help=(
            "pages/request for disjoint layout, or total pages for shared layout; "
            "when layout=auto, specifying this selects legacy shared semantics"
        ),
    )
    parser.add_argument(
        "--cache-layout",
        choices=("auto", "disjoint", "shared"),
        default="auto",
    )
    parser.add_argument(
        "--legacy-hot-33-pages",
        action="store_true",
        help="shared 33-page cache, index_sets=4, batches=1/16/32 unless overridden",
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=576)
    parser.add_argument("--head-dim-v", type=int, default=512)
    parser.add_argument("--num-index-sets", type=int)
    parser.add_argument(
        "--index-sampling",
        choices=("unique", "with-replacement"),
        help="unique matches NSA top-k; replacement reproduces the old benchmark",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--cv-threshold", type=float, default=0.02)
    parser.add_argument("--cold-rep", type=int, default=50)
    parser.add_argument(
        "--flush-mib",
        "--cold-cache-mib",
        dest="flush_mib",
        type=int,
        default=0,
        help="minimum flush-buffer MiB; cold/NCU always use max(4x L2, 256MiB)",
    )
    parser.add_argument("--cudagraph", action="store_true")
    parser.add_argument("--nvtx-range", default="flashmla_profile_target")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    parser.add_argument(
        "--print-json", action="store_true", help="emit JSON records to stdout"
    )
    args = parser.parse_args()

    if args.legacy_hot_33_pages:
        if args.cache_layout not in ("auto", "shared"):
            parser.error("--legacy-hot-33-pages is incompatible with disjoint layout")
        if args.num_pages not in (None, 33):
            parser.error("--legacy-hot-33-pages is incompatible with --num-pages != 33")
        args.cache_layout = "shared"
        args.num_pages = 33
        args.context_length = 33 * args.page_size
        args.provider = args.provider or "both"
        args.num_index_sets = args.num_index_sets or 4
        args.index_sampling = args.index_sampling or "with-replacement"
        if args.batch_size is None and args.batch_sizes is None:
            args.batch_sizes = [1, 16, 32]
    else:
        if args.provider is None:
            parser.error("--provider is required outside --legacy-hot-33-pages")
        args.num_index_sets = args.num_index_sets or 8
        args.index_sampling = args.index_sampling or "unique"

    if args.batch_size is not None:
        args.batch_sizes = [args.batch_size]
    elif args.batch_sizes is None:
        args.batch_sizes = [1, 12, 17, 20] if args.provider == "nvfp4" else [1, 12, 17]

    positive_values = {
        "batch size": min(args.batch_sizes),
        "context length": args.context_length,
        "page size": args.page_size,
        "top-k": args.topk,
        "number of heads": args.num_heads,
        "head dimension": args.head_dim,
        "V head dimension": args.head_dim_v,
        "number of index sets": args.num_index_sets,
        "rounds": args.rounds,
        "max rounds": args.max_rounds,
        "cold repetitions": args.cold_rep,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.num_pages is not None and args.num_pages <= 0:
        parser.error("--num-pages must be positive")
    if args.page_size != 64:
        parser.error("FlashMLA sparse KV cache profiling requires --page-size 64")
    if args.head_dim_v >= args.head_dim:
        parser.error("--head-dim-v must be smaller than --head-dim")
    if args.max_rounds < args.rounds:
        parser.error("--max-rounds must be at least --rounds")
    if args.cv_threshold < 0:
        parser.error("--cv-threshold must be non-negative")
    if args.flush_mib < 0:
        parser.error("--flush-mib must be non-negative")
    if args.mode == "ncu-single":
        if args.provider == "both":
            parser.error("ncu-single requires exactly one provider")
        if len(args.batch_sizes) != 1:
            parser.error("ncu-single requires --batch-size")
        if args.cudagraph:
            parser.error("ncu-single is incompatible with --cudagraph")
    if args.mode == "stage-timing":
        if args.provider != "nvfp4":
            parser.error("stage-timing requires --provider nvfp4")
        if len(args.batch_sizes) != 1:
            parser.error("stage-timing requires --batch-size")
        if args.cudagraph:
            parser.error("stage-timing is incompatible with --cudagraph")
    if args.cudagraph and args.mode != "warm":
        parser.error("--cudagraph is only valid in warm mode")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    result_writer = ResultWriter(
        cli_args.output_jsonl,
        cli_args.output_csv,
        cli_args.print_json,
    )
    try:
        benchmark(cli_args, result_writer)
    finally:
        result_writer.close()
