"""Compare FP8 and NVFP4 DSA KV-cache costs by serving phase.

This benchmark intentionally stops below the model/server layer.  It reproduces
the GLM-5.x TP8 SM90 attention paths closely enough to answer two questions:

* Does a ragged prefill with an existing prefix spend more time materializing
  the full KV cache when it is stored as NVFP4?
* Is fused sparse decode still approximately equal between FP8 and NVFP4?

The prefill chain is timed as one CUDA dependency chain (write current KV,
optionally gather/dequantize the full prefix+extend cache, then run the common
BF16 sparse-prefill kernel).  Per-stage timings are also reported for
attribution. Decode uses production head geometry: FP8 pads GLM TP8 H8 to H64,
while NVFP4 keeps the legacy H8 path for one query token and pads larger
launches to the H64 fast path. Both reuse scheduler metadata generated for H64.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from sgl_kernel.flash_mla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_nvfp4,
    get_mla_metadata,
)
from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton
from sglang.srt.layers.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.srt.layers.attention.dsa.nvfp4_k_cache import (
    NVFP4_BYTES_PER_TOKEN,
    dequantize_nvfp4_k_cache_paged,
    quantize_nvfp4_k_cache_into,
)
from sglang.srt.layers.attention.dsa.quant_k_cache import (
    quantize_k_cache_separate,
)
from sglang.srt.runtime_context import get_parallel


PAGE_SIZE = 64
TOPK = 2048
Q_HEADS = 8
FLASHMLA_HEADS = 64
HEAD_DIM = 576
VALUE_DIM = 512
FP8_BYTES_PER_TOKEN = 656
MIB = 1024 * 1024


@dataclass(frozen=True)
class PrefillCase:
    prefix: int
    extend: int

    @property
    def total(self) -> int:
        return self.prefix + self.extend

    @property
    def name(self) -> str:
        return f"p{self.prefix}_e{self.extend}"


@dataclass
class CachePair:
    fp8_raw: torch.Tensor
    nvfp4_raw: torch.Tensor
    global_scale: torch.Tensor

    @property
    def fp8(self) -> torch.Tensor:
        return self.fp8_raw.view(torch.float8_e4m3fn)


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _repo_manifest() -> dict:
    root = Path(__file__).resolve().parents[2]
    status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], root
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    ).stdout
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"], root),
        "dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_status": status.splitlines() if status else [],
    }


def _file_sha256(path: str | os.PathLike | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _finite(output) -> None:
    tensors: Iterable[torch.Tensor]
    if isinstance(output, torch.Tensor):
        tensors = (output,)
    elif output is None:
        return
    else:
        tensors = (item for item in output if isinstance(item, torch.Tensor))
    for tensor in tensors:
        if not torch.isfinite(tensor).all().item():
            raise AssertionError("timed operation produced NaN or Inf")


class Recorder:
    def __init__(self, output_dir: Path, args: argparse.Namespace) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self._jsonl_file = (output_dir / "timing.jsonl").open("w", encoding="utf-8")
        self._rows: list[dict] = []
        serialized_args = dict(vars(args))
        serialized_args["prefill_cases"] = [
            {"prefix": case.prefix, "extend": case.extend}
            for case in args.prefill_cases
        ]
        extension_path = getattr(
            sys.modules.get("sgl_kernel.flashmla_ops"), "__file__", None
        )
        self.emit(
            {
                "record_type": "manifest",
                "command": shlex.join([sys.executable, *sys.argv]),
                "args": serialized_args,
                "git": _repo_manifest(),
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "gpu_capability": list(torch.cuda.get_device_capability()),
                "benchmark_sha256": _file_sha256(__file__),
                "sgl_kernel_module": extension_path,
                "sgl_kernel_module_sha256": _file_sha256(extension_path),
            }
        )

    def emit(self, row: dict) -> None:
        self._jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")
        self._jsonl_file.flush()
        if row.get("record_type") == "timing":
            self._rows.append(row)
            delta = row.get("nvfp4_vs_fp8_pct")
            delta_text = "" if delta is None else f" delta={delta:+.2f}%"
            print(
                f"{row['phase']:7s} {row['case']:18s} {row['stage']:24s} "
                f"{row['provider']:6s} {row['cache_mode']:4s} "
                f"median={row['median_us']:10.2f} us cv={row['cv']:.3%}"
                f"{delta_text}",
                flush=True,
            )

    def close(self) -> None:
        self._jsonl_file.close()
        fields = sorted({key for row in self._rows for key in row})
        with (self.output_dir / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._rows)


def _physical_rows(num_tokens: int, seed: int, device: torch.device) -> tuple[torch.Tensor, int]:
    num_pages = math.ceil(num_tokens / PAGE_SIZE)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    page_order = torch.randperm(num_pages, generator=generator, dtype=torch.int64)
    logical = torch.arange(num_tokens, dtype=torch.int64)
    physical = page_order[logical // PAGE_SIZE] * PAGE_SIZE + logical % PAGE_SIZE
    return physical.to(device=device, dtype=torch.int32), num_pages * PAGE_SIZE


def _prefill_indices(
    seq_q: int, seq_kv: int, topk: int, device: torch.device
) -> torch.Tensor:
    if topk > seq_kv:
        raise ValueError(f"top-k {topk} exceeds KV length {seq_kv}")
    # Each query gets a distinct cyclic shift of a without-replacement,
    # approximately uniform selection.  This avoids repeatedly touching one
    # tiny cache region while keeping setup deterministic and cheap.
    base = torch.div(
        torch.arange(topk, device=device, dtype=torch.int64) * seq_kv,
        topk,
        rounding_mode="floor",
    )
    shift = (
        torch.arange(seq_q, device=device, dtype=torch.int64).unsqueeze(1) * 257
    ) % seq_kv
    return ((base.unsqueeze(0) + shift) % seq_kv).to(torch.int32).unsqueeze(1)


def _decode_indices(
    batch: int,
    context: int,
    capacity_per_request: int,
    topk: int,
    set_idx: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260721 + 104729 * set_idx + batch)
    scores = torch.rand((batch, context), generator=generator, device=device)
    local = torch.topk(scores, topk, dim=1, sorted=False).indices.to(torch.int32)
    offsets = (
        torch.arange(batch, device=device, dtype=torch.int32) * capacity_per_request
    ).unsqueeze(1)
    return (local + offsets).unsqueeze(1)


def _allocate_caches(capacity: int, device: torch.device) -> CachePair:
    return CachePair(
        fp8_raw=torch.empty(
            (capacity, 1, FP8_BYTES_PER_TOKEN), dtype=torch.uint8, device=device
        ),
        nvfp4_raw=torch.empty(
            (capacity, 1, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8, device=device
        ),
        global_scale=torch.tensor([1.0], dtype=torch.float32, device=device),
    )


def _write_fp8(
    cache: torch.Tensor, loc: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> None:
    nope, rope = quantize_k_cache_separate(k_nope, k_rope)
    set_mla_kv_buffer_triton(cache, loc, nope, rope)


def _write_nvfp4(
    cache: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    quantize_nvfp4_k_cache_into(k_nope, k_rope, cache, loc, global_scale)


def _measure(
    function: Callable[[], object],
    warmup: int,
    samples: int,
    cache_mode: str,
    flush_buffer: torch.Tensor,
) -> tuple[list[float], object]:
    output = None
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    _finite(output)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        if cache_mode == "cold":
            flush_buffer.add_(1)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    _finite(output)
    return values, output


def _capture_cuda_graph(
    function: Callable[[], object], warmup: int
) -> tuple[torch.cuda.CUDAGraph, object]:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        output = None
        for _ in range(warmup):
            output = function()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    torch.cuda.synchronize()
    _finite(output)
    return graph, output


def _emit_pair(
    recorder: Recorder,
    phase: str,
    case: str,
    stage: str,
    functions: dict[str, Callable[[], object]],
    cache_modes: list[str],
    args: argparse.Namespace,
    flush_buffer: torch.Tensor,
    extra: dict | None = None,
) -> dict[tuple[str, str], float]:
    medians: dict[tuple[str, str], float] = {}
    extra = extra or {}
    for cache_mode in cache_modes:
        measured: dict[str, list[float]] = {"fp8": [], "nvfp4": []}
        outputs: dict[str, object] = {}
        for warmup_idx in range(args.warmup):
            order = ("fp8", "nvfp4") if warmup_idx % 2 == 0 else ("nvfp4", "fp8")
            for provider in order:
                outputs[provider] = functions[provider]()
        torch.cuda.synchronize()
        for output in outputs.values():
            _finite(output)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for sample_idx in range(args.samples):
            # AB/BA pairing avoids systematically giving the second provider
            # the warmer clock/cache state.
            order = ("fp8", "nvfp4") if sample_idx % 2 == 0 else ("nvfp4", "fp8")
            for provider in order:
                if cache_mode == "cold":
                    flush_buffer.add_(1)
                start.record()
                outputs[provider] = functions[provider]()
                end.record()
                end.synchronize()
                measured[provider].append(start.elapsed_time(end) * 1000.0)
        for output in outputs.values():
            _finite(output)
        for provider in ("fp8", "nvfp4"):
            medians[(provider, cache_mode)] = statistics.median(measured[provider])
        fp8_median = medians[("fp8", cache_mode)]
        nvfp4_median = medians[("nvfp4", cache_mode)]
        delta = (nvfp4_median / fp8_median - 1.0) * 100.0
        for provider in ("fp8", "nvfp4"):
            values = measured[provider]
            recorder.emit(
                {
                    "record_type": "timing",
                    "phase": phase,
                    "case": case,
                    "stage": stage,
                    "provider": provider,
                    "cache_mode": cache_mode,
                    "median_us": statistics.median(values),
                    "p90_us": _percentile(values, 0.9),
                    "min_us": min(values),
                    "cv": _cv(values),
                    "samples": len(values),
                    "nvfp4_vs_fp8_pct": delta,
                    **extra,
                }
            )
    return medians


def _emit_single(
    recorder: Recorder,
    phase: str,
    case: str,
    stage: str,
    function: Callable[[], object],
    cache_modes: list[str],
    args: argparse.Namespace,
    flush_buffer: torch.Tensor,
    extra: dict | None = None,
) -> None:
    for cache_mode in cache_modes:
        values, _ = _measure(
            function, args.warmup, args.samples, cache_mode, flush_buffer
        )
        recorder.emit(
            {
                "record_type": "timing",
                "phase": phase,
                "case": case,
                "stage": stage,
                "provider": "bf16",
                "cache_mode": cache_mode,
                "median_us": statistics.median(values),
                "p90_us": _percentile(values, 0.9),
                "min_us": min(values),
                "cv": _cv(values),
                "samples": len(values),
                "nvfp4_vs_fp8_pct": None,
                **(extra or {}),
            }
        )


def run_prefill_case(
    case: PrefillCase,
    recorder: Recorder,
    args: argparse.Namespace,
    cache_modes: list[str],
    flush_buffer: torch.Tensor,
    device: torch.device,
) -> None:
    torch.manual_seed(args.seed + case.prefix + case.extend)
    source = torch.randn((case.total, HEAD_DIM), dtype=torch.bfloat16, device=device)
    source.mul_(0.1)
    k_nope = source[:, :VALUE_DIM].contiguous()
    k_rope = source[:, VALUE_DIM:].contiguous()
    physical, capacity = _physical_rows(case.total, args.seed + case.total, device)
    caches = _allocate_caches(capacity, device)

    # Initialize the prefix/current cache outside timed regions.  Timed full
    # prefill overwrites only the current extend rows, matching serving.
    _write_fp8(caches.fp8_raw, physical, k_nope, k_rope)
    _write_nvfp4(caches.nvfp4_raw, physical, k_nope, k_rope, caches.global_scale)
    torch.cuda.synchronize()

    current = slice(case.prefix, case.total)
    current_loc = physical[current].contiguous()
    current_nope = k_nope[current].contiguous()
    current_rope = k_rope[current].contiguous()
    q8 = torch.randn(
        (case.extend, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    q8.mul_(0.1)
    q64 = torch.zeros(
        (case.extend, FLASHMLA_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    q64[:, :Q_HEADS] = q8
    indices = _prefill_indices(case.extend, case.total, args.topk, device)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    write_functions = {
        "fp8": lambda: _write_fp8(
            caches.fp8_raw, current_loc, current_nope, current_rope
        ),
        "nvfp4": lambda: _write_nvfp4(
            caches.nvfp4_raw,
            current_loc,
            current_nope,
            current_rope,
            caches.global_scale,
        ),
    }
    common = {
        "prefix_tokens": case.prefix,
        "extend_tokens": case.extend,
        "total_kv_tokens": case.total,
        "topk": args.topk,
    }
    _emit_pair(
        recorder,
        "prefill",
        case.name,
        "cache_write_current",
        write_functions,
        cache_modes,
        args,
        flush_buffer,
        common,
    )

    def sparse(kv: torch.Tensor):
        return flash_mla_sparse_fwd(q64, kv, indices, scale, d_v=VALUE_DIM)

    _emit_single(
        recorder,
        "prefill",
        case.name,
        "common_sparse_bf16",
        lambda: sparse(source.view(case.total, 1, HEAD_DIM)),
        cache_modes,
        args,
        flush_buffer,
        common,
    )

    if case.prefix == 0:
        def write_then_sparse(provider: str):
            write_functions[provider]()
            return sparse(source.view(case.total, 1, HEAD_DIM))

        _emit_pair(
            recorder,
            "prefill",
            case.name,
            "full_write_then_sparse",
            {
                "fp8": lambda: write_then_sparse("fp8"),
                "nvfp4": lambda: write_then_sparse("nvfp4"),
            },
            cache_modes,
            args,
            flush_buffer,
            common,
        )
        return

    dequant_functions = {
        "fp8": lambda: dequantize_k_cache_paged(caches.fp8, physical),
        "nvfp4": lambda: dequantize_nvfp4_k_cache_paged(
            caches.nvfp4_raw, physical, caches.global_scale
        ),
    }
    _emit_pair(
        recorder,
        "prefill",
        case.name,
        "full_cache_dequant",
        dequant_functions,
        cache_modes,
        args,
        flush_buffer,
        common,
    )

    # Pre-materialize both providers so sparse-only timings contain exactly the
    # same kernel launch sequence and isolate data-value effects.
    fp8_bf16 = dequant_functions["fp8"]()
    nvfp4_bf16 = dequant_functions["nvfp4"]()
    torch.cuda.synchronize()
    _emit_pair(
        recorder,
        "prefill",
        case.name,
        "sparse_after_materialize",
        {"fp8": lambda: sparse(fp8_bf16), "nvfp4": lambda: sparse(nvfp4_bf16)},
        cache_modes,
        args,
        flush_buffer,
        common,
    )

    def dequant_then_sparse(provider: str):
        materialized = dequant_functions[provider]()
        return sparse(materialized)

    _emit_pair(
        recorder,
        "prefill",
        case.name,
        "dequant_then_sparse",
        {
            "fp8": lambda: dequant_then_sparse("fp8"),
            "nvfp4": lambda: dequant_then_sparse("nvfp4"),
        },
        cache_modes,
        args,
        flush_buffer,
        common,
    )

    def full(provider: str):
        write_functions[provider]()
        return dequant_then_sparse(provider)

    _emit_pair(
        recorder,
        "prefill",
        case.name,
        "full_write_dequant_sparse",
        {"fp8": lambda: full("fp8"), "nvfp4": lambda: full("nvfp4")},
        cache_modes,
        args,
        flush_buffer,
        common,
    )


def run_decode_batch(
    batch: int,
    recorder: Recorder,
    args: argparse.Namespace,
    cache_modes: list[str],
    flush_buffer: torch.Tensor,
    device: torch.device,
) -> None:
    pages_per_request = math.ceil(args.decode_context / PAGE_SIZE)
    capacity_per_request = pages_per_request * PAGE_SIZE
    total_capacity = batch * capacity_per_request
    caches = _allocate_caches(total_capacity, device)
    source = torch.randn(
        (total_capacity, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    source.mul_(0.1)
    loc = torch.arange(total_capacity, dtype=torch.int32, device=device)
    _write_fp8(caches.fp8_raw, loc, source[:, :VALUE_DIM], source[:, VALUE_DIM:])
    _write_nvfp4(
        caches.nvfp4_raw,
        loc,
        source[:, :VALUE_DIM],
        source[:, VALUE_DIM:],
        caches.global_scale,
    )
    del source

    current_kv = torch.randn(
        (batch, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    current_kv.mul_(0.1)
    current_loc = (
        torch.arange(batch, dtype=torch.int32, device=device)
        * capacity_per_request
        + args.decode_context
        - 1
    )
    write_functions = {
        "fp8": lambda: _write_fp8(
            caches.fp8_raw,
            current_loc,
            current_kv[:, :VALUE_DIM],
            current_kv[:, VALUE_DIM:],
        ),
        "nvfp4": lambda: _write_nvfp4(
            caches.nvfp4_raw,
            current_loc,
            current_kv[:, :VALUE_DIM],
            current_kv[:, VALUE_DIM:],
            caches.global_scale,
        ),
    }

    q8 = torch.randn(
        (batch, 1, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    q8.mul_(0.1)
    q64 = torch.zeros(
        (batch, 1, FLASHMLA_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    q64[:, :, :Q_HEADS] = q8
    indices_sets = [
        _decode_indices(
            batch,
            args.decode_context,
            capacity_per_request,
            args.topk,
            set_idx,
            device,
        )
        for set_idx in range(args.decode_index_sets)
    ]
    seqlens = torch.full(
        (batch,), args.decode_context, dtype=torch.int32, device=device
    )
    # Literal production metadata: both formats use H64 scheduling metadata.
    # Their wrappers pad GLM TP8's local H8 Q tensor to the H64 specialization.
    fp8_metadata, fp8_num_splits = get_mla_metadata(
        seqlens,
        FLASHMLA_HEADS,
        1,
        FLASHMLA_HEADS,
        is_fp8_kvcache=True,
        topk=args.topk,
    )
    # Keep this separately configurable for explicit legacy diagnostics, while
    # defaulting to the production H64/cluster-1 fast path.
    nvfp4_metadata, nvfp4_num_splits = get_mla_metadata(
        seqlens,
        args.nvfp4_scheduler_heads,
        1,
        args.nvfp4_scheduler_heads,
        is_fp8_kvcache=True,
        topk=args.topk,
    )
    block_table = torch.empty((batch, 0), dtype=torch.int32, device=device)
    step = {"fp8": 0, "nvfp4": 0}
    scale = 1.0 / math.sqrt(HEAD_DIM)

    def fp8_decode_q(q_input: torch.Tensor):
        indices = indices_sets[step["fp8"] % len(indices_sets)]
        step["fp8"] += 1
        out, lse = flash_mla_with_kvcache(
            q=q_input,
            k_cache=caches.fp8.view(
                -1, PAGE_SIZE, 1, FP8_BYTES_PER_TOKEN
            ),
            block_table=block_table,
            cache_seqlens=seqlens,
            head_dim_v=VALUE_DIM,
            tile_scheduler_metadata=fp8_metadata,
            num_splits=fp8_num_splits,
            softmax_scale=scale,
            is_fp8_kvcache=True,
            indices=indices,
        )
        return out[:, :, :Q_HEADS], lse[:, :Q_HEADS]

    def fp8_decode():
        return fp8_decode_q(q64)

    def fp8_decode_with_pad():
        q_input = q8.new_zeros(
            (batch, 1, FLASHMLA_HEADS, HEAD_DIM)
        )
        q_input[:, :, :Q_HEADS] = q8
        return fp8_decode_q(q_input)

    def nvfp4_decode():
        indices = indices_sets[step["nvfp4"] % len(indices_sets)]
        step["nvfp4"] += 1
        return flash_mla_with_kvcache_nvfp4(
            q=q8,
            k_cache=caches.nvfp4_raw.view(
                -1, PAGE_SIZE, 1, NVFP4_BYTES_PER_TOKEN
            ),
            kv_global_scale=caches.global_scale,
            cache_seqlens=seqlens,
            tile_scheduler_metadata=nvfp4_metadata,
            num_splits=nvfp4_num_splits,
            indices=indices,
            head_dim_v=VALUE_DIM,
            softmax_scale=scale,
        )

    common = {
        "batch": batch,
        "context_tokens": args.decode_context,
        "topk": args.topk,
        "fp8_q_heads": FLASHMLA_HEADS,
        "nvfp4_q_heads": Q_HEADS,
        "metadata_q_heads": FLASHMLA_HEADS,
        "fp8_num_sm_parts": fp8_metadata.shape[0],
        "nvfp4_num_sm_parts": nvfp4_metadata.shape[0],
        "nvfp4_scheduler_heads": args.nvfp4_scheduler_heads,
    }
    _emit_pair(
        recorder,
        "decode",
        f"b{batch}_ctx{args.decode_context}",
        "cache_write_current",
        write_functions,
        cache_modes,
        args,
        flush_buffer,
        common,
    )
    _emit_pair(
        recorder,
        "decode",
        f"b{batch}_ctx{args.decode_context}",
        "fused_attention_production_heads",
        {"fp8": fp8_decode, "nvfp4": nvfp4_decode},
        cache_modes,
        args,
        flush_buffer,
        common,
    )
    _emit_pair(
        recorder,
        "decode",
        f"b{batch}_ctx{args.decode_context}",
        "backend_attention_with_padding",
        {"fp8": fp8_decode_with_pad, "nvfp4": nvfp4_decode},
        cache_modes,
        args,
        flush_buffer,
        common,
    )

    def full_decode(provider: str):
        write_functions[provider]()
        if provider == "fp8":
            return fp8_decode_with_pad()
        return nvfp4_decode()

    _emit_pair(
        recorder,
        "decode",
        f"b{batch}_ctx{args.decode_context}",
        "full_write_then_attention",
        {
            "fp8": lambda: full_decode("fp8"),
            "nvfp4": lambda: full_decode("nvfp4"),
        },
        cache_modes,
        args,
        flush_buffer,
        common,
    )

    fp8_graph, fp8_graph_output = _capture_cuda_graph(
        lambda: full_decode("fp8"), args.warmup
    )
    nvfp4_graph, nvfp4_graph_output = _capture_cuda_graph(
        lambda: full_decode("nvfp4"), args.warmup
    )

    def replay_fp8():
        fp8_graph.replay()
        return fp8_graph_output

    def replay_nvfp4():
        nvfp4_graph.replay()
        return nvfp4_graph_output

    _emit_pair(
        recorder,
        "decode",
        f"b{batch}_ctx{args.decode_context}",
        "full_cuda_graph_replay",
        {"fp8": replay_fp8, "nvfp4": replay_nvfp4},
        cache_modes,
        args,
        flush_buffer,
        common,
    )


def _parse_prefill_case(value: str) -> PrefillCase:
    try:
        prefix_text, extend_text = value.split(":", 1)
        case = PrefillCase(int(prefix_text), int(extend_text))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected PREFIX:EXTEND") from exc
    if case.prefix < 0 or case.extend <= 0:
        raise argparse.ArgumentTypeError("prefix must be >=0 and extend must be >0")
    return case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--prefill-cases",
        type=_parse_prefill_case,
        nargs="+",
        default=[
            PrefillCase(0, 8192),
            PrefillCase(8192, 8192),
            PrefillCase(32768, 8192),
        ],
        metavar="PREFIX:EXTEND",
    )
    parser.add_argument("--decode-batches", type=int, nargs="+", default=[1, 12, 17])
    parser.add_argument("--decode-context", type=int, default=32768)
    parser.add_argument("--decode-index-sets", type=int, default=8)
    parser.add_argument(
        "--nvfp4-scheduler-heads",
        type=int,
        default=FLASHMLA_HEADS,
        help="Head count used to generate the H64 NVFP4 scheduler metadata",
    )
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--cache-mode", choices=("warm", "cold", "both"), default="both"
    )
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1:
        parser.error("--warmup and --samples must be positive")
    if args.topk != TOPK:
        parser.error("the production GLM DSA path requires --topk 2048")
    if any(batch <= 0 for batch in args.decode_batches):
        parser.error("decode batch sizes must be positive")
    if args.decode_index_sets <= 0:
        parser.error("--decode-index-sets must be positive")
    if args.nvfp4_scheduler_heads < 64 or args.nvfp4_scheduler_heads % 64 != 0:
        parser.error("--nvfp4-scheduler-heads must be a positive multiple of 64")
    if args.decode_context < args.topk:
        parser.error("decode context must be at least top-k")
    if args.flush_mib <= 0:
        parser.error("--flush-mib must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("this benchmark targets SM90")
    device = torch.device("cuda")
    cache_modes = ["warm", "cold"] if args.cache_mode == "both" else [args.cache_mode]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir
        or f"/tmp/dsa_kv_phase_compare_{timestamp}"
    ).resolve()
    recorder = Recorder(output_dir, args)
    flush_buffer = torch.zeros(
        args.flush_mib * MIB, dtype=torch.uint8, device=device
    )
    try:
        # The standalone benchmark has no distributed process group.  These
        # overrides reproduce TP-local, DCP-disabled production scatter writes.
        with get_parallel().override(
            dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0
        ):
            if not args.skip_prefill:
                for case in args.prefill_cases:
                    run_prefill_case(
                        case, recorder, args, cache_modes, flush_buffer, device
                    )
                    torch.cuda.empty_cache()
            if not args.skip_decode:
                for batch in args.decode_batches:
                    run_decode_batch(
                        batch, recorder, args, cache_modes, flush_buffer, device
                    )
                    torch.cuda.empty_cache()
    finally:
        recorder.close()
    print(f"results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
