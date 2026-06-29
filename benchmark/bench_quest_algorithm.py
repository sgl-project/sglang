"""
Microbenchmark the current Quest sparse attention algorithm implementation.

This script isolates the algorithm layer. It does not start an SGLang server and
does not call a real attention backend. The goal is to measure where Quest spends
time today:

  1. building page min/max representations;
  2. scoring pages for a query;
  3. running the public retrieve_topk path, including Python-side overhead.

Example:
    python benchmark/bench_quest_algorithm.py --preset smoke --device cuda
    python benchmark/bench_quest_algorithm.py --preset custom --output-csv quest.csv
    python benchmark/bench_quest_algorithm.py --preset custom --repeats 5
"""

import argparse
import csv
import importlib.util
import os
import sys
import time
import types
from dataclasses import dataclass
from typing import Callable

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPARSITY_ALGO_DIR = os.path.join(
    REPO_ROOT, "python", "sglang", "srt", "mem_cache", "sparsity", "algorithms"
)


def ensure_package(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_quest_algorithm_cls():
    # Avoid importing top-level sglang, whose package initializer pulls in
    # unrelated runtime dependencies. This benchmark only needs the algorithm
    # files themselves.
    for package in [
        "sglang",
        "sglang.srt",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.sparsity",
        "sglang.srt.mem_cache.sparsity.algorithms",
    ]:
        ensure_package(package)

    load_module(
        "sglang.srt.mem_cache.sparsity.algorithms.base_algorithm",
        os.path.join(SPARSITY_ALGO_DIR, "base_algorithm.py"),
    )
    quest_module = load_module(
        "sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm",
        os.path.join(SPARSITY_ALGO_DIR, "quest_algorithm.py"),
    )
    return quest_module.QuestAlgorithm


QuestAlgorithm = load_quest_algorithm_cls()


@dataclass
class BenchSparseConfig:
    page_size: int
    sparsity_ratio: float
    num_recent_pages: int

    @property
    def sparse_extra_config(self) -> dict:
        return {
            "sparsity_ratio": self.sparsity_ratio,
            "num_recent_pages": self.num_recent_pages,
        }


@dataclass(frozen=True)
class BenchCase:
    batch_size: int
    seq_len: int
    page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    sparsity_ratio: float
    num_recent_pages: int

    @property
    def num_pages(self) -> int:
        return (self.seq_len + self.page_size - 1) // self.page_size


@dataclass
class PreparedCase:
    case: BenchCase
    algorithm: QuestAlgorithm
    k_buffer: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    queries: torch.Tensor
    phys_pages: torch.Tensor
    forward_batch: object
    sparse_mask: torch.Tensor


class FakeReqToTokenPool:
    def __init__(self, req_to_token: torch.Tensor, max_context_len: int):
        self.req_to_token = req_to_token
        self.max_context_len = max_context_len


class FakeTokenToKVPool:
    def __init__(self, key_buffer: torch.Tensor):
        self.key_buffer = key_buffer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.key_buffer


class FakeStates:
    def __init__(self, size: int, device: torch.device):
        self.repr_constructed = torch.zeros(size, dtype=torch.bool, device=device)
        self.prompt_lens = torch.zeros(size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(size, dtype=torch.int64, device=device)


class FakeForwardBatch:
    def __init__(self, seq_lens: torch.Tensor):
        self.seq_lens = seq_lens


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_req_to_token(
    batch_size: int,
    seq_len: int,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    tokens_per_req = ((seq_len + page_size - 1) // page_size) * page_size
    req_to_token = torch.empty((batch_size, seq_len), dtype=torch.int32, device=device)
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    for req_idx in range(batch_size):
        req_to_token[req_idx] = req_idx * tokens_per_req + positions
    return req_to_token, tokens_per_req


def prepare_case(
    case: BenchCase,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> PreparedCase:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    req_to_token, tokens_per_req = build_req_to_token(
        case.batch_size, case.seq_len, case.page_size, device
    )
    total_tokens = case.batch_size * tokens_per_req
    k_buffer = torch.randn(
        total_tokens,
        case.kv_heads,
        case.head_dim,
        dtype=dtype,
        device=device,
    )

    config = BenchSparseConfig(
        page_size=case.page_size,
        sparsity_ratio=case.sparsity_ratio,
        num_recent_pages=case.num_recent_pages,
    )
    algorithm = QuestAlgorithm(config, device)
    algorithm.initialize_representation_pool(
        start_layer=0,
        end_layer=1,
        token_to_kv_pool=FakeTokenToKVPool(k_buffer),
        req_to_token_pool=FakeReqToTokenPool(req_to_token, case.seq_len),
        states=FakeStates(case.batch_size, device),
    )

    req_pool_indices = torch.arange(case.batch_size, dtype=torch.int64, device=device)
    seq_lens = torch.full(
        (case.batch_size,), case.seq_len, dtype=torch.int64, device=device
    )
    page_idx = torch.arange(case.num_pages, dtype=torch.int64, device=device)
    page_starts = (page_idx * case.page_size).clamp(0, case.seq_len - 1)
    phys_pages = (
        req_to_token[req_pool_indices[:, None], page_starts[None, :]].to(torch.int64)
        // case.page_size
    )
    queries = torch.randn(
        case.batch_size,
        case.q_heads,
        case.head_dim,
        dtype=dtype,
        device=device,
    )
    sparse_mask = torch.ones(case.batch_size, dtype=torch.bool, device=device)

    return PreparedCase(
        case=case,
        algorithm=algorithm,
        k_buffer=k_buffer,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        queries=queries,
        phys_pages=phys_pages,
        forward_batch=FakeForwardBatch(seq_lens),
        sparse_mask=sparse_mask,
    )


def compute_page_representations(prepared: PreparedCase) -> None:
    case = prepared.case
    end_pages = torch.full(
        (case.batch_size,),
        case.num_pages,
        dtype=torch.int64,
        device=prepared.k_buffer.device,
    )
    prepared.algorithm._compute_page_representations(
        0,
        prepared.req_pool_indices,
        prepared.seq_lens,
        0,
        end_pages,
        prepared.k_buffer,
    )


def measure_ms(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[list[float], object]:
    result = None
    with torch.inference_mode():
        for _ in range(warmup):
            result = fn()
        sync(device)

        times: list[float] = []
        if device.type == "cuda":
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn()
                end.record()
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
        else:
            for _ in range(iters):
                start_time = time.perf_counter()
                result = fn()
                sync(device)
                times.append((time.perf_counter() - start_time) * 1000.0)
    return times, result


def summarize_times(times: list[float]) -> dict[str, float]:
    t = torch.tensor(times, dtype=torch.float64)
    return {"mean_ms": float(t.mean().item())}


def run_stage_once(
    stage: str,
    case: BenchCase,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
    repeat: int,
) -> dict:
    prepared = prepare_case(case, device, dtype, seed + repeat)

    if stage != "compute_page_reps":
        with torch.inference_mode():
            compute_page_representations(prepared)
        sync(device)

    selected_pages_mean = 0.0

    if stage == "compute_page_reps":
        fn = lambda: compute_page_representations(prepared)
    elif stage == "retrieve_scores":
        fn = lambda: prepared.algorithm._retrieve_page_scores(
            0,
            prepared.phys_pages,
            prepared.req_pool_indices,
            prepared.queries,
        )
    elif stage == "retrieve_topk":
        fn = lambda: prepared.algorithm.retrieve_topk(
            prepared.queries,
            0,
            prepared.req_pool_indices,
            prepared.sparse_mask,
            forward_batch=prepared.forward_batch,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    base_alloc_mb = 0.0
    if device.type == "cuda":
        sync(device)
        base_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(device)

    times, result = measure_ms(fn, device, warmup, iters)

    peak_delta_mb = 0.0
    if device.type == "cuda":
        sync(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        peak_delta_mb = max(0.0, peak_mb - base_alloc_mb)

    if stage.startswith("retrieve_topk") and result is not None:
        _, lengths = result
        selected_pages_mean = float(lengths.float().mean().item())

    stats = summarize_times(times)
    return {
        "stage": stage,
        "batch_size": case.batch_size,
        "seq_len": case.seq_len,
        "num_pages": case.num_pages,
        "page_size": case.page_size,
        "q_heads": case.q_heads,
        "kv_heads": case.kv_heads,
        "head_dim": case.head_dim,
        "sparsity_ratio": case.sparsity_ratio,
        "recent_pages": case.num_recent_pages,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "warmup": warmup,
        "iters": iters,
        "repeat": repeat,
        "selected_pages_mean": selected_pages_mean,
        "peak_delta_mb": peak_delta_mb,
        **stats,
    }


def aggregate_repeats(repeat_results: list[dict]) -> dict:
    if not repeat_results:
        raise ValueError("repeat_results must not be empty")

    first = repeat_results[0]
    result = {
        key: value
        for key, value in first.items()
        if key
        not in {
            "repeat",
            "selected_pages_mean",
            "peak_delta_mb",
            "mean_ms",
        }
    }
    result["repeats"] = len(repeat_results)

    mean_values = torch.tensor(
        [item["mean_ms"] for item in repeat_results], dtype=torch.float64
    )

    mean_ms = float(mean_values.mean().item())
    mean_ms_std = (
        float(mean_values.std(unbiased=False).item())
        if len(repeat_results) > 1
        else 0.0
    )
    result["selected_pages_mean"] = float(
        torch.tensor(
            [item["selected_pages_mean"] for item in repeat_results],
            dtype=torch.float64,
        )
        .mean()
        .item()
    )
    result["peak_delta_mb"] = max(item["peak_delta_mb"] for item in repeat_results)
    result["mean_ms"] = mean_ms
    result["mean_ms_std"] = mean_ms_std
    result["cv_pct"] = (mean_ms_std / mean_ms * 100.0) if mean_ms > 0 else 0.0
    return result


def run_stage(
    stage: str,
    case: BenchCase,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    repeats: int,
    seed: int,
) -> dict:
    repeat_results = [
        run_stage_once(
            stage,
            case,
            device,
            dtype,
            warmup,
            iters,
            seed,
            repeat,
        )
        for repeat in range(repeats)
    ]
    return aggregate_repeats(repeat_results)


def iter_cases(args: argparse.Namespace) -> list[BenchCase]:
    return [
        BenchCase(
            batch_size=batch_size,
            seq_len=seq_len,
            page_size=page_size,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            sparsity_ratio=sparsity_ratio,
            num_recent_pages=args.num_recent_pages,
        )
        for batch_size in args.batch_sizes
        for seq_len in args.seq_lens
        for page_size in args.page_sizes
        for sparsity_ratio in args.sparsity_ratios
    ]


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "smoke":
        args.batch_sizes = [1, 2]
        args.seq_lens = [1024, 4096]
        args.page_sizes = [16, 32]
        args.warmup = min(args.warmup, 3)
        args.iters = min(args.iters, 5)
        args.repeats = min(args.repeats, 2)
    elif args.preset == "custom":
        return
    else:
        raise ValueError(f"Unknown preset: {args.preset}")


def print_results(results: list[dict]) -> None:
    if not results:
        return
    columns = [
        "stage",
        "batch_size",
        "seq_len",
        "num_pages",
        "page_size",
        "sparsity_ratio",
        "repeats",
        "mean_ms",
        "mean_ms_std",
        "cv_pct",
        "peak_delta_mb",
        "selected_pages_mean",
    ]
    widths = {col: len(col) for col in columns}
    rows = []
    for result in results:
        row = {}
        for col in columns:
            value = result[col]
            if isinstance(value, float):
                row[col] = f"{value:.4f}"
            else:
                row[col] = str(value)
            widths[col] = max(widths[col], len(row[col]))
        rows.append(row)

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(row[col].ljust(widths[col]) for col in columns))


def write_csv(path: str, results: list[dict]) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=["smoke", "custom"],
        default="custom",
        help="smoke = tiny built-in sweep for a quick check; "
        "custom (default) honors the explicit list arguments below.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages or all. Stages: compute_page_reps,"
        "retrieve_scores,retrieve_topk",
    )
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 4])
    parser.add_argument("--seq-lens", type=parse_int_list, default=[4096, 16384])
    parser.add_argument("--page-sizes", type=parse_int_list, default=[16, 32, 64])
    parser.add_argument(
        "--sparsity-ratios", type=parse_float_list, default=[0.25, 0.5, 0.7]
    )
    parser.add_argument("--num-recent-pages", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeat each benchmark condition with a fresh setup and report variance.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_preset(args)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if args.stages == "all":
        stages = [
            "compute_page_reps",
            "retrieve_scores",
            "retrieve_topk",
        ]
    else:
        stages = [stage for stage in args.stages.split(",") if stage]

    results = []
    for case in iter_cases(args):
        for stage in stages:
            try:
                result = run_stage(
                    stage,
                    case,
                    device,
                    dtype,
                    args.warmup,
                    args.iters,
                    args.repeats,
                    args.seed,
                )
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            results.append(result)
            print_results([result])

    print("\nSummary")
    print_results(results)

    if args.output_csv:
        write_csv(args.output_csv, results)
        print(f"\nWrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
