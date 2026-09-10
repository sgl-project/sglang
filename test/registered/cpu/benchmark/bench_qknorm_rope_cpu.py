import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

import sgl_kernel  # noqa: F401
from sglang.kernels.ops.diffusion.rope.qknorm_rope_jit import fused_inplace_qknorm_rope

DTYPE = torch.bfloat16
NUM_HEADS = 56
HEAD_DIM = 128
ROPE_DIM = 96
EPS = 1e-6
DEFAULT_CASES = [256, 1024, 4096]


def create_cos_sin_cache(rotary_dim: int, max_position: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(DTYPE)


def make_case(num_tokens: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(10_000 + num_tokens)
    return {
        "q": torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, generator=gen),
        "k": torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE, generator=gen),
        "q_weight": torch.randn(HEAD_DIM, dtype=DTYPE, generator=gen),
        "k_weight": torch.randn(HEAD_DIM, dtype=DTYPE, generator=gen),
        "positions": torch.arange(num_tokens, dtype=torch.int64),
        "cos_sin_cache": create_cos_sin_cache(ROPE_DIM, max_position=num_tokens),
    }


def rmsnorm_baseline(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    norm = nn.RMSNorm(x.shape[-1], eps=eps, dtype=x.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_baseline(x: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    half = cos_sin_cache.shape[-1] // 2
    cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    x_rot, x_pass = x[..., : cos.shape[-1]], x[..., cos.shape[-1] :]
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


def baseline_fn(q: torch.Tensor, k: torch.Tensor, case: dict[str, torch.Tensor]) -> None:
    q_norm = rmsnorm_baseline(q, case["q_weight"], EPS)
    k_norm = rmsnorm_baseline(k, case["k_weight"], EPS)
    cache = case["cos_sin_cache"].index_select(0, case["positions"])
    apply_rope_baseline(q_norm, cache)
    apply_rope_baseline(k_norm, cache)


def fused_fn(q: torch.Tensor, k: torch.Tensor, case: dict[str, torch.Tensor]) -> None:
    fused_inplace_qknorm_rope(
        q,
        k,
        case["q_weight"],
        case["k_weight"],
        case["cos_sin_cache"],
        case["positions"],
        is_neox=True,
        eps=EPS,
        head_dim=HEAD_DIM,
        rope_dim=ROPE_DIM,
        round_norm_before_rope=True,
    )


def time_case(case: dict[str, torch.Tensor], warmup: int, iters: int) -> dict:
    q_work = torch.empty_like(case["q"])
    k_work = torch.empty_like(case["k"])

    def bench(label: str, fn):
        for _ in range(warmup):
            q_work.copy_(case["q"])
            k_work.copy_(case["k"])
            fn(q_work, k_work)
        times = []
        for _ in range(iters):
            q_work.copy_(case["q"])
            k_work.copy_(case["k"])
            start = time.perf_counter()
            fn(q_work, k_work)
            end = time.perf_counter()
            times.append(end - start)
        return {
            "provider": label,
            "latency_median_ms": statistics.median(times) * 1000.0,
            "latency_min_ms": min(times) * 1000.0,
            "latency_max_ms": max(times) * 1000.0,
            "latency_p95_ms": statistics.quantiles(times, n=20)[-1] * 1000.0
            if len(times) >= 20
            else max(times) * 1000.0,
            "samples_ms": [round(t * 1000.0, 6) for t in times],
        }

    baseline = bench(
        "baseline_minimax_cpu",
        lambda q, k: baseline_fn(q, k, case),
    )
    fused = bench("fused_cpu", lambda q, k: fused_fn(q, k, case))
    speedup = baseline["latency_median_ms"] / fused["latency_median_ms"]
    return {
        "num_tokens": case["q"].shape[0],
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "rope_dim": ROPE_DIM,
        "dtype": str(DTYPE),
        "baseline": baseline,
        "fused": fused,
        "speedup": speedup,
    }


def shell_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def format_cpu_ranges(cpus: list[int]) -> str:
    if not cpus:
        return ""
    ranges = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def read_thread_affinity_map() -> dict[int, str]:
    affinity_map: dict[int, str] = {}
    for task_dir in Path("/proc/self/task").iterdir():
        status_path = task_dir / "status"
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("Cpus_allowed_list"):
                    affinity_map[int(task_dir.name)] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            continue
    return affinity_map


def summarize_thread_affinity(affinity_map: dict[int, str]) -> dict:
    counts = Counter(affinity_map.values())
    singleton_cpus = sorted(
        int(mask) for mask in counts if mask.isdigit()
    )
    return {
        "thread_count": len(affinity_map),
        "unique_masks": len(counts),
        "single_cpu_thread_count": sum(counts[mask] for mask in counts if mask.isdigit()),
        "single_cpu_binding_head": singleton_cpus[:16],
        "single_cpu_binding_range": format_cpu_ranges(singleton_cpus),
        "top_masks": [
            {"mask": mask, "threads": count}
            for mask, count in counts.most_common(16)
        ],
    }


def current_thread_affinity_snapshot() -> dict:
    cpus = sorted(os.sched_getaffinity(0))
    return {
        "count": len(cpus),
        "head": cpus[:16],
        "range": format_cpu_ranges(cpus),
    }


def read_cpuset_effective() -> str | None:
    for candidate in [
        Path("/sys/fs/cgroup/cpuset.cpus.effective"),
        Path("/sys/fs/cgroup/cpuset.cpus"),
    ]:
        try:
            value = candidate.read_text().strip()
        except OSError:
            continue
        if value:
            return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument("--require-min-speedup", type=float, default=None)
    args = parser.parse_args()

    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads:
        torch.set_num_threads(int(omp_threads))

    pre_parallel_affinity = current_thread_affinity_snapshot()
    pre_parallel_taskset = shell_output(["bash", "-lc", "taskset -pc $$"])
    pre_parallel_thread_affinity = summarize_thread_affinity(read_thread_affinity_map())

    results = [time_case(make_case(tokens), args.warmup, args.iters) for tokens in args.tokens]
    min_speedup = min(item["speedup"] for item in results)

    post_parallel_affinity = current_thread_affinity_snapshot()
    post_parallel_taskset = shell_output(["bash", "-lc", "taskset -pc $$"])
    post_parallel_thread_affinity = summarize_thread_affinity(read_thread_affinity_map())

    payload = {
        "command": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "cpuset_cgroup_effective": read_cpuset_effective(),
        "pre_parallel_main_thread_affinity": pre_parallel_affinity,
        "pre_parallel_taskset": pre_parallel_taskset,
        "pre_parallel_thread_affinity": pre_parallel_thread_affinity,
        "post_parallel_main_thread_affinity": post_parallel_affinity,
        "post_parallel_taskset": post_parallel_taskset,
        "post_parallel_thread_affinity": post_parallel_thread_affinity,
        "affinity_note": (
            "post_parallel_main_thread_affinity reports the calling thread after the first OpenMP/oneDNN parallel region; "
            "inspect post_parallel_thread_affinity for worker placement across the launch cpuset"
        ),
        "env": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "KMP_AFFINITY",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "SGLANG_CPU_OMP_THREADS_BIND",
            ]
        },
        "lscpu": shell_output(["lscpu"]),
        "numactl_h": shell_output(["numactl", "-H"]),
        "parallel_info": torch.__config__.parallel_info(),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
        "min_speedup": min_speedup,
    }

    if args.result_file is not None:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))

    if args.require_min_speedup is not None and min_speedup < args.require_min_speedup:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
