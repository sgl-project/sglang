#!/usr/bin/env python3
import argparse
import bisect
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PRE_KERNELS = (
    "mega_moe_pre_dispatch_kernel",
    "mega_moe_pre_dispatch_waterfill_rank2_kernel",
)
IMPL_KERNEL = "sm100_fp8_fp4_mega_moe_impl"
PTR_QUERY = "cudaPointerGetAttributes"
LAUNCH_EX = "cuLaunchKernelEx"
CALLS_PER_STEP = 43
IMPL_SHAPE_RE = re.compile(r"sm100_fp8_fp4_mega_moe_impl<(\d+)u")


def load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        return json.load(f)


def infer_rank_phase(path: Path) -> tuple[str, str]:
    name = path.name
    rank_match = re.search(r"TP[-_]?(\d+)|rank[-_]?(\d+)", name, re.IGNORECASE)
    rank = rank_match.group(1) or rank_match.group(2) if rank_match else "0"
    phase = "EXTEND" if "EXTEND" in name.upper() else "DECODE"
    return f"TP{rank}", phase


def event_end(event: dict) -> float:
    return float(event.get("ts", 0.0)) + float(event.get("dur", 0.0))


def count_ts_between(timestamps: list[float], start: float, end: float) -> int:
    return bisect.bisect_right(timestamps, end) - bisect.bisect_left(timestamps, start)


def impl_shape_m(name: str) -> int | None:
    match = IMPL_SHAPE_RE.search(name)
    return int(match.group(1)) if match else None


def extract_file_calls(path: Path) -> tuple[str, str, list[dict]]:
    rank, phase = infer_rank_phase(path)
    trace = load_trace(path)
    pre = []
    impl = []
    ptr = []
    launch_ex = []

    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        if dur <= 0:
            continue
        if any(kernel in name for kernel in PRE_KERNELS):
            pre.append(event)
        elif IMPL_KERNEL in name:
            impl.append(event)
        elif PTR_QUERY in name:
            ptr.append(event)
        elif LAUNCH_EX in name:
            launch_ex.append(event)

    pre.sort(key=lambda e: float(e.get("ts", 0.0)))
    impl.sort(key=lambda e: float(e.get("ts", 0.0)))
    ptr.sort(key=lambda e: float(e.get("ts", 0.0)))
    launch_ex.sort(key=lambda e: float(e.get("ts", 0.0)))
    ptr_ts = [float(event.get("ts", 0.0)) for event in ptr]
    launch_ex_ts = [float(event.get("ts", 0.0)) for event in launch_ex]

    calls = []
    for index, (pre_event, impl_event) in enumerate(zip(pre, impl)):
        impl_name = str(impl_event.get("name", ""))
        pre_start = float(pre_event.get("ts", 0.0))
        pre_end = event_end(pre_event)
        impl_start = float(impl_event.get("ts", 0.0))
        impl_end = event_end(impl_event)
        gap_start = pre_end
        gap_end = impl_start
        ptr_count = count_ts_between(ptr_ts, gap_start, gap_end)
        launch_count = count_ts_between(launch_ex_ts, gap_start, gap_end)
        calls.append(
            {
                "index": index,
                "pre_us": max(0.0, pre_end - pre_start),
                "impl_us": max(0.0, impl_end - impl_start),
                "span_us": max(0.0, impl_end - pre_start),
                "gap_us": max(0.0, gap_end - gap_start),
                "ptr": ptr_count,
                "launch_ex": launch_count,
                "shape_m": impl_shape_m(impl_name),
            }
        )
    return rank, phase, calls


def summarize_profile(profile_dir: Path) -> dict:
    by_phase_rank: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for path in sorted(profile_dir.glob("**/*.json*")):
        rank, phase, calls = extract_file_calls(path)
        if calls:
            by_phase_rank[phase][rank] = calls

    phase_summaries = {}
    for phase, by_rank in sorted(by_phase_rank.items()):
        if not by_rank:
            continue
        call_count = min(len(calls) for calls in by_rank.values())
        max_rank_calls = []
        for index in range(call_count):
            rank_calls = [calls[index] for calls in by_rank.values()]
            max_impl_call = max(rank_calls, key=lambda call: call["impl_us"])
            max_rank_calls.append(
                {
                    "index": index,
                    "pre_us": max(call["pre_us"] for call in rank_calls),
                    "impl_us": max(call["impl_us"] for call in rank_calls),
                    "span_us": max(call["span_us"] for call in rank_calls),
                    "gap_us": max(call["gap_us"] for call in rank_calls),
                    "ptr": sum(call["ptr"] for call in rank_calls),
                    "launch_ex": sum(call["launch_ex"] for call in rank_calls),
                    "shape_m": max_impl_call["shape_m"],
                }
            )

        steps = []
        for step_idx in range(0, call_count, CALLS_PER_STEP):
            chunk = max_rank_calls[step_idx : step_idx + CALLS_PER_STEP]
            if not chunk:
                continue
            shape_counts = Counter(
                call["shape_m"] for call in chunk if call["shape_m"] is not None
            )
            steps.append(
                {
                    "step": step_idx // CALLS_PER_STEP,
                    "calls": len(chunk),
                    "pre_ms": sum(call["pre_us"] for call in chunk) / 1000.0,
                    "impl_ms": sum(call["impl_us"] for call in chunk) / 1000.0,
                    "span_ms": sum(call["span_us"] for call in chunk) / 1000.0,
                    "gap_ms": sum(call["gap_us"] for call in chunk) / 1000.0,
                    "ptr": sum(call["ptr"] for call in chunk),
                    "launch_ex": sum(call["launch_ex"] for call in chunk),
                    "impl_shape_m_counts": dict(sorted(shape_counts.items())),
                }
            )

        shape_counts = Counter(
            call["shape_m"] for call in max_rank_calls if call["shape_m"] is not None
        )
        phase_summaries[phase] = {
            "ranks": sorted(by_rank),
            "calls": call_count,
            "pre_ms": sum(call["pre_us"] for call in max_rank_calls) / 1000.0,
            "impl_ms": sum(call["impl_us"] for call in max_rank_calls) / 1000.0,
            "span_ms": sum(call["span_us"] for call in max_rank_calls) / 1000.0,
            "gap_ms": sum(call["gap_us"] for call in max_rank_calls) / 1000.0,
            "ptr": sum(call["ptr"] for call in max_rank_calls),
            "launch_ex": sum(call["launch_ex"] for call in max_rank_calls),
            "impl_shape_m_counts": dict(sorted(shape_counts.items())),
            "steps": steps,
        }
    return phase_summaries


def pct_speedup(old: float, new: float) -> float:
    if new == 0:
        return 0.0
    return (old / new - 1.0) * 100.0


def format_shape_counts(counts: dict[int, int]) -> str:
    if not counts:
        return "n/a"
    return ",".join(f"{shape}:{count}" for shape, count in sorted(counts.items()))


def print_case(name: str, summary: dict):
    print(f"\n== {name} ==")
    for phase in ("EXTEND", "DECODE"):
        data = summary.get(phase)
        if not data:
            continue
        print(
            f"{phase}: calls={data['calls']} ranks={','.join(data['ranks'])} "
            f"impl={data['impl_ms']:.6f}ms span={data['span_ms']:.6f}ms "
            f"gap={data['gap_ms']:.6f}ms pre={data['pre_ms']:.6f}ms "
            f"ptr={data['ptr']} launchEx={data['launch_ex']} "
            f"impl_shape_m={format_shape_counts(data['impl_shape_m_counts'])}"
        )
        for step in data["steps"][:6]:
            print(
                f"  step{step['step']}: calls={step['calls']} "
                f"impl={step['impl_ms']:.6f} span={step['span_ms']:.6f} "
                f"gap={step['gap_ms']:.6f} pre={step['pre_ms']:.6f} "
                f"ptr={step['ptr']} launchEx={step['launch_ex']} "
                f"impl_shape_m={format_shape_counts(step['impl_shape_m_counts'])}"
            )


def print_compare(left_name: str, left: dict, right_name: str, right: dict):
    print(f"\n== compare {left_name} -> {right_name} ==")
    for phase in ("EXTEND", "DECODE"):
        ldata = left.get(phase)
        rdata = right.get(phase)
        if not ldata or not rdata:
            continue
        print(
            f"{phase}: impl_speedup={pct_speedup(ldata['impl_ms'], rdata['impl_ms']):+.3f}% "
            f"span_speedup={pct_speedup(ldata['span_ms'], rdata['span_ms']):+.3f}% "
            f"gap_delta={rdata['gap_ms'] - ldata['gap_ms']:+.6f}ms "
            f"ptr_delta={rdata['ptr'] - ldata['ptr']:+d} "
            f"launchEx_delta={rdata['launch_ex'] - ldata['launch_ex']:+d}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profiles",
        nargs="+",
        help="case_name=profile_dir entries",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    summaries = {}
    for item in args.profiles:
        if "=" not in item:
            raise SystemExit(f"Expected case_name=profile_dir, got: {item}")
        name, raw_path = item.split("=", 1)
        summaries[name] = summarize_profile(Path(raw_path))
        print_case(name, summaries[name])

    names = list(summaries)
    for idx, left_name in enumerate(names):
        for right_name in names[idx + 1 :]:
            print_compare(left_name, summaries[left_name], right_name, summaries[right_name])

    if args.json_output:
        args.json_output.write_text(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
