#!/usr/bin/env python3
import argparse
import bisect
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path


PRE_KERNELS = (
    "mega_moe_pre_dispatch_kernel",
    "mega_moe_pre_dispatch_waterfill_rank2_kernel",
)
IMPL_KERNEL = "sm100_fp8_fp4_mega_moe_impl"
CALLS_PER_STEP = 43


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


def event_start(event: dict) -> float:
    return float(event.get("ts", 0.0))


def event_end(event: dict) -> float:
    return float(event.get("ts", 0.0)) + float(event.get("dur", 0.0))


def is_gpu_event(event: dict) -> bool:
    cat = str(event.get("cat", "")).lower()
    args = event.get("args", {})
    name = str(event.get("name", "")).lower()
    return (
        "kernel" in cat
        or "gpu" in cat
        or "cuda" in cat
        or "kernel" in name
        or args.get("External id") is not None
    )


def merged_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def overlap_duration(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    clipped = [(max(a, start), min(b, end)) for a, b in intervals if a < end and b > start]
    clipped = [(a, b) for a, b in clipped if b > a]
    return merged_duration(clipped)


def extract_rank_phase(path: Path) -> tuple[str, str, list[dict], list[tuple[float, float]]]:
    rank, phase = infer_rank_phase(path)
    trace = load_trace(path)
    pre = []
    impl = []
    gpu_intervals = []

    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        dur = float(event.get("dur", 0.0))
        if dur <= 0:
            continue
        name = str(event.get("name", ""))
        if any(kernel in name for kernel in PRE_KERNELS):
            pre.append(event)
        elif IMPL_KERNEL in name:
            impl.append(event)
        if is_gpu_event(event):
            gpu_intervals.append((event_start(event), event_end(event)))

    pre.sort(key=event_start)
    impl.sort(key=event_start)
    gpu_intervals.sort()

    calls = []
    for index, (pre_event, impl_event) in enumerate(zip(pre, impl)):
        pre_start = event_start(pre_event)
        impl_end = event_end(impl_event)
        calls.append(
            {
                "index": index,
                "pre_start": pre_start,
                "span_us": max(0.0, impl_end - pre_start),
            }
        )
    return rank, phase, calls, gpu_intervals


def summarize_profile(profile_dir: Path, step_offset: int) -> dict:
    by_phase_rank = defaultdict(dict)
    for path in sorted(profile_dir.glob("**/*.json*")):
        rank, phase, calls, gpu_intervals = extract_rank_phase(path)
        if calls:
            by_phase_rank[phase][rank] = {
                "calls": calls,
                "gpu_intervals": gpu_intervals,
            }

    summaries = {}
    for phase, by_rank in sorted(by_phase_rank.items()):
        if not by_rank:
            continue
        call_count = min(len(data["calls"]) for data in by_rank.values())
        total_steps = call_count // CALLS_PER_STEP
        if total_steps <= step_offset:
            continue

        first_rank = sorted(by_rank)[0]
        anchor_calls = by_rank[first_rank]["calls"][: total_steps * CALLS_PER_STEP]
        step_bounds = []
        for step in range(total_steps):
            start = anchor_calls[step * CALLS_PER_STEP]["pre_start"]
            if step + 1 < total_steps:
                end = anchor_calls[(step + 1) * CALLS_PER_STEP]["pre_start"]
            else:
                last_call_end = max(
                    data["calls"][total_steps * CALLS_PER_STEP - 1]["pre_start"]
                    + data["calls"][total_steps * CALLS_PER_STEP - 1]["span_us"]
                    for data in by_rank.values()
                )
                last_gpu_end = max(
                    (end for data in by_rank.values() for _, end in data["gpu_intervals"]),
                    default=last_call_end,
                )
                end = max(last_call_end, last_gpu_end)
            step_bounds.append((start, end))

        steps = []
        for step, (start, end) in enumerate(step_bounds):
            moe_by_rank = []
            full_span_by_rank = []
            active_by_rank = []
            for rank, data in by_rank.items():
                begin = step * CALLS_PER_STEP
                finish = begin + CALLS_PER_STEP
                calls = data["calls"][begin:finish]
                moe_by_rank.append((rank, sum(call["span_us"] for call in calls)))

                intervals = data["gpu_intervals"]
                starts = [a for a, _ in intervals]
                left = bisect.bisect_left(starts, start)
                right = bisect.bisect_left(starts, end)
                window_intervals = [(a, b) for a, b in intervals[left:right] if b > start]
                if window_intervals:
                    span = max(b for _, b in window_intervals) - min(a for a, _ in window_intervals)
                    active = overlap_duration(window_intervals, start, end)
                else:
                    span = 0.0
                    active = 0.0
                full_span_by_rank.append((rank, span))
                active_by_rank.append((rank, active))

            max_moe_rank, max_moe_us = max(moe_by_rank, key=lambda item: item[1])
            max_full_rank, max_full_span_us = max(full_span_by_rank, key=lambda item: item[1])
            max_active_rank, max_active_us = max(active_by_rank, key=lambda item: item[1])
            steps.append(
                {
                    "step": step,
                    "full_span_ms": max_full_span_us / 1000.0,
                    "full_span_rank": max_full_rank,
                    "gpu_active_ms": max_active_us / 1000.0,
                    "gpu_active_rank": max_active_rank,
                    "moe_span_ms": max_moe_us / 1000.0,
                    "moe_span_rank": max_moe_rank,
                    "moe_share_pct": (max_moe_us / max_full_span_us * 100.0)
                    if max_full_span_us
                    else 0.0,
                }
            )

        used_steps = steps[step_offset:]
        full_ms = sum(step["full_span_ms"] for step in used_steps)
        active_ms = sum(step["gpu_active_ms"] for step in used_steps)
        moe_ms = sum(step["moe_span_ms"] for step in used_steps)
        summaries[phase] = {
            "ranks": sorted(by_rank),
            "steps": steps,
            "step_offset": step_offset,
            "used_steps": [step["step"] for step in used_steps],
            "full_span_ms": full_ms,
            "gpu_active_ms": active_ms,
            "moe_span_ms": moe_ms,
            "moe_share_pct": (moe_ms / full_ms * 100.0) if full_ms else 0.0,
        }
    return summaries


def pct_speedup(old: float, new: float) -> float | None:
    if old <= 0 or new <= 0:
        return None
    return (old / new - 1.0) * 100.0


def target_speedup(old_full: float, old_moe: float, new_moe: float) -> float | None:
    predicted_new = old_full - old_moe + new_moe
    return pct_speedup(old_full, predicted_new)


def print_case(name: str, summary: dict):
    print(f"\n== {name} ==")
    for phase in ("EXTEND", "DECODE"):
        data = summary.get(phase)
        if not data:
            continue
        print(
            f"{phase}: ranks={','.join(data['ranks'])} used_steps={data['used_steps']} "
            f"full_span={data['full_span_ms']:.6f}ms "
            f"gpu_active={data['gpu_active_ms']:.6f}ms "
            f"moe_span={data['moe_span_ms']:.6f}ms "
            f"moe_share={data['moe_share_pct']:.3f}%"
        )
        for step in data["steps"]:
            prefix = "*" if step["step"] in data["used_steps"] else " "
            print(
                f"  {prefix}step{step['step']}: "
                f"full={step['full_span_ms']:.6f}ms({step['full_span_rank']}) "
                f"active={step['gpu_active_ms']:.6f}ms({step['gpu_active_rank']}) "
                f"moe={step['moe_span_ms']:.6f}ms({step['moe_span_rank']}) "
                f"share={step['moe_share_pct']:.3f}%"
            )


def print_compare(left_name: str, left: dict, right_name: str, right: dict):
    print(f"\n== target {left_name} -> {right_name} ==")
    for phase in ("EXTEND", "DECODE"):
        ldata = left.get(phase)
        rdata = right.get(phase)
        if not ldata or not rdata:
            continue
        moe_speed = pct_speedup(ldata["moe_span_ms"], rdata["moe_span_ms"])
        target = target_speedup(ldata["full_span_ms"], ldata["moe_span_ms"], rdata["moe_span_ms"])
        actual_span = pct_speedup(ldata["full_span_ms"], rdata["full_span_ms"])
        print(
            f"{phase}: moe_speedup={moe_speed:+.3f}% "
            f"target_e2e={target:+.3f}% "
            f"observed_trace_full_span={actual_span:+.3f}% "
            f"old_full={ldata['full_span_ms']:.6f}ms old_moe={ldata['moe_span_ms']:.6f}ms "
            f"new_full={rdata['full_span_ms']:.6f}ms new_moe={rdata['moe_span_ms']:.6f}ms"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", help="case_name=profile_dir entries")
    parser.add_argument("--step-offset", type=int, default=1, help="drop early profiler steps")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    summaries = {}
    for item in args.profiles:
        if "=" not in item:
            raise SystemExit(f"Expected case_name=profile_dir, got: {item}")
        name, raw_path = item.split("=", 1)
        summaries[name] = summarize_profile(Path(raw_path), args.step_offset)
        print_case(name, summaries[name])

    names = list(summaries)
    for idx, left_name in enumerate(names):
        for right_name in names[idx + 1 :]:
            print_compare(left_name, summaries[left_name], right_name, summaries[right_name])

    if args.json_output:
        args.json_output.write_text(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
