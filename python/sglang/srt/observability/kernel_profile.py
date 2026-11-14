"""Device-local kernel coverage for an exported Torch profiler window.

Times remain in the Chrome trace clock. Busy is the union of observed kernel
intervals, not device utilization: other processes, memcpy and lost records are
outside the observation. Backend loss is unknown. This runs only after the
existing profiler has stopped; it neither starts CUPTI nor creates a second
profiler.
"""

from __future__ import annotations

import gzip
import json
import logging
import zlib
from collections import defaultdict
from decimal import Decimal

logger = logging.getLogger(__name__)


def _ns(us) -> int:
    return round(Decimal(str(us)) * 1000)


def interval_union_ns(intervals) -> int:
    """Length of a union, including overlap across streams on one device."""
    end = None
    total = 0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


# Bound work on the synchronous profiler-stop path by decompressed size.
# Larger captures remain available for offline analysis.
_INLINE_LIMIT = 16 * 1024 * 1024


def read_profile(path, *, max_bytes=None) -> dict:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        if max_bytes is None:
            payload = json.load(stream, parse_float=Decimal)
        else:
            data = stream.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(
                    "capture exceeds the inline analysis limit; analyze offline with "
                    f"python -m sglang.srt.observability.kernel_profile {path!s}"
                )
            payload = json.loads(data, parse_float=Decimal)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("traceEvents"), list
    ):
        raise ValueError("expected a Chrome traceEvents list")
    return payload


def kernel_intervals(payload: dict):
    """Validate kernel records at the trace boundary; keep raw event identity."""
    records = []
    skipped = 0
    for index, event in enumerate(payload["traceEvents"]):
        if not isinstance(event, dict):
            skipped += 1
            continue
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        try:
            start = _ns(event["ts"])
            end = start + _ns(event["dur"])
            if not isinstance(event.get("name"), str) or not event["name"]:
                raise ValueError("kernel name is required")
            device = event["args"]["device"]
            stream = event["args"]["stream"]
            if (
                type(device) is not int
                or type(stream) is not int
                or start < 0
                or end <= start
                or device < 0
                or stream < 0
            ):
                raise ValueError("invalid kernel device, stream or interval")
        except (KeyError, TypeError, ValueError, ArithmeticError):
            skipped += 1
            continue
        records.append((index, event, device, start, end))
    return records, skipped


def summarize_profile(payload: dict) -> dict:
    records, skipped = kernel_intervals(payload)
    # Torch emits one capture boundary per exported window. If absent, report
    # the kernel envelope explicitly, with no claim of total capture coverage.
    windows = []
    for event in payload["traceEvents"]:
        if (
            isinstance(event, dict)
            and event.get("ph") == "X"
            and str(event.get("name", "")).startswith("PyTorch Profiler")
        ):
            try:
                start = _ns(event["ts"])
                stop = start + _ns(event["dur"])
                if stop > start:
                    windows.append((start, stop))
            except (KeyError, TypeError, ValueError, ArithmeticError):
                pass
    if windows:
        start = min(a for a, _ in windows)
        stop = max(b for _, b in windows)
        source = "profiler_capture"
    elif records:
        start = min(a for _, _, _, a, _ in records)
        stop = max(b for _, _, _, _, b in records)
        source = "observed_kernel_envelope"
    else:
        start = stop = 0
        source = "unavailable"

    by_device = defaultdict(list)
    for _, event, device, a, b in records:
        by_device[device].append((a, b))
    devices = []
    for device, intervals in sorted(by_device.items()):
        clipped = [
            (max(a, start), min(b, stop))
            for a, b in intervals
            if a < stop and b > start
        ]
        busy = interval_union_ns(clipped)
        devices.append(
            {
                "device": device,
                "kernel_records": len(intervals),
                "kernels_started_in_window": sum(
                    start <= a < stop for a, _ in intervals
                ),
                "kernel_busy_union_ns": busy,
                "kernel_time_sum_in_window_ns": sum(b - a for a, b in clipped),
                "uncovered_ns": stop - start - busy,
                "dropped_records": None,
            }
        )
    return {
        "schema_version": 1,
        "clock_domain": "chrome_trace_relative_ns",
        "collection_mode": "window",
        "window_source": source,
        "window_start_ns": start,
        "window_end_ns": stop,
        "busy_scope": "union_of_observed_kernel_intervals",
        "uncovered_scope": "capture_time_without_observed_kernels",
        "skipped_kernel_records": skipped,
        "devices": devices,
    }


def export_kernel_summary(trace_path):
    """Write a sibling summary when enabled; preserve the original trace."""
    from pathlib import Path

    from sglang.srt.environ import envs

    if not (
        envs.SGLANG_PROFILE_KERNEL_SUMMARY.get()
        or envs.SGLANG_PROFILE_STEP_KERNELS.get()
    ):
        return
    path = Path(trace_path)
    output = path.with_name(
        path.name.removesuffix(".gz").removesuffix(".json") + ".kernel-summary.json"
    )
    try:
        payload = read_profile(path, max_bytes=_INLINE_LIMIT)
        summary = summarize_profile(payload)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.replace(output)
        if envs.SGLANG_PROFILE_STEP_KERNELS.get():
            from sglang.srt.observability.step_kernel_profile import export_step_kernels

            steps_path = export_step_kernels(
                path,
                payload,
                max_bytes=envs.SGLANG_PROFILE_STEP_TRACE_MAX_BYTES.get(),
            )
            logger.info("Forward kernel trace saved to %s", steps_path)
    except (OSError, EOFError, zlib.error, ValueError, ArithmeticError):
        logger.warning("Could not summarize profiler trace %s", path, exc_info=True)
    else:
        logger.info("Kernel window summary saved to %s", output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path")
    parser.add_argument(
        "--steps", action="store_true", help="also write a sibling .steps.json"
    )
    parser.add_argument("--max-step-bytes", type=int, default=8 * 1024 * 1024)
    options = parser.parse_args()
    payload = read_profile(options.trace_path)
    if options.steps:
        from sglang.srt.observability.step_kernel_profile import export_step_kernels

        export_step_kernels(
            options.trace_path, payload, max_bytes=options.max_step_bytes
        )
    print(json.dumps(summarize_profile(payload), indent=2))
