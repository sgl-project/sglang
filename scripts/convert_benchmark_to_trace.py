"""Convert benchmark --output-details JSON/JSONL into a client response timeline.

Open the resulting Chrome Trace Event JSON at https://ui.perfetto.dev.
Times describe client-observed output, not server scheduling or GPU execution.
"""

import argparse
import json
import math
from pathlib import Path


def load_benchmark_result(path, run_index=-1):
    """Read one JSON object or select a nonempty JSONL record (last by default)."""
    contents = Path(path).read_text(encoding="utf-8")
    if not contents.strip():
        raise ValueError("Input file is empty")
    try:
        records = [json.loads(contents)]
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(contents.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc
    try:
        result = records[run_index]
    except IndexError as exc:
        raise ValueError(
            f"Run index {run_index} is out of range for {len(records)} record(s)"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("Selected benchmark result must be a JSON object")
    return result


def _nonnegative_number(value, label):
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return value


def _microseconds(value, label):
    result = _nonnegative_number(value, label) * 1e6
    if not math.isfinite(result):
        raise ValueError(f"{label} is too large to represent in microseconds")
    return round(result)


def convert_to_chrome_trace(perf_data):
    """Build a Chrome trace with one lane per successful observed request."""
    required = ("arrival_times", "ttfts", "itls")
    for key in required:
        if key not in perf_data:
            raise ValueError(
                f"Missing {key!r}; rerun the benchmark with --output-details "
                "using a version that exports arrival_times"
            )
        if not isinstance(perf_data[key], list):
            raise ValueError(f"{key} must be an array")
    count = len(perf_data["arrival_times"])
    for key in (*required, "input_lens", "output_lens", "successes", "errors"):
        if key in perf_data and (
            not isinstance(perf_data[key], list) or len(perf_data[key]) != count
        ):
            raise ValueError(f"{key} must have {count} entries to match arrival_times")

    semantics = (
        "Client-observed timing in microseconds relative to benchmark start. "
        "TTFT includes network, queueing, and prefill. Output intervals are "
        "reconstructed from ITLs; streaming chunks may contain multiple tokens. "
        "The native SGLang backend may interpolate ITLs within a chunk. "
        "These are not exact token execution or server scheduling times."
    )
    process_name = str(perf_data.get("backend", "Benchmark client"))
    if perf_data.get("model_id"):
        process_name += f" — {perf_data['model_id']}"
    events = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "args": {"name": process_name},
        }
    ]
    skipped = {"failed": 0, "unavailable_start": 0, "no_first_output": 0}
    included = 0
    for index in range(count):
        start = perf_data["arrival_times"][index]
        if start is not None:
            _microseconds(start, f"arrival_times[{index}]")
        ttft = perf_data["ttfts"][index]
        if isinstance(ttft, list):
            raise ValueError(
                "Nested ttfts from the HiCache benchmark are not supported; "
                "use the general benchmark (python -m sglang.benchmark.serving) "
                "with --output-details"
            )
        _microseconds(ttft, f"ttfts[{index}]")
        intervals = perf_data["itls"][index]
        if not isinstance(intervals, list):
            raise ValueError(f"itls[{index}] must be an array")
        for offset, interval in enumerate(intervals):
            _microseconds(interval, f"itls[{index}][{offset}]")
        for key in ("input_lens", "output_lens"):
            if key in perf_data:
                _nonnegative_number(perf_data[key][index], f"{key}[{index}]")
        if "successes" in perf_data:
            success = perf_data["successes"][index]
            if not isinstance(success, bool):
                raise ValueError(f"successes[{index}] must be a boolean")
        else:
            success = not perf_data["errors"][index] if "errors" in perf_data else True
        if not success:
            skipped["failed"] += 1
            continue
        if start is None:
            skipped["unavailable_start"] += 1
            continue
        if ttft == 0:
            skipped["no_first_output"] += 1
            continue

        included += 1
        # Perfetto normalizes tid=0 to the process's main thread.
        tid = index + 1
        events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": 1,
                "tid": tid,
                "args": {"sort_index": index},
            }
        )
        request_args = {
            "request_index": index,
            "timing_semantics": semantics,
        }
        for key, label in (
            ("input_lens", "input_tokens"),
            ("output_lens", "output_tokens"),
        ):
            if key in perf_data:
                request_args[label] = perf_data[key][index]
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 1,
                "tid": tid,
                "args": {"name": f"Request {index}"},
            }
        )
        timestamp = start + ttft
        start_us = _microseconds(start, f"arrival_times[{index}]")
        end_us = _microseconds(timestamp, f"request {index} cumulative time")
        events.append(
            {
                "name": "TTFT",
                "cat": "client_response",
                "ph": "X",
                "pid": 1,
                "tid": tid,
                "ts": start_us,
                "dur": end_us - start_us,
                "args": request_args,
            }
        )
        for offset, interval in enumerate(intervals, 1):
            # Share rounded boundaries so the viewer cannot round adjacent
            # floating-point timestamps and durations into overlapping slices.
            start_us = end_us
            timestamp += interval
            end_us = _microseconds(timestamp, f"request {index} cumulative time")
            duration_us = end_us - start_us
            event = {
                "name": f"Output interval {offset}",
                "cat": "client_response",
                "ph": "X" if duration_us else "i",
                "pid": 1,
                "tid": tid,
                "ts": start_us,
            }
            if duration_us:
                event["dur"] = duration_us
            else:
                event["s"] = "t"
            events.append(event)
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "timing_semantics": semantics,
            "requests_total": count,
            "requests_included": included,
            "requests_skipped": skipped,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-index",
        type=int,
        default=-1,
        help="Zero-based JSONL record index; negative indexes count from the end (default: -1)",
    )
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_trace.json")
    try:
        if output.resolve() == args.input.resolve() or (
            output.exists() and output.samefile(args.input)
        ):
            raise ValueError("Output path must differ from input path")
        trace = convert_to_chrome_trace(
            load_benchmark_result(args.input, args.run_index)
        )
        output.write_text(json.dumps(trace, allow_nan=False), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    summary = trace["metadata"]
    print(
        f"Wrote {output}: {summary['requests_included']} of "
        f"{summary['requests_total']} requests included; "
        f"skipped {summary['requests_skipped']}"
    )


if __name__ == "__main__":
    main()
