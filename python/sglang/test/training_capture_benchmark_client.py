"""Test-only native benchmark client with post-run capture latency attribution."""

import argparse
import hashlib
import itertools
import json
import math
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np


class RequestRecorder:
    def __init__(self, request):
        self.request = request
        self.prefix = uuid.uuid4().hex
        self.indices = itertools.count()
        self.records = []

    async def __call__(self, request_func_input, pbar=None):
        index = next(self.indices)
        rid = f"capture-benchmark-{self.prefix}-{index}"
        request = replace(
            request_func_input,
            extra_request_body={**request_func_input.extra_request_body, "rid": rid},
        )
        output = await self.request(request, pbar)
        self.records.append(
            {
                "request_index": index,
                "trace_id": hashlib.sha256(rid.encode()).hexdigest(),
                "start_time": output.start_time,
                "latency": output.latency,
                "ttft": output.ttft,
                "itl": list(output.itl),
                "prompt_len": output.prompt_len,
                "output_len": output.output_len,
                "success": output.success,
                "error": output.error,
                "cached_tokens": output.cached_tokens,
            }
        )
        return output


def summarize_requests(records, published_trace_ids, *, count, output_len):
    """Join validated Store manifests to exact native-client latency fields."""
    if len(records) != count or count < 1 or output_len < 2:
        raise ValueError("request records do not cover the measured workload")
    if sorted(row["request_index"] for row in records) != list(range(count)):
        raise ValueError("request indices must cover each measured request once")
    trace_ids = [row["trace_id"] for row in records]
    if len(set(trace_ids)) != count or not published_trace_ids <= set(trace_ids):
        raise ValueError("publication trace IDs do not match unique measured requests")
    for row in records:
        if (
            not row["success"]
            or row["error"]
            or row["output_len"] != output_len
            or not all(
                math.isfinite(row[field]) for field in ("start_time", "latency", "ttft")
            )
            or not 0 < row["ttft"] <= row["latency"]
            or any(not math.isfinite(value) or value < 0 for value in row["itl"])
        ):
            raise ValueError("invalid or incomplete measured request")
    origin = min(row["start_time"] for row in records)
    rows = [
        {
            "request_index": row["request_index"],
            "trace_id": row["trace_id"],
            "published": row["trace_id"] in published_trace_ids,
            "start_offset_ms": (row["start_time"] - origin) * 1000,
            "ttft_ms": row["ttft"] * 1000,
            "e2e_latency_ms": row["latency"] * 1000,
            "tpot_ms": (row["latency"] - row["ttft"]) * 1000 / (output_len - 1),
            "max_itl_ms": max(row["itl"], default=0) * 1000,
        }
        for row in records
    ]
    groups = {}
    for name, selected in (
        ("all", rows),
        ("published", [row for row in rows if row["published"]]),
        ("not_published", [row for row in rows if not row["published"]]),
    ):
        metrics = {"requests": len(selected)}
        if selected:
            for field in ("ttft_ms", "tpot_ms", "e2e_latency_ms", "max_itl_ms"):
                values = [row[field] for row in selected]
                metrics["mean_" + field] = float(np.mean(values))
                for label, percentile in (("median", 50), ("p95", 95), ("p99", 99)):
                    metrics[label + "_" + field] = float(
                        np.percentile(values, percentile)
                    )
        groups[name] = metrics
    return {
        "published_trace_ids": sorted(published_trace_ids),
        "groups": groups,
        "worst_ttft": sorted(rows, key=lambda row: row["ttft_ms"], reverse=True)[:10],
        "worst_tpot": sorted(rows, key=lambda row: row["tpot_ms"], reverse=True)[:10],
        "scope": "Publication identity is joined after validated Store readback. Not-published includes any skipped or failed capture; membership alone does not attribute latency causally. E2E/TPOT use native-client completion timing, including final stream messages.",
    }


def main():
    from sglang.benchmark import serving

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--capture-request-records", type=Path, required=True)
    parser.add_argument("--backend", choices=["sglang"], required=True)
    args, remaining = parser.parse_known_args()
    original = serving.ASYNC_REQUEST_FUNCS["sglang"]
    recorder = RequestRecorder(original)
    report = {"schema_version": 1, "status": "running"}
    argv = sys.argv
    with args.capture_request_records.open("x") as stream:
        try:
            serving.ASYNC_REQUEST_FUNCS["sglang"] = recorder
            sys.argv = [argv[0], "--backend", "sglang", *remaining]
            serving.cli_main()
            report["status"] = "completed"
        except BaseException:
            report["status"] = "failed"
            raise
        finally:
            serving.ASYNC_REQUEST_FUNCS["sglang"] = original
            sys.argv = argv
            report["requests"] = sorted(
                recorder.records, key=lambda row: row["request_index"]
            )
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
