"""Op-trace summaries -> benchmark marker records (`marker/record.py` schema).

Each (workload, op kind) becomes one record whose samples are the op's per-step
GPU time, plus a ``<step>`` record for the whole step, so e2e op times are
baselined and compared exactly like kernel benchmark cases.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from sglang.kernels.jit.benchmark.marker.case import Tolerance, make_case_id
from sglang.kernels.jit.benchmark.marker.record import (
    BenchRecord,
    GPUState,
    append_record,
    collect_env_info,
    collect_run_info,
    make_metrics,
)
from sglang.srt.observability.op_trace.analyze import GRAPH_GLUE, UNATTRIBUTED_KINDS
from sglang.srt.observability.op_trace.schema import BucketStat, TraceSummary

STEP_LINE = "<step>"
# Ops below this share of step GPU time are recorded but not gated.
GATE_MIN_SHARE = 0.02
# Below this attributed share of step kernel time the op table is not trusted;
# the clean A/A runs attribute >= 99.1% (decode) and 99.9% (prefill).
MIN_ATTRIBUTED_SHARE = 0.95
# Per workload mode, from a 4-process A/A (GB300, Llama-3.1-8B-Instruct, leave-
# one-out |delta| of each op's p50): decode gated ops max 0.43% / 1.7 us,
# prefill gated ops max 5.75% (whole-step clock shifts); about 4x / 1.4x margin.
DEFAULT_TOLERANCE = {
    "decode": Tolerance(rel=0.02, abs_us=2.0),
    "prefill": Tolerance(rel=0.08, abs_us=20.0),
}


def summary_to_records(
    summary: TraceSummary,
    *,
    bench_id: str,
    params: dict[str, Any],
    world_size: int,
    gpu_before: GPUState,
    gpu_after: GPUState,
    tolerance: Optional[Tolerance] = None,
) -> list[BenchRecord]:
    """Raises when the kernel -> op join is not trustworthy."""
    # case_id has no bucket component, so one summary must be one workload.
    if len(summary.buckets) != 1:
        raise ValueError(
            f"expected one step bucket per workload, got {[b.bucket for b in summary.buckets]}"
        )
    bucket = summary.buckets[0]
    _check_join(bucket)
    lines = [(STEP_LINE, bucket.step_kernel_sum_samples_us, "")]
    lines += [
        (
            op.op_kind,
            op.step_samples_us,
            _ungated_reason(op_kind=op.op_kind, share=op.share_of_step),
        )
        for op in summary.ops
    ]
    tolerance = tolerance or DEFAULT_TOLERANCE[params["mode"]]
    return [
        BenchRecord(
            bench_id=bench_id,
            params=params,
            line=line,
            world_size=world_size,
            case_id=make_case_id(
                bench_id=bench_id, params=params, line=line, world_size=world_size
            ),
            status="ok",
            skip_reason=None,
            metrics=make_metrics([us * 1e-6 for us in samples_us]),
            gbps=None,
            tflops=None,
            gate=not gate_reason,
            gate_reason=gate_reason,
            tolerance=tolerance,
            gpu_before=gpu_before,
            gpu_after=gpu_after,
            env=collect_env_info(
                device_id=torch.cuda.current_device(), source_file=__file__
            ),
            run=collect_run_info(),
            timestamp=time.time(),
        )
        for line, samples_us, gate_reason in lines
    ]


def write_records(records: list[BenchRecord], *, output_dir: str) -> None:
    for record in records:
        append_record(output_dir=output_dir, record=record)


def _check_join(bucket: BucketStat) -> None:
    if bucket.symbol_mismatches > 0:
        raise ValueError(
            f"{bucket.symbol_mismatches} kernel/op-map symbol mismatches in {bucket.bucket}"
        )
    if bucket.attributed_share < MIN_ATTRIBUTED_SHARE:
        raise ValueError(
            f"only {bucket.attributed_share:.1%} of kernel time attributed in "
            f"{bucket.bucket} (< {MIN_ATTRIBUTED_SHARE:.0%})"
        )


def _ungated_reason(*, op_kind: str, share: float) -> str:
    if op_kind in UNATTRIBUTED_KINDS or op_kind == GRAPH_GLUE:
        return "not a model op"
    if share < GATE_MIN_SHARE:
        return f"share {share:.1%} < {GATE_MIN_SHARE:.0%} of step GPU time"
    return ""
