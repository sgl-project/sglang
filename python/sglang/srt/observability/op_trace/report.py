"""Markdown rendering of op-trace summaries and of the diff of two summaries."""

from __future__ import annotations

from typing import Optional

from sglang.srt.observability.op_trace.schema import OpStat, TraceSummary


def render_summary(summary: TraceSummary, *, top: int) -> str:
    lines = [f"# op trace: `{summary.trace}`", ""]
    for bucket in summary.buckets:
        lines += [
            f"## {bucket.bucket}",
            "",
            f"steps={bucket.num_steps} kernel_sum_p50={bucket.step_kernel_sum_us_p50:.1f}us "
            f"span_p50={bucket.step_span_us_p50:.1f}us "
            f"attributed={bucket.attributed_share:.1%} "
            f"symbol_mismatches={bucket.symbol_mismatches} "
            f"graph_join={bucket.graph_join}",
            "",
            "| op_kind | module | layers | kernels/step | us/step p50 | share | us/layer p50 | top kernel |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        rows = [o for o in summary.ops if o.bucket == bucket.bucket][:top]
        lines += [_op_row(o) for o in rows]
        lines.append("")
    return "\n".join(lines)


def _op_row(o: OpStat) -> str:
    per_layer = f"{o.us_per_layer_p50:.2f}" if o.us_per_layer_p50 is not None else "-"
    return (
        f"| {o.op_kind} | {o.module_class} | {o.num_layers or '-'} | {o.kernels_per_step:.1f} "
        f"| {o.us_per_step_p50:.1f} | {o.share_of_step:.1%} | {per_layer} | {_short(o.top_symbol)} |"
    )


def _short(symbol: str, limit: int = 60) -> str:
    symbol = symbol.removeprefix("void ").split("(")[0].replace("|", "/")
    return symbol if len(symbol) <= limit else symbol[: limit - 3] + "..."


def render_diff(
    *, base: TraceSummary, new: TraceSummary, bucket: Optional[str], top: int
) -> str:
    pairs = _pair_buckets(base=base, new=new, bucket=bucket)
    if not pairs:
        return "no common bucket; pass --bucket or compare summaries of the same shapes"
    lines = []
    for base_bucket, new_bucket in pairs:
        a = {o.op_kind: o for o in base.ops if o.bucket == base_bucket}
        b = {o.op_kind: o for o in new.ops if o.bucket == new_bucket}
        rows = []
        for kind in a.keys() | b.keys():
            ua = a[kind].us_per_step_p50 if kind in a else 0.0
            ub = b[kind].us_per_step_p50 if kind in b else 0.0
            rows.append((kind, ua, ub, ub - ua))
        rows.sort(key=lambda r: -abs(r[3]))
        total_a = _step_total(summary=base, bucket=base_bucket)
        total_b = _step_total(summary=new, bucket=new_bucket)
        lines += [
            f"## {base_bucket} -> {new_bucket}",
            "",
            f"step kernel_sum p50: {total_a:.1f}us -> {total_b:.1f}us ({total_b - total_a:+.1f}us)",
            "",
            "| op_kind | base us/step | new us/step | delta us | delta % |",
            "|---|---:|---:|---:|---:|",
        ]
        for kind, ua, ub, delta in rows[:top]:
            pct = f"{delta / ua:+.1%}" if ua else "new"
            lines.append(f"| {kind} | {ua:.1f} | {ub:.1f} | {delta:+.1f} | {pct} |")
        lines.append("")
    return "\n".join(lines)


def _pair_buckets(
    *, base: TraceSummary, new: TraceSummary, bucket: Optional[str]
) -> list[tuple[str, str]]:
    names_a = [b.bucket for b in base.buckets]
    names_b = [b.bucket for b in new.buckets]
    if bucket is not None:
        return [(bucket, bucket)] if bucket in names_a and bucket in names_b else []
    common = [n for n in names_a if n in names_b]
    if common:
        return [(n, n) for n in common]
    if len(names_a) == 1 and len(names_b) == 1:
        return [(names_a[0], names_b[0])]
    return []


def _step_total(*, summary: TraceSummary, bucket: str) -> float:
    return next(b.step_kernel_sum_us_p50 for b in summary.buckets if b.bucket == bucket)
