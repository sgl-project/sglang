"""Compact triage entrypoint for SGLang torch-profiler analysis."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import triage_kernel_helpers as kernel_helpers
import triage_overlap_helpers as overlap_helpers
from profile_common import (
    discover_trace_targets,
    load_server_args,
    load_trace_json,
    parse_stage,
    run_profiler,
)

MIN_RENDER_SHARE_PCT = 1.0


def build_triage_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_sglang_torch_profile.py",
        description=(
            "Compact SGLang torch-profiler triage entrypoint. "
            "This prints three tables: kernel mapping, overlap opportunities, "
            "and fuse opportunities. "
            "Use either a single trace/profile input or a mapping+formal two-trace pair."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single trace file or profile directory to triage.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Running SGLang server URL for single-trace triage.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Trace output dir when using --url.",
    )
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default="triage-trace",
        help="Profile prefix when generating a single trace from --url.",
    )
    parser.add_argument(
        "--mapping-input",
        type=str,
        default=None,
        help="Graph-off mapping trace file or directory.",
    )
    parser.add_argument(
        "--mapping-url",
        type=str,
        default=None,
        help="Running graph-off SGLang server URL for the mapping trace.",
    )
    parser.add_argument(
        "--formal-input",
        type=str,
        default=None,
        help="Formal graph-on trace file or directory.",
    )
    parser.add_argument(
        "--formal-url",
        type=str,
        default=None,
        help="Running graph-on SGLang server URL for the formal trace.",
    )
    parser.add_argument(
        "--mapping-output-dir",
        type=str,
        default=None,
        help="Trace output dir when using --mapping-url.",
    )
    parser.add_argument(
        "--formal-output-dir",
        type=str,
        default=None,
        help="Trace output dir when using --formal-url.",
    )
    parser.add_argument(
        "--mapping-profile-prefix",
        type=str,
        default="mapping-trace",
        help="Profile prefix for the mapping trace.",
    )
    parser.add_argument(
        "--formal-profile-prefix",
        type=str,
        default="formal-trace",
        help="Profile prefix for the formal trace.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Profiler steps when generating traces from URLs.",
    )
    parser.add_argument(
        "--profile-by-stage", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--merge-profiles", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--probe-requests", type=int, default=1)
    parser.add_argument(
        "--probe-prompt",
        type=str,
        default=(
            "Repeat the word profiler many times with spaces so the server performs several decode steps. "
            "Do not add explanations."
        ),
    )
    parser.add_argument("--probe-max-new-tokens", type=int, default=None)
    parser.add_argument("--probe-delay", type=float, default=0.5)
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="Pass through to sglang.profiler when generating traces from URLs.",
    )
    parser.add_argument(
        "--pid-substring",
        type=str,
        default=None,
        help="Restrict overlap analysis to PIDs containing this substring.",
    )
    parser.add_argument(
        "--kernel-table-limit",
        type=int,
        default=0,
        help="How many kernel rows to print per stage. Use 0 for all kernels.",
    )
    parser.add_argument(
        "--overlap-table-limit",
        type=int,
        default=0,
        help="How many overlap rows to print per stage. Use 0 for all kernels.",
    )
    return parser


def parse_triage_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_triage_parser()
    args = parser.parse_args(argv)

    single_trace_mode = bool(args.input) or bool(args.url)
    dual_trace_mode = any(
        [
            args.mapping_input,
            args.mapping_url,
            args.formal_input,
            args.formal_url,
        ]
    )

    if single_trace_mode and dual_trace_mode:
        parser.error(
            "Use either single-trace mode (--input/--url) or two-trace mode "
            "(--mapping-* plus --formal-*), not both."
        )

    if single_trace_mode:
        if bool(args.input) == bool(args.url):
            parser.error("Provide exactly one of --input or --url.")
        return args

    if bool(args.mapping_input) == bool(args.mapping_url):
        parser.error("Provide exactly one of --mapping-input or --mapping-url.")
    if bool(args.formal_input) == bool(args.formal_url):
        parser.error("Provide exactly one of --formal-input or --formal-url.")
    return args


def resolve_profile_targets(
    *,
    label: str,
    input_path: Optional[str],
    url: Optional[str],
    output_dir: Optional[str],
    profile_prefix: Optional[str],
    args: argparse.Namespace,
) -> Tuple[List[Path], Optional[dict]]:
    if bool(input_path) == bool(url):
        raise ValueError(f"{label} trace requires exactly one of input path or URL.")

    if url:
        target_dir = run_profiler(
            url=url,
            output_dir=output_dir,
            num_steps=args.num_steps,
            profile_by_stage=args.profile_by_stage,
            merge_profiles=args.merge_profiles,
            profile_prefix=profile_prefix,
            probe_requests=max(0, args.probe_requests),
            probe_prompt=args.probe_prompt,
            probe_max_new_tokens=args.probe_max_new_tokens,
            probe_delay=args.probe_delay,
            start_step=args.start_step,
        )
        traces, server_args = discover_trace_targets(target_dir, all_traces=False)
        return traces, server_args

    resolved = Path(input_path).resolve()
    traces, server_args = discover_trace_targets(resolved, all_traces=False)
    if server_args is None:
        server_args = load_server_args(resolved)
    return traces, server_args


def build_mapping_kernel_map(trace_paths: Sequence[Path]) -> dict:
    stage_site_stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(kernel_helpers.MappingSiteAggregate))
    )
    stage_kernel_categories: Dict[str, Dict[str, str]] = defaultdict(dict)
    global_site_stats = defaultdict(
        lambda: defaultdict(kernel_helpers.MappingSiteAggregate)
    )
    global_kernel_categories: Dict[str, str] = {}

    for trace_path in trace_paths:
        trace = load_trace_json(trace_path)
        kernels, cpu_ops, python_frames, launch_events, _, _ = (
            kernel_helpers.extract_trace_data(trace)
        )
        cpu_ops_by_external_id = kernel_helpers.build_cpu_op_index(cpu_ops)
        launches_by_correlation = kernel_helpers.build_launch_index(launch_events)
        local_site_stats = kernel_helpers.aggregate_kernel_sites(
            kernels,
            cpu_ops_by_external_id,
            python_frames,
            launches_by_correlation=launches_by_correlation,
        )
        stage = parse_stage(trace_path)
        kernel_categories = {
            kernel.canonical_name: kernel.category for kernel in kernels
        }
        kernel_helpers.merge_site_stats(stage_site_stats[stage], local_site_stats)
        kernel_helpers.merge_site_stats(global_site_stats, local_site_stats)
        stage_kernel_categories[stage].update(kernel_categories)
        global_kernel_categories.update(kernel_categories)

    stage_payloads = {
        stage: kernel_helpers.build_stage_payload(
            dict(site_stats), stage_kernel_categories.get(stage, {})
        )
        for stage, site_stats in stage_site_stats.items()
    }
    global_payload = kernel_helpers.build_stage_payload(
        dict(global_site_stats), global_kernel_categories
    )
    return {"stages": stage_payloads, "global": global_payload}


def stage_index(stage: str) -> int:
    return {"extend": 0, "prefill": 0, "decode": 1, "all": 2}.get(stage, 99)


def stage_display(stage: str) -> str:
    return kernel_helpers.stage_label(stage)


def pick_trace_for_stage(stage_to_trace: Dict[str, Path], stage: str) -> Optional[Path]:
    if stage in stage_to_trace:
        return stage_to_trace[stage]
    if "all" in stage_to_trace:
        return stage_to_trace["all"]
    if len(stage_to_trace) == 1:
        return next(iter(stage_to_trace.values()))
    return None


def build_stage_trace_map(trace_paths: Sequence[Path]) -> Dict[str, Path]:
    stage_map: Dict[str, Path] = {}
    for trace_path in sorted(
        trace_paths, key=lambda item: (stage_index(parse_stage(item)), item.name)
    ):
        stage_map[parse_stage(trace_path)] = trace_path
    return stage_map


def render_kernel_table(rows: Sequence[dict]) -> List[str]:
    lines = [
        "| Stage | Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {kernel} | {category} | {gpu_time} | {share:.1f}% | {launches} | {location} | {cpu_op} |".format(
                stage=kernel_helpers.escape_md_cell(stage_display(row["stage"])),
                kernel=kernel_helpers.escape_md_cell(row["kernel"]),
                category=kernel_helpers.escape_md_cell(row["category"]),
                gpu_time=kernel_helpers.format_ms(row["total_us"]),
                share=row["share_pct"],
                launches=row["launches"],
                location=kernel_helpers.escape_md_cell(row["location"]),
                cpu_op=kernel_helpers.escape_md_cell(row["cpu_op"]),
            )
        )
    return lines


def render_overlap_table(rows: Sequence[dict]) -> List[str]:
    lines = [
        "| Stage | Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not rows:
        lines.append(
            "| - | - | - | No actionable overlap rows. Use mapping/formal two-trace triage for stronger overlap conclusions. | - | - | - | - |"
        )
        return lines
    for row in rows:
        formal_signal = (
            f"{row['total_us']:.1f} us, share {row['share_pct']:.1f}%, "
            f"excl {row['exclusive_ratio'] * 100:.1f}% / hid {row['hidden_ratio'] * 100:.1f}%"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    kernel_helpers.escape_md_cell(stage_display(row["stage"])),
                    row["priority"],
                    row["verdict"],
                    kernel_helpers.escape_md_cell(row["kernel"]),
                    kernel_helpers.escape_md_cell(row["python_scope"]),
                    kernel_helpers.escape_md_cell(formal_signal),
                    overlap_helpers.dependency_risk_label(row["dependency_signal"]),
                    row["recommendation"],
                ]
            )
            + " |"
        )
    return lines


def render_fuse_table(rows: Sequence[dict]) -> List[str]:
    lines = [
        "| Stage | Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    if not rows:
        lines.append(
            "| - | No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |"
        )
        return lines
    for row in rows:
        lines.append(
            "| {stage} | {pattern} | {confidence} | {gpu_time} | {share:.1f}% | {evidence} | {current_locations} | {candidate_path} | {rationale} |".format(
                stage=kernel_helpers.escape_md_cell(stage_display(row["stage"])),
                pattern=kernel_helpers.escape_md_cell(row["pattern"]),
                confidence=kernel_helpers.escape_md_cell(row["confidence"]),
                gpu_time=kernel_helpers.format_ms(row["related_us"]),
                share=row["share_pct"],
                evidence=kernel_helpers.escape_md_cell(row["evidence"]),
                current_locations=kernel_helpers.escape_md_cell(
                    row["current_locations"]
                ),
                candidate_path=kernel_helpers.escape_md_cell(row["candidate_path"]),
                rationale=kernel_helpers.escape_md_cell(row["rationale"]),
            )
        )
    return lines


def run_triage(args: argparse.Namespace) -> int:
    single_trace_mode = bool(args.input) or bool(args.url)
    if single_trace_mode:
        formal_traces, formal_server_args = resolve_profile_targets(
            label="input",
            input_path=args.input,
            url=args.url,
            output_dir=args.output_dir,
            profile_prefix=args.profile_prefix,
            args=args,
        )
        mapping_traces = formal_traces
        mapping_server_args = formal_server_args
    else:
        mapping_traces, mapping_server_args = resolve_profile_targets(
            label="mapping",
            input_path=args.mapping_input,
            url=args.mapping_url,
            output_dir=args.mapping_output_dir,
            profile_prefix=args.mapping_profile_prefix,
            args=args,
        )
        formal_traces, formal_server_args = resolve_profile_targets(
            label="formal",
            input_path=args.formal_input,
            url=args.formal_url,
            output_dir=args.formal_output_dir,
            profile_prefix=args.formal_profile_prefix,
            args=args,
        )

    mapping_kernel_map = build_mapping_kernel_map(mapping_traces)

    kernel_rows_rendered: List[dict] = []
    fuse_rows_rendered: List[dict] = []

    for formal_trace in formal_traces:
        trace = load_trace_json(formal_trace)
        kernels, _, _, _, _, _ = kernel_helpers.extract_trace_data(trace)
        if not kernels:
            continue
        stage = parse_stage(formal_trace)
        total_us = sum(kernel.dur for kernel in kernels)
        kernel_stats = kernel_helpers.aggregate(
            kernels, key_fn=lambda item: item.canonical_name
        )
        kernel_categories = {
            kernel.canonical_name: kernel.category for kernel in kernels
        }
        full_kernel_rows = kernel_helpers.build_kernel_rows(
            stage=stage,
            kernel_stats=kernel_stats,
            kernel_categories=kernel_categories,
            local_stage_payload=mapping_kernel_map.get("stages", {}).get(
                stage, {"kernels": {}}
            ),
            external_kernel_map=mapping_kernel_map,
        )
        visible_kernel_rows = kernel_helpers.limit_kernel_rows(
            full_kernel_rows, args.kernel_table_limit
        )
        for row in visible_kernel_rows:
            share_pct = kernel_helpers.pct(row.total_us, total_us)
            if share_pct < MIN_RENDER_SHARE_PCT:
                continue
            kernel_rows_rendered.append(
                {
                    "stage": stage,
                    "kernel": row.name,
                    "category": row.category,
                    "total_us": row.total_us,
                    "share_pct": share_pct,
                    "launches": row.aggregate.count,
                    "location": row.location,
                    "cpu_op": row.cpu_op,
                }
            )
        for item in kernel_helpers.detect_fusion_opportunities(
            stage=stage,
            kernel_rows=full_kernel_rows,
            total_us=total_us,
            server_args=formal_server_args or mapping_server_args,
        ):
            share_pct = kernel_helpers.pct(item.related_us, total_us)
            if share_pct < MIN_RENDER_SHARE_PCT:
                continue
            fuse_rows_rendered.append(
                {
                    "stage": stage,
                    "pattern": item.pattern,
                    "confidence": item.confidence,
                    "related_us": item.related_us,
                    "share_pct": share_pct,
                    "evidence": item.evidence,
                    "current_locations": item.current_locations,
                    "candidate_path": item.candidate_path,
                    "rationale": item.rationale,
                }
            )

    mapping_stage_map = build_stage_trace_map(mapping_traces)
    formal_stage_map = build_stage_trace_map(formal_traces)
    overlap_rows_rendered: List[dict] = []
    if not single_trace_mode:
        for stage in sorted(formal_stage_map, key=stage_index):
            formal_trace = formal_stage_map[stage]
            mapping_trace = pick_trace_for_stage(mapping_stage_map, stage)
            if mapping_trace is None:
                continue
            mapping_trace_json = load_trace_json(mapping_trace)
            mapping_events, mapping_pid = overlap_helpers.extract_kernel_events(
                mapping_trace_json, args.pid_substring
            )
            if not mapping_events:
                continue
            formal_trace_json = load_trace_json(formal_trace)
            formal_events, formal_pid = overlap_helpers.extract_kernel_events(
                formal_trace_json, args.pid_substring
            )
            if not formal_events:
                continue
            mapping_bundle = overlap_helpers.TraceBundle(
                label=f"mapping-{stage}",
                trace_path=mapping_trace,
                server_args=mapping_server_args,
                raw_events=mapping_trace_json.get(
                    "traceEvents",
                    mapping_trace_json if isinstance(mapping_trace_json, list) else [],
                ),
                events=mapping_events,
                pid=mapping_pid,
            )
            formal_bundle = overlap_helpers.TraceBundle(
                label=f"formal-{stage}",
                trace_path=formal_trace,
                server_args=formal_server_args,
                raw_events=formal_trace_json.get(
                    "traceEvents",
                    formal_trace_json if isinstance(formal_trace_json, list) else [],
                ),
                events=formal_events,
                pid=formal_pid,
            )
            formal_bundle.overlap_stats = overlap_helpers.analyze_overlap(
                formal_bundle.events
            )
            aggregates = overlap_helpers.aggregate_events(formal_bundle.events)
            source_map = overlap_helpers.build_kernel_source_map(mapping_bundle)
            stage_rows = overlap_helpers.build_action_rows(
                aggregates,
                source_map,
                formal_bundle.events,
                formal_bundle.overlap_stats["total_busy_us"],
                table_limit=max(0, args.overlap_table_limit),
            )
            for row in stage_rows:
                if row.share_pct < MIN_RENDER_SHARE_PCT:
                    continue
                overlap_rows_rendered.append(
                    {
                        "stage": stage,
                        "priority": row.priority,
                        "verdict": row.verdict,
                        "kernel": row.kernel,
                        "python_scope": row.python_scope,
                        "total_us": row.total_us,
                        "share_pct": row.share_pct,
                        "exclusive_ratio": row.exclusive_ratio,
                        "hidden_ratio": row.hidden_ratio,
                        "dependency_signal": row.dependency_signal,
                        "recommendation": row.recommendation,
                    }
                )

    lines: List[str] = []
    lines.append("Triage View")
    if single_trace_mode:
        lines.append(f"Input traces: {', '.join(str(path) for path in formal_traces)}")
    else:
        lines.append(
            f"Mapping traces: {', '.join(str(path) for path in mapping_traces)}"
        )
        lines.append(f"Formal traces: {', '.join(str(path) for path in formal_traces)}")
    if formal_server_args or mapping_server_args:
        server_args = formal_server_args or mapping_server_args
        model = server_args.get("model_path") or server_args.get("model")
        if model:
            lines.append(f"Model: {model}")
    lines.append("")
    lines.append("Kernel Table")
    lines.extend(render_kernel_table(kernel_rows_rendered))
    lines.append("")
    lines.append("Overlap Opportunity Table")
    lines.extend(render_overlap_table(overlap_rows_rendered))
    lines.append("")
    lines.append("Fuse Opportunity Table")
    lines.extend(render_fuse_table(fuse_rows_rendered))
    print("\n".join(lines).rstrip())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv or sys.argv[1:])
    triage_parser = build_triage_parser()

    if not argv or argv[0] in {"-h", "--help"}:
        triage_parser.print_help()
        return 0

    if argv[0] == "triage":
        argv = argv[1:]
    elif not argv[0].startswith("-"):
        triage_parser.error(
            "This skill now exposes only the compact triage workflow. "
            "Use single-trace mode (--input/--url) or mapping+formal two-trace mode."
        )
        return 2

    return run_triage(parse_triage_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
