#!/usr/bin/env python3
"""Unified entrypoint for SGLang torch-profiler analysis workflows."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import analyze_sglang_llm_torch_profile as breakdown_cli
import analyze_sglang_profiler_overlap as overlap_cli
from profile_common import (
    discover_trace_targets,
    load_server_args,
    load_trace_json,
    parse_stage,
    run_profiler,
    write_perfetto_compatible_trace,
)


def build_top_level_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_sglang_torch_profile.py",
        description=(
            "Unified torch-profiler entrypoint for SGLang. "
            "Use `breakdown` for kernel/category share analysis, "
            "`overlap` for two-trace overlap analysis, `triage` for the compact "
            "three-table workflow, or `perfetto-fix` to rewrite a trace into a "
            "more Perfetto-friendly form."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("breakdown", "overlap", "triage", "perfetto-fix"),
        help="Subcommand to run.",
    )
    return parser


def parse_perfetto_fix_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_sglang_torch_profile.py perfetto-fix",
        description="Rewrite a trace so overlapping kernel lanes render more reliably in Perfetto.",
    )
    parser.add_argument(
        "--input", required=True, help="Input trace.json or trace.json.gz path."
    )
    parser.add_argument("--output", default=None, help="Optional output path.")
    return parser.parse_args(argv)


def parse_triage_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_sglang_torch_profile.py triage",
        description=(
            "Run the compact SGLang torch-profiler triage workflow. "
            "This prints three stage-aware tables: kernel mapping, overlap opportunities, "
            "and fuse opportunities."
        ),
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
    args = parser.parse_args(argv)
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
    # The graph-off mapping trace is only used to learn stable
    # kernel -> Python/CPU-op attribution. The final percentages still come from
    # the formal trace.
    stage_site_stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(breakdown_cli.MappingSiteAggregate))
    )
    stage_kernel_categories: Dict[str, Dict[str, str]] = defaultdict(dict)
    global_site_stats = defaultdict(
        lambda: defaultdict(breakdown_cli.MappingSiteAggregate)
    )
    global_kernel_categories: Dict[str, str] = {}

    for trace_path in trace_paths:
        trace = load_trace_json(trace_path)
        kernels, cpu_ops, python_frames, launch_events, _, _ = (
            breakdown_cli.extract_trace_data(trace)
        )
        cpu_ops_by_external_id = breakdown_cli.build_cpu_op_index(cpu_ops)
        launches_by_correlation = breakdown_cli.build_launch_index(launch_events)
        local_site_stats = breakdown_cli.aggregate_kernel_sites(
            kernels,
            cpu_ops_by_external_id,
            python_frames,
            launches_by_correlation=launches_by_correlation,
        )
        stage = parse_stage(trace_path)
        kernel_categories = {
            kernel.canonical_name: kernel.category for kernel in kernels
        }
        breakdown_cli.merge_site_stats(stage_site_stats[stage], local_site_stats)
        breakdown_cli.merge_site_stats(global_site_stats, local_site_stats)
        stage_kernel_categories[stage].update(kernel_categories)
        global_kernel_categories.update(kernel_categories)

    stage_payloads = {
        stage: breakdown_cli.build_stage_payload(
            dict(site_stats), stage_kernel_categories.get(stage, {})
        )
        for stage, site_stats in stage_site_stats.items()
    }
    global_payload = breakdown_cli.build_stage_payload(
        dict(global_site_stats), global_kernel_categories
    )
    return {"stages": stage_payloads, "global": global_payload}


def stage_index(stage: str) -> int:
    return {"extend": 0, "prefill": 0, "decode": 1, "all": 2}.get(stage, 99)


def stage_display(stage: str) -> str:
    return breakdown_cli.stage_label(stage)


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
                stage=breakdown_cli.escape_md_cell(stage_display(row["stage"])),
                kernel=breakdown_cli.escape_md_cell(row["kernel"]),
                category=breakdown_cli.escape_md_cell(row["category"]),
                gpu_time=breakdown_cli.format_ms(row["total_us"]),
                share=row["share_pct"],
                launches=row["launches"],
                location=breakdown_cli.escape_md_cell(row["location"]),
                cpu_op=breakdown_cli.escape_md_cell(row["cpu_op"]),
            )
        )
    return lines


def render_overlap_table(rows: Sequence[dict]) -> List[str]:
    lines = [
        "| Stage | Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        formal_signal = (
            f"{row['total_us']:.1f} us, share {row['share_pct']:.1f}%, "
            f"excl {row['exclusive_ratio'] * 100:.1f}% / hid {row['hidden_ratio'] * 100:.1f}%"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    breakdown_cli.escape_md_cell(stage_display(row["stage"])),
                    row["priority"],
                    row["verdict"],
                    breakdown_cli.escape_md_cell(row["kernel"]),
                    breakdown_cli.escape_md_cell(row["python_scope"]),
                    breakdown_cli.escape_md_cell(formal_signal),
                    overlap_cli.dependency_risk_label(row["dependency_signal"]),
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
                stage=breakdown_cli.escape_md_cell(stage_display(row["stage"])),
                pattern=breakdown_cli.escape_md_cell(row["pattern"]),
                confidence=breakdown_cli.escape_md_cell(row["confidence"]),
                gpu_time=breakdown_cli.format_ms(row["related_us"]),
                share=row["share_pct"],
                evidence=breakdown_cli.escape_md_cell(row["evidence"]),
                current_locations=breakdown_cli.escape_md_cell(
                    row["current_locations"]
                ),
                candidate_path=breakdown_cli.escape_md_cell(row["candidate_path"]),
                rationale=breakdown_cli.escape_md_cell(row["rationale"]),
            )
        )
    return lines


def run_triage(args: argparse.Namespace) -> int:
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
        kernels, _, _, _, _, _ = breakdown_cli.extract_trace_data(trace)
        if not kernels:
            continue
        stage = parse_stage(formal_trace)
        total_us = sum(kernel.dur for kernel in kernels)
        kernel_stats = breakdown_cli.aggregate(
            kernels, key_fn=lambda item: item.canonical_name
        )
        kernel_categories = {
            kernel.canonical_name: kernel.category for kernel in kernels
        }
        full_kernel_rows = breakdown_cli.build_kernel_rows(
            stage=stage,
            kernel_stats=kernel_stats,
            kernel_categories=kernel_categories,
            local_stage_payload=mapping_kernel_map.get("stages", {}).get(
                stage, {"kernels": {}}
            ),
            external_kernel_map=mapping_kernel_map,
        )
        visible_kernel_rows = breakdown_cli.limit_kernel_rows(
            full_kernel_rows, args.kernel_table_limit
        )
        for row in visible_kernel_rows:
            kernel_rows_rendered.append(
                {
                    "stage": stage,
                    "kernel": row.name,
                    "category": row.category,
                    "total_us": row.total_us,
                    "share_pct": breakdown_cli.pct(row.total_us, total_us),
                    "launches": row.aggregate.count,
                    "location": row.location,
                    "cpu_op": row.cpu_op,
                }
            )
        for item in breakdown_cli.detect_fusion_opportunities(
            stage=stage,
            kernel_rows=full_kernel_rows,
            total_us=total_us,
            server_args=formal_server_args or mapping_server_args,
        ):
            fuse_rows_rendered.append(
                {
                    "stage": stage,
                    "pattern": item.pattern,
                    "confidence": item.confidence,
                    "related_us": item.related_us,
                    "share_pct": breakdown_cli.pct(item.related_us, total_us),
                    "evidence": item.evidence,
                    "current_locations": item.current_locations,
                    "candidate_path": item.candidate_path,
                    "rationale": item.rationale,
                }
            )

    mapping_stage_map = build_stage_trace_map(mapping_traces)
    formal_stage_map = build_stage_trace_map(formal_traces)
    overlap_rows_rendered: List[dict] = []
    for stage in sorted(formal_stage_map, key=stage_index):
        formal_trace = formal_stage_map[stage]
        mapping_trace = pick_trace_for_stage(mapping_stage_map, stage)
        if mapping_trace is None:
            continue
        mapping_trace_json = load_trace_json(mapping_trace)
        mapping_events, mapping_pid = overlap_cli.extract_kernel_events(
            mapping_trace_json, args.pid_substring
        )
        if not mapping_events:
            continue
        formal_trace_json = load_trace_json(formal_trace)
        formal_events, formal_pid = overlap_cli.extract_kernel_events(
            formal_trace_json, args.pid_substring
        )
        if not formal_events:
            continue
        mapping_bundle = overlap_cli.TraceBundle(
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
        formal_bundle = overlap_cli.TraceBundle(
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
        formal_bundle.overlap_stats = overlap_cli.analyze_overlap(formal_bundle.events)
        aggregates = overlap_cli.aggregate_events(formal_bundle.events)
        source_map = overlap_cli.build_kernel_source_map(mapping_bundle)
        stage_rows = overlap_cli.build_action_rows(
            aggregates,
            source_map,
            formal_bundle.events,
            formal_bundle.overlap_stats["total_busy_us"],
            table_limit=max(0, args.overlap_table_limit),
        )
        for row in stage_rows:
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
    lines.append(f"Mapping traces: {', '.join(str(path) for path in mapping_traces)}")
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
    top_parser = build_top_level_parser()

    if not argv or argv[0] in {"-h", "--help"}:
        top_parser.print_help()
        return 0

    command = argv[0]
    remainder = argv[1:]

    if command == "breakdown":
        return breakdown_cli.main(remainder)
    if command == "overlap":
        return overlap_cli.main(remainder)
    if command == "triage":
        return run_triage(parse_triage_args(remainder))
    if command == "perfetto-fix":
        args = parse_perfetto_fix_args(remainder)
        output_path = write_perfetto_compatible_trace(
            input_path=Path(args.input),
            output_path=Path(args.output).resolve() if args.output else None,
        )
        print(f"Perfetto-friendly trace written to: {output_path}")
        return 0

    top_parser.error(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
