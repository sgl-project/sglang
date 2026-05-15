"""Compact triage entrypoint for unified LLM torch-profiler analysis."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import triage_kernel_helpers as kernel_helpers
import triage_overlap_helpers as overlap_helpers
from profile_common import (
    DEFAULT_DECODE_INPUT_LEN,
    DEFAULT_DECODE_OUTPUT_LEN,
    DEFAULT_PREFILL_INPUT_LEN,
    DEFAULT_PREFILL_OUTPUT_LEN,
    DEFAULT_WARMUP_STEPS,
    PROFILE_WORKLOAD_CHOICES,
    discover_trace_targets,
    framework_display_name,
    load_server_args,
    load_trace_json,
    parse_stage,
    resolve_framework,
    run_profiler,
)

MIN_RENDER_SHARE_PCT = 1.0
MAPPING_KERNEL_SAMPLE_LIMIT_PER_NAME = 16


def build_triage_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_llm_torch_profile.py",
        description=(
            "Compact LLM torch-profiler triage entrypoint for SGLang, vLLM, and "
            "TensorRT-LLM. "
            "This prints three tables: kernel mapping, overlap opportunities, "
            "and fuse opportunities. "
            "Use either a single trace/profile input or a mapping+formal two-trace pair."
        ),
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="auto",
        choices=["auto", "sglang", "vllm", "trtllm", "tllm", "tensorrt-llm"],
        help=(
            "Serving framework. Use auto to detect from trace contents, path hints, "
            "or URL features."
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
        help=(
            "Running server URL for single-trace triage. SGLang supports direct "
            "capture via sglang.profiler. vLLM and TensorRT-LLM require a server-side "
            "torch-profiler output path exposed via --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Trace output dir when using --url. For vLLM this should match the "
            "server's torch_profiler_dir. For TensorRT-LLM it should match the "
            "directory or file path configured by TLLM_TORCH_PROFILE_TRACE."
        ),
    )
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default="triage-trace",
        help=(
            "Profile prefix when generating a trace from --url. SGLang uses it "
            "directly; vLLM and TensorRT-LLM may ignore it on the HTTP profiler path."
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
        help="Running graph-off server URL for the mapping trace.",
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
        help="Running graph-on server URL for the formal trace.",
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
        help="Active profiler steps when generating traces from URLs.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Warmup steps to run before arming the profiler for URL capture.",
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
        "--profile-workload",
        choices=PROFILE_WORKLOAD_CHOICES,
        default="both",
        help=(
            "Live-capture workload shape. Default 'both' captures separate "
            "prefill and decode profiles instead of one mixed request. Use "
            "'legacy' to keep the old --probe-prompt behavior."
        ),
    )
    parser.add_argument(
        "--prefill-input-len",
        type=int,
        default=DEFAULT_PREFILL_INPUT_LEN,
        help="Synthetic input length for the prefill profile workload.",
    )
    parser.add_argument(
        "--prefill-output-len",
        type=int,
        default=DEFAULT_PREFILL_OUTPUT_LEN,
        help="Output length for the prefill profile workload.",
    )
    parser.add_argument(
        "--decode-input-len",
        type=int,
        default=DEFAULT_DECODE_INPUT_LEN,
        help="Synthetic input length for the decode profile workload.",
    )
    parser.add_argument(
        "--decode-output-len",
        type=int,
        default=DEFAULT_DECODE_OUTPUT_LEN,
        help="Output length for the decode profile workload.",
    )
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
) -> Tuple[List[Path], Optional[dict], str]:
    if bool(input_path) == bool(url):
        raise ValueError(f"{label} trace requires exactly one of input path or URL.")

    if url:
        framework = resolve_framework(
            args.framework,
            input_path=Path(output_dir).resolve() if output_dir else None,
            url=url,
        )
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
            warmup_steps=args.warmup_steps,
            start_step=args.start_step,
            framework=framework,
            framework_hint_path=output_dir,
            profile_workload=args.profile_workload,
            prefill_input_len=args.prefill_input_len,
            prefill_output_len=args.prefill_output_len,
            decode_input_len=args.decode_input_len,
            decode_output_len=args.decode_output_len,
        )
        traces, server_args = discover_trace_targets(target_dir, all_traces=False)
        resolved_framework = resolve_framework(
            args.framework,
            input_path=target_dir,
            url=url,
            server_args=server_args,
        )
        return traces, server_args, resolved_framework

    resolved = Path(input_path).resolve()
    traces, server_args = discover_trace_targets(resolved, all_traces=False)
    if server_args is None:
        server_args = load_server_args(resolved)
    framework = resolve_framework(
        args.framework, input_path=resolved, server_args=server_args
    )
    return traces, server_args, framework


def build_mapping_kernel_map(trace_paths: Sequence[Path], framework: str) -> dict:
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
        if not kernels:
            continue
        cpu_ops_by_external_id = kernel_helpers.build_cpu_op_index(cpu_ops)
        launches_by_correlation = kernel_helpers.build_launch_index(launch_events)
        site_context_cache = {}
        default_stage = parse_stage(trace_path)
        for stage, stage_kernels in kernel_helpers.group_kernels_by_stage(
            kernels, default_stage
        ).items():
            sampled_stage_kernels = (
                stage_kernels
                if framework == "sglang"
                else sample_kernels_for_mapping(stage_kernels)
            )
            local_site_stats = kernel_helpers.aggregate_kernel_sites(
                sampled_stage_kernels,
                cpu_ops_by_external_id,
                python_frames,
                launches_by_correlation=launches_by_correlation,
                site_context_cache=site_context_cache,
            )
            kernel_categories = {
                kernel.canonical_name: kernel.category for kernel in stage_kernels
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


def sample_kernels_for_mapping(
    kernels: Sequence[kernel_helpers.KernelEvent],
    per_name_limit: int = MAPPING_KERNEL_SAMPLE_LIMIT_PER_NAME,
) -> List[kernel_helpers.KernelEvent]:
    if per_name_limit <= 0:
        return list(kernels)

    grouped: Dict[str, List[kernel_helpers.KernelEvent]] = defaultdict(list)
    for kernel in kernels:
        grouped[kernel.canonical_name].append(kernel)

    sampled: List[kernel_helpers.KernelEvent] = []
    for kernel_name in sorted(grouped):
        items = grouped[kernel_name]
        if len(items) <= per_name_limit:
            sampled.extend(items)
            continue
        for sample_idx in range(per_name_limit):
            pos = round(sample_idx * (len(items) - 1) / (per_name_limit - 1))
            sampled.append(items[pos])
    sampled.sort(key=lambda kernel: (kernel.ts, kernel.name))
    return sampled


def stage_display(stage: str) -> str:
    return kernel_helpers.stage_label(stage)


def pick_stage_value(stage_to_value: Dict[str, object], stage: str) -> Optional[object]:
    if stage in stage_to_value:
        return stage_to_value[stage]
    if "all" in stage_to_value:
        return stage_to_value["all"]
    if len(stage_to_value) == 1:
        return next(iter(stage_to_value.values()))
    return None


def render_stages(stage_to_value: Dict[str, object]) -> List[str]:
    stages = set(stage_to_value)
    if any(stage != "all" for stage in stages):
        stages.discard("all")
    return sorted(stages, key=stage_index)


def build_overlap_stage_bundle_map(
    trace_paths: Sequence[Path],
    *,
    label_prefix: str,
    server_args: Optional[dict],
    pid_substring: Optional[str],
) -> Dict[str, overlap_helpers.TraceBundle]:
    stage_bundles: Dict[str, overlap_helpers.TraceBundle] = {}
    for trace_path in sorted(
        trace_paths, key=lambda item: (stage_index(parse_stage(item)), item.name)
    ):
        trace_json = load_trace_json(trace_path)
        raw_events = trace_json.get(
            "traceEvents",
            trace_json if isinstance(trace_json, list) else [],
        )
        events, pid = overlap_helpers.extract_kernel_events(trace_json, pid_substring)
        if not events:
            continue
        default_stage = parse_stage(trace_path)
        stage_groups = overlap_helpers.group_events_by_stage(events, default_stage)
        for stage in render_stages(stage_groups):
            if stage in stage_bundles:
                continue
            stage_bundles[stage] = overlap_helpers.TraceBundle(
                label=f"{label_prefix}-{stage}",
                trace_path=trace_path,
                server_args=server_args,
                raw_events=raw_events,
                events=stage_groups[stage],
                pid=pid,
            )
        if "all" in stage_groups and not stage_bundles:
            stage_bundles["all"] = overlap_helpers.TraceBundle(
                label=f"{label_prefix}-all",
                trace_path=trace_path,
                server_args=server_args,
                raw_events=raw_events,
                events=stage_groups["all"],
                pid=pid,
            )
    return stage_bundles


def group_rows_by_stage(rows: Sequence[dict]) -> List[Tuple[str, List[dict]]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stage") or "all")].append(row)
    return [
        (stage, grouped[stage]) for stage in sorted(grouped.keys(), key=stage_index)
    ]


def render_kernel_table_for_stage(rows: Sequence[dict]) -> List[str]:
    lines = [
        "| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    if not rows:
        lines.append(
            "| No kernel rows at or above 1.0% share. | - | - | - | - | - | - |"
        )
        return lines
    for row in rows:
        lines.append(
            "| {kernel} | {category} | {gpu_time} | {share:.1f}% | {launches} | {location} | {cpu_op} |".format(
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


def render_stage_section_tables(
    rows: Sequence[dict],
    *,
    render_stage_fn,
    stage_label_prefix: str = "#####",
) -> List[str]:
    if not rows:
        return render_stage_fn([])
    stage_groups = group_rows_by_stage(rows)
    if len(stage_groups) == 1 and stage_groups[0][0] == "all":
        return render_stage_fn(stage_groups[0][1])

    lines: List[str] = []
    for index, (stage, stage_rows) in enumerate(stage_groups):
        lines.append(f"{stage_label_prefix} {stage_display(stage)}")
        lines.extend(render_stage_fn(stage_rows))
        if index != len(stage_groups) - 1:
            lines.append("")
    return lines


def render_kernel_tables(rows: Sequence[dict]) -> List[str]:
    return render_stage_section_tables(
        rows, render_stage_fn=render_kernel_table_for_stage
    )


def render_overlap_table_for_stage(rows: Sequence[dict]) -> List[str]:
    lines = [
        "| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not rows:
        lines.append(
            "| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |"
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


def render_overlap_tables(rows: Sequence[dict]) -> List[str]:
    return render_stage_section_tables(
        rows,
        render_stage_fn=render_overlap_table_for_stage,
    )


def render_fuse_table_for_stage(rows: Sequence[dict]) -> List[str]:
    lines = [
        "| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    if not rows:
        lines.append(
            "| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |"
        )
        return lines
    for row in rows:
        lines.append(
            "| {pattern} | {confidence} | {gpu_time} | {share:.1f}% | {evidence} | {current_locations} | {candidate_path} | {rationale} |".format(
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


def render_fuse_tables(rows: Sequence[dict]) -> List[str]:
    return render_stage_section_tables(
        rows,
        render_stage_fn=render_fuse_table_for_stage,
    )


def run_triage(args: argparse.Namespace) -> int:
    single_trace_mode = bool(args.input) or bool(args.url)
    if single_trace_mode:
        formal_traces, formal_server_args, formal_framework = resolve_profile_targets(
            label="input",
            input_path=args.input,
            url=args.url,
            output_dir=args.output_dir,
            profile_prefix=args.profile_prefix,
            args=args,
        )
        mapping_traces = formal_traces
        mapping_server_args = formal_server_args
        mapping_framework = formal_framework
    else:
        mapping_traces, mapping_server_args, mapping_framework = (
            resolve_profile_targets(
                label="mapping",
                input_path=args.mapping_input,
                url=args.mapping_url,
                output_dir=args.mapping_output_dir,
                profile_prefix=args.mapping_profile_prefix,
                args=args,
            )
        )
        formal_traces, formal_server_args, formal_framework = resolve_profile_targets(
            label="formal",
            input_path=args.formal_input,
            url=args.formal_url,
            output_dir=args.formal_output_dir,
            profile_prefix=args.formal_profile_prefix,
            args=args,
        )

    mapping_kernel_map = build_mapping_kernel_map(mapping_traces, mapping_framework)

    kernel_rows_rendered: List[dict] = []
    fuse_rows_rendered: List[dict] = []
    formal_stage_payloads: Dict[str, dict] = {}

    for formal_trace in formal_traces:
        trace = load_trace_json(formal_trace)
        kernels, cpu_ops, python_frames, launch_events, _, _ = (
            kernel_helpers.extract_trace_data(trace)
        )
        if not kernels:
            continue
        default_stage = parse_stage(formal_trace)
        stage_groups = kernel_helpers.group_kernels_by_stage(kernels, default_stage)
        formal_cpu_ops_by_external_id = kernel_helpers.build_cpu_op_index(cpu_ops)
        formal_launches_by_correlation = kernel_helpers.build_launch_index(
            launch_events
        )
        formal_site_context_cache = {}
        for stage_name, stage_kernels in stage_groups.items():
            local_site_stats = kernel_helpers.aggregate_kernel_sites(
                stage_kernels,
                formal_cpu_ops_by_external_id,
                python_frames,
                launches_by_correlation=formal_launches_by_correlation,
                site_context_cache=formal_site_context_cache,
            )
            formal_stage_payloads[stage_name] = kernel_helpers.build_stage_payload(
                local_site_stats,
                {kernel.canonical_name: kernel.category for kernel in stage_kernels},
            )
        trace_total_us = sum(kernel.dur for kernel in kernels)
        for stage in sorted(stage_groups, key=stage_index):
            stage_kernels = stage_groups[stage]
            if not stage_kernels:
                continue
            total_us = sum(kernel.dur for kernel in stage_kernels)
            if (
                stage == "all"
                and default_stage == "all"
                and kernel_helpers.pct(total_us, trace_total_us) < MIN_RENDER_SHARE_PCT
            ):
                continue
            kernel_stats = kernel_helpers.aggregate(
                stage_kernels, key_fn=lambda item: item.canonical_name
            )
            kernel_categories = {
                kernel.canonical_name: kernel.category for kernel in stage_kernels
            }
            full_kernel_rows = kernel_helpers.build_kernel_rows(
                stage=stage,
                kernel_stats=kernel_stats,
                kernel_categories=kernel_categories,
                local_stage_payload=formal_stage_payloads.get(stage, {"kernels": {}}),
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
                kernel_rows=full_kernel_rows,
                total_us=total_us,
                server_args=formal_server_args or mapping_server_args,
                framework=formal_framework,
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

    overlap_rows_rendered: List[dict] = []
    if not single_trace_mode:
        mapping_overlap_bundles = build_overlap_stage_bundle_map(
            mapping_traces,
            label_prefix="mapping",
            server_args=mapping_server_args,
            pid_substring=args.pid_substring,
        )
        formal_overlap_bundles = build_overlap_stage_bundle_map(
            formal_traces,
            label_prefix="formal",
            server_args=formal_server_args,
            pid_substring=args.pid_substring,
        )
        for stage in render_stages(formal_overlap_bundles):
            formal_bundle = pick_stage_value(formal_overlap_bundles, stage)
            mapping_bundle = pick_stage_value(mapping_overlap_bundles, stage)
            if formal_bundle is None or mapping_bundle is None:
                continue
            formal_bundle.overlap_stats = overlap_helpers.analyze_overlap(
                formal_bundle.events
            )
            aggregates = overlap_helpers.aggregate_events(formal_bundle.events)
            source_map = overlap_helpers.build_kernel_source_map(
                mapping_bundle,
                kernel_map_entry_lookup=lambda stage_name, kernel_name: (
                    kernel_helpers.lookup_kernel_map_entry(
                        mapping_kernel_map, stage_name, kernel_name
                    )
                    if mapping_kernel_map
                    else None
                ),
                stage=stage,
            )
            source_map = overlap_helpers.merge_source_map_from_kernel_payload(
                source_map,
                pick_stage_value(formal_stage_payloads, stage),
            )
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
    lines.append(f"Mode: {'single-trace' if single_trace_mode else 'mapping-formal'}")
    if single_trace_mode:
        lines.append(f"Framework: {framework_display_name(formal_framework)}")
        lines.append(f"Input traces: {', '.join(str(path) for path in formal_traces)}")
    else:
        if mapping_framework == formal_framework:
            lines.append(f"Framework: {framework_display_name(formal_framework)}")
        else:
            lines.append(
                f"Mapping framework: {framework_display_name(mapping_framework)}"
            )
            lines.append(
                f"Formal framework: {framework_display_name(formal_framework)}"
            )
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
    lines.extend(render_kernel_tables(kernel_rows_rendered))
    lines.append("")
    lines.append("Overlap Opportunity Table")
    lines.extend(render_overlap_tables(overlap_rows_rendered))
    lines.append("")
    lines.append("Fuse Opportunity Table")
    lines.extend(render_fuse_tables(fuse_rows_rendered))
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
            "This skill exposes only the triage workflow. "
            "Use single-trace mode (--input/--url) or mapping+formal two-trace mode."
        )
        return 2

    return run_triage(parse_triage_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
