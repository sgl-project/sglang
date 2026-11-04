#!/usr/bin/env python3
"""Recompute lifecycle fixed-work results and check the archived task ledger."""

import argparse
import gzip
import json
from pathlib import Path

from lifecycle_bench import (
    arm_summary,
    cases_for_arm,
    logical_working_set,
    output_equal,
    resume_prompt,
    summarize,
    write_json,
)
from session_kv_bench import digest


def open_text(path):
    return path.open() if path.exists() else gzip.open(str(path) + ".gz", "rt")


def load_json(path):
    with open_text(path) as source:
        return json.load(source)


def analyze(directory):
    root = Path(directory)
    manifest = load_json(root / "lifecycle-manifest.json")
    args = argparse.Namespace(**manifest["args"])
    expected = {item["id"]: item for lane in manifest["fixed_plan"] for item in lane}
    controls, references, publications, errors = {}, {}, set(), []
    source_records, source_phases, source_calls = {}, {}, {}
    schedule = {spec["id"]: spec for spec in manifest["schedule"]}
    with open_text(root / "lifecycle-raw.jsonl") as source:
        for line in source:
            record = json.loads(line)
            if record.get("kind") == "lifecycle_control":
                identity = record["id"]
                if identity in controls:
                    errors.append("duplicate control result: " + identity)
                controls[identity] = {
                    key: record.get(key)
                    for key in ("errors", "quiescent", "completed_monotonic")
                }
            elif record.get("kind") == "lifecycle_publication":
                publications.add(record["id"])
                arm_id = record["id"].rsplit("-source-", 1)[0]
                spec = schedule[arm_id]
                key = (spec["block"], record["case"])
                response = record["generation"]
                origin = record.get("reused_from")
                if origin:
                    original = source_records.get(origin)
                    if (
                        not getattr(args, "reuse_published_sources", False)
                        or original is None
                        or original.get("reused_from")
                        or not record.get("reference_only")
                        or "publication" in record
                        or "initial_generation" in record
                        or any(
                            record.get(field) != original.get(field)
                            for field in ("case", "prefix_sha256", "cache_salt")
                        )
                        or schedule[origin.rsplit("-source-", 1)[0]]["block"]
                        != spec["block"]
                        or not output_equal(original["generation"], response)
                    ):
                        errors.append(
                            "invalid reused source provenance: " + record["id"]
                        )
                elif getattr(args, "reuse_published_sources", False) and (
                    record.get("reference_only")
                    or not record.get("publication")
                    or not record.get("initial_generation")
                ):
                    errors.append("missing actual source publication: " + record["id"])
                source_records[record["id"]] = record
                if not output_equal(references.setdefault(key, response), response):
                    errors.append("source output differs: " + record["id"])
            elif record.get("kind") == "lifecycle_source_phase":
                source_phases[record["id"]] = record
            elif record.get("kind") == "generate" and record.get("origin") in (
                "publication",
                "publication_reference",
            ):
                arm_id = record["trial"].rsplit("-source-", 1)[0]
                source_calls[arm_id] = source_calls.get(arm_id, 0) + 1
    expected_publications = {
        spec["id"] + "-source-" + str(case)
        for spec in manifest["schedule"]
        for case in range(args.cases)
    }
    if publications != expected_publications:
        errors.append("source publication ledger differs from complete plan")
    if len(controls) != len(expected) * len(schedule):
        errors.append("asynchronous control ledger count differs from complete plan")
    arms = []
    for spec in manifest["schedule"]:
        path = root / (spec["id"] + ".json")
        if not path.exists() and not Path(str(path) + ".gz").exists():
            errors.append("missing arm: " + spec["id"])
            continue
        recorded = load_json(path)
        rows = recorded["rows"]
        if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
            errors.append("task plan differs: " + spec["id"])
        cases = cases_for_arm(args, manifest["input_material"], spec)
        if getattr(args, "reuse_published_sources", False):
            phase = source_phases.get(spec["id"], {})
            origin = phase.get("source_reused_from")
            shared = source_phases.get(origin, phase)
            expected_calls = 0 if origin else args.cases * 2
            if (
                not phase
                or shared.get("source_reused_from")
                or source_calls.get(spec["id"], 0) != expected_calls
                or phase.get("source_generation_calls") != expected_calls
                or (origin and phase.get("source_execution_wall_s") != 0)
                or phase.get("source_allocation_parts") != len(args.arms)
                or abs(
                    recorded["source_publication_s"]
                    - shared.get("source_execution_wall_s", -1) / len(args.arms)
                )
                > 1e-9
                or any(
                    recorded.get(key) != value
                    for key, value in phase.items()
                    if key
                    not in ("kind", "id", "wall_time", "monotonic", "allocated_wall_s")
                )
            ):
                errors.append(
                    "source execution/allocation ledger differs: " + spec["id"]
                )
            for case in cases:
                record = source_records.get(
                    spec["id"] + "-source-" + str(case["id"]), {}
                )
                if (
                    record.get("prefix_sha256") != digest(case["prefix"])
                    or record.get("cache_salt") != case["salt"]
                    or record.get("reused_from")
                    != (origin + "-source-" + str(case["id"]) if origin else None)
                ):
                    errors.append("source scope/identity differs: " + spec["id"])
        for row in rows:
            planned = expected.get(row["id"])
            if planned is None:
                continue
            if any(row.get(key) != value for key, value in planned.items()):
                errors.append("replay fields differ: " + row["trial"])
            if row["input_sha256"] != digest(resume_prompt(cases[planned["case"]])):
                errors.append("input digest differs: " + row["trial"])
            control = controls.get(row["trial"])
            if control is None:
                errors.append("missing asynchronous control result: " + row["trial"])
            elif (
                not control["quiescent"]
                or control["errors"]
                or control["completed_monotonic"] > recorded["completed_monotonic"]
            ):
                errors.append("control error or outside final drain: " + row["trial"])
            if row.get("generation"):
                reply = row["generation"]
                if reply["meta_info"].get("completion_tokens") != args.output_tokens:
                    errors.append("fixed output count differs: " + row["trial"])
        wall = recorded["completed_monotonic"] - recorded["started_monotonic"]
        if abs(wall - recorded["wall_s"]) > 1e-6:
            errors.append("arm wall interval differs: " + spec["id"])
        if (
            logical_working_set(cases, args.page_size)
            != recorded["logical_working_set_tokens"]
        ):
            errors.append("logical working set differs: " + spec["id"])
        recomputed = arm_summary(
            spec,
            rows,
            recorded["wall_s"],
            recorded["drain"],
            args,
            recorded["source_publication_s"],
            references,
        )
        for key in (
            "planned_tasks",
            "expected_effective_tasks",
            "completed",
            "effective_completed",
            "effective_tasks_per_s",
            "repeated_prefill_tokens",
            "valid_tasks",
        ):
            if recomputed[key] != recorded[key]:
                errors.append("recomputed " + key + " differs: " + spec["id"])
        arms.append({**recorded, **recomputed})
    result = summarize(arms)
    result["verification"] = {
        "errors": errors,
        "verified": not errors,
        "archived_controls": len(controls),
        "expected_controls": len(expected) * len(schedule),
        "source_reference_cases": len(references),
    }
    result["acceptance_passed"] = (
        not errors
        and len(arms) == len(schedule)
        and all(
            arm["completed"] == arm["expected_effective_tasks"]
            and arm["effective_completed"] == arm["expected_effective_tasks"]
            and arm["valid_tasks"] == arm["planned_tasks"]
            and arm["meets_minimum_duration"]
            and arm["meets_minimum_resumes"]
            and not arm["drain"]["errors"]
            for arm in arms
        )
    )
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results", required=True)
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    result = analyze(args.results)
    write_json(Path(args.output), result)
    print(
        json.dumps(
            {"acceptance_passed": result["acceptance_passed"], **result["verification"]}
        )
    )
    return 0 if result["acceptance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
