"""Fail-closed E2E evidence checks; CPU-only and independent of runtime imports."""

import argparse
import json
import math
import os
import re
from pathlib import Path

from pr_report import runner
from select_pr_e2e import (
    ROOT,
    GitHub,
    check_live,
    output,
    phase,
    read_json,
    validate_plan,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def entry_for(plan, name):
    entries = [entry for entry in plan["entries"] if entry["id"] == name]
    require(len(entries) == 1, "Entry is not uniquely selected")
    return entries[0]


def identity(plan, entry):
    return {
        "id": entry["id"],
        "plan_hash": plan["plan_hash"],
        "tested_sha": plan["tested_sha"],
        "run_id": plan["run_id"],
        "attempt": plan["attempt"],
        "config_digest": entry["config_digest"],
    }


def duration(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def orchestration_metrics(outputs):
    result = {"github_queue_seconds": None, "pod_startup_seconds": None}
    for key in (
        "duration_seconds",
        "queue_duration_seconds",
        "occupied_ppu_card_hours",
        "execution_ppu_card_hours",
    ):
        try:
            value = float(outputs.get(key, ""))
        except (TypeError, ValueError):
            value = None
        result[key] = value if duration(value) else None
    return result


def verify_pytest(directory, entry):
    report = read_json(directory / "summary.json")
    selected = ["/workspace/source/" + name for name in entry["files"]]
    require(
        report.get("success") is True and report.get("selected") == selected,
        "Missing or incorrect registered-file selection",
    )
    require(not report.get("allowed_skips"), "Unapproved skip policy")
    records = report["files"]
    require([r["file"] for r in records] == selected, "Missing/duplicate file result")
    xmls = {path.name for path in directory.glob("*.xml")}
    require(
        xmls == {r["junit"] for r in records} and len(xmls) == len(selected),
        "Missing/extra per-file JUnit",
    )
    total = 0
    for record in records:
        require(
            record.get("passed") is True and record.get("returncode") == 0,
            "Failed pytest file",
        )
        for key in ("junit", "log"):
            name = record[key]
            require(
                Path(name).name == name and (directory / name).is_file(),
                "Missing/unsafe evidence path",
            )
        counts = runner.inspect_junit(directory / record["junit"])
        require(
            all(record.get(k) == v for k, v in counts.items()),
            "JUnit/count disagreement",
        )
        require(duration(record.get("elapsed")), "Missing test timing")
        total += counts["executed"]
    return total


def verify_evaluation(directory, entry):
    report = read_json(directory / "results/result.json")
    config = read_json(ROOT / entry["config"])
    summary = report["summary"]
    provenance = report["provenance"]
    require(
        provenance.get("test_config_id") == entry["id"]
        and provenance.get("test_config_sha256") == entry["config_digest"],
        "Evaluator config identity mismatch",
    )
    require(
        type(summary.get("total")) is int and summary["total"] > 0,
        "Evaluator produced no cases/measurements",
    )
    if entry["category"] == "answer":
        require(
            summary.get("gate", {}).get("regressed") is False,
            "Answer regression gate failed or is missing",
        )
        dataset = read_json(
            ROOT / "test/registered/ppu/answer_eval" / config["evaluation"]["dataset"]
        )
        require(
            [c["case_id"] for c in report["cases"]]
            == [c["id"] for c in dataset["cases"]],
            "Answer case inventory mismatch",
        )
        require(summary["total"] == len(report["cases"]), "Answer count mismatch")
        quality = "Answer regression gate (configured known failures retained)"
    else:
        require(
            report.get("test_id") == entry["id"]
            and report.get("config_digest") == entry["config_digest"],
            "Measurement identity mismatch",
        )
        measurements = report["measurements"]
        require(
            summary.get("verdict") == "passed"
            and summary.get("failed") == 0
            and summary["total"] == len(measurements)
            and summary.get("measured") == len(measurements),
            "Failed/incomplete measurements",
        )
        require(
            all(m.get("status") == "measured" for m in measurements),
            "Failed/unmeasured measurement",
        )
        if entry["category"] == "accuracy":
            require(
                len(measurements) == 1,
                "Accuracy execution must report one reviewed dataset",
            )
            quality = "Accuracy smoke" if entry["mode"] == "smoke" else "Accuracy full"
            if config["evaluation"].get("baseline") is None:
                quality += " (measurement-only; no regression baseline)"
        else:
            require(
                [m["id"] for m in measurements]
                == [m["id"] for m in config["workload"]["measurements"]],
                "Performance measurement inventory mismatch",
            )
            quality = "Performance measurement-only (no regression threshold)"
    require((directory / "results/junit.xml").is_file(), "Missing evaluator JUnit")
    if entry["category"] != "answer":
        counts = runner.inspect_junit(directory / "results/junit.xml")
        require(
            counts["executed"] == summary["total"], "Evaluator JUnit count mismatch"
        )
    gated = entry["category"] == "answer" or (
        entry["category"] == "accuracy"
        and config["evaluation"].get("baseline") is not None
    )
    result = {
        "count": summary["total"],
        "quality": quality,
        "quality_gate": "passed" if gated else "not_configured",
    }
    if entry["category"] == "accuracy":
        result["score"] = report["measurements"][0].get("score")
        result["samples"] = report["measurements"][0].get("samples")
    return result


def verify(directory, plan, entry):
    directory = Path(directory)
    expected_ranks = {f"rank-{rank}" for rank in range(entry["nodes"])}
    require(
        {p.name for p in directory.glob("rank-*")} == expected_ranks,
        "Missing/extra worker rank",
    )
    config = read_json(ROOT / entry["config"])
    ranks = []
    images = set()
    models = set()
    for rank in range(entry["nodes"]):
        rank_dir = directory / f"rank-{rank}"
        worker = read_json(rank_dir / "worker.json")
        require(
            all(worker.get(k) == v for k, v in identity(plan, entry).items()),
            "Stale worker identity",
        )
        require(
            worker.get("rank") == rank
            and worker.get("exit_code") == 0
            and duration(worker.get("elapsed")),
            "Missing/failed rank completion",
        )
        environment = read_json(rank_dir / "environment.json")
        require(
            environment.get("commit") == plan["tested_sha"], "Wrong tested checkout"
        )
        require(
            all(environment.get(k) for k in ("python", "torch", "checkout_import")),
            "Incomplete environment",
        )
        require(
            environment["checkout_import"].startswith("/workspace/source/python/"),
            "Wrong import source",
        )
        image_digest = environment.get("digest", "")
        require(
            re.fullmatch(r"sha256:[a-f0-9]{64}", image_digest)
            and environment.get("image", "").endswith("@" + image_digest),
            "Unpinned worker image",
        )
        images.add(environment["image"])
        require(
            len(environment.get("devices", []))
            == len(config["hardware"]["visible_devices"]),
            "Device count mismatch",
        )
        model = environment.get("model", {})
        require(
            model.get("path") == config["model"]["path"]
            and model.get("weights")
            and re.fullmatch(r"[a-f0-9]{64}", model.get("metadata_digest", "")),
            "Incomplete model identity",
        )
        models.add(json.dumps(model, sort_keys=True))
        require(
            duration(environment.get("setup_seconds"))
            and (rank_dir / "worker.log").is_file(),
            "Missing setup evidence",
        )
        tests = verify_pytest(rank_dir / "pytest", entry)
        ranks.append(
            {
                "rank": rank,
                "role": "evaluator" if rank == 0 else "serving peer",
                "pytest_executed": tests,
                "setup_seconds": environment["setup_seconds"],
                "elapsed": worker["elapsed"],
            }
        )
    require(len(images) == 1, "Ranks used different images")
    require(len(models) == 1, "Ranks used different checkpoints")
    return {
        **identity(plan, entry),
        "success": True,
        "ranks": ranks,
        "allocation": {k: entry[k] for k in ("nodes", "cards_per_node", "board")},
        "hardware": config["hardware"],
        "evaluation": verify_evaluation(directory, entry),
    }


def verify_terminal(record, plan, entry):
    require(
        all(record.get(k) == v for k, v in identity(plan, entry).items()),
        "Stale terminal evidence",
    )
    require(
        record.get("success") is True,
        "Selected entry did not pass evidence verification",
    )
    require(
        [r["rank"] for r in record.get("ranks", [])] == list(range(entry["nodes"])),
        "Missing terminal rank",
    )
    require(
        record.get("evaluation", {}).get("count", 0) > 0, "Empty terminal evaluation"
    )


def aggregate(directory, plan):
    paths = list(Path(directory).rglob("entry.json"))
    records = [read_json(path) for path in paths]
    require(
        sorted(r["id"] for r in records) == sorted(e["id"] for e in plan["entries"]),
        "Selected-ID evidence incomplete, duplicated, or unexpected",
    )
    for record in records:
        verify_terminal(record, plan, entry_for(plan, record["id"]))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify", "aggregate", "outer"])
    parser.add_argument("--plan", default="selection.json")
    parser.add_argument("--entry")
    parser.add_argument("--directory", default="e2e-results")
    parser.add_argument("--destination", default="entry.json")
    args = parser.parse_args()
    plan = read_json(args.plan)
    validate_plan(plan)
    if args.mode == "outer":
        check_live(plan, GitHub(), bool(plan["entries"]), require_latest=True)
        needs = json.loads(os.environ["NEEDS_JSON"])
        require(needs["e2e-selection"]["result"] == "success", "Selection failed")
        if plan["entries"]:
            require(
                needs["ci"]["result"] == "success"
                and needs["ci"].get("outputs", {}).get("e2e_evidence_ok") == "true",
                "E2E workflow did not finish",
            )
        print("E2E gate passed" if plan["entries"] else "No E2E selectors active")
        return
    if args.mode == "verify":
        entry = entry_for(plan, args.entry)
        result = {**identity(plan, entry), "success": False}
        try:
            require(
                os.environ.get("WORKER_OUTCOME") == "success",
                "Worker action failed or did not run",
            )
            result = verify(args.directory, plan, entry)
        except Exception as error:
            result["error"] = runner.sanitize(str(error))
            raise
        finally:
            result["orchestration"] = orchestration_metrics(
                json.loads(os.environ.get("ACTION_OUTPUTS", "{}"))
            )
            Path(args.destination).write_text(
                runner.sanitize(json.dumps(result, indent=2)) + "\n"
            )
        print(json.dumps(result))
    else:
        check_live(plan, GitHub())
        needs = json.loads(os.environ["NEEDS_JSON"])
        require(needs["prepare"]["result"] == "success", "E2E preparation failed")
        for name in ("default", "extended", "distributed"):
            expected = (
                "success"
                if any(phase(e) == name for e in plan["entries"])
                else "skipped"
            )
            require(
                needs[name]["result"] == expected,
                f"Phase {name} did not finish: expected {expected}",
            )
        records = aggregate(args.directory, plan)
        text = "\n".join(f"- {r['id']}: {r['evaluation']['quality']}" for r in records)
        print(text)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
                stream.write("## Verified PPU E2E executions\n" + text + "\n")
        output("evidence_ok", "true")


if __name__ == "__main__":
    main()
