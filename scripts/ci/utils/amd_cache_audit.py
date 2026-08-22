#!/usr/bin/env python3
"""Audit the AMD CI model cache: which checkpoints are needed, missing, or dead.

The AMD runners share one HuggingFace cache PVC per cluster, nothing evicts
from it, and it fills up -- a full cache kills a multi-hundred-GB download
partway through with ENOSPC. Deciding what is safe to drop needs the join this
script performs: `register_amd_ci` test registrations -> the checkpoints those
tests name -> whether any AMD workflow actually dispatches them.

Getting that join wrong is expensive in both directions. A 2026-08-20 audit
joined only the two nightly workflows and concluded 939 GiB was reclaimable;
937 GiB of it was live, because 289 of 429 AMD registrations are dispatched
*only* by the per-commit `pr-test-amd*.yml` workflows, which run on
`linux-mi35x-gpu-1` and share the same PVC as the 8-GPU nightly jobs. Deleting
by that list would have broken AMD CI on every PR. So `--reachability` counts
per-commit and nightly separately and never reports one without the other.

The suite -> models half is not reimplemented here: it comes from
`scripts/ci/list_stage_models.py`, which already does the AST extraction,
constant resolution and override handling for any backend.

Usage:
    # What each AMD suite needs, and who dispatches it
    python3 scripts/ci/utils/amd_cache_audit.py

    # Diff against a live cache (run where /sgl-data is mounted)
    python3 scripts/ci/utils/amd_cache_audit.py --cache-dir /sgl-data/hf-cache/hub

    # Same, from a listing captured elsewhere (one `models--org--name` per line)
    ls /sgl-data/hf-cache/hub > hub.txt
    python3 scripts/ci/utils/amd_cache_audit.py --cache-listing hub.txt --json out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Workflows that dispatch AMD suites, split by cadence. Both families mount the
# same per-cluster cache PVC, so both must be consulted before calling a
# checkpoint unused.
NIGHTLY_WORKFLOWS = ("nightly-test-amd.yml", "nightly-test-amd-rocm720.yml")
PER_COMMIT_WORKFLOWS = (
    "pr-test-amd.yml",
    "pr-test-amd-rocm720.yml",
    "pr-test-amd-extra.yml",
)

# HuggingFace stores `org/name` under `models--org--name`.
CACHE_DIR_RE = re.compile(r"^models--(?P<slug>.+)$")


def _load_inventory(
    include_disabled: bool,
) -> Tuple[Dict[str, object], Dict[str, List[str]]]:
    """Reuse list_stage_models.py rather than duplicating its extraction.

    Returns the inventory (suite -> models) plus the suite -> test files map,
    which the inventory itself does not expose in full -- it only carries the
    files whose models could not be resolved.
    """
    path = os.path.join(REPO_ROOT, "scripts", "ci", "list_stage_models.py")
    spec = importlib.util.spec_from_file_location("list_stage_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    overrides = mod.load_overrides(
        os.path.join(REPO_ROOT, "scripts", "ci", "stage_models_overrides.json")
    )
    inventory = mod.build_inventory(
        REPO_ROOT, "amd", overrides, commit="", include_disabled=include_disabled
    )
    suite_files = mod.collect_suite_files(REPO_ROOT, "amd", include_disabled)[0]
    return inventory, suite_files


def _job_runner_labels(job: dict) -> List[str]:
    """`runs-on` as a flat label list, covering the matrix form AMD jobs use."""
    runs_on = job.get("runs-on")
    labels: List[str] = []
    if isinstance(runs_on, str):
        labels.append(runs_on)
    elif isinstance(runs_on, list):
        labels.extend(str(x) for x in runs_on)
    # `runs-on: ${{ matrix.runner }}` with `strategy.matrix.runner: [label, ...]`
    matrix = (
        ((job.get("strategy") or {}).get("matrix") or {})
        if isinstance(job.get("strategy"), dict)
        else {}
    )
    for value in matrix.values():
        if isinstance(value, list):
            labels.extend(str(x) for x in value if isinstance(x, (str, int)))
    return [l for l in labels if "${{" not in l]


def _job_text(job: dict) -> str:
    """Everything a job's steps run, so both dispatch styles are visible."""
    parts: List[str] = []
    for step in job.get("steps") or []:
        if isinstance(step, dict) and isinstance(step.get("run"), str):
            parts.append(step["run"])
    return "\n".join(parts)


def _scan_workflows() -> Dict[str, List[dict]]:
    """suite/file token -> [{workflow, job, cadence, labels}] for AMD workflows.

    Indexed by both `--suite <name>` and bare test filename, because the AMD
    workflows use both: several perf steps call
    `python3 registered/amd/perf/mi35x/<file>.py` directly with no `--suite`,
    and matching only `--suite` would report those checkpoints as unused.
    """
    import yaml

    index: Dict[str, List[dict]] = {}
    families = (
        ("nightly", NIGHTLY_WORKFLOWS),
        ("per-commit", PER_COMMIT_WORKFLOWS),
    )
    for cadence, names in families:
        for name in names:
            path = os.path.join(REPO_ROOT, ".github", "workflows", name)
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh)
            except (OSError, yaml.YAMLError):
                continue
            for job_name, job in (doc.get("jobs") or {}).items():
                if not isinstance(job, dict):
                    continue
                entry = {
                    "workflow": name,
                    "job": job_name,
                    "cadence": cadence,
                    "labels": _job_runner_labels(job),
                }
                text = _job_text(job)
                for token in re.findall(r"--suite\s+(\S+)", text):
                    index.setdefault(token, []).append(entry)
                for token in re.findall(r"([A-Za-z0-9_.-]+\.py)", text):
                    index.setdefault(token, []).append(entry)
    return index


def compute(
    include_disabled: bool = False, runner: Optional[str] = None
) -> Dict[str, object]:
    """Join AMD suites to the jobs that dispatch them, optionally per cluster.

    `runner` is a substring matched against the dispatching job's `runs-on`
    labels (e.g. "mi35x" for the tw cluster, "mi300" for ccs). Scoping matters
    for a cache diff: tw only ever runs the mi35x jobs, so comparing it against
    every AMD suite reports the mi300-only checkpoints as missing.
    """
    inventory, suite_files_map = _load_inventory(include_disabled)
    suites: Dict[str, dict] = inventory.get("suites") or {}  # type: ignore[assignment]
    index = _scan_workflows()

    suite_reach: Dict[str, List[str]] = {}
    suite_jobs: Dict[str, List[str]] = {}
    # model -> {"nightly": [suites], "per-commit": [suites], "undispatched": [suites]}
    model_need: Dict[str, Dict[str, List[str]]] = {}

    for suite, info in suites.items():
        tokens = [suite] + [
            os.path.basename(f) for f in suite_files_map.get(suite) or []
        ]
        entries = [e for t in tokens for e in index.get(t, [])]
        if runner:
            entries = [e for e in entries if any(runner in l for l in e["labels"])]

        where = sorted({e["cadence"] for e in entries})
        suite_reach[suite] = where
        suite_jobs[suite] = sorted({e["job"] for e in entries})

        bucket = where or ["undispatched"]
        for model in info.get("models") or []:
            entry = model_need.setdefault(
                model, {"nightly": [], "per-commit": [], "undispatched": []}
            )
            for w in bucket:
                entry[w].append(suite)

    def verdict(entry: Dict[str, List[str]]) -> str:
        if entry["nightly"]:
            return "nightly"
        if entry["per-commit"]:
            return "per-commit"
        return "undispatched"

    models = {
        model: {
            "verdict": verdict(entry),
            "nightly_suites": sorted(set(entry["nightly"])),
            "per_commit_suites": sorted(set(entry["per-commit"])),
            "undispatched_suites": sorted(set(entry["undispatched"])),
        }
        for model, entry in model_need.items()
    }

    return {
        "suite_count": len(suites),
        "runner_filter": runner,
        "suite_jobs": suite_jobs,
        "suite_reachability": {
            "nightly_only": sorted(
                s for s, w in suite_reach.items() if w == ["nightly"]
            ),
            "per_commit_only": sorted(
                s for s, w in suite_reach.items() if w == ["per-commit"]
            ),
            "both": sorted(s for s, w in suite_reach.items() if len(w) == 2),
            "undispatched": sorted(s for s, w in suite_reach.items() if not w),
        },
        "models": models,
    }


def _read_cache(cache_dir: Optional[str], listing: Optional[str]) -> Set[str]:
    """Cached repo ids, from a live cache dir or a captured `ls` listing."""
    names: List[str] = []
    if cache_dir:
        try:
            names = os.listdir(cache_dir)
        except OSError as exc:
            print(f"ERROR: cannot read {cache_dir}: {exc}", file=sys.stderr)
            return set()
    elif listing:
        with open(listing, encoding="utf-8") as fh:
            names = [line.strip() for line in fh if line.strip()]

    cached: Set[str] = set()
    for name in names:
        match = CACHE_DIR_RE.match(os.path.basename(name.rstrip("/")))
        if match:
            cached.add(match.group("slug").replace("--", "/", 1))
    return cached


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-dir",
        help="Live HF hub cache to diff against, e.g. /sgl-data/hf-cache/hub.",
    )
    parser.add_argument(
        "--cache-listing",
        help="File with one `models--org--name` entry per line, from `ls` elsewhere.",
    )
    parser.add_argument(
        "--runner",
        help="Scope to jobs whose runs-on label contains this substring, e.g. "
        "'mi35x' for the tw cluster or 'mi300' for ccs. Without it every AMD "
        "suite counts, which reports the other cluster's models as missing.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Count tests registered with disabled=. Off by default: a disabled "
        "test does not pull weights, so its models are not needed.",
    )
    parser.add_argument("--json", help="Write the full result as JSON to this path.")
    args = parser.parse_args()

    result = compute(include_disabled=args.include_disabled, runner=args.runner)
    reach = result["suite_reachability"]  # type: ignore[index]
    models: Dict[str, dict] = result["models"]  # type: ignore[assignment]

    print("AMD suite reachability")
    for key in ("nightly_only", "per_commit_only", "both", "undispatched"):
        print(f"  {key:<16} {len(reach[key]):>4} suites")
    needed = {m for m, i in models.items() if i["verdict"] != "undispatched"}
    print(
        f"\n{len(models)} distinct checkpoints referenced; "
        f"{len(needed)} needed by a dispatched suite, "
        f"{len(models) - len(needed)} referenced only by undispatched tests."
    )

    if args.cache_dir or args.cache_listing:
        cached = _read_cache(args.cache_dir, args.cache_listing)
        missing = sorted(needed - cached)
        reclaimable = sorted(cached - set(models))
        dead = sorted(
            c for c in cached if c in models and models[c]["verdict"] == "undispatched"
        )

        print(f"\nCache holds {len(cached)} checkpoints")
        print(f"  needed but MISSING : {len(missing)}")
        for m in missing:
            info = models[m]
            src = ", ".join(info["nightly_suites"] + info["per_commit_suites"])[:70]
            print(f"      {m}  <- {src}")
        print(f"  cached, no AMD test reference (safe to drop): {len(reclaimable)}")
        for m in reclaimable:
            print(f"      {m}")
        print(f"  cached, test exists but no job dispatches it: {len(dead)}")
        for m in dead:
            print(f"      {m}  <- {', '.join(models[m]['undispatched_suites'])[:60]}")
        print(
            "\nOnly the 'no AMD test reference' group is unconditionally safe. The "
            "dead group becomes needed again the moment someone wires its suite up."
        )
        result["cache"] = {
            "cached_count": len(cached),
            "missing": missing,
            "reclaimable": reclaimable,
            "dead_coverage": dead,
        }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
