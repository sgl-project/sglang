#!/usr/bin/env python3
"""
Bucket coverage-recommended tests per CI suite.

Runs after test_selector.py has produced recommended_pytest_paths.txt (one
pytest path per line, repo-relative, optional ::case suffix). For each
requested suite, enumerate the suite's registered test files via list_tests.py
and intersect with the recommended files, then write per-suite GitHub Actions
step outputs:

    only_tests_<key>   multi-line list of intersected files (repo-relative);
                       exported non-empty ONLY in precision mode, so callers
                       can reference it without any extra guard expression
    should_run_<key>   'false' when precision mode is active, the suite
                       enumerated a non-empty registered pool, and this
                       suite's bucket is empty -> the caller skips the whole
                       job; also 'false' for ALL suites when nothing was
                       recommended and the changed-file list is available;
                       'true' otherwise (incl. full suites). A suite forced
                       full by the multimodal safety gate stays 'true'
                       regardless of both skip paths.

Suite keys replace '-' with '_' because GitHub Actions output names must be
valid shell identifiers. Any failure path yields empty/missing outputs, which
callers treat as "run the full suite".

Decision model (precision vs full)
----------------------------------
Precision mode has the highest priority: when --precision-mode is not 'true',
every suite runs in full, no matter what was recommended. Otherwise a full run
is triggered by only two signals:

  - a changed path in _FULL_RUN_PATHS (the pipeline configuration files),
  - a missing changed-file list (test_selector.py failed),

plus the multimodal baseline safety gate described below.

Everything else is decided by the recommendation: an empty one skips every
suite, a non-empty one runs only the buckets. Precision mode is therefore not
keyed off "is any bucket non-empty" — a recommendation that no scheduled suite
contains (a base-a CPU unit test, a nightly-only test, a deleted test file) is
not runnable in this workflow, and a full run would not run it either. This is
what keeps a PR that only edits a CPU unit test from re-running the whole a3
matrix.

Diffusion (multimodal_gen) suites are enumerated from the NPU diffusion case
config instead of test/registered, and they recommend at test-case
granularity: one file holds several expensive cases, so a file-level list
would be equivalent to running the whole suite. ``--diffusion-suites`` takes
``job-name:suite`` pairs, e.g.
``multimodal-gen-test-1-npu-a3:1-npu,multimodal-gen-test-4-npu-a3:2-npu``.
Their ``only_tests_<key>`` lines are pytest ``-k`` tokens rather than paths:

    [<case_id>]            parametrized diffusion case
    <standalone file stem> standalone diffusion test file

Safety gate: if the coverage baseline (``--case-map``) has no multimodal_gen
entry, the diffusion buckets are forced to full-suite. Without that gate a
baseline built before multimodal collection existed would make every
multimodal bucket empty, and precision mode would silently skip the diffusion
cases. The same full-suite fallback applies per suite when the baseline is
missing any of that suite's cases or standalone files: a newly added case has
no coverage-map key by definition, so no PR could ever recommend it until the
baseline is rebuilt. A forced suite ignores BOTH the precision-mode skip and
the skip-all rule below — "the baseline cannot judge this suite" is not the
same as "no test is affected".

When --github-summary points at $GITHUB_STEP_SUMMARY, a markdown section is
appended with the recommended test cases, counts, per-suite buckets, and the
files that matched no suite pool (not run by the bucketed a3 suites).
--report-file writes the same section to a plain file (e.g. for artifact
upload, so the report is downloadable from the run page).

Usage:
    python3 bucket_recommended_tests.py \
        --recommended recommended_pytest_paths.txt \
        --changed-files pr_changed_files.txt \
        --suites base-a-test-npu,base-b-test-1-npu-a3 \
        --diffusion-suites multimodal-gen-test-1-npu-a3:1-npu,multimodal-gen-test-4-npu-a3:2-npu \
        --case-map test_case_map.json \
        --precision-mode true \
        --github-output "$GITHUB_OUTPUT" \
        --github-summary "$GITHUB_STEP_SUMMARY" \
        --report-file recommended-tests-report.md
"""

import argparse
import ast
import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Repo layout: this script lives at <repo>/scripts/ci/npu/precise-test/.
SCRIPT_DIR = Path(__file__).resolve().parent
LIST_TESTS = SCRIPT_DIR / "list_tests.py"
REPO_ROOT = SCRIPT_DIR.parents[3]

DIFFUSION_UTILS_DIR = REPO_ROOT / "scripts" / "ci" / "utils" / "diffusion"
# Any coverage-map key under this prefix proves the baseline includes
# multimodal_gen cases (see the safety gate above).
DIFFUSION_TEST_PREFIX = "python/sglang/multimodal_gen/test/"
# Standalone files in that config are relative to the server/ test root.
DIFFUSION_TEST_ROOT_REL = DIFFUSION_TEST_PREFIX + "server"
DIFFUSION_CASE_CONFIG = (
    REPO_ROOT / DIFFUSION_TEST_ROOT_REL / "ascend/testcase_configs_npu.py"
)

# diffusion_case_parser reads this at import time to resolve ONE_NPU_CASES /
# TWO_NPU_CASES instead of the GPU case lists.
os.environ["USE_NPU_CONFIGS"] = "true"
sys.path.insert(0, str(DIFFUSION_UTILS_DIR))

from diffusion_case_parser import (  # noqa: E402
    parse_standalone_data,
    parse_testcase_configs,
)

# Trailing "[case_id]" of a parametrized pytest node id.
_CASE_ID_RE = re.compile(r"\[([^\[\]]+)\]$")


def normalize_rel(path: str) -> str:
    """Normalize a pytest path entry to a repo-relative posix file path.

    Strips an optional ::case suffix and the leading ./, converts absolute
    paths (any runner topology) to repo-relative.
    """
    p = path.strip().replace("\\", "/").removeprefix("./")
    p = p.split("::", 1)[0]
    if os.path.isabs(p):
        p = os.path.relpath(p, REPO_ROOT).replace(os.sep, "/")
    return p


# The only changes that force a full run. They are exactly the non-test paths
# that trigger this workflow's paths filter and shape its environment or its
# selection logic, so their effect cannot be judged from the coverage baseline
# (and a selector change must not be validated by the selector's own output).
_FULL_RUN_PATHS = {
    ".github/workflows/pr-test-npu.yml",
    "python/pyproject_npu.toml",
    "scripts/ci/npu/npu_ci_install_dependency.sh",
}


def needs_full_run(path: str) -> bool:
    """True when this changed path must not be judged by the recommendation."""
    return path.strip().replace("\\", "/").removeprefix("./") in _FULL_RUN_PATHS


def enumerate_suite_files(suite: str, work_dir: Path) -> list:
    """List the suite's registered NPU PR-mode test files via list_tests.py."""
    out_file = work_dir / (suite.replace("-", "_") + ".txt")
    subprocess.run(
        [
            sys.executable,
            str(LIST_TESTS),
            "--hw",
            "npu",
            "--suite",
            suite,
            "-o",
            str(out_file),
        ],
        check=True,
    )
    with open(out_file) as f:
        return [line.strip() for line in f if line.strip()]


@dataclass
class DiffusionPool:
    """Recommendation pool of one diffusion suite."""

    cases_by_file: dict  # repo-relative test file -> [case_id, ...]
    standalone_files: list  # repo-relative standalone test files

    @property
    def size(self) -> int:
        return sum(len(ids) for ids in self.cases_by_file.values()) + len(
            self.standalone_files
        )

    @property
    def case_ids(self) -> list:
        return [cid for ids in self.cases_by_file.values() for cid in ids]

    def files(self) -> set:
        return set(self.cases_by_file) | set(self.standalone_files)


def _diffusion_rel(file_rel: str) -> str:
    """Resolve a config-relative test file to a repo-relative posix path."""
    return posixpath.normpath(
        posixpath.join(DIFFUSION_TEST_ROOT_REL, file_rel.replace("\\", "/"))
    )


def _parse_parametrized_groups(case_config: Path) -> dict:
    """Parse PARAMETRIZED_CASE_GROUPS into suite -> [(file, case list var)].

    diffusion_case_parser reports case ids and standalone files but not which
    file a case lives in, which is needed to expand file-level
    recommendations. The mapping dict is parsed here with AST only, so this
    stays import-free of torch.
    """
    tree = ast.parse(case_config.read_text(encoding="utf-8"), filename=str(case_config))
    groups = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "PARAMETRIZED_CASE_GROUPS" not in targets or not isinstance(
            node.value, ast.Dict
        ):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if not isinstance(value, ast.List):
                continue
            entries = []
            for elt in value.elts:
                if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
                    continue
                file_node, cases_node = elt.elts
                if isinstance(file_node, ast.Constant) and isinstance(
                    cases_node, ast.Name
                ):
                    entries.append((file_node.value, cases_node.id))
            groups[key.value] = entries
    return groups


def collect_diffusion_pools() -> dict:
    """Enumerate the NPU diffusion suites and their recommendation pools."""
    case_lists = parse_testcase_configs(DIFFUSION_CASE_CONFIG)
    standalone_files, _ = parse_standalone_data(DIFFUSION_CASE_CONFIG)
    parametrized_groups = _parse_parametrized_groups(DIFFUSION_CASE_CONFIG)
    # A suite may have only standalone files, so enumerate the union instead
    # of the parametrized groups alone (a missing pool means "run full").
    pools = {}
    for suite in sorted(set(parametrized_groups) | set(standalone_files)):
        cases_by_file = {}
        for file_rel, list_name in parametrized_groups.get(suite, []):
            rel = _diffusion_rel(file_rel)
            case_ids = cases_by_file.setdefault(rel, [])
            for case_id in case_lists.get(list_name, []):
                if case_id not in case_ids:
                    case_ids.append(case_id)
        pools[suite] = DiffusionPool(
            cases_by_file=cases_by_file,
            standalone_files=[
                _diffusion_rel(f) for f in standalone_files.get(suite, [])
            ],
        )
    _warn_standalone_token_collisions(pools)
    return pools


def _warn_standalone_token_collisions(pools: dict) -> None:
    """Warn when a standalone -k token is not unique inside the diffusion run.

    Standalone files are selected by pytest module-name keyword, so two files
    sharing a stem (or a stem that is a substring of a case id) would make one
    token select more than intended. Over-running is harmless, hence a warning
    instead of an error.
    """
    stems = {}
    case_ids = set()
    for suite, pool in pools.items():
        case_ids.update(pool.case_ids)
        for rel in pool.standalone_files:
            stems.setdefault(Path(rel).stem, []).append(f"{suite}:{rel}")
    for stem, owners in sorted(stems.items()):
        if len(owners) > 1:
            print(
                f"::warning::Standalone -k token {stem!r} covers multiple files: "
                f"{', '.join(owners)}"
            )
        hits = sorted(case_id for case_id in case_ids if stem in case_id)
        if hits:
            print(
                f"::warning::Standalone -k token {stem!r} also matches case "
                f"id(s): {', '.join(hits)}"
            )


def match_diffusion_pool(pool: DiffusionPool, raw_entries: list) -> tuple:
    """Match recommended entries against one diffusion pool.

    A file-level entry (a changed test file, or a whole-file coverage target)
    selects every case of that file; a case-level entry selects just that
    case. Standalone files are selected as whole files.
    """
    selected_cases = []
    selected_standalones = []
    for entry in raw_entries:
        rel = normalize_rel(entry)
        _, _, node = entry.strip().replace("\\", "/").partition("::")
        if rel in pool.standalone_files:
            if rel not in selected_standalones:
                selected_standalones.append(rel)
            continue
        case_ids = pool.cases_by_file.get(rel)
        if case_ids is None:
            continue
        match = _CASE_ID_RE.search(node)
        if match:
            if match.group(1) in case_ids and match.group(1) not in selected_cases:
                selected_cases.append(match.group(1))
        else:
            for case_id in case_ids:
                if case_id not in selected_cases:
                    selected_cases.append(case_id)
    return selected_cases, selected_standalones


def diffusion_tokens(selected_cases: list, selected_standalones: list) -> list:
    """Render a diffusion bucket as pytest -k tokens.

    Bracket the case id so ``[x]`` cannot match ``[x_v2]``. Standalone files
    are matched by module-name keyword (the runner applies -k to pytest, whose
    keywords include the module name).
    """
    tokens = [f"[{case_id}]" for case_id in selected_cases]
    tokens.extend(Path(rel).stem for rel in selected_standalones)
    return tokens


def load_multimodal_map_keys(case_map: Path):
    """Return the coverage baseline's multimodal_gen keys, or None.

    None means "the baseline cannot be trusted for multimodal" (missing,
    unreadable, or built before multimodal collection existed); callers then
    run the diffusion suites in full.
    """
    if not case_map.is_file():
        print(f"::warning::Test case map not found: {case_map}")
        return None
    try:
        with open(case_map, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"::warning::Cannot read test case map {case_map}: {exc}")
        return None
    if not isinstance(data, dict):
        print(f"::warning::Unexpected test case map format: {case_map}")
        return None
    keys = {key for key in data if key.startswith(DIFFUSION_TEST_PREFIX)}
    if not keys:
        print(
            "::warning::Test case map has no multimodal_gen entries; the diffusion "
            "suites must run in full. Rebuild the coverage baseline."
        )
        return None
    return keys


def missing_from_baseline(pool: DiffusionPool, map_keys: set) -> list:
    """Pool entries that the coverage baseline never collected.

    A case or standalone file added after the last coverage run has no map
    key, so no PR diff can ever recommend it. Such a suite must run in full
    until the baseline is rebuilt.
    """
    covered_cases = set()
    for key in map_keys:
        match = _CASE_ID_RE.search(key)
        if match:
            covered_cases.add(match.group(1))
    missing = [cid for cid in pool.case_ids if cid not in covered_cases]
    missing.extend(f for f in pool.standalone_files if f not in map_keys)
    return missing


def write_github_summary(
    target_files: list,
    raw_entries: list,
    recommended: set,
    buckets: dict,
    suite_pool_sizes: dict,
    force_full: set,
    baseline_gaps: dict,
    precision_mode: bool,
    skip_all_suites: bool = False,
    decision: str = "",
    decision_reason: str = "",
    full_run_paths: list | None = None,
    unknown_recommended: list | None = None,
) -> None:
    """Append a markdown section with the recommendation results to every
    target file: $GITHUB_STEP_SUMMARY and/or a plain report file."""
    full_run_paths = full_run_paths or []
    unknown_recommended = unknown_recommended or []
    lines = []
    lines.append("### Recommended tests from coverage")
    lines.append("")
    if decision:
        lines.append(f"**Decision**: `{decision}` — {decision_reason}")
        lines.append("")
    if precision_mode:
        lines.append(
            "Precision mode **on** — a3 suites run only the bucketed subset below."
        )
    elif skip_all_suites and not force_full:
        lines.append("**All suites skipped**: the changes recommend no test.")
    else:
        lines.append("Precision mode **off** — suites run in full.")
    if full_run_paths:
        lines.append("")
        lines.append(
            "**Full run forced — pipeline configuration changed** "
            "(not judgeable from coverage):"
        )
        lines.append("")
        lines.extend(f"- `{p}`" for p in full_run_paths)
    lines.append("")
    lines.append(
        f"Recommended **{len(raw_entries)}** test cases "
        f"({len(recommended)} unique files); "
        f"{len(unknown_recommended)} are not run by this workflow."
    )
    lines.append("")
    lines.append("| Suite | Registered | Bucketed | Action |")
    lines.append("|---|---|---|---|")
    for suite, files in buckets.items():
        if suite in force_full:
            action = "run full suite (safety gate)"
        elif skip_all_suites or (
            precision_mode and suite_pool_sizes[suite] and not files
        ):
            action = "skipped (no recommended test)"
        elif precision_mode and files:
            action = "run subset"
        else:
            action = "run full suite"
        lines.append(
            f"| {suite} | {suite_pool_sizes[suite]} | {len(files)} | {action} |"
        )
    if baseline_gaps:
        lines.append("")
        lines.append("**Full suite forced — baseline gaps**")
        lines.append("")
        for suite, missing in baseline_gaps.items():
            listed = ", ".join(f"`{item}`" for item in missing)
            lines.append(f"- `{suite}`: {listed}")
    if raw_entries:
        lines.append("")
        lines.append(
            f"<details><summary>Recommended test cases ({len(raw_entries)})</summary>"
        )
        lines.append("")
        lines.append("```")
        lines.extend(raw_entries)
        lines.append("```")
        lines.append("</details>")
    if unknown_recommended:
        lines.append("")
        lines.append(
            f"<details><summary>Not run by this workflow ({len(unknown_recommended)})</summary>"
        )
        lines.append("")
        lines.append("```")
        lines.extend(unknown_recommended)
        lines.append("```")
        lines.append("")
        lines.append(
            "Recommended but not registered in any suite this workflow runs "
            "(base-a / CPU / nightly suites belong to other pipelines, or the "
            "file was deleted). Running the full suites would not run them "
            "either."
        )
        lines.append("</details>")
    if any(buckets.values()):
        lines.append("")
        lines.append("<details><summary>Bucketed tests per suite</summary>")
        lines.append("")
        for suite, files in buckets.items():
            lines.append(f"**{suite} ({len(files)})**")
            lines.append("")
            lines.extend(f"- {f}" for f in files)
            lines.append("")
        lines.append("</details>")
    for target in target_files:
        with open(target, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Summary written to: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Bucket coverage-recommended tests per CI suite."
    )
    parser.add_argument(
        "--recommended",
        type=str,
        default=str(SCRIPT_DIR / "recommended_pytest_paths.txt"),
        help="recommended_pytest_paths.txt written by test_selector.py",
    )
    parser.add_argument(
        "--suites",
        type=str,
        required=True,
        help="Comma-separated suite names to bucket (e.g. base-a-test-npu,base-b-test-1-npu-a3)",
    )
    parser.add_argument(
        "--diffusion-suites",
        type=str,
        default="",
        help="Comma-separated job:suite pairs for multimodal_gen suites "
        "(e.g. multimodal-gen-test-1-npu-a3:1-npu,multimodal-gen-test-4-npu-a3:2-npu)",
    )
    parser.add_argument(
        "--case-map",
        type=str,
        default="",
        help="test_case_map.json used as the multimodal safety gate",
    )
    parser.add_argument(
        "--precision-mode",
        type=str,
        default="false",
        help="'true' to enable precision mode (run only the recommended tests). "
        "Any other value has the highest priority and forces every suite to "
        "run in full",
    )
    parser.add_argument(
        "--changed-files",
        type=str,
        default="",
        help="pr_changed_files.txt written by test_selector.py. A path listed "
        "in _FULL_RUN_PATHS (pr-test-npu.yml, pyproject_npu.toml, "
        "npu_ci_install_dependency.sh) forces a full run; a missing or empty "
        "file forces it too, and everything else is decided by the "
        "recommendation",
    )
    parser.add_argument(
        "--github-output",
        type=str,
        default="",
        help="GITHUB_OUTPUT file to write step outputs to (empty = print only)",
    )
    parser.add_argument(
        "--github-summary",
        type=str,
        default="",
        help="GITHUB_STEP_SUMMARY file to append a markdown report to (empty = skip)",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="",
        help="Also write the markdown report to this file, e.g. for artifact "
        "upload (empty = skip)",
    )
    args = parser.parse_args()

    # Recommended set: file-level, repo-relative. raw_entries keeps the
    # original case-level lines (with optional ::case suffix) for the summary.
    rec_file = Path(args.recommended)
    recommended = set()
    raw_entries = []
    if rec_file.exists():
        with open(rec_file) as f:
            raw_entries = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("ERROR")
            ]
            recommended = {normalize_rel(line) for line in raw_entries}
    else:
        print(f"::warning::Recommended file not found: {rec_file}")
    print(f"Recommended test entries: {len(raw_entries)} ({len(recommended)} files)")

    # Only the pipeline configuration paths force a full run; everything else
    # is decided by the coverage recommendation (or the missing-list fallback).
    changed_files: list[str] = []
    if args.changed_files:
        cf = Path(args.changed_files)
        if cf.exists():
            changed_files = [
                ln.strip() for ln in cf.read_text().splitlines() if ln.strip()
            ]
    full_run_paths = sorted(p for p in changed_files if needs_full_run(p))

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    diffusion_suites = {}
    for item in args.diffusion_suites.split(","):
        item = item.strip()
        if not item:
            continue
        job, sep, suite = item.partition(":")
        if not sep or not job or not suite:
            parser.error(f"--diffusion-suites entry must be job:suite, got {item!r}")
        diffusion_suites[job] = suite
    work_dir = Path(tempfile.mkdtemp(prefix="bucket_tests_"))

    buckets = {}
    # Registered-pool size per suite. An empty ENUMERATION (0 files) must
    # never let precision mode skip the whole job — that would silently drop
    # the suite from the PR gate — so the output loop falls back to a
    # full-suite run in that case.
    suite_pool_sizes = {}
    # Suites that must run in full even when precision mode is on (safety-gate
    # failures, unreadable configs). A forced suite also opts out of the
    # skip-all rule: an incomplete baseline cannot prove it is unaffected.
    force_full = set()
    # job -> tests absent from the coverage baseline (reported in the summary).
    baseline_gaps = {}
    all_pool = set()
    for suite in suites:
        suite_files = {normalize_rel(f) for f in enumerate_suite_files(suite, work_dir)}
        suite_pool_sizes[suite] = len(suite_files)
        all_pool |= suite_files
        if not suite_files:
            print(f"::warning::Suite {suite}: enumerated 0 registered test files")
        intersection = sorted(suite_files & recommended)
        buckets[suite] = intersection
        print(
            f"Suite {suite}: {len(suite_files)} files, {len(intersection)} recommended"
        )

    if diffusion_suites:
        map_keys = (
            load_multimodal_map_keys(Path(args.case_map)) if args.case_map else None
        )
        if map_keys is None:
            force_full |= set(diffusion_suites)
        try:
            diffusion_pools = collect_diffusion_pools()
        except Exception as exc:
            print(f"::warning::Cannot enumerate diffusion suites: {exc!r}")
            diffusion_pools = {}
            force_full |= set(diffusion_suites)
        for job, suite in diffusion_suites.items():
            pool = diffusion_pools.get(suite)
            if pool is None:
                print(
                    f"::warning::Diffusion suite {suite} ({job}) not found in "
                    f"{DIFFUSION_CASE_CONFIG.name}"
                )
                buckets[job] = []
                suite_pool_sizes[job] = 0
                force_full.add(job)
                continue
            suite_pool_sizes[job] = pool.size
            all_pool |= pool.files()
            if map_keys is None:
                buckets[job] = []
                print(
                    f"Suite {job} (diffusion {suite}): {pool.size} cases, "
                    "forced full suite by the safety gate"
                )
                continue
            missing = missing_from_baseline(pool, map_keys)
            if missing:
                buckets[job] = []
                force_full.add(job)
                baseline_gaps[job] = missing
                print(
                    f"::warning::Diffusion suite {suite} ({job}): {len(missing)} "
                    f"test(s) missing from the coverage baseline "
                    f"({', '.join(missing)}); running the full suite. "
                    "Rebuild the baseline to restore precision."
                )
                continue
            selected_cases, selected_standalones = match_diffusion_pool(
                pool, raw_entries
            )
            tokens = diffusion_tokens(selected_cases, selected_standalones)
            buckets[job] = tokens
            print(
                f"Suite {job} (diffusion {suite}): {pool.size} cases, "
                f"{len(tokens)} recommended"
            )

    precision_switch = args.precision_mode == "true"
    if not precision_switch:
        # Highest priority: the switch is the operator's override, so it must
        # win over every automatic decision below.
        decision = "full"
        decision_reason = "precision mode disabled (--precision-mode is not 'true')"
    elif not changed_files:
        # test_selector.py writes this file on every PR run; missing/empty
        # means it failed, so there is no evidence to narrow anything down.
        decision = "full"
        decision_reason = "changed-file list unavailable (test_selector.py did not run)"
    elif full_run_paths:
        decision = "full"
        decision_reason = "pipeline configuration changed: " + ", ".join(full_run_paths)
    elif not raw_entries:
        decision = "skip_all"
        decision_reason = "no test recommended"
    else:
        decision = "precision"
        decision_reason = f"{len(raw_entries)} recommended test(s)"
    skip_all_suites = decision == "skip_all"
    print(
        f"Decision: {decision} ({decision_reason}); "
        f"{len(changed_files)} changed path(s), {len(full_run_paths)} full-run path(s)"
    )

    total_count = sum(len(b) for b in buckets.values())
    # Recommendations no suite here contains: base-a / CPU / nightly tests run
    # by other pipelines, or a deleted file. None of them can be run by this
    # workflow, and a full run would not run them either, so this is reported
    # but never escalates to a full run.
    unknown_recommended = sorted(recommended - all_pool)
    if unknown_recommended:
        print(
            "::notice::Recommended test(s) not run by any suite in this "
            f"workflow: {', '.join(unknown_recommended)}"
        )

    if skip_all_suites and not force_full:
        print("::notice::No test recommended — skipping all suites.")
    precision_mode = decision == "precision"
    print(
        f"total_count={total_count} decision={decision} precision_mode={precision_mode}"
    )

    # Outputs first, summary last: a summary failure must never cost the
    # downstream jobs their bucket outputs (they'd fall back to full suites).
    if args.github_output:
        with open(args.github_output, "a") as out:
            for suite, files in buckets.items():
                key = suite.replace("-", "_")
                # Gate bucket contents on precision_mode: a run without precision
                # mode enabled never exports a non-empty bucket, so downstream
                # jobs can reference only_tests_<key> without any extra guard
                # expression.
                bucket_files = files if precision_mode else []
                out.write(f"only_tests_{key}<<EOF\n")
                for line in bucket_files:
                    out.write(line + "\n")
                out.write("EOF\n")
                # 'false' when precision mode is on, the suite enumerated a
                # non-empty registered pool, and this bucket is empty -> the
                # caller skips the whole suite job; or for ALL suites when
                # nothing was recommended and the changed-file list was
                # available (decision 'skip_all'). An empty ENUMERATION (0
                # registered files, e.g. a list_tests/run_suite filter drift)
                # falls back to 'true' so the job runs the full suite instead of
                # silently vanishing, except under 'skip_all', where no suite is
                # expected to run at all. Anything else ('true', or a missing
                # output from a failed step) means a full-suite run.
                # A force_full suite never takes either skip path: its baseline
                # is incomplete, so "nothing recommended" proves nothing.
                should_run = (
                    "false"
                    if (
                        not files
                        and suite not in force_full
                        and (
                            skip_all_suites
                            or (precision_mode and suite_pool_sizes[suite])
                        )
                    )
                    else "true"
                )
                out.write(f"should_run_{key}={should_run}\n")

    if args.github_summary or args.report_file:
        write_github_summary(
            [p for p in (args.github_summary, args.report_file) if p],
            raw_entries,
            recommended,
            buckets,
            suite_pool_sizes,
            force_full,
            baseline_gaps,
            precision_mode,
            skip_all_suites,
            decision=decision,
            decision_reason=decision_reason,
            full_run_paths=full_run_paths,
            unknown_recommended=unknown_recommended,
        )


if __name__ == "__main__":
    main()
