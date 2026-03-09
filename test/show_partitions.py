#!/usr/bin/env python3
"""Print test partition assignments for all CI stages.

Parses stage configs (hw, suite, partition_size) directly from pr-test.yml
so they never go out of sync. Uses the same deterministic LPT algorithm as
run_suite.py to show which test files land in which partition.

Runs on ubuntu-latest without third-party deps (stdlib + ci_register).
"""

import glob
import importlib.util
import os
import re

# Import ci_register directly by file path to avoid pulling in the full
# sglang package (which has heavy deps like numpy, torch, etc.).
_ci_register_path = os.path.join(
    os.path.dirname(__file__), "..", "python", "sglang", "test", "ci", "ci_register.py"
)
_spec = importlib.util.spec_from_file_location("ci_register", _ci_register_path)
_ci_register = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ci_register)

CIRegistry = _ci_register.CIRegistry
HWBackend = _ci_register.HWBackend
auto_partition = _ci_register.auto_partition
collect_tests = _ci_register.collect_tests

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "amd": HWBackend.AMD,
    "npu": HWBackend.NPU,
}

# Regex to detect run_suite.py invocations from pr-test.yml.
# Uses lookaheads so --hw and --suite can appear in any order.
_RUN_SUITE_RE = re.compile(
    r"^\s*(?:\w+=\S+\s+)?"  # optional env prefix (e.g. IS_BLACKWELL=1)
    r"python3\s+run_suite\.py\b"
    r"(?=.*\s--hw\s+(?P<hw>\S+))"
    r"(?=.*\s--suite\s+(?P<suite>\S+))"
)
_PARTITION_SIZE_RE = re.compile(r"--auto-partition-size\s+(\d+)")


def parse_stage_configs(workflow_path):
    """Parse (hw, suite, partition_count) tuples from pr-test.yml."""
    configs = []
    seen = set()
    with open(workflow_path) as f:
        for line in f:
            # Skip commented-out lines
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            m = _RUN_SUITE_RE.search(line)
            if not m:
                continue
            hw = m.group("hw")
            suite = m.group("suite")
            size_match = _PARTITION_SIZE_RE.search(line)
            size = int(size_match.group(1)) if size_match else 1
            key = (hw, suite)
            if key not in seen:
                seen.add(key)
                configs.append((hw, suite, size))
    return configs


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(repo_root, "test")
    workflow_path = os.path.join(repo_root, ".github", "workflows", "pr-test.yml")

    os.chdir(test_dir)

    stage_configs = parse_stage_configs(workflow_path)

    files = [
        f
        for f in glob.glob("registered/**/*.py", recursive=True)
        if not f.endswith("/conftest.py") and not f.endswith("/__init__.py")
    ]
    all_tests = collect_tests(files, sanity_check=True)

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    md_lines = []

    def out(line=""):
        print(line)
        md_lines.append(line)

    out("## Test Partition Assignments")
    out()

    for hw_str, suite, partition_count in stage_configs:
        hw = HW_MAPPING.get(hw_str)
        if hw is None:
            continue

        suite_tests = [
            t
            for t in all_tests
            if t.backend == hw and t.suite == suite and not t.nightly
        ]
        enabled = [t for t in suite_tests if t.disabled is None]
        disabled = [t for t in suite_tests if t.disabled is not None]

        total_time = sum(t.est_time for t in enabled)
        out(
            f"### {suite} (hw={hw_str}, {partition_count} partition(s), "
            f"{len(enabled)} enabled, {len(disabled)} disabled, est {total_time:.0f}s total)"
        )
        out()

        if disabled:
            out(f"**Disabled ({len(disabled)}):**")
            for t in disabled:
                out(f"- ~~{t.filename}~~ (reason: {t.disabled})")
            out()

        if not enabled:
            out("_(no enabled tests)_")
            out()
            continue

        out("| Partition | Tests | Est Time | Files |")
        out("|-----------|-------|----------|-------|")
        for rank in range(partition_count):
            part_tests = auto_partition(enabled, rank, partition_count)
            part_time = sum(t.est_time for t in part_tests)
            file_list = ", ".join(
                f"`{t.filename}` ({t.est_time:.0f}s)" for t in part_tests
            )
            out(f"| {rank} | {len(part_tests)} | {part_time:.0f}s | {file_list} |")
        out()

    if summary_file:
        with open(summary_file, "a") as f:
            f.write("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()
