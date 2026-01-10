#!/usr/bin/env python3
"""
CI Coverage Report Generator

Collects all CI test registrations from test/registered/ and generates
a coverage report organized by folder, backend, and suite.

Usage:
    python scripts/ci/ci_coverage_report.py [--output-format markdown|json]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add the python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from sglang.test.ci.ci_register import CIRegistry, HWBackend, ut_parse_one_file


def collect_all_tests(registered_dir: str) -> list[CIRegistry]:
    """Collect all CI registrations from registered directory."""
    files = glob.glob(f"{registered_dir}/**/*.py", recursive=True)
    all_tests = []

    for file in sorted(files):
        try:
            registries = ut_parse_one_file(file)
            all_tests.extend(registries)
        except Exception as e:
            print(f"Warning: Failed to parse {file}: {e}", file=sys.stderr)

    return all_tests


def get_folder_name(filename: str) -> str:
    """Extract folder name from test filename."""
    # e.g., "registered/models/test_foo.py" -> "models"
    parts = Path(filename).parts
    if "registered" in parts:
        idx = parts.index("registered")
        if idx + 1 < len(parts) - 1:  # Has subfolder
            return parts[idx + 1]
    return "root"


def generate_markdown_report(tests: list[CIRegistry]) -> str:
    """Generate markdown report for GitHub step summary."""
    lines = []
    lines.append("# CI Coverage Overview\n")

    # Summary stats
    total = len(tests)
    by_backend = defaultdict(list)
    by_folder = defaultdict(list)
    by_suite = defaultdict(list)
    disabled_tests = []

    for t in tests:
        by_backend[t.backend.name].append(t)
        by_folder[get_folder_name(t.filename)].append(t)
        by_suite[t.suite].append(t)
        if t.disabled:
            disabled_tests.append(t)

    enabled = total - len(disabled_tests)
    lines.append(f"**Total Tests:** {total} ({enabled} enabled, {len(disabled_tests)} disabled)\n")

    # Backend summary
    lines.append("## By Backend\n")
    lines.append("| Backend | Total | Enabled | Disabled | Per-Commit | Nightly |")
    lines.append("|---------|-------|---------|----------|------------|---------|")

    for backend in ["CUDA", "AMD", "NPU", "CPU"]:
        backend_tests = by_backend.get(backend, [])
        if not backend_tests:
            continue
        b_total = len(backend_tests)
        b_disabled = sum(1 for t in backend_tests if t.disabled)
        b_enabled = b_total - b_disabled
        b_per_commit = sum(1 for t in backend_tests if not t.nightly and not t.disabled)
        b_nightly = sum(1 for t in backend_tests if t.nightly and not t.disabled)
        lines.append(f"| {backend} | {b_total} | {b_enabled} | {b_disabled} | {b_per_commit} | {b_nightly} |")

    lines.append("")

    # Folder breakdown
    lines.append("## By Folder\n")
    lines.append("| Folder | CUDA | AMD | NPU | CPU | Total |")
    lines.append("|--------|------|-----|-----|-----|-------|")

    for folder in sorted(by_folder.keys()):
        folder_tests = by_folder[folder]
        cuda = sum(1 for t in folder_tests if t.backend == HWBackend.CUDA)
        amd = sum(1 for t in folder_tests if t.backend == HWBackend.AMD)
        npu = sum(1 for t in folder_tests if t.backend == HWBackend.NPU)
        cpu = sum(1 for t in folder_tests if t.backend == HWBackend.CPU)
        lines.append(f"| {folder} | {cuda} | {amd} | {npu} | {cpu} | {len(folder_tests)} |")

    lines.append("")

    # Suite breakdown (non-NVIDIA focus)
    lines.append("## Non-NVIDIA Coverage (AMD, NPU, CPU)\n")

    non_nvidia_tests = [t for t in tests if t.backend != HWBackend.CUDA]
    if non_nvidia_tests:
        suite_backend = defaultdict(lambda: defaultdict(list))
        for t in non_nvidia_tests:
            suite_backend[t.suite][t.backend.name].append(t)

        lines.append("| Suite | Backend | Tests | Est. Time (s) |")
        lines.append("|-------|---------|-------|---------------|")

        for suite in sorted(suite_backend.keys()):
            for backend in ["AMD", "NPU", "CPU"]:
                suite_tests = suite_backend[suite].get(backend, [])
                if suite_tests:
                    est_time = sum(t.est_time for t in suite_tests)
                    lines.append(f"| {suite} | {backend} | {len(suite_tests)} | {est_time:.0f} |")
    else:
        lines.append("*No non-NVIDIA tests found.*\n")

    lines.append("")

    # CUDA suite breakdown
    lines.append("## CUDA Suites\n")
    lines.append("<details>")
    lines.append("<summary>Click to expand CUDA suite details</summary>\n")

    cuda_tests = [t for t in tests if t.backend == HWBackend.CUDA]
    cuda_suites = defaultdict(list)
    for t in cuda_tests:
        cuda_suites[t.suite].append(t)

    lines.append("| Suite | Tests | Enabled | Est. Time (s) | Nightly |")
    lines.append("|-------|-------|---------|---------------|---------|")

    for suite in sorted(cuda_suites.keys()):
        suite_tests = cuda_suites[suite]
        s_total = len(suite_tests)
        s_disabled = sum(1 for t in suite_tests if t.disabled)
        s_enabled = s_total - s_disabled
        s_est_time = sum(t.est_time for t in suite_tests if not t.disabled)
        s_nightly = "Yes" if any(t.nightly for t in suite_tests) else "No"
        lines.append(f"| {suite} | {s_total} | {s_enabled} | {s_est_time:.0f} | {s_nightly} |")

    lines.append("\n</details>\n")

    # Disabled tests
    if disabled_tests:
        lines.append("## Disabled Tests\n")
        lines.append("<details>")
        lines.append("<summary>Click to expand disabled tests</summary>\n")
        lines.append("| File | Backend | Reason |")
        lines.append("|------|---------|--------|")
        for t in sorted(disabled_tests, key=lambda x: x.filename):
            reason = t.disabled[:50] + "..." if len(t.disabled) > 50 else t.disabled
            lines.append(f"| {t.filename} | {t.backend.name} | {reason} |")
        lines.append("\n</details>\n")

    return "\n".join(lines)


def generate_json_report(tests: list[CIRegistry]) -> str:
    """Generate JSON report."""
    data = {
        "total": len(tests),
        "by_backend": {},
        "by_folder": {},
        "by_suite": {},
        "tests": [],
    }

    by_backend = defaultdict(list)
    by_folder = defaultdict(list)
    by_suite = defaultdict(list)

    for t in tests:
        by_backend[t.backend.name].append(t)
        by_folder[get_folder_name(t.filename)].append(t)
        by_suite[t.suite].append(t)

        data["tests"].append({
            "filename": t.filename,
            "backend": t.backend.name,
            "suite": t.suite,
            "est_time": t.est_time,
            "nightly": t.nightly,
            "disabled": t.disabled,
        })

    for backend, tests_list in by_backend.items():
        data["by_backend"][backend] = {
            "total": len(tests_list),
            "enabled": sum(1 for t in tests_list if not t.disabled),
            "disabled": sum(1 for t in tests_list if t.disabled),
        }

    for folder, tests_list in by_folder.items():
        data["by_folder"][folder] = len(tests_list)

    for suite, tests_list in by_suite.items():
        data["by_suite"][suite] = len(tests_list)

    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate CI coverage report")
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--registered-dir",
        default="test/registered",
        help="Path to registered test directory",
    )
    args = parser.parse_args()

    # Change to repo root if needed
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    os.chdir(repo_root)

    tests = collect_all_tests(args.registered_dir)

    if args.output_format == "markdown":
        report = generate_markdown_report(tests)
    else:
        report = generate_json_report(tests)

    print(report)

    # Write to GITHUB_STEP_SUMMARY if available
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file and args.output_format == "markdown":
        with open(summary_file, "a") as f:
            f.write(report)


if __name__ == "__main__":
    main()
