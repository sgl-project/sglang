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

# Add the ci_register module path directly to avoid heavy sglang imports
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "python" / "sglang" / "test" / "ci")
)

from ci_register import CIRegistry, HWBackend, ut_parse_one_file


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


def get_test_basename(filename: str) -> str:
    """Extract just the test file name from the path."""
    return Path(filename).name


def organize_test_data(tests: list[CIRegistry]) -> dict:
    """Organize tests into various groupings."""
    by_backend = defaultdict(list)
    by_folder = defaultdict(list)
    disabled_tests = []

    for t in tests:
        by_backend[t.backend.name].append(t)
        by_folder[get_folder_name(t.filename)].append(t)
        if t.disabled:
            disabled_tests.append(t)

    # Count unique test files (a file may be registered for multiple backends)
    unique_files = set(t.filename for t in tests)
    unique_enabled_files = set(t.filename for t in tests if not t.disabled)
    unique_disabled_files = set(t.filename for t in tests if t.disabled)

    return {
        "total": len(tests),
        "total_unique_files": len(unique_files),
        "enabled": len(tests) - len(disabled_tests),
        "enabled_unique_files": len(unique_enabled_files),
        "disabled_count": len(disabled_tests),
        "disabled_unique_files": len(unique_disabled_files),
        "by_backend": by_backend,
        "by_folder": by_folder,
        "disabled_tests": disabled_tests,
    }


def generate_summary_section(data: dict) -> str:
    """Generate the summary/overview section."""
    lines = []
    lines.append("# CI Coverage Overview\n")
    lines.append(
        f"**Unique Test Files:** {data['total_unique_files']} ({data['enabled_unique_files']} enabled, {data['disabled_unique_files']} disabled)\n"
    )
    lines.append(
        f"**Total Registrations:** {data['total']} ({data['enabled']} enabled, {data['disabled_count']} disabled)\n"
    )
    lines.append(
        "*Note: A test file may be registered for multiple backends (e.g., CUDA + AMD), so total registrations > unique files.*\n"
    )

    by_backend = data["by_backend"]
    by_folder = data["by_folder"]
    disabled_tests = data["disabled_tests"]

    # Backend summary (collapsible)
    lines.append("<details>")
    lines.append("<summary><h2>Backend Summary</h2></summary>\n")
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
        lines.append(
            f"| {backend} | {b_total} | {b_enabled} | {b_disabled} | {b_per_commit} | {b_nightly} |"
        )

    lines.append("\n</details>\n")

    # Folder summary (collapsible)
    lines.append("<details>")
    lines.append("<summary><h2>Folder Summary</h2></summary>\n")
    lines.append("| Folder | CUDA | AMD | NPU | CPU | Total |")
    lines.append("|--------|------|-----|-----|-----|-------|")

    for folder in sorted(by_folder.keys()):
        folder_tests = by_folder[folder]
        cuda = sum(1 for t in folder_tests if t.backend == HWBackend.CUDA)
        amd = sum(1 for t in folder_tests if t.backend == HWBackend.AMD)
        npu = sum(1 for t in folder_tests if t.backend == HWBackend.NPU)
        cpu = sum(1 for t in folder_tests if t.backend == HWBackend.CPU)
        lines.append(
            f"| {folder} | {cuda} | {amd} | {npu} | {cpu} | {len(folder_tests)} |"
        )

    lines.append("\n</details>\n")

    # Disabled tests section (collapsible)
    if disabled_tests:
        lines.append("<details>")
        lines.append("<summary><h2>Disabled Tests</h2></summary>\n")
        lines.append("| File | Backend | Suite | Reason |")
        lines.append("|------|---------|-------|--------|")
        for t in sorted(disabled_tests, key=lambda x: (x.backend.name, x.filename)):
            test_name = get_test_basename(t.filename)
            reason = t.disabled[:50] + "..." if len(t.disabled) > 50 else t.disabled
            lines.append(f"| `{test_name}` | {t.backend.name} | {t.suite} | {reason} |")
        lines.append("\n</details>\n")

    return "\n".join(lines)


def generate_by_folder_section(data: dict) -> str:
    """Generate the 'All Tests by Folder' section."""
    lines = []
    by_folder = data["by_folder"]

    lines.append("# All Tests by Folder\n")

    for folder in sorted(by_folder.keys()):
        folder_tests = by_folder[folder]
        lines.append("<details>")
        lines.append(
            f"<summary><h2>{folder}/ ({len(folder_tests)} tests)</h2></summary>\n"
        )

        # Group by backend within folder
        folder_by_backend = defaultdict(list)
        for t in folder_tests:
            folder_by_backend[t.backend.name].append(t)

        for backend in ["CUDA", "AMD", "NPU", "CPU"]:
            backend_tests = folder_by_backend.get(backend, [])
            if not backend_tests:
                continue

            lines.append(f"### {backend} ({len(backend_tests)} tests)\n")
            lines.append("| Test File | Suite | Est. Time | Status |")
            lines.append("|-----------|-------|-----------|--------|")

            for t in sorted(backend_tests, key=lambda x: x.filename):
                test_name = get_test_basename(t.filename)
                status = (
                    "Disabled"
                    if t.disabled
                    else ("Nightly" if t.nightly else "Per-Commit")
                )
                lines.append(
                    f"| `{test_name}` | {t.suite} | {t.est_time:.0f}s | {status} |"
                )

            lines.append("")

        lines.append("</details>\n")

    return "\n".join(lines)


def generate_by_suite_section(data: dict) -> str:
    """Generate the 'All Tests by Test Suite' section."""
    lines = []
    by_backend = data["by_backend"]

    lines.append("# All Tests by Test Suite\n")

    for backend in ["CUDA", "AMD", "NPU", "CPU"]:
        backend_tests = by_backend.get(backend, [])
        if not backend_tests:
            continue

        b_total = len(backend_tests)
        b_disabled = sum(1 for t in backend_tests if t.disabled)
        b_enabled = b_total - b_disabled

        lines.append("<details>")
        lines.append(
            f"<summary><h2>{backend} Backend ({b_enabled} enabled, {b_disabled} disabled)</h2></summary>\n"
        )

        # Group by suite within backend
        backend_suites = defaultdict(list)
        for t in backend_tests:
            backend_suites[t.suite].append(t)

        for suite in sorted(backend_suites.keys()):
            suite_tests = backend_suites[suite]
            s_enabled = sum(1 for t in suite_tests if not t.disabled)
            s_disabled = sum(1 for t in suite_tests if t.disabled)
            s_est_time = sum(t.est_time for t in suite_tests if not t.disabled)
            is_nightly = any(t.nightly for t in suite_tests if not t.disabled)

            suite_type = "Nightly" if is_nightly else "Per-Commit"
            lines.append("<details>")
            lines.append(
                f"<summary><h3>{suite} ({s_enabled} enabled, {s_disabled} disabled) - {suite_type}</h3></summary>\n"
            )
            lines.append(f"*Estimated total time: {s_est_time:.0f}s*\n")

            lines.append("| Test File | Folder | Est. Time | Status |")
            lines.append("|-----------|--------|-----------|--------|")

            for t in sorted(suite_tests, key=lambda x: x.filename):
                test_name = get_test_basename(t.filename)
                folder = get_folder_name(t.filename)
                if t.disabled:
                    status = (
                        f"Disabled: {t.disabled[:30]}..."
                        if len(t.disabled) > 30
                        else f"Disabled: {t.disabled}"
                    )
                else:
                    status = "Nightly" if t.nightly else "Per-Commit"
                lines.append(
                    f"| `{test_name}` | {folder} | {t.est_time:.0f}s | {status} |"
                )

            lines.append("\n</details>\n")

        lines.append("</details>\n")

    return "\n".join(lines)


def generate_markdown_report(tests: list[CIRegistry], section: str = "all") -> str:
    """Generate markdown report for GitHub step summary."""
    data = organize_test_data(tests)

    if section == "summary":
        return generate_summary_section(data)
    elif section == "by-folder":
        return generate_by_folder_section(data)
    elif section == "by-suite":
        return generate_by_suite_section(data)
    else:  # "all"
        parts = [
            generate_summary_section(data),
            "---",
            generate_by_folder_section(data),
            "---",
            generate_by_suite_section(data),
        ]
        return "\n".join(parts)


def generate_json_report(tests: list[CIRegistry]) -> str:
    """Generate JSON report with detailed test listings."""
    by_backend = defaultdict(list)
    by_folder = defaultdict(list)

    for t in tests:
        by_backend[t.backend.name].append(t)
        by_folder[get_folder_name(t.filename)].append(t)

    disabled_tests = [t for t in tests if t.disabled]

    # Build structured data
    data = {
        "summary": {
            "total": len(tests),
            "enabled": len(tests) - len(disabled_tests),
            "disabled": len(disabled_tests),
        },
        "tests_by_folder": {},
        "tests_by_suite": {},
        "backend_summary": {},
        "folder_summary": {},
        "disabled_tests": [],
    }

    # Section 1: Tests by Folder
    for folder in sorted(by_folder.keys()):
        folder_tests = by_folder[folder]
        folder_by_backend = defaultdict(list)
        for t in folder_tests:
            folder_by_backend[t.backend.name].append(t)

        data["tests_by_folder"][folder] = {
            "total": len(folder_tests),
            "backends": {},
        }

        for backend in ["CUDA", "AMD", "NPU", "CPU"]:
            backend_tests = folder_by_backend.get(backend, [])
            if backend_tests:
                data["tests_by_folder"][folder]["backends"][backend] = [
                    {
                        "filename": get_test_basename(t.filename),
                        "suite": t.suite,
                        "est_time": t.est_time,
                        "status": (
                            "disabled"
                            if t.disabled
                            else ("nightly" if t.nightly else "per-commit")
                        ),
                    }
                    for t in sorted(backend_tests, key=lambda x: x.filename)
                ]

    # Section 2: Tests by Suite (Backend -> Suite)
    for backend in ["CUDA", "AMD", "NPU", "CPU"]:
        backend_tests = by_backend.get(backend, [])
        if not backend_tests:
            continue

        backend_suites = defaultdict(list)
        for t in backend_tests:
            backend_suites[t.suite].append(t)

        data["tests_by_suite"][backend] = {
            "total": len(backend_tests),
            "enabled": sum(1 for t in backend_tests if not t.disabled),
            "disabled": sum(1 for t in backend_tests if t.disabled),
            "suites": {},
        }

        for suite in sorted(backend_suites.keys()):
            suite_tests = backend_suites[suite]
            is_nightly = any(t.nightly for t in suite_tests if not t.disabled)

            data["tests_by_suite"][backend]["suites"][suite] = {
                "total": len(suite_tests),
                "enabled": sum(1 for t in suite_tests if not t.disabled),
                "disabled": sum(1 for t in suite_tests if t.disabled),
                "est_time": sum(t.est_time for t in suite_tests if not t.disabled),
                "type": "nightly" if is_nightly else "per-commit",
                "tests": [
                    {
                        "filename": get_test_basename(t.filename),
                        "folder": get_folder_name(t.filename),
                        "est_time": t.est_time,
                        "status": (
                            "disabled"
                            if t.disabled
                            else ("nightly" if t.nightly else "per-commit")
                        ),
                        "disabled_reason": t.disabled if t.disabled else None,
                    }
                    for t in sorted(suite_tests, key=lambda x: x.filename)
                ],
            }

    # Backend summary
    for backend in ["CUDA", "AMD", "NPU", "CPU"]:
        backend_tests = by_backend.get(backend, [])
        if backend_tests:
            data["backend_summary"][backend] = {
                "total": len(backend_tests),
                "enabled": sum(1 for t in backend_tests if not t.disabled),
                "disabled": sum(1 for t in backend_tests if t.disabled),
                "per_commit": sum(
                    1 for t in backend_tests if not t.nightly and not t.disabled
                ),
                "nightly": sum(
                    1 for t in backend_tests if t.nightly and not t.disabled
                ),
            }

    # Folder summary
    for folder in sorted(by_folder.keys()):
        folder_tests = by_folder[folder]
        data["folder_summary"][folder] = {
            "CUDA": sum(1 for t in folder_tests if t.backend == HWBackend.CUDA),
            "AMD": sum(1 for t in folder_tests if t.backend == HWBackend.AMD),
            "NPU": sum(1 for t in folder_tests if t.backend == HWBackend.NPU),
            "CPU": sum(1 for t in folder_tests if t.backend == HWBackend.CPU),
            "total": len(folder_tests),
        }

    # Disabled tests
    for t in sorted(disabled_tests, key=lambda x: (x.backend.name, x.filename)):
        data["disabled_tests"].append(
            {
                "filename": get_test_basename(t.filename),
                "backend": t.backend.name,
                "suite": t.suite,
                "reason": t.disabled,
            }
        )

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
        "--section",
        choices=["all", "summary", "by-folder", "by-suite"],
        default="all",
        help="Which section to output (default: all). Only applies to markdown format.",
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
        report = generate_markdown_report(tests, section=args.section)
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
