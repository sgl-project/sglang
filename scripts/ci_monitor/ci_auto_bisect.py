#!/usr/bin/env python3
"""
SGLang CI Auto Bisect

Fetches recent Nvidia scheduled PR Test runs, identifies consistently failing
tests, and calls Claude to classify each as regression/flaky/hardware/environment.

Self-contained: does its own lightweight GitHub API analysis instead of running
the full ci_failures_analysis.py, keeping API usage to ~30-40 calls.

Usage:
    python ci_auto_bisect.py \
        --github-token $GITHUB_TOKEN \
        --anthropic-api-key $ANTHROPIC_API_KEY \
        --output bisect_results.json \
        --max-failures 10
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import anthropic
import requests

REPO = "sgl-project/sglang"
GITHUB_API = "https://api.github.com"

# Claude model to use
CLAUDE_MODEL = "claude-sonnet-4-5-20250514"

# Path to the bisect skill definition (relative to repo root)
BISECT_SKILL_PATH = ".claude/skills/sglang-bisect-ci-regression/SKILL.md"

# Jobs to exclude from analysis (administrative/setup, not actual tests)
EXCLUDED_JOBS = [
    "check-changes",
    "pr-test-finish",
    "call-gate",
    "pr-gate",
    "check-all-jobs",
]

# Number of recent scheduled runs to analyze
SCHEDULED_RUN_LIMIT = 6

# Compiled regex for stripping ANSI escape codes from CI logs
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FailureTarget:
    """A single test failure that needs bisection analysis."""

    job_name: str
    test_file: str
    hardware: str
    current_streak: int
    first_failure_sha: str
    last_failure_sha: str
    first_failure_date: str
    last_failure_date: str
    first_failure_job_url: str
    last_failure_job_url: str
    first_failure_job_id: Optional[int]
    last_failure_job_id: Optional[int]
    recent_run_statuses: List[str] = field(default_factory=list)
    test_streak: int = 0
    test_total_failures: int = 0


@dataclass
class BisectionContext:
    """All gathered context for a single bisection."""

    target: FailureTarget
    commits_between: List[str] = field(default_factory=list)
    error_signature: str = ""
    runner_correlation: Dict = field(default_factory=dict)
    candidate_commits: List[str] = field(default_factory=list)


@dataclass
class BisectionResult:
    """Claude's analysis result for a single failure."""

    target: FailureTarget
    classification: str = "unknown"
    confidence: str = "low"
    suspected_commit: Optional[str] = None
    suspected_pr: Optional[int] = None
    evidence_summary: str = ""
    recommended_fix: str = ""
    raw_response: str = ""
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Focused GitHub API analysis (Nvidia scheduled only)
# ---------------------------------------------------------------------------


def _gh_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _gh_get(url: str, token: str, params: Optional[dict] = None) -> Optional[dict]:
    """Make a GitHub API GET request. Raises on auth/permission errors."""
    try:
        resp = requests.get(url, headers=_gh_headers(token), params=params, timeout=30)
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"GitHub API auth/permission error ({resp.status_code}) "
                f"for {url}: {resp.text[:200]}"
            )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  ERROR: GitHub API request failed for {url}: {e}")
        raise


def _gh_get_all_pages(
    url: str, token: str, params: Optional[dict] = None
) -> List[dict]:
    """Fetch all pages for a paginated GitHub API endpoint. Raises on auth errors."""
    all_items = []
    current_params = dict(params or {})
    current_url: Optional[str] = url

    while current_url:
        resp = requests.get(
            current_url,
            headers=_gh_headers(token),
            params=current_params,
            timeout=30,
        )
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"GitHub API auth/permission error ({resp.status_code}): "
                f"{resp.text[:200]}"
            )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("jobs", data.get("workflow_runs", []))
        all_items.extend(items)

        # Follow pagination
        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(", "):
            if 'rel="next"' in part:
                next_url = part.split(";")[0].strip("<>")
                break
        current_url = next_url
        current_params = {}  # params are in the URL for subsequent pages

    return all_items


def fetch_nvidia_scheduled_runs(token: str) -> List[dict]:
    """Fetch recent scheduled PR Test runs on main. ~1 API call."""
    print(f"Fetching {SCHEDULED_RUN_LIMIT} recent scheduled PR Test (Nvidia) runs...")
    url = f"{GITHUB_API}/repos/{REPO}/actions/workflows/pr-test.yml/runs"
    data = _gh_get(url, token, {"event": "schedule", "per_page": SCHEDULED_RUN_LIMIT})
    if not data:
        return []
    runs = data.get("workflow_runs", [])
    print(f"  Found {len(runs)} runs")
    return runs


def fetch_jobs_for_run(run_id: int, token: str) -> List[dict]:
    """Fetch all jobs for a workflow run, handling pagination. ~1-2 API calls."""
    url = f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}/jobs"
    return _gh_get_all_pages(url, token, {"per_page": 100})


def fetch_job_logs(job_id: int, token: str, max_chars: int = 2000000) -> str:
    """Fetch logs for a specific job. 1 API call. Returns empty string on failure."""
    if not job_id:
        return ""
    try:
        url = f"{GITHUB_API}/repos/{REPO}/actions/jobs/{job_id}/logs"
        resp = requests.get(
            url, headers=_gh_headers(token), timeout=60, allow_redirects=True
        )
        if resp.status_code == 200:
            text = resp.text
            return text[-max_chars:] if len(text) > max_chars else text
        print(f"  Warning: Log fetch for job {job_id} returned HTTP {resp.status_code}")
    except requests.RequestException as e:
        print(f"  Warning: Failed to fetch logs for job {job_id}: {e}")
    return ""


def parse_test_summary(logs: str) -> Optional[Dict]:
    """Parse the test summary block from job logs.

    Returns dict with passed/total counts and list of failed tests,
    or None if no summary found.
    """
    # Strip ANSI escape codes
    logs = _ANSI_ESCAPE_RE.sub("", logs)

    summary_match = re.search(r"Test Summary:\s*(\d+)/(\d+)\s*passed", logs)
    if not summary_match:
        # Try to find the last running test (timeout scenario)
        last_test = _find_last_running_test(logs)
        if last_test:
            return {"passed": 0, "total": 0, "failed_tests": [last_test]}
        return None

    try:
        passed = int(summary_match.group(1))
        total = int(summary_match.group(2))
    except (ValueError, TypeError):
        return None

    failed_tests = []
    failed_section_match = re.search(
        r".?\s*FAILED:\s*\n(.*?)(?:={10,}|$)", logs, re.DOTALL
    )
    if failed_section_match:
        for match in re.finditer(r"(\S+\.py)", failed_section_match.group(1)):
            full_path = match.group(1)
            test_file = full_path.split("/")[-1] if "/" in full_path else full_path
            failed_tests.append({"test_file": test_file, "full_path": full_path})

    return {"passed": passed, "total": total, "failed_tests": failed_tests}


def _find_last_running_test(logs: str) -> Optional[Dict]:
    """Find the last test running before logs cut off (timeout scenarios)."""
    lines = logs.split("\n")
    test_patterns = [r"(\S+\.py)::", r"python3?\s+(\S+\.py)"]

    # Find last "server_args:" and look above it for test file
    server_args_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "server_args:" in lines[i].lower() or "server_args =" in lines[i]:
            server_args_idx = i
            break

    if server_args_idx is not None:
        for j in range(1, 11):
            line_idx = server_args_idx - j
            if line_idx >= 0:
                for pattern in test_patterns:
                    match = re.search(pattern, lines[line_idx])
                    if match:
                        full_path = match.group(1)
                        test_file = (
                            full_path.split("/")[-1] if "/" in full_path else full_path
                        )
                        if test_file.endswith(".py"):
                            return {"test_file": test_file, "full_path": full_path}
    return None


def analyze_scheduled_failures(
    token: str, min_streak: int = 2, max_failures: int = 10
) -> Tuple[List[FailureTarget], Dict[int, str]]:
    """
    Fetch Nvidia scheduled runs, analyze job/test failure streaks, return targets.

    Returns (targets, logs_cache) where logs_cache maps job_id -> log text,
    so callers can reuse fetched logs without re-fetching.

    API calls: 1 (list runs) + ~12 (jobs per run) + ~5-10 (logs for broken jobs)
    = ~20-25 total.
    """
    logs_cache: Dict[int, str] = {}

    runs = fetch_nvidia_scheduled_runs(token)
    if not runs:
        print("No scheduled runs found")
        return [], logs_cache

    # Sort oldest-first for streak tracking
    sorted_runs = sorted(runs, key=lambda r: r.get("created_at", ""))

    # Track per-job streaks
    job_streak: Dict[str, int] = {}
    job_first_fail: Dict[str, dict] = {}
    job_last_fail: Dict[str, dict] = {}
    job_recent: Dict[str, List[str]] = {}

    print(f"\nAnalyzing {len(sorted_runs)} runs for job failure streaks...")
    api_calls = 1  # The initial list-runs call

    for run in sorted_runs:
        try:
            run_id: int = run["id"]
        except (KeyError, TypeError):
            print(f"  Warning: Skipping malformed run entry: {run}")
            continue

        head_sha = run.get("head_sha", "")[:8]
        created_at = run.get("created_at", "")
        run_url = f"https://github.com/{REPO}/actions/runs/{run_id}"

        jobs = fetch_jobs_for_run(run_id, token)
        api_calls += 1
        time.sleep(0.05)

        for job in jobs:
            name = job.get("name", "")
            if any(name.startswith(ex) for ex in EXCLUDED_JOBS):
                continue

            conclusion = job.get("conclusion")
            job_id = job.get("id")
            job_url = job.get("html_url", run_url)

            if name not in job_streak:
                job_streak[name] = 0
                job_recent[name] = []

            if conclusion == "failure":
                job_streak[name] += 1
                if job_streak[name] == 1:
                    job_first_fail[name] = {
                        "head_sha": head_sha,
                        "created_at": created_at,
                        "job_url": job_url,
                        "job_id": job_id,
                    }
                job_last_fail[name] = {
                    "head_sha": head_sha,
                    "created_at": created_at,
                    "job_url": job_url,
                    "job_id": job_id,
                }
                job_recent[name].append("❌")
            elif conclusion == "success":
                job_streak[name] = 0
                job_first_fail.pop(name, None)
                job_last_fail.pop(name, None)
                job_recent[name].append("✅")
            else:
                job_recent[name].append("⚪")

    # Find jobs with streak >= min_streak
    broken_jobs = {
        name: {
            "streak": streak,
            "first_fail": job_first_fail.get(name, {}),
            "last_fail": job_last_fail.get(name, {}),
            "recent": job_recent.get(name, [])[-10:],
        }
        for name, streak in job_streak.items()
        if streak >= min_streak
    }

    print(f"Found {len(broken_jobs)} jobs with streak >= {min_streak}")
    if not broken_jobs:
        print(f"Total GitHub API calls: {api_calls}")
        return [], logs_cache

    # For broken jobs, fetch logs and parse test-level failures
    # Only fetch logs for the MOST RECENT failure of each broken job
    print("\nFetching logs for broken jobs to identify failing tests...")
    targets = []

    for job_name, data in broken_jobs.items():
        last_fail = data["last_fail"]
        last_job_id = last_fail.get("job_id")

        test_failures = []
        if last_job_id:
            logs = fetch_job_logs(last_job_id, token)
            api_calls += 1
            if logs:
                logs_cache[last_job_id] = logs
                summary = parse_test_summary(logs)
                if summary and summary.get("failed_tests"):
                    test_failures = summary["failed_tests"]

        first_fail = data["first_fail"]

        def _make_target(test_file: str) -> FailureTarget:
            return FailureTarget(
                job_name=job_name,
                test_file=test_file,
                hardware="Nvidia",
                current_streak=data["streak"],
                first_failure_sha=first_fail.get("head_sha", ""),
                last_failure_sha=last_fail.get("head_sha", ""),
                first_failure_date=first_fail.get("created_at", ""),
                last_failure_date=last_fail.get("created_at", ""),
                first_failure_job_url=first_fail.get("job_url", ""),
                last_failure_job_url=last_fail.get("job_url", ""),
                first_failure_job_id=first_fail.get("job_id"),
                last_failure_job_id=last_fail.get("job_id"),
                recent_run_statuses=data["recent"],
                test_streak=data["streak"],
                test_total_failures=data["streak"],
            )

        if test_failures:
            for tf in test_failures:
                targets.append(_make_target(tf["test_file"]))
        else:
            targets.append(_make_target("<job-level>"))

        time.sleep(0.1)

    print(f"Total GitHub API calls: {api_calls}")

    # Deduplicate: same test across partitions -> keep highest streak
    # For job-level targets, include job_name to avoid collapsing distinct failures
    seen: Dict[str, FailureTarget] = {}
    for t in targets:
        if t.test_file == "<job-level>":
            key = f"<job-level>:{t.job_name}"
        else:
            key = t.test_file
        if key not in seen or t.current_streak > seen[key].current_streak:
            seen[key] = t
    targets = list(seen.values())

    # Prioritize by streak, descending
    targets.sort(
        key=lambda t: t.current_streak * 10 + t.test_total_failures, reverse=True
    )

    return targets[:max_failures], logs_cache


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def resolve_sha(short_sha: str) -> str:
    """Resolve a short SHA to a full SHA using git rev-parse."""
    if not short_sha:
        return ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", short_sha],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return short_sha


def get_commits_between(first_sha: str, last_sha: str) -> List[str]:
    """Get commit list between two SHAs using git log.

    Uses first_sha~1..last_sha to include the first failure commit itself,
    since that commit may be the one that introduced the regression.
    """
    if not first_sha or not last_sha:
        return []

    full_first = resolve_sha(first_sha)
    full_last = resolve_sha(last_sha)

    # Try first_sha~1 to include the first failure commit itself
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"{full_first}~1..{full_last}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l]
            if len(lines) > 50:
                return (
                    lines[:25]
                    + [f"... ({len(lines) - 50} commits omitted) ..."]
                    + lines[-25:]
                )
            return lines
        # Fallback: ~1 may fail on root commit or missing SHA
        print(f"    Note: first_sha~1 failed, falling back to exclusive range")
        result = subprocess.run(
            ["git", "log", "--oneline", f"{full_first}..{full_last}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l]
            if len(lines) > 50:
                return (
                    lines[:25]
                    + [f"... ({len(lines) - 50} commits omitted) ..."]
                    + lines[-25:]
                )
            return lines
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []


def get_candidate_commits(first_sha: str, last_sha: str, test_file: str) -> List[str]:
    """Get commits that touch files related to the failing test."""
    if not first_sha or not last_sha or test_file == "<job-level>":
        return []

    full_first = resolve_sha(first_sha)
    full_last = resolve_sha(last_sha)

    related_paths = _infer_related_paths(test_file)
    if not related_paths:
        return []

    try:
        # Use first_sha~1 to include the first failure commit itself
        result = subprocess.run(
            ["git", "log", "--oneline", f"{full_first}~1..{full_last}", "--"]
            + related_paths,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l]
            return lines[:15]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []


def _infer_related_paths(test_file: str) -> List[str]:
    """Heuristically infer source paths related to a test file."""
    paths = ["test/"]

    core = test_file
    if core.startswith("test_"):
        core = core[5:]
    if core.endswith(".py"):
        core = core[:-3]

    path_hints = {
        "lora": ["python/sglang/srt/lora/"],
        "moe": ["python/sglang/srt/layers/moe/"],
        "tp": ["python/sglang/srt/distributed/"],
        "dp": ["python/sglang/srt/distributed/"],
        "endpoint": ["python/sglang/srt/entrypoints/"],
        "openai": ["python/sglang/srt/entrypoints/openai/"],
        "anthropic": ["python/sglang/srt/entrypoints/anthropic/"],
        "server": ["python/sglang/srt/entrypoints/"],
        "engine": ["python/sglang/srt/"],
        "sampling": ["python/sglang/srt/sampling/"],
        "tokenizer": ["python/sglang/srt/managers/tokenizer_manager.py"],
        "schedule": ["python/sglang/srt/managers/schedule_batch.py"],
        "radix": ["python/sglang/srt/mem_cache/"],
        "cuda_graph": ["python/sglang/srt/layers/cuda_graph_runner.py"],
        "attention": ["python/sglang/srt/layers/attention/"],
        "quantiz": ["python/sglang/srt/layers/quantization/"],
        "specul": ["python/sglang/srt/speculative/"],
        "vision": ["python/sglang/srt/models/"],
        "embed": ["python/sglang/srt/layers/"],
        "kernel": ["sgl-kernel/", "python/sglang/srt/layers/"],
        "bench": ["benchmark/"],
        "constrained": ["python/sglang/srt/constrained/"],
    }

    for hint, hint_paths in path_hints.items():
        if hint in core:
            paths.extend(hint_paths)

    if len(paths) == 1:
        paths.append("python/sglang/srt/")

    return paths


# ---------------------------------------------------------------------------
# Error extraction
# ---------------------------------------------------------------------------


def extract_error_signature(logs: str, test_file: str) -> str:
    """Extract error-relevant lines from job logs.

    When a specific test_file is given, tries to find the error context
    closest to where that test ran (not just the last error in the log).
    """
    if not logs:
        return ""

    logs = _ANSI_ESCAPE_RE.sub("", logs)
    lines = logs.split("\n")

    # If we have a specific test file, try to find the error near its FAILED marker
    test_stem = ""
    if test_file and test_file != "<job-level>":
        test_stem = re.escape(test_file.replace(".py", ""))

        # Strategy: find the "FAILED: .../test_name.py" line and look backwards
        # for the traceback/error that caused it. CI logs have this pattern:
        #   <traceback>
        #   FAILED (errors=N)
        #   ...
        #   FAILED: /path/to/test_name.py returned exit code 1
        failed_marker = None
        for i, line in enumerate(lines):
            if re.search(r"FAILED:.*" + test_stem, line):
                failed_marker = i
                break

        if failed_marker:
            # Look backwards from the FAILED marker for the traceback
            # CI logs can have ~150 lines between the error and the FAILED marker
            # (metrics, report writing, rate limit messages, etc.)
            search_start = max(0, failed_marker - 200)
            region = lines[search_start : failed_marker + 5]

            # Find the last Traceback or error in this region
            error_indices = []
            for j, line in enumerate(region):
                if re.search(
                    r"Traceback|AssertionError|Exception:|FAILED \(errors",
                    line,
                ):
                    error_indices.append(j)

            if error_indices:
                last_err = error_indices[-1]
                ctx_start = max(0, last_err - 5)
                ctx_end = min(len(region), last_err + 20)
                excerpt = "\n".join(region[ctx_start:ctx_end])
                return excerpt[:2000]

    # Fallback: find the last error/traceback anywhere in the log
    error_patterns = [
        r"AssertionError",
        r"FAIL(?:ED)?:",
        r"Error:",
        r"Exception:",
        r"Traceback",
        r"raise ",
    ]
    if test_stem:
        error_patterns.append(test_stem)

    combined_pattern = "|".join(error_patterns)

    match_indices = []
    for i, line in enumerate(lines):
        if re.search(combined_pattern, line, re.IGNORECASE):
            match_indices.append(i)

    if not match_indices:
        return "\n".join(lines[-50:])[:2000]

    last_match = match_indices[-1]
    start = max(0, last_match - 10)
    end = min(len(lines), last_match + 40)
    excerpt = "\n".join(lines[start:end])

    summary_match = re.search(r"Test Summary:.{0,2000}?(?:={10,}|$)", logs, re.DOTALL)
    if summary_match:
        excerpt += "\n\n--- Test Summary ---\n" + summary_match.group(0)[:500]

    return excerpt[:2000]


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def gather_bisection_context(
    target: FailureTarget,
    github_token: str,
    logs_cache: Optional[Dict[int, str]] = None,
) -> BisectionContext:
    """Gather all context needed for bisection analysis.

    Args:
        logs_cache: Pre-fetched logs from analyze_scheduled_failures to avoid
            re-fetching the same job logs.
    """
    print(f"  Gathering context for {target.test_file} in {target.job_name}...")

    commits = get_commits_between(target.first_failure_sha, target.last_failure_sha)
    print(f"    Found {len(commits)} commits in range")

    candidates = get_candidate_commits(
        target.first_failure_sha, target.last_failure_sha, target.test_file
    )
    print(f"    Found {len(candidates)} candidate commits")

    # Fetch error logs from the most recent failure (reuse cache if available)
    error_sig = ""
    if target.last_failure_job_id:
        cached = (logs_cache or {}).get(target.last_failure_job_id)
        if cached:
            print(f"    Using cached logs for job {target.last_failure_job_id}")
            logs = cached
        else:
            print(f"    Fetching logs for job {target.last_failure_job_id}...")
            logs = fetch_job_logs(target.last_failure_job_id, github_token)
        if logs:
            error_sig = extract_error_signature(logs, target.test_file)
            print(f"    Extracted {len(error_sig)} chars of error context")
        else:
            print("    Warning: No logs retrieved")

    return BisectionContext(
        target=target,
        commits_between=commits,
        error_signature=error_sig,
        candidate_commits=candidates,
    )


# ---------------------------------------------------------------------------
# Skill loading
# ---------------------------------------------------------------------------


class SkillLoadError(Exception):
    """Raised when the bisect skill SKILL.md cannot be loaded or parsed."""


# Required sections to extract from SKILL.md. If any are missing after a
# rename or restructuring, the script raises SkillLoadError so the team
# gets a Slack notification instead of silently falling back.
_REQUIRED_SKILL_SECTIONS = {
    "Key Patterns to Recognize": r"## Key Patterns to Recognize\n(.*?)(?=\n## |\Z)",
    "Important Notes": r"## Important Notes\n(.*?)(?=\n## |\Z)",
    "Root Cause Classification": r"### Root Cause Classification\n(.*?)(?=\n### |\Z)",
}


def load_bisect_skill() -> str:
    """Load the bisect skill SKILL.md and extract analysis methodology sections.

    Reads the skill definition from the repo and extracts the sections that
    are useful as Claude prompt context: Key Patterns, Important Notes, and
    Root Cause Classification. This keeps the automated workflow in sync with
    any updates to the skill definition.

    Raises:
        SkillLoadError: If SKILL.md is not found or required sections are missing.
    """
    # Try repo-relative path first, then look in common locations
    candidates = [
        BISECT_SKILL_PATH,
        os.path.join(os.path.dirname(__file__), "..", "..", BISECT_SKILL_PATH),
    ]

    content = ""
    for path in candidates:
        try:
            with open(path) as f:
                content = f.read()
            break
        except FileNotFoundError:
            continue

    if not content:
        raise SkillLoadError(
            f"Could not find {BISECT_SKILL_PATH}. Searched: {candidates}"
        )

    # Extract the required analysis sections
    sections = []
    missing = []

    for section_name, pattern in _REQUIRED_SKILL_SECTIONS.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            sections.append(f"## {section_name}\n{match.group(1).strip()}")
        else:
            missing.append(section_name)

    if missing:
        raise SkillLoadError(
            f"SKILL.md is missing required sections: {missing}. "
            f"Was SKILL.md restructured? Update _REQUIRED_SKILL_SECTIONS in "
            f"ci_auto_bisect.py to match the new section names."
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Claude API integration
# ---------------------------------------------------------------------------


def build_prompt(context: BisectionContext, skill_content: str = "") -> str:
    """Build a structured prompt for Claude to analyze a CI failure.

    Args:
        skill_content: Extracted sections from SKILL.md to use as analysis
            methodology.
    """
    t = context.target

    statuses_str = " ".join(t.recent_run_statuses) if t.recent_run_statuses else "N/A"
    commits_str = (
        "\n".join(context.commits_between)
        if context.commits_between
        else "No commits found in range (SHAs may be identical or unresolvable)"
    )
    candidates_str = (
        "\n".join(context.candidate_commits)
        if context.candidate_commits
        else "No commits found touching related files"
    )

    runner_str = "No runner-specific data available"
    if context.runner_correlation:
        runner_lines = []
        for runner_instance, data in context.runner_correlation.items():
            runner_lines.append(
                f"  - {runner_instance} / {data['runner_name']} "
                f"({data['runner_labels']}): {data['count']} failures"
            )
        runner_str = "\n".join(runner_lines)

    error_str = context.error_signature or "No error logs available"

    methodology = f"""## Analysis Methodology (from bisect skill definition)

{skill_content}

## Additional Classification Guidance
Classify as exactly ONE of: code_regression, flaky_test, hardware_issue, environment_change.
- If recent run pattern shows alternating pass/fail -> likely flaky
- If recent run pattern shows solid block of failures -> likely regression or environment
- If commit range is empty (same SHA) -> the failure predates this range, check if flaky
- If candidate commits are empty but failures are consistent -> environment change or hardware"""

    return f"""You are an expert CI regression analyst for the SGLang project (a high-performance LLM serving framework).

## Task
Analyze this CI test failure and classify its root cause. Be precise and evidence-based.

## Failure Details
- **Test**: {t.test_file}
- **Job**: {t.job_name}
- **Hardware**: {t.hardware}
- **Job consecutive failures**: {t.current_streak}
- **Test consecutive failures**: {t.test_streak}
- **First failure**: {t.first_failure_date} (SHA: {t.first_failure_sha})
  URL: {t.first_failure_job_url}
- **Last failure**: {t.last_failure_date} (SHA: {t.last_failure_sha})
  URL: {t.last_failure_job_url}
- **Recent run pattern** (oldest to newest): {statuses_str}

## Error Signature (from most recent failure)
```
{error_str}
```

## All Commits in Range ({t.first_failure_sha}..{t.last_failure_sha})
```
{commits_str}
```

## Commits Touching Related Files
```
{candidates_str}
```

## Runner Correlation
{runner_str}

Note: PR numbers appear in squash-merged commit messages as (#1234). Extract the PR number from the suspected commit message if possible.

{methodology}

## Required Output
Respond with ONLY a JSON object (no markdown fencing, no extra text):
{{"classification": "code_regression|flaky_test|hardware_issue|environment_change", "confidence": "high|medium|low", "suspected_commit": "short SHA or null", "suspected_pr": PR_NUMBER_or_null, "evidence_summary": "2-3 sentence explanation of your reasoning", "recommended_fix": "1-2 sentence actionable recommendation"}}"""


def call_claude_api(
    prompt: str,
    api_key: str,
    max_retries: int = 3,
) -> Tuple[str, int]:
    """Call Claude API with retry logic. Returns (response_text, total_tokens)."""
    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
            if not message.content:
                print("    Warning: Claude returned empty content")
                return "", 0
            # With extended thinking, content has thinking + text blocks
            response_text = ""
            for block in message.content:
                if block.type == "text":
                    response_text = block.text
                    break
            tokens_used = message.usage.input_tokens + message.usage.output_tokens
            return response_text, tokens_used
        except anthropic.AuthenticationError as e:
            # Auth errors will never self-resolve -- fail fast
            raise RuntimeError(f"Anthropic API authentication failed: {e}") from e
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Rate limited after {max_retries} retries, giving up")
                return "", 0
        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    API error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error after {max_retries} retries: {e}")
                return "", 0

    return "", 0


def parse_claude_response(response_text: str) -> dict:
    """Parse Claude's JSON response."""
    if not response_text:
        return {
            "classification": "unknown",
            "confidence": "low",
            "suspected_commit": None,
            "suspected_pr": None,
            "evidence_summary": "Failed to get analysis from Claude API",
            "recommended_fix": "Manual investigation required",
        }

    # First try: the entire response is JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Second try: find JSON block (possibly with nested braces)
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "classification": "unknown",
        "confidence": "low",
        "suspected_commit": None,
        "suspected_pr": None,
        "evidence_summary": f"Could not parse Claude response: {response_text[:200]}",
        "recommended_fix": "Manual investigation required",
    }


# ---------------------------------------------------------------------------
# GitHub Actions summary
# ---------------------------------------------------------------------------


def generate_github_summary(results: List[BisectionResult]) -> None:
    """Write markdown summary to $GITHUB_STEP_SUMMARY."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        print("Not running in GitHub Actions, skipping step summary")
        return

    by_class: Dict[str, List[BisectionResult]] = {}
    for r in results:
        by_class.setdefault(r.classification, []).append(r)

    class_order = [
        ("code_regression", "🔴"),
        ("hardware_issue", "🟠"),
        ("environment_change", "🟡"),
        ("flaky_test", "🔵"),
        ("unknown", "⚪"),
    ]

    lines = ["# CI Auto Bisect Results\n"]

    lines.append("## Summary\n")
    for cls, emoji in class_order:
        count = len(by_class.get(cls, []))
        if count > 0:
            lines.append(f"- {emoji} **{cls.replace('_', ' ').title()}**: {count}")
    lines.append("")

    if not results:
        lines.append("No failures requiring bisection analysis.\n")
    else:
        lines.append("## Details\n")
        lines.append(
            "| Classification | Test | Job | Confidence "
            "| Suspected Cause | Recommendation |"
        )
        lines.append("|---|---|---|---|---|---|")

        for cls, emoji in class_order:
            for r in by_class.get(cls, []):
                suspected = ""
                if r.suspected_commit:
                    suspected = f"`{r.suspected_commit}`"
                if r.suspected_pr:
                    suspected += f" (PR #{r.suspected_pr})"
                suspected = suspected or "N/A"

                test_display = r.target.test_file
                if len(test_display) > 30:
                    test_display = "..." + test_display[-27:]

                job_display = r.target.job_name
                if len(job_display) > 30:
                    job_display = "..." + job_display[-27:]

                lines.append(
                    f"| {emoji} {cls} | `{test_display}` | `{job_display}` | "
                    f"{r.confidence} | {suspected} | {r.recommended_fix[:80]} |"
                )

    total_tokens = sum(r.tokens_used for r in results)
    lines.append(
        f"\n---\n*Analyzed {len(results)} failures using {total_tokens} tokens*"
    )

    with open(summary_path, "a") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

VALID_CLASSIFICATIONS = {
    "code_regression",
    "flaky_test",
    "hardware_issue",
    "environment_change",
}


def run_bisection_analysis(
    github_token: str,
    api_key: str,
    max_failures: int = 10,
    min_streak: int = 2,
    output_file: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Main orchestration: fetch failures, gather context, call Claude, report."""
    print("=" * 80)
    print("SGLang CI Auto Bisect")
    print("=" * 80)

    # Load bisect skill methodology for prompt construction
    # Raises SkillLoadError if SKILL.md is missing or sections were renamed
    skill_content = load_bisect_skill()
    print(f"Loaded bisect skill ({len(skill_content)} chars)")

    # Fetch and analyze failures directly (no external report file needed)
    targets, logs_cache = analyze_scheduled_failures(
        github_token, min_streak, max_failures
    )
    print(f"\n{len(targets)} failure targets to analyze")

    if not targets:
        print("No failures requiring bisection analysis.")
        output = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_failures_analyzed": 0,
            "total_tokens_used": 0,
            "results": [],
            "summary": {
                "code_regressions": 0,
                "flaky_tests": 0,
                "hardware_issues": 0,
                "environment_changes": 0,
                "unknown": 0,
            },
        }
        if output_file:
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
        generate_github_summary([])
        return output

    for i, t in enumerate(targets, 1):
        print(f"  [{i}] {t.test_file} in {t.job_name} (streak: {t.current_streak})")

    # Process each target
    results: List[BisectionResult] = []
    total_tokens = 0

    for i, target in enumerate(targets, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(targets)}] Analyzing: {target.test_file}")
        print(f"  Job: {target.job_name}")
        print(f"  Streak: {target.current_streak} (test-level: {target.test_streak})")
        print(f"  SHA range: {target.first_failure_sha}..{target.last_failure_sha}")

        context = gather_bisection_context(target, github_token, logs_cache)

        if dry_run:
            prompt = build_prompt(context, skill_content)
            print("  [DRY RUN] Skipping Claude API call")
            print(f"  Prompt length: {len(prompt)} chars")
            result = BisectionResult(
                target=target,
                classification="dry_run",
                confidence="n/a",
                evidence_summary="Dry run - no API call made",
                recommended_fix="N/A",
            )
            results.append(result)
            continue

        prompt = build_prompt(context, skill_content)
        print(f"  Calling Claude ({CLAUDE_MODEL})...")
        response_text, tokens = call_claude_api(prompt, api_key)
        total_tokens += tokens
        print(f"  Tokens used: {tokens}")

        parsed = parse_claude_response(response_text)
        classification = parsed.get("classification", "unknown")
        if classification not in VALID_CLASSIFICATIONS:
            classification = "unknown"

        result = BisectionResult(
            target=target,
            classification=classification,
            confidence=parsed.get("confidence", "low"),
            suspected_commit=parsed.get("suspected_commit"),
            suspected_pr=parsed.get("suspected_pr"),
            evidence_summary=parsed.get("evidence_summary", ""),
            recommended_fix=parsed.get("recommended_fix", ""),
            raw_response=response_text,
            tokens_used=tokens,
        )
        results.append(result)

        print(f"  Classification: {result.classification} ({result.confidence})")
        if result.suspected_commit:
            print(f"  Suspected commit: {result.suspected_commit}")
        print(f"  Evidence: {result.evidence_summary[:100]}...")

        if i < len(targets):
            time.sleep(1)

    # Aggregate
    summary = {
        "code_regressions": sum(
            1 for r in results if r.classification == "code_regression"
        ),
        "flaky_tests": sum(1 for r in results if r.classification == "flaky_test"),
        "hardware_issues": sum(
            1 for r in results if r.classification == "hardware_issue"
        ),
        "environment_changes": sum(
            1 for r in results if r.classification == "environment_change"
        ),
        "unknown": sum(1 for r in results if r.classification == "unknown"),
    }

    output = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_failures_analyzed": len(results),
        "total_tokens_used": total_tokens,
        "results": [
            {
                "target": asdict(r.target),
                "classification": r.classification,
                "confidence": r.confidence,
                "suspected_commit": r.suspected_commit,
                "suspected_pr": r.suspected_pr,
                "evidence_summary": r.evidence_summary,
                "recommended_fix": r.recommended_fix,
                "tokens_used": r.tokens_used,
            }
            for r in results
        ],
        "summary": summary,
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")

    generate_github_summary(results)

    print(f"\n{'=' * 80}")
    print("BISECTION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total failures analyzed: {len(results)}")
    print(f"Total tokens used: {total_tokens}")
    for cls, count in summary.items():
        if count > 0:
            print(f"  {cls}: {count}")

    return output


def main():
    parser = argparse.ArgumentParser(description="SGLang CI Auto Bisect")
    parser.add_argument(
        "--github-token",
        required=True,
        help="GitHub token for API access",
    )
    parser.add_argument(
        "--anthropic-api-key",
        required=True,
        help="Anthropic API key for Claude",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=10,
        help="Maximum number of failures to analyze (default: 10)",
    )
    parser.add_argument(
        "--min-streak",
        type=int,
        default=2,
        help="Minimum consecutive failure streak to trigger bisection (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Claude API calls, only gather context",
    )
    args = parser.parse_args()

    try:
        run_bisection_analysis(
            github_token=args.github_token,
            api_key=args.anthropic_api_key,
            max_failures=args.max_failures,
            min_streak=args.min_streak,
            output_file=args.output,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"Error during bisection analysis: {e}")
        import traceback

        traceback.print_exc()

        # Write an error result file so the Slack notification step can
        # report the failure instead of silently skipping
        if args.output:
            error_output = {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_failures_analyzed": 0,
                "total_tokens_used": 0,
                "error": str(e),
                "results": [],
                "summary": {
                    "code_regressions": 0,
                    "flaky_tests": 0,
                    "hardware_issues": 0,
                    "environment_changes": 0,
                    "unknown": 0,
                },
            }
            try:
                with open(args.output, "w") as f:
                    json.dump(error_output, f, indent=2)
                print(f"Error report saved to {args.output}")
            except OSError:
                pass

        sys.exit(1)


if __name__ == "__main__":
    main()
