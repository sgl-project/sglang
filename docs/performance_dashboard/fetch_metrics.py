#!/usr/bin/env python3
"""
Fetch and process SGLang nightly test metrics from GitHub Actions artifacts.

This script fetches consolidated metrics from GitHub Actions workflow runs
and outputs them as JSON for the performance dashboard.

Usage:
    python fetch_metrics.py --output metrics_data.json
    python fetch_metrics.py --output metrics_data.json --days 30
    python fetch_metrics.py --output metrics_data.json --run-id 21338741812
"""

import argparse
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

GITHUB_REPO = "sgl-project/sglang"
WORKFLOW_NAME = "nightly-test-nvidia.yml"
ARTIFACT_PREFIX = "consolidated-metrics-"


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or gh CLI."""
    # Check environment variable first
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    # Try gh CLI
    try:
        import subprocess

        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def get_headers(token: Optional[str]) -> dict:
    """Get request headers with optional authentication."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_workflow_runs(
    token: Optional[str],
    days: int = 30,
    event: Optional[str] = None,
) -> list:
    """Fetch completed workflow runs from GitHub Actions."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_NAME}/runs"

    params = {
        "status": "completed",
        "per_page": 100,
    }

    if event:
        params["event"] = event

    response = requests.get(url, headers=get_headers(token), params=params, timeout=30)
    response.raise_for_status()

    runs = response.json().get("workflow_runs", [])

    # Filter by date
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    runs = [
        run
        for run in runs
        if datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) > cutoff
    ]

    return runs


def fetch_run_artifacts(token: Optional[str], run_id: int) -> list:
    """Fetch artifacts for a specific workflow run."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}/artifacts"

    response = requests.get(url, headers=get_headers(token), timeout=30)
    response.raise_for_status()

    return response.json().get("artifacts", [])


def download_artifact(token: Optional[str], artifact_id: int) -> Optional[bytes]:
    """Download an artifact by ID."""
    if not token:
        print(f"Warning: GitHub token required to download artifacts", file=sys.stderr)
        return None

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/artifacts/{artifact_id}/zip"

    headers = get_headers(token)
    response = requests.get(url, headers=headers, allow_redirects=True, timeout=60)

    if response.status_code == 200:
        return response.content

    print(
        f"Failed to download artifact {artifact_id}: {response.status_code}",
        file=sys.stderr,
    )
    return None


def extract_metrics_from_zip(zip_content: bytes) -> Optional[dict]:
    """Extract metrics JSON from a zip file."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            # Find the JSON file in the archive
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not json_files:
                return None

            with zf.open(json_files[0]) as f:
                return json.load(f)
    except (zipfile.BadZipFile, json.JSONDecodeError) as e:
        print(f"Failed to extract metrics: {e}", file=sys.stderr)
        return None


def fetch_metrics_for_run(token: Optional[str], run: dict) -> Optional[dict]:
    """Fetch metrics for a single workflow run."""
    run_id = run["id"]
    print(f"Fetching metrics for run {run_id}...", file=sys.stderr)

    artifacts = fetch_run_artifacts(token, run_id)

    # Find consolidated metrics artifact
    metrics_artifact = None
    for artifact in artifacts:
        if artifact["name"].startswith(ARTIFACT_PREFIX):
            metrics_artifact = artifact
            break

    if not metrics_artifact:
        print(f"No consolidated metrics found for run {run_id}", file=sys.stderr)
        return None

    # Download and extract
    zip_content = download_artifact(token, metrics_artifact["id"])
    if not zip_content:
        return None

    metrics = extract_metrics_from_zip(zip_content)
    if not metrics:
        return None

    # Ensure required fields are present
    if "run_id" not in metrics:
        metrics["run_id"] = str(run_id)
    if "run_date" not in metrics:
        metrics["run_date"] = run["created_at"]
    if "commit_sha" not in metrics:
        metrics["commit_sha"] = run["head_sha"]
    if "branch" not in metrics:
        metrics["branch"] = run["head_branch"]

    return metrics


def fetch_single_run(token: Optional[str], run_id: int) -> Optional[dict]:
    """Fetch metrics for a single run by ID."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}"

    response = requests.get(url, headers=get_headers(token), timeout=30)
    response.raise_for_status()

    run = response.json()
    return fetch_metrics_for_run(token, run)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch SGLang nightly test metrics from GitHub Actions"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="metrics_data.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to fetch (default: 30)",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Fetch a specific run by ID",
    )
    parser.add_argument(
        "--event",
        type=str,
        choices=["schedule", "workflow_dispatch", "push"],
        help="Filter by trigger event type",
    )
    parser.add_argument(
        "--scheduled-only",
        action="store_true",
        help="Only fetch scheduled (nightly) runs",
    )

    args = parser.parse_args()

    token = get_github_token()
    if not token:
        print(
            "Warning: No GitHub token found. Some features may be limited.",
            file=sys.stderr,
        )
        print(
            "Set GITHUB_TOKEN env var or login with 'gh auth login'",
            file=sys.stderr,
        )

    all_metrics = []

    if args.run_id:
        # Fetch single run
        metrics = fetch_single_run(token, args.run_id)
        if metrics:
            all_metrics.append(metrics)
    else:
        # Fetch multiple runs
        event = "schedule" if args.scheduled_only else args.event
        runs = fetch_workflow_runs(token, days=args.days, event=event)
        print(f"Found {len(runs)} workflow runs", file=sys.stderr)

        for run in runs:
            metrics = fetch_metrics_for_run(token, run)
            if metrics:
                all_metrics.append(metrics)

    # Sort by date descending
    all_metrics.sort(key=lambda x: x.get("run_date", ""), reverse=True)

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Wrote {len(all_metrics)} metrics records to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
