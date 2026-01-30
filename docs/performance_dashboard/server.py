#!/usr/bin/env python3
"""
Simple development server for the SGLang Performance Dashboard.

This server:
1. Serves the static HTML/JS files
2. Provides an API endpoint to fetch metrics from GitHub
3. Caches metrics data to reduce API calls

Usage:
    python server.py
    python server.py --port 8080
    python server.py --host 0.0.0.0  # Allow external access
    python server.py --fetch-on-start
"""

import argparse
import http.server
import io
import json
import os
import socketserver
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

GITHUB_REPO = "sgl-project/sglang"
WORKFLOW_NAME = "nightly-test-nvidia.yml"
ARTIFACT_PREFIX = "consolidated-metrics-"

# Cache for metrics data with thread-safe lock
cache_lock = threading.Lock()
metrics_cache = {
    "data": [],
    "last_updated": None,
    "updating": False,
}

CACHE_TTL = 300  # 5 minutes
REQUEST_TIMEOUT = 30  # seconds


def get_github_token():
    """Get GitHub token from environment or gh CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

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


def fetch_metrics_from_github(days=30):
    """Fetch metrics from GitHub Actions artifacts."""
    token = get_github_token()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Get workflow runs - only scheduled (nightly) runs, not workflow_dispatch
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_NAME}/runs"
    params = {"status": "completed", "per_page": 50, "event": "schedule"}

    try:
        response = requests.get(
            url, headers=headers, params=params, timeout=REQUEST_TIMEOUT
        )
        if not response.ok:
            print(f"Failed to fetch workflow runs: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching workflow runs: {e}")
        return []

    runs = response.json().get("workflow_runs", [])

    # Filter by date
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    runs = [
        run
        for run in runs
        if datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) > cutoff
    ]

    all_metrics = []

    for run in runs[:20]:  # Limit to 20 most recent
        run_id = run["id"]

        # Get artifacts
        artifacts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}/artifacts"
        try:
            artifacts_resp = requests.get(
                artifacts_url, headers=headers, timeout=REQUEST_TIMEOUT
            )
            if not artifacts_resp.ok:
                continue
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching artifacts for run {run_id}: {e}")
            continue

        artifacts = artifacts_resp.json().get("artifacts", [])

        # Find consolidated metrics
        for artifact in artifacts:
            if artifact["name"].startswith(ARTIFACT_PREFIX):
                if not token:
                    # Without token, we can't download - return metadata only
                    all_metrics.append(
                        {
                            "run_id": str(run_id),
                            "run_date": run["created_at"],
                            "commit_sha": run["head_sha"],
                            "branch": run["head_branch"],
                            "results": [],
                        }
                    )
                    break

                # Download artifact
                download_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/artifacts/{artifact['id']}/zip"
                try:
                    download_resp = requests.get(
                        download_url,
                        headers=headers,
                        allow_redirects=True,
                        timeout=REQUEST_TIMEOUT,
                    )
                except requests.exceptions.RequestException as e:
                    print(f"Network error downloading artifact: {e}")
                    break

                if download_resp.ok:
                    try:
                        with zipfile.ZipFile(io.BytesIO(download_resp.content)) as zf:
                            json_files = [
                                f for f in zf.namelist() if f.endswith(".json")
                            ]
                            if json_files:
                                with zf.open(json_files[0]) as f:
                                    metrics = json.load(f)
                                    # Ensure required fields
                                    metrics.setdefault("run_id", str(run_id))
                                    metrics.setdefault("run_date", run["created_at"])
                                    metrics.setdefault("commit_sha", run["head_sha"])
                                    metrics.setdefault("branch", run["head_branch"])
                                    all_metrics.append(metrics)
                    except (zipfile.BadZipFile, json.JSONDecodeError) as e:
                        print(f"Failed to process artifact: {e}")
                break

    return all_metrics


def update_cache_async():
    """Update the metrics cache in background with thread safety."""
    with cache_lock:
        if metrics_cache["updating"]:
            return
        metrics_cache["updating"] = True

    try:
        data = fetch_metrics_from_github()
        with cache_lock:
            metrics_cache["data"] = data
            metrics_cache["last_updated"] = time.time()
        print(f"Cache updated with {len(data)} metrics records")
    finally:
        with cache_lock:
            metrics_cache["updating"] = False


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # Prevent directory traversal attacks
        if ".." in parsed.path or parsed.path.startswith("//"):
            self.send_error(400, "Invalid path")
            return

        if parsed.path == "/api/metrics":
            self.handle_metrics_api(parsed)
        elif parsed.path == "/api/refresh":
            self.handle_refresh_api()
        else:
            super().do_GET()

    def handle_metrics_api(self, parsed):
        """Handle /api/metrics endpoint."""
        # Check cache with thread safety
        with cache_lock:
            cache_valid = (
                metrics_cache["last_updated"]
                and time.time() - metrics_cache["last_updated"] < CACHE_TTL
            )
            data = metrics_cache["data"].copy()

        if not cache_valid:
            # Trigger background update
            threading.Thread(target=update_cache_async, daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def handle_refresh_api(self):
        """Handle /api/refresh endpoint."""
        threading.Thread(target=update_cache_async, daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "refreshing"}).encode())

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="SGLang Performance Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (use 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--fetch-on-start", action="store_true", help="Fetch metrics on startup"
    )
    args = parser.parse_args()

    # Change to dashboard directory
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)

    if args.fetch_on_start:
        print("Fetching initial metrics data...")
        update_cache_async()

    handler = lambda *a, **kw: DashboardHandler(*a, directory=str(dashboard_dir), **kw)

    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving dashboard at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
