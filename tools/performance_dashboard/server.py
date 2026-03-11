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
    python server.py --username admin --password secret  # Enable authentication
    DASHBOARD_USERNAME=admin DASHBOARD_PASSWORD=secret python server.py  # Via env vars
    python server.py --refresh-interval 12  # Auto-refresh data every 12 hours
"""

import argparse
import hashlib
import hmac
import http.server
import io
import json
import os
import secrets
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

# Authentication configuration (set via CLI flags)
auth_config = {
    "enabled": False,
    "username": None,
    "password_hash": None,  # SHA-256 hash of the password
    "active_tokens": {},  # token -> expiry timestamp
}
auth_lock = threading.Lock()
AUTH_TOKEN_TTL = 3600  # 1 hour


def hash_password(password):
    """Hash a password using SHA-256 for constant-time comparison."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_auth_token():
    """Create a new session token."""
    token = secrets.token_hex(32)
    with auth_lock:
        # Clean up expired tokens
        now = time.time()
        auth_config["active_tokens"] = {
            t: exp for t, exp in auth_config["active_tokens"].items() if exp > now
        }
        auth_config["active_tokens"][token] = now + AUTH_TOKEN_TTL
    return token


def verify_auth_token(token):
    """Verify a session token is valid and not expired."""
    if not token:
        return False
    with auth_lock:
        expiry = auth_config["active_tokens"].get(token)
        if expiry and expiry > time.time():
            return True
        # Remove expired token
        auth_config["active_tokens"].pop(token, None)
        return False


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


def start_periodic_refresh(interval_hours):
    """Start a background thread that refreshes the cache periodically."""
    interval_seconds = interval_hours * 3600

    def refresh_loop():
        while True:
            time.sleep(interval_seconds)
            print(f"Periodic refresh triggered (every {interval_hours}h)")
            update_cache_async()

    thread = threading.Thread(target=refresh_loop, daemon=True)
    thread.start()
    print(f"Periodic refresh enabled: every {interval_hours} hours")


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def _send_json(self, data, status=200):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _check_auth(self):
        """Check if request is authenticated. Returns True if OK, sends 401 and returns False otherwise."""
        if not auth_config["enabled"]:
            return True
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if verify_auth_token(token):
                return True
        self._send_json({"error": "Unauthorized"}, status=401)
        return False

    def do_GET(self):
        parsed = urlparse(self.path)

        # Prevent directory traversal attacks
        if ".." in parsed.path or parsed.path.startswith("//"):
            self.send_error(400, "Invalid path")
            return

        if parsed.path == "/api/auth-check":
            self.handle_auth_check()
        elif parsed.path == "/api/metrics":
            if self._check_auth():
                self.handle_metrics_api(parsed)
        elif parsed.path == "/api/refresh":
            if self._check_auth():
                self.handle_refresh_api()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/login":
            self.handle_login()
        else:
            self.send_error(404, "Not Found")

    def handle_auth_check(self):
        """Tell the frontend whether authentication is required."""
        self._send_json({"auth_required": auth_config["enabled"]})

    def handle_login(self):
        """Validate username/password and return a session token."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0 or content_length > 4096:
            self._send_json({"error": "Invalid request"}, status=400)
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, ValueError):
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        username = body.get("username", "")
        password = body.get("password", "")

        if hmac.compare_digest(
            username, auth_config["username"]
        ) and hmac.compare_digest(
            hash_password(password), auth_config["password_hash"]
        ):
            token = create_auth_token()
            self._send_json({"token": token})
        else:
            self._send_json({"error": "Invalid username or password"}, status=401)

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

        self._send_json(data)

    def handle_refresh_api(self):
        """Handle /api/refresh endpoint."""
        threading.Thread(target=update_cache_async, daemon=True).start()
        self._send_json({"status": "refreshing"})

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
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=12,
        help="Auto-refresh interval in hours (default: 12, set to 0 to disable)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("DASHBOARD_USERNAME"),
        help="Username for dashboard authentication (or set DASHBOARD_USERNAME env var)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("DASHBOARD_PASSWORD"),
        help="Password for dashboard authentication (or set DASHBOARD_PASSWORD env var)",
    )
    args = parser.parse_args()

    # Configure authentication if both username and password are provided
    if args.username and args.password:
        auth_config["enabled"] = True
        auth_config["username"] = args.username
        auth_config["password_hash"] = hash_password(args.password)
        print(f"Authentication enabled for user: {args.username}")
    elif args.username or args.password:
        parser.error("Both --username and --password must be provided together")

    # Change to dashboard directory
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)

    if args.fetch_on_start:
        print("Fetching initial metrics data...")
        update_cache_async()

    if args.refresh_interval > 0:
        start_periodic_refresh(args.refresh_interval)

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
