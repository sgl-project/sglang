#!/usr/bin/env python3
"""Client for submitting SGLang MLU CI work to a Cambricon MLU CI bridge.

The script intentionally mirrors the mlu-ops GitHub Actions client: it posts a
small task description to a local bridge service, polls by task id, and maps the
returned status to the GitHub Actions exit code. The bridge/master/Jenkins side
owns the actual MLU environment, image, and test execution.
"""

import argparse
import json
import os
import sys
import time
from typing import Optional
from urllib import error, parse, request

DEFAULT_BRIDGE_URL = "http://localhost:14547"
# "unstable" is a terminal failure state reported by the external runner.
TERMINAL_MARKERS = ("success", "fail", "error", "unstable")


def _request_json(url: str, payload: Optional[dict] = None, timeout: int = 30) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers)
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        if not body:
            return {}
        return json.loads(body)


def _is_terminal(status: str) -> bool:
    return any(marker in status for marker in TERMINAL_MARKERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit and monitor SGLang MLU CI")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--trigger-id", required=True)
    parser.add_argument("--pr-id", default="")
    parser.add_argument("--repo-url", default="")
    parser.add_argument("--git-ref", default="")
    parser.add_argument("--commit-sha", default="")
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--trigger-type", default="ci")
    parser.add_argument("--repeat-times", default="3")
    parser.add_argument(
        "--bridge-url",
        default=os.environ.get("SGLANG_MLU_CI_BRIDGE_URL", DEFAULT_BRIDGE_URL),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=int(os.environ.get("SGLANG_MLU_CI_POLL_INTERVAL", "10")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("SGLANG_MLU_CI_TIMEOUT_SECONDS", "21600")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bridge_url = args.bridge_url.rstrip("/")
    payload = {
        "timestamp": args.timestamp,
        "repo": args.repo,
        "pr_id": args.pr_id,
        "repo_url": args.repo_url,
        "git_ref": args.git_ref,
        "commit_sha": args.commit_sha,
        "trigger_type": args.trigger_type,
        "trigger_id": args.trigger_id,
        "repeat_times": args.repeat_times,
        "status": "running",
    }

    print(f"Submitting MLU CI task to {bridge_url}: {payload}", flush=True)
    try:
        task_obj = _request_json(bridge_url, payload)
    except Exception as exc:
        print(f"Failed to submit MLU CI task: {exc}", file=sys.stderr)
        return 1

    task_id = task_obj.get("id")
    if not task_id:
        print(f"Bridge response does not contain task id: {task_obj}", file=sys.stderr)
        return 1
    print(f"MLU CI task id: {task_id}", flush=True)

    start = time.monotonic()
    last_status = None
    while True:
        if time.monotonic() - start > args.timeout_seconds:
            print(
                f"Timed out waiting for MLU CI task {task_id} after "
                f"{args.timeout_seconds} seconds",
                file=sys.stderr,
            )
            return 1

        query = parse.urlencode({"aiming": "get_status", "id": task_id})
        try:
            result = _request_json(f"{bridge_url}/{query}")
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"Failed to query MLU CI task {task_id}: {exc}", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        status = str(result.get("status", ""))
        if status != last_status:
            print(f"MLU CI task {task_id} status: {status}", flush=True)
            last_status = status

        if _is_terminal(status):
            log = result.get("log", "")
            if log:
                print(log)
            end_query = parse.urlencode({"aiming": "end_job", "id": task_id})
            try:
                _request_json(f"{bridge_url}/{end_query}")
            except Exception as exc:
                print(f"Failed to end MLU CI task {task_id}: {exc}", file=sys.stderr)
            return 0 if "success" in status else 1

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
