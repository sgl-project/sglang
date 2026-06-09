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
from pathlib import Path
from typing import Optional
from urllib import error, parse, request

DEFAULT_BRIDGE_URL = "http://localhost:14547"
# "unstable" is a terminal failure state reported by the external runner.
TERMINAL_MARKERS = ("success", "fail", "error", "unstable")
DEFAULT_LOG_DIR = "mlu-ci-logs"
LOG_DOWNLOAD_RETRIES = 3
PR_TIMEOUT_SECONDS = 19800
NIGHTLY_TIMEOUT_SECONDS = 27000


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


def _request_text(url: str, timeout: int = 60) -> str:
    with request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _is_terminal(status: str) -> bool:
    return any(marker in status for marker in TERMINAL_MARKERS)


def _should_cleanup_remote_task(
    status: str,
    log_path: Optional[Path],
    log_is_complete: bool,
) -> bool:
    return "success" in status and log_path is not None and log_is_complete


def _download_jenkins_log(bridge_url: str, task_id: str, log_dir: str) -> Optional[Path]:
    query = parse.urlencode({"aiming": "get_log", "id": task_id})
    log_path = Path(log_dir) / f"jenkins-{task_id}.log"
    for attempt in range(1, LOG_DOWNLOAD_RETRIES + 1):
        try:
            text = _request_text(f"{bridge_url}/{query}")
            break
        except Exception as exc:
            print(
                f"Failed to download MLU Jenkins log for task {task_id} "
                f"(attempt {attempt}/{LOG_DOWNLOAD_RETRIES}): {exc}",
                file=sys.stderr,
            )
            if attempt < LOG_DOWNLOAD_RETRIES:
                time.sleep(3)
    else:
        print(
            "MLU Jenkins log download failed; keep the remote task for retention-based cleanup.",
            file=sys.stderr,
        )
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    print(f"Saved MLU Jenkins log to {log_path}", flush=True)
    return log_path


def _write_result_metadata(
    log_dir: str,
    task_id: str,
    status: str,
    failure_type: str,
    failure_stage: str,
) -> Path:
    metadata_path = Path(log_dir) / f"result-{task_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "status": status,
                "failure_type": failure_type,
                "failure_stage": failure_stage,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved MLU CI result metadata to {metadata_path}", flush=True)
    return metadata_path


def _append_failure_classification(
    log_path: Optional[Path],
    failure_type: str,
    failure_stage: str,
) -> None:
    if log_path is None or not log_path.is_file():
        return
    if not failure_type and not failure_stage:
        return

    line = (
        "MLU CI failure classification: "
        f"type={failure_type or 'unknown'}, "
        f"stage={failure_stage or 'unknown'}"
    )
    encoded_line = (line + "\n").encode("utf-8")
    log_size = log_path.stat().st_size
    with log_path.open("rb") as log_file:
        log_file.seek(max(0, log_size - len(encoded_line)))
        if log_file.read().endswith(encoded_line):
            return
        needs_newline = False
        if log_size:
            log_file.seek(-1, os.SEEK_END)
            needs_newline = log_file.read(1) != b"\n"

    with log_path.open("ab") as log_file:
        if needs_newline:
            log_file.write(b"\n")
        log_file.write(encoded_line)


def _print_log_progress(bridge_url: str, task_id: str,
                        last_fetched: int, log_size: int) -> int:
    """Fetch and print new log content from the log file since *last_fetched*.
    Returns the new *last_fetched* byte offset (may be unchanged on failure)."""
    if log_size <= last_fetched:
        return last_fetched
    query = parse.urlencode({
        "aiming": "get_log", "id": task_id, "start": str(last_fetched),
    })
    try:
        chunk = _request_text(f"{bridge_url}/{query}")
    except Exception as exc:
        print(f"Failed to fetch MLU Jenkins log chunk: {exc}", file=sys.stderr)
        return last_fetched
    if not chunk:
        return last_fetched
    print(chunk.rstrip(), flush=True)
    # Advance by the actual bytes received, not by the reported file size.
    # get_status's log_size may be ahead of what is actually on disk.
    return last_fetched + len(chunk.encode("utf-8"))


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
        default=(
            int(os.environ["SGLANG_MLU_CI_TIMEOUT_SECONDS"])
            if "SGLANG_MLU_CI_TIMEOUT_SECONDS" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("SGLANG_MLU_CI_LOG_DIR", DEFAULT_LOG_DIR),
        help="Directory where the downloaded Jenkins log is saved for upload-artifact.",
    )
    args = parser.parse_args()
    if args.timeout_seconds is None:
        args.timeout_seconds = (
            NIGHTLY_TIMEOUT_SECONDS
            if args.trigger_type == "nightly"
            else PR_TIMEOUT_SECONDS
        )
    return args


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
    last_fetched = 0
    while True:
        if time.monotonic() - start > args.timeout_seconds:
            print(
                f"Timed out waiting for MLU CI task {task_id} after "
                f"{args.timeout_seconds} seconds",
                file=sys.stderr,
            )
            print(
                "MLU CI failure classification: type=timeout, stage=github_client",
                file=sys.stderr,
            )
            _write_result_metadata(
                args.log_dir,
                task_id,
                "error",
                "timeout",
                "github_client",
            )
            log_path = _download_jenkins_log(bridge_url, task_id, args.log_dir)
            if log_path and log_path.is_file():
                full = log_path.read_text(encoding="utf-8", errors="replace")
                if len(full) > last_fetched:
                    remaining = full[last_fetched:]
                    if remaining.strip():
                        print(remaining.rstrip(), flush=True)
            _append_failure_classification(log_path, "timeout", "github_client")
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

        # Fetch and print incremental log content from the log file
        log_size = int(result.get("log_size", 0))
        last_fetched = _print_log_progress(
            bridge_url, task_id, last_fetched, log_size,
        )

        if _is_terminal(status):
            log_status = str(result.get("log_status", ""))
            log_error = str(result.get("log_error", ""))
            failure_type = str(result.get("failure_type", ""))
            failure_stage = str(result.get("failure_stage", ""))
            if failure_type or failure_stage:
                print(
                    "MLU CI failure classification: "
                    f"type={failure_type or 'unknown'}, "
                    f"stage={failure_stage or 'unknown'}",
                    flush=True,
                )
            _write_result_metadata(
                args.log_dir,
                task_id,
                status,
                failure_type,
                failure_stage,
            )
            if log_status:
                print(f"MLU CI task {task_id} Jenkins log status: {log_status}", flush=True)
            if log_status == "failed":
                print(
                    f"MLU Jenkins log sync failed; downloaded artifact may be incomplete. {log_error}",
                    file=sys.stderr,
                )
            log_path = _download_jenkins_log(bridge_url, task_id, args.log_dir)
            # Print any log content that arrived after the last poll, so the
            # Actions page output matches the artifact.
            if log_path and log_path.is_file():
                full = log_path.read_text(encoding="utf-8", errors="replace")
                if len(full) > last_fetched:
                    remaining = full[last_fetched:]
                    if remaining.strip():
                        print(remaining.rstrip(), flush=True)
            _append_failure_classification(log_path, failure_type, failure_stage)
            log_is_complete = log_status in {"", "complete"}
            if _should_cleanup_remote_task(status, log_path, log_is_complete):
                end_query = parse.urlencode({"aiming": "end_job", "id": task_id})
                try:
                    _request_json(f"{bridge_url}/{end_query}")
                except Exception as exc:
                    print(f"Failed to end MLU CI task {task_id}: {exc}", file=sys.stderr)
            else:
                print(
                    f"Skip end_job for MLU CI task {task_id}; failed or incomplete "
                    "tasks are kept until retention cleanup.",
                    file=sys.stderr,
                )
            return 0 if "success" in status else 1

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
