#!/usr/bin/env python3
"""
Update the per-batch status icon in a /rerun-test reply comment.

State machine for one batch line (anchored by a unique HTML-comment marker
written by the slash-command handler):

  dispatched   ⏳ ... <!--rrt:i-->          (handler, on dispatch)
  running      🔄 ... <!--rrt:i-->          (start-beacon, on the test runner)
  done         ✅/❌ ... <!--rrt:i:done-->  (finalizer, after the test job)

The leading 🚀 on the line is the visual anchor that this came from a
slash-command trigger and is preserved across all states.

Idempotency:
- :done marker present  -> no-op (covers reruns and start-after-finalizer race)
- running and line already has 🔄  -> no-op
- marker not found after retries  -> warn and exit 0 so a single placeholder
  glitch does not amplify into N noisy job failures.

Concurrent updates against the same comment from different batches can race
on the body (read-modify-write of the same field). The finalizer
serializes itself via job-level concurrency; the beacon does not, because
splitting it into its own job would re-queue the GPU runner. Worst case
for the beacon race is one missed 🔄 flicker - comment stays consistent.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

RETRY_DELAYS_SEC = [0, 5, 15]

STATUS_ICONS = {
    "running": "🔄",
    "success": "✅",
    "failure": "❌",
}

TERMINAL_STATUSES = {"success", "failure"}


def gh_request(method, url, token, body=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comment-id", required=True, type=int)
    ap.add_argument(
        "--marker", required=True, help="Per-batch marker, e.g. <!--rrt:0-->"
    )
    ap.add_argument(
        "--status",
        required=True,
        choices=list(STATUS_ICONS.keys()),
    )
    ap.add_argument("--repo", required=True, help="owner/repo")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set")
        return 1

    icon = STATUS_ICONS[args.status]
    is_terminal = args.status in TERMINAL_STATUSES
    done_marker = args.marker.replace("-->", ":done-->")

    url = f"https://api.github.com/repos/{args.repo}/issues/comments/{args.comment_id}"

    body = None
    for attempt, delay in enumerate(RETRY_DELAYS_SEC):
        if delay:
            time.sleep(delay)
        status, text = gh_request("GET", url, token)
        if status != 200:
            print(f"GET failed: {status} {text}")
            return 1
        body = json.loads(text).get("body") or ""
        if done_marker in body:
            print(f"Marker {done_marker} already present; nothing to do.")
            return 0
        if args.marker in body:
            break
        print(
            f"Marker {args.marker} not found "
            f"(attempt {attempt + 1}/{len(RETRY_DELAYS_SEC)}); will retry."
        )
    else:
        print(
            f"WARNING: marker {args.marker} not found after "
            f"{len(RETRY_DELAYS_SEC)} attempts; skipping. "
            f"The handler may have failed to edit the placeholder comment."
        )
        return 0

    new_lines = []
    for line in body.splitlines(keepends=True):
        if args.marker in line:
            if not is_terminal and icon in line:
                print(f"Line already has {icon}; nothing to do.")
                return 0
            for prior in ("⏳", "🔄"):
                if prior in line:
                    line = line.replace(prior, icon, 1)
                    break
            if is_terminal:
                line = line.replace(args.marker, done_marker)
        new_lines.append(line)

    new_body = "".join(new_lines)
    status, text = gh_request("PATCH", url, token, body={"body": new_body})
    if status != 200:
        print(f"PATCH failed: {status} {text}")
        return 1
    print(f"Updated comment {args.comment_id}: {args.marker} -> {icon}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
