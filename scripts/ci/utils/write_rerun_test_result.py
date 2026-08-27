#!/usr/bin/env python3
"""
Write back the result of a Rerun Test workflow run into its parent reply
comment, by replacing the hourglass placeholder on the marker's line with
a success/failure icon. The leading rocket on the line is preserved as a
visual anchor that this line came from a slash-command trigger.

Concurrent dispatches against the same reply comment are serialized at the
GitHub Actions layer via job-level `concurrency`, so a simple
read-modify-write here is sufficient. Marker is rewritten to a `:done`
variant to make the operation idempotent against accidental reruns.
"""

import argparse
import os
import sys
import time

import requests

# When the marker isn't found, retry a few times before giving up. The
# usual cause is a brief race where the dispatched workflow finishes its
# writeback step before the handler edits the placeholder comment with the
# final body containing markers. If retries don't help (e.g. the handler
# failed to edit the placeholder at all), we still return 0 — failing here
# would amplify a single placeholder-edit failure into N noisy finalizer
# job failures, one per dispatched batch.
RETRY_DELAYS_SEC = [0, 5, 15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comment-id", required=True, type=int)
    ap.add_argument(
        "--marker", required=True, help="Per-batch marker, e.g. <!--rrt:0-->"
    )
    ap.add_argument("--status", required=True, choices=["success", "failure"])
    ap.add_argument("--repo", required=True, help="owner/repo")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set")
        return 1

    icon = "✅" if args.status == "success" else "❌"
    done_marker = args.marker.replace("-->", ":done-->")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{args.repo}/issues/comments/{args.comment_id}"

    body = None
    for attempt, delay in enumerate(RETRY_DELAYS_SEC):
        if delay:
            time.sleep(delay)
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"GET failed: {resp.status_code} {resp.text}")
            return 1
        body = resp.json().get("body") or ""
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
            f"{len(RETRY_DELAYS_SEC)} attempts; skipping writeback. "
            f"The handler may have failed to edit the placeholder comment."
        )
        return 0

    new_lines = []
    for line in body.splitlines(keepends=True):
        if args.marker in line:
            line = line.replace("⏳", icon, 1)
            line = line.replace(args.marker, done_marker)
        new_lines.append(line)
    new_body = "".join(new_lines)

    patch = requests.patch(
        url,
        headers=headers,
        json={"body": new_body},
        timeout=15,
    )
    if patch.status_code != 200:
        print(f"PATCH failed: {patch.status_code} {patch.text}")
        return 1
    print(f"Updated comment {args.comment_id}: {args.marker} -> {icon}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
