"""Refresh marked CI-command replies from live workflow runs, never PR code.

Like /rerun-test replies, each line has a hidden marker. The minimum attempt
prevents a newly requested rerun from displaying the previous attempt's result.
All writers use ci-command-status.yml's per-PR concurrency group and refresh all
marked runs, so coalesced events and multiple workflows in one reply are safe.
"""

import argparse
import json
import os
import re
from pathlib import Path

MARKER = re.compile(r"<!--ci-run:(\d+):(\d+)-->")
ICONS = {
    "queued": "⏳",
    "in_progress": "🔄",
    "success": "✅",
    "failure": "❌",
    "cancelled": "🚫",
    "timed_out": "❌",
    "skipped": "⏭️",
    "neutral": "➖",
    "action_required": "⚠️",
    "stale": "⚠️",
}


def status_line(run, minimum_attempt):
    attempt = run.run_attempt or 1
    status = run.conclusion if run.status == "completed" else run.status
    if attempt < minimum_attempt:
        status = "queued"
    icon = ICONS.get(status, "⏳")
    # Keep the original minimum so later manual reruns also update this line.
    return (
        f"{icon} [`{run.name}`]({run.html_url}) — {status or 'pending'} "
        f"<!--ci-run:{run.id}:{minimum_attempt}-->"
    )


def refresh_comments(repo, pr_number):
    from github import GithubException

    runs = {}
    for comment in repo.get_issue(pr_number).get_comments():
        if comment.user.login != "github-actions[bot]" or comment.user.type != "Bot":
            continue
        body = comment.body or ""
        if not MARKER.search(body):
            continue
        lines = []
        for line in body.splitlines(keepends=True):
            marker = MARKER.search(line)
            if marker:
                run_id, minimum = map(int, marker.groups())
                if run_id not in runs:
                    try:
                        runs[run_id] = repo.get_workflow_run(run_id)
                    except GithubException as exc:
                        if exc.status != 404:
                            raise
                        # A deleted historical run must not block current replies.
                        runs[run_id] = None
                if runs[run_id] is None:
                    lines.append(line)
                    continue
                line = status_line(runs[run_id], minimum) + (
                    "\n" if line.endswith("\n") else ""
                )
            lines.append(line)
        updated = "".join(lines)
        if updated != body:
            comment.edit(updated)


def linked_pr_numbers(repo, event):
    """Fork workflow_run payloads often have an empty pull_requests list."""
    if "workflow_run" not in event:
        number = str(event.get("inputs", {}).get("pr_number", ""))
        if not number.isdecimal() or int(number) < 1:
            raise ValueError("pr_number must be a positive integer")
        return [int(number)]
    run = event["workflow_run"]
    if run["event"] not in ("pull_request", "pull_request_target"):
        return []
    if run.get("pull_requests"):
        return sorted({pr["number"] for pr in run["pull_requests"]})
    head_repo = run.get("head_repository") or {}
    owner = (head_repo.get("owner") or {}).get("login")
    if not owner or not run.get("head_branch"):
        return []
    # Include closed PRs: jobs may complete after merge/close. Markers bind the
    # exact run ID, so reused branches cannot update another run's status.
    return [
        pr.number
        for pr in repo.get_pulls(state="all", head=f"{owner}:{run['head_branch']}")
    ]


def main():
    from github import Auth, Github

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--pr-number", type=int)
    args = parser.parse_args()
    repo = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])).get_repo(
        os.environ["GITHUB_REPOSITORY"]
    )
    if args.resolve:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        numbers = linked_pr_numbers(repo, event)
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"prs={json.dumps(numbers)}\n")
    else:
        if not args.pr_number or args.pr_number < 1:
            parser.error("--pr-number must be positive")
        refresh_comments(repo, args.pr_number)


if __name__ == "__main__":
    main()
