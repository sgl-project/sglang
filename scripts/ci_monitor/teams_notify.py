#!/usr/bin/env python3
"""
Post a failed scheduled CI run to a Microsoft Teams channel, with the
auto-bisect verdicts if available.

Stdlib only. Sends an Adaptive Card to a Teams Workflows webhook ("Post to a
channel when a webhook request is received") or a legacy Incoming Webhook;
both accept this payload. If TEAMS_WEBHOOK is unset the script prints a
warning and exits 0, so a missing secret never fails the calling workflow.

Environment:
    TEAMS_WEBHOOK  webhook URL (the secret)
    GITHUB_TOKEN   actions:read, to list failed jobs

Usage:
    python teams_notify.py --run-id 123 --workflow-name "PR Test (XPU)" \
        [--bisect-json bisect_results.json] [--dry-run]
"""

import argparse
import json
import os
import sys
import urllib.request
from typing import List, Optional

REPO = "sgl-project/sglang"
GITHUB_API = "https://api.github.com"
# Bookkeeping jobs that fail only because a test job did
IGNORED_JOBS = {"finish", "check-changes", "pr-gate", "pr-gate / pr-gate"}


def _gh_get(url: str, token: str) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_run(run_id: str, token: str) -> dict:
    return _gh_get(f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}", token)


def fetch_failed_jobs(run_id: str, token: str) -> List[dict]:
    jobs, page = [], 1
    while True:
        data = _gh_get(
            f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}/jobs"
            f"?per_page=100&page={page}",
            token,
        )
        jobs.extend(data.get("jobs", []))
        if len(jobs) >= data.get("total_count", 0) or not data.get("jobs"):
            break
        page += 1
    return [
        j
        for j in jobs
        if j.get("conclusion") in ("failure", "timed_out")
        and j.get("name") not in IGNORED_JOBS
    ]


def _text(text: str, **kw) -> dict:
    return {"type": "TextBlock", "text": text, "wrap": True, **kw}


def bisect_blocks(path: Optional[str]) -> List[dict]:
    head = _text("Auto-bisect", weight="Bolder", spacing="Medium", separator=True)
    if not path or not os.path.exists(path):
        return [head, _text("Results not available.", isSubtle=True)]
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return [head, _text(f"Results unreadable: {e}", color="Warning")]
    if data.get("error"):
        return [head, _text(f"Bisect failed: {data['error']}", color="Warning")]
    results = data.get("results", [])
    if not results:
        return [head, _text("No failure streaks found.", isSubtle=True)]

    blocks = [head]
    for r in results:
        t = r.get("target", {})
        pr = r.get("suspected_pr")
        suspect = (
            f"[#{pr}](https://github.com/{REPO}/pull/{pr})"
            if pr
            else r.get("suspected_commit") or "none"
        )
        blocks += [
            _text(
                f"**{t.get('test_file')}** ({t.get('job_name')}), "
                f"streak {t.get('current_streak')}",
                spacing="Medium",
            ),
            {
                "type": "FactSet",
                "facts": [
                    {
                        "title": "Verdict",
                        "value": f"{r.get('classification')} ({r.get('confidence')})",
                    },
                    {"title": "Suspect", "value": suspect},
                    {"title": "Evidence", "value": r.get("evidence_summary", "")},
                    {"title": "Fix", "value": r.get("recommended_fix", "")},
                ],
            },
        ]
    return blocks


def build_payload(
    workflow_name: str, run: dict, failed_jobs: List[dict], bisect: List[dict]
) -> dict:
    sha = (run.get("head_sha") or "")[:10]
    url = run.get("html_url", "")
    jobs = (
        "\n".join(
            f"- [{j.get('name')}]({j.get('html_url')}) on {j.get('runner_name') or '?'}"
            for j in failed_jobs
        )
        or "- none reported; check the run"
    )
    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "msteams": {"width": "Full"},
        "body": [
            _text(
                f"{workflow_name}: scheduled run on main failed",
                size="Large",
                weight="Bolder",
                color="Attention",
            ),
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Commit", "value": sha},
                    {
                        "title": "Started",
                        "value": run.get("run_started_at") or run.get("created_at", ""),
                    },
                ],
            },
            _text(f"Failed jobs ({len(failed_jobs)})", weight="Bolder"),
            _text(jobs),
            *bisect,
        ],
        "actions": [{"type": "Action.OpenUrl", "title": "Open run", "url": url}],
    }
    return {
        "type": "message",
        "attachments": [
            {"contentType": "application/vnd.microsoft.card.adaptive", "content": card}
        ],
    }


def post(payload: dict, webhook: str) -> None:
    req = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # Workflows webhooks answer 202 with no body; legacy connectors answer 200 "1".
    with urllib.request.urlopen(req, timeout=30) as resp:
        if not 200 <= resp.status < 300:
            raise RuntimeError(f"Teams webhook returned HTTP {resp.status}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workflow-name", required=True)
    ap.add_argument("--bisect-json", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Print instead of post")
    args = ap.parse_args()

    webhook = os.environ.get("TEAMS_WEBHOOK", "")
    if not args.dry_run and not webhook:
        print("::warning::TEAMS_WEBHOOK not set; skipping Teams notification.")
        return 0

    token = os.environ.get("GITHUB_TOKEN", "")
    payload = build_payload(
        args.workflow_name,
        fetch_run(args.run_id, token),
        fetch_failed_jobs(args.run_id, token),
        bisect_blocks(args.bisect_json),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0
    post(payload, webhook)
    print("Posted failure card to Teams.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
