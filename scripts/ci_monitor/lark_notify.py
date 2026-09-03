#!/usr/bin/env python3
"""
Post CUDA CI health cards to a Lark group via an incoming webhook.

Used by .github/workflows/ci-lark-notify.yml; see scripts/ci_monitor/README.md.
Needs GITHUB_TOKEN (an admin PAT for runner-health) and LARK_WEBHOOK, or
--dry-run to print the card JSON instead of posting.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

DEFAULT_REPO = "sgl-project/sglang"
GITHUB_API = "https://api.github.com"

# Primary pool labels only; aliases (1-gpu-runner, 8-gpu-h200-deepep, ...) are
# excluded so every runner is counted under exactly one label.
CUDA_LABEL_RE = re.compile(r"^\d+-gpu-(h100|h200|h20|5090|b200|b300|gb200|gb300|a10)$")

# Workflows whose jobs run on the CUDA pools; used for queue-digest.
CUDA_WORKFLOW_FILES = [
    "pr-test.yml",
    "pr-test-extra.yml",
    "nightly-test-nvidia.yml",
    "weekly-test-nvidia.yml",
]

FAILED_CONCLUSIONS = {"failure", "timed_out", "startup_failure", "action_required"}
# Aggregator jobs fail whenever any other job fails; listing them is noise.
AGGREGATOR_JOB_RE = re.compile(r"^(check-all-jobs|pr-test-finish)$")
MAX_LISTED_JOBS = 15


# --------------------------------------------------------------------------
# GitHub API
# --------------------------------------------------------------------------


class GitHub:
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo

    def get(self, path: str, params: Optional[dict] = None, retries: int = 5) -> Any:
        url = f"{GITHUB_API}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                transient = e.code in (429, 502, 503, 504) or (
                    e.code == 403 and "rate limit" in body.lower()
                )
                if not transient or attempt == retries - 1:
                    raise RuntimeError(f"GET {url} -> {e.code}: {body[:300]}") from e
            except urllib.error.URLError as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"GET {url} failed: {e}") from e
            time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def paginate(
        self, path: str, key: str, params: Optional[dict] = None, max_pages: int = 30
    ) -> list:
        params = dict(params or {})
        params.setdefault("per_page", 100)
        items: list = []
        for page in range(1, max_pages + 1):
            params["page"] = page
            data = self.get(path, params)
            chunk = data.get(key, [])
            items.extend(chunk)
            if len(chunk) < params["per_page"]:
                break
        return items

    def run(self, run_id: int) -> dict:
        return self.get(f"repos/{self.repo}/actions/runs/{run_id}")

    def run_jobs(self, run_id: int) -> list:
        return self.paginate(
            f"repos/{self.repo}/actions/runs/{run_id}/jobs",
            "jobs",
            # latest attempt per job; jobs not rerun keep their earlier result
            {"filter": "latest"},
        )

    def run_attempt_jobs(self, run_id: int, attempt: int) -> list:
        return self.paginate(
            f"repos/{self.repo}/actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs"
        )

    def workflow_runs(
        self, workflow_id: Any, params: dict, max_pages: int = 30
    ) -> list:
        return self.paginate(
            f"repos/{self.repo}/actions/workflows/{workflow_id}/runs",
            "workflow_runs",
            params,
            max_pages=max_pages,
        )

    def runners(self) -> list:
        return self.paginate(f"repos/{self.repo}/actions/runners", "runners")


# --------------------------------------------------------------------------
# Lark
# --------------------------------------------------------------------------


def build_card(title: str, color: str, body_md: str, buttons: list) -> dict:
    elements: list = [{"tag": "div", "text": {"tag": "lark_md", "content": body_md}}]
    if buttons:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": text},
                        "url": url,
                        "type": "default",
                    }
                    for text, url in buttons
                ],
            }
        )
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": color,
            },
            "elements": elements,
        },
    }


def post_card(card: dict, webhook: str, dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(card, indent=2))
        return
    req = urllib.request.Request(
        webhook,
        data=json.dumps(card).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if body.get("code", body.get("StatusCode")) not in (0, None):
        raise RuntimeError(f"Lark webhook rejected message: {body}")
    print(f"posted: {card['card']['header']['title']['content']}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def parse_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def percentile(values: list, p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    return ordered[idx]


def primary_cuda_label(labels: list) -> Optional[str]:
    for label in labels:
        if CUDA_LABEL_RE.match(label):
            return label
    return None


def list_jobs_md(jobs: list, limit: int = MAX_LISTED_JOBS) -> str:
    lines = [f"  - [{j['name']}]({j['html_url']})" for j in jobs[:limit]]
    if len(jobs) > limit:
        lines.append(f"  - ... and {len(jobs) - limit} more")
    return "\n".join(lines)


def compress_runner_names(names: list) -> str:
    groups: dict = {}
    for n in sorted(names):
        m = re.match(r"^(.*?)(\d+)$", n)
        if m:
            groups.setdefault(m.group(1), []).append(int(m.group(2)))
        else:
            groups.setdefault(n, [])
    parts = []
    for prefix, nums in groups.items():
        if not nums:
            parts.append(prefix)
        elif len(nums) == 1:
            parts.append(f"{prefix}{nums[0]}")
        else:
            parts.append(prefix + "{" + ",".join(str(x) for x in sorted(nums)) + "}")
    return ", ".join(parts)


# --------------------------------------------------------------------------
# ci-status
# --------------------------------------------------------------------------


def is_reportable_job(job: dict) -> bool:
    return job.get("conclusion") not in (
        None,
        "skipped",
    ) and not AGGREGATOR_JOB_RE.match(job["name"])


def failed_job_names(jobs: list) -> dict:
    return {
        j["name"]: j
        for j in jobs
        if is_reportable_job(j) and j.get("conclusion") in FAILED_CONCLUSIONS
    }


def diff_attempts(current: dict, previous: dict) -> dict:
    return {
        "fixed": [j for n, j in previous.items() if n not in current],
        "still": [j for n, j in current.items() if n in previous],
        "new": [j for n, j in current.items() if n not in previous],
    }


def render_ci_status(run: dict, jobs: list, prev_failed: Optional[dict]) -> dict:
    name = run["name"]
    attempt = run.get("run_attempt", 1)
    conclusion = run.get("conclusion") or "unknown"
    counted = [j for j in jobs if is_reportable_job(j)]
    failed = failed_job_names(jobs)
    cancelled = [j for j in jobs if j.get("conclusion") == "cancelled"]
    started = parse_time(run.get("run_started_at"))
    updated = parse_time(run.get("updated_at"))
    duration = (
        fmt_duration((updated - started).total_seconds())
        if started and updated
        else "-"
    )

    sha = run["head_sha"][:9]
    commit_msg = (run.get("head_commit") or {}).get("message", "").splitlines()
    commit_line = f"`{sha}` {commit_msg[0] if commit_msg else ''}".strip()
    rerun_prefix = f"Rerun #{attempt} - " if attempt > 1 else ""

    if conclusion == "cancelled":
        title = f"{rerun_prefix}{name}: CANCELLED ({duration})"
        color = "grey"
    elif failed:
        title = f"{rerun_prefix}{name}: FAILED ({len(failed)} failed / {len(counted)} jobs, {duration})"
        color = "red"
    else:
        title = f"{rerun_prefix}{name}: PASSED ({len(counted)} jobs, {duration})"
        color = "green"

    lines = [commit_line]
    # None: first attempt, nothing to compare against
    if prev_failed is None:
        if failed:
            lines.append(f"**Failed jobs ({len(failed)})**")
            lines.append(list_jobs_md(list(failed.values())))
    else:
        diff = diff_attempts(failed, prev_failed)
        if diff["fixed"]:
            lines.append(f"**Fixed by rerun ({len(diff['fixed'])})**")
            lines.append(list_jobs_md(diff["fixed"]))
        if diff["still"]:
            lines.append(f"**Still failing ({len(diff['still'])})**")
            lines.append(list_jobs_md(diff["still"]))
        if diff["new"]:
            lines.append(f"**New failures ({len(diff['new'])})**")
            lines.append(list_jobs_md(diff["new"]))
    if cancelled:
        lines.append(f"Cancelled jobs: {len(cancelled)}")

    buttons = [("Run", run["html_url"])]
    if attempt > 1:
        buttons.append(
            (f"Attempt {attempt - 1}", f"{run['html_url']}/attempts/{attempt - 1}")
        )
    return build_card(title, color, "\n".join(lines), buttons)


def cmd_ci_status(args: argparse.Namespace, gh: GitHub) -> None:
    run = gh.run(args.run_id)
    if run["event"] != "schedule" and not args.any_event:
        print(
            f"run {args.run_id} event={run['event']} is not a scheduled run; skipping"
        )
        return
    if run.get("status") != "completed":
        print(
            f"run {args.run_id} status={run.get('status')} is not completed; skipping"
        )
        return
    jobs = gh.run_jobs(run["id"])
    attempt = run.get("run_attempt", 1)
    prev_failed = None
    if attempt > 1:
        prev_failed = failed_job_names(gh.run_attempt_jobs(run["id"], attempt - 1))
    post_card(render_ci_status(run, jobs, prev_failed), args.webhook, args.dry_run)


# --------------------------------------------------------------------------
# runner-health
# --------------------------------------------------------------------------


def summarize_pools(runners: list) -> dict:
    pools: dict = {}
    for r in runners:
        label = primary_cuda_label([l["name"] for l in r.get("labels", [])])
        if label is None:
            continue
        pool = pools.setdefault(
            label,
            {"total": 0, "online": 0, "offline": 0, "busy": 0, "offline_names": []},
        )
        pool["total"] += 1
        if r.get("status") == "online":
            pool["online"] += 1
            if r.get("busy"):
                pool["busy"] += 1
        else:
            pool["offline"] += 1
            pool["offline_names"].append(r["name"])
    return pools


def is_degraded(pool: dict, threshold: float) -> bool:
    if pool["total"] < 2:
        return pool["offline"] == pool["total"] and pool["total"] > 0
    return pool["offline"] / pool["total"] >= threshold


def plan_health_events(
    pools: dict, state: dict, now: datetime, threshold: float, remind_hours: float
) -> tuple:
    events = []  # (kind, label, pool, since)
    new_state: dict = {}
    for label, pool in sorted(pools.items()):
        prev = state.get(label)
        degraded = is_degraded(pool, threshold)
        if degraded:
            since = parse_time(prev["degraded_since"]) if prev else now
            last = parse_time(prev["last_notified"]) if prev else None
            if prev is None:
                events.append(("degraded", label, pool, since))
                last = now
            elif last is None or (now - last) >= timedelta(hours=remind_hours):
                events.append(("still_degraded", label, pool, since))
                last = now
            new_state[label] = {
                "degraded_since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_notified": last.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "offline": pool["offline"],
            }
        elif prev is not None:
            events.append(
                ("recovered", label, pool, parse_time(prev["degraded_since"]))
            )
    return events, new_state


def render_health_event(
    kind: str, label: str, pool: dict, since: datetime, now: datetime, repo: str
) -> dict:
    counts = (
        f"online {pool['online']} / offline {pool['offline']} / total {pool['total']}, "
        f"busy {pool['busy']}"
    )
    runners_url = f"https://github.com/{repo}/settings/actions/runners"
    since_s = since.strftime("%Y-%m-%d %H:%M UTC")
    elapsed = fmt_duration((now - since).total_seconds())
    if kind == "recovered":
        title = f"Runner pool recovered: {label}"
        body = f"{counts}\nDegraded for {elapsed} (since {since_s})."
        return build_card(title, "green", body, [("Runners", runners_url)])
    all_down = pool["offline"] == pool["total"]
    state_word = "DOWN" if all_down else "degraded"
    if kind == "degraded":
        title = f"Runner pool {state_word}: {label}"
    else:
        title = f"Runner pool still {state_word}: {label} ({elapsed})"
    body = "\n".join(
        [
            counts,
            f"Offline: {compress_runner_names(pool['offline_names'])}",
            f"Since {since_s}",
        ]
    )
    return build_card(
        title, "red" if all_down else "orange", body, [("Runners", runners_url)]
    )


def cmd_runner_health(args: argparse.Namespace, gh: GitHub) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = {}
    if os.path.exists(args.state_file):
        with open(args.state_file) as f:
            state = json.load(f)
    pools = summarize_pools(gh.runners())
    for label, pool in sorted(pools.items()):
        print(
            f"{label}: online {pool['online']} offline {pool['offline']} busy {pool['busy']}"
        )
    events, new_state = plan_health_events(
        pools, state, now, args.threshold, args.remind_hours
    )
    for kind, label, pool, since in events:
        post_card(
            render_health_event(kind, label, pool, since, now, gh.repo),
            args.webhook,
            args.dry_run,
        )
    if not events:
        print("no runner-health transitions")
    if not args.dry_run:
        with open(args.state_file, "w") as f:
            json.dump(new_state, f, indent=2)


# --------------------------------------------------------------------------
# queue-digest
# --------------------------------------------------------------------------


def job_queue_seconds(job: dict, now: datetime) -> Optional[float]:
    created = parse_time(job.get("created_at"))
    if created is None:
        return None
    if job.get("status") == "queued":
        return (now - created).total_seconds()
    started = parse_time(job.get("started_at"))
    if started is None or started < created:
        return None
    return (started - created).total_seconds()


def summarize_queue(jobs: list, now: datetime) -> dict:
    per_label: dict = {}
    for job in jobs:
        label = primary_cuda_label(job.get("labels") or [])
        if label is None:
            continue
        q = job_queue_seconds(job, now)
        if q is None:
            continue
        entry = per_label.setdefault(
            label, {"waits": [], "queued_now": [], "started": 0}
        )
        if job.get("status") == "queued":
            entry["queued_now"].append(q)
        else:
            entry["waits"].append(q)
            entry["started"] += 1
    result = {}
    for label, e in per_label.items():
        result[label] = {
            "n": e["started"],
            "p50": percentile(e["waits"], 0.5),
            "p90": percentile(e["waits"], 0.9),
            "max": max(e["waits"]) if e["waits"] else None,
            "queued_now": len(e["queued_now"]),
            "oldest_queued": max(e["queued_now"]) if e["queued_now"] else None,
        }
    return result


def render_queue_digest(
    stats: dict, hours: float, slow_minutes: float, repo: str
) -> dict:
    slow = slow_minutes * 60
    ordered = sorted(stats.items(), key=lambda kv: -(kv[1]["p90"] or 0))
    lines = []
    for label, s in ordered:
        flag = " (!)" if (s["p90"] or 0) >= slow else ""
        queued = (
            f", queued now {s['queued_now']} (oldest {fmt_duration(s['oldest_queued'])})"
            if s["queued_now"]
            else ""
        )
        lines.append(
            f"**{label}**{flag}: {s['n']} jobs, p50 {fmt_duration(s['p50'])}, "
            f"p90 {fmt_duration(s['p90'])}, max {fmt_duration(s['max'])}{queued}"
        )
    if not lines:
        lines.append("_No CUDA jobs in this window._")
    any_slow = any((s["p90"] or 0) >= slow for s in stats.values())
    title = f"CUDA queue time, last {int(hours)}h"
    if any_slow:
        title += f" - p90 over {int(slow_minutes)}m on some pools"
    return build_card(
        title,
        "orange" if any_slow else "blue",
        "\n".join(lines),
        [("Actions", f"https://github.com/{repo}/actions")],
    )


def fetch_window_jobs(
    gh: GitHub, hours: float, workflow_files: list, workers: int
) -> list:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    runs: list = []
    for wf in workflow_files:
        runs.extend(
            gh.workflow_runs(
                wf,
                {"created": ">=" + since.strftime("%Y-%m-%dT%H:%M:%SZ")},
                max_pages=10,
            )
        )
    print(f"{len(runs)} runs in window across {len(workflow_files)} workflows")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        job_lists = list(pool.map(lambda r: gh.run_jobs(r["id"]), runs))
    jobs = [j for jl in job_lists for j in jl]
    print(f"{len(jobs)} jobs fetched")
    return jobs


def cmd_queue_digest(args: argparse.Namespace, gh: GitHub) -> None:
    now = datetime.now(timezone.utc)
    jobs = fetch_window_jobs(gh, args.hours, args.workflows.split(","), args.workers)
    stats = summarize_queue(jobs, now)
    post_card(
        render_queue_digest(stats, args.hours, args.slow_minutes, gh.repo),
        args.webhook,
        args.dry_run,
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--webhook", default=os.environ.get("LARK_WEBHOOK"))
    parser.add_argument(
        "--dry-run", action="store_true", help="print card JSON instead of posting"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ci-status", help="summarize a finished scheduled run")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument(
        "--any-event", action="store_true", help="also report non-schedule runs"
    )

    p = sub.add_parser("runner-health", help="per-label online/offline transitions")
    p.add_argument("--state-file", required=True)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="offline ratio that counts as degraded",
    )
    p.add_argument("--remind-hours", type=float, default=1.0)

    p = sub.add_parser("queue-digest", help="per-label queue time percentiles")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument(
        "--slow-minutes", type=float, default=30.0, help="p90 above this is flagged"
    )
    p.add_argument("--workflows", default=",".join(CUDA_WORKFLOW_FILES))
    p.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    if not args.token:
        print("GITHUB_TOKEN (or --token) is required", file=sys.stderr)
        return 2
    if not args.webhook and not args.dry_run:
        print(
            "LARK_WEBHOOK (or --webhook) is required unless --dry-run", file=sys.stderr
        )
        return 2

    gh = GitHub(args.token, args.repo)
    {
        "ci-status": cmd_ci_status,
        "runner-health": cmd_runner_health,
        "queue-digest": cmd_queue_digest,
    }[args.command](args, gh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
