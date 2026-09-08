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
from zoneinfo import ZoneInfo

DEFAULT_REPO = "sgl-project/sglang"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
GITHUB_API = "https://api.github.com"
UTILIZATION_WORKFLOW = "runner-utilization.yml"

# Primary pool labels only; aliases (1-gpu-runner, 8-gpu-h200-deepep, ...) are
# excluded so every runner is counted under exactly one label. a10 is left out
# as well: it serves no per-commit test, so its transitions are noise.
CUDA_LABEL_RE = re.compile(r"^\d+-gpu-(h100|h200|h20|5090|b200|b300|gb200|gb300)$")

FAILED_CONCLUSIONS = {"failure", "timed_out", "startup_failure", "action_required"}
# Aggregator jobs fail whenever any other job fails; listing them is noise.
# Jobs from a called workflow arrive prefixed ("call-pr-test-extra / <name>"),
# so the aggregator name is matched on the last segment.
AGGREGATOR_JOB_RE = re.compile(
    r"^(?:.+ / )?(check-all-jobs|pr-test-finish|pr-test-extra-finish)$"
)
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

    def latest_run_url(self, workflow_file: str) -> str:
        runs = self.workflow_runs(
            workflow_file, {"status": "success", "per_page": 1}, max_pages=1
        )
        if runs:
            return runs[0]["html_url"]
        return f"https://github.com/{self.repo}/actions/workflows/{workflow_file}"

    def runners(self) -> list:
        return self.paginate(f"repos/{self.repo}/actions/runners", "runners")


# --------------------------------------------------------------------------
# Lark card (schema 2.0)
# --------------------------------------------------------------------------


def md(text: str) -> dict:
    return {"tag": "markdown", "content": text}


def grey(text: str) -> str:
    return f"<font color='grey'>{text}</font>"


def kv_columns(pairs: list) -> dict:
    return {
        "tag": "column_set",
        "flex_mode": "flow",
        "horizontal_spacing": "default",
        "columns": [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [md(f"{grey(k)}\n**{v}**")],
            }
            for k, v in pairs
        ],
    }


def button(text: str, url: str) -> dict:
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": text},
        "type": "default",
        "behaviors": [{"type": "open_url", "default_url": url}],
    }


def chart(spec: dict, aspect_ratio: str = "16:9") -> dict:
    # The spec is VChart JSON rendered by the Lark client, so no image upload
    # (hence no Lark app credentials) is involved. Needs Lark client 7.1+.
    return {
        "tag": "chart",
        "aspect_ratio": aspect_ratio,
        "color_theme": "brand",
        "chart_spec": spec,
    }


HR = {"tag": "hr"}


def build_card(title: str, color: str, elements: list, buttons: list) -> dict:
    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": color,  # red | orange | green | blue | grey
            },
            "body": {"elements": elements + [button(t, u) for t, u in buttons]},
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


def fmt_local(dt: Optional[datetime]) -> str:
    if dt is None:
        return "-"
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p %Z")


def fmt_local_hour(dt: Optional[datetime]) -> str:
    """Timeline bucket label: the local hour alone, e.g. "9am"."""
    if dt is None:
        return "-"
    return dt.astimezone(LOCAL_TZ).strftime("%I%p").lstrip("0").lower()


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def primary_cuda_label(labels: list) -> Optional[str]:
    for name in labels:
        if CUDA_LABEL_RE.match(name):
            return name
    return None


def list_jobs_md(jobs: list, limit: int = MAX_LISTED_JOBS) -> str:
    lines = [f"- [{j['name']}]({j['html_url']})" for j in jobs[:limit]]
    if len(jobs) > limit:
        lines.append(f"- ... and {len(jobs) - limit} more")
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

    repo_url = run["html_url"].split("/actions/")[0]
    sha = run["head_sha"]
    subject = ((run.get("head_commit") or {}).get("message") or "").splitlines()
    commit_md = (
        f"[`{sha[:9]}`]({repo_url}/commit/{sha}) {subject[0] if subject else ''}"
    )
    rerun_prefix = f"Rerun #{attempt} - " if attempt > 1 else ""

    if conclusion == "cancelled":
        title = f"{rerun_prefix}{name}: CANCELLED"
        color = "grey"
    elif failed:
        title = f"{rerun_prefix}{name}: FAILED ({len(failed)} of {plural(len(counted), 'job')})"
        color = "red"
    else:
        title = f"{rerun_prefix}{name}: PASSED ({plural(len(counted), 'job')})"
        color = "green"

    jobs_summary = f"{len(counted)} total, {len(failed)} failed"
    if cancelled:
        jobs_summary += f", {len(cancelled)} cancelled"
    elements = [
        md(f"{grey('Commit')}  {commit_md}"),
        kv_columns(
            [
                ("Started", fmt_local(started)),
                ("Finished", fmt_local(updated)),
                ("Duration", duration),
                ("Jobs", jobs_summary),
            ]
        ),
    ]

    sections = []
    # None: first attempt, nothing to compare against
    if prev_failed is None:
        if failed:
            sections.append(
                f"**Failed jobs ({len(failed)})**\n{list_jobs_md(list(failed.values()))}"
            )
    else:
        diff = diff_attempts(failed, prev_failed)
        for key, heading in (
            ("fixed", "Fixed by rerun"),
            ("still", "Still failing"),
            ("new", "New failures"),
        ):
            if diff[key]:
                sections.append(
                    f"**{heading} ({len(diff[key])})**\n{list_jobs_md(diff[key])}"
                )
    if sections:
        elements.append(HR)
        elements.append(md("\n\n".join(sections)))

    buttons = [("View run on GitHub", run["html_url"])]
    if attempt > 1:
        buttons.append(
            (f"View attempt {attempt - 1}", f"{run['html_url']}/attempts/{attempt - 1}")
        )
    return build_card(title, color, elements, buttons)


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
        pool_label = primary_cuda_label([lb["name"] for lb in r.get("labels", [])])
        if pool_label is None:
            continue
        pool = pools.setdefault(
            pool_label,
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
    for pool_label, pool in sorted(pools.items()):
        prev = state.get(pool_label)
        degraded = is_degraded(pool, threshold)
        if degraded:
            since = parse_time(prev["degraded_since"]) if prev else now
            last = parse_time(prev["last_notified"]) if prev else None
            if prev is None:
                events.append(("degraded", pool_label, pool, since))
                last = now
            elif last is None or (now - last) >= timedelta(hours=remind_hours):
                events.append(("still_degraded", pool_label, pool, since))
                last = now
            new_state[pool_label] = {
                "degraded_since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_notified": last.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "offline": pool["offline"],
            }
        elif prev is not None:
            events.append(
                ("recovered", pool_label, pool, parse_time(prev["degraded_since"]))
            )
    return events, new_state


def render_health_event(
    kind: str, pool_label: str, pool: dict, since: datetime, now: datetime, repo: str
) -> dict:
    runners_url = f"https://github.com/{repo}/settings/actions/runners"
    elapsed = fmt_duration((now - since).total_seconds())
    status = f"{pool['online']} online / {pool['offline']} offline of {pool['total']}"
    all_down = pool["offline"] == pool["total"]
    if kind == "recovered":
        title = f"Runner pool recovered: {pool_label}"
        color = "green"
        elapsed_key = "Was degraded for"
    else:
        state_word = "DOWN" if all_down else "degraded"
        if kind == "degraded":
            title = f"Runner pool {state_word}: {pool_label}"
        else:
            title = f"Runner pool still {state_word}: {pool_label} ({elapsed})"
        color = "red" if all_down else "orange"
        elapsed_key = "Degraded for"
    elements = [
        kv_columns(
            [
                ("Pool", pool_label),
                ("Status", status),
                ("Busy", str(pool["busy"])),
                (elapsed_key, elapsed),
            ]
        ),
        md(f"{grey('Degraded since')}  {fmt_local(since)}"),
    ]
    if kind != "recovered":
        elements += [
            HR,
            md(f"**Offline runners**\n{compress_runner_names(pool['offline_names'])}"),
        ]
    return build_card(title, color, elements, [("View runners on GitHub", runners_url)])


def cmd_runner_health(args: argparse.Namespace, gh: GitHub) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = {}
    if os.path.exists(args.state_file):
        with open(args.state_file) as f:
            state = json.load(f)
    pools = summarize_pools(gh.runners())
    for pool_label, pool in sorted(pools.items()):
        print(
            f"{pool_label}: online {pool['online']} offline {pool['offline']} busy {pool['busy']}"
        )
    events, new_state = plan_health_events(
        pools, state, now, args.threshold, args.remind_hours
    )
    for kind, pool_label, pool, since in events:
        post_card(
            render_health_event(kind, pool_label, pool, since, now, gh.repo),
            args.webhook,
            args.dry_run,
        )
    if not events:
        print("no runner-health transitions")
    if not args.dry_run:
        with open(args.state_file, "w") as f:
            json.dump(new_state, f, indent=2)


# --------------------------------------------------------------------------
# queue-timeline
# --------------------------------------------------------------------------


def merge_timeline(series: dict) -> list:
    """Fold the per-label series into one CUDA-wide series, bucket by bucket.

    Backlog sums across pools; the wait takes the max of the per-pool p90s,
    since averaging would let idle pools mask the one pool that is stuck. The
    merge costs the answer to "which pool?", so each bucket keeps the label
    behind the deepest backlog and the longest wait -- routinely not the same.
    """
    buckets: dict = {}
    for label, rows in series.get("labels", {}).items():
        if not CUDA_LABEL_RE.match(label):
            continue
        for row in rows:
            b = buckets.setdefault(
                row["start"],
                {
                    "backlog": 0,
                    "started": 0,
                    "p90": 0.0,
                    "p90_pool": "-",
                    "top_pool": "-",
                    "top_backlog": 0,
                },
            )
            b["backlog"] += row["backlog"]
            b["started"] += row["started"]
            if row["p90_wait_min"] > b["p90"]:
                b["p90"], b["p90_pool"] = row["p90_wait_min"], label
            if row["backlog"] > b["top_backlog"]:
                b["top_backlog"], b["top_pool"] = row["backlog"], label
    return [dict(start=k, **v) for k, v in sorted(buckets.items())]


def timeline_chart_spec(rows: list) -> dict:
    hours = [fmt_local_hour(parse_time(r["start"])) for r in rows]
    return {
        "type": "common",
        "data": [
            {
                "id": "backlog",
                "values": [
                    {"hour": h, "value": r["backlog"]} for h, r in zip(hours, rows)
                ],
            },
            {
                "id": "wait",
                "values": [
                    {"hour": h, "value": round(r["p90"], 1)}
                    for h, r in zip(hours, rows)
                ],
            },
        ],
        "series": [
            {
                "type": "bar",
                "id": "backlog",
                "dataIndex": 0,
                "xField": "hour",
                "yField": "value",
                "name": "Jobs waiting (peak)",
            },
            {
                "type": "line",
                "id": "wait",
                "dataIndex": 1,
                "xField": "hour",
                "yField": "value",
                "name": "p90 wait (min)",
            },
        ],
        "axes": [
            {"orient": "left", "seriesIndex": [0], "title": {"visible": False}},
            {"orient": "right", "seriesId": ["wait"], "grid": {"visible": False}},
            {"orient": "bottom", "type": "band", "label": {"visible": True}},
        ],
        "legends": {"visible": True, "orient": "bottom"},
    }


def render_queue_timeline(rows: list, report_url: str) -> dict:
    peak = max(rows, key=lambda r: r["backlog"])
    slowest = max(rows, key=lambda r: r["p90"])
    span = f"{fmt_local(parse_time(rows[0]['start']))} to {fmt_local(parse_time(rows[-1]['start']))}"
    peak_hour = fmt_local_hour(parse_time(peak["start"]))
    slowest_hour = fmt_local_hour(parse_time(slowest["start"]))
    elements = [
        md(f"{grey('Window')}  {span}  {grey('(bucket: 1h)')}"),
        kv_columns(
            [
                ("Jobs started", str(sum(r["started"] for r in rows))),
                ("Peak backlog", f"{peak['backlog']} jobs"),
                ("Worst p90 wait", fmt_duration(slowest["p90"] * 60)),
            ]
        ),
        chart(timeline_chart_spec(rows)),
        md(
            f"{grey('Peak backlog')}  {peak_hour}, mostly "
            f"**{peak['top_pool']}** ({peak['top_backlog']})\n"
            f"{grey('Worst wait')}  {slowest_hour}, **{slowest['p90_pool']}**"
        ),
    ]
    return build_card(
        "CUDA queue over the day",
        "blue",
        elements,
        [("View utilization report", report_url)],
    )


def cmd_queue_timeline(args: argparse.Namespace, gh: GitHub) -> None:
    with open(args.series_file) as f:
        series = json.load(f)
    rows = merge_timeline(series)
    if not rows:
        print("no CUDA buckets in the series; skipping")
        return
    # The card is built from THIS run's scan, so link to it rather than to the
    # last successful one, which would be yesterday's report.
    run_id = os.environ.get("GITHUB_RUN_ID")
    report_url = (
        f"https://github.com/{gh.repo}/actions/runs/{run_id}"
        if run_id
        else gh.latest_run_url(UTILIZATION_WORKFLOW)
    )
    post_card(render_queue_timeline(rows, report_url), args.webhook, args.dry_run)


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

    p = sub.add_parser("queue-timeline", help="daily queue backlog / wait chart")
    p.add_argument(
        "--series-file",
        required=True,
        help="JSON written by runner_utilization_report.py --queue-series-out",
    )

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
        "queue-timeline": cmd_queue_timeline,
    }[args.command](args, gh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
