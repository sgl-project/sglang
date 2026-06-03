#!/usr/bin/env python3
"""
Runner Utilization Report

Analyzes GitHub Actions job data to calculate runner utilization metrics.
Reports idle time, active time, and utilization percentage per runner label.
"""

import argparse
import json
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

# Display all clock times in California time (auto PST/PDT). Internals stay in
# UTC — this only affects how timestamps are rendered in the report.
try:
    from zoneinfo import ZoneInfo

    DISPLAY_TZ = ZoneInfo("America/Los_Angeles")
    TZ_LABEL = "PT"
except Exception:  # zoneinfo/tzdata unavailable — fall back to UTC
    DISPLAY_TZ = timezone.utc
    TZ_LABEL = "UTC"

# Labels to skip when grouping runners (GitHub default labels)
DEFAULT_LABELS_TO_IGNORE = {"self-hosted", "Linux", "X64", "ARM64"}
GITHUB_HOSTED_LABELS = {"ubuntu-latest", "ubuntu-22.04", "ubuntu-24.04"}

# Human-facing job outcome buckets, in display order, with emoji.
STATUS_ORDER = ("pass", "fail", "cancel", "running", "queued")
STATUS_EMOJI = {
    "pass": "✅",
    "fail": "❌",
    "cancel": "🚫",
    "running": "🔄",
    "queued": "⏳",
}


def format_status_counts(counts: dict) -> str:
    """Compact per-label outcome summary, e.g. '✅120 ❌3 🔄2 ⏳4'."""
    parts = [f"{STATUS_EMOJI[s]}{counts[s]}" for s in STATUS_ORDER if counts.get(s)]
    return " ".join(parts) if parts else "—"


def run_gh_command(args: list[str], max_retries: int = 10) -> dict:
    """Run gh CLI command and return JSON result.

    Retries on transient failures (5xx, secondary rate limits, network
    blips) with exponential backoff. The previous fail-fast behavior
    combined with `except Exception: return None` in the threadpool
    callers caused entire workflow runs to be silently dropped from
    the utilization numerator whenever GH API hiccuped, severely
    undercounting busy time on busy days.
    """
    last_err = ""
    for attempt in range(max_retries):
        result = subprocess.run(
            ["gh", "api"] + args,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        last_err = result.stderr or "(no stderr)"
        # Detect retryable conditions: HTTP 5xx, secondary rate limit, abuse
        # detection, network resets. 4xx other than 429 are non-retryable.
        retryable = any(
            s in last_err
            for s in (
                "rate limit",
                "abuse",
                "Internal Server Error",
                "502",
                "503",
                "504",
                "Bad Gateway",
                "Gateway Time-out",
                "connection reset",
                "Connection reset",
                "EOF",
                "timeout",
            )
        )
        if not retryable:
            break
        # Exponential backoff with jitter, capped at 60s.
        delay = min(60, (2**attempt) + random.uniform(0, 1))
        time.sleep(delay)
    raise Exception(f"gh api failed after {max_retries} attempts: {last_err[:300]}")


def get_workflow_runs(repo: str, hours: int = 24) -> list[dict]:
    """Get workflow runs created in the last N hours.

    Paginates the runs API newest-first and filters by `created_at`
    client-side. Important: do NOT add a server-side `created=...`
    filter -- the runs API silently invokes a search-style backend
    when filter params are present, which caps total results at 1000
    regardless of pagination. Plain unfiltered pagination has no such
    cap, so we collect what we need and stop at the first run older
    than the cutoff.

    Previously this function had a hard `page > 50` (5000-run) safety
    cap. Once daily CI volume passed that threshold, the cap silently
    truncated the OLDEST end of the window -- the Daily report's first
    ~10 hours looked empty for that reason. The cap is now raised to
    200 pages (20k runs), and we log a warning if it ever fires
    instead of breaking silently.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    runs = []
    page = 1
    # Soft guard against runaway pagination. At ~6 runs/min sustained
    # SGLang volume, 200 pages = ~57h of runs -- comfortable headroom
    # for a 24h window. If this ever fires we want to know, not have
    # the chart silently lose data again.
    MAX_PAGES = 200
    while page <= MAX_PAGES:
        data = run_gh_command(
            [
                f"repos/{repo}/actions/runs?per_page=100&page={page}",
            ]
        )
        page_runs = data.get("workflow_runs", [])
        if not page_runs:
            return runs

        for run in page_runs:
            created_at = parse_time(run.get("created_at"))
            if created_at and created_at >= since:
                runs.append(run)
            elif created_at and created_at < since:
                # Runs are ordered by created_at desc, so the first
                # run older than `since` means we've passed the cutoff.
                return runs

        if len(page_runs) < 100:
            return runs
        page += 1

    print(
        f"WARNING: pagination reached MAX_PAGES={MAX_PAGES} "
        f"({len(runs)} runs collected). The window may be truncated -- "
        f"consider narrowing --hours or raising MAX_PAGES."
    )
    return runs


def get_jobs_for_run(repo: str, run_id: int) -> list[dict]:
    """Get all jobs for a workflow run, including all retry attempts.

    `filter=all` is required so that re-run attempts of the same job
    appear separately. Each attempt consumed host time on the runner
    pool, so for utilization we want them all summed in. The default
    (`filter=latest`) only returns the most recent attempt and silently
    hides time spent on prior retries.
    """
    jobs = []
    page = 1
    while True:
        data = run_gh_command(
            [
                f"repos/{repo}/actions/runs/{run_id}/jobs"
                f"?per_page=100&page={page}&filter=all",
            ]
        )
        jobs.extend(data.get("jobs", []))
        if len(data.get("jobs", [])) < 100:
            break
        page += 1
        if page > 20:  # Safety limit (2000 jobs per run)
            break
    return jobs


def get_runners(repo: str, online_only: bool = True) -> list[dict]:
    """Get all self-hosted runners with pagination. Returns empty if no permission."""
    try:
        all_runners = []
        page = 1
        while True:
            data = run_gh_command(
                [f"repos/{repo}/actions/runners?per_page=100&page={page}"]
            )
            runners = data.get("runners", [])
            all_runners.extend(runners)
            if len(runners) < 100:
                break
            page += 1
            if page > 10:  # Safety limit
                break
        if online_only:
            all_runners = [r for r in all_runners if r.get("status") == "online"]
        return all_runners
    except Exception as e:
        print(f"Warning: Cannot access runners API (need admin): {e}")
        return []


def parse_time(time_str: str) -> datetime:
    """Parse ISO timestamp to datetime."""
    if not time_str:
        return None
    return datetime.fromisoformat(time_str.replace("Z", "+00:00"))


def classify_job(job: dict, now: datetime):
    """Derive the queue-wait and busy interval for a single job.

    Returns a job_info dict, or None when the job neither waited for nor
    occupied a runner (skipped / cancelled-before-start / missing data).

    The queue wait runs from when the job entered the runner queue
    (`created_at`) until a runner picked it up (`started_at`) — or until
    `now` if it is still waiting.

    GitHub API gotcha this exists to handle: a still-queued job reports
    status="queued", runner_name="" and `started_at` set to a PLACEHOLDER
    equal to `created_at` (not null). The previous code required both a
    runner_name and a `completed_at`, so every in-flight wait — the
    multi-hour 8-gpu jobs still sitting in the queue, i.e. the worst cases —
    was dropped, undercounting max/avg queue time. We therefore measure a
    queued job's wait against `now` rather than its bogus `started_at`, and
    don't require completion.
    """
    status = job.get("status")
    runner_name = job.get("runner_name") or ""
    created_at = parse_time(job.get("created_at"))
    started_at = parse_time(job.get("started_at"))
    completed_at = parse_time(job.get("completed_at"))

    if status == "queued":
        # Still waiting for a runner; ignore the placeholder started_at.
        queue_end, start, end = now, None, None
    elif status == "in_progress" and started_at is not None:
        # Running now: the wait is final and it still occupies the runner.
        queue_end, start, end = started_at, started_at, now
    elif (
        status == "completed"
        and started_at is not None
        and completed_at is not None
        and runner_name
    ):
        queue_end, start, end = started_at, started_at, completed_at
    else:
        # Skipped, cancelled before start, or missing timestamps: never
        # waited for or occupied a runner.
        return None

    if created_at is None:
        return None

    queue_time = max(0.0, (queue_end - created_at).total_seconds())
    duration = (end - start).total_seconds() if start is not None else 0.0
    labels = [
        label
        for label in job.get("labels", [])
        if label not in DEFAULT_LABELS_TO_IGNORE | GITHUB_HOSTED_LABELS
    ]

    # Human-facing outcome bucket used by the report's status breakdown.
    if status == "queued":
        outcome = "queued"
    elif status == "in_progress":
        outcome = "running"
    else:  # completed and actually ran
        outcome = {"success": "pass", "cancelled": "cancel"}.get(
            job.get("conclusion"), "fail"
        )

    return {
        "start": start,
        "end": end,
        "created_at": created_at,
        "queue_end": queue_end,
        "duration": duration,
        "queue_time": queue_time,
        "job_name": job.get("name", ""),
        "runner_name": runner_name,
        "labels": labels,
        "status": outcome,
        "html_url": job.get("html_url", ""),
    }


def calculate_concurrency_metrics(
    jobs: list,
    window_start: datetime,
    window_end: datetime,
    num_runners: int,
) -> dict:
    """Sweep-line algorithm: peak/avg concurrent, saturation time, peak queue."""
    if not jobs:
        return {
            "peak_concurrent": 0,
            "avg_concurrent": 0.0,
            "saturation_seconds": 0,
            "saturation_pct": 0.0,
            "peak_queue": 0,
        }
    window_seconds = (window_end - window_start).total_seconds()
    if window_seconds <= 0:
        return {
            "peak_concurrent": 0,
            "avg_concurrent": 0.0,
            "saturation_seconds": 0,
            "saturation_pct": 0.0,
            "peak_queue": 0,
        }
    # Group running intervals by runner_name and merge per-runner before
    # building (start, +1) / (end, -1) events. Same overlap sources as
    # _wallclock_busy_seconds (timestamp slop, `filter=all` retry rows,
    # in_progress end=now straddling completed_at) -- without the merge
    # they push `current_running` above num_runners and Avg Concurrent
    # renders >100% (e.g. 4-gpu-b200 showed 7.5/7=107% in the previous
    # run). Merge per-runner, not globally: jobs on DIFFERENT runners
    # do legitimately overlap and must contribute separate events.
    intervals_by_runner = defaultdict(list)
    for job in jobs:
        start, end = job.get("start"), job.get("end")
        if start is None or end is None:
            continue
        if end < window_start or start > window_end:
            continue
        s = max(start, window_start)
        e = min(end, window_end)
        if e <= s:
            continue
        # Bucket runnerless jobs (rare in_progress edge case) individually
        # so we don't merge unrelated jobs into one interval.
        runner = job.get("runner_name") or f"_unknown_{id(job)}"
        intervals_by_runner[runner].append((s, e))

    running_events = []
    for intervals in intervals_by_runner.values():
        intervals.sort()
        covered_until = window_start
        for s, e in intervals:
            s = max(s, covered_until)
            if e > s:
                running_events.append((s, 1))
                running_events.append((e, -1))
                covered_until = e
    queue_events = []
    for job in jobs:
        created_at = job.get("created_at")
        # The wait ends when a runner picks the job up, or `now` if it is
        # still queued (queue_end was set to now upstream). Counting the
        # still-open waits is what makes peak_queue reflect the real backlog.
        queue_end = job.get("queue_end") or job["start"]
        if created_at and queue_end and created_at < queue_end:
            if queue_end < window_start or created_at > window_end:
                continue
            queue_events.append((max(created_at, window_start), 1))
            queue_events.append((min(queue_end, window_end), -1))
    running_events.sort(key=lambda e: (e[0], e[1] == 1))
    current_running = 0
    peak_running = 0
    prev_time = window_start
    total_running_seconds = 0.0
    saturation_seconds = 0.0
    for event_time, delta in running_events:
        td = (event_time - prev_time).total_seconds()
        if td > 0:
            total_running_seconds += current_running * td
            if current_running >= num_runners:
                saturation_seconds += td
        current_running += delta
        peak_running = max(peak_running, current_running)
        prev_time = event_time
    if prev_time < window_end:
        td = (window_end - prev_time).total_seconds()
        total_running_seconds += current_running * td
        if current_running >= num_runners:
            saturation_seconds += td
    queue_events.sort(key=lambda e: (e[0], e[1] == 1))
    current_queued = 0
    peak_queue = 0
    for _, delta in queue_events:
        current_queued += delta
        peak_queue = max(peak_queue, current_queued)
    avg_concurrent = total_running_seconds / window_seconds if window_seconds > 0 else 0
    return {
        "peak_concurrent": peak_running,
        "avg_concurrent": avg_concurrent,
        "saturation_seconds": saturation_seconds,
        "saturation_pct": (
            (saturation_seconds / window_seconds * 100) if window_seconds > 0 else 0
        ),
        "peak_queue": peak_queue,
    }


_NON_GPU_WORKFLOW_HINTS = (
    "lint",
    "deploy",
    "release",
    "publish",
    "docs",
    "doc",
    "mintlify",
    "runner utilization",  # this very script
    "tag-and-rerun",
    "auto",  # auto-merge etc.
    "label",
    "stale",
    "dependabot",
    "codeql",
)


def _likely_no_gpu_jobs(workflow_name: str) -> bool:
    """Heuristic: skip per-run job-fetch for workflows that don't dispatch
    to self-hosted GPU runners. The GH API rate limit (~5000 req/hr per
    token) is the bottleneck on busy 24h windows where ~4000 workflow
    runs fire — but only a fraction of those (pr-test, nightly-test,
    pr-test-*kernel, etc.) actually run on GPU runners. Skipping the
    docs/lint/release runs cuts the API call budget by 2-4x.
    """
    if not workflow_name:
        return False
    n = workflow_name.lower()
    return any(h in n for h in _NON_GPU_WORKFLOW_HINTS)


def _wallclock_busy_seconds(jobs, window_start, window_end):
    """Wall-clock busy seconds on a single runner across the window.

    Each job contributes its `[start, end]` interval clipped to the
    window. Overlapping intervals are merged with a sweep so the
    result is bounded by `(window_end - window_start).total_seconds()`
    per runner. Without the merge, three pathologies double-count
    busy time:

    1. GitHub API timestamp slop -- consecutive back-to-back jobs on
       the same runner aren't reliably monotonic, so job N+1's
       `started_at` can land slightly before job N's `completed_at`.
    2. `filter=all` on the jobs API returns every retry attempt as a
       separate row; same runner_name, near-adjacent intervals.
    3. An `in_progress` job uses `end=now`, which can straddle a
       just-completed job's `completed_at` by a few seconds.

    These pushed per-label utilization slightly above 100% on busy
    multi-GPU pools (e.g. `2-gpu-h100` rendered at 108.8% before this
    helper landed). After merging, utilization is mathematically
    capped at 100% per label.
    """
    intervals = []
    for j in jobs:
        cs = max(j["start"], window_start)
        ce = min(j["end"], window_end)
        if ce > cs:
            intervals.append((cs, ce))
    intervals.sort()
    busy = 0.0
    covered_until = window_start
    for s, e in intervals:
        s = max(s, covered_until)
        if e > s:
            busy += (e - s).total_seconds()
            covered_until = e
    return busy


def calculate_utilization(repo: str, hours: int = 24, runner_filter: str = None):
    """Calculate runner utilization metrics."""

    print(f"Fetching workflow runs from last {hours} hours...")
    all_runs = get_workflow_runs(repo, hours)
    runs = [r for r in all_runs if not _likely_no_gpu_jobs(r.get("name", ""))]
    skipped = len(all_runs) - len(runs)
    print(
        f"Found {len(all_runs)} workflow runs "
        f"({skipped} skipped as non-GPU: docs/lint/release/etc.)"
    )

    # Try to get online runners from API
    print("Fetching online runners...")
    runners = get_runners(repo, online_only=True)

    # Build label -> set of online runner names from API
    api_label_runners = defaultdict(set)
    if runners:
        for runner in runners:
            for label in runner.get("labels", []):
                label_name = label.get("name", "")
                if label_name not in DEFAULT_LABELS_TO_IGNORE:
                    api_label_runners[label_name].add(runner["name"])
        print(f"Got {len(runners)} online runners from API")
    else:
        print("No runner API access, will use observed runners from job data")

    # Track runners seen in jobs (for labels not in API or when API unavailable)
    job_label_runners = defaultdict(set)
    label_jobs = defaultdict(list)  # label -> list of job_info
    # Per-host accumulation: each physical machine appears once regardless of
    # how many overlapping labels it advertises. This is what we use for the
    # "Per Host Utilization" section (the source-of-truth view).
    host_jobs = defaultdict(list)  # runner_name -> list of job_info
    host_labels = defaultdict(set)  # runner_name -> set of labels it ran jobs under

    # Fetch jobs for all runs in parallel. Cap concurrency lower than the
    # GH API secondary rate-limit threshold to avoid bursts that silently
    # drop runs even with retries.
    total_runs = len(runs)
    print(f"Fetching jobs for {total_runs} runs in parallel...")

    def fetch_jobs_for_run(run):
        """Fetch jobs for a single run.

        Returns (run_id, jobs, error_msg). `error_msg` is None on success.
        We surface failures rather than silently dropping the run so the
        caller can report how many runs' jobs are missing — silently
        dropping previously caused 4-gpu-b200 (and every other label) to
        report wildly different numbers depending on transient API hiccups.
        """
        try:
            return (run["id"], get_jobs_for_run(repo, run["id"]), None)
        except Exception as e:
            return (run["id"], None, str(e)[:200])

    all_jobs = []
    failed_runs = []
    # Concurrency=4 with longer retry budget keeps us well below the GH
    # API secondary rate-limit threshold (~10 req/s). On a 24h window
    # with ~1500 GPU-relevant runs (post-filter), this completes in ~5
    # min and almost never hits the rate limit.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_jobs_for_run, run) for run in runs]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(
                    f"Fetched jobs for {completed}/{total_runs} runs "
                    f"({len(failed_runs)} failed so far)..."
                )
            run_id, jobs, err = future.result()
            if err:
                failed_runs.append((run_id, err))
            elif jobs:
                all_jobs.extend(jobs)

    print(f"Processing {len(all_jobs)} jobs...")
    if failed_runs:
        print(
            f"WARNING: {len(failed_runs)}/{total_runs} runs failed to fetch "
            f"after retries. Utilization will be undercounted. "
            f"First few errors:"
        )
        for rid, err in failed_runs[:5]:
            print(f"  run {rid}: {err}")
    fetch_failure_pct = len(failed_runs) / total_runs * 100 if total_runs > 0 else 0

    # `now` anchors the wait of jobs that are still queued or running. It is
    # captured once so every in-flight job is measured against a single
    # reference (matches window_end below to within processing time).
    now = datetime.now(timezone.utc)
    all_job_infos = []  # one entry per job (deduped across labels) for detail views
    for job in all_jobs:
        job_info = classify_job(job, now)
        if job_info is None:
            continue
        all_job_infos.append(job_info)
        runner_name = job_info["runner_name"]

        # Per-host busy time only applies to jobs that actually occupied a
        # runner (ran or still running); a still-queued job has no host yet.
        if job_info["start"] is not None and runner_name:
            host_jobs[runner_name].append(job_info)

        for label in job_info["labels"]:
            if runner_name:
                job_label_runners[label].add(runner_name)
                host_labels[runner_name].add(label)
            label_jobs[label].append(job_info)

    # Merge API runners and job-observed runners
    # Prefer API count (online runners) when available
    # Include labels seen only on still-queued jobs (no online runner, no
    # completed job under them yet) so a fully-backed-up pool still reports.
    all_labels = (
        set(api_label_runners.keys())
        | set(job_label_runners.keys())
        | set(label_jobs.keys())
    )

    # Filter labels if specified
    if runner_filter:
        all_labels = {lbl for lbl in all_labels if runner_filter in lbl}

    print(f"Tracking {len(all_labels)} runner labels: {sorted(all_labels)}")

    window_seconds = hours * 3600
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=hours)

    # Per-host window-clamped busy time (each physical machine counted once).
    # This is the source of truth for how loaded each host actually is.
    host_busy_seconds = {
        host: _wallclock_busy_seconds(jobs, window_start, window_end)
        for host, jobs in host_jobs.items()
    }

    results = []
    for label in sorted(all_labels):
        # Hosts to attribute to this label = union of currently-online
        # runners advertising the label PLUS hosts that actually ran a
        # job under it during the window. The union catches hosts that
        # went offline mid-window (their busy time is still real
        # capacity consumed) and hosts that came online late.
        hosts = api_label_runners.get(label, set()) | job_label_runners.get(
            label, set()
        )
        num_runners = len(hosts) if hosts else 1

        # Pool busy time: sum of busy time across the hosts that could
        # serve this label, regardless of which sibling label actually
        # dispatched the job. This is the right denominator/numerator for
        # asking "how saturated is the underlying hardware that this
        # label depends on?" — sibling labels (e.g. `4-gpu-b200` and
        # `4-gpu-b200-low-disk`) compete for the same physical machines,
        # so their busy time should not be double-counted into separate
        # capacity buckets.
        active_seconds = sum(host_busy_seconds.get(h, 0.0) for h in hosts)
        capacity_seconds = num_runners * window_seconds
        utilization = (
            (active_seconds / capacity_seconds * 100) if capacity_seconds > 0 else 0
        )

        # Job count + queue stats stay label-specific (only jobs that
        # were dispatched under THIS label).
        jobs = label_jobs.get(label, [])
        queue_times = [j["queue_time"] for j in jobs if j["queue_time"] > 0]
        avg_queue = sum(queue_times) / len(queue_times) if queue_times else 0
        max_queue = max(queue_times) if queue_times else 0
        # Outcome breakdown for this label (pass/fail/cancel/running/queued).
        status_counts = dict(Counter(j["status"] for j in jobs))

        # Concurrency / saturation / queue-depth metrics. Use observed
        # peak as effective capacity if it's lower than the API count
        # (e.g. for autoscaling pools where most listeners sit idle).
        conc_initial = calculate_concurrency_metrics(
            jobs, window_start, window_end, num_runners
        )
        effective_runners = (
            min(num_runners, conc_initial["peak_concurrent"]) or num_runners
        )
        if effective_runners < num_runners and effective_runners > 0:
            conc = calculate_concurrency_metrics(
                jobs, window_start, window_end, effective_runners
            )
        else:
            conc = conc_initial

        results.append(
            {
                "label": label,
                "num_runners": num_runners,
                "effective_runners": effective_runners,
                "num_jobs": len(jobs),
                "total_active_hours": active_seconds / 3600,
                "utilization_pct": utilization,
                "avg_queue_min": avg_queue / 60,
                "max_queue_min": max_queue / 60,
                "peak_concurrent": conc_initial["peak_concurrent"],
                "avg_concurrent": conc["avg_concurrent"],
                "saturation_hours": conc["saturation_seconds"] / 3600,
                "saturation_pct": conc["saturation_pct"],
                "peak_queue": conc["peak_queue"],
                "status_counts": status_counts,
            }
        )

    # Per-job detail (deduped across labels), longest waits first, for the
    # links + status section of the report.
    longest_waits = sorted(all_job_infos, key=lambda j: j["queue_time"], reverse=True)
    queue_timeline = build_queue_timeline(label_jobs, window_start, window_end)
    # Hand utilization to build_load_buckets so it can include
    # small-but-saturated pools (e.g. 8-gpu-b200 at 89.5% util on a
    # single runner) that the peak-demand ranking alone would demote.
    utilization_by_label = {r["label"]: r["utilization_pct"] for r in results}
    load_buckets = build_load_buckets(
        label_jobs,
        window_start,
        window_end,
        utilization_by_label=utilization_by_label,
    )
    return (
        results,
        fetch_failure_pct,
        longest_waits,
        queue_timeline,
        load_buckets,
    )


# Distinct line colors paired with a matching legend emoji, in the same order,
# so the emoji swatch identifies each line — xychart-beta has no built-in
# legend. Series get palette colors in declaration order.
QUEUE_CHART_COLORS = [
    ("🔴", "#e6194B"),
    ("🟢", "#3cb44b"),
    ("🔵", "#4363d8"),
    ("🟠", "#f58231"),
    ("🟣", "#911eb4"),
    ("🟤", "#9A6324"),
    ("⚫", "#000000"),
    ("🟡", "#ffe119"),
]


def _bucket_label_fmt(window_hours: float) -> str:
    return "%m-%d %H:%M" if window_hours > 24 else "%H:%M"


def build_queue_timeline(label_jobs, window_start, window_end, max_series=8):
    """Sample each runner pool's longest in-queue wait across the window.

    At each sample time t, a pool's value is the max (t - created_at) over
    jobs dispatched under that pool that were still waiting at t
    (created_at <= t < queue_end), in minutes — i.e. how long the worst waiter
    had been queued at that instant. Queue time is inherently per-pool: a
    queued job has no host yet, so it can't be tied to a physical runner.

    Returns (sample_labels, [(pool, [values...]), ...]) for the pools with the
    highest peak wait, capped at max_series (the palette size).
    """
    total = (window_end - window_start).total_seconds()
    if total <= 0 or not label_jobs:
        return [], []
    hours = total / 3600
    # Keep the number of x-axis labels low (one per point in mermaid) so they
    # stay readable — ~2h resolution for a 24h window (~13 labels).
    n = max(8, min(12, round(hours)))
    step = total / n
    samples = [window_start + timedelta(seconds=step * i) for i in range(n + 1)]
    fmt = _bucket_label_fmt(hours)
    sample_labels = [t.astimezone(DISPLAY_TZ).strftime(fmt) for t in samples]

    series = []
    for label, jobs in label_jobs.items():
        waits = [
            (j["created_at"], j["queue_end"])
            for j in jobs
            if j.get("created_at")
            and j.get("queue_end")
            and j["queue_end"] > j["created_at"]
        ]
        if not waits:
            continue
        values = []
        for t in samples:
            best = 0.0
            for c, qe in waits:
                # Inclusive at qe so the peak (at pickup, or `now` for a job
                # still queued at the window end) lands on a sample.
                if c <= t <= qe:
                    best = max(best, (t - c).total_seconds() / 60.0)
            values.append(round(best, 1))
        peak = max(values)
        if peak > 0:
            series.append((label, values, peak))
    series.sort(key=lambda s: s[2], reverse=True)
    return sample_labels, [(lbl, vals) for lbl, vals, _ in series[:max_series]]


def build_load_buckets(
    label_jobs,
    window_start,
    window_end,
    max_pools=8,
    *,
    utilization_by_label=None,
    max_total=12,
):
    """Per runner pool, count jobs running and queued during each hourly bucket.

    A job is *running* in a bucket if its [start, end] interval overlaps it, and
    *queued* if its waiting interval [created_at, queue_end] overlaps it.
    Returns (bucket_labels, [(pool, running[], queued[]), ...]) for the
    pools selected by the ranking below.

    Selection:
      - When `utilization_by_label` is None: top `max_pools` by peak
        (running+queued). This is the legacy behavior.
      - When given: union of (top `max_pools` by peak) and (top
        `max_pools` by utilization%), de-duplicated and capped at
        `max_total`. Peak-ranked pools come first so the visual order
        matches the queue chart; utilization-ranked additions follow.

    Why the hybrid: ranking purely by peak (running+queued) demotes
    small-but-saturated pools. e.g. 8-gpu-b200 has 1 runner busy 89.5%
    of the window but peak running+queued is ~7, putting it around
    15th place. With the hybrid it lands via its 94% utilization
    instead. `max_pools=8` matches `build_queue_timeline`'s
    `max_series=8` so both charts cover the same primary pool set.
    """
    total = (window_end - window_start).total_seconds()
    if total <= 0 or not label_jobs:
        return [], []
    hours = total / 3600
    n = max(6, min(48, round(hours)))  # ~hourly buckets
    step = total / n
    edges = [window_start + timedelta(seconds=step * i) for i in range(n + 1)]
    fmt = _bucket_label_fmt(hours)
    bucket_labels = [edges[i].astimezone(DISPLAY_TZ).strftime(fmt) for i in range(n)]

    out = []
    for label, jobs in label_jobs.items():
        running = [0] * n
        queued = [0] * n
        for j in jobs:
            s, e = j.get("start"), j.get("end")
            c, qe = j.get("created_at"), j.get("queue_end")
            for i in range(n):
                b0, b1 = edges[i], edges[i + 1]
                if s is not None and e is not None and s < b1 and e > b0:
                    running[i] += 1
                if c is not None and qe is not None and c < b1 and qe > b0:
                    queued[i] += 1
        peak = max((r + q for r, q in zip(running, queued)), default=0)
        if peak > 0:
            out.append((label, running, queued, peak))

    by_peak = sorted(out, key=lambda x: x[3], reverse=True)
    peak_top = [x[0] for x in by_peak[:max_pools]]

    if utilization_by_label is None:
        selected = peak_top
    else:
        # Restrict utilization ranking to pools that had nonzero
        # peak (running+queued) data this window -- a pool with high
        # historical utilization but no observed activity in this
        # window has nothing to plot.
        candidates = {x[0] for x in out}
        util_top = sorted(
            (lbl for lbl in candidates if utilization_by_label.get(lbl, 0) > 0),
            key=lambda lbl: utilization_by_label[lbl],
            reverse=True,
        )[:max_pools]
        # dict.fromkeys preserves first-seen order: peak-rank first,
        # utilization-only additions after, then cap at max_total.
        selected = list(dict.fromkeys(peak_top + util_top))[:max_total]

    by_label = {label: (label, r, q) for label, r, q, _ in out}
    return bucket_labels, [by_label[lbl] for lbl in selected if lbl in by_label]


def format_report(
    results: list[dict],
    hours: int,
    fetch_failure_pct: float = 0.0,
    longest_waits: list = None,
    top_n: int = 20,
    queue_timeline: tuple = None,
    load_buckets: tuple = None,
) -> str:
    """One compact summary table — original schema, fixed columns.

    Active (hrs) and Utilization now reflect the actual host pool's
    busy time (sum across all jobs on the hosts that advertise this
    label, regardless of which sibling label dispatched them). This
    makes the column meaningful for shared host pools — e.g.
    `4-gpu-b200` and `4-gpu-b200-low-disk` both consume the same
    physical hosts, so their utilization now reflects real hardware
    saturation instead of being divided across labels.
    """
    lines = [
        "# Runner Utilization Report",
        "",
        f"**Time window:** Last {hours} hours · "
        f"**Generated:** "
        f"{datetime.now(DISPLAY_TZ).strftime('%Y-%m-%d %H:%M')} {TZ_LABEL}",
        "",
    ]
    if fetch_failure_pct > 1.0:
        lines.append(
            f"⚠️ **Data completeness warning**: {fetch_failure_pct:.0f}% of "
            f"GPU-relevant workflow runs failed to fetch jobs after retries "
            f"(GH API rate limit). Active hours and utilization below are "
            f"under-counted by approximately this fraction."
        )
        lines.append("")
    lines.extend(
        [
            "| Label | Runners | Jobs | Active (hrs) | Utilization | Avg Queue | Max Queue | Status |",
            "|-------|---------|------|--------------|-------------|-----------|-----------|--------|",
        ]
    )
    for r in results:
        bar = "█" * int(r["utilization_pct"] / 10) + "░" * (
            10 - int(r["utilization_pct"] / 10)
        )
        lines.append(
            f"| {r['label']} | {r['num_runners']} | {r['num_jobs']} | "
            f"{r['total_active_hours']:.1f} | "
            f"{r['utilization_pct']:.1f}% {bar} | "
            f"{r['avg_queue_min']:.1f}m | {r['max_queue_min']:.1f}m | "
            f"{format_status_counts(r.get('status_counts', {}))} |"
        )

    # Queue wait over time — one big multi-line chart, one line per runner pool.
    if queue_timeline and queue_timeline[1]:
        sample_labels, series = queue_timeline
        palette = ",".join(c for _, c in QUEUE_CHART_COLORS[: len(series)])
        ymax = int(max(max(vals) for _, vals in series) * 1.1) + 5
        x_cats = ", ".join(f'"{s}"' for s in sample_labels)
        legend = "  ".join(
            f"{emoji} `{lbl}`"
            for (emoji, _), (lbl, _) in zip(QUEUE_CHART_COLORS, series)
        )
        lines.extend(
            [
                "",
                "## Queue Wait Over Time",
                "",
                f"Longest in-queue wait per runner pool across the window "
                f"(minutes; times in {TZ_LABEL}). Queue time is per-pool — a "
                f"queued job isn't on a host yet, so it can't be attributed to "
                f"a physical runner.",
                "",
                f"**Pools:** {legend}",
                "",
                "```mermaid",
                '%%{init: {"xyChart": {"width": 1500, "height": 520}, '
                '"themeVariables": {"xyChart": {"plotColorPalette": "'
                + palette
                + '"}}}}%%',
                "xychart-beta",
                '    title "Queue Wait Over Time (min, per runner pool)"',
                f'    x-axis "Time ({TZ_LABEL})" [{x_cats}]',
                f'    y-axis "Wait (min)" 0 --> {ymax}',
            ]
        )
        for _, vals in series:
            lines.append("    line [" + ", ".join(str(v) for v in vals) + "]")
        lines.append("```")

    # Running vs queued per hour — one bar+line chart per busy runner pool.
    if load_buckets and load_buckets[1]:
        bucket_labels, pools = load_buckets
        x_cats = ", ".join(f'"{s}"' for s in bucket_labels)
        lines.extend(
            [
                "",
                "## Running vs Queued Per Hour",
                "",
                f"Per runner pool: jobs **running** (🔵 line) and **queued** "
                f"(🟠 bars) during each hourly bucket (times in {TZ_LABEL}). "
                f"Bars rising above the line mean demand outran capacity and a "
                f"backlog built up.",
            ]
        )
        for lbl, running, queued in pools:
            ymax = int(max(max(running), max(queued), 1) * 1.1) + 1
            lines.extend(
                [
                    "",
                    f"### {lbl}",
                    "",
                    "```mermaid",
                    '%%{init: {"xyChart": {"width": 1200, "height": 340}, '
                    '"themeVariables": {"xyChart": {"plotColorPalette": '
                    '"#f58231,#4363d8"}}}}%%',
                    "xychart-beta",
                    f'    title "{lbl} — running (line) & queued (bars) per hour"',
                    f"    x-axis [{x_cats}]",
                    f'    y-axis "Jobs" 0 --> {ymax}',
                    "    bar [" + ", ".join(str(v) for v in queued) + "]",
                    "    line [" + ", ".join(str(v) for v in running) + "]",
                    "```",
                ]
            )

    # Longest queue waits — links to the actual jobs, with live status, so the
    # worst waits (including jobs still queued/running right now) are one click
    # away. This is the detail behind the Max Queue column.
    waits = [j for j in (longest_waits or []) if j.get("queue_time", 0) > 0][:top_n]
    if waits:
        lines.extend(
            [
                "",
                f"## Longest Queue Waits (top {len(waits)})",
                "",
                "| Wait | Status | Label | Job |",
                "|------|--------|-------|-----|",
            ]
        )
        for j in waits:
            status = j.get("status", "")
            emoji = STATUS_EMOJI.get(status, "")
            label = ", ".join(j.get("labels", [])) or "—"
            name = j.get("job_name", "job")
            url = j.get("html_url", "")
            job_cell = f"[{name}]({url})" if url else name
            lines.append(
                f"| {j['queue_time'] / 60:.0f}m | {emoji} {status} | "
                f"{label} | {job_cell} |"
            )

    # Concurrency Analysis section
    lines.extend(
        [
            "",
            "## Concurrency Analysis",
            "",
            "| Label | Runners (API/Effective) | Peak Concurrent | Avg Concurrent | Saturation Time | Peak Queue |",
            "|-------|-------------------------|-----------------|----------------|-----------------|------------|",
        ]
    )
    for r in results:
        effective = r["effective_runners"]
        avg_pct = (r["avg_concurrent"] / effective * 100) if effective > 0 else 0
        runner_str = (
            f"{r['num_runners']}/{effective}"
            if effective != r["num_runners"]
            else str(r["num_runners"])
        )
        lines.append(
            f"| {r['label']} | {runner_str} | "
            f"{r['peak_concurrent']} | "
            f"{r['avg_concurrent']:.1f} ({avg_pct:.0f}%) | "
            f"{r['saturation_hours']:.1f}h ({r['saturation_pct']:.0f}%) | "
            f"{r['peak_queue']} jobs |"
        )

    # Recommendations
    lines.extend(["", "## Recommendations", ""])
    has_recs = False
    for r in results:
        label = r["label"]
        sat_pct = r["saturation_pct"]
        peak_q = r["peak_queue"]
        effective = r["effective_runners"]
        avg_pct = (r["avg_concurrent"] / effective * 100) if effective > 0 else 0
        if sat_pct > 50 or peak_q > 5:
            lines.append(
                f"⚠️ **{label}**: High saturation ({sat_pct:.0f}%) "
                f"with queue buildup ({peak_q} jobs). Consider adding runners."
            )
            has_recs = True
        elif sat_pct > 20 or peak_q > 0:
            lines.append(
                f"📊 **{label}**: Moderate saturation ({sat_pct:.0f}%), "
                f"peak queue {peak_q} jobs. Monitor for trends."
            )
            has_recs = True
        elif avg_pct < 30 and r["num_jobs"] > 0:
            lines.append(
                f"💡 **{label}**: Low average utilization ({avg_pct:.0f}%). "
                f"Runner pool may be oversized."
            )
            has_recs = True
        else:
            lines.append(f"✓ **{label}**: Healthy utilization with minimal queueing.")
    if not has_recs and results:
        lines.append("All runner pools have healthy utilization.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate runner utilization report")
    parser.add_argument("--repo", default="sgl-project/sglang", help="GitHub repo")
    parser.add_argument(
        "--hours", type=float, default=24, help="Time window in hours (fractional ok)"
    )
    parser.add_argument(
        "--filter", type=str, help="Filter runner labels (e.g., '5090', 'h200')"
    )
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()

    results, fetch_failure_pct, longest_waits, queue_timeline, load_buckets = (
        calculate_utilization(args.repo, args.hours, args.filter)
    )
    report = format_report(
        results,
        args.hours,
        fetch_failure_pct,
        longest_waits=longest_waits,
        queue_timeline=queue_timeline,
        load_buckets=load_buckets,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Also write to GITHUB_STEP_SUMMARY if available
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(report)


if __name__ == "__main__":
    main()
