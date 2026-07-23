#!/usr/bin/env python3
"""Build a public MLU CI reliability report from GitHub Actions artifacts."""

import argparse
import io
import json
import os
import re
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
DEFAULT_WORKFLOW = "pr-test-mlu.yml"
DEFAULT_JOB = "pr-test-mlu"
ARTIFACT_PREFIX = "mlu-ci-result-pr"
INFRA_TIMEOUT_STAGES = {
    "external_task",
    "github_client",
    "mlu_resource_queue",
}
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GitHubApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class StripCrossHostAuthorization(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            old_host = urllib.parse.urlsplit(req.full_url).netloc
            new_host = urllib.parse.urlsplit(newurl).netloc
            if old_host.lower() != new_host.lower():
                redirected.remove_header("Authorization")
        return redirected


class GitHubApi:
    def __init__(self, token: str):
        if not token:
            raise ValueError("GITHUB_TOKEN or GH_TOKEN is required")
        self.token = token
        self.opener = urllib.request.build_opener(StripCrossHostAuthorization())

    def request(
        self, path: str, *, accept: str = "application/vnd.github+json"
    ) -> bytes:
        request = urllib.request.Request(
            f"{API_ROOT}{path}",
            headers={
                "Accept": accept,
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "sglang-mlu-ci-reliability-report",
                "X-GitHub-Api-Version": API_VERSION,
            },
        )
        try:
            with self.opener.open(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GitHubApiError(
                exc.code, f"GitHub API GET {path} failed: HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"GitHub API GET {path} unavailable: {exc.reason}"
            ) from exc

    def get_json(self, path: str) -> dict[str, Any]:
        payload = json.loads(self.request(path))
        if not isinstance(payload, dict):
            raise RuntimeError(f"GitHub API returned non-object JSON for {path}")
        return payload


def parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def isoformat(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def paginate(api: GitHubApi, path: str, key: str) -> list[dict[str, Any]]:
    separator = "&" if "?" in path else "?"
    values: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = api.get_json(f"{path}{separator}per_page=100&page={page}")
        batch = payload.get(key)
        if not isinstance(batch, list):
            raise RuntimeError(f"GitHub API response is missing {key!r}")
        values.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            return values
        page += 1


def list_workflow_runs(
    api: GitHubApi,
    repo: str,
    workflow: str,
    start: datetime,
    end: datetime,
    event: str,
) -> list[dict[str, Any]]:
    encoded_workflow = urllib.parse.quote(workflow, safe="")
    query_values = {"created": f">={isoformat(start)}"}
    if event != "all":
        query_values["event"] = event
    query = urllib.parse.urlencode(query_values)
    path = f"/repos/{repo}/actions/workflows/{encoded_workflow}/runs?{query}"
    runs = paginate(api, path, "workflow_runs")
    return [
        run
        for run in runs
        if (created_at := parse_timestamp(str(run.get("created_at", ""))))
        and start <= created_at < end
    ]


def list_attempt_jobs(
    api: GitHubApi, repo: str, run_id: int, attempt: int, latest_attempt: int
) -> list[dict[str, Any]]:
    attempt_path = f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
    try:
        return paginate(api, attempt_path, "jobs")
    except GitHubApiError as exc:
        if exc.status != 404 or attempt != latest_attempt:
            raise
    return paginate(api, f"/repos/{repo}/actions/runs/{run_id}/jobs", "jobs")


def job_name_matches(name: str, target: str) -> bool:
    return name == target or name.endswith(f" / {target}")


def find_artifact(
    artifacts: list[dict[str, Any]], run_id: int, attempt: int
) -> Optional[dict[str, Any]]:
    expected_name = f"{ARTIFACT_PREFIX}-{run_id}-{attempt}"
    matches = [
        artifact for artifact in artifacts if artifact.get("name") == expected_name
    ]
    if not matches:
        return None
    return max(matches, key=lambda artifact: str(artifact.get("created_at", "")))


def read_result_metadata(archive: bytes) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            candidates = sorted(
                name
                for name in bundle.namelist()
                if Path(name).name.startswith("result-") and name.endswith(".json")
            )
            if not candidates:
                raise ValueError("artifact does not contain result-*.json")
            payload = json.loads(bundle.read(candidates[-1]))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid result artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("result metadata is not a JSON object")
    return payload


def duration_seconds(start: str, end: str) -> Optional[float]:
    start_time = parse_timestamp(start)
    end_time = parse_timestamp(end)
    if start_time is None or end_time is None or end_time < start_time:
        return None
    return (end_time - start_time).total_seconds()


def collect_records(
    api: GitHubApi,
    repo: str,
    workflow: str,
    job_name: str,
    start: datetime,
    end: datetime,
    event: str = "pull_request_target",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    collection_errors: list[dict[str, Any]] = []
    for run in list_workflow_runs(api, repo, workflow, start, end, event):
        run_id = int(run["id"])
        latest_attempt = max(1, int(run.get("run_attempt", 1)))
        artifacts: Optional[list[dict[str, Any]]] = None
        for attempt in range(1, latest_attempt + 1):
            try:
                jobs = list_attempt_jobs(api, repo, run_id, attempt, latest_attempt)
            except Exception as exc:
                collection_errors.append(
                    {
                        "run_id": run_id,
                        "attempt": attempt,
                        "stage": "jobs",
                        "error": str(exc),
                    }
                )
                continue
            target_jobs = [
                job
                for job in jobs
                if job_name_matches(str(job.get("name", "")), job_name)
                and job.get("conclusion") != "skipped"
            ]
            for job in target_jobs:
                if artifacts is None:
                    try:
                        artifacts = paginate(
                            api,
                            f"/repos/{repo}/actions/runs/{run_id}/artifacts",
                            "artifacts",
                        )
                    except Exception as exc:
                        artifacts = []
                        collection_errors.append(
                            {
                                "run_id": run_id,
                                "attempt": attempt,
                                "stage": "artifacts",
                                "error": str(exc),
                            }
                        )
                artifact = find_artifact(artifacts, run_id, attempt)
                metadata: Optional[dict[str, Any]] = None
                metadata_error = ""
                if artifact is not None and not artifact.get("expired", False):
                    try:
                        archive = api.request(
                            f"/repos/{repo}/actions/artifacts/{int(artifact['id'])}/zip",
                            accept="application/vnd.github+json",
                        )
                        metadata = read_result_metadata(archive)
                    except Exception as exc:
                        metadata_error = str(exc)

                created_at = str(job.get("created_at") or run.get("created_at") or "")
                started_at = str(job.get("started_at") or "")
                completed_at = str(job.get("completed_at") or "")
                runner_name = str(job.get("runner_name") or "")
                records.append(
                    {
                        "run_id": run_id,
                        "attempt": attempt,
                        "run_url": str(run.get("html_url") or ""),
                        "job_id": int(job.get("id") or 0),
                        "job_url": str(
                            job.get("html_url") or run.get("html_url") or ""
                        ),
                        "event": str(run.get("event") or ""),
                        "created_at": created_at,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "runner_assigned": bool(runner_name),
                        "runner_name": runner_name,
                        "job_status": str(job.get("status") or ""),
                        "job_conclusion": str(job.get("conclusion") or ""),
                        "queue_seconds": duration_seconds(created_at, started_at),
                        "runtime_seconds": duration_seconds(started_at, completed_at),
                        "artifact_uploaded": artifact is not None,
                        "artifact_expired": bool(
                            artifact and artifact.get("expired", False)
                        ),
                        "metadata": metadata,
                        "metadata_error": metadata_error,
                    }
                )
    return records, collection_errors


def result_bucket(record: dict[str, Any]) -> tuple[str, str]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        status = str(metadata.get("status") or "")
        failure_type = str(metadata.get("failure_type") or "")
        failure_stage = str(metadata.get("failure_stage") or "")
        if "success" in status and not failure_type:
            return "success", "success"
        if failure_type:
            return failure_type, failure_stage or "unknown"
        return "unknown", failure_stage or status or "unknown"
    if record.get("job_conclusion") == "cancelled":
        return "cancelled", "no_result_metadata"
    return "missing_metadata", "no_result_metadata"


def is_infrastructure_result(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return False
    failure_type = str(metadata.get("failure_type") or "")
    failure_stage = str(metadata.get("failure_stage") or "")
    return failure_type == "infrastructure" or (
        failure_type == "timeout" and failure_stage in INFRA_TIMEOUT_STAGES
    )


def percentile(values: list[float], percentile_value: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def build_summary(
    records: list[dict[str, Any]],
    collection_errors: list[dict[str, Any]],
    repo: str,
    workflow: str,
    start: datetime,
    end: datetime,
    event: str = "pull_request_target",
) -> dict[str, Any]:
    classifications = Counter(result_bucket(record) for record in records)
    assigned = [record for record in records if record["runner_assigned"]]
    completed = [record for record in records if record["job_status"] == "completed"]
    completed_assigned = [record for record in completed if record["runner_assigned"]]
    with_metadata = [
        record for record in records if isinstance(record.get("metadata"), dict)
    ]
    infra_results = [
        record for record in with_metadata if is_infrastructure_result(record)
    ]
    queue_values = [
        float(record["queue_seconds"])
        for record in assigned
        if record.get("queue_seconds") is not None
    ]
    runtime_values = [
        float(record["runtime_seconds"])
        for record in assigned
        if record.get("runtime_seconds") is not None
    ]
    classified_reliability = None
    if with_metadata:
        classified_reliability = 1 - len(infra_results) / len(with_metadata)
    metadata_coverage = None
    if completed_assigned:
        metadata_coverage = len(with_metadata) / len(completed_assigned)
    return {
        "schema_version": 1,
        "repository": repo,
        "workflow": workflow,
        "event": event,
        "period_start": isoformat(start),
        "period_end": isoformat(end),
        "generated_at": isoformat(datetime.now(timezone.utc)),
        "counts": {
            "mlu_job_attempts": len(records),
            "runner_assigned": len(assigned),
            "runner_unassigned": len(records) - len(assigned),
            "job_completed": len(completed),
            "job_pending": len(records) - len(completed),
            "job_cancelled": sum(
                record["job_conclusion"] == "cancelled" for record in records
            ),
            "unclaimed_completed": sum(
                not record["runner_assigned"] for record in completed
            ),
            "artifact_uploaded": sum(record["artifact_uploaded"] for record in records),
            "result_metadata": len(with_metadata),
            "infrastructure_results": len(infra_results),
            "collection_errors": len(collection_errors),
        },
        "classified_reliability": classified_reliability,
        "metadata_coverage_for_assigned_jobs": metadata_coverage,
        "classifications": [
            {"failure_type": key[0], "failure_stage": key[1], "count": count}
            for key, count in sorted(classifications.items())
        ],
        "latency_seconds": {
            "queue_median": statistics.median(queue_values) if queue_values else None,
            "queue_p95": percentile(queue_values, 0.95),
            "runtime_median": statistics.median(runtime_values)
            if runtime_values
            else None,
            "runtime_p95": percentile(runtime_values, 0.95),
        },
        "records": records,
        "collection_errors": collection_errors,
    }


def percentage(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    latency = summary["latency_seconds"]
    lines = [
        "<!-- mlu-ci-reliability-report -->",
        "## MLU CI reliability report",
        "",
        f"Period: `{summary['period_start']}` to `{summary['period_end']}`",
        "",
        f"Workflow: `{summary['workflow']}` in `{summary['repository']}`",
        f"Event filter: `{summary['event']}`",
        "",
        "### Coverage",
        "",
        f"- MLU job attempts: **{counts['mlu_job_attempts']}**",
        f"- Runner assigned / unassigned: **{counts['runner_assigned']} / {counts['runner_unassigned']}**",
        f"- Completed / pending: **{counts['job_completed']} / {counts['job_pending']}**",
        f"- Cancelled / completed without a runner: **{counts['job_cancelled']} / {counts['unclaimed_completed']}**",
        f"- Result metadata: **{counts['result_metadata']}**",
        f"- Artifact uploaded: **{counts['artifact_uploaded']}**",
        "- Metadata coverage among completed runner-assigned jobs: "
        f"**{percentage(summary['metadata_coverage_for_assigned_jobs'])}**",
        "- Classified non-infrastructure reliability: "
        f"**{percentage(summary['classified_reliability'])}**",
        "",
        "> Reliability is calculated only from jobs with structured result metadata. "
        "Unassigned jobs and missing metadata are reported separately rather than hidden.",
        "",
        "### Classifications",
        "",
        "| Type | Stage | Count |",
        "| --- | --- | ---: |",
    ]
    for item in summary["classifications"]:
        failure_type = str(item["failure_type"]).replace("|", "\\|")
        failure_stage = str(item["failure_stage"]).replace("|", "\\|")
        lines.append(f"| `{failure_type}` | `{failure_stage}` | {item['count']} |")
    if not summary["classifications"]:
        lines.append("| `n/a` | `n/a` | 0 |")

    lines.extend(
        [
            "",
            "### Latency",
            "",
            "| Metric | Median | p95 |",
            "| --- | ---: | ---: |",
            f"| GitHub runner queue | {format_duration(latency['queue_median'])} | "
            f"{format_duration(latency['queue_p95'])} |",
            f"| MLU GitHub job runtime | {format_duration(latency['runtime_median'])} | "
            f"{format_duration(latency['runtime_p95'])} |",
        ]
    )

    missing = [record for record in summary["records"] if record["metadata"] is None]
    if missing:
        lines.extend(
            [
                "",
                "### Jobs without structured result metadata",
                "",
            ]
        )
        for record in missing[:20]:
            label = f"run {record['run_id']} attempt {record['attempt']}"
            link = record["job_url"] or record["run_url"]
            if record["job_status"] != "completed":
                reason = f"job {record['job_status'] or 'pending'}"
            elif not record["runner_assigned"]:
                reason = "runner not assigned"
            else:
                reason = "metadata missing"
            if record["artifact_expired"]:
                reason = "artifact expired"
            elif record["metadata_error"]:
                reason = "artifact metadata unreadable"
            lines.append(
                f"- [{label}]({link}): {reason}, conclusion `{record['job_conclusion'] or 'unknown'}`"
            )
        if len(missing) > 20:
            lines.append(f"- ... and {len(missing) - 20} more")

    if summary["collection_errors"]:
        lines.extend(
            [
                "",
                f"> Warning: {len(summary['collection_errors'])} GitHub API collection error(s) "
                "occurred. Treat this report as incomplete.",
            ]
        )

    lines.extend(
        [
            "",
            "Generated from `result-<task-id>.json` artifacts and GitHub job metadata.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--job-name", default=DEFAULT_JOB)
    parser.add_argument(
        "--event",
        choices=(
            "pull_request",
            "pull_request_target",
            "push",
            "workflow_dispatch",
            "all",
        ),
        default="pull_request_target",
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--end-time", help="UTC ISO-8601 report end time; defaults to now"
    )
    parser.add_argument("--output", type=Path, default=Path("mlu-ci-reliability.md"))
    parser.add_argument(
        "--json-output", type=Path, default=Path("mlu-ci-reliability.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not REPOSITORY_RE.fullmatch(args.repo):
        print("--repo must be in owner/repository form", file=sys.stderr)
        return 2
    if args.days < 1 or args.days > 90:
        print("--days must be between 1 and 90", file=sys.stderr)
        return 2
    end = (
        parse_timestamp(args.end_time) if args.end_time else datetime.now(timezone.utc)
    )
    if end is None:
        print("--end-time must be an ISO-8601 timestamp", file=sys.stderr)
        return 2
    start = end - timedelta(days=args.days)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    try:
        api = GitHubApi(token)
        records, collection_errors = collect_records(
            api, args.repo, args.workflow, args.job_name, start, end, args.event
        )
        summary = build_summary(
            records,
            collection_errors,
            args.repo,
            args.workflow,
            start,
            end,
            args.event,
        )
    except Exception as exc:
        print(f"Failed to collect MLU CI reliability data: {exc}", file=sys.stderr)
        return 1

    args.output.write_text(render_markdown(summary), encoding="utf-8")
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} and {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
