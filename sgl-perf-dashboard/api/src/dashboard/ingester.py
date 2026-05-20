"""Periodic ingester: MinIO result JSONs -> SQLite.

Design:
- Idempotent: INSERT OR IGNORE on (github_run_id, run_attempt, config, concurrency)
- Resumable: a cursor in `ingester_state` records the last-seen object key.
  On first startup (no cursor) we do a full bucket scan; subsequent runs only
  look at objects whose LastModified > cursor.
- Graceful: GitHub enrichment failures don't block metric insertion — the
  run row gets the SHA we parsed and stays null for message/author/PR.
  A later retry can fill them in.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.client import Config as BotoConfig
from dashboard import anomaly, reconciler
from dashboard.config import settings
from dashboard.db import connect
from dashboard.github_client import CommitInfo, GitHubClient, PRInfo
from dashboard.s3_paths import ParsedPath, parse_result_key

logger = logging.getLogger(__name__)


# Metric field names we pull out of the result JSON. Kept verbatim per design-doc
# decision: "keep result JSON field names verbatim" (DLR-5301 §16). Anything
# numeric top-level gets captured even if not in this list — this is just to
# help reviewers see what the UI cares about by default.
KNOWN_METRIC_FIELDS = {
    "total_token_throughput",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "median_ttft_ms",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "median_e2e_latency_ms",
    "mean_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "completed",
    "failed",
    "latency_s",
    "mean_latency_s",
}

# Metrics with obvious units. Everything else leaves `unit` null.
UNIT_HINTS = {
    "ms": "milliseconds",
    "throughput": "tokens/sec",
    "latency_s": "seconds",
    "e2e_latency": "milliseconds",
}


def _infer_unit(metric_name: str) -> str | None:
    if metric_name.endswith("_ms"):
        return "milliseconds"
    if metric_name.endswith("_s"):
        return "seconds"
    if "throughput" in metric_name:
        return "tokens/sec"
    return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        region_name=settings.minio_region,
        config=BotoConfig(signature_version="s3v4"),
    )


def _read_json(s3, key: str) -> dict[str, Any] | None:
    try:
        obj = s3.get_object(Bucket=settings.minio_bucket, Key=key)
        body = obj["Body"].read()
        return json.loads(body)
    except (
        Exception
    ) as exc:  # noqa: BLE001 — log + move on; one bad file shouldn't halt
        logger.warning("failed to fetch/parse %s: %s", key, exc)
        return None


def _read_text(s3, key: str) -> str | None:
    try:
        obj = s3.get_object(Bucket=settings.minio_bucket, Key=key)
        return obj["Body"].read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to fetch %s: %s", key, exc)
        return None


def _list_ai_analysis_keys(s3) -> list[str]:
    """Every object ending in `/ai_analysis.md` across the bucket."""
    paginator = s3.get_paginator("list_objects_v2")
    out: list[str] = []
    for page in paginator.paginate(Bucket=settings.minio_bucket):
        for obj in page.get("Contents") or []:
            if obj["Key"].endswith("/ai_analysis.md"):
                out.append(obj["Key"])
    return out


def _sync_ai_analyses(conn: sqlite3.Connection, s3) -> int:
    """For each `ai_analysis.md` in S3, attach it to every run row that shares
    the same (github_run_id, attempt, config_name). A single markdown file
    applies to the whole matrix job — every concurrency row under that config.

    Idempotent: skips analyses already stored in ai_summaries for all matching
    runs. Returns count of newly-inserted rows.
    """
    keys = _list_ai_analysis_keys(s3)
    inserted = 0
    for key in keys:
        parts = key.split("/")
        # Expected: <trigger>/<run-attempt>/<seq_len>/<config>/ai_analysis.md
        if len(parts) != 5 or "-" not in parts[1]:
            continue
        run_attempt = parts[1]
        config_name = parts[3]
        try:
            github_run_id, attempt_str = run_attempt.rsplit("-", 1)
            attempt = int(attempt_str)
        except ValueError:
            continue

        run_rows = conn.execute(
            """
            SELECT id FROM runs
            WHERE github_run_id = ? AND github_run_attempt = ? AND config_name = ?
            """,
            (github_run_id, attempt, config_name),
        ).fetchall()
        if not run_rows:
            continue

        run_ids = [r["id"] for r in run_rows]
        placeholders = ",".join("?" for _ in run_ids)
        existing = conn.execute(
            f"""
            SELECT run_id FROM ai_summaries
            WHERE summary_type = 'log_analysis' AND run_id IN ({placeholders})
            """,
            run_ids,
        ).fetchall()
        existing_ids = {e["run_id"] for e in existing}

        missing = [rid for rid in run_ids if rid not in existing_ids]
        if not missing:
            continue

        body = _read_text(s3, key)
        if body is None:
            continue

        now = _now_iso()
        for rid in missing:
            conn.execute(
                """
                INSERT OR IGNORE INTO ai_summaries
                    (run_id, summary_type, body, model, generated_at)
                VALUES (?, 'log_analysis', ?, 'opencode-via-modal', ?)
                """,
                (rid, body, now),
            )
            inserted += 1
    return inserted


def _list_result_keys(s3, cursor: str | None) -> list[tuple[str, datetime]]:
    """Return (key, last_modified) for every result_concurrency_*.json newer than cursor.

    `cursor` is the ISO timestamp of the latest LastModified we processed.
    LastModified filtering is robust across trigger prefixes (lex ordering
    broke when `cron/` < `manual/` and a later cron run got skipped because
    its keys lex-sorted before the manual cursor).

    Old-format cursors (S3 keys, not timestamps) are treated as None — the
    next pass does a full rescan, then writes the new timestamp format.
    """
    cutoff: datetime | None = None
    if cursor:
        try:
            cutoff = datetime.fromisoformat(cursor)
        except ValueError:
            cutoff = None  # legacy key-format cursor; rescan once.

    paginator = s3.get_paginator("list_objects_v2")
    results: list[tuple[str, datetime]] = []
    for page in paginator.paginate(Bucket=settings.minio_bucket):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            if "/results_concurrency_" not in key:
                continue
            lm = obj["LastModified"]
            if cutoff is not None and lm <= cutoff:
                continue
            results.append((key, lm))
    return results


def _get_cursor(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT value FROM ingester_state WHERE key = 's3_cursor'"
    ).fetchone()
    return row["value"] if row else None


def _set_cursor(conn: sqlite3.Connection, cursor: str) -> None:
    conn.execute(
        """
        INSERT INTO ingester_state (key, value, updated_at)
        VALUES ('s3_cursor', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (cursor, _now_iso()),
    )


def _set_heartbeat(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO ingester_state (key, value, updated_at)
        VALUES ('last_run', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (_now_iso(), _now_iso()),
    )


def _insert_run(
    conn: sqlite3.Connection,
    parsed: ParsedPath,
    commit: CommitInfo | None,
    pr: PRInfo | None,
    started_at: datetime,
    s3_log_prefix: str,
) -> int | None:
    """INSERT OR IGNORE the run. Returns the row id (new or existing) or None."""
    run_url = (
        f"https://github.com/{settings.github_repo}/actions/runs/"
        f"{parsed.github_run_id}/attempts/{parsed.github_run_attempt}"
    )
    try:
        # Upgrade any placeholder row (failed/partial) that the reconciler may
        # have inserted for this same unique key. A late-arriving JSON should
        # replace the placeholder, not lose to it on INSERT OR IGNORE.
        conn.execute(
            """
            DELETE FROM runs
            WHERE github_run_id = ? AND github_run_attempt = ?
              AND config_name = ? AND concurrency = ?
              AND status IN ('failed', 'partial')
            """,
            (
                parsed.github_run_id,
                parsed.github_run_attempt,
                parsed.config_name,
                parsed.concurrency,
            ),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO runs (
                github_run_id, github_run_attempt, github_run_url,
                commit_sha, commit_short_sha, commit_message, commit_author, commit_date,
                pr_number, pr_title,
                trigger, config_name,
                model_prefix, precision, seq_len, isl, osl, recipe,
                concurrency, num_gpus, prefill_gpus, decode_gpus,
                started_at, completed_at, status,
                s3_log_prefix, slurm_job_id, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parsed.github_run_id,
                parsed.github_run_attempt,
                run_url,
                commit.sha if commit else None,
                commit.short_sha if commit else None,
                commit.message if commit else None,
                commit.author_login if commit else None,
                commit.date if commit else None,
                pr.number if pr else None,
                pr.title if pr else None,
                parsed.trigger,
                parsed.config_name,
                parsed.model_prefix,
                parsed.precision,
                parsed.seq_len,
                parsed.isl,
                parsed.osl,
                parsed.recipe,
                parsed.concurrency,
                parsed.num_gpus,
                parsed.prefill_gpus,
                parsed.decode_gpus,
                started_at.isoformat(),
                None,  # completed_at: unknown from the result JSON alone
                "passed",  # status: if the JSON exists at all, the run produced it
                s3_log_prefix,
                parsed.slurm_job_id,
                _now_iso(),
            ),
        )
    except sqlite3.IntegrityError as exc:
        logger.warning("insert failed for run %s: %s", asdict(parsed), exc)
        return None

    row = conn.execute(
        """
        SELECT id FROM runs
        WHERE github_run_id = ? AND github_run_attempt = ?
          AND config_name = ? AND concurrency = ?
        """,
        (
            parsed.github_run_id,
            parsed.github_run_attempt,
            parsed.config_name,
            parsed.concurrency,
        ),
    ).fetchone()
    return row["id"] if row else None


def _extract_metrics(payload: dict[str, Any]) -> list[tuple[str, float, str | None]]:
    """Turn a result JSON into [(name, value, unit), ...]. Numeric scalars only."""
    out: list[tuple[str, float, str | None]] = []
    for key, val in payload.items():
        if isinstance(val, bool):
            continue  # bool is a subtype of int — skip
        if isinstance(val, (int, float)):
            out.append((key, float(val), _infer_unit(key)))
    return out


def _insert_metrics(
    conn: sqlite3.Connection, run_id: int, metrics: list[tuple[str, float, str | None]]
) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO metrics (run_id, name, value, unit)
        VALUES (?, ?, ?, ?)
        """,
        [(run_id, name, value, unit) for name, value, unit in metrics],
    )


def run_once() -> dict[str, Any]:
    """Single pass: list MinIO, parse, enrich, insert. Returns stats dict."""
    s3 = _make_s3_client()
    github = GitHubClient(token=settings.github_token, repo=settings.github_repo)

    stats = {
        "listed": 0,
        "parsed": 0,
        "inserted": 0,
        "enriched": 0,
        "metrics_written": 0,
        "regressions_flagged": 0,
        "regressions_resolved": 0,
        "skipped": 0,
        "errors": 0,
        "reconcile_failed": 0,
        "reconcile_partial": 0,
        "reconcile_orphans": 0,
        "ai_summaries_fetched": 0,
    }

    newly_inserted_run_ids: list[int] = []

    with connect(settings.db_path) as conn:
        cursor = _get_cursor(conn)
        keys = _list_result_keys(s3, cursor)
        stats["listed"] = len(keys)

        # Sort by LastModified ascending so the cursor advances monotonically
        # with upload time, independent of key prefix (cron/ vs manual/ etc).
        keys.sort(key=lambda kv: kv[1])
        latest_lm: datetime | None = None

        def _advance(lm: datetime) -> None:
            nonlocal latest_lm
            if latest_lm is None or lm > latest_lm:
                latest_lm = lm

        for key, last_modified in keys:
            parsed = parse_result_key(key)
            if parsed is None:
                stats["skipped"] += 1
                _advance(last_modified)
                continue
            stats["parsed"] += 1

            payload = _read_json(s3, key)
            if payload is None:
                stats["errors"] += 1
                _advance(last_modified)
                continue

            commit: CommitInfo | None = None
            pr: PRInfo | None = None
            if github.enabled:
                try:
                    gh_run = github.get_workflow_run(parsed.github_run_id)
                    sha = (gh_run or {}).get("head_sha")
                    if sha:
                        commit = github.get_commit(sha)
                        pr = github.get_pr_for_commit(sha)
                        if commit or pr:
                            stats["enriched"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("github enrichment failed for %s: %s", key, exc)

            s3_log_prefix = "/".join(key.split("/")[:-1]) + "/"
            started_at = last_modified if last_modified else datetime.now(UTC)

            run_id = _insert_run(conn, parsed, commit, pr, started_at, s3_log_prefix)
            if run_id is None:
                stats["errors"] += 1
                _advance(last_modified)
                continue

            metrics = _extract_metrics(payload)
            if metrics:
                _insert_metrics(conn, run_id, metrics)
                stats["metrics_written"] += len(metrics)

            stats["inserted"] += 1
            newly_inserted_run_ids.append(run_id)
            _advance(last_modified)

        # Run anomaly detection for each new run. Do this after everything
        # is inserted so history queries include the freshest data.
        for new_id in newly_inserted_run_ids:
            try:
                flags = anomaly.detect_for_run(conn, new_id)
                stats["regressions_flagged"] += len(flags)
            except Exception as exc:  # noqa: BLE001
                logger.warning("anomaly detection failed for run %d: %s", new_id, exc)

        try:
            stats["regressions_resolved"] = anomaly.resolve_stale_regressions(conn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("regression resolution failed: %s", exc)

        if latest_lm is not None:
            _set_cursor(conn, latest_lm.isoformat())
        _set_heartbeat(conn)
        conn.commit()

        try:
            rc = reconciler.reconcile(conn)
            stats["reconcile_failed"] = rc.get("failed_inserted", 0)
            stats["reconcile_partial"] = rc.get("partial_inserted", 0)
            stats["reconcile_orphans"] = rc.get("orphans_found", 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconciler failed: %s", exc)

        try:
            stats["ai_summaries_fetched"] = _sync_ai_analyses(conn, s3)
            if stats["ai_summaries_fetched"]:
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ai-analysis sync failed: %s", exc)

    github.close()
    logger.info("ingest complete: %s", stats)
    return stats
