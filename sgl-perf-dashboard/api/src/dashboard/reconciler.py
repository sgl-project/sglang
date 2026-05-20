"""Phase-1 failed-run reconciler.

For each workflow run that has at least one ingested result in the DB, cross-
reference the expected matrix from `nightly-configs.yaml` and insert placeholder
rows for any missing (config, concurrency) pairs:

  - `status='failed'`  → the whole matrix job produced zero outputs (likely
                         crashed before any benchmark finished)
  - `status='partial'` → the matrix job produced some outputs but is missing
                         some concurrencies (usually a late-run OOM)

Inserted rows carry no metrics and are excluded from anomaly detection. They
inherit commit/PR metadata from a sibling row in the same workflow so that
failure pages still show "which commit was being tested".

Late-arriving JSONs: the ingester deletes any failed/partial placeholder with
the same unique key before inserting a real row, so delayed uploads upgrade
cleanly to `status='passed'`.

Phase 2 (GH reconciler) will add detection of complete-workflow failures where
zero rows exist yet for a workflow_run_id.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
import yaml
from botocore.client import Config as BotoConfig
from dashboard.s3_paths import parse_config_name, parse_seq_len

logger = logging.getLogger(__name__)


def _seq_len_str(isl: int, osl: int) -> str:
    def fmt(n: int) -> str:
        return f"{n // 1024}k" if n % 1024 == 0 else str(n)

    return f"{fmt(isl)}{fmt(osl)}"


def load_expected_matrix(
    config_path: str, runner: str = "gb200"
) -> dict[str, list[int]]:
    """Return {config_name: [expected_concurrencies]} for the given runner.

    Mirrors the naming convention in `scripts/ci/slurm/generate_matrix.py`:
        name = f"{model-prefix}-{precision}-{seq-len-str}-{topology}"
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "nightly-configs.yaml not found at %s; reconciler disabled", path
        )
        return {}

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    result: dict[str, list[int]] = {}
    for exp in data.values():
        if exp.get("runner") != runner:
            continue
        for seq_cfg in exp.get("seq-len-configs", []):
            isl, osl = seq_cfg["isl"], seq_cfg["osl"]
            sl = _seq_len_str(isl, osl)
            for entry in seq_cfg.get("search-space", []):
                topology = entry["config_file"].rsplit("/", 1)[-1].replace(".yaml", "")
                config_name = (
                    f"{exp['model-prefix']}-{exp['precision']}-{sl}-{topology}"
                )
                concs = entry.get("conc-list", []) or []
                result.setdefault(config_name, []).extend(concs)

    # Dedupe while preserving first-seen order.
    return {k: list(dict.fromkeys(v)) for k, v in result.items()}


# Skip workflows whose most recent row is newer than this — they may still be
# producing output. Conservative default: 3 hours past the last seen row.
INFLIGHT_GRACE = timedelta(hours=3)


def _workflow_runs_in_window(
    conn: sqlite3.Connection, window_days: int
) -> list[sqlite3.Row]:
    """Distinct (github_run_id, attempt) pairs eligible for reconciliation.

    Skip workflows whose most recent row is within INFLIGHT_GRACE of now —
    the job may still be producing output and we don't want to race it.

    Trigger-specific rules are applied later in ``reconcile()``:
      - cron: flag both whole-config failures and within-config partials.
      - manual: flag only within-config partials (users may run a filtered
        matrix via the workflow's ``configs`` input, so a whole-config gap is
        ambiguous without querying GH Actions — deferred to Phase 2).
    """
    now = datetime.now(UTC)
    cutoff_start = (now - timedelta(days=window_days)).isoformat()
    cutoff_inflight = (now - INFLIGHT_GRACE).isoformat()
    return conn.execute(
        """
        SELECT github_run_id, github_run_attempt, trigger,
               MIN(started_at) AS first_started,
               MAX(started_at) AS last_started
        FROM runs
        WHERE started_at >= ?
        GROUP BY github_run_id, github_run_attempt
        HAVING MAX(started_at) < ?
        """,
        (cutoff_start, cutoff_inflight),
    ).fetchall()


def _rows_for_workflow(
    conn: sqlite3.Connection, github_run_id: str, attempt: int
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT config_name, concurrency, status, github_run_url, trigger,
               commit_sha, commit_short_sha, commit_message, commit_author,
               commit_date, pr_number, pr_title, seq_len, started_at
        FROM runs
        WHERE github_run_id = ? AND github_run_attempt = ?
        """,
        (github_run_id, attempt),
    ).fetchall()


def _insert_placeholder(
    conn: sqlite3.Connection,
    github_run_id: str,
    attempt: int,
    config_name: str,
    concurrency: int,
    status: str,
    context: sqlite3.Row,
    now_iso: str,
) -> bool:
    """Insert a failed/partial placeholder. Returns True if a new row was inserted."""
    model_prefix, precision, recipe = parse_config_name(config_name)
    seq_len = context["seq_len"]
    isl, osl = parse_seq_len(seq_len) if seq_len else (None, None)

    failure_reason = (
        "job failed" if status == "failed" else "concurrency missing from matrix job"
    )

    cur = conn.execute(
        """
        INSERT OR IGNORE INTO runs (
            github_run_id, github_run_attempt, github_run_url,
            commit_sha, commit_short_sha, commit_message, commit_author, commit_date,
            pr_number, pr_title,
            trigger, config_name,
            model_prefix, precision, seq_len, isl, osl, recipe,
            concurrency,
            started_at, status,
            s3_log_prefix, ingested_at, failure_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            github_run_id,
            attempt,
            context["github_run_url"],
            context["commit_sha"],
            context["commit_short_sha"],
            context["commit_message"],
            context["commit_author"],
            context["commit_date"],
            context["pr_number"],
            context["pr_title"],
            context["trigger"],
            config_name,
            model_prefix,
            precision,
            seq_len,
            isl,
            osl,
            recipe,
            concurrency,
            context["started_at"],
            status,
            "",  # s3_log_prefix: no JSON to link to for a placeholder
            now_iso,
            failure_reason,
        ),
    )
    return cur.rowcount > 0


def _make_s3_client():
    from dashboard.config import settings

    return boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        region_name=settings.minio_region,
        config=BotoConfig(signature_version="s3v4"),
    )


def _discover_orphan_workflows(s3: Any) -> list[dict]:
    """Find workflow_run_id prefixes in S3 that contain no result JSONs.

    These represent workflows that crashed before any benchmark finished but
    still uploaded log artifacts. The ingester can't see them because it only
    indexes `results_concurrency_*.json`.

    Returns a list of dicts: {trigger, github_run_id, github_run_attempt,
    first_seen_at}.
    """
    from dashboard.config import settings

    orphans: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")

    now = datetime.now(UTC)
    cutoff_inflight = now - INFLIGHT_GRACE

    for trigger in ("cron", "manual"):
        resp = s3.list_objects_v2(
            Bucket=settings.minio_bucket, Prefix=f"{trigger}/", Delimiter="/"
        )
        for p in resp.get("CommonPrefixes") or []:
            prefix = p["Prefix"]  # e.g. "manual/24803056480-1/"
            has_result = False
            earliest: datetime | None = None
            latest: datetime | None = None
            # Paginated scan — a workflow's log upload can exceed 1000 keys.
            # Don't break early on has_result: we still need MAX(LastModified)
            # to apply the inflight grace correctly.
            for page in paginator.paginate(Bucket=settings.minio_bucket, Prefix=prefix):
                for obj in page.get("Contents") or []:
                    if "/results_concurrency_" in obj["Key"]:
                        has_result = True
                    lm = obj.get("LastModified")
                    if lm:
                        if earliest is None or lm < earliest:
                            earliest = lm
                        if latest is None or lm > latest:
                            latest = lm
            if has_result:
                continue

            # Inflight guard: the workflow may still be uploading. Wait until
            # the most recent S3 write is older than INFLIGHT_GRACE before
            # declaring this an orphan. Mirrors the per-workflow reconciler's
            # rule and prevents the bug where logs land minutes before JSONs
            # and we'd race the upload.
            if latest is not None and latest >= cutoff_inflight:
                continue

            # Parse "<trigger>/<run_id>-<attempt>/"
            parts = prefix.strip("/").split("/")
            if len(parts) != 2 or "-" not in parts[1]:
                continue
            run_id, attempt_str = parts[1].rsplit("-", 1)
            try:
                attempt = int(attempt_str)
            except ValueError:
                continue

            orphans.append(
                {
                    "trigger": trigger,
                    "github_run_id": run_id,
                    "github_run_attempt": attempt,
                    "started_at": (earliest or now).isoformat(),
                }
            )

    return orphans


def _insert_orphan_placeholder(
    conn: sqlite3.Connection,
    orphan: dict,
    config_name: str,
    concurrency: int,
    now_iso: str,
) -> bool:
    """Insert a failed placeholder for a workflow that produced no JSONs at all."""
    from dashboard.config import settings

    model_prefix, precision, recipe = parse_config_name(config_name)
    github_run_url = (
        f"https://github.com/{settings.github_repo}/actions/runs/"
        f"{orphan['github_run_id']}/attempts/{orphan['github_run_attempt']}"
    )
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO runs (
            github_run_id, github_run_attempt, github_run_url,
            trigger, config_name,
            model_prefix, precision, recipe,
            concurrency,
            started_at, status,
            s3_log_prefix, ingested_at, failure_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            orphan["github_run_id"],
            orphan["github_run_attempt"],
            github_run_url,
            orphan["trigger"],
            config_name,
            model_prefix,
            precision,
            recipe,
            concurrency,
            orphan["started_at"],
            "failed",
            f"{orphan['trigger']}/{orphan['github_run_id']}-{orphan['github_run_attempt']}/",
            now_iso,
            "workflow produced no result JSONs",
        ),
    )
    return cur.rowcount > 0


def reconcile(conn: sqlite3.Connection) -> dict[str, int]:
    """Scan recent workflows, insert failed/partial rows for missing expected pairs."""
    from dashboard.config import settings

    stats = {
        "workflows_examined": 0,
        "failed_inserted": 0,
        "partial_inserted": 0,
        "orphans_found": 0,
    }

    expected = load_expected_matrix(
        settings.nightly_configs_path, settings.nightly_runner
    )
    if not expected:
        return stats

    conn.row_factory = sqlite3.Row
    workflows = _workflow_runs_in_window(conn, settings.reconcile_window_days)
    stats["workflows_examined"] = len(workflows)
    now_iso = datetime.now(UTC).isoformat()

    for wf in workflows:
        rows = _rows_for_workflow(conn, wf["github_run_id"], wf["github_run_attempt"])
        if not rows:
            continue
        context = rows[0]
        is_cron = wf["trigger"] == "cron"

        actual_by_config: dict[str, set[int]] = {}
        for r in rows:
            actual_by_config.setdefault(r["config_name"], set()).add(r["concurrency"])

        for cfg_name, expected_concs in expected.items():
            actual_concs = actual_by_config.get(cfg_name, set())
            missing = [c for c in expected_concs if c not in actual_concs]
            if not missing:
                continue

            # No actual data for this config = whole-job failure ('failed').
            # Some data present but missing some concurrencies = 'partial'.
            # For manual runs, skip whole-config gaps — may have been an
            # intentional filtered matrix via the workflow's `configs` input.
            if not actual_concs:
                if not is_cron:
                    continue
                status = "failed"
            else:
                status = "partial"

            for conc in missing:
                if _insert_placeholder(
                    conn,
                    wf["github_run_id"],
                    wf["github_run_attempt"],
                    cfg_name,
                    conc,
                    status,
                    context,
                    now_iso,
                ):
                    stats[f"{status}_inserted"] += 1

        conn.execute(
            """
            INSERT INTO reconciliation_state (workflow_run_id, reconciled_at)
            VALUES (?, ?)
            ON CONFLICT(workflow_run_id) DO UPDATE SET reconciled_at = excluded.reconciled_at
            """,
            (f"{wf['github_run_id']}-{wf['github_run_attempt']}", now_iso),
        )

    # Orphan workflows: S3 prefixes with no result JSONs. The ingester never
    # sees these, so they'd be invisible to the dashboard without this pass.
    try:
        s3 = _make_s3_client()
        orphans = _discover_orphan_workflows(s3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("orphan discovery failed: %s", exc)
        orphans = []

    for orphan in orphans:
        existing = conn.execute(
            """
            SELECT 1 FROM runs
            WHERE github_run_id = ? AND github_run_attempt = ?
            LIMIT 1
            """,
            (orphan["github_run_id"], orphan["github_run_attempt"]),
        ).fetchone()
        if existing:
            # Either an earlier orphan pass already populated this, or the
            # ingester has since picked up a JSON (unlikely for orphans).
            continue
        stats["orphans_found"] += 1
        for cfg_name, concs in expected.items():
            for conc in concs:
                if _insert_orphan_placeholder(conn, orphan, cfg_name, conc, now_iso):
                    stats["failed_inserted"] += 1
        conn.execute(
            """
            INSERT INTO reconciliation_state (workflow_run_id, reconciled_at)
            VALUES (?, ?)
            ON CONFLICT(workflow_run_id) DO UPDATE SET reconciled_at = excluded.reconciled_at
            """,
            (
                f"{orphan['github_run_id']}-{orphan['github_run_attempt']}",
                now_iso,
            ),
        )

    conn.commit()
    if stats["failed_inserted"] or stats["partial_inserted"] or stats["orphans_found"]:
        logger.info("reconcile: %s", stats)
    return stats
