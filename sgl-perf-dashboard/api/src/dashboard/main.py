"""FastAPI entry point. Exposes /api/* routes and runs the APScheduler-driven
ingester alongside the HTTP server in the same process.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from dashboard import ingester
from dashboard.config import settings
from dashboard.db import connect, init_db
from dashboard.models import (
    CommitRunsResult,
    CompareResult,
    ConfigSparkline,
    ConfigSummary,
    HealthStatus,
    LatestNightlyConfigResult,
    LatestNightlySummary,
    Metric,
    RegressionDetail,
    RegressionSummary,
    RunDetail,
    RunMetricDelta,
    RunSummary,
    RunSummaryAI,
    SparklineSeries,
    TrendPoint,
)
from dashboard.reconciler import load_expected_matrix
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_scheduler: BackgroundScheduler | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db(settings.db_path)
    logger.info("db initialized at %s", settings.db_path)

    global _scheduler
    _scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
    _scheduler.add_job(
        _safe_ingest,
        "interval",
        seconds=settings.ingester_interval_seconds,
        id="ingest",
    )
    _scheduler.start()
    logger.info(
        "scheduler started; ingest cadence=%ds, github enrichment=%s",
        settings.ingester_interval_seconds,
        settings.github_enrichment_enabled,
    )

    # Kick off an immediate ingest in the background (non-blocking).
    _scheduler.add_job(_safe_ingest, id="ingest-bootstrap")

    try:
        yield
    finally:
        if _scheduler:
            _scheduler.shutdown(wait=False)


def _safe_ingest() -> None:
    try:
        ingester.run_once()
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingester run failed: %s", exc)


app = FastAPI(
    title="sglang Perf Dashboard",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten when auth lands
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


def _maybe(row: sqlite3.Row, key: str) -> Any:
    """Defensive read — returns None if a column was added in a later migration
    but the caller still holds a pre-migration row.
    """
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _row_to_summary(row: sqlite3.Row) -> RunSummary:
    return RunSummary(
        id=row["id"],
        github_run_id=row["github_run_id"],
        github_run_attempt=row["github_run_attempt"],
        github_run_url=row["github_run_url"],
        commit_sha=row["commit_sha"],
        commit_short_sha=row["commit_short_sha"],
        commit_author=row["commit_author"],
        pr_number=row["pr_number"],
        pr_title=row["pr_title"],
        trigger=row["trigger"],
        config_name=row["config_name"],
        model_prefix=row["model_prefix"],
        precision=row["precision"],
        seq_len=row["seq_len"],
        concurrency=row["concurrency"],
        started_at=row["started_at"],
        status=row["status"],
        failure_reason=_maybe(row, "failure_reason"),
        gh_job_url=_maybe(row, "gh_job_url"),
    )


def _row_to_detail(row: sqlite3.Row, metrics: list[Metric]) -> RunDetail:
    return RunDetail(
        id=row["id"],
        github_run_id=row["github_run_id"],
        github_run_attempt=row["github_run_attempt"],
        github_run_url=row["github_run_url"],
        commit_sha=row["commit_sha"],
        commit_short_sha=row["commit_short_sha"],
        commit_author=row["commit_author"],
        commit_message=row["commit_message"],
        commit_date=row["commit_date"],
        pr_number=row["pr_number"],
        pr_title=row["pr_title"],
        trigger=row["trigger"],
        config_name=row["config_name"],
        model_prefix=row["model_prefix"],
        precision=row["precision"],
        seq_len=row["seq_len"],
        isl=row["isl"],
        osl=row["osl"],
        recipe=row["recipe"],
        concurrency=row["concurrency"],
        num_gpus=row["num_gpus"],
        prefill_gpus=row["prefill_gpus"],
        decode_gpus=row["decode_gpus"],
        started_at=row["started_at"],
        status=row["status"],
        failure_reason=_maybe(row, "failure_reason"),
        gh_job_url=_maybe(row, "gh_job_url"),
        s3_log_prefix=row["s3_log_prefix"],
        slurm_job_id=row["slurm_job_id"],
        ingested_at=row["ingested_at"],
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthStatus)
def health() -> HealthStatus:
    with connect(settings.db_path) as conn:
        runs_count = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS c FROM runs GROUP BY status"
        ).fetchall()
        status_counts = {r["status"]: r["c"] for r in status_rows}
        metrics_count = conn.execute("SELECT COUNT(*) AS c FROM metrics").fetchone()[
            "c"
        ]
        cursor_row = conn.execute(
            "SELECT updated_at FROM ingester_state WHERE key = 's3_cursor'"
        ).fetchone()
        heartbeat_row = conn.execute(
            "SELECT updated_at FROM ingester_state WHERE key = 'last_run'"
        ).fetchone()
    return HealthStatus(
        status="ok",
        runs=runs_count,
        runs_passed=status_counts.get("passed", 0),
        runs_failed=status_counts.get("failed", 0),
        runs_partial=status_counts.get("partial", 0),
        metrics=metrics_count,
        last_ingest_at=cursor_row["updated_at"] if cursor_row else None,
        last_scheduler_run_at=heartbeat_row["updated_at"] if heartbeat_row else None,
        github_enrichment=settings.github_enrichment_enabled,
    )


@app.get("/api/runs", response_model=list[RunSummary])
def list_runs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    config: str | None = None,
    trigger: str | None = None,
    status: str | None = None,
) -> list[RunSummary]:
    clauses: list[str] = []
    params: list[Any] = []
    if config:
        clauses.append("config_name = ?")
        params.append(config)
    if trigger:
        clauses.append("trigger = ?")
        params.append(trigger)
    if status and status != "all":
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT * FROM runs
        {where}
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    with connect(settings.db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_summary(r) for r in rows]


@app.get("/api/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: int) -> RunDetail:
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        metric_rows = conn.execute(
            "SELECT name, value, unit FROM metrics WHERE run_id = ? ORDER BY name",
            (run_id,),
        ).fetchall()
    metrics = [
        Metric(name=m["name"], value=m["value"], unit=m["unit"]) for m in metric_rows
    ]
    return _row_to_detail(row, metrics)


@app.get("/api/runs/{run_id}/summary", response_model=RunSummaryAI)
def get_run_summary(run_id: int) -> RunSummaryAI:
    """Return the cached AI log analysis for a run, if one exists.

    The analysis is generated by `analyze_logs_with_modal.py` during the
    workflow, uploaded to S3, and attached to all matching run rows by the
    ingester's `_sync_ai_analyses` pass.
    """
    with connect(settings.db_path) as conn:
        row = conn.execute(
            """
            SELECT body, model, tokens_used, generated_at
            FROM ai_summaries
            WHERE run_id = ? AND summary_type = 'log_analysis'
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no summary for this run")
    return RunSummaryAI(
        body=row["body"],
        model=row["model"],
        tokens_used=row["tokens_used"],
        generated_at=row["generated_at"],
    )


@app.get("/api/runs/{run_id}/previous")
def get_previous_run(run_id: int) -> dict[str, int | None]:
    """Find the previous run at the same (config_name, concurrency) —
    used by the UI to build a 'compare to previous' link.
    """
    with connect(settings.db_path) as conn:
        current = conn.execute(
            "SELECT config_name, concurrency, started_at FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail="run not found")
        prev = conn.execute(
            """
            SELECT id FROM runs
            WHERE config_name = ?
              AND concurrency = ?
              AND started_at < ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (current["config_name"], current["concurrency"], current["started_at"]),
        ).fetchone()
    return {"previous_run_id": prev["id"] if prev else None}


@app.get("/api/configs", response_model=list[ConfigSummary])
def list_configs() -> list[ConfigSummary]:
    """Per-config summary.

    `latest_status` aggregates the most recent *workflow's* rows for a config
    (not a single row): any `failed` → failed, else any `partial` → partial,
    else `passed`. Makes the card colour a reliable "was the last nightly
    clean?" signal.

    `latest_run_id` points at the most problematic row in that workflow
    (failed > partial > passed) so clicking the card lands on whatever needs
    attention.
    """
    out: list[ConfigSummary] = []
    with connect(settings.db_path) as conn:
        config_rows = conn.execute("""
            SELECT
                config_name,
                GROUP_CONCAT(DISTINCT concurrency) AS concurrency_csv
            FROM runs
            GROUP BY config_name
            ORDER BY config_name
        """).fetchall()

        for cr in config_rows:
            config_name = cr["config_name"]

            latest_wf = conn.execute(
                """
                SELECT github_run_id, github_run_attempt,
                       MAX(started_at) AS started_at
                FROM runs
                WHERE config_name = ?
                GROUP BY github_run_id, github_run_attempt
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (config_name,),
            ).fetchone()
            if latest_wf is None:
                continue

            agg = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END) AS n_failed,
                    SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END) AS n_partial,
                    MAX(started_at) AS latest_started_at
                FROM runs
                WHERE config_name = ?
                  AND github_run_id = ?
                  AND github_run_attempt = ?
                """,
                (
                    config_name,
                    latest_wf["github_run_id"],
                    latest_wf["github_run_attempt"],
                ),
            ).fetchone()

            if agg["n_failed"]:
                status = "failed"
            elif agg["n_partial"]:
                status = "partial"
            else:
                status = "passed"

            representative = conn.execute(
                """
                SELECT id FROM runs
                WHERE config_name = ?
                  AND github_run_id = ?
                  AND github_run_attempt = ?
                ORDER BY
                  CASE status WHEN 'failed' THEN 0 WHEN 'partial' THEN 1 ELSE 2 END,
                  concurrency
                LIMIT 1
                """,
                (
                    config_name,
                    latest_wf["github_run_id"],
                    latest_wf["github_run_attempt"],
                ),
            ).fetchone()

            concs = sorted(
                int(c) for c in (cr["concurrency_csv"] or "").split(",") if c
            )
            out.append(
                ConfigSummary(
                    config_name=config_name,
                    latest_run_id=representative["id"] if representative else None,
                    latest_started_at=agg["latest_started_at"],
                    latest_status=status,
                    concurrency_levels=concs,
                )
            )

    return out


@app.get("/api/configs/{config_name}/trend", response_model=list[TrendPoint])
def config_trend(
    config_name: str,
    metric: str = Query(
        ..., description="Metric name (verbatim, e.g. 'total_token_throughput')"
    ),
    concurrency: int = Query(..., ge=1),
    window_days: int = Query(default=30, ge=1, le=365),
) -> list[TrendPoint]:
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.id AS run_id, r.github_run_id, r.commit_short_sha, r.commit_author,
                   r.started_at, m.value
            FROM runs r
            JOIN metrics m ON m.run_id = r.id
            WHERE r.config_name = ?
              AND r.concurrency = ?
              AND m.name = ?
              AND r.started_at > datetime('now', ?)
              AND r.status = 'passed'
            ORDER BY r.started_at
            """,
            (config_name, concurrency, metric, f"-{window_days} day"),
        ).fetchall()
    return [
        TrendPoint(
            run_id=r["run_id"],
            github_run_id=r["github_run_id"],
            commit_short_sha=r["commit_short_sha"],
            commit_author=r["commit_author"],
            started_at=r["started_at"],
            value=r["value"],
        )
        for r in rows
    ]


@app.get("/api/configs/{config_name}/sparkline", response_model=ConfigSparkline)
def config_sparkline(
    config_name: str,
    metric: str = Query(default="total_token_throughput"),
    window_days: int = Query(default=14, ge=1, le=90),
) -> ConfigSparkline:
    """Compact multi-concurrency trend for home-page sparklines.

    Returns one series per concurrency observed at this config over the
    window. Only `passed` runs contribute points — failed/partial show as
    gaps in the timeline.
    """
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.id AS run_id, r.github_run_id, r.commit_short_sha,
                   r.commit_author, r.started_at, r.concurrency, m.value
            FROM runs r
            JOIN metrics m ON m.run_id = r.id
            WHERE r.config_name = ?
              AND m.name = ?
              AND r.started_at > datetime('now', ?)
              AND r.status = 'passed'
            ORDER BY r.concurrency, r.started_at
            """,
            (config_name, metric, f"-{window_days} day"),
        ).fetchall()

    by_conc: dict[int, list[TrendPoint]] = {}
    for r in rows:
        by_conc.setdefault(r["concurrency"], []).append(
            TrendPoint(
                run_id=r["run_id"],
                github_run_id=r["github_run_id"],
                commit_short_sha=r["commit_short_sha"],
                commit_author=r["commit_author"],
                started_at=r["started_at"],
                value=r["value"],
            )
        )
    return ConfigSparkline(
        config_name=config_name,
        metric=metric,
        series=[
            SparklineSeries(concurrency=c, points=pts)
            for c, pts in sorted(by_conc.items())
        ],
    )


@app.get("/api/latest-nightly", response_model=LatestNightlySummary | None)
def latest_nightly() -> LatestNightlySummary | None:
    """The most recent cron workflow, summarized per config.

    Aggregates each config's outcome (passed/failed/partial concurrencies),
    a representative run_id to link into, and a headline metric for the
    highest passed concurrency with a 7-day delta.
    """
    with connect(settings.db_path) as conn:
        latest = conn.execute("""
            SELECT github_run_id, github_run_attempt, MAX(started_at) AS started_at
            FROM runs
            WHERE trigger = 'cron'
            GROUP BY github_run_id, github_run_attempt
            ORDER BY started_at DESC
            LIMIT 1
            """).fetchone()
        if latest is None:
            return None

        run_id, attempt = latest["github_run_id"], latest["github_run_attempt"]

        meta = conn.execute(
            """
            SELECT github_run_url, commit_sha, commit_short_sha, commit_author,
                   commit_message, pr_number
            FROM runs
            WHERE github_run_id = ? AND github_run_attempt = ?
            ORDER BY started_at LIMIT 1
            """,
            (run_id, attempt),
        ).fetchone()

        try:
            expected = load_expected_matrix(
                settings.nightly_configs_path, settings.nightly_runner
            )
        except Exception:
            expected = {}

        rows = conn.execute(
            """
            SELECT id, config_name, concurrency, status, started_at
            FROM runs
            WHERE github_run_id = ? AND github_run_attempt = ?
            ORDER BY config_name, concurrency
            """,
            (run_id, attempt),
        ).fetchall()

        by_config: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            by_config.setdefault(r["config_name"], []).append(r)

        configs: list[LatestNightlyConfigResult] = []
        for cfg_name, cfg_rows in by_config.items():
            passed = sorted(
                r["concurrency"] for r in cfg_rows if r["status"] == "passed"
            )
            failed = sorted(
                r["concurrency"] for r in cfg_rows if r["status"] == "failed"
            )
            partial = sorted(
                r["concurrency"] for r in cfg_rows if r["status"] == "partial"
            )

            # Representative row: prefer failed > partial > passed (drill in to
            # what needs attention).
            representative = sorted(
                cfg_rows,
                key=lambda r: (
                    {"failed": 0, "partial": 1}.get(r["status"], 2),
                    r["concurrency"],
                ),
            )[0]

            # Headline metric: at the highest passed concurrency, if any.
            headline_metric = headline_value = headline_unit = headline_delta = None
            if passed:
                top_conc = passed[-1]
                row_with_metric = conn.execute(
                    """
                    SELECT m.value, m.unit
                    FROM runs r JOIN metrics m ON m.run_id = r.id
                    WHERE r.github_run_id = ? AND r.github_run_attempt = ?
                      AND r.config_name = ? AND r.concurrency = ?
                      AND m.name = 'total_token_throughput'
                    LIMIT 1
                    """,
                    (run_id, attempt, cfg_name, top_conc),
                ).fetchone()
                if row_with_metric:
                    headline_metric = "total_token_throughput"
                    headline_value = row_with_metric["value"]
                    headline_unit = row_with_metric["unit"]
                    baseline = conn.execute(
                        """
                        WITH window_vals AS (
                            SELECT m.value
                            FROM runs r JOIN metrics m ON m.run_id = r.id
                            WHERE r.config_name = ?
                              AND r.concurrency = ?
                              AND m.name = 'total_token_throughput'
                              AND r.status = 'passed'
                              AND r.started_at > datetime('now', '-7 day')
                              AND r.id != (
                                  SELECT id FROM runs
                                  WHERE github_run_id = ? AND github_run_attempt = ?
                                    AND config_name = ? AND concurrency = ?
                              )
                        )
                        SELECT AVG(value) AS median FROM (
                            SELECT value FROM window_vals
                            ORDER BY value
                            LIMIT 2 - (SELECT COUNT(*) FROM window_vals) % 2
                            OFFSET (SELECT (COUNT(*) - 1) / 2 FROM window_vals)
                        )
                        """,
                        (
                            cfg_name,
                            top_conc,
                            run_id,
                            attempt,
                            cfg_name,
                            top_conc,
                        ),
                    ).fetchone()
                    if baseline and baseline["median"]:
                        headline_delta = (
                            (headline_value - baseline["median"])
                            / baseline["median"]
                            * 100
                        )

            configs.append(
                LatestNightlyConfigResult(
                    config_name=cfg_name,
                    expected_concurrencies=expected.get(cfg_name, []),
                    passed_concurrencies=passed,
                    failed_concurrencies=failed,
                    partial_concurrencies=partial,
                    representative_run_id=representative["id"],
                    headline_metric=headline_metric,
                    headline_value=headline_value,
                    headline_unit=headline_unit,
                    headline_delta_pct_7d=headline_delta,
                )
            )

        configs.sort(key=lambda c: c.config_name)

    return LatestNightlySummary(
        github_run_id=run_id,
        github_run_attempt=attempt,
        github_run_url=meta["github_run_url"] if meta else "",
        started_at=latest["started_at"],
        commit_sha=meta["commit_sha"] if meta else None,
        commit_short_sha=meta["commit_short_sha"] if meta else None,
        commit_author=meta["commit_author"] if meta else None,
        commit_message=meta["commit_message"] if meta else None,
        pr_number=meta["pr_number"] if meta else None,
        configs=configs,
    )


@app.get("/api/regressions", response_model=list[RegressionSummary])
def list_regressions(
    status: str = Query(default="active", regex="^(active|resolved|all)$"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[RegressionSummary]:
    clauses: list[str] = []
    if status == "active":
        clauses.append("reg.resolved_at IS NULL")
    elif status == "resolved":
        clauses.append("reg.resolved_at IS NOT NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT reg.id, reg.run_id, reg.metric_name, reg.severity,
               reg.delta_percent, reg.z_score, reg.baseline_window_days,
               reg.detected_at, reg.resolved_at,
               r.config_name, r.concurrency, r.commit_short_sha,
               r.commit_author, r.started_at, r.github_run_url
        FROM regressions reg
        JOIN runs r ON r.id = reg.run_id
        {where}
        ORDER BY
          CASE reg.severity WHEN 'critical' THEN 0 WHEN 'major' THEN 1 ELSE 2 END,
          reg.detected_at DESC
        LIMIT ?
    """
    with connect(settings.db_path) as conn:
        rows = conn.execute(sql, [limit]).fetchall()
    return [
        RegressionSummary(
            id=r["id"],
            run_id=r["run_id"],
            metric_name=r["metric_name"],
            severity=r["severity"],
            delta_percent=r["delta_percent"],
            z_score=r["z_score"],
            baseline_window_days=r["baseline_window_days"],
            detected_at=r["detected_at"],
            resolved_at=r["resolved_at"],
            config_name=r["config_name"],
            concurrency=r["concurrency"],
            commit_short_sha=r["commit_short_sha"],
            commit_author=r["commit_author"],
            started_at=r["started_at"],
            github_run_url=r["github_run_url"],
        )
        for r in rows
    ]


@app.get("/api/regressions/{reg_id}", response_model=RegressionDetail)
def get_regression(reg_id: int) -> RegressionDetail:
    with connect(settings.db_path) as conn:
        row = conn.execute(
            """
            SELECT reg.id, reg.run_id, reg.metric_name, reg.severity,
                   reg.delta_percent, reg.z_score, reg.baseline_window_days,
                   reg.detected_at, reg.resolved_at,
                   r.config_name, r.concurrency, r.commit_sha, r.commit_short_sha,
                   r.commit_message, r.commit_author, r.started_at, r.github_run_url,
                   r.pr_number, r.pr_title
            FROM regressions reg
            JOIN runs r ON r.id = reg.run_id
            WHERE reg.id = ?
            """,
            (reg_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="regression not found")

        metric_row = conn.execute(
            "SELECT value FROM metrics WHERE run_id = ? AND name = ?",
            (row["run_id"], row["metric_name"]),
        ).fetchone()

        baseline_median = None
        if metric_row and row["delta_percent"] is not None and metric_row["value"] != 0:
            # value = median * (1 + delta_percent/100) → median = value / (1 + dp/100)
            factor = 1 + (row["delta_percent"] / 100)
            if factor != 0:
                baseline_median = metric_row["value"] / factor

        # Last passing run at same (config, concurrency) before this regression
        last_good = conn.execute(
            """
            SELECT r.id, r.commit_sha, r.commit_short_sha, r.commit_author,
                   r.started_at
            FROM runs r
            LEFT JOIN regressions reg
              ON reg.run_id = r.id AND reg.metric_name = ?
            WHERE r.config_name = ?
              AND r.concurrency = ?
              AND r.started_at < ?
              AND r.status = 'passed'
              AND reg.id IS NULL
            ORDER BY r.started_at DESC
            LIMIT 1
            """,
            (
                row["metric_name"],
                row["config_name"],
                row["concurrency"],
                row["started_at"],
            ),
        ).fetchone()

    return RegressionDetail(
        id=row["id"],
        run_id=row["run_id"],
        metric_name=row["metric_name"],
        severity=row["severity"],
        delta_percent=row["delta_percent"],
        z_score=row["z_score"],
        baseline_window_days=row["baseline_window_days"],
        detected_at=row["detected_at"],
        resolved_at=row["resolved_at"],
        config_name=row["config_name"],
        concurrency=row["concurrency"],
        commit_short_sha=row["commit_short_sha"],
        commit_author=row["commit_author"],
        commit_message=row["commit_message"],
        pr_number=row["pr_number"],
        pr_title=row["pr_title"],
        started_at=row["started_at"],
        github_run_url=row["github_run_url"],
        metric_current_value=metric_row["value"] if metric_row else 0.0,
        metric_baseline_median=baseline_median,
        last_passing_run_id=last_good["id"] if last_good else None,
        last_passing_commit_sha=last_good["commit_sha"] if last_good else None,
        last_passing_commit_short_sha=(
            last_good["commit_short_sha"] if last_good else None
        ),
        last_passing_commit_author=last_good["commit_author"] if last_good else None,
        last_passing_started_at=last_good["started_at"] if last_good else None,
    )


@app.get("/api/compare", response_model=CompareResult)
def compare_runs(a: int = Query(..., ge=1), b: int = Query(..., ge=1)) -> CompareResult:
    """Side-by-side diff of two runs. Metrics absent on one side get null values."""
    with connect(settings.db_path) as conn:
        row_a = conn.execute("SELECT * FROM runs WHERE id = ?", (a,)).fetchone()
        row_b = conn.execute("SELECT * FROM runs WHERE id = ?", (b,)).fetchone()
        if row_a is None or row_b is None:
            raise HTTPException(status_code=404, detail="run(s) not found")
        metrics_a = {
            m["name"]: {"value": m["value"], "unit": m["unit"]}
            for m in conn.execute(
                "SELECT name, value, unit FROM metrics WHERE run_id = ?", (a,)
            ).fetchall()
        }
        metrics_b = {
            m["name"]: {"value": m["value"], "unit": m["unit"]}
            for m in conn.execute(
                "SELECT name, value, unit FROM metrics WHERE run_id = ?", (b,)
            ).fetchall()
        }

    all_names = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    deltas: list[RunMetricDelta] = []
    for name in all_names:
        va = metrics_a.get(name)
        vb = metrics_b.get(name)
        a_val = va["value"] if va else None
        b_val = vb["value"] if vb else None
        unit = (va or vb or {}).get("unit")
        delta_pct: float | None = None
        is_regr: bool | None = None
        if a_val is not None and b_val is not None and a_val != 0:
            delta_pct = (b_val - a_val) / a_val * 100
            from dashboard.anomaly import _is_regression

            is_regr = _is_regression(name, delta_pct)
        deltas.append(
            RunMetricDelta(
                name=name,
                unit=unit,
                a_value=a_val,
                b_value=b_val,
                delta_percent=delta_pct,
                is_regression=is_regr,
            )
        )

    return CompareResult(
        a=_row_to_summary(row_a),
        b=_row_to_summary(row_b),
        metric_deltas=deltas,
    )


@app.get("/api/commits/{sha}", response_model=CommitRunsResult)
def get_commit_runs(sha: str) -> CommitRunsResult:
    """All runs testing this commit (match by full SHA OR short SHA prefix)."""
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM runs
            WHERE commit_sha = ? OR commit_short_sha = ? OR commit_sha LIKE ?
            ORDER BY started_at DESC
            """,
            (sha, sha[:7], f"{sha}%"),
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="no runs found for commit")

    first = rows[0]
    return CommitRunsResult(
        sha=first["commit_sha"] or sha,
        short_sha=first["commit_short_sha"] or sha[:7],
        commit_message=first["commit_message"],
        commit_author=first["commit_author"],
        pr_number=first["pr_number"],
        pr_title=first["pr_title"],
        runs=[_row_to_summary(r) for r in rows],
    )


@app.post("/api/admin/ingest")
def trigger_ingest() -> dict[str, Any]:
    """Manual trigger — useful during M1 testing. Replaces scheduled waiting."""
    stats = ingester.run_once()
    return {"ok": True, "stats": stats}
