"""Pydantic response schemas. Kept separate from DB layer so HTTP shape can
evolve independently of storage shape.
"""

from __future__ import annotations

from pydantic import BaseModel


class Metric(BaseModel):
    name: str
    value: float
    unit: str | None = None


class RunSummary(BaseModel):
    """Compact shape for lists (recent runs, config pages)."""

    id: int
    github_run_id: str
    github_run_attempt: int
    github_run_url: str
    commit_sha: str | None
    commit_short_sha: str | None
    commit_author: str | None
    pr_number: int | None
    pr_title: str | None
    trigger: str
    config_name: str
    model_prefix: str | None
    precision: str | None
    seq_len: str | None
    concurrency: int
    started_at: str
    status: str
    failure_reason: str | None = None
    gh_job_url: str | None = None


class RunDetail(RunSummary):
    """Full run with commit message + metrics + log prefix."""

    commit_message: str | None
    commit_date: str | None
    isl: int | None
    osl: int | None
    recipe: str | None
    num_gpus: int | None
    prefill_gpus: int | None
    decode_gpus: int | None
    s3_log_prefix: str
    slurm_job_id: str | None
    ingested_at: str
    metrics: list[Metric]


class TrendPoint(BaseModel):
    """One point on a config-trend chart."""

    run_id: int
    github_run_id: str
    commit_short_sha: str | None
    commit_author: str | None
    started_at: str
    value: float


class ConfigSummary(BaseModel):
    """For home-page status cards."""

    config_name: str
    latest_run_id: int | None
    latest_started_at: str | None
    latest_status: str | None
    concurrency_levels: list[int]


class HealthStatus(BaseModel):
    status: str
    runs: int
    runs_passed: int
    runs_failed: int
    runs_partial: int
    metrics: int
    last_ingest_at: str | None
    last_scheduler_run_at: str | None
    github_enrichment: bool


class RegressionSummary(BaseModel):
    """One flagged anomaly, denormalized with the run context."""

    id: int
    run_id: int
    metric_name: str
    severity: str
    delta_percent: float | None
    z_score: float | None
    baseline_window_days: int | None
    detected_at: str
    resolved_at: str | None
    # Denormalized from runs
    config_name: str
    concurrency: int
    commit_short_sha: str | None
    commit_author: str | None
    started_at: str
    github_run_url: str


class RegressionDetail(RegressionSummary):
    """Single regression + context for diagnosis."""

    commit_message: str | None
    pr_number: int | None
    pr_title: str | None
    metric_current_value: float
    metric_baseline_median: float | None
    # Context: the last-passing run of same (config, concurrency, metric)
    # — useful for showing "what commit was last good".
    last_passing_run_id: int | None
    last_passing_commit_sha: str | None
    last_passing_commit_short_sha: str | None
    last_passing_commit_author: str | None
    last_passing_started_at: str | None


class RunMetricDelta(BaseModel):
    """One metric shown in a side-by-side run comparison."""

    name: str
    unit: str | None
    a_value: float | None
    b_value: float | None
    delta_percent: float | None
    is_regression: bool | None  # True when b is worse than a


class CompareResult(BaseModel):
    """Compare two runs. Metrics present in both sides get a delta."""

    a: RunSummary
    b: RunSummary
    metric_deltas: list[RunMetricDelta]


class CommitRunsResult(BaseModel):
    """Lookup runs by commit SHA."""

    sha: str
    short_sha: str
    commit_message: str | None
    commit_author: str | None
    pr_number: int | None
    pr_title: str | None
    runs: list[RunSummary]


class RunSummaryAI(BaseModel):
    """Claude-generated narrative attached to a single run."""

    body: str
    model: str | None
    tokens_used: int | None
    generated_at: str


class SparklineSeries(BaseModel):
    """One concurrency's time-series, used inside ConfigSparkline."""

    concurrency: int
    points: list[TrendPoint]


class ConfigSparkline(BaseModel):
    """All concurrencies' trends for one config, used on the home page."""

    config_name: str
    metric: str
    series: list[SparklineSeries]


class LatestNightlyConfigResult(BaseModel):
    """One config's outcome within the latest cron workflow."""

    config_name: str
    expected_concurrencies: list[int]
    passed_concurrencies: list[int]
    failed_concurrencies: list[int]
    partial_concurrencies: list[int]
    representative_run_id: int
    headline_metric: str | None
    headline_value: float | None
    headline_unit: str | None
    headline_delta_pct_7d: float | None  # vs 7-day median at same (config, conc)


class LatestNightlySummary(BaseModel):
    """Summary of the most recent cron workflow run, for the home-page hero."""

    github_run_id: str
    github_run_attempt: int
    github_run_url: str
    started_at: str
    commit_sha: str | None
    commit_short_sha: str | None
    commit_author: str | None
    commit_message: str | None
    pr_number: int | None
    configs: list[LatestNightlyConfigResult]
