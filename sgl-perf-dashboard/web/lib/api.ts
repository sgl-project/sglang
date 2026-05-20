/**
 * Thin typed API client. Same-origin calls; Next.js rewrites /api/* to the
 * backend container. SSR-friendly: works from both server and client components.
 */

// Server components (SSR) run in Node, where fetch requires absolute URLs.
// Client components run in the browser, where relative URLs resolve against
// the current origin. Next.js rewrites `/api/*` to the backend container.
const BASE =
  typeof window === "undefined"
    ? `${process.env.API_INTERNAL_URL ?? "http://dashboard-api:8000"}/api`
    : process.env.NEXT_PUBLIC_API_URL ?? "/api";

export interface Metric {
  name: string;
  value: number;
  unit: string | null;
}

export interface RunSummary {
  id: number;
  github_run_id: string;
  github_run_attempt: number;
  github_run_url: string;
  commit_sha: string | null;
  commit_short_sha: string | null;
  commit_author: string | null;
  pr_number: number | null;
  pr_title: string | null;
  trigger: string;
  config_name: string;
  model_prefix: string | null;
  precision: string | null;
  seq_len: string | null;
  concurrency: number;
  started_at: string;
  status: string;
  failure_reason: string | null;
  gh_job_url: string | null;
}

export interface RunDetail extends RunSummary {
  commit_message: string | null;
  commit_date: string | null;
  isl: number | null;
  osl: number | null;
  recipe: string | null;
  num_gpus: number | null;
  prefill_gpus: number | null;
  decode_gpus: number | null;
  s3_log_prefix: string;
  slurm_job_id: string | null;
  ingested_at: string;
  metrics: Metric[];
}

export interface ConfigSummary {
  config_name: string;
  latest_run_id: number | null;
  latest_started_at: string | null;
  latest_status: string | null;
  concurrency_levels: number[];
}

export interface TrendPoint {
  run_id: number;
  github_run_id: string;
  commit_short_sha: string | null;
  commit_author: string | null;
  started_at: string;
  value: number;
}

export interface RunSummaryAI {
  body: string;
  model: string | null;
  tokens_used: number | null;
  generated_at: string;
}

export interface SparklineSeries {
  concurrency: number;
  points: TrendPoint[];
}

export interface ConfigSparkline {
  config_name: string;
  metric: string;
  series: SparklineSeries[];
}

export interface LatestNightlyConfigResult {
  config_name: string;
  expected_concurrencies: number[];
  passed_concurrencies: number[];
  failed_concurrencies: number[];
  partial_concurrencies: number[];
  representative_run_id: number;
  headline_metric: string | null;
  headline_value: number | null;
  headline_unit: string | null;
  headline_delta_pct_7d: number | null;
}

export interface LatestNightlySummary {
  github_run_id: string;
  github_run_attempt: number;
  github_run_url: string;
  started_at: string;
  commit_sha: string | null;
  commit_short_sha: string | null;
  commit_author: string | null;
  commit_message: string | null;
  pr_number: number | null;
  configs: LatestNightlyConfigResult[];
}

export interface HealthStatus {
  status: string;
  runs: number;
  runs_passed: number;
  runs_failed: number;
  runs_partial: number;
  metrics: number;
  last_ingest_at: string | null;
  last_scheduler_run_at: string | null;
  github_enrichment: boolean;
}

export interface RegressionSummary {
  id: number;
  run_id: number;
  metric_name: string;
  severity: "critical" | "major" | "minor";
  delta_percent: number | null;
  z_score: number | null;
  baseline_window_days: number | null;
  detected_at: string;
  resolved_at: string | null;
  config_name: string;
  concurrency: number;
  commit_short_sha: string | null;
  commit_author: string | null;
  started_at: string;
  github_run_url: string;
}

export interface RegressionDetail extends RegressionSummary {
  commit_message: string | null;
  pr_number: number | null;
  pr_title: string | null;
  metric_current_value: number;
  metric_baseline_median: number | null;
  last_passing_run_id: number | null;
  last_passing_commit_sha: string | null;
  last_passing_commit_short_sha: string | null;
  last_passing_commit_author: string | null;
  last_passing_started_at: string | null;
}

export interface RunMetricDelta {
  name: string;
  unit: string | null;
  a_value: number | null;
  b_value: number | null;
  delta_percent: number | null;
  is_regression: boolean | null;
}

export interface CompareResult {
  a: RunSummary;
  b: RunSummary;
  metric_deltas: RunMetricDelta[];
}

export interface CommitRunsResult {
  sha: string;
  short_sha: string;
  commit_message: string | null;
  commit_author: string | null;
  pr_number: number | null;
  pr_title: string | null;
  runs: RunSummary[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}${path}`;
  const resp = await fetch(url, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText} — ${url}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthStatus>("/health"),
  listRegressions: (status: "active" | "resolved" | "all" = "active") =>
    request<RegressionSummary[]>(`/regressions?status=${status}`),
  getRegression: (id: number) => request<RegressionDetail>(`/regressions/${id}`),
  compare: (a: number, b: number) =>
    request<CompareResult>(`/compare?a=${a}&b=${b}`),
  getCommit: (sha: string) => request<CommitRunsResult>(`/commits/${sha}`),
  listRuns: (params?: {
    limit?: number;
    offset?: number;
    config?: string;
    trigger?: string;
    status?: string;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params ?? {}).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    });
    const q = qs.toString();
    return request<RunSummary[]>(`/runs${q ? `?${q}` : ""}`);
  },
  getRun: (id: number) => request<RunDetail>(`/runs/${id}`),
  getRunSummary: (id: number) => request<RunSummaryAI>(`/runs/${id}/summary`),
  listConfigs: () => request<ConfigSummary[]>("/configs"),
  configSparkline: (config: string, params?: { metric?: string; window_days?: number }) => {
    const qs = new URLSearchParams();
    if (params?.metric) qs.set("metric", params.metric);
    if (params?.window_days) qs.set("window_days", String(params.window_days));
    const q = qs.toString();
    return request<ConfigSparkline>(`/configs/${config}/sparkline${q ? `?${q}` : ""}`);
  },
  latestNightly: () => request<LatestNightlySummary | null>("/latest-nightly"),
  configTrend: (
    config: string,
    params: { metric: string; concurrency: number; window_days?: number },
  ) => {
    const qs = new URLSearchParams({
      metric: params.metric,
      concurrency: String(params.concurrency),
    });
    if (params.window_days) qs.set("window_days", String(params.window_days));
    return request<TrendPoint[]>(`/configs/${config}/trend?${qs.toString()}`);
  },
};
