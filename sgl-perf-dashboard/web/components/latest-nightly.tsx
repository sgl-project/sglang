import Link from "next/link";
import type { LatestNightlySummary } from "@/lib/api";
import { formatNumber, formatRelative, compactUnit } from "@/lib/format";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function LatestNightlyCard({ summary }: { summary: LatestNightlySummary }) {
  return (
    <Card className="overflow-hidden">
      <CardContent className="space-y-4 p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border/40 pb-3">
          <div className="space-y-0.5">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Latest nightly
            </h2>
            <p className="text-[12px] text-muted-foreground">
              {formatRelative(summary.started_at)}
              <span className="mx-1.5 text-muted-foreground/60">·</span>
              <a
                className="text-muted-foreground transition hover:text-foreground"
                href={summary.github_run_url}
                target="_blank"
                rel="noreferrer"
              >
                workflow #{summary.github_run_id}
              </a>
            </p>
          </div>
          {summary.commit_short_sha && (
            <p className="font-mono text-[11px] text-muted-foreground">
              <a
                className="text-primary transition hover:underline"
                href={`https://github.com/sgl-project/sglang/commit/${summary.commit_sha}`}
                target="_blank"
                rel="noreferrer"
              >
                {summary.commit_short_sha}
              </a>
              {summary.commit_author && (
                <span className="ml-2 font-sans text-muted-foreground">
                  by {summary.commit_author}
                </span>
              )}
              {summary.commit_message && (
                <span className="ml-2 hidden font-sans text-muted-foreground sm:inline">
                  — {summary.commit_message.split("\n")[0]}
                </span>
              )}
            </p>
          )}
        </div>

        <div className="space-y-2">
          {summary.configs.map((c) => (
            <ConfigRow key={c.config_name} cfg={c} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function ConfigRow({ cfg }: { cfg: import("@/lib/api").LatestNightlyConfigResult }) {
  const expected = cfg.expected_concurrencies.length || (cfg.passed_concurrencies.length + cfg.failed_concurrencies.length + cfg.partial_concurrencies.length);
  const passed = cfg.passed_concurrencies.length;
  const allPassed = passed === expected && expected > 0;
  const noneRan = passed === 0;
  const someFailed = cfg.failed_concurrencies.length > 0 || cfg.partial_concurrencies.length > 0;
  const variant = allPassed ? "success" : noneRan ? "destructive" : "warning";

  return (
    <Link
      href={`/runs/${cfg.representative_run_id}`}
      className="group block rounded-md border border-transparent px-2 py-1.5 transition hover:border-border/60 hover:bg-muted/30"
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Badge variant={variant} className="font-mono">
          {passed}/{expected || "?"} passed
        </Badge>
        <span className="font-mono text-[13px] text-foreground/90">{cfg.config_name}</span>
        {cfg.headline_value !== null && (
          <>
            <span className="ml-auto font-mono tabular-numbers text-[13px] text-foreground/80">
              {formatNumber(cfg.headline_value)}
              {compactUnit(cfg.headline_unit) && (
                <span className="ml-1 text-[11px] text-muted-foreground">
                  {compactUnit(cfg.headline_unit)}
                </span>
              )}
            </span>
            {cfg.headline_delta_pct_7d !== null && (
              <DeltaBadge delta={cfg.headline_delta_pct_7d} metric={cfg.headline_metric} />
            )}
          </>
        )}
        {someFailed && cfg.headline_value === null && (
          <span className="ml-auto text-[12px] text-destructive">workflow failed</span>
        )}
      </div>
    </Link>
  );
}

function DeltaBadge({ delta, metric }: { delta: number; metric: string | null }) {
  const higherIsBetter = (metric ?? "").includes("throughput");
  const good = higherIsBetter ? delta >= 0 : delta <= 0;
  const sign = delta > 0 ? "+" : "";
  return (
    <span
      className={`font-mono tabular-numbers text-[11px] ${good ? "text-success" : "text-destructive"}`}
      title={metric ? `vs 7-day median for ${metric}` : "vs 7-day median"}
    >
      {sign}
      {delta.toFixed(1)}%
    </span>
  );
}
