import Link from "next/link";
import {
  api,
  type ConfigSparkline,
  type ConfigSummary,
  type LatestNightlySummary,
} from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AutoRefresh } from "@/components/auto-refresh";
import { StatusTooltip } from "@/components/status-tooltip";
import { Sparkline } from "@/components/sparkline";
import { LatestNightlyCard } from "@/components/latest-nightly";

export const dynamic = "force-dynamic";

async function loadHomeData() {
  try {
    const [configs, health, latest] = await Promise.all([
      api.listConfigs(),
      api.health(),
      api.latestNightly().catch(() => null),
    ]);
    const sparklines = await Promise.all(
      configs.map((c) =>
        api
          .configSparkline(c.config_name, { metric: "total_token_throughput", window_days: 14 })
          .catch(() => null),
      ),
    );
    const sparklineByConfig: Record<string, ConfigSparkline | null> = {};
    configs.forEach((c, i) => (sparklineByConfig[c.config_name] = sparklines[i]));
    return { configs, health, latest, sparklineByConfig, error: null };
  } catch (err) {
    return {
      configs: [],
      health: null,
      latest: null as LatestNightlySummary | null,
      sparklineByConfig: {} as Record<string, ConfigSparkline | null>,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export default async function HomePage() {
  const { configs, health, latest, sparklineByConfig, error } = await loadHomeData();

  return (
    <div className="space-y-8 animate-fade-in-up">
      <AutoRefresh />

      {/* Header */}
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">GB200 Nightly</h1>
          <p className="text-[13px] text-muted-foreground">
            {health ? (
              <>
                <span className="tabular-numbers font-medium text-foreground">
                  {health.runs.toLocaleString()}
                </span>{" "}
                runs{" "}
                <span className="tabular-numbers text-success">
                  {health.runs_passed.toLocaleString()} passed
                </span>
                {health.runs_failed > 0 && (
                  <>
                    <span className="mx-1 text-muted-foreground/60">·</span>
                    <span className="tabular-numbers text-destructive">
                      {health.runs_failed.toLocaleString()} failed
                    </span>
                  </>
                )}
                {health.runs_partial > 0 && (
                  <>
                    <span className="mx-1 text-muted-foreground/60">·</span>
                    <span className="tabular-numbers text-warning">
                      {health.runs_partial.toLocaleString()} partial
                    </span>
                  </>
                )}
                <span className="mx-1.5 text-muted-foreground/60">·</span>
                new data {formatRelative(health.last_ingest_at)}
                <span className="mx-1.5 text-muted-foreground/60">·</span>
                <span title={`last scheduler tick: ${health.last_scheduler_run_at ?? "never"}`}>
                  sync {formatRelative(health.last_scheduler_run_at)}
                </span>
              </>
            ) : (
              "loading…"
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {health && !health.github_enrichment && (
            <Badge variant="warning">GitHub enrichment disabled</Badge>
          )}
          {health && (
            <Badge variant="outline" className="font-mono">
              status: {health.status}
            </Badge>
          )}
        </div>
      </section>

      {error && <ErrorBanner message={error} />}

      {/* Configs grid with sparklines */}
      <section className="space-y-3">
        <SectionHeader title="Configs" hint={`${configs.length} tracked`} />
        {configs.length === 0 ? (
          <EmptyState
            title="No configs yet"
            hint="Ingester hasn't seen any result JSONs — kick one off on the GB200 nightly workflow."
          />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {configs.map((c) => (
              <ConfigCard
                key={c.config_name}
                config={c}
                sparkline={sparklineByConfig[c.config_name]}
              />
            ))}
          </div>
        )}
      </section>

      {/* Latest nightly hero */}
      <section className="space-y-3">
        <SectionHeader title="Latest nightly" hint={latest ? "cron" : undefined} />
        {latest ? (
          <LatestNightlyCard summary={latest} />
        ) : (
          <EmptyState
            title="No nightly runs yet"
            hint="First scheduled run will appear here at 02:00 UTC."
          />
        )}
      </section>
    </div>
  );
}

function SectionHeader({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between">
      <div className="flex items-baseline gap-2">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          {title}
        </h2>
        {hint && <span className="text-[11px] text-muted-foreground/70">· {hint}</span>}
      </div>
      {action}
    </div>
  );
}

function ConfigCard({
  config,
  sparkline,
}: {
  config: ConfigSummary;
  sparkline: ConfigSparkline | null | undefined;
}) {
  const isPassing = config.latest_status === "passed";
  const isPartial = config.latest_status === "partial";
  const hasData = sparkline && sparkline.series.some((s) => s.points.length > 0);
  return (
    <Link href={`/configs/${config.config_name}`} className="group block">
      <Card className="h-full">
        <CardContent className="space-y-3 p-4">
          <div className="flex items-start justify-between gap-3">
            <p className="font-mono text-[12.5px] leading-tight text-foreground/90">
              {config.config_name}
            </p>
            <StatusTooltip status={config.latest_status ?? ""}>
              <Badge variant={isPassing ? "success" : isPartial ? "warning" : "destructive"}>
                <StatusDot passing={isPassing} />
                {config.latest_status ?? "—"}
              </Badge>
            </StatusTooltip>
          </div>

          {hasData ? (
            <Sparkline
              series={sparkline.series}
              ariaLabel={`14-day total_token_throughput trend for ${config.config_name}`}
            />
          ) : (
            <div className="flex h-9 items-center text-[10px] text-muted-foreground/60">
              no recent passed data
            </div>
          )}

          <div className="flex flex-wrap items-baseline justify-between gap-2 text-[11px] text-muted-foreground">
            <span>
              {config.concurrency_levels.length} conc level
              {config.concurrency_levels.length === 1 ? "" : "s"}
            </span>
            <span>latest {formatRelative(config.latest_started_at)}</span>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

function StatusDot({ passing }: { passing: boolean }) {
  return (
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full ${
        passing ? "bg-success" : "bg-destructive"
      }`}
      aria-hidden
    />
  );
}

function EmptyState({ title, hint }: { title: string; hint: string }) {
  return (
    <Card>
      <CardContent className="py-10 text-center">
        <p className="text-[14px] font-medium">{title}</p>
        <p className="mt-1 text-[13px] text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-4">
      <p className="text-[13px] font-medium text-destructive">Dashboard is unreachable</p>
      <p className="mt-1 font-mono text-[11px] text-muted-foreground">{message}</p>
    </div>
  );
}
