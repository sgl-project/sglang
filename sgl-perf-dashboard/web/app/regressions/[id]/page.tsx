import Link from "next/link";
import { notFound } from "next/navigation";
import { api } from "@/lib/api";
import { formatNumber, formatRelative } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CopyLinkButton } from "@/components/copy-link-button";

export const dynamic = "force-dynamic";

export default async function RegressionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const regId = Number(id);
  if (!Number.isFinite(regId)) notFound();

  const reg = await api.getRegression(regId).catch(() => null);
  if (!reg) notFound();

  const delta = reg.delta_percent ?? 0;
  return (
    <div className="space-y-8 animate-fade-in-up">
      {/* Header */}
      <section className="border-b border-border/60 pb-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
              Regression
            </p>
            <h1 className="font-mono text-lg font-semibold leading-tight">
              {reg.metric_name}
            </h1>
            <p className="text-[13px] text-muted-foreground">
              {reg.config_name}
              <span className="mx-1.5">·</span>
              concurrency{" "}
              <span className="tabular-numbers text-foreground/80">
                {reg.concurrency.toLocaleString()}
              </span>
              <span className="mx-1.5">·</span>
              detected {formatRelative(reg.detected_at)}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {reg.resolved_at ? (
              <Badge variant="success">resolved</Badge>
            ) : (
              <Badge variant="outline">active</Badge>
            )}
            <CopyLinkButton />
          </div>
        </div>
      </section>

      {/* Delta visualization */}
      <section className="grid gap-4 md:grid-cols-3">
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>Delta</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p
              className={`font-mono text-3xl font-semibold tabular-numbers ${
                delta > 0 ? "text-success" : "text-destructive"
              }`}
            >
              {delta > 0 ? "+" : ""}
              {delta.toFixed(2)}%
            </p>
            {reg.z_score !== null && (
              <p className="text-[12px] text-muted-foreground">
                <span className="font-mono tabular-numbers">z = {reg.z_score.toFixed(2)}</span>
                <span className="mx-1.5">·</span>
                baseline window {reg.baseline_window_days}d
              </p>
            )}
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Values</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
                This run
              </p>
              <p className="mt-1 font-mono text-xl font-semibold tabular-numbers">
                {formatNumber(reg.metric_current_value)}
              </p>
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
                Baseline (median)
              </p>
              <p className="mt-1 font-mono text-xl font-semibold tabular-numbers text-muted-foreground">
                {reg.metric_baseline_median !== null
                  ? formatNumber(reg.metric_baseline_median)
                  : "—"}
              </p>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Bad run + last good run */}
      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-destructive">First failing run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-[13px]">
            <KV label="Started">{formatRelative(reg.started_at)}</KV>
            <KV label="Commit">
              {reg.commit_short_sha ? (
                <Link
                  href={`/commits/${reg.commit_short_sha}`}
                  className="font-mono text-primary hover:underline"
                >
                  {reg.commit_short_sha}
                </Link>
              ) : (
                "—"
              )}
            </KV>
            <KV label="Author">{reg.commit_author ?? "—"}</KV>
            {reg.commit_message && (
              <p className="pt-1 text-[12px] text-foreground/80">
                {reg.commit_message.split("\n")[0]}
              </p>
            )}
            {reg.pr_number && (
              <p className="pt-1 text-[12px]">
                <a
                  className="text-primary hover:underline"
                  href={`https://github.com/sgl-project/sglang/pull/${reg.pr_number}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  PR #{reg.pr_number}
                </a>
                {reg.pr_title && <span className="text-foreground/70"> · {reg.pr_title}</span>}
              </p>
            )}
            <div className="flex gap-2 pt-2">
              <Link
                href={`/runs/${reg.run_id}`}
                className="text-[12px] text-primary hover:underline"
              >
                → run detail
              </Link>
              <a
                className="text-[12px] text-primary hover:underline"
                href={reg.github_run_url}
                target="_blank"
                rel="noreferrer"
              >
                → GitHub Actions
              </a>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-success">Last passing run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-[13px]">
            {reg.last_passing_run_id ? (
              <>
                <KV label="Started">{formatRelative(reg.last_passing_started_at)}</KV>
                <KV label="Commit">
                  {reg.last_passing_commit_short_sha ? (
                    <Link
                      href={`/commits/${reg.last_passing_commit_short_sha}`}
                      className="font-mono text-primary hover:underline"
                    >
                      {reg.last_passing_commit_short_sha}
                    </Link>
                  ) : (
                    "—"
                  )}
                </KV>
                <KV label="Author">{reg.last_passing_commit_author ?? "—"}</KV>
                <div className="flex gap-2 pt-2">
                  <Link
                    href={`/runs/${reg.last_passing_run_id}`}
                    className="text-[12px] text-primary hover:underline"
                  >
                    → run detail
                  </Link>
                  <Link
                    href={`/compare/${reg.last_passing_run_id}/${reg.run_id}`}
                    className="text-[12px] text-primary hover:underline"
                  >
                    → compare with failing
                  </Link>
                </div>
              </>
            ) : (
              <p className="text-[12px] text-muted-foreground">
                No prior passing run on this metric — this may be the first data point
                on a fresh baseline.
              </p>
            )}
          </CardContent>
        </Card>
      </section>

      {/* Commit range hint */}
      {reg.last_passing_commit_sha && reg.commit_short_sha && (
        <Card>
          <CardHeader>
            <CardTitle>Suspect commits</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-[13px] text-muted-foreground">
              Compare the commit range on GitHub to find the culprit:
            </p>
            <a
              className="inline-block break-all font-mono text-[12px] text-primary hover:underline"
              href={`https://github.com/sgl-project/sglang/compare/${reg.last_passing_commit_sha}...${reg.commit_short_sha}`}
              target="_blank"
              rel="noreferrer"
            >
              {reg.last_passing_commit_short_sha}...{reg.commit_short_sha}
            </a>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function KV({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="w-20 shrink-0 text-[11px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span className="min-w-0">{children}</span>
    </div>
  );
}
