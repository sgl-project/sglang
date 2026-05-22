import Link from "next/link";
import { api } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent } from "@/components/ui/card";
import { AutoRefresh } from "@/components/auto-refresh";

export const dynamic = "force-dynamic";

export default async function RegressionsPage({
  searchParams,
}: {
  searchParams: Promise<{ status?: "active" | "resolved" | "all" }>;
}) {
  const sp = await searchParams;
  const status = sp.status ?? "active";
  const regressions = await api.listRegressions(status).catch(() => []);

  return (
    <div className="space-y-6 animate-fade-in-up">
      <AutoRefresh />
      <section className="flex items-baseline justify-between border-b border-border/60 pb-4">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">Regressions</h1>
          <p className="text-[12px] text-muted-foreground">
            Auto-flagged by rolling median absolute deviation (|z| &gt; 2.5, 30-day window).
          </p>
        </div>
        <div className="flex gap-1">
          {(["active", "resolved", "all"] as const).map((s) => (
            <Link
              key={s}
              href={`?status=${s}`}
              className={`rounded-md border px-2 py-0.5 text-[12px] transition ${
                status === s
                  ? "border-primary/60 bg-primary/10 text-primary"
                  : "border-border/60 text-muted-foreground hover:border-border hover:text-foreground"
              }`}
            >
              {s}
            </Link>
          ))}
        </div>
      </section>

      {regressions.length === 0 ? (
        <Card>
          <CardContent className="py-10 text-center text-[13px] text-muted-foreground">
            {status === "active"
              ? "No active regressions — everything's within baseline."
              : `No ${status} regressions.`}
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {regressions.map((r) => (
            <Link key={r.id} href={`/regressions/${r.id}`} className="block">
              <Card className="transition-colors hover:border-destructive/60">
                <CardContent className="flex flex-wrap items-center gap-3 p-4">
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-mono text-[13px]">
                      <span className="text-foreground/80">{r.config_name}</span>
                      <span className="mx-1.5 text-muted-foreground">·</span>
                      <span>{r.metric_name}</span>
                      <span className="mx-1.5 text-muted-foreground">·</span>
                      <span className="text-muted-foreground">
                        conc {r.concurrency.toLocaleString()}
                      </span>
                    </p>
                    <p className="mt-0.5 text-[11px] text-muted-foreground">
                      {r.delta_percent !== null && (
                        <span
                          className={`font-mono tabular-numbers ${
                            r.delta_percent > 0 ? "text-success" : "text-destructive"
                          }`}
                        >
                          {r.delta_percent > 0 ? "+" : ""}
                          {r.delta_percent.toFixed(1)}%
                        </span>
                      )}
                      {r.z_score !== null && (
                        <>
                          <span className="mx-1 text-muted-foreground/60">·</span>
                          <span className="font-mono">z={r.z_score.toFixed(1)}</span>
                        </>
                      )}
                      <span className="mx-1 text-muted-foreground/60">·</span>
                      detected {formatRelative(r.detected_at)}
                      {r.resolved_at && (
                        <>
                          <span className="mx-1 text-muted-foreground/60">·</span>
                          <span className="text-success">resolved {formatRelative(r.resolved_at)}</span>
                        </>
                      )}
                    </p>
                  </div>
                  <span className="text-[12px] text-muted-foreground">→</span>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
