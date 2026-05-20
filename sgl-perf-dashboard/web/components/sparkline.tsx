"use client";

import type { SparklineSeries } from "@/lib/api";

const WIDTH = 220;
const HEIGHT = 36;
const PAD_X = 2;
const PAD_Y = 4;

export function Sparkline({
  series,
  ariaLabel,
}: {
  series: SparklineSeries[];
  ariaLabel?: string;
}) {
  if (!series.length || series.every((s) => s.points.length === 0)) {
    return (
      <div
        className="flex h-9 w-full items-center text-[10px] text-muted-foreground/60"
        style={{ width: WIDTH }}
        aria-label={ariaLabel}
      >
        no data
      </div>
    );
  }

  // X domain: union of all timestamps. Y domain: union of all values.
  let minT = Infinity;
  let maxT = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const s of series) {
    for (const p of s.points) {
      const t = Date.parse(p.started_at);
      if (t < minT) minT = t;
      if (t > maxT) maxT = t;
      if (p.value < minV) minV = p.value;
      if (p.value > maxV) maxV = p.value;
    }
  }
  const xSpan = Math.max(1, maxT - minT);
  const ySpan = Math.max(1e-9, maxV - minV);
  const fx = (t: number) =>
    PAD_X + ((t - minT) / xSpan) * (WIDTH - 2 * PAD_X);
  const fy = (v: number) =>
    PAD_Y + (1 - (v - minV) / ySpan) * (HEIGHT - 2 * PAD_Y);

  // Highest concurrency = most opaque. Lower concs muted.
  const sortedConcs = [...series.map((s) => s.concurrency)].sort((a, b) => b - a);
  const opacity = (conc: number) => {
    const rank = sortedConcs.indexOf(conc);
    // 0 (highest) → 1.0, then 0.55, 0.4, 0.3, ...
    const steps = [1.0, 0.55, 0.4, 0.3, 0.22];
    return steps[Math.min(rank, steps.length - 1)];
  };

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={ariaLabel}
      className="overflow-visible"
    >
      {series.map((s) => {
        if (s.points.length === 0) return null;
        const d = s.points
          .map((p, i) => {
            const x = fx(Date.parse(p.started_at)).toFixed(2);
            const y = fy(p.value).toFixed(2);
            return `${i === 0 ? "M" : "L"}${x},${y}`;
          })
          .join(" ");
        return (
          <g key={s.concurrency} opacity={opacity(s.concurrency)}>
            <path
              d={d}
              fill="none"
              stroke="hsl(var(--primary))"
              strokeWidth={1.2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.points.map((p, i) => (
              <circle
                key={i}
                cx={fx(Date.parse(p.started_at))}
                cy={fy(p.value)}
                r={1.2}
                fill="hsl(var(--primary))"
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}
