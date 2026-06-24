#!/usr/bin/env python3
"""Generate KV-transfer latency comparison charts from the Markdown report.

The original CSV files referenced by the report may live under /tmp and expire.
This script treats the checked-in Markdown tables as the reproducible source.
"""

from __future__ import annotations

import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "docs/superpowers/reports/2026-06-23-kv-transfer-background-traffic-report.md"
OUT_DIR = ROOT / "docs/superpowers/reports/figures/kv-transfer-latency-groups"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

SERIES_STYLE = {
    "bg0": {"color": "#5477C4", "dash": "", "label": "bg0 baseline"},
    "bg1": {"color": "#71B436", "dash": "7 4", "label": "bg1"},
    "bg10": {"color": "#B8A037", "dash": "2 4", "label": "bg10"},
    "bg50": {"color": "#CC6F47", "dash": "9 3 2 3", "label": "bg50"},
    "bg90": {"color": "#BD569B", "dash": "11 5", "label": "bg90"},
}

GROUPS = {
    "rdma_200_1x200_mlx5_0": {
        "title": "KV-transfer p50 latency: RDMA 200G mlx5_0",
        "subtitle": "Foreground 099 -> 102, p50 latency in ms; colors show background traffic level.",
        "filename": "rdma_200_1x200_mlx5_0_p50_latency.svg",
    },
    "rdma_ipv6_bond0_200_1x200": {
        "title": "KV-transfer p50 latency: IPv6 mlx5_bond_0",
        "subtitle": "bg0 uses the 1x400 baseline table; bg1-bg90 use manual 200_1x200 dense runs.",
        "filename": "rdma_ipv6_bond0_200_1x200_p50_latency.svg",
    },
    "rdma_ipv6_400_2x200": {
        "title": "KV-transfer p50 latency: IPv6 400_2x200",
        "subtitle": "Two-shard aggregation; logical latency is the max shard latency per size and iteration.",
        "filename": "rdma_ipv6_400_2x200_p50_latency.svg",
    },
    "rdma_ipv6_400_4x100": {
        "title": "KV-transfer p50 latency: IPv6 400_4x100",
        "subtitle": "Four-shard aggregation; logical latency is the max shard latency per size and iteration.",
        "filename": "rdma_ipv6_400_4x100_p50_latency.svg",
    },
    "rdma_ipv6_800_4x200": {
        "title": "KV-transfer p50 latency: IPv6 800_4x200",
        "subtitle": "Four-shard aggregation; logical latency is the max shard latency per size and iteration.",
        "filename": "rdma_ipv6_800_4x200_p50_latency.svg",
    },
}


@dataclass(frozen=True)
class Row:
    group: str
    group_title: str
    bg: str
    section: str
    size_label: str
    size_mib: float
    p50_latency_ms: float
    p90_latency_ms: float | None
    p99_latency_ms: float | None


def parse_size_mib(label: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(MiB|GiB)", label.strip())
    if not match:
        raise ValueError(f"Unsupported size label: {label!r}")
    value = float(match.group(1))
    unit = match.group(2)
    return value * (1024.0 if unit == "GiB" else 1.0)


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def split_md_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def classify_section(section: str) -> tuple[str, str] | None:
    if "Smoke" in section:
        return None

    if section == "RDMA 200G mlx5_0 bg0 完整基线":
        return "rdma_200_1x200_mlx5_0", "bg0"

    match = re.fullmatch(r"RDMA 200G mlx5_0 bg(1|10|50|90) 完整测试", section)
    if match:
        return "rdma_200_1x200_mlx5_0", f"bg{match.group(1)}"

    if section == "RDMA IPv6 mlx5_bond_0 1x400 完整基线":
        return "rdma_ipv6_bond0_200_1x200", "bg0"

    match = re.fullmatch(
        r"RDMA IPv6 mlx5_bond_0 manual 200_1x200 bg(1|10|50|90)(?: dense)? 完整测试",
        section,
    )
    if match:
        return "rdma_ipv6_bond0_200_1x200", f"bg{match.group(1)}"

    match = re.fullmatch(
        r"RDMA IPv6 manual (400_2x200|400_4x100|800_4x200) bg(1|10|50|90) dense 完整测试",
        section,
    )
    if match:
        profile, bg = match.groups()
        return f"rdma_ipv6_{profile}", f"bg{bg}"

    return None


def parse_report(report_path: Path) -> list[Row]:
    rows: list[Row] = []
    lines = report_path.read_text(encoding="utf-8").splitlines()
    section = ""
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.startswith("### "):
            section = line[4:].strip()

        if line.startswith("| Size | p50 latency ms |"):
            classification = classify_section(section)
            header = split_md_row(line)
            index += 2

            while index < len(lines) and lines[index].startswith("|"):
                cells = split_md_row(lines[index])
                if len(cells) == len(header):
                    record = dict(zip(header, cells))
                    if classification is not None:
                        group, bg = classification
                        rows.append(
                            Row(
                                group=group,
                                group_title=GROUPS[group]["title"],
                                bg=bg,
                                section=section,
                                size_label=record["Size"],
                                size_mib=parse_size_mib(record["Size"]),
                                p50_latency_ms=float(record["p50 latency ms"]),
                                p90_latency_ms=parse_float(record.get("p90 latency ms", "")),
                                p99_latency_ms=parse_float(record.get("p99 latency ms", "")),
                            )
                        )
                index += 1
            continue

        index += 1

    return rows


def nice_ticks(max_value: float, target_count: int = 6) -> list[float]:
    if max_value <= 0:
        return [0.0, 1.0]
    raw_step = max_value / max(target_count, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = 1 * magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    top = math.ceil(max_value / step) * step
    ticks = []
    value = 0.0
    while value <= top + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}" if value == round(value) else f"{value:.1f}"
    if value == round(value):
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def polyline(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def marker(cx: float, cy: float, color: str, bg: str) -> str:
    if bg == "bg0":
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.2" fill="{color}" stroke="#2E4780" stroke-width="1" />'
    if bg == "bg1":
        return f'<rect x="{cx-3:.2f}" y="{cy-3:.2f}" width="6" height="6" fill="{color}" stroke="#386411" stroke-width="1" />'
    if bg == "bg10":
        points = [(cx, cy - 3.8), (cx + 3.3, cy + 2.2), (cx - 3.3, cy + 2.2)]
        return f'<polygon points="{polyline(points)}" fill="{color}" stroke="#736422" stroke-width="1" />'
    if bg == "bg50":
        points = [(cx, cy - 3.8), (cx + 3.8, cy), (cx, cy + 3.8), (cx - 3.8, cy)]
        return f'<polygon points="{polyline(points)}" fill="{color}" stroke="#804126" stroke-width="1" />'
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.4" fill="#FFFFFF" '
        f'stroke="{color}" stroke-width="2" />'
    )


def draw_chart(group: str, rows: list[Row], out_path: Path) -> None:
    width, height = 1120, 720
    margin_left, margin_right = 90, 36
    margin_top, margin_bottom = 124, 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x_min = min(row.size_mib for row in rows)
    x_max = max(row.size_mib for row in rows)
    y_max = max(row.p50_latency_ms for row in rows)
    y_ticks = nice_ticks(y_max * 1.08)
    y_top = max(y_ticks)

    def sx(size_mib: float) -> float:
        low = math.log2(x_min)
        high = math.log2(x_max)
        return margin_left + ((math.log2(size_mib) - low) / (high - low)) * plot_w

    def sy(latency_ms: float) -> float:
        return margin_top + (1 - latency_ms / y_top) * plot_h

    bg_order = ["bg0", "bg1", "bg10", "bg50", "bg90"]
    present_bgs = [bg for bg in bg_order if any(row.bg == bg for row in rows)]
    x_ticks = [
        (1, "1MiB"),
        (2, "2MiB"),
        (4, "4MiB"),
        (8, "8MiB"),
        (16, "16MiB"),
        (32, "32MiB"),
        (64, "64MiB"),
        (128, "128MiB"),
        (256, "256MiB"),
        (512, "512MiB"),
        (1024, "1GiB"),
        (2048, "2GiB"),
    ]
    x_ticks = [(value, label) for value, label in x_ticks if x_min <= value <= x_max]

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(GROUPS[group]['title'])}</title>",
        f"<desc>{html.escape(GROUPS[group]['subtitle'])}</desc>",
        f'<rect width="{width}" height="{height}" fill="{TOKENS["surface"]}" />',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="{TOKENS["panel"]}" />',
    ]

    for tick in y_ticks:
        y = sy(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" '
            f'stroke="{TOKENS["grid"]}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{margin_left-12}" y="{y+4:.2f}" text-anchor="end" '
            f'font-family="Menlo, Consolas, monospace" font-size="12" fill="{TOKENS["muted"]}">{fmt_tick(tick)}</text>'
        )

    for value, label in x_ticks:
        x = sx(value)
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height-margin_bottom}" '
            f'stroke="{TOKENS["grid"]}" stroke-width="1" stroke-dasharray="2 6" />'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{height-margin_bottom+28}" text-anchor="middle" '
            f'font-family="Menlo, Consolas, monospace" font-size="12" fill="{TOKENS["muted"]}">{label}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" '
        f'stroke="{TOKENS["axis"]}" stroke-width="1.2" />'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" '
        f'stroke="{TOKENS["axis"]}" stroke-width="1.2" />'
    )

    parts.append(
        f'<text x="{margin_left}" y="32" text-anchor="start" font-family="Aptos, Inter, Segoe UI, Arial, sans-serif" '
        f'font-size="19" font-weight="600" fill="{TOKENS["ink"]}">{html.escape(GROUPS[group]["title"])}</text>'
    )
    parts.append(
        f'<text x="{margin_left}" y="58" text-anchor="start" font-family="Aptos, Inter, Segoe UI, Arial, sans-serif" '
        f'font-size="13" fill="{TOKENS["muted"]}">{html.escape(GROUPS[group]["subtitle"])}</text>'
    )

    legend_x = margin_left
    legend_y = 88
    for bg in present_bgs:
        style = SERIES_STYLE[bg]
        label = style["label"]
        color = style["color"]
        dash_attr = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+34}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="2.2"{dash_attr} />'
        )
        parts.append(marker(legend_x + 17, legend_y, color, bg))
        parts.append(
            f'<text x="{legend_x+44}" y="{legend_y+4}" font-family="Aptos, Inter, Segoe UI, Arial, sans-serif" '
            f'font-size="12.5" fill="{TOKENS["ink"]}">{html.escape(label)}</text>'
        )
        legend_x += 116 if bg != "bg0" else 150

    for bg in present_bgs:
        style = SERIES_STYLE[bg]
        color = style["color"]
        dash_attr = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        series = sorted((row for row in rows if row.bg == bg), key=lambda row: row.size_mib)
        points = [(sx(row.size_mib), sy(row.p50_latency_ms)) for row in series]
        parts.append(
            f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"{dash_attr} />'
        )
        for x, y in points:
            parts.append(marker(x, y, color, bg))

        last = series[-1]
        label_x = min(sx(last.size_mib) + 8, width - margin_right - 20)
        label_y = sy(last.p50_latency_ms)
        parts.append(
            f'<text x="{label_x:.2f}" y="{label_y-7:.2f}" text-anchor="end" '
            f'font-family="Menlo, Consolas, monospace" font-size="11.5" fill="{color}">'
            f'{html.escape(style["label"])} {last.p50_latency_ms:.1f}ms</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_w / 2:.2f}" y="{height-24}" text-anchor="middle" '
        f'font-family="Aptos, Inter, Segoe UI, Arial, sans-serif" font-size="13" fill="{TOKENS["ink"]}">'
        "Data size, log2 scale</text>"
    )
    parts.append(
        f'<text x="24" y="{margin_top + plot_h / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {margin_top + plot_h / 2:.2f})" '
        f'font-family="Aptos, Inter, Segoe UI, Arial, sans-serif" font-size="13" fill="{TOKENS["ink"]}">'
        "p50 latency (ms)</text>"
    )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_data_csv(rows: list[Row], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "bg",
                "section",
                "size_label",
                "size_mib",
                "p50_latency_ms",
                "p90_latency_ms",
                "p99_latency_ms",
            ]
        )
        for row in sorted(rows, key=lambda item: (item.group, item.bg, item.size_mib)):
            writer.writerow(
                [
                    row.group,
                    row.bg,
                    row.section,
                    row.size_label,
                    f"{row.size_mib:g}",
                    row.p50_latency_ms,
                    row.p90_latency_ms if row.p90_latency_ms is not None else "",
                    row.p99_latency_ms if row.p99_latency_ms is not None else "",
                ]
            )


def write_index(rows: list[Row], chart_paths: dict[str, Path], out_path: Path) -> None:
    cards = []
    for group, meta in GROUPS.items():
        group_rows = [row for row in rows if row.group == group]
        if not group_rows:
            continue
        present = ", ".join(SERIES_STYLE[bg]["label"] for bg in ["bg0", "bg1", "bg10", "bg50", "bg90"] if any(row.bg == bg for row in group_rows))
        rel_path = chart_paths[group].name
        cards.append(
            f"""
            <section class="chart-card">
              <h2>{html.escape(meta["title"])}</h2>
              <p>{html.escape(meta["subtitle"])} Series: {html.escape(present)}.</p>
              <img src="{html.escape(rel_path)}" alt="{html.escape(meta["title"])}" />
            </section>
            """
        )

    out_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KV Transfer Latency Groups</title>
  <style>
    :root {{
      color-scheme: light;
      --surface: {TOKENS["surface"]};
      --panel: {TOKENS["panel"]};
      --ink: {TOKENS["ink"]};
      --muted: {TOKENS["muted"]};
      --grid: {TOKENS["grid"]};
    }}
    body {{
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: Aptos, Inter, "Segoe UI", Arial, sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      line-height: 1.2;
    }}
    .intro {{
      margin: 0 0 24px;
      color: var(--muted);
      line-height: 1.55;
    }}
    .chart-card {{
      margin: 22px 0 32px;
      padding-top: 10px;
      border-top: 1px solid var(--grid);
    }}
    .chart-card h2 {{
      margin: 0 0 4px;
      font-size: 18px;
      line-height: 1.25;
    }}
    .chart-card p {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      background: var(--panel);
    }}
    code {{
      font-family: Menlo, Consolas, monospace;
      font-size: 0.95em;
    }}
  </style>
</head>
<body>
  <main>
    <h1>KV Transfer 背景流量分组延迟曲线</h1>
    <p class="intro">
      数据来自 <code>{html.escape(str(REPORT.relative_to(ROOT)))}</code> 中的 Markdown 汇总表。
      每张图固定同一 baseline/profile，使用不同颜色和线型区分背景流量，x 轴为数据量 log2 刻度，y 轴为 p50 latency ms。
    </p>
    {"".join(cards)}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_markdown_index(rows: list[Row], chart_paths: dict[str, Path], out_path: Path) -> None:
    lines = [
        "# KV Transfer 背景流量分组延迟曲线",
        "",
        "数据来自 `../2026-06-23-kv-transfer-background-traffic-report.md` 中的 Markdown 汇总表。",
        "每张图固定同一 baseline/profile，颜色和线型区分背景流量；x 轴为数据量 log2 刻度，y 轴为 p50 latency ms。",
        "",
    ]

    for group, meta in GROUPS.items():
        group_rows = [row for row in rows if row.group == group]
        if not group_rows:
            continue
        present = ", ".join(
            SERIES_STYLE[bg]["label"]
            for bg in ["bg0", "bg1", "bg10", "bg50", "bg90"]
            if any(row.bg == bg for row in group_rows)
        )
        rel_path = chart_paths[group].name
        lines.extend(
            [
                f"## {meta['title']}",
                "",
                f"{meta['subtitle']} Series: {present}.",
                "",
                f"![{meta['title']}]({rel_path})",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = parse_report(REPORT)
    if not rows:
        raise SystemExit(f"No chart rows parsed from {REPORT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_data_csv(rows, OUT_DIR / "latency_groups_data.csv")

    chart_paths: dict[str, Path] = {}
    for group, meta in GROUPS.items():
        group_rows = [row for row in rows if row.group == group]
        if not group_rows:
            continue
        out_path = OUT_DIR / meta["filename"]
        draw_chart(group, group_rows, out_path)
        chart_paths[group] = out_path

    write_index(rows, chart_paths, OUT_DIR / "index.html")
    write_markdown_index(rows, chart_paths, OUT_DIR / "index.md")

    print(f"parsed_rows={len(rows)}")
    for group in GROUPS:
        group_rows = [row for row in rows if row.group == group]
        bgs = sorted({row.bg for row in group_rows}, key=lambda bg: ["bg0", "bg1", "bg10", "bg50", "bg90"].index(bg))
        print(f"{group}: rows={len(group_rows)} series={','.join(bgs)}")
    print(f"output_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
