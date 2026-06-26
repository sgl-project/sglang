#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SCRIPT_DIR, "../../../..");
const DATA_DIR = path.join(ROOT, "kv_muti_hca_unaverage");
const REPORT_DIR = path.join(ROOT, "docs/superpowers/reports");
const OUT_DIR = path.join(REPORT_DIR, "figures/kv-transfer-multi-hca");
const REPORT_PATH = path.join(REPORT_DIR, "2026-06-25-kv-transfer-multi-hca-background-report.md");
const DRIVER_LOGS = [
  path.join(DATA_DIR, "multi-hca-bg.log"),
  path.join(DATA_DIR, "multi-hca-compare-4x100.log"),
];

const KEY_BYTES = new Set(["536870912", "1073741824", "2147483648"]);
const KEY_SIZE_ORDER = ["512.00MiB", "1.00GiB", "2.00GiB"];
const BG_ORDER = [1, 10, 50, 90];
const MULTI_CONFIG_ORDER = ["2x100", "4x50", "2x200", "4x100"];
const GIBPS_TO_GBPS = 8.589934592;

const TOKENS = {
  surface: "#FCFCFD",
  panel: "#FFFFFF",
  ink: "#1F2430",
  muted: "#6F768A",
  grid: "#E6E8F0",
  axis: "#D7DBE7",
};

const COLORS = {
  blue: { base: "#A3BEFA", mid: "#5477C4", dark: "#2E4780" },
  gold: { base: "#FFE15B", mid: "#B8A037", dark: "#736422" },
  orange: { base: "#F0986E", mid: "#CC6F47", dark: "#804126" },
  olive: { base: "#A3D576", mid: "#71B436", dark: "#386411" },
  pink: { base: "#F390CA", mid: "#BD569B", dark: "#8A3A6F" },
  neutral: { base: "#C5CAD3", mid: "#7A828F", dark: "#464C55" },
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(header.map((key, index) => [key, cells[index] ?? ""]));
  });
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\n\r]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function toCsv(rows, columns) {
  return [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
  ].join("\n") + "\n";
}

function parseRunName(run) {
  let match = run.match(/^(\d+)_(\d+)x(\d+)_bg(\d+)_multi_hca_moonbg$/);
  if (match) {
    const [, total, lanes, hcaGbps, bg] = match;
    return {
      run,
      mode: "multi-HCA",
      total_bandwidth_gbps: Number(total),
      hca_config: `${lanes}x${hcaGbps}`,
      hca_count: Number(lanes),
      per_hca_gbps: Number(hcaGbps),
      bg_percent: Number(bg),
      bg_label: `bg${bg}`,
      split_per_shard_cap_gbps: "",
    };
  }

  match = run.match(/^(\d+)_(\d+)x(\d+)_bg(\d+)_cap(\d+)_moonbg_split$/);
  if (match) {
    const [, total, lanes, hcaGbps, bg, cap] = match;
    return {
      run,
      mode: "split",
      total_bandwidth_gbps: Number(total),
      hca_config: `${lanes}x${hcaGbps}`,
      hca_count: Number(lanes),
      per_hca_gbps: Number(hcaGbps),
      bg_percent: Number(bg),
      bg_label: `bg${bg}`,
      split_per_shard_cap_gbps: Number(cap),
    };
  }

  return null;
}

function expectedRates(meta) {
  const expected_bg_limit_gbps = meta.total_bandwidth_gbps * meta.bg_percent / 100;
  const expected_fg_limit_gbps = meta.total_bandwidth_gbps - expected_bg_limit_gbps;
  const expected_bg_per_flow_gbps = meta.mode === "split"
    ? expected_bg_limit_gbps / meta.hca_count
    : expected_bg_limit_gbps;
  const expected_fg_per_flow_gbps = meta.mode === "split"
    ? expected_fg_limit_gbps / meta.hca_count
    : expected_fg_limit_gbps;
  return {
    expected_bg_limit_gbps,
    expected_fg_limit_gbps,
    expected_bg_per_flow_gbps,
    expected_fg_per_flow_gbps,
  };
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readDriverLogText() {
  const parts = [];
  for (const file of DRIVER_LOGS) {
    if (await fileExists(file)) {
      parts.push(await fs.readFile(file, "utf8"));
    }
  }
  return parts.join("\n");
}

function extractDriverInfo(driverText, run) {
  const commands = driverText
    .split(/\r?\n/)
    .filter((line) => line.includes(`/auto/${run}/raw/`) && line.includes("kv_transfer_latency.py --role initiator"));

  const fg = [];
  const bg = [];
  for (const line of commands) {
    const flow = line.match(/--flow-id\s+([^'\s]+)/)?.[1] ?? "";
    const rate = Number(line.match(/--rate-limit-gbps\s+([0-9.]+)/)?.[1] ?? NaN);
    const ibDevice = line.match(/--ib-device\s+([^'\s]+)/)?.[1] ?? "";
    const sizes = line.match(/--sizes\s+([^'\s]+)/)?.[1] ?? "";
    const record = { flow, rate, ibDevice, sizes };
    if (flow.startsWith("bg-")) bg.push(record);
    if (flow.startsWith("fg-")) fg.push(record);
  }

  return {
    fg,
    bg,
    monitor_setup_in_driver_log: driverText.includes(`/auto/${run}/raw/rdma-rcv-monitor.csv`),
  };
}

async function rawSampleStats(runDir) {
  const rawDir = path.join(runDir, "raw");
  const entries = await fs.readdir(rawDir, { withFileTypes: true }).catch(() => []);
  const sampleFiles = entries
    .filter((entry) => entry.isFile() && entry.name.startsWith("shard-") && entry.name.endsWith("-samples.jsonl"))
    .map((entry) => path.join(rawDir, entry.name));

  let sample_count = 0;
  let ret_error_count = 0;
  const rateLimits = new Set();
  const flowIds = new Set();

  for (const file of sampleFiles) {
    const lines = (await fs.readFile(file, "utf8")).split(/\r?\n/).filter(Boolean);
    sample_count += lines.length;
    for (const line of lines) {
      const record = JSON.parse(line);
      rateLimits.add(String(record.rate_limit_gbps));
      flowIds.add(record.flow_id);
      if (record.ret !== 0) ret_error_count += 1;
    }
  }

  return {
    sample_files: sampleFiles.length,
    sample_count,
    ret_error_count,
    raw_fg_rate_limit_gbps_values: [...rateLimits].sort((a, b) => Number(a) - Number(b)).join(";"),
    raw_fg_flow_ids: [...flowIds].sort().join(";"),
  };
}

function round(value, digits = 3) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(value).toFixed(digits);
}

function pct(value, digits = 1) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return `${Number(value).toFixed(digits)}%`;
}

function humanMode(mode) {
  return mode === "multi-HCA" ? "multi-HCA single logical flow" : "manual split shards";
}

function mdEscape(text) {
  return String(text).replaceAll("|", "\\|");
}

function markdownTable(rows, columns) {
  const header = `| ${columns.map((column) => mdEscape(column.label)).join(" | ")} |`;
  const separator = `| ${columns.map((column) => column.align === "right" ? "---:" : "---").join(" | ")} |`;
  const body = rows.map((row) => {
    return `| ${columns.map((column) => mdEscape(column.format ? column.format(row[column.key], row) : row[column.key])).join(" | ")} |`;
  });
  return [header, separator, ...body].join("\n");
}

function valueBySize(rows, sizeLabel, key) {
  return rows.find((row) => row.human_bytes === sizeLabel)?.[key] ?? "";
}

function metricCompact(rows, sizeLabel) {
  const row = rows.find((item) => item.human_bytes === sizeLabel);
  if (!row) return "";
  return `${round(row.latency_ms_p50)}/${round(row.latency_ms_p90)}/${round(row.latency_ms_p99)} ms, ${round(row.bandwidth_GBps_p50)} GiB/s`;
}

function wrap(text, maxChars) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let current = "";
  for (const word of words) {
    if ((current + " " + word).trim().length > maxChars && current) {
      lines.push(current);
      current = word;
    } else {
      current = `${current} ${word}`.trim();
    }
  }
  if (current) lines.push(current);
  return lines;
}

function svgText(x, y, text, attrs = "") {
  return `<text x="${x}" y="${y}" ${attrs}>${escapeXml(text)}</text>`;
}

function escapeXml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function lineChartSvg({ title, subtitle, series, xLabels, yLabel, output, logY = false, yDomain = null }) {
  const width = 980;
  const height = 540;
  const margin = { top: 118, right: 48, bottom: 78, left: 92 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const values = series.flatMap((item) => item.values).filter((value) => Number.isFinite(value) && value > 0);
  const maxValue = yDomain?.[1] ?? Math.max(...values) * 1.12;
  const minValue = yDomain?.[0] ?? (logY ? Math.max(1, Math.min(...values) * 0.72) : 0);
  const x = (index) => margin.left + (xLabels.length === 1 ? plotW / 2 : index * plotW / (xLabels.length - 1));
  const y = (value) => {
    if (logY) {
      const lo = Math.log10(minValue);
      const hi = Math.log10(maxValue);
      return margin.top + plotH - (Math.log10(value) - lo) / (hi - lo) * plotH;
    }
    return margin.top + plotH - (value - minValue) / (maxValue - minValue) * plotH;
  };
  const ticks = logY
    ? [40, 80, 160, 320, 640, 1000].filter((tick) => tick >= minValue && tick <= maxValue)
    : niceTicks(maxValue, 5);

  const grid = ticks.map((tick) => {
    const yy = y(tick);
    return [
      `<line x1="${margin.left}" x2="${margin.left + plotW}" y1="${yy}" y2="${yy}" stroke="${TOKENS.grid}" stroke-width="1" />`,
      svgText(margin.left - 12, yy + 4, formatAxis(tick), `text-anchor="end" class="tick"`),
    ].join("\n");
  }).join("\n");

  const xAxis = xLabels.map((label, index) => {
    const xx = x(index);
    return [
      `<line x1="${xx}" x2="${xx}" y1="${margin.top + plotH}" y2="${margin.top + plotH + 6}" stroke="${TOKENS.axis}" />`,
      svgText(xx, margin.top + plotH + 28, label, `text-anchor="middle" class="tick"`),
    ].join("\n");
  }).join("\n");

  const paths = series.map((item) => {
    const d = item.values.map((value, index) => `${index === 0 ? "M" : "L"} ${x(index).toFixed(2)} ${y(value).toFixed(2)}`).join(" ");
    const points = item.values.map((value, index) => `<circle cx="${x(index).toFixed(2)}" cy="${y(value).toFixed(2)}" r="4.5" fill="${item.color}" stroke="${item.stroke}" stroke-width="1.2"><title>${escapeXml(item.label)} ${xLabels[index]}: ${round(value)}</title></circle>`).join("\n");
    return `<path d="${d}" fill="none" stroke="${item.stroke}" stroke-width="2.2" ${item.dash ? `stroke-dasharray="${item.dash}"` : ""} />\n${points}`;
  }).join("\n");

  let legendX = margin.left;
  let legendY = 78;
  const legend = series.map((item) => {
    const textWidth = item.label.length * 7.3 + 48;
    const piece = [
      `<line x1="${legendX}" x2="${legendX + 28}" y1="${legendY}" y2="${legendY}" stroke="${item.stroke}" stroke-width="2.2" ${item.dash ? `stroke-dasharray="${item.dash}"` : ""} />`,
      `<circle cx="${legendX + 14}" cy="${legendY}" r="4.2" fill="${item.color}" stroke="${item.stroke}" stroke-width="1" />`,
      svgText(legendX + 36, legendY + 4, item.label, `class="legend"`),
    ].join("\n");
    legendX += textWidth;
    return piece;
  }).join("\n");

  const subtitleLines = wrap(subtitle, 132);
  const subtitleSvg = subtitleLines.map((line, index) => svgText(margin.left, 38 + index * 17, line, `class="subtitle"`)).join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(title)}</title>
  <desc id="desc">${escapeXml(subtitle)}</desc>
  <style>
    .title { font: 700 18px "Segoe UI", Arial, sans-serif; fill: ${TOKENS.ink}; }
    .subtitle { font: 400 12px "Segoe UI", Arial, sans-serif; fill: ${TOKENS.muted}; }
    .tick { font: 400 11px "Consolas", "Segoe UI", monospace; fill: ${TOKENS.muted}; }
    .axis-label { font: 600 12px "Segoe UI", Arial, sans-serif; fill: ${TOKENS.ink}; }
    .legend { font: 500 12px "Segoe UI", Arial, sans-serif; fill: ${TOKENS.ink}; }
  </style>
  <rect width="${width}" height="${height}" fill="${TOKENS.surface}" />
  <rect x="${margin.left}" y="${margin.top}" width="${plotW}" height="${plotH}" fill="${TOKENS.panel}" />
  ${svgText(margin.left, 24, title, `class="title"`)}
  ${subtitleSvg}
  ${legend}
  ${grid}
  <line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${margin.top + plotH}" stroke="${TOKENS.axis}" stroke-width="1.2" />
  <line x1="${margin.left}" x2="${margin.left + plotW}" y1="${margin.top + plotH}" y2="${margin.top + plotH}" stroke="${TOKENS.axis}" stroke-width="1.2" />
  ${xAxis}
  ${svgText(18, margin.top + plotH / 2, yLabel, `class="axis-label" transform="rotate(-90 18 ${margin.top + plotH / 2})" text-anchor="middle"`)}
  ${paths}
</svg>
`;
}

function niceTicks(maxValue, targetCount) {
  const rawStep = maxValue / targetCount;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / magnitude;
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
  const ticks = [];
  for (let value = 0; value <= maxValue + step * 0.2; value += step) ticks.push(value);
  return ticks;
}

function formatAxis(value) {
  if (value >= 100) return String(Math.round(value));
  if (value >= 10) return value.toFixed(0);
  return value.toFixed(1);
}

function get2GiB(rows, predicate) {
  return rows.filter((row) => row.human_bytes === "2.00GiB" && predicate(row));
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const driverText = await readDriverLogText();
  const entries = await fs.readdir(DATA_DIR, { withFileTypes: true });
  const runDirs = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .map((run) => ({ run, meta: parseRunName(run) }))
    .filter((item) => item.meta)
    .sort((a, b) => {
      const mode = a.meta.mode.localeCompare(b.meta.mode);
      if (mode !== 0) return mode;
      const config = MULTI_CONFIG_ORDER.indexOf(a.meta.hca_config) - MULTI_CONFIG_ORDER.indexOf(b.meta.hca_config);
      if (config !== 0) return config;
      return a.meta.bg_percent - b.meta.bg_percent;
    });

  const allRows = [];
  const manifest = [];
  const monitorRows = [];

  for (const { run, meta } of runDirs) {
    const runDir = path.join(DATA_DIR, run);
    const rates = expectedRates(meta);
    const driverInfo = extractDriverInfo(driverText, run);
    const samples = await rawSampleStats(runDir);
    const monitorPath = path.join(runDir, "raw", "rdma-rcv-monitor.csv");
    const monitorPresent = await fileExists(monitorPath);
    const summaryRows = parseCsv(await fs.readFile(path.join(runDir, "aggregated-summary.csv"), "utf8"));
    const totalErrors = summaryRows.reduce((sum, row) => sum + Number(row.error_count), 0);

    const fgRateValues = [...new Set(driverInfo.fg.map((record) => record.rate).filter(Number.isFinite))].sort((a, b) => a - b);
    const bgRateValues = [...new Set(driverInfo.bg.map((record) => record.rate).filter(Number.isFinite))].sort((a, b) => a - b);
    const ibDevices = [...new Set([...driverInfo.fg, ...driverInfo.bg].map((record) => record.ibDevice).filter(Boolean))].sort();

    manifest.push({
      run,
      mode: meta.mode,
      hca_config: meta.hca_config,
      total_bandwidth_gbps: meta.total_bandwidth_gbps,
      bg_label: meta.bg_label,
      bg_percent: meta.bg_percent,
      expected_bg_limit_gbps: round(rates.expected_bg_limit_gbps, 1),
      expected_fg_limit_gbps: round(rates.expected_fg_limit_gbps, 1),
      expected_bg_per_flow_gbps: round(rates.expected_bg_per_flow_gbps, 1),
      expected_fg_per_flow_gbps: round(rates.expected_fg_per_flow_gbps, 1),
      driver_fg_rate_limit_gbps_values: fgRateValues.join(";"),
      driver_bg_rate_limit_gbps_values: bgRateValues.join(";"),
      ib_device_values: ibDevices.join(";"),
      aggregated_rows: summaryRows.length,
      aggregated_error_count_sum: totalErrors,
      raw_sample_files: samples.sample_files,
      raw_sample_count: samples.sample_count,
      raw_sample_ret_error_count: samples.ret_error_count,
      raw_fg_rate_limit_gbps_values: samples.raw_fg_rate_limit_gbps_values,
      raw_fg_flow_ids: samples.raw_fg_flow_ids,
      rdma_monitor_present: monitorPresent ? "yes" : "no",
      monitor_setup_in_driver_log: driverInfo.monitor_setup_in_driver_log ? "yes" : "no",
    });

    monitorRows.push({
      run,
      mode: meta.mode,
      hca_config: meta.hca_config,
      bg_label: meta.bg_label,
      expected_bg_limit_gbps: round(rates.expected_bg_limit_gbps, 1),
      monitor_setup_in_driver_log: driverInfo.monitor_setup_in_driver_log ? "yes" : "no",
      local_raw_rdma_monitor_csv_present: monitorPresent ? "yes" : "no",
      local_raw_rdma_monitor_csv_path: path.relative(ROOT, monitorPath).replaceAll("\\", "/"),
      validation_status: monitorPresent
        ? "available for receiver-side validation"
        : "missing from local result bundle; receiver-side validation not possible from checked-in raw files",
    });

    for (const row of summaryRows) {
      const bw = Number(row.bandwidth_GBps_p50);
      const actualGbps = bw * GIBPS_TO_GBPS;
      allRows.push({
        run,
        mode: meta.mode,
        hca_config: meta.hca_config,
        total_bandwidth_gbps: meta.total_bandwidth_gbps,
        bg_label: meta.bg_label,
        bg_percent: meta.bg_percent,
        expected_bg_limit_gbps: round(rates.expected_bg_limit_gbps, 1),
        expected_fg_limit_gbps: round(rates.expected_fg_limit_gbps, 1),
        driver_fg_rate_limit_gbps_values: fgRateValues.join(";"),
        driver_bg_rate_limit_gbps_values: bgRateValues.join(";"),
        ib_device_values: ibDevices.join(";"),
        size_bytes: row.bytes,
        human_bytes: row.human_bytes,
        shard_count: row.shard_count,
        repeat_count: row.repeat_count,
        error_count: row.error_count,
        latency_ms_mean: Number(row.latency_ms_mean),
        latency_ms_p50: Number(row.latency_ms_p50),
        latency_ms_p90: Number(row.latency_ms_p90),
        latency_ms_p99: Number(row.latency_ms_p99),
        latency_ms_min: Number(row.latency_ms_min),
        latency_ms_max: Number(row.latency_ms_max),
        bandwidth_GBps_p50: bw,
        bandwidth_GBps_mean: Number(row.bandwidth_GBps_mean),
        bandwidth_gbps_p50_binary: actualGbps,
        foreground_cap_utilization_pct: actualGbps / rates.expected_fg_limit_gbps * 100,
      });
    }
  }

  const keyRows = allRows.filter((row) => KEY_BYTES.has(String(row.size_bytes)));
  const keyByRun = Object.groupBy(keyRows, (row) => row.run);

  const compare4x100 = [];
  for (const bg of BG_ORDER) {
    for (const size of KEY_SIZE_ORDER) {
      const split = keyRows.find((row) => row.mode === "split" && row.hca_config === "4x100" && row.bg_percent === bg && row.human_bytes === size);
      const multi = keyRows.find((row) => row.mode === "multi-HCA" && row.hca_config === "4x100" && row.bg_percent === bg && row.human_bytes === size);
      if (!split || !multi) continue;
      compare4x100.push({
        bg_label: `bg${bg}`,
        size,
        split_latency_ms_p50: round(split.latency_ms_p50),
        split_latency_ms_p90: round(split.latency_ms_p90),
        split_latency_ms_p99: round(split.latency_ms_p99),
        split_bandwidth_GBps_p50: round(split.bandwidth_GBps_p50),
        multi_latency_ms_p50: round(multi.latency_ms_p50),
        multi_latency_ms_p90: round(multi.latency_ms_p90),
        multi_latency_ms_p99: round(multi.latency_ms_p99),
        multi_bandwidth_GBps_p50: round(multi.bandwidth_GBps_p50),
        multi_vs_split_bandwidth_pct: round((multi.bandwidth_GBps_p50 / split.bandwidth_GBps_p50 - 1) * 100, 1),
        multi_vs_split_latency_pct: round((multi.latency_ms_p50 / split.latency_ms_p50 - 1) * 100, 1),
      });
    }
  }

  const trend2g = get2GiB(allRows, (row) => row.mode === "multi-HCA").map((row) => ({
    run: row.run,
    hca_config: row.hca_config,
    total_bandwidth_gbps: row.total_bandwidth_gbps,
    bg_label: row.bg_label,
    bg_percent: row.bg_percent,
    expected_fg_limit_gbps: row.expected_fg_limit_gbps,
    latency_ms_p50: round(row.latency_ms_p50),
    latency_ms_p90: round(row.latency_ms_p90),
    latency_ms_p99: round(row.latency_ms_p99),
    bandwidth_GBps_p50: round(row.bandwidth_GBps_p50),
    bandwidth_gbps_p50_binary: round(row.bandwidth_gbps_p50_binary, 1),
    foreground_cap_utilization_pct: round(row.foreground_cap_utilization_pct, 1),
  })).sort((a, b) => MULTI_CONFIG_ORDER.indexOf(a.hca_config) - MULTI_CONFIG_ORDER.indexOf(b.hca_config) || a.bg_percent - b.bg_percent);

  const anomalies = get2GiB(allRows, () => true)
    .filter((row) => row.foreground_cap_utilization_pct < 90 || row.latency_ms_p99 / row.latency_ms_p50 > 1.2)
    .map((row) => ({
      run: row.run,
      mode: row.mode,
      hca_config: row.hca_config,
      bg_label: row.bg_label,
      size: row.human_bytes,
      expected_fg_limit_gbps: row.expected_fg_limit_gbps,
      bandwidth_gbps_p50_binary: round(row.bandwidth_gbps_p50_binary, 1),
      foreground_cap_utilization_pct: round(row.foreground_cap_utilization_pct, 1),
      latency_ms_p50: round(row.latency_ms_p50),
      latency_ms_p99: round(row.latency_ms_p99),
      possible_reason: row.mode === "multi-HCA" && row.total_bandwidth_gbps === 400
        ? "single logical multi-HCA foreground flow does not scale like manual shard split at this residual foreground cap"
        : "tail latency spread or cap utilization needs raw log follow-up",
    }));

  await fs.writeFile(path.join(OUT_DIR, "run-manifest.csv"), toCsv(manifest, [
    "run", "mode", "hca_config", "total_bandwidth_gbps", "bg_label", "bg_percent",
    "expected_bg_limit_gbps", "expected_fg_limit_gbps", "expected_bg_per_flow_gbps", "expected_fg_per_flow_gbps",
    "driver_fg_rate_limit_gbps_values", "driver_bg_rate_limit_gbps_values", "ib_device_values",
    "aggregated_rows", "aggregated_error_count_sum", "raw_sample_files", "raw_sample_count", "raw_sample_ret_error_count",
    "raw_fg_rate_limit_gbps_values", "raw_fg_flow_ids", "rdma_monitor_present", "monitor_setup_in_driver_log",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "all-aggregated-summary-with-metadata.csv"), toCsv(allRows, [
    "run", "mode", "hca_config", "total_bandwidth_gbps", "bg_label", "bg_percent",
    "expected_bg_limit_gbps", "expected_fg_limit_gbps", "driver_fg_rate_limit_gbps_values", "driver_bg_rate_limit_gbps_values",
    "ib_device_values", "size_bytes", "human_bytes", "shard_count", "repeat_count", "error_count",
    "latency_ms_mean", "latency_ms_p50", "latency_ms_p90", "latency_ms_p99", "latency_ms_min", "latency_ms_max",
    "bandwidth_GBps_p50", "bandwidth_GBps_mean", "bandwidth_gbps_p50_binary", "foreground_cap_utilization_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "key-results-512mib-1gib-2gib.csv"), toCsv(keyRows, [
    "run", "mode", "hca_config", "total_bandwidth_gbps", "bg_label", "bg_percent",
    "expected_bg_limit_gbps", "expected_fg_limit_gbps", "human_bytes",
    "latency_ms_p50", "latency_ms_p90", "latency_ms_p99", "bandwidth_GBps_p50",
    "bandwidth_gbps_p50_binary", "foreground_cap_utilization_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "compare-4x100-split-vs-multi-hca.csv"), toCsv(compare4x100, [
    "bg_label", "size", "split_latency_ms_p50", "split_latency_ms_p90", "split_latency_ms_p99", "split_bandwidth_GBps_p50",
    "multi_latency_ms_p50", "multi_latency_ms_p90", "multi_latency_ms_p99", "multi_bandwidth_GBps_p50",
    "multi_vs_split_bandwidth_pct", "multi_vs_split_latency_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "multi-hca-2gib-trends.csv"), toCsv(trend2g, [
    "run", "hca_config", "total_bandwidth_gbps", "bg_label", "bg_percent", "expected_fg_limit_gbps",
    "latency_ms_p50", "latency_ms_p90", "latency_ms_p99", "bandwidth_GBps_p50",
    "bandwidth_gbps_p50_binary", "foreground_cap_utilization_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "rdma-monitor-inventory.csv"), toCsv(monitorRows, [
    "run", "mode", "hca_config", "bg_label", "expected_bg_limit_gbps", "monitor_setup_in_driver_log",
    "local_raw_rdma_monitor_csv_present", "local_raw_rdma_monitor_csv_path", "validation_status",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "anomalies.csv"), toCsv(anomalies, [
    "run", "mode", "hca_config", "bg_label", "size", "expected_fg_limit_gbps",
    "bandwidth_gbps_p50_binary", "foreground_cap_utilization_pct", "latency_ms_p50", "latency_ms_p99", "possible_reason",
  ]));

  const bgLabels = BG_ORDER.map((bg) => `bg${bg}`);
  const split4 = get2GiB(allRows, (row) => row.mode === "split" && row.hca_config === "4x100").sort((a, b) => a.bg_percent - b.bg_percent);
  const multi4 = get2GiB(allRows, (row) => row.mode === "multi-HCA" && row.hca_config === "4x100").sort((a, b) => a.bg_percent - b.bg_percent);
  const multi2gByConfig = MULTI_CONFIG_ORDER.map((config, index) => {
    const palette = [COLORS.blue, COLORS.gold, COLORS.orange, COLORS.olive][index];
    return {
      label: config,
      color: palette.base,
      stroke: palette.dark,
      dash: index === 0 ? "" : index === 1 ? "7 4" : index === 2 ? "2 4" : "9 3 2 3",
      rows: get2GiB(allRows, (row) => row.mode === "multi-HCA" && row.hca_config === config).sort((a, b) => a.bg_percent - b.bg_percent),
    };
  });

  await fs.writeFile(path.join(OUT_DIR, "compare-4x100-2gib-bandwidth.svg"), lineChartSvg({
    title: "4x100 2GiB foreground bandwidth: split vs multi-HCA",
    subtitle: "p50 bandwidth_GBps from aggregated summaries; split uses four manual 100G shards, multi-HCA uses one logical foreground flow across four HCA names.",
    xLabels: bgLabels,
    yLabel: "bandwidth_GBps_p50",
    series: [
      { label: "split 4 shards", color: COLORS.blue.base, stroke: COLORS.blue.dark, values: split4.map((row) => row.bandwidth_GBps_p50) },
      { label: "multi-HCA single flow", color: COLORS.orange.base, stroke: COLORS.orange.dark, dash: "7 4", values: multi4.map((row) => row.bandwidth_GBps_p50) },
    ],
  }));

  await fs.writeFile(path.join(OUT_DIR, "compare-4x100-2gib-latency.svg"), lineChartSvg({
    title: "4x100 2GiB foreground p50 latency: split vs multi-HCA",
    subtitle: "p50 latency in milliseconds; y-axis uses a log scale so bg1/bg10/bg50/bg90 remain visible together.",
    xLabels: bgLabels,
    yLabel: "p50 latency ms (log)",
    logY: true,
    series: [
      { label: "split 4 shards", color: COLORS.blue.base, stroke: COLORS.blue.dark, values: split4.map((row) => row.latency_ms_p50) },
      { label: "multi-HCA single flow", color: COLORS.orange.base, stroke: COLORS.orange.dark, dash: "7 4", values: multi4.map((row) => row.latency_ms_p50) },
    ],
  }));

  await fs.writeFile(path.join(OUT_DIR, "multi-hca-2gib-bandwidth.svg"), lineChartSvg({
    title: "multi-HCA 2GiB foreground bandwidth by total bandwidth and background share",
    subtitle: "All rows are single logical foreground flows; bandwidth falls with background share and the 400G configurations do not reach the low-background foreground caps.",
    xLabels: bgLabels,
    yLabel: "bandwidth_GBps_p50",
    series: multi2gByConfig.map((item) => ({
      label: item.label,
      color: item.color,
      stroke: item.stroke,
      dash: item.dash,
      values: item.rows.map((row) => row.bandwidth_GBps_p50),
    })),
  }));

  await fs.writeFile(path.join(OUT_DIR, "multi-hca-2gib-cap-utilization.svg"), lineChartSvg({
    title: "multi-HCA 2GiB foreground cap utilization",
    subtitle: "Utilization converts bandwidth_GBps_p50 to Gbps and divides by configured foreground cap; 400G bg1/bg10 sit near half of target.",
    xLabels: bgLabels,
    yLabel: "cap utilization",
    yDomain: [0, 110],
    series: multi2gByConfig.map((item) => ({
      label: item.label,
      color: item.color,
      stroke: item.stroke,
      dash: item.dash,
      values: item.rows.map((row) => row.foreground_cap_utilization_pct),
    })),
  }));

  const chartMap = [
    {
      file: "compare-4x100-2gib-bandwidth.svg",
      section: "4x100 split vs multi-HCA",
      question: "Does a single multi-HCA logical flow match manual split throughput at 2GiB?",
      chart_type: "line",
      supported_claim: "manual split is about 1.94x faster at bg1/bg10 and about 1.43x faster at bg50; bg90 converges.",
    },
    {
      file: "compare-4x100-2gib-latency.svg",
      section: "4x100 split vs multi-HCA",
      question: "How much latency does multi-HCA add relative to split?",
      chart_type: "log-scale line",
      supported_claim: "multi-HCA roughly doubles p50 latency at bg1/bg10 and is 43% higher at bg50; bg90 converges.",
    },
    {
      file: "multi-hca-2gib-bandwidth.svg",
      section: "multi-HCA trend",
      question: "How do 2x100, 4x50, 2x200, and 4x100 multi-HCA runs trend with background share?",
      chart_type: "multi-series line",
      supported_claim: "200G total configs track their caps; 400G configs plateau around 200Gbps at low background and underuse bg50 foreground cap.",
    },
    {
      file: "multi-hca-2gib-cap-utilization.svg",
      section: "multi-HCA trend",
      question: "How closely does p50 throughput match the configured foreground rate limit?",
      chart_type: "multi-series line",
      supported_claim: "2x100 and 4x50 stay near 100%; 2x200/4x100 are near 51-52% at bg1/bg10, about 70% at bg50, and 100% at bg90.",
    },
  ];
  await fs.writeFile(path.join(OUT_DIR, "chart-map.csv"), toCsv(chartMap, [
    "file", "section", "question", "chart_type", "supported_claim",
  ]));

  const wideMultiRows = MULTI_CONFIG_ORDER.flatMap((config) => {
    return BG_ORDER.map((bg) => {
      const rows = keyRows
        .filter((row) => row.mode === "multi-HCA" && row.hca_config === config && row.bg_percent === bg)
        .sort((a, b) => KEY_SIZE_ORDER.indexOf(a.human_bytes) - KEY_SIZE_ORDER.indexOf(b.human_bytes));
      return {
        profile: config,
        bg: `bg${bg}`,
        fg_cap: valueBySize(rows, "2.00GiB", "expected_fg_limit_gbps"),
        s512: metricCompact(rows, "512.00MiB"),
        s1g: metricCompact(rows, "1.00GiB"),
        s2g: metricCompact(rows, "2.00GiB"),
        util_2g: pct(valueBySize(rows, "2.00GiB", "foreground_cap_utilization_pct")),
      };
    });
  });

  const compare4Rows = compare4x100.map((row) => ({
    bg: row.bg_label,
    size: row.size,
    split: `${row.split_latency_ms_p50}/${row.split_latency_ms_p90}/${row.split_latency_ms_p99} ms, ${row.split_bandwidth_GBps_p50} GiB/s`,
    multi: `${row.multi_latency_ms_p50}/${row.multi_latency_ms_p90}/${row.multi_latency_ms_p99} ms, ${row.multi_bandwidth_GBps_p50} GiB/s`,
    bw_delta: `${row.multi_vs_split_bandwidth_pct}%`,
    latency_delta: `${row.multi_vs_split_latency_pct}%`,
  }));

  const trendRowsForReport = trend2g.map((row) => ({
    profile: row.hca_config,
    bg: row.bg_label,
    fg_cap: row.expected_fg_limit_gbps,
    p50: row.latency_ms_p50,
    p90: row.latency_ms_p90,
    p99: row.latency_ms_p99,
    bw: row.bandwidth_GBps_p50,
    util: `${row.foreground_cap_utilization_pct}%`,
  }));

  const manifestRowsForReport = manifest.map((row) => ({
    mode: humanMode(row.mode),
    profile: row.hca_config,
    bg: row.bg_label,
    fg_cap: row.expected_fg_limit_gbps,
    bg_cap: row.expected_bg_limit_gbps,
    fg_rate: row.driver_fg_rate_limit_gbps_values || row.raw_fg_rate_limit_gbps_values,
    bg_rate: row.driver_bg_rate_limit_gbps_values,
    ib: row.ib_device_values,
    rows: row.aggregated_rows,
    samples: row.raw_sample_count,
    errors: row.aggregated_error_count_sum,
    monitor: row.rdma_monitor_present,
  }));

  const report = `# KV Transfer / Mooncake multi-HCA 不指定均分背景流实验报告

**日期:** 2026-06-25

**数据目录:** \`kv_muti_hca_unaverage/\`

**报告产物目录:** \`docs/superpowers/reports/figures/kv-transfer-multi-hca/\`

## Executive Summary

- **multi-HCA 单逻辑流不等价于手动 shard。** 在 4x100 对照中，manual split 的 2GiB p50 带宽在 bg1/bg10/bg50 分别为 46.036、41.847、23.264 GiB/s；multi-HCA 单逻辑流只有 23.720、21.693、16.285 GiB/s。换算成相对 split，multi-HCA 在前三档分别低约 48.5%、48.2%、30.0%；到 bg90 时两者都被 40Gbps 前景 cap 限住，结果收敛。
- **200G 总带宽配置下，multi-HCA 基本贴住限速目标。** 2x100 和 4x50 在 2GiB/bg1-bg90 下的 foreground cap utilization 都约为 99.8%-100.0%，说明当剩余前景带宽不超过约 200Gbps 时，单逻辑流表现稳定。
- **400G 总带宽配置下，multi-HCA 低背景档出现明显上限。** 2x200 和 4x100 在 bg1/bg10 的 2GiB 前景吞吐只约 204/185-186Gbps，只有 396/360Gbps 目标的约 51%-52%；bg50 也只有约 140Gbps，约 70% cap utilization。
- **接收侧 RDMA monitor 没有随本地结果包落地。** driver log 显示每个 run 都设置了 \`raw/rdma-rcv-monitor.csv\`，但本地 \`kv_muti_hca_unaverage/*/raw/\` 下没有该文件。因此本报告只能验证限速命令和前景样本结果，不能用接收侧 monitor 实测值证明背景流量到位。

## 实验目的

本次实验比较两种多网卡使用方式在 KV Transfer / Mooncake 背景流竞争下的差异：

1. **manual split:** 手动把一个逻辑传输拆成多条 shard，每条 shard 固定绑定一张 HCA，并按每张网卡的 100Gbps cap 拆分前景/背景限速。
2. **multi-HCA 不指定均分:** 不手动切 shard，只启动一个逻辑前景流，把多个 HCA 名称作为逗号分隔的 \`--ib-device\` 传给 Mooncake，让 Mooncake 自己在多 HCA 上调度。

报告重点回答两个问题：第一，4x100 下 multi-HCA 单逻辑流是否接近手动均分 split；第二，2x100、4x50、4x100、2x200 在 bg1/bg10/bg50/bg90 背景占比下是否能按前景剩余 cap 扩展。

## 实验方法

### run 命名和配置

\`<total>_<hca_count>x<per_hca>_bg<percent>_multi_hca_moonbg\` 表示一个 multi-HCA 单逻辑流实验。例如 \`400_4x100_bg10_multi_hca_moonbg\` 表示总预算 400Gbps、4 张 HCA、背景占比 10%，前景是一个逻辑流，\`--ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3\`。

\`400_4x100_bg<percent>_cap100_moonbg_split\` 表示 4x100 manual split 对照：4 条 shard，每条 shard 固定一张 HCA，每条按 100Gbps cap 拆分背景和前景。

主实验的 multi-HCA 配置为：

| Profile | Total cap | HCA set | 语义 |
|---|---:|---|---|
| \`200_2x100\` | 200Gbps | \`mlx5_bond_0,mlx5_bond_1\` | 两张 100G HCA，单逻辑前景流 |
| \`200_4x50\` | 200Gbps | \`mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3\` | 四张 HCA，总 cap 仍为 200G |
| \`400_4x100\` | 400Gbps | \`mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3\` | 四张 100G HCA，单逻辑前景流 |
| \`400_2x200\` | 400Gbps | \`mlx5_bond_0,mlx5_bond_1\` | 两张 HCA，总 cap 400G，单逻辑前景流 |

### 背景比例、size 和限速

每个 profile 跑四组背景占比：\`bg1\`、\`bg10\`、\`bg50\`、\`bg90\`。前景 size 固定为完整 21 个点：

\`1MB, 2MB, 4MB, 8MB, 16MB, 24MB, 32MB, 48MB, 64MB, 96MB, 128MB, 192MB, 256MB, 384MB, 512MB, 768MB, 1GB, 1.25GB, 1.5GB, 1.75GB, 2GB\`。

multi-HCA 每组只有一个逻辑前景流和一个逻辑背景流：

\`\`\`text
background_limit_gbps = total_bandwidth_gbps * bg_percent
foreground_limit_gbps = total_bandwidth_gbps - background_limit_gbps
\`\`\`

split 对照则把同样比例拆到每条 shard 上，例如 4x100/bg10 下每条 shard 背景 10Gbps、前景 90Gbps。

## 数据完整性

所有 20 个 run 都有 \`aggregated-summary.csv\`，每个汇总表 21 行，\`error_count\` 合计为 0。multi-HCA run 每组保留 1 个前景 raw samples 文件，split run 每组保留 4 个前景 raw samples 文件；前景 raw sample 的 \`ret\` 也没有错误记录。

${markdownTable(manifestRowsForReport, [
    { key: "mode", label: "Mode" },
    { key: "profile", label: "Profile" },
    { key: "bg", label: "BG" },
    { key: "fg_cap", label: "FG cap Gbps", align: "right" },
    { key: "bg_cap", label: "BG cap Gbps", align: "right" },
    { key: "fg_rate", label: "FG rate-limit" },
    { key: "bg_rate", label: "BG rate-limit" },
    { key: "rows", label: "Rows", align: "right" },
    { key: "samples", label: "FG samples", align: "right" },
    { key: "errors", label: "Errors", align: "right" },
    { key: "monitor", label: "RDMA monitor" },
  ])}

## 关键结果表

下表单元格格式为 \`p50/p90/p99 latency ms, bandwidth_GBps_p50 GiB/s\`。完整长表在 [key-results-512mib-1gib-2gib.csv](figures/kv-transfer-multi-hca/key-results-512mib-1gib-2gib.csv)。

### multi-HCA 主实验重点 size

${markdownTable(wideMultiRows, [
    { key: "profile", label: "Profile" },
    { key: "bg", label: "BG" },
    { key: "fg_cap", label: "FG cap Gbps", align: "right" },
    { key: "s512", label: "512MiB p50/p90/p99, bw" },
    { key: "s1g", label: "1GiB p50/p90/p99, bw" },
    { key: "s2g", label: "2GiB p50/p90/p99, bw" },
    { key: "util_2g", label: "2GiB cap util", align: "right" },
  ])}

### 4x100 split vs multi-HCA 重点 size

${markdownTable(compare4Rows, [
    { key: "bg", label: "BG" },
    { key: "size", label: "Size" },
    { key: "split", label: "manual split p50/p90/p99, bw" },
    { key: "multi", label: "multi-HCA p50/p90/p99, bw" },
    { key: "bw_delta", label: "multi bw vs split", align: "right" },
    { key: "latency_delta", label: "multi p50 latency vs split", align: "right" },
  ])}

## 4x100 split vs multi-HCA 分析

![4x100 2GiB bandwidth](figures/kv-transfer-multi-hca/compare-4x100-2gib-bandwidth.svg)

4x100 split 在 bg1/bg10/bg50 下基本贴住前景剩余 cap：2GiB p50 带宽分别是 46.036、41.847、23.264 GiB/s，对应约 395.4、359.5、199.8Gbps。multi-HCA 单逻辑流在同样三个背景档只有 23.720、21.693、16.285 GiB/s，低背景档近似只拿到 split 的一半。

![4x100 2GiB latency](figures/kv-transfer-multi-hca/compare-4x100-2gib-latency.svg)

延迟表现与带宽一致：bg1 下 split 的 2GiB p50 latency 是 43.444ms，multi-HCA 是 84.318ms；bg10 是 47.794ms vs 92.195ms；bg50 是 85.970ms vs 122.811ms。bg90 时两者都被前景 40Gbps cap 限住，p50 latency 都约 429.6ms，带宽也都约 4.655 GiB/s。

这个对照说明，multi-HCA 的“传多个 HCA 名称给 Mooncake”在本批数据里不是手动 shard 的替代物。manual split 显式创造了 4 条并行前景 shard，聚合后能吃满 4x100 的剩余前景预算；multi-HCA 仍然是一个逻辑前景流，它的调度和聚合行为没有等价地产生 4 条 shard。

## multi-HCA 主实验趋势

![multi-HCA 2GiB bandwidth](figures/kv-transfer-multi-hca/multi-hca-2gib-bandwidth.svg)

${markdownTable(trendRowsForReport, [
    { key: "profile", label: "Profile" },
    { key: "bg", label: "BG" },
    { key: "fg_cap", label: "FG cap Gbps", align: "right" },
    { key: "p50", label: "p50 ms", align: "right" },
    { key: "p90", label: "p90 ms", align: "right" },
    { key: "p99", label: "p99 ms", align: "right" },
    { key: "bw", label: "bw GiB/s", align: "right" },
    { key: "util", label: "cap util", align: "right" },
  ])}

![multi-HCA 2GiB cap utilization](figures/kv-transfer-multi-hca/multi-hca-2gib-cap-utilization.svg)

趋势可以分成两类：

- **200G 总带宽组稳定贴 cap。** \`200_2x100\` 和 \`200_4x50\` 在 bg1/bg10/bg50/bg90 下，2GiB 带宽分别落在约 23.03、20.93、11.64、2.328 GiB/s，换算后几乎等于 198/180/100/20Gbps 前景 cap。说明 200G 总预算下，不指定均分的 multi-HCA 单逻辑流足以达到限速目标。
- **400G 总带宽组在低背景档受单逻辑流上限影响。** \`400_2x200\` 和 \`400_4x100\` 在 bg1/bg10 下都只在约 23.7/21.6 GiB/s，约等于 204/185-186Gbps，而不是 396/360Gbps。bg50 时也只有约 16.2 GiB/s，约 139-140Gbps，不到 200Gbps 前景 cap；bg90 时前景 cap 只有 40Gbps，两个 400G profile 都能贴住。
- **4x50 与 2x100 在 200G 总预算下几乎重合。** 这说明本实验里“更多 HCA 名称”本身没有自动带来超出总前景 cap 的收益；当总 cap 一样且不超过单逻辑流可承载区间时，2 张和 4 张 HCA 的宏观结果接近。
- **2x200 与 4x100 在 multi-HCA 下也几乎重合。** 在 400G 总预算里，两张 200G 和四张 100G 的差异小于 single logical multi-HCA 的调度上限影响。换句话说，瓶颈更像是单逻辑流 multi-HCA 聚合行为，而不是某个具体 HCA 组合。

## RDMA 接收监控验证

driver log 中每个 run 都有创建 \`raw/rdma-rcv-monitor.csv\` 的命令，监控内容设计为 \`ts,dev,rcv_Gbps\`。但是当前本地结果目录没有这些 CSV 文件，见 [rdma-monitor-inventory.csv](figures/kv-transfer-multi-hca/rdma-monitor-inventory.csv)。

因此，本报告可以确认两件事：

1. 前景和背景 Mooncake 命令里的 \`--rate-limit-gbps\` 与实验设计一致，见 [run-manifest.csv](figures/kv-transfer-multi-hca/run-manifest.csv)。
2. 前景 raw samples 与 \`aggregated-summary.csv\` 均显示传输成功，汇总 error 为 0。

但本报告不能完成“用接收侧 \`rdma-rcv-monitor.csv\` 实测背景流量是否到位”的验证。若要补齐这一项，需要从实验机器同步每个 run 的 \`raw/rdma-rcv-monitor.csv\` 和 \`raw/rdma-rcv-monitor.err\`。

## 明显异常点

异常点明细见 [anomalies.csv](figures/kv-transfer-multi-hca/anomalies.csv)。本批最明显的异常不是错误码，而是 400G multi-HCA 单逻辑流没有贴住低背景档前景 cap：

- \`400_2x200_bg1_multi_hca_moonbg\`: 2GiB p50 带宽约 23.765 GiB/s，换算约 204.1Gbps，只是 396Gbps 前景 cap 的 51.5%。
- \`400_4x100_bg1_multi_hca_moonbg\`: 2GiB p50 带宽约 23.720 GiB/s，换算约 203.7Gbps，只是 396Gbps 前景 cap 的 51.5%。
- \`400_2x200_bg50_multi_hca_moonbg\` 和 \`400_4x100_bg50_multi_hca_moonbg\`: 2GiB p50 带宽约 16.2 GiB/s，约 139-140Gbps，只达到 200Gbps 前景 cap 的约 70%。

可能原因包括：

1. Mooncake multi-HCA 对单个逻辑 transfer 的内部调度没有像 manual shard 那样并行打满多张 HCA。
2. 单逻辑前景流与单逻辑背景流同时跨 HCA 时，内部路径选择或队列竞争导致 bg50 这种“前景/背景都很重”的档位没有线性平分。
3. 本地缺失接收侧 RDMA monitor，无法确认背景流实际占用是否完全达到命令 cap；这会影响对 bg50 异常的归因强度。

## Caveats

- \`bandwidth_GBps_p50\` 沿用原始汇总列名。报告中的 GiB/s 文案按原始计算方式解释；换算 cap utilization 时使用 \`bandwidth_GBps_p50 * 8.589934592\` 转为 Gbps。
- multi-HCA 是 **单逻辑流跨多 HCA 名称**，不是多 shard；本报告不会把 multi-HCA 结果描述成多个 foreground shard 的聚合。
- split 对照只覆盖 4x100；2x100、4x50、2x200 没有对应 manual split 对照。
- 当前本地结果包没有 \`raw/rdma-rcv-monitor.csv\`，所以背景流“实测到位”未完成验证；只验证了命令限速设置和前景样本结果。
- 每组 20 次 repeat，报告重点看 p50/p90/p99；若要判断自然调度稳定性，建议追加多次独立 run 或不同 start offset。

## 简短结论

在这批结果里，multi-HCA 不指定均分 **弱于** 4x100 手动均分 split，尤其在 bg1/bg10/bg50 这类前景剩余带宽较高的档位；它没有表现出等价于 4 条手动 shard 的聚合能力。multi-HCA 在 200G 总带宽配置下能稳定贴住前景 cap，在 400G 总带宽配置下低背景档接近约 200Gbps 上限，bg50 约 140Gbps，只有 bg90 这种前景 cap 很低的场景与 split 收敛。

因此，如果目标是可预测地吃满 4x100 的前景剩余带宽，当前数据支持继续使用 manual split；如果目标是简化配置且前景需求不超过约 200Gbps，multi-HCA 单逻辑流是可用的。

## 产物索引

- [run-manifest.csv](figures/kv-transfer-multi-hca/run-manifest.csv)
- [all-aggregated-summary-with-metadata.csv](figures/kv-transfer-multi-hca/all-aggregated-summary-with-metadata.csv)
- [key-results-512mib-1gib-2gib.csv](figures/kv-transfer-multi-hca/key-results-512mib-1gib-2gib.csv)
- [compare-4x100-split-vs-multi-hca.csv](figures/kv-transfer-multi-hca/compare-4x100-split-vs-multi-hca.csv)
- [multi-hca-2gib-trends.csv](figures/kv-transfer-multi-hca/multi-hca-2gib-trends.csv)
- [rdma-monitor-inventory.csv](figures/kv-transfer-multi-hca/rdma-monitor-inventory.csv)
- [anomalies.csv](figures/kv-transfer-multi-hca/anomalies.csv)
- [chart-map.csv](figures/kv-transfer-multi-hca/chart-map.csv)
- [compare-4x100-2gib-bandwidth.svg](figures/kv-transfer-multi-hca/compare-4x100-2gib-bandwidth.svg)
- [compare-4x100-2gib-latency.svg](figures/kv-transfer-multi-hca/compare-4x100-2gib-latency.svg)
- [multi-hca-2gib-bandwidth.svg](figures/kv-transfer-multi-hca/multi-hca-2gib-bandwidth.svg)
- [multi-hca-2gib-cap-utilization.svg](figures/kv-transfer-multi-hca/multi-hca-2gib-cap-utilization.svg)
`;

  await fs.writeFile(REPORT_PATH, report, "utf8");
  console.log(`wrote ${path.relative(ROOT, REPORT_PATH)}`);
  console.log(`wrote ${path.relative(ROOT, OUT_DIR)}`);
}

await main();
