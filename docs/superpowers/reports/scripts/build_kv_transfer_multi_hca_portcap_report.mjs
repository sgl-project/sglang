#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SCRIPT_DIR, "../../../..");
const DATA_DIR = path.join(ROOT, "kv_portcap_multi_hca");
const PRIOR_DIR = path.join(ROOT, "kv_muti_hca_unaverage");
const REPORT_DIR = path.join(ROOT, "docs/superpowers/reports");
const OUT_DIR = path.join(REPORT_DIR, "figures/kv-transfer-multi-hca-portcap");
const REPORT_PATH = path.join(REPORT_DIR, "2026-06-25-kv-transfer-multi-hca-portcap-report.md");
const DRIVER_LOG = path.join(DATA_DIR, "multi-hca-portcap-bg.log");

const KEY_BYTES = new Set(["536870912", "1073741824", "2147483648"]);
const KEY_SIZE_ORDER = ["512.00MiB", "1.00GiB", "2.00GiB"];
const BG_ORDER = [1, 10, 50, 90];
const CONFIG_ORDER = ["2x100", "4x50", "4x100", "2x200"];
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
  blue: { base: "#A3BEFA", dark: "#2E4780" },
  gold: { base: "#FFE15B", dark: "#736422" },
  orange: { base: "#F0986E", dark: "#804126" },
  olive: { base: "#A3D576", dark: "#386411" },
  pink: { base: "#F390CA", dark: "#8A3A6F" },
  neutral: { base: "#C5CAD3", dark: "#464C55" },
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
  const match = run.match(/^(\d+)_(\d+)x(\d+)_bg(\d+)_multi_hca_portcap_moonbg$/);
  if (!match) return null;
  const [, total, lanes, perPort, bg] = match;
  const hcaConfig = `${lanes}x${perPort}`;
  const hcaCount = Number(lanes);
  const perPortTargetGbps = Number(perPort);
  return {
    run,
    total_bandwidth_gbps: Number(total),
    hca_config: hcaConfig,
    hca_count: hcaCount,
    per_port_target_gbps: perPortTargetGbps,
    physical_port_capacity_gbps: 200,
    bg_percent: Number(bg),
    bg_label: `bg${bg}`,
    expected_capfill_per_port_gbps: Math.max(0, 200 - perPortTargetGbps),
    expected_capfill_total_gbps: Math.max(0, 200 - perPortTargetGbps) * hcaCount,
  };
}

function expectedRates(meta) {
  const expected_bg_limit_gbps = meta.total_bandwidth_gbps * meta.bg_percent / 100;
  const expected_fg_limit_gbps = meta.total_bandwidth_gbps - expected_bg_limit_gbps;
  return { expected_bg_limit_gbps, expected_fg_limit_gbps };
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function round(value, digits = 3) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(value).toFixed(digits);
}

function pct(value, digits = 1) {
  if (value === "" || value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return `${Number(value).toFixed(digits)}%`;
}

function extractDriverCommands(driverText, run) {
  const lines = driverText
    .split(/\r?\n/)
    .filter((line) => line.includes(`/auto/${run}/raw/`) && line.includes("kv_transfer_latency.py --role initiator"));

  const records = lines.map((line) => {
    const flow = line.match(/--flow-id\s+([^'\s]+)/)?.[1] ?? "";
    const rate = Number(line.match(/--rate-limit-gbps\s+([0-9.]+)/)?.[1] ?? NaN);
    const ibDevice = line.match(/--ib-device\s+([^'\s]+)/)?.[1] ?? "";
    const sizes = line.match(/--sizes\s+([^'\s]+)/)?.[1] ?? "";
    let kind = "other";
    if (flow.startsWith("capfill-")) kind = "capfill";
    if (flow.startsWith("bg-")) kind = "background";
    if (flow.startsWith("fg-")) kind = "foreground";
    return { kind, flow, rate, ibDevice, sizes, line };
  });

  return {
    records,
    fg: records.filter((record) => record.kind === "foreground"),
    bg: records.filter((record) => record.kind === "background"),
    capfill: records.filter((record) => record.kind === "capfill"),
    monitor_setup_in_driver_log: driverText.includes(`/auto/${run}/raw/rdma-rcv-monitor.csv`),
  };
}

async function readLogTopology(filePath) {
  if (!(await fileExists(filePath))) return "";
  const text = await fs.readFile(filePath, "utf8");
  const match = text.match(/Topology discovery complete\. Found\s+(\d+)\s+HCAs\./);
  return match?.[1] ?? "";
}

async function rawStats(runDir) {
  const rawDir = path.join(runDir, "raw");
  const entries = await fs.readdir(rawDir, { withFileTypes: true }).catch(() => []);
  const sampleFiles = entries
    .filter((entry) => entry.isFile() && entry.name.startsWith("shard-") && entry.name.endsWith("-samples.jsonl"))
    .map((entry) => path.join(rawDir, entry.name));
  const capfillLogs = entries.filter((entry) => entry.isFile() && entry.name.startsWith("capfill-init-") && entry.name.endsWith(".log"));
  const bgLogs = entries.filter((entry) => entry.isFile() && entry.name.startsWith("bgmoon-init-") && entry.name.endsWith(".log"));
  const fgLogs = entries.filter((entry) => entry.isFile() && entry.name.startsWith("init-") && entry.name.endsWith("-dense.log"));

  let sample_count = 0;
  let ret_error_count = 0;
  const rateLimits = new Set();
  const flowIds = new Set();

  for (const file of sampleFiles) {
    const lines = (await fs.readFile(file, "utf8")).split(/\r?\n/).filter(Boolean);
    sample_count += lines.length;
    for (const line of lines) {
      const record = JSON.parse(line);
      flowIds.add(record.flow_id);
      rateLimits.add(String(record.rate_limit_gbps));
      if (record.ret !== 0) ret_error_count += 1;
    }
  }

  const capfillTopologies = [];
  for (const entry of capfillLogs) {
    const topology = await readLogTopology(path.join(rawDir, entry.name));
    capfillTopologies.push(`${entry.name}:${topology}`);
  }
  const bgTopologies = [];
  for (const entry of bgLogs) {
    const topology = await readLogTopology(path.join(rawDir, entry.name));
    bgTopologies.push(`${entry.name}:${topology}`);
  }
  const fgTopologies = [];
  for (const entry of fgLogs) {
    const topology = await readLogTopology(path.join(rawDir, entry.name));
    fgTopologies.push(`${entry.name}:${topology}`);
  }

  return {
    sample_files: sampleFiles.length,
    sample_count,
    ret_error_count,
    raw_fg_rate_limit_gbps_values: [...rateLimits].sort((a, b) => Number(a) - Number(b)).join(";"),
    raw_fg_flow_ids: [...flowIds].sort().join(";"),
    capfill_log_count: capfillLogs.length,
    bg_log_count: bgLogs.length,
    fg_log_count: fgLogs.length,
    capfill_log_topology_hcas: capfillTopologies.sort().join(";"),
    bg_log_topology_hcas: bgTopologies.sort().join(";"),
    fg_log_topology_hcas: fgTopologies.sort().join(";"),
  };
}

function mdEscape(text) {
  return String(text).replaceAll("|", "\\|");
}

function markdownTable(rows, columns) {
  const header = `| ${columns.map((column) => mdEscape(column.label)).join(" | ")} |`;
  const separator = `| ${columns.map((column) => column.align === "right" ? "---:" : "---").join(" | ")} |`;
  const body = rows.map((row) => (
    `| ${columns.map((column) => mdEscape(column.format ? column.format(row[column.key], row) : row[column.key])).join(" | ")} |`
  ));
  return [header, separator, ...body].join("\n");
}

function metricCompact(rows, sizeLabel) {
  const row = rows.find((item) => item.human_bytes === sizeLabel);
  if (!row) return "";
  return `${round(row.latency_ms_p50)}/${round(row.latency_ms_p90)}/${round(row.latency_ms_p99)} ms, ${round(row.bandwidth_GBps_p50)} GiB/s`;
}

function valueBySize(rows, sizeLabel, key) {
  return rows.find((item) => item.human_bytes === sizeLabel)?.[key] ?? "";
}

function bandwidthFromLatencyGiBps(sizeBytes, latencyMs) {
  const gib = Number(sizeBytes) / 1024 / 1024 / 1024;
  return gib / (Number(latencyMs) / 1000);
}

function escapeXml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
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

function lineChartSvg({ title, subtitle, series, xLabels, yLabel, yDomain = null, logY = false }) {
  const width = 980;
  const height = 540;
  const margin = { top: 118, right: 52, bottom: 78, left: 92 };
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
    return `<line x1="${margin.left}" x2="${margin.left + plotW}" y1="${yy}" y2="${yy}" stroke="${TOKENS.grid}" stroke-width="1" />\n${svgText(margin.left - 12, yy + 4, formatAxis(tick), `text-anchor="end" class="tick"`)}`;
  }).join("\n");
  const xAxis = xLabels.map((label, index) => {
    const xx = x(index);
    return `<line x1="${xx}" x2="${xx}" y1="${margin.top + plotH}" y2="${margin.top + plotH + 6}" stroke="${TOKENS.axis}" />\n${svgText(xx, margin.top + plotH + 28, label, `text-anchor="middle" class="tick"`)}`;
  }).join("\n");
  const paths = series.map((item) => {
    const d = item.values.map((value, index) => `${index === 0 ? "M" : "L"} ${x(index).toFixed(2)} ${y(value).toFixed(2)}`).join(" ");
    const points = item.values.map((value, index) => `<circle cx="${x(index).toFixed(2)}" cy="${y(value).toFixed(2)}" r="4.5" fill="${item.color}" stroke="${item.stroke}" stroke-width="1.2"><title>${escapeXml(item.label)} ${xLabels[index]}: ${round(value)}</title></circle>`).join("\n");
    return `<path d="${d}" fill="none" stroke="${item.stroke}" stroke-width="2.2" ${item.dash ? `stroke-dasharray="${item.dash}"` : ""} />\n${points}`;
  }).join("\n");
  let legendX = margin.left;
  const legendY = 78;
  const legend = series.map((item) => {
    const piece = `<line x1="${legendX}" x2="${legendX + 28}" y1="${legendY}" y2="${legendY}" stroke="${item.stroke}" stroke-width="2.2" ${item.dash ? `stroke-dasharray="${item.dash}"` : ""} />\n<circle cx="${legendX + 14}" cy="${legendY}" r="4.2" fill="${item.color}" stroke="${item.stroke}" stroke-width="1" />\n${svgText(legendX + 36, legendY + 4, item.label, `class="legend"`)}`;
    legendX += item.label.length * 7.3 + 48;
    return piece;
  }).join("\n");
  const subtitleSvg = wrap(subtitle, 132).map((line, index) => svgText(margin.left, 38 + index * 17, line, `class="subtitle"`)).join("\n");

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

async function loadPriorSplitRows() {
  const rows = [];
  for (const bg of BG_ORDER) {
    const run = `400_4x100_bg${bg}_cap100_moonbg_split`;
    const file = path.join(PRIOR_DIR, run, "aggregated-summary.csv");
    if (!(await fileExists(file))) continue;
    for (const row of parseCsv(await fs.readFile(file, "utf8"))) {
      if (!KEY_BYTES.has(row.bytes)) continue;
      rows.push({
        run,
        bg_label: `bg${bg}`,
        bg_percent: bg,
        human_bytes: row.human_bytes,
        latency_ms_p50: Number(row.latency_ms_p50),
        latency_ms_p90: Number(row.latency_ms_p90),
        latency_ms_p99: Number(row.latency_ms_p99),
        bandwidth_GBps_p50: Number(row.bandwidth_GBps_p50),
      });
    }
  }
  return rows;
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const driverText = await fs.readFile(DRIVER_LOG, "utf8");
  const entries = await fs.readdir(DATA_DIR, { withFileTypes: true });
  const runDirs = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => ({ run: entry.name, meta: parseRunName(entry.name) }))
    .filter((item) => item.meta)
    .sort((a, b) => CONFIG_ORDER.indexOf(a.meta.hca_config) - CONFIG_ORDER.indexOf(b.meta.hca_config) || a.meta.bg_percent - b.meta.bg_percent);

  const allRows = [];
  const manifest = [];
  const monitorInventory = [];

  for (const { run, meta } of runDirs) {
    const runDir = path.join(DATA_DIR, run);
    const rawDir = path.join(runDir, "raw");
    const summaryRows = parseCsv(await fs.readFile(path.join(runDir, "aggregated-summary.csv"), "utf8"));
    const rates = expectedRates(meta);
    const driver = extractDriverCommands(driverText, run);
    const raw = await rawStats(runDir);
    const monitorPath = path.join(rawDir, "rdma-rcv-monitor.csv");
    const monitorPresent = await fileExists(monitorPath);
    const totalErrors = summaryRows.reduce((sum, row) => sum + Number(row.error_count), 0);
    const fgRates = [...new Set(driver.fg.map((record) => record.rate).filter(Number.isFinite))].sort((a, b) => a - b);
    const bgRates = [...new Set(driver.bg.map((record) => record.rate).filter(Number.isFinite))].sort((a, b) => a - b);
    const capfillRates = [...new Set(driver.capfill.map((record) => record.rate).filter(Number.isFinite))].sort((a, b) => a - b);
    const ibDevices = [...new Set([...driver.fg, ...driver.bg].map((record) => record.ibDevice).filter(Boolean))].sort();

    manifest.push({
      run,
      hca_config: meta.hca_config,
      hca_count: meta.hca_count,
      total_bandwidth_gbps: meta.total_bandwidth_gbps,
      bg_label: meta.bg_label,
      expected_capfill_per_port_gbps: round(meta.expected_capfill_per_port_gbps, 1),
      expected_capfill_total_gbps: round(meta.expected_capfill_total_gbps, 1),
      expected_bg_limit_gbps: round(rates.expected_bg_limit_gbps, 1),
      expected_fg_limit_gbps: round(rates.expected_fg_limit_gbps, 1),
      driver_capfill_flow_count: driver.capfill.length,
      driver_capfill_rate_limit_gbps_values: capfillRates.join(";"),
      driver_bg_rate_limit_gbps_values: bgRates.join(";"),
      driver_fg_rate_limit_gbps_values: fgRates.join(";"),
      ib_device_values: ibDevices.join(";"),
      aggregated_rows: summaryRows.length,
      aggregated_error_count_sum: totalErrors,
      raw_sample_files: raw.sample_files,
      raw_sample_count: raw.sample_count,
      raw_sample_ret_error_count: raw.ret_error_count,
      raw_fg_rate_limit_gbps_values: raw.raw_fg_rate_limit_gbps_values,
      capfill_log_count: raw.capfill_log_count,
      bg_log_count: raw.bg_log_count,
      fg_log_count: raw.fg_log_count,
      capfill_log_topology_hcas: raw.capfill_log_topology_hcas,
      bg_log_topology_hcas: raw.bg_log_topology_hcas,
      fg_log_topology_hcas: raw.fg_log_topology_hcas,
      rdma_monitor_present: monitorPresent ? "yes" : "no",
      monitor_setup_in_driver_log: driver.monitor_setup_in_driver_log ? "yes" : "no",
    });

    monitorInventory.push({
      run,
      hca_config: meta.hca_config,
      bg_label: meta.bg_label,
      expected_capfill_total_gbps: round(meta.expected_capfill_total_gbps, 1),
      expected_background_gbps: round(rates.expected_bg_limit_gbps, 1),
      monitor_setup_in_driver_log: driver.monitor_setup_in_driver_log ? "yes" : "no",
      local_raw_rdma_monitor_csv_present: monitorPresent ? "yes" : "no",
      local_raw_rdma_monitor_csv_path: path.relative(ROOT, monitorPath).replaceAll("\\", "/"),
      validation_status: monitorPresent
        ? "receiver monitor available"
        : "missing from local result bundle; use driver/raw logs for setup validation only",
    });

    for (const row of summaryRows) {
      const bw = Number(row.bandwidth_GBps_p50);
      const gbps = bw * GIBPS_TO_GBPS;
      const calculatedBw = bandwidthFromLatencyGiBps(row.bytes, row.latency_ms_p50);
      const calculatedGbps = calculatedBw * GIBPS_TO_GBPS;
      allRows.push({
        run,
        hca_config: meta.hca_config,
        hca_count: meta.hca_count,
        total_bandwidth_gbps: meta.total_bandwidth_gbps,
        per_port_target_gbps: meta.per_port_target_gbps,
        bg_label: meta.bg_label,
        bg_percent: meta.bg_percent,
        expected_capfill_per_port_gbps: round(meta.expected_capfill_per_port_gbps, 1),
        expected_capfill_total_gbps: round(meta.expected_capfill_total_gbps, 1),
        expected_bg_limit_gbps: round(rates.expected_bg_limit_gbps, 1),
        expected_fg_limit_gbps: round(rates.expected_fg_limit_gbps, 1),
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
        bandwidth_gbps_p50_binary: gbps,
        calculated_bandwidth_GiBps_from_p50_latency: calculatedBw,
        calculated_bandwidth_gbps_binary_from_p50_latency: calculatedGbps,
        foreground_cap_utilization_pct: calculatedGbps / rates.expected_fg_limit_gbps * 100,
      });
    }
  }

  const keyRows = allRows.filter((row) => KEY_BYTES.has(String(row.size_bytes)));
  const perExperimentCalculatedRows = allRows.map((row) => ({
    run: row.run,
    hca_config: row.hca_config,
    total_bandwidth_gbps: row.total_bandwidth_gbps,
    bg_label: row.bg_label,
    bg_percent: row.bg_percent,
    expected_capfill_per_port_gbps: row.expected_capfill_per_port_gbps,
    expected_capfill_total_gbps: row.expected_capfill_total_gbps,
    expected_bg_limit_gbps: row.expected_bg_limit_gbps,
    expected_fg_limit_gbps: row.expected_fg_limit_gbps,
    human_bytes: row.human_bytes,
    size_bytes: row.size_bytes,
    latency_ms_p50: round(row.latency_ms_p50),
    latency_ms_p90: round(row.latency_ms_p90),
    latency_ms_p99: round(row.latency_ms_p99),
    calculated_bandwidth_GiBps_from_p50_latency: round(row.calculated_bandwidth_GiBps_from_p50_latency),
    calculated_bandwidth_gbps_binary_from_p50_latency: round(row.calculated_bandwidth_gbps_binary_from_p50_latency, 1),
    reported_bandwidth_GBps_p50: round(row.bandwidth_GBps_p50),
    foreground_cap_utilization_pct: round(row.foreground_cap_utilization_pct, 1),
  }));
  const trend2g = allRows
    .filter((row) => row.human_bytes === "2.00GiB")
    .sort((a, b) => CONFIG_ORDER.indexOf(a.hca_config) - CONFIG_ORDER.indexOf(b.hca_config) || a.bg_percent - b.bg_percent)
    .map((row) => ({
      run: row.run,
      hca_config: row.hca_config,
      total_bandwidth_gbps: row.total_bandwidth_gbps,
      bg_label: row.bg_label,
      bg_percent: row.bg_percent,
      expected_capfill_total_gbps: row.expected_capfill_total_gbps,
      expected_bg_limit_gbps: row.expected_bg_limit_gbps,
      expected_fg_limit_gbps: row.expected_fg_limit_gbps,
      latency_ms_p50: round(row.latency_ms_p50),
      latency_ms_p90: round(row.latency_ms_p90),
      latency_ms_p99: round(row.latency_ms_p99),
      bandwidth_GBps_p50: round(row.bandwidth_GBps_p50),
      bandwidth_gbps_p50_binary: round(row.bandwidth_gbps_p50_binary, 1),
      foreground_cap_utilization_pct: round(row.foreground_cap_utilization_pct, 1),
    }));

  const priorSplitRows = await loadPriorSplitRows();
  const comparePriorSplit = [];
  for (const bg of BG_ORDER) {
    for (const size of KEY_SIZE_ORDER) {
      const split = priorSplitRows.find((row) => row.bg_percent === bg && row.human_bytes === size);
      const portcap = keyRows.find((row) => row.hca_config === "4x100" && row.bg_percent === bg && row.human_bytes === size);
      if (!split || !portcap) continue;
      comparePriorSplit.push({
        bg_label: `bg${bg}`,
        size,
        prior_split_latency_ms_p50: round(split.latency_ms_p50),
        prior_split_latency_ms_p90: round(split.latency_ms_p90),
        prior_split_latency_ms_p99: round(split.latency_ms_p99),
        prior_split_bandwidth_GBps_p50: round(split.bandwidth_GBps_p50),
        portcap_multi_latency_ms_p50: round(portcap.latency_ms_p50),
        portcap_multi_latency_ms_p90: round(portcap.latency_ms_p90),
        portcap_multi_latency_ms_p99: round(portcap.latency_ms_p99),
        portcap_multi_bandwidth_GBps_p50: round(portcap.bandwidth_GBps_p50),
        portcap_vs_split_bandwidth_pct: round((portcap.bandwidth_GBps_p50 / split.bandwidth_GBps_p50 - 1) * 100, 1),
        portcap_vs_split_latency_pct: round((portcap.latency_ms_p50 / split.latency_ms_p50 - 1) * 100, 1),
      });
    }
  }

  const anomalies = allRows
    .filter((row) => row.human_bytes === "2.00GiB")
    .filter((row) => row.foreground_cap_utilization_pct < 90 || row.latency_ms_p99 / row.latency_ms_p50 > 1.15)
    .map((row) => ({
      run: row.run,
      hca_config: row.hca_config,
      bg_label: row.bg_label,
      size: row.human_bytes,
      expected_capfill_total_gbps: row.expected_capfill_total_gbps,
      expected_bg_limit_gbps: row.expected_bg_limit_gbps,
      expected_fg_limit_gbps: row.expected_fg_limit_gbps,
      bandwidth_gbps_p50_binary: round(row.bandwidth_gbps_p50_binary, 1),
      foreground_cap_utilization_pct: round(row.foreground_cap_utilization_pct, 1),
      latency_ms_p50: round(row.latency_ms_p50),
      latency_ms_p99: round(row.latency_ms_p99),
      possible_reason: row.hca_config === "4x50"
        ? "capfill is heavy on every physical port, leaving fragmented 50G-per-port target capacity for one logical multi-HCA flow"
        : row.hca_config === "4x100"
          ? "single logical multi-HCA flow plus capfill does not behave like four manual shards and appears limited below the 400G target"
          : "single logical multi-HCA flow under capfill/background competition does not fully use the configured foreground cap",
    }));

  await fs.writeFile(path.join(OUT_DIR, "run-manifest.csv"), toCsv(manifest, [
    "run", "hca_config", "hca_count", "total_bandwidth_gbps", "bg_label",
    "expected_capfill_per_port_gbps", "expected_capfill_total_gbps", "expected_bg_limit_gbps", "expected_fg_limit_gbps",
    "driver_capfill_flow_count", "driver_capfill_rate_limit_gbps_values", "driver_bg_rate_limit_gbps_values", "driver_fg_rate_limit_gbps_values",
    "ib_device_values", "aggregated_rows", "aggregated_error_count_sum", "raw_sample_files", "raw_sample_count", "raw_sample_ret_error_count",
    "raw_fg_rate_limit_gbps_values", "capfill_log_count", "bg_log_count", "fg_log_count",
    "capfill_log_topology_hcas", "bg_log_topology_hcas", "fg_log_topology_hcas", "rdma_monitor_present", "monitor_setup_in_driver_log",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "all-aggregated-summary-with-metadata.csv"), toCsv(allRows, [
    "run", "hca_config", "hca_count", "total_bandwidth_gbps", "per_port_target_gbps", "bg_label", "bg_percent",
    "expected_capfill_per_port_gbps", "expected_capfill_total_gbps", "expected_bg_limit_gbps", "expected_fg_limit_gbps",
    "size_bytes", "human_bytes", "shard_count", "repeat_count", "error_count",
    "latency_ms_mean", "latency_ms_p50", "latency_ms_p90", "latency_ms_p99", "latency_ms_min", "latency_ms_max",
    "bandwidth_GBps_p50", "bandwidth_GBps_mean", "bandwidth_gbps_p50_binary",
    "calculated_bandwidth_GiBps_from_p50_latency", "calculated_bandwidth_gbps_binary_from_p50_latency", "foreground_cap_utilization_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "per-experiment-calculated-bandwidth.csv"), toCsv(perExperimentCalculatedRows, [
    "run", "hca_config", "total_bandwidth_gbps", "bg_label", "bg_percent",
    "expected_capfill_per_port_gbps", "expected_capfill_total_gbps", "expected_bg_limit_gbps", "expected_fg_limit_gbps",
    "human_bytes", "size_bytes", "latency_ms_p50", "latency_ms_p90", "latency_ms_p99",
    "calculated_bandwidth_GiBps_from_p50_latency", "calculated_bandwidth_gbps_binary_from_p50_latency",
    "reported_bandwidth_GBps_p50", "foreground_cap_utilization_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "key-results-512mib-1gib-2gib.csv"), toCsv(keyRows, [
    "run", "hca_config", "total_bandwidth_gbps", "bg_label", "bg_percent",
    "expected_capfill_total_gbps", "expected_bg_limit_gbps", "expected_fg_limit_gbps", "human_bytes",
    "latency_ms_p50", "latency_ms_p90", "latency_ms_p99", "bandwidth_GBps_p50",
    "bandwidth_gbps_p50_binary", "calculated_bandwidth_GiBps_from_p50_latency",
    "calculated_bandwidth_gbps_binary_from_p50_latency", "foreground_cap_utilization_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "multi-hca-portcap-2gib-trends.csv"), toCsv(trend2g, [
    "run", "hca_config", "total_bandwidth_gbps", "bg_label", "bg_percent", "expected_capfill_total_gbps",
    "expected_bg_limit_gbps", "expected_fg_limit_gbps", "latency_ms_p50", "latency_ms_p90", "latency_ms_p99",
    "bandwidth_GBps_p50", "bandwidth_gbps_p50_binary", "foreground_cap_utilization_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "compare-portcap-4x100-vs-prior-split.csv"), toCsv(comparePriorSplit, [
    "bg_label", "size", "prior_split_latency_ms_p50", "prior_split_latency_ms_p90", "prior_split_latency_ms_p99", "prior_split_bandwidth_GBps_p50",
    "portcap_multi_latency_ms_p50", "portcap_multi_latency_ms_p90", "portcap_multi_latency_ms_p99", "portcap_multi_bandwidth_GBps_p50",
    "portcap_vs_split_bandwidth_pct", "portcap_vs_split_latency_pct",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "rdma-monitor-and-log-validation.csv"), toCsv(monitorInventory, [
    "run", "hca_config", "bg_label", "expected_capfill_total_gbps", "expected_background_gbps",
    "monitor_setup_in_driver_log", "local_raw_rdma_monitor_csv_present", "local_raw_rdma_monitor_csv_path", "validation_status",
  ]));
  await fs.writeFile(path.join(OUT_DIR, "anomalies.csv"), toCsv(anomalies, [
    "run", "hca_config", "bg_label", "size", "expected_capfill_total_gbps", "expected_bg_limit_gbps", "expected_fg_limit_gbps",
    "bandwidth_gbps_p50_binary", "foreground_cap_utilization_pct", "latency_ms_p50", "latency_ms_p99", "possible_reason",
  ]));

  const bgLabels = BG_ORDER.map((bg) => `bg${bg}`);
  const configSeries = CONFIG_ORDER.map((config, index) => {
    const palette = [COLORS.blue, COLORS.gold, COLORS.orange, COLORS.olive][index];
    const rows = allRows.filter((row) => row.human_bytes === "2.00GiB" && row.hca_config === config).sort((a, b) => a.bg_percent - b.bg_percent);
    return {
      label: config,
      color: palette.base,
      stroke: palette.dark,
      dash: index === 0 ? "" : index === 1 ? "7 4" : index === 2 ? "2 4" : "9 3 2 3",
      rows,
    };
  });
  await fs.writeFile(path.join(OUT_DIR, "portcap-2gib-bandwidth.svg"), lineChartSvg({
    title: "portcap multi-HCA 2GiB foreground bandwidth",
    subtitle: "p50 bandwidth_GBps by target profile and background share; all rows are single logical foreground flows, not manual shards.",
    xLabels: bgLabels,
    yLabel: "bandwidth_GBps_p50",
    series: configSeries.map((item) => ({ ...item, values: item.rows.map((row) => row.bandwidth_GBps_p50) })),
  }));
  await fs.writeFile(path.join(OUT_DIR, "portcap-2gib-cap-utilization.svg"), lineChartSvg({
    title: "portcap multi-HCA 2GiB foreground cap utilization",
    subtitle: "bandwidth_GBps_p50 converted to Gbps and divided by foreground rate limit; bg90 converges because foreground cap is low.",
    xLabels: bgLabels,
    yLabel: "foreground cap utilization",
    yDomain: [0, 110],
    series: configSeries.map((item) => ({ ...item, values: item.rows.map((row) => row.foreground_cap_utilization_pct) })),
  }));
  await fs.writeFile(path.join(OUT_DIR, "portcap-2gib-latency.svg"), lineChartSvg({
    title: "portcap multi-HCA 2GiB foreground p50 latency",
    subtitle: "p50 latency in milliseconds; y-axis is log scaled so low and high background cases stay readable.",
    xLabels: bgLabels,
    yLabel: "p50 latency ms (log)",
    logY: true,
    series: configSeries.map((item) => ({ ...item, values: item.rows.map((row) => row.latency_ms_p50) })),
  }));

  const splitSeries = BG_ORDER.map((bg) => priorSplitRows.find((row) => row.bg_percent === bg && row.human_bytes === "2.00GiB")?.bandwidth_GBps_p50 ?? NaN);
  const portcap4x100Series = BG_ORDER.map((bg) => allRows.find((row) => row.hca_config === "4x100" && row.bg_percent === bg && row.human_bytes === "2.00GiB")?.bandwidth_GBps_p50 ?? NaN);
  await fs.writeFile(path.join(OUT_DIR, "portcap-4x100-vs-prior-split-2gib-bandwidth.svg"), lineChartSvg({
    title: "4x100 2GiB bandwidth: portcap multi-HCA vs prior manual split",
    subtitle: "Prior split uses four manual 100G shards; portcap multi-HCA uses one logical foreground flow with capfill on each physical 200G port.",
    xLabels: bgLabels,
    yLabel: "bandwidth_GBps_p50",
    series: [
      { label: "prior manual split", color: COLORS.blue.base, stroke: COLORS.blue.dark, values: splitSeries },
      { label: "portcap multi-HCA", color: COLORS.orange.base, stroke: COLORS.orange.dark, dash: "7 4", values: portcap4x100Series },
    ],
  }));

  const chartMap = [
    { file: "portcap-2gib-bandwidth.svg", section: "multi-HCA portcap trend", question: "How does 2GiB bandwidth change by target profile and background share?", chart_type: "multi-series line", supported_claim: "capfill makes 4x50 and 4x100 substantially weaker at low background; bg90 converges." },
    { file: "portcap-2gib-cap-utilization.svg", section: "cap utilization", question: "Do portcap multi-HCA runs reach the configured foreground cap?", chart_type: "multi-series line", supported_claim: "2x100 partly underuses low-background cap, 4x50/4x100 underuse more, 2x200 remains best but not full 400G." },
    { file: "portcap-2gib-latency.svg", section: "latency trend", question: "How does p50 latency grow as background share increases?", chart_type: "log-scale line", supported_claim: "Latency rises as effective bandwidth falls; all profiles converge when foreground cap is only 20-40Gbps." },
    { file: "portcap-4x100-vs-prior-split-2gib-bandwidth.svg", section: "prior split comparison", question: "How far is portcap multi-HCA from manual split?", chart_type: "two-series line", supported_claim: "4x100 portcap multi-HCA is far below prior split for bg1/bg10/bg50 and only converges at bg90." },
  ];
  await fs.writeFile(path.join(OUT_DIR, "chart-map.csv"), toCsv(chartMap, ["file", "section", "question", "chart_type", "supported_claim"]));

  const wideRows = CONFIG_ORDER.flatMap((config) => (
    BG_ORDER.map((bg) => {
      const rows = keyRows.filter((row) => row.hca_config === config && row.bg_percent === bg).sort((a, b) => KEY_SIZE_ORDER.indexOf(a.human_bytes) - KEY_SIZE_ORDER.indexOf(b.human_bytes));
      return {
        profile: config,
        bg: `bg${bg}`,
        capfill: valueBySize(rows, "2.00GiB", "expected_capfill_total_gbps"),
        fg_cap: valueBySize(rows, "2.00GiB", "expected_fg_limit_gbps"),
        s512: metricCompact(rows, "512.00MiB"),
        s1g: metricCompact(rows, "1.00GiB"),
        s2g: metricCompact(rows, "2.00GiB"),
        util_2g: pct(valueBySize(rows, "2.00GiB", "foreground_cap_utilization_pct")),
      };
    })
  ));
  const trendRowsForReport = trend2g.map((row) => ({
    profile: row.hca_config,
    bg: row.bg_label,
    capfill: row.expected_capfill_total_gbps,
    bg_cap: row.expected_bg_limit_gbps,
    fg_cap: row.expected_fg_limit_gbps,
    p50: row.latency_ms_p50,
    p90: row.latency_ms_p90,
    p99: row.latency_ms_p99,
    bw: row.bandwidth_GBps_p50,
    util: `${row.foreground_cap_utilization_pct}%`,
  }));
  const compareRowsForReport = comparePriorSplit.filter((row) => row.size === "2.00GiB").map((row) => ({
    bg: row.bg_label,
    split: `${row.prior_split_latency_ms_p50}/${row.prior_split_latency_ms_p90}/${row.prior_split_latency_ms_p99} ms, ${row.prior_split_bandwidth_GBps_p50} GiB/s`,
    portcap: `${row.portcap_multi_latency_ms_p50}/${row.portcap_multi_latency_ms_p90}/${row.portcap_multi_latency_ms_p99} ms, ${row.portcap_multi_bandwidth_GBps_p50} GiB/s`,
    bw_delta: `${row.portcap_vs_split_bandwidth_pct}%`,
    latency_delta: `${row.portcap_vs_split_latency_pct}%`,
  }));
  const manifestRowsForReport = manifest.map((row) => ({
    profile: row.hca_config,
    bg: row.bg_label,
    capfill: row.expected_capfill_per_port_gbps,
    capfill_total: row.expected_capfill_total_gbps,
    bg_cap: row.expected_bg_limit_gbps,
    fg_cap: row.expected_fg_limit_gbps,
    capfill_rate: row.driver_capfill_rate_limit_gbps_values || "none",
    bg_rate: row.driver_bg_rate_limit_gbps_values,
    fg_rate: row.driver_fg_rate_limit_gbps_values,
    rows: row.aggregated_rows,
    samples: row.raw_sample_count,
    errors: row.aggregated_error_count_sum,
    monitor: row.rdma_monitor_present,
  }));
  const orderedExperimentKeys = CONFIG_ORDER.flatMap((config) => BG_ORDER.map((bg) => ({ config, bg })));
  const perExperimentSections = orderedExperimentKeys.map(({ config, bg }, index) => {
    const rows = keyRows
      .filter((row) => row.hca_config === config && row.bg_percent === bg)
      .sort((a, b) => KEY_SIZE_ORDER.indexOf(a.human_bytes) - KEY_SIZE_ORDER.indexOf(b.human_bytes));
    const first = rows[0];
    if (!first) return "";
    const runManifest = manifest.find((row) => row.run === first.run);
    const twoG = rows.find((row) => row.human_bytes === "2.00GiB");
    const calculatedRows = rows.map((row) => ({
      size: row.human_bytes,
      latency: `${round(row.latency_ms_p50)}/${round(row.latency_ms_p90)}/${round(row.latency_ms_p99)}`,
      calc_gib: round(row.calculated_bandwidth_GiBps_from_p50_latency),
      calc_gbps: round(row.calculated_bandwidth_gbps_binary_from_p50_latency, 1),
      fg_cap: row.expected_fg_limit_gbps,
      util: pct(row.foreground_cap_utilization_pct),
    }));
    const util = Number(twoG?.foreground_cap_utilization_pct ?? NaN);
    const gap = Number(twoG?.expected_fg_limit_gbps ?? NaN) - Number(twoG?.calculated_bandwidth_gbps_binary_from_p50_latency ?? NaN);
    const status = util >= 98
      ? "2GiB 基本贴住配置的 foreground cap。"
      : util >= 90
        ? "2GiB 接近配置的 foreground cap。"
        : `2GiB 比配置的 foreground cap 低约 ${round(gap, 1)}Gbps。`;
    const capfillText = Number(first.expected_capfill_per_port_gbps) > 0
      ? `每端口 ${first.expected_capfill_per_port_gbps}Gbps，总计 ${first.expected_capfill_total_gbps}Gbps`
      : "无";
    const monitorText = runManifest?.rdma_monitor_present === "yes"
      ? "本地存在 rdma-rcv-monitor.csv"
      : "本地缺少 rdma-rcv-monitor.csv；这里只能用 driver/raw logs 验证命令配置";
    return `### ${index + 1}. ${first.run}

**配置。** profile=${first.hca_config}，total target=${first.total_bandwidth_gbps}Gbps，background=${first.bg_label} (${first.bg_percent}%)，capfill=${capfillText}，background cap=${first.expected_bg_limit_gbps}Gbps，foreground cap=${first.expected_fg_limit_gbps}Gbps。本实验是一个 \`shard_count=1\` 的逻辑 foreground flow，不是手动 shard 拆分。

**计算带宽。** ${twoG ? `2GiB 的 p50 latency 是 ${round(twoG.latency_ms_p50)}ms，因此计算带宽是 ${round(twoG.calculated_bandwidth_GiBps_from_p50_latency)}GiB/s (${round(twoG.calculated_bandwidth_gbps_binary_from_p50_latency, 1)}Gbps)，相当于 foreground cap 的 ${pct(twoG.foreground_cap_utilization_pct)}。${status}` : "缺少 2GiB 结果。"}

${markdownTable(calculatedRows, [
      { key: "size", label: "Size" },
      { key: "latency", label: "p50/p90/p99 延迟 ms" },
      { key: "calc_gib", label: "计算带宽 GiB/s", align: "right" },
      { key: "calc_gbps", label: "计算带宽 Gbps", align: "right" },
      { key: "fg_cap", label: "前景 cap Gbps", align: "right" },
      { key: "util", label: "cap 利用率", align: "right" },
    ])}

**验证备注。** aggregated rows=${runManifest?.aggregated_rows ?? ""}，foreground samples=${runManifest?.raw_sample_count ?? ""}，aggregate errors=${runManifest?.aggregated_error_count_sum ?? ""}，monitor=${monitorText}。`;
  }).filter(Boolean).join("\n\n");

  const report = `# KV Transfer / Mooncake multi-HCA portcap 实验报告

**日期:** 2026-06-25

**数据目录:** \`kv_portcap_multi_hca/\`

**报告产物目录:** \`docs/superpowers/reports/figures/kv-transfer-multi-hca-portcap/\`

## Executive Summary

- **portcap/capfill 后，multi-HCA 单逻辑流仍然不是手动 shard 的替代物。** 4x100/bg1 的 2GiB p50 带宽只有 16.978 GiB/s，约 145.8Gbps；此前 4x100 manual split 对照是 46.036 GiB/s，约 395.4Gbps。bg10/bg50 也分别低 63.4% 和 50.4%，只有 bg90 因前景 cap 只有 40Gbps 而收敛。
- **capfill 让“每端口容量限制”的效果显性化，但也进一步暴露 single logical flow 的带宽利用问题。** 2x100 在 bg1/bg10 只能达到前景 cap 的 84.6%/88.9%；4x50 在 bg1/bg10/bg50 只有 67.0%/67.6%/85.2%；4x100 在 bg1/bg10/bg50 只有 36.8%/36.5%/49.6%。
- **2x200 是本轮 portcap 中表现最好的高带宽配置，但仍未接近 400G 目标。** 2x200/bg1 的 2GiB p50 带宽为 25.353 GiB/s，约 217.8Gbps，只达到 396Gbps 前景 cap 的 55.0%；bg10 为 51.8%，bg50 为 67.1%，bg90 才贴住 40Gbps cap。
- **本地结果包缺少实际 \`rdma-rcv-monitor.csv\`。** driver log 显示每组都创建了 monitor 输出，但同步到本地的 raw 目录没有该 CSV；因此本报告用 driver/raw logs 验证 capfill/background/foreground 的命令限速与 HCA 拓扑发现，不能用接收侧 monitor 实测值证明流量完全到位。

## 实验目的

本次实验验证 KV Transfer / Mooncake 在 **不手动 shard** 时，一个逻辑 foreground transfer flow 使用多个 HCA 的实际表现。重点是：

1. multi-HCA 单逻辑流是否能在多网卡上有效使用带宽；
2. 在 2x100、4x50、4x100、2x200 和 bg1/bg10/bg50/bg90 下，前景 KV transfer 的 latency/bandwidth 如何变化；
3. 与此前 4x100 manual split shard 对照相比，multi-HCA 不指定均分是否仍有差距。

## 方法：portcap/capfill 如何模拟每端口限制

物理上每张 \`mlx5_bond_*\` 按 200Gbps 能力理解。为了模拟目标配置里的 per-port cap，脚本先在每张物理端口上启动一个 capfill Mooncake 背景流，占用掉多余容量：

| Profile | Physical port assumption | capfill per port | Simulated remaining capacity |
|---|---:|---:|---:|
| \`2x100\` | 2 x 200Gbps | 100Gbps | 2 x 100Gbps |
| \`4x50\` | 4 x 200Gbps | 150Gbps | 4 x 50Gbps |
| \`4x100\` | 4 x 200Gbps | 100Gbps | 4 x 100Gbps |
| \`2x200\` | 2 x 200Gbps | 0Gbps | 2 x 200Gbps |

每组实验随后启动一个 background multi-HCA flow 和一个 foreground multi-HCA flow。二者都用逗号分隔的 \`--ib-device\`，例如 4x100 使用 \`mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3\`。foreground **仍然是一个逻辑流**，没有把 2GB 逻辑数据手动拆成多个 shard。

背景比例 \`bg1/bg10/bg50/bg90\` 按目标总带宽计算：

\`\`\`text
background_limit_gbps = total_target_gbps * bg_percent
foreground_limit_gbps = total_target_gbps - background_limit_gbps
\`\`\`

前景完整跑 21 个 size：\`1MB,2MB,4MB,8MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,384MB,512MB,768MB,1GB,1.25GB,1.5GB,1.75GB,2GB\`。

## 数据完整性与命令验证

所有 16 个 run 都有 \`aggregated-summary.csv\`，每个 21 行，汇总 \`error_count=0\`。每组有一个 foreground samples 文件，样本数为 420，raw sample 的 \`ret\` 没有错误。capfill/background/foreground 的限速命令从 \`multi-hca-portcap-bg.log\` 解析；raw log 也显示 capfill flow 发现 1 个 HCA，background/foreground flow 发现对应的 2 或 4 个 HCA。

${markdownTable(manifestRowsForReport, [
    { key: "profile", label: "Profile" },
    { key: "bg", label: "BG" },
    { key: "capfill", label: "capfill/port Gbps", align: "right" },
    { key: "capfill_total", label: "capfill total Gbps", align: "right" },
    { key: "bg_cap", label: "BG cap Gbps", align: "right" },
    { key: "fg_cap", label: "FG cap Gbps", align: "right" },
    { key: "capfill_rate", label: "capfill rate-limit" },
    { key: "bg_rate", label: "BG rate-limit" },
    { key: "fg_rate", label: "FG rate-limit" },
    { key: "rows", label: "Rows", align: "right" },
    { key: "samples", label: "FG samples", align: "right" },
    { key: "errors", label: "Errors", align: "right" },
    { key: "monitor", label: "local monitor CSV" },
  ])}

## 逐实验结果与计算带宽

下面按 16 个实验逐个展开，不再把所有结果挤在一张总表里。报告正文展示 512MiB、1GiB、2GiB 三个关键 size；完整 21 个 size 的逐实验计算结果见 [per-experiment-calculated-bandwidth.csv](figures/kv-transfer-multi-hca-portcap/per-experiment-calculated-bandwidth.csv)，关键 size 长表见 [key-results-512mib-1gib-2gib.csv](figures/kv-transfer-multi-hca-portcap/key-results-512mib-1gib-2gib.csv)。

计算带宽统一按 p50 latency 反推：

\`\`\`text
calculated_bandwidth_GiBps = size_GiB / (latency_ms_p50 / 1000)
calculated_bandwidth_Gbps = calculated_bandwidth_GiBps * 8.589934592
cap_utilization = calculated_bandwidth_Gbps / foreground_limit_gbps
\`\`\`

${perExperimentSections}

## 2GiB 趋势：capfill 后 single logical flow 更难吃满

![portcap 2GiB bandwidth](figures/kv-transfer-multi-hca-portcap/portcap-2gib-bandwidth.svg)

2GiB 是最能反映大块 KV transfer 稳态吞吐的点。结果显示，随着背景占比上升，所有 profile 的带宽都下降；但下降不是只由 foreground cap 决定，profile 本身和 capfill 方式也强烈影响单逻辑流能拿到的有效带宽。

${markdownTable(trendRowsForReport, [
    { key: "profile", label: "Profile" },
    { key: "bg", label: "BG" },
    { key: "capfill", label: "capfill total", align: "right" },
    { key: "bg_cap", label: "BG cap", align: "right" },
    { key: "fg_cap", label: "FG cap", align: "right" },
    { key: "p50", label: "p50 ms", align: "right" },
    { key: "p90", label: "p90 ms", align: "right" },
    { key: "p99", label: "p99 ms", align: "right" },
    { key: "bw", label: "bw GiB/s", align: "right" },
    { key: "util", label: "cap util", align: "right" },
  ])}

![portcap 2GiB utilization](figures/kv-transfer-multi-hca-portcap/portcap-2gib-cap-utilization.svg)

按 profile 看：

- **2x100:** bg1/bg10 只能达到 198/180Gbps 前景 cap 的 84.6%/88.9%，bg50/bg90 则贴近 100%。这说明 capfill 每卡 100Gbps 后，单逻辑 multi-HCA 在高剩余带宽档已经有损耗，但在 100Gbps 或 20Gbps 前景 cap 下仍可控。
- **4x50:** bg1/bg10/bg50 分别只有 67.0%/67.6%/85.2% cap utilization。每张物理口先用 capfill 占 150Gbps，只留下 50Gbps/口的碎片化容量；一个逻辑 flow 跨 4 张卡时没有像 4 条 shard 那样稳定聚合这些剩余容量。
- **4x100:** bg1/bg10/bg50 分别只有 36.8%/36.5%/49.6%，是本轮最能说明问题的配置。它名义上和 prior manual split 一样有 4 x 100Gbps 目标前景空间，但 portcap single logical flow 只拿到约 146/132/99Gbps。
- **2x200:** 没有 capfill，因此表现最好；bg1 达到约 217.8Gbps，bg10 约 186.6Gbps，bg50 约 134.1Gbps，bg90 贴住 40Gbps cap。即便如此，bg1/bg10 仍远低于 396/360Gbps 前景 cap。

![portcap 2GiB latency](figures/kv-transfer-multi-hca-portcap/portcap-2gib-latency.svg)

latency 与有效带宽基本反向变化：4x50/bg1 的 2GiB p50 latency 是 129.470ms，高于 2x100/bg1 的 102.562ms；4x100/bg1 是 117.800ms；2x200/bg1 最低，为 78.886ms。bg90 时因为前景 cap 很低，2x100/4x50 都约 859ms，4x100/2x200 都约 429.6ms，符合 20Gbps vs 40Gbps 前景 cap 的量级差异。

## 与此前 manual split 的差距

![4x100 portcap vs prior split](figures/kv-transfer-multi-hca-portcap/portcap-4x100-vs-prior-split-2gib-bandwidth.svg)

此前 4x100 manual split 是 4 条 foreground shard，每条固定绑定一张 HCA；本轮 portcap multi-HCA 是一个 foreground flow 跨 4 张 HCA 名称。两者不能视作同一执行模型。

${markdownTable(compareRowsForReport, [
    { key: "bg", label: "BG" },
    { key: "split", label: "prior manual split 2GiB p50/p90/p99, bw" },
    { key: "portcap", label: "portcap multi-HCA 2GiB p50/p90/p99, bw" },
    { key: "bw_delta", label: "portcap bw vs split", align: "right" },
    { key: "latency_delta", label: "portcap p50 latency vs split", align: "right" },
  ])}

4x100/bg1 下，portcap multi-HCA 的 2GiB p50 带宽比 manual split 低 63.1%，p50 latency 高 171.2%。bg10 低 63.4%，latency 高 173.2%；bg50 低 50.4%，latency 高 101.6%。bg90 收敛，是因为两边前景 cap 都只有 40Gbps，single logical flow 的内部聚合上限不再是主要瓶颈。

## capfill/background 流量验证

本地 raw 目录没有实际 \`rdma-rcv-monitor.csv\`，但总 driver log 中每个 run 都有创建 monitor 的命令，见 [rdma-monitor-and-log-validation.csv](figures/kv-transfer-multi-hca-portcap/rdma-monitor-and-log-validation.csv)。因此本轮能验证的是：

1. **capfill 命令限速符合设计。** 2x100 每个 capfill flow 是 100Gbps；4x50 每个 capfill flow 是 150Gbps；4x100 每个 capfill flow 是 100Gbps；2x200 没有 capfill flow。
2. **background/foreground 命令限速符合设计。** 例如 2x100/bg10 是 background 20Gbps、foreground 180Gbps；400G/bg50 是 background 200Gbps、foreground 200Gbps。
3. **raw log 拓扑发现符合执行模型。** capfill log 发现 1 个 HCA；background/foreground log 发现对应的 2 或 4 个 HCA。

但因为缺少本地 monitor CSV，本报告不能用接收侧 counter 证明 capfill/background 实际持续达到了命令 cap。若要做端口级定论，需要补传每个 run 的 \`raw/rdma-rcv-monitor.csv\` 和 \`raw/rdma-rcv-monitor.err\`。

## 异常点与可能原因

异常明细见 [anomalies.csv](figures/kv-transfer-multi-hca-portcap/anomalies.csv)。最重要的异常是：多个低背景或中背景档没有达到 foreground cap。

- **4x100/bg1 和 bg10:** 只有约 145.8Gbps / 131.6Gbps，有效利用率 36.8% / 36.5%。这比上一轮无 portcap 的 4x100 multi-HCA 还低，说明 capfill 与 foreground/background 的并发可能进一步挤压了单逻辑流调度。
- **4x50/bg1 和 bg10:** 有效利用率约 67%。capfill 每口 150Gbps 后，只剩 50Gbps/口；单逻辑流没有稳定拼出 4 x 50Gbps。
- **2x100/bg1 和 bg10:** 虽然比 4x50 好，但仍只有 84.6%/88.9%。这说明 capfill 竞争会影响 single logical multi-HCA 的低背景带宽，即使总目标只有 200Gbps。
- **2x200/bg1:** 无 capfill 也只有 55.0% cap utilization，延续了此前 400G multi-HCA 单逻辑流存在约 200Gbps 级别上限的趋势。

可能原因包括：

1. Mooncake multi-HCA 的单逻辑 transfer 没有按端口剩余容量做等价均分，和手动 shard 的并行聚合模型不同。
2. capfill/background/foreground 都是 Mooncake flow 时，内部路径选择、队列竞争或连接级调度可能让 foreground 无法稳定拿到理论剩余容量。
3. 端口级 monitor CSV 缺失，无法判断实际流量是否集中在少数 HCA，或者 capfill 是否对每张卡都稳定打满。

## Caveats

- \`bandwidth_GBps_p50\` 沿用原始汇总列名；报告中的 GiB/s 按原始列解释，换算 cap utilization 时使用 \`bandwidth_GBps_p50 * 8.589934592\`。
- 本轮结果是 **multi-HCA 单逻辑 foreground flow**，不是手动 shard，不能把 \`shard_count=1\` 的结果解读成多个 foreground shard 的聚合。
- prior manual split 对照来自 \`kv_muti_hca_unaverage/400_4x100_bg*_cap100_moonbg_split\`，执行模型不同，因此只用于说明“手动 shard 能达到的上界/差距”，不是严格同一脚本条件。
- 本地结果包缺失 \`raw/rdma-rcv-monitor.csv\`；报告只能验证命令、raw samples 和 raw log 拓扑，不能验证接收侧端口实测流量。

## 简短结论

portcap/capfill 实验进一步说明：**multi-HCA 不指定均分在单逻辑 foreground flow 下可用，但不适合当作手动 shard 的替代方案来吃满多端口高带宽。** 它在低 foreground cap 档位（例如 bg90）能贴住目标；在 100-200Gbps 档位部分可用但会有损耗；在 400G 目标和每端口 capfill 约束下，距离 manual split 很远。

如果目标是稳定利用 4x100 的剩余前景带宽，当前证据仍支持手动 shard；如果目标是简化配置，并且可接受约 130-220Gbps 级别的大块传输吞吐，multi-HCA 单逻辑流可以作为简化方案继续评估。

## 产物索引

- [run-manifest.csv](figures/kv-transfer-multi-hca-portcap/run-manifest.csv)
- [all-aggregated-summary-with-metadata.csv](figures/kv-transfer-multi-hca-portcap/all-aggregated-summary-with-metadata.csv)
- [per-experiment-calculated-bandwidth.csv](figures/kv-transfer-multi-hca-portcap/per-experiment-calculated-bandwidth.csv)
- [key-results-512mib-1gib-2gib.csv](figures/kv-transfer-multi-hca-portcap/key-results-512mib-1gib-2gib.csv)
- [multi-hca-portcap-2gib-trends.csv](figures/kv-transfer-multi-hca-portcap/multi-hca-portcap-2gib-trends.csv)
- [compare-portcap-4x100-vs-prior-split.csv](figures/kv-transfer-multi-hca-portcap/compare-portcap-4x100-vs-prior-split.csv)
- [rdma-monitor-and-log-validation.csv](figures/kv-transfer-multi-hca-portcap/rdma-monitor-and-log-validation.csv)
- [anomalies.csv](figures/kv-transfer-multi-hca-portcap/anomalies.csv)
- [chart-map.csv](figures/kv-transfer-multi-hca-portcap/chart-map.csv)
- [portcap-2gib-bandwidth.svg](figures/kv-transfer-multi-hca-portcap/portcap-2gib-bandwidth.svg)
- [portcap-2gib-cap-utilization.svg](figures/kv-transfer-multi-hca-portcap/portcap-2gib-cap-utilization.svg)
- [portcap-2gib-latency.svg](figures/kv-transfer-multi-hca-portcap/portcap-2gib-latency.svg)
- [portcap-4x100-vs-prior-split-2gib-bandwidth.svg](figures/kv-transfer-multi-hca-portcap/portcap-4x100-vs-prior-split-2gib-bandwidth.svg)
`;

  await fs.writeFile(REPORT_PATH, report, "utf8");
  console.log(`wrote ${path.relative(ROOT, REPORT_PATH)}`);
  console.log(`wrote ${path.relative(ROOT, OUT_DIR)}`);
}

await main();
