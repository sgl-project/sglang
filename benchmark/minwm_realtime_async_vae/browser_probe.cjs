#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

function percentile(sorted, ratio) {
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * ratio;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

function rounded(value) {
  return Math.round(value * 1000) / 1000;
}

function summarize(values) {
  const sorted = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return { count: 0, min: null, p50: null, p95: null, max: null, mean: null };
  return {
    count: sorted.length,
    min: rounded(sorted[0]),
    p50: rounded(percentile(sorted, 0.5)),
    p95: rounded(percentile(sorted, 0.95)),
    max: rounded(sorted.at(-1)),
    mean: rounded(sorted.reduce((sum, value) => sum + value, 0) / sorted.length),
  };
}

function traceHttpUrl(pageUrl, traceId) {
  const url = new URL(pageUrl);
  return `${url.origin}/v1/realtime_video/traces/${encodeURIComponent(traceId)}`;
}

function parseArgs(argv) {
  const values = {};
  for (let index = 2; index < argv.length; index += 1) {
    const name = argv[index];
    if (!name.startsWith("--")) throw new Error(`unexpected argument ${name}`);
    values[name.slice(2)] = argv[++index];
  }
  if (!values.url || !values.output) throw new Error("--url and --output are required");
  return {
    url: values.url,
    output: values.output,
    screenshot: values.screenshot || "",
    minFrames: Number(values["min-frames"] || 64),
    warmupChunks: Number(values["warmup-chunks"] || 2),
    timeoutMs: Number(values["timeout-ms"] || 300000),
    traceTimeoutMs: Number(values["trace-timeout-ms"] || 90000),
  };
}

async function collectTrace(request, endpoint, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let cursor = 0;
  let stable = 0;
  const events = new Map();
  while (Date.now() < deadline) {
    const response = await request.get(endpoint, {
      params: { after: String(cursor), limit: "500" },
      timeout: 15000,
    });
    if (response.ok()) {
      const payload = await response.json();
      let added = 0;
      for (const event of payload.events || []) {
        const sequence = Number(event.trace_seq || 0);
        if (!sequence || events.has(sequence)) continue;
        events.set(sequence, event);
        added += 1;
      }
      cursor = Math.max(cursor, Number(payload.next_cursor || 0), ...events.keys());
      stable = added ? 0 : stable + 1;
      const hasDisplay = [...events.values()].some(
        (event) => event.event === "client.chunk_first_rendered" && Number.isFinite(Number(event.display_lag_ms)),
      );
      if (hasDisplay && stable >= 2) break;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  return [...events.values()].sort((left, right) => Number(left.trace_seq) - Number(right.trace_seq));
}

async function run(args) {
  const { chromium } = require("playwright");
  const launch = { headless: true };
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) {
    launch.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  }
  const browser = await chromium.launch(launch);
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await context.newPage();
  try {
    const url = new URL(args.url);
    url.searchParams.set("mode", "t2v");
    await page.goto(url.toString(), { waitUntil: "networkidle", timeout: 60000 });
    await page.selectOption("#generationMode", "t2v");
    await page.fill("#fps", "24");
    await page.fill("#numFrames", "121");
    await page.fill("#guidance", "0");
    await page.fill("#prompt", "A smooth forward camera move through a mountain valley in daylight");
    await page.click("#connectBtn");
    await page.waitForFunction(
      (minFrames) => {
        const debug = window.__sglangRealtimeDebug?.();
        return debug && debug.frames >= minFrames && Number.isFinite(debug.lastDisplayLagMs);
      },
      args.minFrames,
      { timeout: args.timeoutMs, polling: 250 },
    );
    await page.keyboard.down("w");
    await page.waitForTimeout(350);
    await page.keyboard.up("w");
    await page.waitForTimeout(1500);
    const debug = await page.evaluate(() => window.__sglangRealtimeDebug());
    const traceId = debug.currentSessionArtifact?.traceId;
    if (!traceId) throw new Error("browser session did not expose a trace id");
    if (args.screenshot) {
      await page.screenshot({ path: args.screenshot, fullPage: true });
    }
    const events = await collectTrace(
      context.request,
      traceHttpUrl(url.toString(), traceId),
      args.traceTimeoutMs,
    );
    const displayEvents = events.filter(
      (event) =>
        event.event === "client.chunk_first_rendered" &&
        Number(event.chunk_index || 0) >= args.warmupChunks &&
        Number.isFinite(Number(event.display_lag_ms)),
    );
    if (!displayEvents.length) throw new Error("CloudWatch Trace has no warm display-lag events");
    return {
      trace_id: traceId,
      display_lag_ms: summarize(displayEvents.map((event) => Number(event.display_lag_ms))),
      rendered_frames: Number(debug.frames || 0),
      render_fps: Number(debug.renderedFps || 0),
      playback: debug.playback,
      evidence_events: displayEvents.length,
    };
  } finally {
    await browser.close();
  }
}

if (require.main === module) {
  const args = parseArgs(process.argv);
  run(args)
    .then((result) => {
      fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });
      fs.writeFileSync(args.output, `${JSON.stringify(result, null, 2)}\n`);
      console.log(`browser probe passed: ${args.output}`);
    })
    .catch((error) => {
      console.error(error.stack || error);
      process.exitCode = 1;
    });
}

module.exports = { collectTrace, parseArgs, run, summarize, traceHttpUrl };
