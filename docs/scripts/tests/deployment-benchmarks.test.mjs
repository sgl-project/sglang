import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// Exercise the card's real data builders without a browser or JSX dependency.
const source = readFileSync(new URL("../../src/snippets/_deployment.jsx", import.meta.url), "utf8");
const section = (start, end) => {
  const a = source.indexOf(start);
  const b = source.indexOf(end, a);
  assert.ok(a >= 0 && b > a, `Missing source section: ${start}`);
  return source.slice(a, b);
};
const normalize = section("  const normalizeSpeed =", "  // Variant default accuracy");
const accuracy = section("  const effectiveAccuracy =", "  // Empty =");
const tableSetup = section("    const pct =", "    const renderBenchTable =");
const tableBuilder = section("    const buildSpeedTable =", "    // One row per ACCURACY_LABELS");
const commandBuilder = section("  const buildBenchCommands =", "  const buildHardwareGroups =");
const speedTable = new Function("entry", "config", `${normalize}\n${tableSetup}\n${tableBuilder}\nreturn buildSpeedTable(normalizeSpeed(entry.speed));`);
const benchCommands = new Function("entry", "config", "sel", `${normalize}\n${accuracy}\nconst ACCURACY_LABELS = config.accuracyLabels || [];\n${commandBuilder}\nreturn buildBenchCommands(entry, sel);`);
const workload = { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 };

test("legacy speed rows retain their labels, units, values, and total-throughput legend", () => {
  const table = speedTable({ speed: { workload, ttft_ms: 400, tpot_ms: 8, tokens_per_sec_per_gpu: 2000 } }, {});
  assert.equal(table.sharedText, "random, in/out=8192/1024");
  assert.deepEqual(table.colHeaders, ["max-concurrency=16"]);
  assert.deepEqual(table.rows, [
    { label: "TTFT (P50)", values: ["400 ms"] },
    { label: "TPOT (P50)", values: ["8 ms"] },
    { label: "throughput per gpu", values: ["2000 tok/s"] },
    { label: "interactivity", values: ["125 tokens/s/user"] },
  ]);
  assert.deepEqual(table.legend, [
    "throughput per gpu = (input+output tokens)/elapsed/GPU",
    "interactivity = 1000/TPOT(ms) (tokens/s/user)",
  ]);
});

test("trace measurements display mean ITL and output-only throughput without relabeling them as TPOT or total", () => {
  const table = speedTable({ latencyPercentile: "Mean", speed: [
    { workload: { dataset: "Mooncake trace 4%", max_concurrency: 1 }, ttft_ms: 70, itl_ms: 5, output_tokens_per_sec_per_gpu: 100 },
    { workload: { dataset: "Mooncake trace 4%", max_concurrency: 16 }, ttft_ms: 120, itl_ms: 8, output_tokens_per_sec_per_gpu: 250 },
  ] }, { latencyPercentile: "P50" });
  assert.deepEqual(table.colHeaders, ["max-concurrency=1", "max-concurrency=16"]);
  assert.deepEqual(table.rows, [
    { label: "TTFT (Mean)", values: ["70 ms", "120 ms"] },
    { label: "ITL (Mean)", values: ["5 ms", "8 ms"] },
    { label: "output throughput per gpu", values: ["100 tok/s", "250 tok/s"] },
    { label: "interactivity", values: ["200 tokens/s/user", "125 tokens/s/user"] },
  ]);
  assert.deepEqual(table.legend, [
    "output throughput per gpu = output tokens/elapsed/GPU",
    "interactivity = 1000/ITL(ms) (tokens/s/user)",
  ]);
});

test("missing or zero token latency does not produce an infinite interactivity value", () => {
  const table = speedTable({ speed: [{ itl_ms: 0 }, { itl_ms: null }] }, {});
  assert.deepEqual(table.rows.at(-1).values, [null, null]);
});

test("an entry can suppress reproduction templates while other entries retain them", () => {
  const config = { benchmarkCommands: { speed: "bench {{MAX_CONCURRENCY}}" } };
  const entry = { speed: { workload, tpot_ms: 8 } };
  const commands = benchCommands(entry, config, {});
  assert.equal(commands.speed.template, "bench {{MAX_CONCURRENCY}}");
  assert.deepEqual(commands.speed.concurrencies, [16]);
  assert.equal(commands.speed.numPromptsOf(16), 200);
  assert.equal(benchCommands({ ...entry, benchmarkCommands: false }, config, {}), null);
});

test("explicit accuracy nulls suppress inherited scores and reproduction commands", () => {
  const config = {
    defaultAccuracy: { default: { gsm8k_pct: 98, aime25_pct: 88 } },
    accuracyLabels: [["gsm8k_pct", "GSM8K"], ["aime25_pct", "AIME25"]],
    benchmarkCommands: { accuracy: { gsm8k_pct: "eval gsm8k", aime25_pct: "eval aime25" } },
  };
  assert.equal(benchCommands({}, config, { variant: "default" }).accuracy.length, 2);
  assert.equal(benchCommands({ accuracy: { gsm8k_pct: null, aime25_pct: null } }, config, { variant: "default" }), null);
});
