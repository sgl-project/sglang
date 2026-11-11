import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");
const text = read("../../src/snippets/configs/Qwen/qwen3.8-27b.jsx");
const { config } = await import("data:text/javascript," + encodeURIComponent(text));
const deployment = read("../../src/snippets/_deployment.jsx");
const playground = read("../../src/snippets/_playground.jsx");
const setup = deployment.slice(
  deployment.indexOf("  const LEGACY_MATCH_DIMS ="),
  deployment.indexOf("  const ACCURACY_LABELS ="),
);
const api = new Function("config", `${setup}
return { findCell, renderCommand, snapToValidCell, validateSelection };`)(config);
const selection = { hw: "xeon", variant: "default", quant: "fp8", nodes: "single" };
const cell = api.findCell(config.cells, selection);

test("Qwen3.8 FP8 Xeon emits a concise TP6 text recipe", () => {
  const command = api.renderCommand(cell, {
    ...selection, spec: "dflash", tier: "high-throughput", ssmDtype: "bfloat16",
  }, { HOST_IP: "127.0.0.1", PORT: "31001" });
  for (const flag of [
    "--model Qwen/Qwen3.8-27B-FP8", "--device cpu", "--tp 6", "--port 31001",
  ]) assert.ok(command.includes(flag), flag);
  assert.match(command, /^sglang serve/);
  assert.doesNotMatch(command, /--revision|--dtype|--max-total-tokens|--trust-remote-code|--disable-overlap-schedule|--speculative|--mamba-|--kv-cache|--mem-fraction|--reasoning-parser|--tool-call-parser|docker/);
  assert.deepEqual(config.runModes(selection), ["python"]);
  assert.match(cell.warn, /Only basic text generation.*unverified/);
});

test("Qwen3.8 hardware switches retain validated FP8 and reject NVFP4", () => {
  const switched = api.snapToValidCell(config.cells, { ...selection, hw: "h200" }, "hw", "xeon");
  assert.equal(switched.quant, "fp8");
  assert.ok(api.findCell(config.cells, switched));
  const stale = api.validateSelection(config.cells, { ...selection, quant: "nvfp4-nvidia", nodes: "multi-2" });
  assert.ok(api.findCell(config.cells, stale));
  assert.equal(stale.nodes, "single");
  assert.notEqual(stale.quant, "nvfp4-nvidia");
});

test("Qwen3.8 FP8 CPU hides and clears GPU Playground overrides only", () => {
  const features = config.playgroundFeatures;
  for (const feature of [features.parsers, features.speculative, ...features.flagSelects]) {
    assert.equal(feature.showWhen(selection), false);
    assert.equal(feature.showWhen({ ...selection, hw: "h200" }), true);
    assert.equal(feature.showWhen({ ...selection, quant: "bf16" }), true);
  }
  const stale = { parsers: { reasoning: true }, speculative: "dflash", flagSelects: { prefixCache: "off" } };
  assert.deepEqual(features.normalizeDeltas(stale, {}, selection), {});
  assert.equal(features.normalizeDeltas(stale, {}, { ...selection, quant: "bf16" }), stale);
  assert.equal(features.normalizeDeltas(stale, {}, { ...selection, hw: "h200" }), stale);
});

test("fixed CPU recipes ignore stale calculator ratios in Deploy and Playground", () => {
  const deployBody = deployment.slice(
    deployment.indexOf("  const cellWithRatio = (() => {") + "  const cellWithRatio = (() => {".length,
    deployment.indexOf("  const commandEnv =", deployment.indexOf("  const cellWithRatio =")),
  ).split("  })();")[0];
  const applyDeploy = new Function("cell", "mambaRatio", deployBody);
  const pgBody = playground.slice(
    playground.indexOf("  const withRatio = (fl, value) => {") + "  const withRatio = (fl, value) => {".length,
    playground.indexOf("  if (baseCell) {", playground.indexOf("  const withRatio =")),
  ).split("  };")[0];
  const applyPlayground = new Function("baseCell", "fl", "value", pgBody);
  assert.equal(applyDeploy(cell, "3.12"), cell);
  assert.deepEqual(applyPlayground(cell, cell.flags, "3.12"), cell.flags);
  const gpu = config.cells.find((candidate) => candidate.match.hw === "h200");
  assert.ok(applyDeploy(gpu, "3.12").flags.includes("--mamba-full-memory-ratio 3.12"));
  assert.ok(applyPlayground(gpu, gpu.flags, "3.12").includes("--mamba-full-memory-ratio 3.12"));
});
