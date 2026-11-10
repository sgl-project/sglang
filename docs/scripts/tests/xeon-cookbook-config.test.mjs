import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const source = readFileSync(new URL("../../src/snippets/_deployment.jsx", import.meta.url), "utf8");
const setup = source.slice(
  source.indexOf("  const LEGACY_MATCH_DIMS ="),
  source.indexOf("  const ACCURACY_LABELS ="),
);
const engine = (config) => new Function("config", `${setup}
return { findCell, snapToValidCell, validateSelection, isOptionAvailable, renderCommand };
`)(config);
const load = async (name, vendor = "IFM") => {
  const text = readFileSync(new URL(`../../src/snippets/configs/${vendor}/${name}.jsx`, import.meta.url), "utf8");
  return (await import("data:text/javascript," + encodeURIComponent(text))).config;
};

for (const [vendor, name, checkpoint, revision, cap, gpu] of [
  ["IFM", "k2-horizon", "K2-Horizon-0.9B", "d50881f4f8dc6923fb8f301e1c2313fe3a2b6c1d", 4096, "h200"],
  ["IFM", "jais-3", "Jais-3-0.9B", "ef8b0e93cae13e7fdb6c711770ab00e03616baac", 8192, "h100"],
  ["ibm-granite", "granite-4.2", "granite-4.2-3b", "4e9c275e232f10211ec88644d106a1f6a1b60e06", 8192, "h200"],
]) {
const config = await load(name, vendor);
const firstVariant = vendor === "IFM" ? "0.9b" : "3b";
const staleVariant = vendor === "IFM" ? "375b" : "30b";

test(`${name} exposes only validated Python-only Xeon recipes`, () => {
  assert.ok(config.supportedHardware.includes("xeon"));
  assert.ok(config.hardware.some((hw) => hw.id === "xeon" && hw.vendor === "cpu"));
  const modes = (hw) => typeof config.runModes === "function" ? config.runModes({ hw }) : config.runModes;
  assert.deepEqual(modes("xeon"), ["python"]);
  if (vendor === "ibm-granite") assert.deepEqual(modes(gpu), ["python", "docker"]);
  const cells = config.cells.filter((cell) => cell.match.hw === "xeon");
  assert.deepEqual(cells.map((cell) => cell.match.variant).sort(),
    vendor === "IFM" ? ["0.9b", "3.7b"] : [firstVariant]);
  const api = engine(config);
  const cell = cells.find((candidate) => candidate.match.variant === firstVariant);
  const command = api.renderCommand(cell, cell.match, {
    HOST_IP: "127.0.0.1", PORT: "31000",
  });
  for (const flag of [
    `--model ${vendor}/${checkpoint}`, "--device cpu", "--disable-overlap-schedule",
    "--tp 6", "--dtype bfloat16", `--max-total-tokens ${cap}`, "--port 31000",
    `--revision ${revision}`,
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /docker|--gpus|fa3|--ep |--nnodes|--reasoning-parser|--speculative/);
  assert.match(cell.warn, /Quick generation.*full context.*unverified/);
});

test(`${name} hardware switches and stale URLs snap to a validated Xeon cell`, () => {
  const api = engine(config);
  const stale = { hw: gpu, variant: staleVariant, quant: "bf16", strategy: "balanced", nodes: "single" };
  const switched = api.snapToValidCell(config.cells, stale, "hw", "xeon");
  assert.equal(switched.variant, firstVariant);
  assert.ok(api.findCell(config.cells, switched));
  assert.equal(api.isOptionAvailable(config.cells, switched, "variant", staleVariant), false);
  const fromUrl = api.validateSelection(config.cells, { ...stale, hw: "xeon", nodes: "multi-2" });
  assert.equal(fromUrl.variant, firstVariant);
  assert.equal(fromUrl.nodes, "single");
});

test(`${name} hides unvalidated Playground axes and discards stale GPU overrides`, () => {
  const features = config.playgroundFeatures;
  for (const axis of Object.keys(features).filter((key) => key !== "normalizeDeltas")) {
    assert.equal(features[axis].showWhen({ hw: "xeon" }), false, axis);
    assert.equal(features[axis].showWhen({ hw: gpu }), true, axis);
  }
  const stale = {
    attention: { tp: 8 }, moe: { ep: 8 }, parsers: { toolCall: true },
    speculative: "ngram", pdDisagg: { mode: "prefill" }, hicache: { enabled: true },
  };
  assert.deepEqual(features.normalizeDeltas(stale, {}, { hw: "xeon" }), {});
  assert.equal(features.normalizeDeltas(stale, {}, { hw: gpu }), stale);
});
}

for (const [vendor, name, variant, checkpoint, revision] of [
  ["IFM", "k2-horizon", "3.7b", "K2-Horizon-3.7B", "4525090a8f0fce1026d861701b9bde2fb726b7a0"],
  ["IFM", "jais-3", "3.7b", "Jais-3-3.7B", "78d52c03900ef18ee3ebae631426556df6bd925a"],
]) {
test(`${checkpoint} emits its independently validated checkpoint and cache limit`, async () => {
  const config = await load(name, vendor);
  const cell = config.cells.find((cell) => cell.match.hw === "xeon" && cell.match.variant === variant);
  const command = engine(config).renderCommand(cell, cell.match, {
    HOST_IP: "127.0.0.1", PORT: "31001",
  });
  for (const flag of [
    `--model ${vendor}/${checkpoint}`,
    `--revision ${revision}`,
    "--tp 6", "--max-total-tokens 8192", "OPENBLAS_NUM_THREADS=1",
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /docker|--gpus|fa3|--ep |--nnodes|--reasoning-parser|--speculative/);
});
}
