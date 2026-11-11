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

for (const [vendor, name, checkpoint, gpu] of [
  ["IFM", "k2-horizon", "K2-Horizon-0.9B", "h200"],
  ["IFM", "jais-3", "Jais-3-0.9B", "h100"],
  ["ibm-granite", "granite-4.2", "granite-4.2-3b", "h200"],
]) {
const config = await load(name, vendor);
const firstVariant = vendor === "IFM" ? "0.9b" : "3b";
const staleVariant = vendor === "IFM" ? "375b" : "invalid";

test(`${name} exposes only validated Python-only Xeon recipes`, () => {
  assert.ok(config.supportedHardware.includes("xeon"));
  assert.ok(config.hardware.some((hw) => hw.id === "xeon" && hw.vendor === "cpu"));
  const modes = (hw) => typeof config.runModes === "function" ? config.runModes({ hw }) : config.runModes;
  assert.deepEqual(modes("xeon"), ["python"]);
  if (vendor === "ibm-granite") assert.deepEqual(modes(gpu), ["python", "docker"]);
  const cells = config.cells.filter((cell) => cell.match.hw === "xeon");
  const expectedVariants = vendor === "IFM" ? ["0.9b", "3.7b"] : ["3b", "8b", "30b"];
  if (vendor === "IFM") expectedVariants.push("7b");
  if (vendor === "IFM") expectedVariants.push("32b");
  assert.deepEqual(cells.map((cell) => cell.match.variant).sort(), expectedVariants.sort());
  const api = engine(config);
  const cell = cells.find((candidate) => candidate.match.variant === firstVariant);
  const command = api.renderCommand(cell, cell.match, {
    HOST_IP: "127.0.0.1", PORT: "31000",
  });
  for (const flag of [
    `--model ${vendor}/${checkpoint}`, "--device cpu", "--tp 6", "--port 31000",
  ]) assert.ok(command.includes(flag), flag);
  assert.match(command, /^sglang serve/);
  assert.doesNotMatch(command, /--revision|--dtype|--max-total-tokens|--disable-overlap-schedule|docker|--gpus|fa3|--ep |--nnodes|--reasoning-parser|--speculative/);
  assert.match(cell.warn, /Only basic text generation.*full context.*unverified/);
  const defaults = api.renderCommand(cell, cell.match, { HOST_IP: "127.0.0.1", PORT: "30000" });
  assert.doesNotMatch(defaults, /--host|--port/);
  const custom = api.renderCommand(cell, cell.match, { HOST_IP: "0.0.0.0", PORT: "31001" });
  assert.match(custom, /--host 0\.0\.0\.0/);
  assert.match(custom, /--port 31001/);
  const gpuCell = config.cells.find((candidate) => candidate.match.hw === gpu);
  assert.match(api.renderCommand(gpuCell, gpuCell.match, { HOST_IP: "127.0.0.1", PORT: "30000" }), /--port 30000/);
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

for (const [vendor, name, variant, checkpoint] of [
  ["IFM", "k2-horizon", "3.7b", "K2-Horizon-3.7B"],
  ["IFM", "k2-horizon", "7b", "K2-Horizon-7B"],
  ["IFM", "k2-horizon", "32b", "K2-Horizon-32B"],
  ["IFM", "jais-3", "3.7b", "Jais-3-3.7B"],
  ["IFM", "jais-3", "7b", "Jais-3-7B"],
  ["IFM", "jais-3", "32b", "Jais-3-32B"],
  ["ibm-granite", "granite-4.2", "8b", "granite-4.2-8b"],
  ["ibm-granite", "granite-4.2", "30b", "granite-4.2-30b"],
]) {
test(`${checkpoint} emits its model and CPU topology without redundant flags`, async () => {
  const config = await load(name, vendor);
  const cell = config.cells.find((cell) => cell.match.hw === "xeon" && cell.match.variant === variant);
  const command = engine(config).renderCommand(cell, cell.match, {
    HOST_IP: "127.0.0.1", PORT: "31001",
  });
  for (const flag of [
    `--model ${vendor}/${checkpoint}`,
    "--device cpu", "--tp 6",
  ]) assert.ok(command.includes(flag), flag);
  assert.match(command, /^sglang serve/);
  assert.doesNotMatch(command, /--revision|--dtype|--max-total-tokens|--disable-overlap-schedule|docker|--gpus|fa3|--ep |--nnodes|--reasoning-parser|--speculative/);
});
}
