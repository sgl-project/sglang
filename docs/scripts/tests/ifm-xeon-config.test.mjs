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
const load = async (name) => {
  const text = readFileSync(new URL(`../../src/snippets/configs/IFM/${name}.jsx`, import.meta.url), "utf8");
  return (await import("data:text/javascript," + encodeURIComponent(text))).config;
};

for (const [name, checkpoint, revision, cap, gpu] of [
  ["k2-horizon", "K2-Horizon-0.9B", "d50881f4f8dc6923fb8f301e1c2313fe3a2b6c1d", 4096, "h200"],
  ["jais-3", "Jais-3-0.9B", "ef8b0e93cae13e7fdb6c711770ab00e03616baac", 8192, "h100"],
]) {
const config = await load(name);

test(`${name} exposes only validated Python-only Xeon recipes`, () => {
  assert.ok(config.supportedHardware.includes("xeon"));
  assert.ok(config.hardware.some((hw) => hw.id === "xeon" && hw.vendor === "cpu"));
  assert.deepEqual(config.runModes, ["python"]);
  const cells = config.cells.filter((cell) => cell.match.hw === "xeon");
  assert.deepEqual(cells.map((cell) => cell.match.variant), ["0.9b"]);
  const api = engine(config);
  const cell = cells[0];
  const command = api.renderCommand(cell, cell.match, {
    HOST_IP: "127.0.0.1", PORT: "31000",
  });
  for (const flag of [
    `--model IFM/${checkpoint}`, "--device cpu", "--disable-overlap-schedule",
    "--tp 6", "--dtype bfloat16", `--max-total-tokens ${cap}`, "--port 31000",
    `--revision ${revision}`,
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /docker|--gpus|fa3|--ep |--nnodes|--reasoning-parser|--speculative/);
  assert.match(cell.warn, /Quick generation.*full context.*unverified/);
});

test(`${name} hardware switches and stale URLs snap to a validated Xeon cell`, () => {
  const api = engine(config);
  const stale = { hw: gpu, variant: "375b", quant: "bf16", strategy: "balanced", nodes: "single" };
  const switched = api.snapToValidCell(config.cells, stale, "hw", "xeon");
  assert.equal(switched.variant, "0.9b");
  assert.ok(api.findCell(config.cells, switched));
  assert.equal(api.isOptionAvailable(config.cells, switched, "variant", "375b"), false);
  const fromUrl = api.validateSelection(config.cells, { ...stale, hw: "xeon", nodes: "multi-2" });
  assert.equal(fromUrl.variant, "0.9b");
  assert.equal(fromUrl.nodes, "single");
});

test(`${name} hides unvalidated Playground axes and discards stale GPU overrides`, () => {
  const features = config.playgroundFeatures;
  for (const axis of ["attention", "moe", "parsers", "speculative", "pdDisagg", "hicache"]) {
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
