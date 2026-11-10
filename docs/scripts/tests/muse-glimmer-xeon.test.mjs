import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const text = readFileSync(new URL("../../src/snippets/configs/meta-models/muse-glimmer.jsx", import.meta.url), "utf8");
const { config } = await import("data:text/javascript," + encodeURIComponent(text));
const source = readFileSync(new URL("../../src/snippets/_deployment.jsx", import.meta.url), "utf8");
const setup = source.slice(source.indexOf("  const LEGACY_MATCH_DIMS ="), source.indexOf("  const ACCURACY_LABELS ="));
const api = new Function("config", `${setup} return { findCell, renderCommand, snapToValidCell, validateSelection };`)(config);
const cell = config.cells.find((candidate) => candidate.match.hw === "xeon");

test("Muse Glimmer CPU preserves the complete checkpoint and tested command", () => {
  const command = api.renderCommand(cell, { ...cell.match, modality: "text" }, {
    HOST_IP: "0.0.0.0", PORT: "30000",
  });
  for (const flag of [
    "--model-path meta-models/Muse-Glimmer-30B",
    "--revision cec94fcfd6512f2cfe708138c32533b346ea4f69",
    "--device cpu", "--disable-overlap-schedule", "--tp 6", "--dtype bfloat16",
    "--max-total-tokens 8192", "--port 30000", "--trust-remote-code",
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /--language-model-only|--speculative|--reasoning-parser|--tool-call-parser|docker/);
  assert.deepEqual(config.runModes(cell.match), ["python"]);
  assert.deepEqual(config.runModes({ hw: "mac" }), ["python"]);
  assert.deepEqual(config.runModes({ hw: "h200" }), ["python", "docker"]);
});

test("Muse Glimmer CPU rejects stale GPU and MLX combinations", () => {
  const stale = { hw: "mac", variant: "default", quant: "mlx-q4km", strategy: "dflash", nodes: "single", modality: "mm" };
  const switched = api.snapToValidCell(config.cells, stale, "hw", "xeon");
  assert.equal(switched.quant, "bf16");
  assert.equal(switched.strategy, "standard");
  assert.ok(api.findCell(config.cells, switched));
  assert.equal(config.overlayDims[0].showWhen(switched), false);
  assert.deepEqual(config.playgroundFeatures.normalizeDeltas({ attention: { tp: 2 }, speculative: "dflash" }, {}, switched), {});
  for (const feature of [config.playgroundFeatures.attention, config.playgroundFeatures.speculative]) {
    assert.equal(feature.showWhen(switched), false);
    assert.equal(feature.showWhen({ hw: "h200" }), true);
  }
});
