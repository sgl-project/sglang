import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const text = readFileSync(new URL("../../src/snippets/configs/poolside/laguna-xs21.jsx", import.meta.url), "utf8");
const { config } = await import("data:text/javascript," + encodeURIComponent(text));
const source = readFileSync(new URL("../../src/snippets/_deployment.jsx", import.meta.url), "utf8");
const setup = source.slice(source.indexOf("  const LEGACY_MATCH_DIMS ="), source.indexOf("  const ACCURACY_LABELS ="));
const api = new Function("config", `${setup} return { findCell, renderCommand, snapToValidCell };`)(config);
const cell = config.cells.find((candidate) => candidate.match.hw === "xeon");

test("Laguna XS2.1 CPU emits exactly its validated BF16 command shape", () => {
  const command = api.renderCommand(cell, cell.match, { HOST_IP: "0.0.0.0", PORT: "30000" });
  for (const flag of [
    "--model-path poolside/Laguna-XS-2.1",
    "--revision c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db",
    "--device cpu", "--disable-overlap-schedule", "--tp 4", "--dtype bfloat16",
    "--max-total-tokens 8192", "--trust-remote-code", "--port 30000",
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /--reasoning-parser|--tool-call-parser|--speculative|--ep-size|--attention-backend|docker/);
  assert.deepEqual(config.runModes(cell.match), ["python"]);
  assert.deepEqual(config.runModes({ hw: "gb300" }), ["python", "docker"]);
});

test("Laguna XS2.1 hardware switches clear stale GPU quantization and speculation", () => {
  const stale = { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" };
  const switched = api.snapToValidCell(config.cells, stale, "hw", "xeon");
  assert.equal(switched.quant, "bf16");
  assert.equal(switched.strategy, "high-throughput");
  assert.ok(api.findCell(config.cells, switched));
  assert.deepEqual(config.playgroundFeatures.normalizeDeltas({ attention: { tp: 8 }, parsers: { reasoning: true } }, {}, switched), {});
  for (const feature of [config.playgroundFeatures.attention, config.playgroundFeatures.parsers]) {
    assert.equal(feature.showWhen(switched), false);
    assert.equal(feature.showWhen({ hw: "gb300" }), true);
  }
  for (const quant of ["fp8", "nvfp4", "int4"]) {
    assert.equal(api.findCell(config.cells, { ...cell.match, quant }), undefined);
  }
});
