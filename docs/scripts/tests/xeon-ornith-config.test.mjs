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

for (const [vendor, name, model] of [
  ["deepreinforce-ai", "ornith-1.0", "Ornith-1.0-9B"],
  ["ornith-ai", "ornith-1.5", "Ornith-1.5-9B"],
]) {
  const text = readFileSync(new URL(`../../src/snippets/configs/${vendor}/${name}.jsx`, import.meta.url), "utf8");
  const { config } = await import("data:text/javascript," + encodeURIComponent(text));
  const api = engine(config);

  test(`${name} offers only independently validated Python-only CPU recipes`, () => {
    assert.ok(config.supportedHardware.includes("xeon"));
    assert.ok(config.hardware.some((hw) => hw.id === "xeon" && hw.vendor === "cpu"));
    assert.deepEqual(config.runModes({ hw: "xeon" }), ["python"]);
    assert.deepEqual(config.runModes({ hw: "h100" }), ["python", "docker"]);
    const cells = config.cells.filter((cell) => cell.match.hw === "xeon");
    assert.equal(cells.length, 2);
    const [cell] = cells;
    assert.equal(cell.match.variant, "9b");
    assert.equal(cell.match.quant, "bf16");
    const command = api.renderCommand(cell, cell.match, {
      HOST_IP: "127.0.0.1", PORT: "31001",
    });
    for (const flag of [
      `--model ${vendor}/${model}`, "--device cpu", "--disable-overlap-schedule", "--tp 6", "--port 31001",
    ]) assert.ok(command.includes(flag), flag);
    assert.match(command, /^sglang serve/);
    assert.doesNotMatch(command, /--revision|--dtype|--max-total-tokens|--served-model-name|docker|--gpus|--context-length|--reasoning-parser|--tool-call-parser/);
    assert.match(cell.warn, /Only basic text generation.*full context.*unverified/);
  });

  test(`${name} resets stale GPU variants, precision and topology on CPU`, () => {
    const stale = { hw: "h200", variant: "397b", quant: "fp8", strategy: "balanced", nodes: "single" };
    const switched = api.snapToValidCell(config.cells, stale, "hw", "xeon");
    assert.equal(switched.variant, "9b");
    assert.equal(switched.quant, "bf16");
    assert.ok(api.findCell(config.cells, switched));
    assert.equal(api.isOptionAvailable(config.cells, switched, "variant", "397b"), false);
    assert.equal(api.isOptionAvailable(config.cells, switched, "quant", "fp8"), false);
    const fromUrl = api.validateSelection(config.cells, { ...stale, hw: "xeon", nodes: "multi-2" });
    assert.equal(fromUrl.variant, "9b");
    assert.equal(fromUrl.quant, "bf16");
    assert.equal(fromUrl.nodes, "single");
  });

  test(`${name} discards hidden CPU Playground overrides without changing GPU features`, () => {
    const features = config.playgroundFeatures;
    const stale = { parsers: { reasoning: true, toolCall: true }, attention: { tp: 8 } };
    assert.deepEqual(features.normalizeDeltas(stale, {}, { hw: "xeon" }), {});
    assert.equal(features.normalizeDeltas(stale, {}, { hw: "h100" }), stale);
    for (const [axis, feature] of Object.entries(features)) {
      if (axis === "normalizeDeltas") continue;
      assert.equal(feature.showWhen({ hw: "xeon" }), false, axis);
      assert.equal(feature.showWhen({ hw: "h100" }), true, axis);
    }

  });
}

test("Ornith 1.0 35B uses its independent BF16 TP6 record, not its FP8 sibling", async () => {
  const text = readFileSync(new URL("../../src/snippets/configs/deepreinforce-ai/ornith-1.0.jsx", import.meta.url), "utf8");
  const { config } = await import("data:text/javascript," + encodeURIComponent(text));
  const cell = config.cells.find((candidate) => candidate.match.hw === "xeon" && candidate.match.variant === "35b");
  const command = engine(config).renderCommand(cell, cell.match, {});
  for (const flag of [
    "--model deepreinforce-ai/Ornith-1.0-35B", "--device cpu", "--disable-overlap-schedule", "--tp 6",
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /-FP8|--revision|--served-model-name|--reasoning-parser|--tool-call-parser|--speculative/);
});

test("Ornith 1.5 35B-A3B uses its independent BF16 TP6 record, not its FP8 sibling", async () => {
  const text = readFileSync(new URL("../../src/snippets/configs/ornith-ai/ornith-1.5.jsx", import.meta.url), "utf8");
  const { config } = await import("data:text/javascript," + encodeURIComponent(text));
  const cell = config.cells.find((candidate) => candidate.match.hw === "xeon" && candidate.match.variant === "35b");
  const command = engine(config).renderCommand(cell, cell.match, {});
  for (const flag of [
    "--model ornith-ai/Ornith-1.5-35B-A3B", "--device cpu", "--disable-overlap-schedule", "--tp 6",
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /-FP8|--revision|--served-model-name|--reasoning-parser|--tool-call-parser|--speculative/);
});
