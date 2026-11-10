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

for (const [vendor, name, model, revision] of [
  ["deepreinforce-ai", "ornith-1.0", "Ornith-1.0-9B", "83dc1f5e24ef8527af019a6b3bf66ac0f1c2c999"],
  ["ornith-ai", "ornith-1.5", "Ornith-1.5-9B", "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"],
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
      `--model ${vendor}/${model}`, `--revision ${revision}`,
      `--served-model-name ${model}`, "--device cpu", "--disable-overlap-schedule",
      "--tp 6", "--dtype bfloat16", "--max-total-tokens 8192", "--port 31001",
      "SGLANG_USE_CPU_ENGINE=1", "OPENBLAS_NUM_THREADS=1",
    ]) assert.ok(command.includes(flag), flag);
    assert.doesNotMatch(command, /docker|--gpus|--context-length|--reasoning-parser|--tool-call-parser/);
    assert.match(cell.warn, /Quick generation.*full context.*unverified/);
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
    "--model deepreinforce-ai/Ornith-1.0-35B", "--served-model-name Ornith-1.0-35B",
    "--revision 5df2ed3f675c7beaa490328cc70bb573b65fb660",
    "--device cpu", "--tp 6", "--dtype bfloat16", "--max-total-tokens 8192",
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /-FP8|--reasoning-parser|--tool-call-parser|--speculative/);
});

test("Ornith 1.5 35B-A3B uses its independent BF16 TP6 record, not its FP8 sibling", async () => {
  const text = readFileSync(new URL("../../src/snippets/configs/ornith-ai/ornith-1.5.jsx", import.meta.url), "utf8");
  const { config } = await import("data:text/javascript," + encodeURIComponent(text));
  const cell = config.cells.find((candidate) => candidate.match.hw === "xeon" && candidate.match.variant === "35b");
  const command = engine(config).renderCommand(cell, cell.match, {});
  for (const flag of [
    "--model ornith-ai/Ornith-1.5-35B-A3B", "--served-model-name Ornith-1.5-35B-A3B",
    "--revision 10fbf86fed7ecee4a061f8b499a618f46001cac1",
    "--device cpu", "--tp 6", "--dtype bfloat16", "--max-total-tokens 8192",
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /-FP8|--reasoning-parser|--tool-call-parser|--speculative/);
});
