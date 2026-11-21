// Run with: node --test docs/scripts/tests/deepseek-v4_1-playground.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const root = new URL("../../", import.meta.url);
const configSource = readFileSync(
  new URL("src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx", root),
  "utf8",
);
const configLiteral = configSource
  .slice(configSource.indexOf("export const config =") + "export const config =".length)
  .trim()
  .replace(/;\s*$/, "");
const config = new Function(`return (${configLiteral});`)();

const playgroundSource = readFileSync(
  new URL("src/snippets/_playground.jsx", root),
  "utf8",
);
const rendererStart = playgroundSource.indexOf("  const renderCommandLines =");
const rendererEnd = playgroundSource.indexOf("\n  // ===", rendererStart);
assert.ok(rendererStart >= 0 && rendererEnd > rendererStart, "Could not locate renderCommandLines");
const rendererSource = playgroundSource.slice(rendererStart, rendererEnd);
const renderCommandLines = new Function(
  "config",
  "resolveModelName",
  "interpolate",
  "parseNnodes",
  "ATTN_DP_HEADS",
  "PD_PORTS",
  "pgFeatures",
  "findFlagArg",
  "resolveRouter",
  "matchConstraint",
  `${rendererSource}\nreturn renderCommandLines;`,
)(
  config,
  (selection) => config.modelNames[`${selection.variant}|${selection.quant}`] || "test-model",
  (text, values, modelName) => text
    .replace(/\{\{([A-Z0-9_]+)\}\}/g, (_, key) => values[key] ?? `{{${key}}}`)
    .replaceAll("{{MODEL_NAME}}", modelName),
  (nodes) => Number.parseInt(nodes, 10) || 1,
  ["--attn-dp-size", "--attention-data-parallel-size"],
  {},
  config.playgroundFeatures || {},
  () => null,
  () => null,
  () => true,
);

const selection = {
  hw: "a5",
  variant: "flash-1017",
  quant: "fp4",
  strategy: "balanced",
  nodes: "single",
};
const cell = config.cells.find((candidate) =>
  candidate.match.hw === selection.hw
  && candidate.match.variant === selection.variant
  && candidate.match.quant === selection.quant
  && candidate.match.strategy === selection.strategy
  && candidate.match.nodes === selection.nodes,
);
assert.ok(cell, "A5 DeepSeek-V4.1 recipe was not found");
const envValues = {
  MODEL_PATH: "/models/dsv4.1",
  NODE0_IP: "192.168.25.209",
  NODE_RANK: "0",
  HCCL_INTERFACE: "lo",
  GLOO_INTERFACE: "lo",
  PORT: "30000",
  HF_TOKEN: "test-token",
};

test("A5 Playground Python mode uses the configured setup and launch hooks", () => {
  const command = renderCommandLines(cell, cell.flags, cell.env, selection, envValues, null, "python");
  assert.ok(command.includes("source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"));
  assert.ok(command.includes("source /usr/local/Ascend/nnal/atb/set_env.sh"));
  assert.ok(command.includes("python3 -m sglang.launch_server"));
  assert.ok(command.includes("HCCL_OP_EXPANSION_MODE=AIV"));
  assert.ok(command.includes("--attn-dp-size 8"));
  assert.doesNotMatch(command, /--enable-dp-attention|--dp-size 8/);
  assert.ok(command.includes("--model-path /models/dsv4.1"));
});

test("A5 Cookbook exposes only Python mode and has no A5 Docker recipe", () => {
  assert.deepEqual(config.runModes(selection), ["python"]);
  assert.equal(config.dockerImages.a5, undefined);
  assert.equal(config.dockerRunCommand, undefined);
  assert.equal(config.dockerMounts, undefined);
});
