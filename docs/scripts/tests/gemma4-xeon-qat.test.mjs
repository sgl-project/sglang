import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const source = readFileSync(new URL("../../src/snippets/autoregressive/gemma4-deployment.jsx", import.meta.url), "utf8");
const setup = source.slice(source.indexOf("  const options ="), source.indexOf("  const getInitialState"));
const handler = source.slice(source.indexOf("  const handleRadioChange ="), source.indexOf("  const handleCheckboxChange ="));
const load = (setValues = () => {}) => new Function("setValues", `${setup}
${handler}
return { options, generateCommand, handleRadioChange };
`)(setValues);
const base = {
  hardware: "xeon", modelSize: "e2b", checkpoint: "qat",
  reasoning: "enabled", toolcall: "enabled", speculative: "enabled",
};

for (const [modelSize, checkpoint, revision] of [
  ["e2b", "E2B", "6befbaca7398925921802abd1f277b495b78b738"],
]) {
  test(`Gemma 4 QAT ${checkpoint} emits only the independently verified CPU command`, () => {
    const command = load().generateCommand({ ...base, modelSize });
    for (const flag of [
      `--model-path google/gemma-4-${checkpoint}-it-qat-q4_0-unquantized`,
      `--revision ${revision}`, "--device cpu", "--tp 4",
      "--dtype bfloat16", "--max-total-tokens 8192",
      "--host 0.0.0.0 --port 30000", "SGLANG_USE_CPU_ENGINE=1",
    ]) assert.ok(command.includes(flag), flag);
    assert.doesNotMatch(command, /--reasoning-parser|--tool-call-parser|--speculative|--mem-fraction|docker|--gpus/);
  });
}

test("Xeon QAT hides unvalidated axes but keeps GPU options unchanged", () => {
  const { options } = load();
  for (const axis of ["reasoning", "toolcall", "speculative"]) {
    assert.equal(options[axis].condition(base), false);
    assert.equal(options[axis].condition({ ...base, hardware: "h200" }), true);
  }
});

test("Xeon QAT hardware switches clear stale parser and speculative selections", () => {
  let state = { ...base, hardware: "h200" };
  const api = load((update) => { state = update(state); });
  api.handleRadioChange("hardware", "xeon");
  assert.equal(state.checkpoint, "qat");
  for (const axis of ["reasoning", "toolcall", "speculative"]) {
    assert.equal(state[axis], "disabled");
  }
  api.handleRadioChange("modelSize", "31b");
  assert.equal(state.checkpoint, "standard");
});

test("Unvalidated Xeon QAT variants cannot generate a launch command", () => {
  const { generateCommand } = load();
  for (const modelSize of ["31b", "26b-a4b"]) {
    assert.match(generateCommand({ ...base, modelSize }), /^# Error: No validated Xeon QAT/);
  }
});
