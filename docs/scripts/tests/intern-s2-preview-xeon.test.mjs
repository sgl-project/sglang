import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const source = readFileSync(new URL("../../src/snippets/autoregressive/intern-s2-preview-deployment.jsx", import.meta.url), "utf8");
const load = (text, values, setValues = () => {}) => new Function("values", "setValues", `
${text.slice(text.indexOf("  const handleRadioChange ="), text.indexOf("  const containerStyle ="))}
return { generateCommand, handleRadioChange };`)(values, setValues);

test("Intern-S2-Preview Xeon command keeps only the model and CPU topology", () => {
  const command = load(source, { hardware: "xeon", reasoning: "enabled", toolcall: "enabled", mtp: "enabled" }).generateCommand();
  for (const flag of [
    "sglang serve", "--model internLM/Intern-S2-Preview",
    "--device cpu", "--disable-overlap-schedule", "--tp 4", "--host 0.0.0.0",
  ]) assert.ok(command.includes(flag), flag);
  assert.match(command, /^sglang serve/);
  assert.doesNotMatch(command, /--revision|--dtype|--max-total-tokens|--port|--trust-remote-code|--reasoning-parser|--tool-call-parser|--speculative|--mamba-radix|docker/);
});

test("Intern-S2-Preview hardware switches reset and lock unvalidated axes", () => {
  let state = { hardware: "h200", reasoning: "enabled", toolcall: "enabled", mtp: "enabled" };
  const api = load(source, state, (update) => { state = update(state); });
  api.handleRadioChange("hardware", "xeon");
  assert.deepEqual(state, { hardware: "xeon", reasoning: "disabled", toolcall: "disabled", mtp: "disabled" });
  const before = state;
  for (const axis of ["reasoning", "toolcall", "mtp"]) api.handleRadioChange(axis, "enabled");
  assert.equal(state, before);
  api.handleRadioChange("hardware", "h200");
  api.handleRadioChange("mtp", "enabled");
  assert.equal(state.mtp, "enabled");
  assert.match(source, /disabled=\{isDisabled\}/);
  assert.match(source, /Only basic text generation has been validated/);
});
