import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const source = readFileSync(new URL("../../src/snippets/autoregressive/laguna-xs2-deployment.jsx", import.meta.url), "utf8");
const setup = source.slice(source.indexOf("  const options ="), source.indexOf("  const [values, setValues]"));
const handler = source.slice(source.indexOf("  const handleRadioChange ="), source.indexOf("  const containerStyle ="));
const load = (values, setValues = () => {}) => new Function("values", "setValues", `${setup}
${handler}
return { options, generateCommand, handleRadioChange, getInitialState };`)(values, setValues);
const base = { hardware: "xeon", quantization: "bf16", reasoning: "enabled", toolcall: "enabled", dpAttention: "enabled" };

test("Laguna XS.2 emits only the independently validated TP4 BF16 CPU recipe", () => {
  const command = load(base).generateCommand();
  for (const flag of [
    "--model poolside/Laguna-XS.2", "--device cpu", "--tp 4", "--host 0.0.0.0",
  ]) assert.ok(command.includes(flag), flag);
  assert.match(command, /^sglang serve/);
  assert.doesNotMatch(command, /--revision|--dtype|--max-total-tokens|--port|--trust-remote-code|--disable-overlap-schedule|--reasoning-parser|--tool-call-parser|--attn-dp-size|docker/);
  for (const quantization of ["fp8", "nvfp4"]) {
    assert.match(load({ ...base, quantization }).generateCommand(), /^# Error: No validated Xeon/);
  }
});

test("Laguna XS.2 hardware changes clear stale quantization and hidden GPU axes", () => {
  let state = { ...base, hardware: "b200", quantization: "nvfp4" };
  const api = load(state, (update) => { state = update(state); });
  assert.deepEqual(api.getInitialState(), {
    hardware: "h200", quantization: "bf16", reasoning: "enabled",
    toolcall: "enabled", dpAttention: "disabled",
  });
  api.handleRadioChange("hardware", "xeon");
  assert.equal(state.quantization, "bf16");
  for (const axis of ["reasoning", "toolcall", "dpAttention"]) {
    assert.equal(state[axis], "disabled");
    assert.equal(api.options[axis].condition(state), false);
    assert.equal(api.options[axis].condition({ ...state, hardware: "b200" }), true);
  }
  assert.match(source, /if \(option\.condition && !option\.condition\(values\)\) return null/);
});
