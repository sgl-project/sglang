import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const file = "docs/src/snippets/autoregressive/intern-s2-preview-deployment.jsx";
const source = readFileSync(new URL("../../src/snippets/autoregressive/intern-s2-preview-deployment.jsx", import.meta.url), "utf8");
const load = (text, values, setValues = () => {}) => new Function("values", "setValues", `
${text.slice(text.indexOf("  const handleRadioChange ="), text.indexOf("  const containerStyle ="))}
return { generateCommand, handleRadioChange };`)(values, setValues);

test("Intern-S2-Preview Xeon command locks the verified checkpoint and topology", () => {
  const command = load(source, { hardware: "xeon", reasoning: "enabled", toolcall: "enabled", mtp: "enabled" }).generateCommand();
  for (const flag of [
    "SGLANG_USE_CPU_ENGINE=1 sglang serve", "--model internLM/Intern-S2-Preview",
    "--device cpu", "--disable-overlap-schedule",
    "--revision 4f57cab513689b089019fce4ad24e26520df183c",
    "--trust-remote-code", "--tp 4", "--dtype bfloat16",
    "--max-total-tokens 8192", "--port 30000",
  ]) assert.ok(command.includes(flag), flag);
  assert.doesNotMatch(command, /--reasoning-parser|--tool-call-parser|--speculative|--mamba-radix|docker/);
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

test("Intern-S2-Preview preserves every original H200 command", () => {
  const before = execFileSync("git", ["show", `2fb4f3db431190255a56ebcf03c8dd35a27573e3:${file}`], { encoding: "utf8" });
  for (const reasoning of ["enabled", "disabled"]) {
    for (const toolcall of ["enabled", "disabled"]) {
      for (const mtp of ["enabled", "disabled"]) {
        const values = { hardware: "h200", reasoning, toolcall, mtp };
        assert.equal(load(source, values).generateCommand(), load(before, values).generateCommand());
      }
    }
  }
});
