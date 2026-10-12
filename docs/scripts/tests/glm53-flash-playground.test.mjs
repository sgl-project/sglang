import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/snippets/_playground.jsx", import.meta.url), "utf8");
const configSource = readFileSync(new URL("../../src/snippets/configs/zai-org/glm-5.3-flash.jsx", import.meta.url), "utf8");
const { config } = await import("data:text/javascript," + encodeURIComponent(configSource));

// Execute the production flag pipeline without React/Mintlify. Only the JSX
// render methods are omitted; the config, constraints and flag helpers are real.
const prefix = source.slice(source.indexOf("  const DIMENSIONS"), source.indexOf("  const AXIS_HANDLERS"));
const axis = (id) => {
  const start = source.indexOf(`\n    ${id}: {`);
  const end = source.indexOf("\n      render:", start);
  return source.slice(start, end) + "\n    },";
};
const pipelineStart = source.indexOf("  const applyAllDeltas =");
const pipelineEnd = source.indexOf("\n  // Renderer", pipelineStart);
const engine = new Function("config", prefix + "\nconst AXIS_HANDLERS = {"
  + ["attention", "moe", "pdDisagg"].map(axis).join("\n") + "};\n"
  + source.slice(pipelineStart, pipelineEnd)
  + "return { AXIS_HANDLERS, helpers, applyAllDeltas, evaluateChip };")(config);
const { AXIS_HANDLERS: handlers, helpers, evaluateChip } = engine;
const cellFor = (hw) => config.cells.find((c) => c.match.hw === hw && c.match.quant === "fp8" && c.match.strategy === "low-latency");
const compose = (hw, picks = {}, suppliedCell) => {
  const cell = suppliedCell || cellFor(hw);
  const base = { ...cell.match, pdMode: "off" };
  const deltas = {}, derived = {};
  for (const [id, handler] of Object.entries(handlers)) {
    const fc = config.playgroundFeatures[id];
    if (!fc) continue;
    deltas[id] = { ...handler.initState(fc, base), ...(picks[id] || {}) };
    derived[id] = handler.deriveFromBase?.(cell, fc, helpers);
  }
  return engine.applyAllDeltas(cell.flags, cell.env, deltas, base, derived).flags;
};
const arg = (flags, head) => helpers.findFlagArg(flags, head);

test("a fresh cookbook opens on the B300 FP8 low-latency recipe", () => {
  assert.deepEqual(config.cells[0].match, { hw: "b300", strategy: "low-latency", quant: "fp8" });
});

test("CP8 is available on B200/B300 and emits interleave CP flags", () => {
  const cp = config.playgroundFeatures.attention.knobs.find((k) => k.id === "cp");
  const entry = helpers.findEntry(cp.values, 8);
  assert.notEqual(entry, null);
  for (const hw of config.supportedHardware) {
    assert.equal(evaluateChip(entry, { hw }).disabled, !["b200", "b300"].includes(hw), hw);
  }
  for (const hw of ["b200", "b300"]) {
    const flags = compose(hw, { attention: { cp: 8 } });
    assert.equal(arg(flags, "--attn-cp-size"), "8");
    assert.ok(flags.includes("--enable-prefill-cp"));
    assert.equal(arg(flags, "--cp-strategy"), "interleave");
  }
});

test("CP restricts both displayed and emitted EP to 1 or CP size", () => {
  for (const cp of [4, 8]) {
    for (const ep of [null, 1, 2, 4, 8]) {
      const flags = compose("b300", { attention: { tp: cp, cp }, moe: { ep } });
      const expected = ep === cp ? cp : 1;
      assert.equal(Number(arg(flags, "--ep") || arg(flags, "--ep-size") || 1), expected);
      assert.equal(arg(flags, "--moe-dense-tp-size"), expected > 1 ? String(cp) : null);
      const state = handlers.moe.resolveEp({ ep }, config.playgroundFeatures.moe, { ep: null }, cp);
      assert.equal(state.ep, expected);
      assert.deepEqual(state.entries, [1, cp]);
    }
  }
});

test("inherited invalid EP is normalized and CP-off restores ordinary EP", () => {
  const cell = { ...cellFor("b300"), flags: [...cellFor("b300").flags, "--ep-size 2"] };
  assert.equal(arg(compose("b300", { attention: { cp: 8 } }, cell), "--ep-size"), null);
  const off = compose("b300", { attention: { cp: 1 }, moe: { ep: 2 } });
  assert.equal(arg(off, "--ep"), "2");
  assert.equal(arg(off, "--moe-dense-tp-size"), null);
});

test("CP coupling is opt-in and does not mutate other cookbook commands", () => {
  const fc = { ep: { values: [null, 1, 2, 4, 8] } };
  const flags = ["--model-path example", "--tp 8", "--attn-cp-size 8", "--enable-prefill-cp", "--ep 2"];
  const value = handlers.moe.initState();
  const derived = handlers.moe.deriveFromBase({ flags }, fc, helpers);
  const result = handlers.moe.apply({ flags, env: [], value, fc, sel: {}, h: helpers, derived });
  assert.deepEqual(result.flags, flags);
  assert.deepEqual(handlers.moe.resolveEp(value, fc, derived, 8), {
    coupled: false, ep: 2, entries: [null, 1, 2, 4, 8],
  });
});

test("CP+EP replaces a baked dense-TP flag exactly once without mutating the recipe", () => {
  const cell = { ...cellFor("b300"), flags: [...cellFor("b300").flags, "--moe-dense-tp-size 1"] };
  const before = [...cell.flags];
  const flags = compose("b300", { attention: { cp: 8 }, moe: { ep: 8 } }, cell);
  assert.deepEqual(flags.filter((f) => f.startsWith("--moe-dense-tp-size")), ["--moe-dense-tp-size 8"]);
  assert.deepEqual(cell.flags, before);
});

test("Blackwell and Hopper inherit their respective MoE backend selections", () => {
  const fc = config.playgroundFeatures.moe;
  for (const hw of ["b200", "b300", "gb200", "gb300", "h100", "h200", "mi355x"]) {
    const expected = ["h100", "h200"].includes(hw) ? "deep_gemm" : hw === "mi355x" ? "aiter" : "flashinfer_trtllm";
    const derived = handlers.moe.deriveFromBase(cellFor(hw), fc, helpers);
    const opt = fc.backend.options.find((o) => o.id === derived.backend);
    assert.ok(opt, `${hw}: missing selected backend option`);
    assert.equal(derived.backend, expected);
    assert.equal(evaluateChip(opt, { hw }).disabled, false);
    assert.equal(arg(compose(hw), "--moe-runner-backend"), expected);
  }
});

test("PD defaults off and emits only Mooncake for either selected role", () => {
  const fc = config.playgroundFeatures.pdDisagg;
  assert.ok(fc);
  assert.deepEqual(fc.transferBackends.map((o) => o.id), ["mooncake"]);
  assert.equal(handlers.pdDisagg.initState(fc, cellFor("b300").match).mode, "off");
  assert.equal(arg(compose("b300"), "--disaggregation-mode"), null);
  for (const mode of ["prefill", "decode"]) {
    const flags = compose("b300", { attention: { cp: 8 }, pdDisagg: { mode } });
    assert.equal(arg(flags, "--disaggregation-mode"), mode);
    assert.equal(arg(flags, "--disaggregation-transfer-backend"), "mooncake");
    assert.equal(arg(flags, "--attn-cp-size"), null);
    assert.ok(!flags.includes("--enable-prefill-cp"));
  }
});
