import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/snippets/_playground.jsx", import.meta.url), "utf8");
const configSource = readFileSync(new URL("../../src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx", import.meta.url), "utf8");
const { config } = await import("data:text/javascript," + encodeURIComponent(configSource));

// Mintlify inlines these component-local functions. Exercise the real command
// handler and mutation helpers without needing a browser or duplicating the
// flag-generation logic in the test.
const helpersStart = source.indexOf("  const matchConstraint =");
const helpersEnd = source.indexOf("  // -------- Prefill-CP flag family", helpersStart);
const axisStart = source.indexOf("    hicache: {");
const applyStart = source.indexOf("      apply: ", axisStart) + "      apply: ".length;
const renderStart = source.indexOf("\n      render: ", applyStart);
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart && axisStart >= 0 && renderStart > applyStart);
const applySource = source.slice(applyStart, renderStart).trim().replace(/,$/, "");
const makeEngine = new Function("config", `${source.slice(helpersStart, helpersEnd)}
  return { h: helpers, apply: (${applySource}) };
`);
const { h, apply } = makeEngine(config);
const fc = config.playgroundFeatures.hicache;
const selection = { hw: "h200", variant: "flash-1017", quant: "fp4", strategy: "high-throughput", nodes: "single" };
const base = config.cells.find((cell) => Object.entries(selection).every(([k, v]) => cell.match[k] === v));
assert.ok(base);
const enabled = { enable: true, backend: null, writePolicy: "write_back" };
const inherited = { enable: null, backend: null, writePolicy: "auto" };

const withRole = (role, flags = base.flags) => h.insertBeforeTail(
  h.stripFlagsByFirstToken(flags, ["--disaggregation-mode"]),
  role === "off" ? [] : [`--disaggregation-mode ${role}`],
);
const generate = (role, value = enabled, flags = withRole(role), feature = fc, derived = { enable: false }) => apply({
  flags, env: ["NCCL_SOCKET_IFNAME=eth0"], value, fc: feature,
  sel: { ...selection, pdMode: role }, h, derived,
});
const hasHiCache = (flags) => flags.some((f) => /^(?:--enable-hierarchical-cache|--hicache-)/.test(f));
const hasDecodeRadix = (flags) => h.hasFlag(flags, "--disaggregation-decode-enable-radix-cache");

test("V4.1 PD decode must not generate HiCache or its unsupported decode-radix flag", () => {
  const { flags, env } = generate("decode");
  assert.equal(hasHiCache(flags), false);
  assert.equal(hasDecodeRadix(flags), false);
  assert.ok(h.hasFlag(flags, "--tp"));
  assert.equal(h.findFlagArg(flags, "--disaggregation-mode"), "decode");
  assert.deepEqual(env, ["NCCL_SOCKET_IFNAME=eth0"]);
});

test("V4.1 aggregated and prefill roles keep the selected HiCache recipe", () => {
  for (const role of ["off", "prefill"]) {
    const { flags } = generate(role);
    assert.equal(h.hasFlag(flags, "--enable-hierarchical-cache"), true);
    assert.equal(h.findFlagArg(flags, "--hicache-write-policy"), "write_back");
    assert.equal(hasDecodeRadix(flags), false);
  }
});

test("switching from a HiCache-enabled prefill to decode clears inherited HiCache flags", () => {
  const { flags: prefillFlags } = generate("prefill");
  const { flags } = generate("decode", inherited, withRole("decode", prefillFlags), fc, { enable: true });
  assert.equal(hasHiCache(flags), false);
  assert.equal(hasDecodeRadix(flags), false);
});

test("models without a role exclusion retain the generic decode HiCache behavior", () => {
  const { flags } = generate("decode", enabled, withRole("decode"), { writePolicies: fc.writePolicies });
  assert.equal(h.hasFlag(flags, "--enable-hierarchical-cache"), true);
  assert.equal(hasDecodeRadix(flags), true);
});
