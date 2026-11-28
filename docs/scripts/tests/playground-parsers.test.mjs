import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const readSnippet = (path) =>
  readFileSync(new URL(`../../src/snippets/${path}`, import.meta.url), "utf8");
const source = readSnippet("_playground.jsx");
const slice = (startMarker, endMarker) => {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `missing Playground section ${startMarker}`);
  return source.slice(start, end);
};
// Load the production parser overrides without running the surrounding JSX.
const { parsers, helpers } = new Function(`
  ${slice("  const stripFlagsByFirstToken", "  // Insert after the first present anchor")}
  ${slice("  const hasFlag", "  // String arg")}
  const { parsers } = { ${slice("    parsers: {", "      render:")} } };
  return { parsers, helpers: { hasFlag, stripFlagsByFirstToken, insertBeforeTail } };
`)();
const { config } = await import(
  "data:text/javascript," + encodeURIComponent(
    readSnippet("configs/thinkingmachines/inkling-2-mini.jsx"),
  )
);
const parserConfig = config.playgroundFeatures.parsers;
const tpuCell = config.cells.find(({ match }) => match.hw === "tpu7x" && match.strategy === "dspark");

function applyParsers(cell, value) {
  return parsers.apply({
    flags: cell.flags,
    env: cell.env,
    value,
    fc: parserConfig,
    h: helpers,
    derived: parsers.deriveFromBase(cell, parserConfig, helpers),
  }).flags;
}

test("turning off reasoning preserves the TPU7x Inkling tool-call parser", () => {
  assert.ok(tpuCell.flags.includes("--tool-call-parser inkling"));

  const flags = applyParsers(tpuCell, { reasoning: false });
  assert.deepEqual(flags, tpuCell.flags.filter(flag => flag !== "--reasoning-parser inkling"));
});

test("turning off both parsers removes both flags", () => {
  const flags = applyParsers(tpuCell, { reasoning: false, toolCall: false });
  assert.ok(flags.every(flag => !/^--(reasoning|tool-call)-parser\b/.test(flag)));
});

test("enabling a parser absent from the base uses the configured flag", () => {
  const cell = config.cells.find(({ match }) => match.hw === "gb300" && match.strategy === "balanced");
  const flags = cell.flags.filter(flag => flag !== "--tool-call-parser inkling2");
  assert.deepEqual(applyParsers({ ...cell, flags }, { toolCall: true }), cell.flags);
});
