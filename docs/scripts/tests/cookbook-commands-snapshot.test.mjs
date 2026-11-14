// Temporary: guards the cookbook engine through the CLI Guide refactor; delete once its last PR lands.
// Checks whether engine changes affect existing command text on cookbook.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const docs = fileURLToPath(new URL("../..", import.meta.url));
const configsDir = join(docs, "src/snippets/configs");
const snapshotPath = fileURLToPath(new URL("cookbook-commands.snapshot.json", import.meta.url));
const engineSource = readFileSync(join(docs, "src/snippets/_deployment.jsx"), "utf8");
const START = "  // ==== COMMAND ENGINE ====";
const END = "  // ==== end COMMAND ENGINE ====";
const start = engineSource.indexOf(START);
const end = engineSource.indexOf(END);
assert.ok(start > 0 && end > start, `_deployment.jsx no longer contains "${START}" ... "${END}"; update the slice markers`);
const createEngine = new Function(
  "config",
  `${engineSource.slice(start, end)}\nreturn { renderCommand, findCell };`,
);

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
  );
const hash = (text) => createHash("sha256").update(text).digest("hex").slice(0, 12);
const selectionKey = (sel) =>
  Object.keys(sel).sort().map((key) => `${key}=${sel[key]}`).join(",");
const RECIPE_META = new Set(["id", "default", "unverified", "label", "description"]);

const renderAll = async () => {
  const configs = {};
  const commands = {};
  const files = walk(configsDir).filter((path) => path.endsWith(".jsx") && !path.endsWith("-benchmarks.jsx")).sort();
  for (const file of files) {
    const source = readFileSync(file, "utf8");
    const { config } = await import("data:text/javascript," + encodeURIComponent(source));
    if (!config) continue;
    const name = relative(configsDir, file);
    const entries = {};
    const engine = createEngine(config);
    const env = Object.fromEntries(Object.entries(config.placeholders || {}).map(([key, spec]) => [key, spec.default ?? ""]));
    const overlayDefaults = Object.fromEntries((config.overlayDims || []).map((spec) => [spec.id, spec.default ?? spec.options?.[0]?.id ?? ""]));
    const modesOf = (sel) => (typeof config.runModes === "function" ? config.runModes(sel) : config.runModes) || ["python", "docker"];
    const record = (sel, label, cell) => {
      const command = engine.renderCommand(cell, sel, env, label === "docker" ? "docker" : "python");
      const key = `${selectionKey(sel)} :: ${label}`;
      entries[key] = hash(command);
      commands[`${name} :: ${key}`] = command;
    };

    if (config.commandBuilder) {
      const builder = config.commandBuilder;
      const recipes = [{}, ...(builder.resource?.verifiedRecipes || [])];
      for (const recipe of recipes) {
        const picks = Object.fromEntries(Object.entries(recipe).filter(([key, value]) => !RECIPE_META.has(key) && typeof value !== "function"));
        const sel = { ...builder.defaultSelection, ...overlayDefaults, ...picks };
        const { cell } = builder.resolveDeployment(sel);
        for (const mode of modesOf(sel)) record(sel, mode, cell);
      }
    } else {
      for (const base of config.cells || []) {
        const sel = { ...base.match, ...overlayDefaults };
        const cell = engine.findCell(config.cells, sel);
        for (const mode of modesOf(sel)) record(sel, mode, cell);
        const baseline = engine.renderCommand(cell, sel, env, "python");
        for (const spec of config.overlayDims || []) {
          for (const option of spec.options || []) {
            if (option.id === sel[spec.id]) continue;
            const variant = { ...sel, [spec.id]: option.id };
            const variantCell = engine.findCell(config.cells, variant);
            if (engine.renderCommand(variantCell, variant, env, "python") === baseline) continue;
            record(variant, "python", variantCell);
          }
        }
      }
    }
    configs[name] = {
      source: hash(source),
      commands: Object.fromEntries(Object.keys(entries).sort().map((key) => [key, entries[key]])),
    };
  }
  return { configs, commands };
};

test("cookbook commands match the snapshot", async (t) => {
  const { configs, commands } = await renderAll();
  const total = Object.values(configs).reduce((sum, config) => sum + Object.keys(config.commands).length, 0);
  assert.ok(total > 1000, "expected the snapshot to cover every cookbook cell");

  if (process.env.UPDATE_SNAPSHOTS) {
    writeFileSync(snapshotPath, JSON.stringify(configs, null, 1) + "\n");
    return;
  }

  const expected = JSON.parse(readFileSync(snapshotPath, "utf8"));
  const compared = Object.keys(configs).filter((name) => expected[name]?.source === configs[name].source);
  const skipped = Object.keys(configs).length - compared.length;
  if (skipped) t.diagnostic(`${skipped} configs changed since the snapshot and were not compared.`);

  const changed = [];
  const added = [];
  const removed = [];
  for (const name of compared) {
    const now = configs[name].commands;
    const before = expected[name].commands;
    for (const key of Object.keys(now)) {
      if (!(key in before)) added.push(`${name} :: ${key}`);
      else if (before[key] !== now[key]) changed.push(`${name} :: ${key}`);
    }
    for (const key of Object.keys(before)) if (!(key in now)) removed.push(`${name} :: ${key}`);
  }
  if (changed.length || added.length || removed.length) {
    const show = (keys) => keys.slice(0, 5).map((key) => `  ${key}\n${commands[key] ?? ""}`).join("\n\n");
    assert.fail([
      `cookbook commands differ from ${relative(docs, snapshotPath)}:`,
      `${changed.length} changed, ${added.length} added, ${removed.length} removed.`,
      changed.length ? `Changed (command rendered now):\n${show(changed)}` : "",
      added.length ? `Added:\n${show(added)}` : "",
      removed.length ? `Removed:\n  ${removed.slice(0, 5).join("\n  ")}` : "",
      "Only if the change is intended, regenerate the snapshot with:",
      "  UPDATE_SNAPSHOTS=1 node --test docs/scripts/tests/cookbook-commands-snapshot.test.mjs",
    ].filter(Boolean).join("\n"));
  }
});
