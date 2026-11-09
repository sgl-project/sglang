// Writes python/sglang/cli/_commands/, the Cookbook commands `sglang serve <model>` launches,
// one JSON file per checkpoint. The export-cookbook-commands pre-commit hook runs it; never
// edit the files by hand.
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../..", import.meta.url));
const docs = join(root, "docs");
const configsDir = join(docs, "src/snippets/configs");
const outputDir = join(root, "python/sglang/cli/_commands");
const MODES = ["balanced", "low-latency", "high-throughput"];
// TODO(Yichi): export every LLM/VLM config once the CLI Guide covers them.
const EXPORTED_CONFIGS = ["Qwen/qwen3.8-flash-next.jsx"];

const engineSource = readFileSync(join(docs, "src/snippets/_deployment.jsx"), "utf8");
const start = engineSource.indexOf("// ==== COMMAND ENGINE ====");
const end = engineSource.indexOf("// ==== end COMMAND ENGINE ====");
if (start < 0 || end < start) throw new Error("_deployment.jsx is missing the COMMAND ENGINE markers");
const createEngine = new Function(
  "config",
  `${engineSource.slice(start, end)}
  return { renderCommand, findCell, cellNnodes, cellVerifyStatus, resolveModelName, matchDimSpecs, overlayDimSpecs };`,
);

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
  );
const pages = walk(join(docs, "cookbook")).filter((path) => path.endsWith(".mdx"));
const pageUrl = (name) => {
  const hits = pages.filter((path) => {
    const source = readFileSync(path, "utf8");
    return source.includes(`/src/snippets/configs/${name}"`) && source.includes("<Deployment");
  });
  if (hits.length !== 1) throw new Error(`${name}: expected one Cookbook page rendering it, found ${hits.length}`);
  return `https://docs.sglang.io/${relative(docs, hits[0]).replace(/\.mdx$/, "")}`;
};

const exportConfig = async (name) => {
  const { config } = await import(
    "data:text/javascript," + encodeURIComponent(readFileSync(join(configsDir, name), "utf8"))
  );
  const engine = createEngine(config);
  if (engine.overlayDimSpecs.some((spec) => (spec.options || []).some((option) => MODES.includes(option.id)))) {
    throw new Error(`${name}: modes declared as an overlay dim are not exported yet`);
  }
  const modeDim = engine.matchDimSpecs.find((spec) => spec.options.some((option) => MODES.includes(option.id)));
  const placeholders = Object.fromEntries(
    Object.entries(config.placeholders || {}).map(([key, spec]) => [key, spec.default ?? ""]),
  );
  const overlayDefaults = Object.fromEntries(
    engine.overlayDimSpecs.map((spec) => [spec.id, spec.default ?? spec.options?.[0]?.id ?? ""]),
  );
  const url = pageUrl(name);
  const entries = new Map();
  let skipped = 0;
  for (const base of config.cells) {
    const sel = { ...base.match, ...overlayDefaults };
    const cell = engine.findCell(config.cells, sel);
    const runModes = (typeof config.runModes === "function" ? config.runModes(sel) : config.runModes) || ["python"];
    const command = engine.renderCommand(cell, sel, placeholders, "python");
    if (cell.pd || engine.cellNnodes(cell, sel) > 1 || !runModes.includes("python") || /<[\w-]+>|{{/.test(command)) {
      skipped++;
      continue;
    }
    const hash = new URLSearchParams(sel).toString();
    const [prefix] = command.split(/^sglang serve/m);
    const env = Object.fromEntries(
      prefix
        .split("\n")
        .map((line) => line.replace(/\s*\\$/, "").trim())
        .filter((line) => line && !line.startsWith("#"))
        .map((line) => line.split(/=(.*)/s).slice(0, 2)),
    );
    // test_serve_backends.py cross-checks this against ServerArgs.
    const size = (flag) => Number(command.match(new RegExp(`--${flag}(?:-size)? (\\d+)`))?.[1] ?? 1);
    entries.set(hash, {
      model: engine.resolveModelName(sel),
      hw: sel.hw,
      mode: modeDim ? sel[modeDim.id] : null,
      verified: engine.cellVerifyStatus(cell, sel),
      gpus: size("tp") * size("pp") * size("dp"),
      env,
      url: `${url}#${hash}`,
      command,
    });
  }
  if (!entries.size) throw new Error(`${name}: no command exported`);
  console.log(`${name}: ${entries.size} commands, ${skipped} multi-node, PD, docker-only or placeholder cells skipped`);
  return [...entries.values()];
};

// "Qwen/Qwen3-8B" -> "Qwen--Qwen3-8B.json"; Hugging Face names never contain "--".
const fileName = (model) => {
  if (!/^[\w.-]+\/[\w.-]+$/.test(model) || model.includes("--")) {
    throw new Error(`${model}: expected an org/name model path`);
  }
  return `${model.replace("/", "--")}.json`;
};

const byModel = {};
for (const name of EXPORTED_CONFIGS) {
  for (const { model, ...entry } of await exportConfig(name)) (byModel[model] ||= []).push(entry);
}
const files = Object.fromEntries(
  Object.entries(byModel).map(([model, entries]) => [
    fileName(model),
    `[\n${entries.map((e) => `  ${JSON.stringify(e)}`).join(",\n")}\n]\n`,
  ]),
);
mkdirSync(outputDir, { recursive: true });
const changed = [];
for (const name of readdirSync(outputDir)) {
  if (name in files) continue;
  rmSync(join(outputDir, name));
  changed.push(`removed ${name}`);
}
for (const [name, json] of Object.entries(files)) {
  const path = join(outputDir, name);
  if (existsSync(path) && readFileSync(path, "utf8") === json) continue;
  writeFileSync(path, json);
  changed.push(`wrote ${name}`);
}
console.log(`${relative(root, outputDir)}: ${Object.keys(files).length} files, ${changed.length} changed`);
// pre-commit only notices edits to tracked files, so a new file would pass unless this fails.
if (changed.length) {
  console.log(`  ${changed.join("\n  ")}`);
  process.exit(1);
}
