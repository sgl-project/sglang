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
// Every config a Deployment renders on an LLM/VLM page; diffusion pages come later.
const EXPORTED_PAGES = join(docs, "cookbook/autoregressive");
// Hardware the CLI Guide cannot detect from a device name: CPU, Apple, TPU and Ascend NPU, whose
// commands target other builds, and Vera Rubin, whose device name no cell has recorded yet.
const UNDETECTED_HARDWARE = new Set(["xeon", "mac", "tpu7x", "a3", "a5", "vera-rubin"]);

const engineSource = readFileSync(join(docs, "src/snippets/_deployment.jsx"), "utf8");
const start = engineSource.indexOf("// ==== COMMAND ENGINE ====");
const end = engineSource.indexOf("// ==== end COMMAND ENGINE ====");
if (start < 0 || end < start) throw new Error("_deployment.jsx is missing the COMMAND ENGINE markers");
const createEngine = new Function(
  "config",
  `${engineSource.slice(start, end)}
  return { renderCommand, findCell, cellNnodes, cellVerifyStatus, resolveModelName, validateSelection, matchDimSpecs, overlayDimSpecs };`,
);

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
  );
const imported = (source, binding) =>
  source.match(new RegExp(`import\\s*{\\s*${binding}\\s*}\\s*from\\s*"/src/snippets/configs/([^"]+)"`))?.[1];
const load = async (name) =>
  import("data:text/javascript," + encodeURIComponent(readFileSync(join(configsDir, name), "utf8")));
// Config name -> the Cookbook pages whose Deployment renders it.
const pagesByConfig = new Map();
for (const path of walk(join(docs, "cookbook")).filter((path) => path.endsWith(".mdx"))) {
  const source = readFileSync(path, "utf8");
  const name = imported(source, "config");
  if (name && source.includes("<Deployment")) pagesByConfig.set(name, [...(pagesByConfig.get(name) || []), path]);
}
const exportedConfigs = [...pagesByConfig]
  .filter(([, paths]) => paths.some((path) => path.startsWith(`${EXPORTED_PAGES}/`)))
  .map(([name, paths]) => {
    if (paths.length !== 1) throw new Error(`${name}: expected one Cookbook page rendering it, found ${paths.length}`);
    return { name, url: `https://docs.sglang.io/${relative(docs, paths[0]).replace(/\.mdx$/, "")}` };
  })
  .sort((a, b) => a.name.localeCompare(b.name));

// `sglang serve <org>/<name>` keys the Guide; a local directory or a single GGUF file does not.
const HF_MODEL = /^[\w.-]+\/[\w.-]+$/;

// GPUs one replica takes, from the flags after `sglang serve`; test_serve_backends.py
// cross-checks it against ServerArgs.
const gpuCount = (flags) => {
  const size = (...names) => {
    for (const name of names) {
      const value = flags.match(new RegExp(`(?:^|\\s)--${name}[ =](\\d+)`))?.[1];
      if (value) return Number(value);
    }
    return 1;
  };
  return size("tp", "tp-size", "tensor-parallel-size") * size("pp", "pp-size", "pipeline-parallel-size") * size("dp", "dp-size", "data-parallel-size");
};

const exportConfig = async ({ name, url }) => {
  const { config } = await load(name);
  const engine = createEngine(config);
  const isModeDim = (spec) => (spec.options || []).some((option) => MODES.includes(option.id));
  const modeDim = [...engine.matchDimSpecs, ...engine.overlayDimSpecs].find(isModeDim);
  const overlayModes = modeDim && engine.overlayDimSpecs.includes(modeDim) ? modeDim.options : [null];
  const placeholders = Object.fromEntries(
    Object.entries(config.placeholders || {}).map(([key, spec]) => [key, spec.default ?? ""]),
  );
  const byRecipe = new Map();
  let skipped = 0;
  for (const base of config.cells) {
    for (const option of overlayModes) {
      // The selection the page shows for this cell, overlays at their defaults.
      const sel = engine.validateSelection(config.cells, option ? { ...base.match, [modeDim.id]: option.id } : base.match);
      if (option && sel[modeDim.id] !== option.id) continue;
      const cell = engine.findCell(config.cells, sel);
      const runModes = (typeof config.runModes === "function" ? config.runModes(sel) : config.runModes) || ["python"];
      const command = engine.renderCommand(cell, sel, placeholders, "python");
      // Hint comments may name a flag or a placeholder without needing it.
      const uncommented = command.replace(/^#.*$/gm, "");
      const model = engine.resolveModelName(sel);
      if (
        !HF_MODEL.test(model) ||
        UNDETECTED_HARDWARE.has(sel.hw) ||
        // A "Not offered" banner in place of a command.
        cell.redirect ||
        cell.pd ||
        /--disaggregation-mode\b/.test(command) ||
        engine.cellNnodes(cell, sel) > 1 ||
        !runModes.includes("python") ||
        // Placeholders left to fill.
        /<[\w-]+>|{{/.test(uncommented)
      ) {
        skipped++;
        continue;
      }
      const [prefix, flags] = uncommented.split(/^sglang serve/m);
      const env = Object.fromEntries(
        prefix
          .split("\n")
          .map((line) => line.replace(/\s*\\$/, "").trim())
          .filter(Boolean)
          .map((line) => line.split(/=(.*)/s).slice(0, 2)),
      );
      const quoted = Object.keys(env).find((key) => /["']/.test(env[key]));
      if (quoted) throw new Error(`${name}: ${quoted} is quoted, but the CLI Guide passes env values verbatim`);
      const recipe = {
        model,
        hw: sel.hw,
        mode: modeDim ? sel[modeDim.id] : null,
        verified: engine.cellVerifyStatus(cell, sel),
        gpus: gpuCount(flags),
        env,
        url: `${url}#${new URLSearchParams(sel).toString()}`,
        command,
      };
      // The Guide picks by hardware, GPU count and mode; when the page has several commands for one
      // pick, keep the first cell's, as the page seeds from its first cell and falls back to the first match.
      const key = `${recipe.model} ${recipe.gpus}x ${recipe.hw} ${recipe.mode ?? "(no mode)"}`;
      if (!byRecipe.has(key)) byRecipe.set(key, recipe);
    }
  }
  console.log(`${name}: ${byRecipe.size} commands, ${skipped} not-offered, local-path, undetected-hardware, multi-node, PD, docker-only or placeholder cells skipped`);
  return [...byRecipe.values()];
};

// "Qwen/Qwen3-8B" -> "Qwen--Qwen3-8B.json"; Hugging Face names never contain "--".
const fileName = (model) => {
  if (model.includes("--")) throw new Error(`${model}: expected an org/name model path`);
  return `${model.replace("/", "--")}.json`;
};

const byModel = {};
for (const exported of exportedConfigs) {
  for (const { model, ...entry } of await exportConfig(exported)) (byModel[model] ||= []).push(entry);
}
// The CLI Guide picks by hardware, GPU count and mode, so a second match would never run.
for (const [model, entries] of Object.entries(byModel)) {
  const keys = entries.map(({ hw, gpus, mode }) => `${gpus}x ${hw} ${mode ?? "(no mode)"}`);
  const duplicate = keys.find((key, i) => keys.indexOf(key) !== i);
  if (duplicate) throw new Error(`${model}: two commands for ${duplicate}`);
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
