#!/usr/bin/env node
// Static guard for the cookbook deployment/playground engines and their configs.
// Zero dependencies, no browser, no Mintlify — plain `node`.
//
//   node docs/scripts/check_cookbook_configs.mjs
//
// What it protects, in order of how expensive the bug is to find by hand:
//
//   1. MIRROR drift. The overlay-resolution rule is written in both engines
//      because Mintlify snippets cannot import each other. If the copies drift,
//      the Deploy command and the playground's base disagree and the reader sees
//      phantom +/- lines in the diff — with no error anywhere.
//   2. Sibling identity. Overlay resolution clones the base cell, so sibling
//      detection must compare match dimensions rather than object references.
//   3. Config/engine contract. A cell keyed on a dimension the config no longer
//      declares silently stops matching; the panel just shows a different cell.
//   4. Predicate safety. showWhen / disabled / flags run against selections the
//      author never clicked through; a throw there blanks the whole widget.

import { readFileSync, readdirSync } from "node:fs";
import { basename, dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const SNIPPETS = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "snippets");
const CONFIGS = join(SNIPPETS, "configs");
const DIFFUSION_COOKBOOK = join(SNIPPETS, "..", "..", "cookbook", "diffusion");
const LEGACY_DIMS = ["variants", "quantizations", "strategies", "nodesOptions"];

const failures = [];
const fail = (where, msg) => failures.push(`${where}: ${msg}`);

// ---------------------------------------------------------------- 1. MIRROR
// Compare the marked blocks with comments and whitespace normalized away, so
// wording may differ per file but the rule may not.
const mirrorBody = (file) => {
  const src = readFileSync(join(SNIPPETS, file), "utf8");
  const start = src.indexOf("==== MIRROR");
  const end = src.indexOf("==== end MIRROR");
  if (start === -1 || end === -1) return null;
  return src
    .slice(src.indexOf("\n", start), end)
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("//"))
    .join(" ")
    .replace(/\s+/g, " ");
};

const a = mirrorBody("_deployment.jsx");
const b = mirrorBody("_playground.jsx");
if (a === null) fail("_deployment.jsx", "MIRROR markers missing");
if (b === null) fail("_playground.jsx", "MIRROR markers missing");
if (a && b && a !== b) {
  fail("MIRROR", "overlay resolution has drifted between the two engines");
  const [la, lb] = [a.split(" "), b.split(" ")];
  const i = la.findIndex((t, k) => t !== lb[k]);
  fail("MIRROR", `first divergence near token ${i}: `
    + `_deployment "${la.slice(i, i + 8).join(" ")}" vs `
    + `_playground "${lb.slice(i, i + 8).join(" ")}"`);
}

// `withOverlay` returns a clone, so object identity can never distinguish the
// current base cell from a true sibling. This previously made every cookbook
// show a spurious "matches … / switch base" hint before the reader changed
// anything.
const playgroundSource = readFileSync(join(SNIPPETS, "_playground.jsx"), "utf8");
if (/\bmatchedCell\s*!==\s*baseCell\b/.test(playgroundSource)) {
  fail("_playground.jsx", "sibling detection compares cloned cells by object identity");
}

// --------------------------------------------------------------- 3/4. Configs
// Configs are .jsx with a single `export const config` literal; import them
// through a data: URL so no temp file is needed.
const loadConfig = async (path) => {
  const src = readFileSync(path, "utf8");
  const mod = await import(
    "data:text/javascript," + encodeURIComponent(src)
  );
  return mod.config;
};

// Every combination of match dims + overlay dims the reader can produce.
const selectionSpace = (config) => {
  const dims = [
    { id: "hw", options: (config.supportedHardware || []).map((id) => ({ id })) },
    ...(config.matchDims || []),
    ...(config.overlayDims || []),
  ];
  let space = [{}];
  for (const d of dims) {
    const next = [];
    for (const partial of space) {
      for (const opt of (d.options || [])) next.push({ ...partial, [d.id]: opt.id });
    }
    space = next.length ? next : space;
    if (space.length > 20000) return space.slice(0, 20000); // cheap blow-up guard
  }
  return space;
};

const walk = (dir) => readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
  e.isDirectory() ? walk(join(dir, e.name))
    : (e.name.endsWith(".jsx")
      && !e.name.includes("benchmark")
      && e.name !== "popular-models.jsx"
      ? [join(dir, e.name)] : []));

for (const path of walk(CONFIGS)) {
  const where = relative(join(SNIPPETS, ".."), path);
  let config;
  try {
    config = await loadConfig(path);
  } catch (e) {
    fail(where, `does not parse as a module: ${e.message}`);
    continue;
  }
  if (!config) { fail(where, "no `export const config`"); continue; }

  const custom = Array.isArray(config.matchDims);

  // A config either declares its own dims or carries the full legacy set —
  // half of each means the engine silently renders a dimension nobody authored.
  if (!custom) {
    for (const k of LEGACY_DIMS) {
      if (!Array.isArray(config[k])) fail(where, `legacy config is missing \`${k}\``);
    }
  }

  const matchIds = ["hw", ...(custom
    ? config.matchDims.map((d) => d.id)
    : LEGACY_DIMS.map((k) => ({ variants: "variant", quantizations: "quant",
        strategies: "strategy", nodesOptions: "nodes" })[k]))];

  for (const [i, cell] of (config.cells || []).entries()) {
    const keys = Object.keys(cell.match || {}).sort();
    const want = [...matchIds].sort();
    if (keys.join(",") !== want.join(",")) {
      fail(where, `cells[${i}].match keys [${keys}] != declared dims [${want}]`);
    }
    for (const dim of (config.matchDims || [])) {
      const v = cell.match[dim.id];
      if (!(dim.options || []).some((o) => o.id === v)) {
        fail(where, `cells[${i}].match.${dim.id}="${v}" is not an option of that dim`);
      }
    }
    // Without a `nodes` dim the node count rides on the cell; a missing one
    // silently degrades a multi-node recipe to single-node.
    if (custom && !matchIds.includes("nodes") && cell.nnodes === undefined) {
      fail(where, `cells[${i}] has no \`nnodes\` and the config declares no nodes dim`);
    }
  }

  for (const dim of (config.overlayDims || [])) {
    const ids = (dim.options || []).map((o) => o.id);
    if (dim.default !== undefined && !ids.includes(dim.default)) {
      fail(where, `overlayDims.${dim.id}.default="${dim.default}" is not one of [${ids}]`);
    }
  }

  // Predicates and flag builders must survive every reachable selection.
  const space = selectionSpace(config);
  const probe = (fn, label) => {
    for (const sel of space) {
      try { fn(sel); } catch (e) {
        fail(where, `${label} throws on ${JSON.stringify(sel)}: ${e.message}`);
        return;
      }
    }
  };
  for (const dim of [...(config.matchDims || []), ...(config.overlayDims || [])]) {
    if (typeof dim.showWhen === "function") probe(dim.showWhen, `${dim.id}.showWhen`);
    for (const opt of (dim.options || [])) {
      const tag = `${dim.id}.${opt.id}`;
      if (typeof opt.showWhen === "function") probe(opt.showWhen, `${tag}.showWhen`);
      if (typeof opt.disabled === "function") probe(opt.disabled, `${tag}.disabled`);
      for (const key of ["flags", "env", "hints"]) {
        if (typeof opt[key] !== "function") continue;
        probe((sel) => {
          const out = opt[key](sel);
          if (out !== undefined && !Array.isArray(out)) throw new Error(`${key} returned ${typeof out}, expected an array`);
          for (const f of (out || [])) {
            if (typeof f !== "string") throw new Error(`${key} yielded a non-string entry`);
            if (/undefined|NaN/.test(f)) throw new Error(`${key} produced "${f}"`);
          }
        }, `${tag}.${key}`);
      }
    }
  }
  if (typeof config.curl === "function") {
    probe((sel) => {
      const out = config.curl(sel, null);
      if (typeof out !== "string") {
        throw new Error(`curl returned ${typeof out}, expected a string`);
      }
    }, "curl");
  }
}

// ----------------------------------------------------- Diffusion page opening
// Keep the first screen consistent across model pages. This check intentionally
// guards structure, not editorial judgment; the authoring skill carries the
// capability/strength/boundary rubric that cannot be reduced to keywords.
const walkMdx = (dir) => readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
  e.isDirectory() ? walkMdx(join(dir, e.name))
    : (e.name.endsWith(".mdx") ? [join(dir, e.name)] : []));

for (const path of walkMdx(DIFFUSION_COOKBOOK)) {
  if (["README.mdx", "intro.mdx"].includes(basename(path))) continue;

  const where = relative(join(SNIPPETS, "..", ".."), path);
  const src = readFileSync(path, "utf8");
  const heading = /^## 1\. Model Introduction\s*$/m.exec(src);
  if (!heading) {
    fail(where, "missing exact `## 1. Model Introduction` heading");
    continue;
  }

  const importPattern = /import\s+\{\s*DiffusionModelTags\s*\}\s+from\s+['"]\/src\/snippets\/diffusion\/model-tags\.jsx['"]/;
  if (!importPattern.test(src.slice(0, heading.index))) {
    fail(where, "missing the shared DiffusionModelTags import before section 1");
  }

  const tagMatch = /<DiffusionModelTags\s+tags=\{\[([^\]]+)]}\s*\/>/.exec(src.slice(0, heading.index));
  if (!tagMatch) {
    fail(where, "missing `<DiffusionModelTags tags={[...]} />` before section 1");
  } else {
    const tags = [...tagMatch[1].matchAll(/["']([^"']+)["']/g)].map((m) => m[1].trim());
    if (tags.length < 4 || tags.length > 6) {
      fail(where, `tag widget has ${tags.length} tags; expected 4–6`);
    }
    if (tags.some((tag) => !tag)) fail(where, "tag widget contains an empty tag");
  }

  const introStart = heading.index + heading[0].length;
  const rest = src.slice(introStart);
  const boundaries = ["\n|", "\n<Warning", "\n<Note", "\n## 2"]
    .map((marker) => rest.indexOf(marker))
    .filter((i) => i >= 0);
  const lead = rest.slice(0, boundaries.length ? Math.min(...boundaries) : rest.length).trim();
  const paragraphs = lead.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  if (paragraphs.length < 2) {
    fail(where, "model introduction needs at least two lead paragraphs");
  }
  const prose = lead
    .replace(/\[([^\]]+)]\([^)]+\)/g, "$1")
    .replace(/`[^`]+`/g, "value")
    .replace(/[*_#]/g, " ");
  const words = prose.match(/[A-Za-z0-9][A-Za-z0-9+./@–—-]*/g) || [];
  if (words.length < 45 || words.length > 180) {
    fail(where, `lead introduction has ${words.length} words; expected 45–180`);
  }
}

if (failures.length) {
  console.error(`FAIL (${failures.length})`);
  for (const f of failures) console.error("  - " + f);
  process.exit(1);
}
console.log("cookbook config check: OK");
