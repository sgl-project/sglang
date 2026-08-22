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
//   5. PD/speculation reachability. A PD card must stay reachable when a base
//      cell inherits an algorithm it declares incompatible, so choosing a PD
//      role can remove that algorithm from the generated command.

import { readFileSync, readdirSync } from "node:fs";
import { basename, dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const SNIPPETS = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "snippets");
const CONFIGS = join(SNIPPETS, "configs");
const DIFFUSION_COOKBOOK = join(SNIPPETS, "..", "..", "cookbook", "diffusion");
const COOKBOOK_MODEL_TEMPLATE = join(
  SNIPPETS, "..", "..", "..", ".claude", "skills", "cookbook-add-model",
  "templates", "config.jsx.tmpl");
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

const cookbookModelTemplate = readFileSync(COOKBOOK_MODEL_TEMPLATE, "utf8");
for (const oldName of [
  "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS",
  "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND",
]) {
  if (cookbookModelTemplate.includes(oldName)) {
    fail("config.jsx.tmpl", `still emits removed W4A4 setting ${oldName}`);
  }
}
if (!cookbookModelTemplate.includes("--enable-w4a4-mxfp4-megamoe")) {
  fail("config.jsx.tmpl", "W4A4 MegaMoE option is missing the server flag");
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
  let space = [{ ...(config.commandBuilder?.defaultSelection || {}) }];
  for (const d of dims) {
    const next = [];
    const options = d.kind === "number"
      ? [...new Set([d.default, d.min, d.max])].map((id) => ({ id }))
      : (d.options || []);
    for (const partial of space) {
      for (const opt of options) next.push({ ...partial, [d.id]: opt.id });
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
    if (dim.kind === "number") {
      if (!Number.isInteger(dim.min) || !Number.isInteger(dim.max) || dim.min > dim.max) {
        fail(where, `overlayDims.${dim.id} has invalid numeric bounds`);
      }
      if (!Number.isInteger(dim.default) || dim.default < dim.min || dim.default > dim.max) {
        fail(where, `overlayDims.${dim.id}.default is outside [${dim.min}, ${dim.max}]`);
      }
    } else if (dim.default !== undefined && !ids.includes(dim.default)) {
      fail(where, `overlayDims.${dim.id}.default="${dim.default}" is not one of [${ids}]`);
    }
  }

  const builder = config.commandBuilder;
  if (builder) {
    const scopes = new Set(["base", "serve", "request"]);
    for (const dim of (config.overlayDims || [])) {
      if (!scopes.has(dim.scope)) {
        fail(where, `builder dimension ${dim.id} has invalid scope "${dim.scope}"`);
      }
    }
    if (!builder.defaultSelection || typeof builder.defaultSelection !== "object") {
      fail(where, "commandBuilder.defaultSelection is required");
    }
    if (typeof builder.resource?.autoTopology !== "function") {
      fail(where, "commandBuilder.resource.autoTopology must be a function");
    }
    if (typeof builder.resource?.validateTopology !== "function") {
      fail(where, "commandBuilder.resource.validateTopology must be a function");
    }
    if (typeof builder.resolveDeployment !== "function") {
      fail(where, "commandBuilder.resolveDeployment must be a function");
    }
    for (const [key, bounds] of Object.entries(builder.resource?.limits || {})) {
      if (!Number.isInteger(bounds.min) || !Number.isInteger(bounds.max) || bounds.min > bounds.max) {
        fail(where, `commandBuilder.resource.limits.${key} is invalid`);
      }
      const value = builder.defaultSelection?.[key];
      if (!Number.isInteger(value) || value < bounds.min || value > bounds.max) {
        fail(where, `commandBuilder.defaultSelection.${key} is outside its bounds`);
      }
    }

    const selectionOf = (extra = {}) => {
      const defaults = { ...(builder.defaultSelection || {}) };
      for (const dim of (config.overlayDims || [])) defaults[dim.id] = dim.default;
      return { ...defaults, ...extra };
    };
    const validateResolved = (selection, label, expectVerified = false) => {
      let resolved;
      try {
        resolved = builder.resolveDeployment(selection);
      } catch (e) {
        fail(where, `${label} resolver throws: ${e.message}`);
        return null;
      }
      if (!resolved || !Array.isArray(resolved.flags) || !resolved.builder) {
        fail(where, `${label} resolver must return a cell with flags and builder metadata`);
        return resolved;
      }
      if (!Array.isArray(resolved.builder.errors) || !Array.isArray(resolved.builder.warnings)) {
        fail(where, `${label} resolver errors/warnings must be arrays`);
      }
      if (expectVerified && (resolved.builder.errors?.length || resolved.builder.verification?.serve !== "verified")) {
        fail(where, `${label} is declared verified but resolved as ${resolved.builder.verification?.serve || "invalid"}`);
      }

      let flags = [...resolved.flags];
      for (const dim of (config.overlayDims || [])) {
        const option = (dim.options || []).find((entry) => entry.id === selection[dim.id]);
        if (!option) continue;
        const strip = typeof option.stripPrefixes === "function"
          ? option.stripPrefixes(selection) : (option.stripPrefixes || []);
        if (strip.length) flags = flags.filter((flag) => !strip.includes(flag.split(/[\s=]/)[0]));
        const extra = typeof option.flags === "function" ? option.flags(selection) : option.flags;
        flags.push(...(extra || []));
      }
      const families = flags.map((flag) => flag.split(/[\s=]/)[0]);
      const duplicate = families.find((family, index) => families.indexOf(family) !== index);
      if (duplicate) fail(where, `${label} emits duplicate flag family ${duplicate}`);
      return resolved;
    };

    validateResolved(selectionOf(), "commandBuilder default");
    const recipeIds = new Set();
    const recipeSignatures = new Set();
    for (const [index, recipe] of (builder.resource?.verifiedRecipes || []).entries()) {
      if (!recipe.id || recipeIds.has(recipe.id)) fail(where, `verifiedRecipes[${index}] has a duplicate/missing id`);
      recipeIds.add(recipe.id);
      const signature = [recipe.hw, recipe.nodes, recipe.gpus_per_node, recipe.placement,
        recipe.tp_size, recipe.ulysses_degree, recipe.ring_degree].join("|");
      if (recipeSignatures.has(signature)) fail(where, `verifiedRecipes[${index}] duplicates ${signature}`);
      recipeSignatures.add(signature);
      const selection = selectionOf({ ...recipe, topology_mode: "auto" });
      const topology = builder.resource.autoTopology(selection);
      const errors = builder.resource.validateTopology(selection, topology);
      if (!Array.isArray(errors) || errors.length) {
        fail(where, `verifiedRecipes[${index}] fails topology validation: ${(errors || []).join("; ")}`);
      }
      validateResolved(selection, `verifiedRecipes[${index}]`, true);
    }

    // H3's architectural contract is important enough to pin directly: exact
    // platform recipes, legal custom admission, and each invalidity family.
    if (config.modelName === "MiniMax-H3") {
      const checkH3 = (label, extra, expected, verified = true) => {
        const selection = selectionOf(extra);
        const resolved = validateResolved(selection, `H3 ${label}`, verified);
        if (!resolved) return;
        for (const [key, value] of Object.entries(expected)) {
          if (resolved.builder.topology?.[key] !== value) {
            fail(where, `H3 ${label} topology.${key}=${resolved.builder.topology?.[key]}, expected ${value}`);
          }
        }
      };
      checkH3("B200 1x8", { hw: "b200", nodes: 1, gpus_per_node: 8, placement: "resident" }, { tp_size: 1, ulysses_degree: 8, ring_degree: 1 });
      checkH3("H100 1x4", { hw: "h100", nodes: 1, gpus_per_node: 4, placement: "resident" }, { tp_size: 2, ulysses_degree: 2, ring_degree: 1 });
      checkH3("H200 2x8", { hw: "h200", nodes: 2, gpus_per_node: 8, placement: "resident" }, { tp_size: 1, ulysses_degree: 8, ring_degree: 2 });
      for (const hw of ["mi300x", "mi355x"]) {
        for (const count of [1, 2, 4, 8]) {
          checkH3(`${hw} 1x${count}`, { hw, nodes: 1, gpus_per_node: count, placement: "resident" }, { tp_size: 1, ulysses_degree: count, ring_degree: 1 });
        }
      }
      const custom = validateResolved(selectionOf({ hw: "b200", nodes: 1, gpus_per_node: 2, placement: "resident" }), "H3 legal custom");
      if (custom?.builder.errors?.length || custom?.builder.verification?.serve !== "unverified") {
        fail(where, "H3 legal custom topology must be copyable and Unverified");
      }
      for (const [label, extra] of [
        ["3 GPU", { hw: "h200", nodes: 1, gpus_per_node: 3, placement: "resident" }],
        ["TP3", { hw: "h100", nodes: 1, gpus_per_node: 4, placement: "resident", topology_mode: "manual", tp_size: 3, ulysses_degree: 1, ring_degree: 1 }],
        ["head divisibility", { hw: "h200", nodes: 2, gpus_per_node: 8, placement: "resident", topology_mode: "manual", tp_size: 2, ulysses_degree: 8, ring_degree: 1 }],
        ["sequence alignment", { hw: "mi300x", nodes: 2, gpus_per_node: 3, placement: "resident", topology_mode: "manual", tp_size: 1, ulysses_degree: 2, ring_degree: 3 }],
      ]) {
        const resolved = validateResolved(selectionOf(extra), `H3 invalid ${label}`);
        if (!resolved?.builder.errors?.length) fail(where, `H3 invalid ${label} was not rejected`);
      }
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
  // A cell may report its badge per selection (`verificationStatus` as a
  // function of sel), so it has to survive the same space the predicates do —
  // it renders on every pick, and an unrecognized return silently downgrades
  // the badge to "Not Verified" rather than erroring in the browser.
  const VERIFY_STATES = ["verified", "in-progress", "unverified"];
  for (const [i, cell] of (config.cells || []).entries()) {
    if (typeof cell.verificationStatus !== "function") continue;
    probe((sel) => {
      const out = cell.verificationStatus(sel);
      if (out !== undefined && !VERIFY_STATES.includes(out)) {
        throw new Error(`returned ${JSON.stringify(out)}, expected one of [${VERIFY_STATES}]`);
      }
    }, `cells[${i}].verificationStatus`);
  }
  for (const dim of [...(config.matchDims || []), ...(config.overlayDims || [])]) {
    if (typeof dim.showWhen === "function") probe(dim.showWhen, `${dim.id}.showWhen`);
    if (typeof dim.verifiedWhen === "function") probe(dim.verifiedWhen, `${dim.id}.verifiedWhen`);
    for (const opt of (dim.options || [])) {
      const tag = `${dim.id}.${opt.id}`;
      if (typeof opt.showWhen === "function") probe(opt.showWhen, `${tag}.showWhen`);
      if (typeof opt.disabled === "function") probe(opt.disabled, `${tag}.disabled`);
      if (typeof opt.soft === "function") probe(opt.soft, `${tag}.soft`);
      if (typeof opt.verifiedWhen === "function") probe(opt.verifiedWhen, `${tag}.verifiedWhen`);
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

  const pd = (config.playgroundFeatures || {}).pdDisagg;
  if (pd && typeof pd.showWhen === "function") {
    const incompatible = (pd.incompatibleSpeculativeAlgorithms || [])
      .map((name) => String(name).toUpperCase());
    for (const [i, cell] of (config.cells || []).entries()) {
      const algorithmFlag = (cell.flags || []).find((flag) =>
        flag.split(/[\s=]/)[0] === "--speculative-algorithm");
      const algorithm = algorithmFlag
        ? algorithmFlag.split(/[\s=]/).filter(Boolean)[1]?.toUpperCase()
        : null;
      if (!algorithm || !incompatible.includes(algorithm)) continue;
      const selection = { ...(cell.match || {}), specAlgorithm: algorithm };
      if (!pd.showWhen(selection)) {
        fail(where, `cells[${i}] hides PD Disagg for incompatible ${algorithm}; `
          + "the card must remain reachable so a PD role can disable speculation");
      }
    }
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
  const quickStart = /^## 1\. Quick start\s*$/m.exec(src);
  const legacyIntroduction = /^## 1\. Model Introduction\s*$/m.exec(src);
  const sectionOne = quickStart || legacyIntroduction;
  if (!sectionOne) {
    fail(where, "missing `## 1. Quick start` or legacy `## 1. Model Introduction` heading");
    continue;
  }

  const capabilityHeading = quickStart
    ? /^## 2\. Model capabilities\s*$/m.exec(src)
    : legacyIntroduction;
  if (!capabilityHeading) {
    fail(where, "Quick start pages need an exact `## 2. Model capabilities` heading");
    continue;
  }

  const importPattern = /import\s+\{\s*DiffusionModelTags\s*\}\s+from\s+['"]\/src\/snippets\/diffusion\/model-tags\.jsx['"]/;
  if (!importPattern.test(src.slice(0, sectionOne.index))) {
    fail(where, "missing the shared DiffusionModelTags import before section 1");
  }

  const tagMatch = /<DiffusionModelTags\s+tags=\{\[([^\]]+)]}\s*\/>/.exec(src.slice(0, sectionOne.index));
  if (!tagMatch) {
    fail(where, "missing `<DiffusionModelTags tags={[...]} />` before section 1");
  } else {
    const tags = [...tagMatch[1].matchAll(/["']([^"']+)["']/g)].map((m) => m[1].trim());
    if (tags.length < 4 || tags.length > 6) {
      fail(where, `tag widget has ${tags.length} tags; expected 4–6`);
    }
    if (tags.some((tag) => !tag)) fail(where, "tag widget contains an empty tag");
  }

  if (quickStart) {
    const quickStartBody = src.slice(quickStart.index + quickStart[0].length, capabilityHeading.index);
    if (!quickStartBody.includes("<Deployment config={config} />")) {
      fail(where, "Quick start must render the command builder before model capabilities");
    }
    if (!quickStartBody.includes('uv pip install "sglang[diffusion]"')) {
      fail(where, "Quick start must include the diffusion installation command");
    }
  }

  const introStart = capabilityHeading.index + capabilityHeading[0].length;
  const rest = src.slice(introStart);
  const boundaries = ["\n|", "\n<Warning", "\n<Note", quickStart ? "\n## 3" : "\n## 2"]
    .map((marker) => rest.indexOf(marker))
    .filter((i) => i >= 0);
  const lead = rest.slice(0, boundaries.length ? Math.min(...boundaries) : rest.length).trim();
  const paragraphs = lead.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  if (paragraphs.length < 2) {
    fail(where, "model capability introduction needs at least two lead paragraphs");
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
