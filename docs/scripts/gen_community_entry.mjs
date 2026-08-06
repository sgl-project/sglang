#!/usr/bin/env node
// Generate a Community Configs entry from a cookbook contribution PR.
//
//   node docs/scripts/gen_community_entry.mjs --pr 31006
//   node docs/scripts/gen_community_entry.mjs --pr 31006 --repo sgl-project/sglang
//
// Why this exists: a community entry's config and benchmarks must be BYTE-FAITHFUL to
// what the contributor actually ran. Hand-transcribing a diff into the data file works
// right up until it doesn't, and a silently mistyped flag or throughput number is
// exactly the failure this section is supposed to prevent. So everything mechanical is
// extracted here. The ONLY field a human writes is `source.org` — GitHub does not know
// a contributor's affiliation.
//
// How it extracts (structural, not textual): the cookbook config files are plain
// `export const config = {...}` data modules, so both the PR-head and base versions can
// be imported and compared as OBJECTS. Cells present at head but absent at base are the
// contribution. This is immune to diff-hunk formatting, reordering, and context lines —
// a regex over `gh pr diff` is not.
//
// What each generated field is derived from:
//   flags, env          the added cell, verbatim
//   modelName           head config `modelNames[hw|variant|quant]` → `[variant|quant]`
//   dockerImage         head config `dockerImages[hw|quant|strategy]` → `[hw|quant]` → `[hw]`
//   hardware            the hw entry's label + vram, from the PR's `config.hardware`
//                       addition, else the shared HARDWARE_CATALOG in _deployment.jsx.
//                       GPU MODEL ONLY — the card derives the count from --tp × --nnodes
//   sglangVersion       the added benchmarks entry's `sglang_version`
//   (benchmark numbers are NOT extracted — they stay in the PR; the section shows the
//    config and links there)
//   source              PR number/url + author login; `org` is emitted as "TODO"
//   reportedAt          the PR's last-updated date
//   title               "<hw> · <QUANT> · <strategy>", the chip label when a PR
//                       contributes several configs
//
// Requires `gh` authenticated. Read-only: prints to stdout, writes nothing.

import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SNIPPETS = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "snippets");

const args = process.argv.slice(2);
const argOf = (name, dflt) => {
  const i = args.indexOf(name);
  return i === -1 ? dflt : args[i + 1];
};
const PR = argOf("--pr");
const REPO = argOf("--repo", "sgl-project/sglang");
if (!PR) {
  console.error("usage: gen_community_entry.mjs --pr <number> [--repo owner/name]");
  process.exit(2);
}

const gh = (...a) => execFileSync("gh", a, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
const ghJson = (...a) => JSON.parse(gh(...a));

// ------------------------------------------------------------------ PR metadata
const pr = ghJson("api", `repos/${REPO}/pulls/${PR}`);
const headRepo = pr.head.repo.full_name;
const headSha = pr.head.sha;
const baseSha = pr.base.sha;
const files = ghJson("api", `repos/${REPO}/pulls/${PR}/files`, "--paginate")
  .map((f) => f.filename);

// The cookbook config path moved docs_new/ → docs/ mid-2026, so a PR opened before the
// move names the old prefix. Match on the tail rather than the full path.
const configPath = files.find((f) =>
  /src\/snippets\/configs\/.+\.jsx$/.test(f)
  && !f.includes("benchmark") && !f.endsWith("-community.jsx"));
const benchPath = files.find((f) => /src\/snippets\/configs\/.+benchmarks\.jsx$/.test(f));
if (!configPath) {
  console.error(`PR #${PR} touches no cookbook config file; nothing to extract.\n`
    + `changed files:\n  ${files.join("\n  ")}`);
  process.exit(1);
}

// ------------------------------------------------------- load config objects at 2 refs
// `?ref=` must be quoted when typed in a shell; execFileSync passes it literally.
const fetchFile = (repo, path, ref) => {
  const res = ghJson("api", `repos/${repo}/contents/${path}?ref=${ref}`);
  return Buffer.from(res.content, "base64").toString("utf8");
};
const loadExport = async (source, name) => {
  const mod = await import("data:text/javascript," + encodeURIComponent(source));
  return mod[name];
};
// A file added by the PR has no base version; treat that as "everything is new".
const tryFetch = (repo, path, ref) => {
  try { return fetchFile(repo, path, ref); } catch { return null; }
};

const headConfigSrc = fetchFile(headRepo, configPath, headSha);
const baseConfigSrc = tryFetch(REPO, configPath, baseSha);
const headConfig = await loadExport(headConfigSrc, "config");
const baseConfig = baseConfigSrc ? await loadExport(baseConfigSrc, "config") : { cells: [] };

let headBench = [];
let baseBench = [];
if (benchPath) {
  headBench = (await loadExport(fetchFile(headRepo, benchPath, headSha), "benchmarks")) || [];
  const bs = tryFetch(REPO, benchPath, baseSha);
  baseBench = bs ? ((await loadExport(bs, "benchmarks")) || []) : [];
}

// ------------------------------------------------------------------- structural diff
const matchKey = (m) => JSON.stringify(Object.entries(m || {}).sort());
const baseCellKeys = new Set((baseConfig.cells || []).map((c) => matchKey(c.match)));
const addedCells = (headConfig.cells || []).filter((c) => !baseCellKeys.has(matchKey(c.match)));
if (addedCells.length === 0) {
  console.error(`PR #${PR} adds no new cells (it may only edit existing ones).`);
  process.exit(1);
}
const benchFor = (match) => {
  const key = matchKey(match);
  const baseKeys = new Set(baseBench.map((b) => matchKey(b.match)));
  const hit = headBench.find((b) => matchKey(b.match) === key);
  // A benchmarks entry that already existed at base was not measured by this
  // contributor; do not attribute it to them.
  return hit && !baseKeys.has(key) ? hit : null;
};

// ------------------------------------------------------------------ hardware label
// Prefer the PR's own `config.hardware` addition; fall back to the shared catalog in
// _deployment.jsx, sliced out of the source and evaluated (a plain object literal).
const sharedCatalog = (() => {
  const src = readFileSync(join(SNIPPETS, "_deployment.jsx"), "utf8");
  const start = src.indexOf("const HARDWARE_CATALOG = {");
  if (start === -1) return {};
  const open = src.indexOf("{", start);
  let depth = 0, end = open;
  for (let i = open; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") { depth--; if (depth === 0) { end = i; break; } }
  }
  try {
    return new Function("return " + src.slice(open, end + 1))();
  } catch { return {}; }
})();
const VENDOR_BRAND = { nvidia: "NVIDIA", amd: "AMD", blackwell: "NVIDIA", hopper: "NVIDIA" };
const hardwareOf = (hwId) => {
  const own = (headConfig.hardware || []).find((h) => h.id === hwId);
  const found = own || Object.entries(sharedCatalog).flatMap(([group, list]) =>
    list.filter((h) => h.id === hwId).map((h) => ({ ...h, group }))).at(0);
  if (!found) return { text: null, why: `hw id "${hwId}" is in neither config.hardware nor HARDWARE_CATALOG` };
  const brand = VENDOR_BRAND[found.vendor || found.group] || "";
  return {
    text: [brand, found.label, found.vram ? `(${found.vram})` : ""].filter(Boolean).join(" "),
    why: own ? "config.hardware (added by this PR)" : "shared HARDWARE_CATALOG",
  };
};

const modelNameOf = (m) => {
  const mn = headConfig.modelNames || {};
  return mn[`${m.hw}|${m.variant}|${m.quant}`] ?? mn[`${m.variant}|${m.quant}`] ?? null;
};
const dockerImageOf = (m) => {
  const di = headConfig.dockerImages || {};
  return di[`${m.hw}|${m.quant}|${m.strategy}`] ?? di[`${m.hw}|${m.quant}`] ?? di[m.hw] ?? null;
};

// ------------------------------------------------------------------------- emit
const q = (s) => JSON.stringify(s);
const slug = (m) => [m.hw, m.quant, m.strategy].filter(Boolean).join("-");
const lines = [];
const warnings = [];

// One CONTRIBUTION per PR, holding every config that PR added. Grouping by PR is the
// shape the section renders: a reader browsing community work wants a PR's configs
// together under the credit and link they came from.
lines.push(`  // GENERATED from PR #${PR} — ${headRepo}@${headSha.slice(0, 7)}`);
lines.push(`  {`);
lines.push(`    source: {`);
lines.push(`      label: ${q("PR #" + PR)},`);
lines.push(`      url: ${q(pr.html_url)},`);
lines.push(`      author: ${q(pr.user.login)},`);
lines.push(`      org: "TODO",`);
lines.push(`      authorUrl: ${q(pr.user.html_url)},`);
lines.push(`    },`);
lines.push(`    reportedAt: ${q(pr.updated_at.slice(0, 10))},`);
lines.push(`    configs: [`);

for (const cell of addedCells) {
  const m = cell.match || {};
  const bench = benchFor(m);
  const hw = hardwareOf(m.hw);
  const modelName = modelNameOf(m);
  const image = dockerImageOf(m);
  if (!hw.text) warnings.push(hw.why);
  if (!modelName) warnings.push(`no modelNames entry for ${m.hw}|${m.variant}|${m.quant}`);
  if (!bench) warnings.push(`config ${slug(m)} has NO new benchmarks entry — it will render config-only`);
  if (bench && !bench.sglang_version) warnings.push(`benchmarks entry for ${slug(m)} has no sglang_version`);
  if (cell.verified) warnings.push(`cell ${slug(m)} carries verified:true — dropped, community configs are not team-verified`);

  // The chip label when a PR contributes several configs, so it must be SHORT: the
  // hardware and checkpoint already render on the identity line beside it.
  const title = [
    hw.text ? hw.text.replace(/^(NVIDIA|AMD) /, "").replace(/ \(.*\)$/, "") : m.hw,
    m.quant ? String(m.quant).toUpperCase() : null,
    m.strategy || null,
  ].filter(Boolean).join(" · ");

  lines.push(`      {`);
  lines.push(`        id: ${q(slug(m))},`);
  lines.push(`        title: ${q(title)},`);
  lines.push(`        // GPU model only — the card derives the count from --tp × --nnodes.`);
  lines.push(`        hardware: ${q(hw.text || "TODO")},   // from ${hw.why}`);
  lines.push(`        modelName: ${q(modelName || "TODO")},`);
  if (bench && bench.sglang_version) lines.push(`        sglangVersion: ${q(bench.sglang_version)},`);
  if (image) lines.push(`        dockerImage: ${q(image)},`);
  lines.push(`        env: ${JSON.stringify(cell.env || [])},`);
  lines.push(`        flags: [`);
  for (const f of cell.flags || []) lines.push(`          ${q(f)},`);
  lines.push(`        ],`);

  lines.push(`      },`);
}
lines.push(`    ],`);
lines.push(`  },`);

console.log(lines.join("\n"));
console.error(`\nPaste the block above into the model's -community.jsx \`community\` array`);
console.error(`(one such block per contributing PR; several blocks stack as several groups).`);
if (warnings.length) {
  console.error(`\n${warnings.length} thing(s) needing a human:`);
  for (const w of [...new Set(warnings)]) console.error("  - " + w);
}
