// Prints every recipe id a Cookbook Deployment can show a rating for, one per line, sorted.
// Use it to audit or clean up the rating service's data, e.g. to drop rows whose id no page
// can produce:
//
//   node docs/scripts/export_rating_ids.mjs > recipe-ids.txt
//
// A recipe id is "<page path>|<ratingKey(sel)>", the same string _deployment.jsx builds in the
// browser: ratingKey comes from the engine block, so the two cannot drift apart.
import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const docs = fileURLToPath(new URL("..", import.meta.url));
const configsDir = join(docs, "src/snippets/configs");

const engineSource = readFileSync(join(docs, "src/snippets/_deployment.jsx"), "utf8");
const start = engineSource.indexOf("// ==== COMMAND ENGINE ====");
const end = engineSource.indexOf("// ==== end COMMAND ENGINE ====");
if (start < 0 || end < start) throw new Error("_deployment.jsx is missing the COMMAND ENGINE markers");
const createEngine = new Function(
  "config",
  `${engineSource.slice(start, end)}
  return { findCell, validateSelection, ratingKey, overlayDimSpecs, rowVisible, optionVisible, optionDisabled };`,
);

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
  );
// The config file behind each `<Deployment config={X}>` on a page: X resolved through the page's
// imports, which may rename (`{ config as ascendConfig }`) and use either quote style.
const deploymentConfigs = (source) => {
  const imports = new Map();
  for (const m of source.matchAll(/import\s*{([^}]*)}\s*from\s*["']\/src\/snippets\/configs\/([^"']+)["']/g)) {
    for (const part of m[1].split(",")) {
      const [name, alias] = part.trim().split(/\s+as\s+/);
      if (name) imports.set(alias ?? name, { file: m[2], name });
    }
  }
  // A tag this cannot resolve throws rather than being skipped: a page missing from the output
  // would have its live votes treated as orphans.
  return [...source.matchAll(/<Deployment\b[^>]*>/g)].map(([tag]) => {
    const ref = imports.get(tag.match(/\bconfig=\{\s*([A-Za-z_$][\w$]*)\s*\}/)?.[1]);
    if (!ref) throw new Error(`cannot resolve the config of ${tag}`);
    return ref;
  });
};
const load = async (name) =>
  import("data:text/javascript," + encodeURIComponent(readFileSync(join(configsDir, name), "utf8")));

// Every combination of overlay options; visibility is checked per selection afterwards.
const overlayCombos = (specs) =>
  specs.reduce(
    (combos, spec) => combos.flatMap((c) => (spec.options || []).map((o) => ({ ...c, [spec.id]: o.id }))),
    [{}],
  );

const idsForPage = async (pagePath, { file, name }) => {
  const config = (await load(file))[name];
  // The component shows no rating on builder pages or pages that opt out.
  if (!config || config.commandBuilder || config.ratings === false) return [];
  const engine = createEngine(config);
  const combos = overlayCombos(engine.overlayDimSpecs);
  const ids = new Set();
  for (const base of config.cells) {
    for (const combo of combos) {
      const sel = engine.validateSelection(config.cells, { ...base.match, ...combo });
      const cell = engine.findCell(config.cells, sel);
      if (!cell || cell.redirect) continue;
      // Keep only selections a reader can reach: each showing overlay row on an option it offers,
      // unless the row offers none (every option disabled here). The page then keeps whatever
      // pick the row held, and ratingKey leaves that row out, so any pick gives the same id.
      const reachable = engine.overlayDimSpecs
        .filter((spec) => engine.rowVisible(spec, sel))
        .every((spec) => {
          const usable = (spec.options || []).filter(
            (o) => engine.optionVisible(o, sel) && !engine.optionDisabled(o, sel),
          );
          return !usable.length || usable.some((o) => o.id === sel[spec.id]);
        });
      if (reachable) ids.add(`${pagePath}|${engine.ratingKey(sel)}`);
    }
  }
  return [...ids];
};

const all = new Set();
for (const path of walk(join(docs, "cookbook")).filter((p) => p.endsWith(".mdx")).sort()) {
  const source = readFileSync(path, "utf8");
  // The browser's location.pathname, lower-cased as the component does.
  const pagePath = `/${relative(docs, path).replace(/\.mdx$/, "")}`.toLowerCase();
  let refs;
  try {
    refs = deploymentConfigs(source);
  } catch (e) {
    throw new Error(`${relative(docs, path)}: ${e.message}`);
  }
  for (const ref of refs) {
    for (const id of await idsForPage(pagePath, ref)) all.add(id);
  }
}

process.stdout.write(`${[...all].sort().join("\n")}\n`);
console.error(`${all.size} recipe ids`);
