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
const imported = (source, binding) =>
  source.match(new RegExp(`import\\s*{\\s*${binding}\\s*}\\s*from\\s*"/src/snippets/configs/([^"]+)"`))?.[1];
const load = async (name) =>
  import("data:text/javascript," + encodeURIComponent(readFileSync(join(configsDir, name), "utf8")));

// Every combination of overlay options; visibility is checked per selection afterwards.
const overlayCombos = (specs) =>
  specs.reduce(
    (combos, spec) => combos.flatMap((c) => (spec.options || []).map((o) => ({ ...c, [spec.id]: o.id }))),
    [{}],
  );

const idsForPage = async (pagePath, configName) => {
  const { config } = await load(configName);
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
      // Keep only selections a reader can reach: each showing overlay row on an option it offers.
      const reachable = engine.overlayDimSpecs
        .filter((spec) => engine.rowVisible(spec, sel))
        .every((spec) => {
          const opt = (spec.options || []).find((o) => o.id === sel[spec.id]);
          return opt && engine.optionVisible(opt, sel) && !engine.optionDisabled(opt, sel);
        });
      if (reachable) ids.add(`${pagePath}|${engine.ratingKey(sel)}`);
    }
  }
  return [...ids];
};

const all = new Set();
for (const path of walk(join(docs, "cookbook")).filter((p) => p.endsWith(".mdx")).sort()) {
  const source = readFileSync(path, "utf8");
  const name = imported(source, "config");
  if (!name || !source.includes("<Deployment")) continue;
  // The browser's location.pathname, lower-cased as the component does.
  const pagePath = `/${relative(docs, path).replace(/\.mdx$/, "")}`.toLowerCase();
  for (const id of await idsForPage(pagePath, name)) all.add(id);
}

process.stdout.write(`${[...all].sort().join("\n")}\n`);
console.error(`${all.size} recipe ids`);
