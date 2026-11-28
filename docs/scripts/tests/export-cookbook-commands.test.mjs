import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

const config = "docs/src/snippets/configs/thinkingmachines/inkling-2-mini.jsx";
const page = "docs/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini.mdx";
const output = "python/sglang/cli/_commands/thinkingmachines--Inkling-2-Mini.json";
const exporter = "docs/scripts/export_commands.mjs";

function fixture(t, includeAutoregressivePage) {
  const root = mkdtempSync(join(tmpdir(), "sglang-cookbook-export-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const copy = (path) => {
    const source = new URL(`../../../${path}`, import.meta.url);
    const destination = join(root, path);
    mkdirSync(dirname(destination), { recursive: true });
    copyFileSync(source, destination);
  };
  copy(exporter);
  copy("docs/src/snippets/_deployment.jsx");
  copy(config);
  const excludedPage = join(root, "docs/cookbook/autoregressive-other/Excluded.mdx");
  mkdirSync(dirname(excludedPage), { recursive: true });
  writeFileSync(
    excludedPage,
    'import { config } from "/src/snippets/configs/excluded.jsx";\n<Deployment config={config} />\n',
  );
  if (includeAutoregressivePage) copy(page);
  return root;
}

function runExport(root) {
  return spawnSync(process.execPath, [join(root, exporter)], { encoding: "utf8" });
}

test("exports autoregressive Cookbook recipes with canonical URLs on the host platform", (t) => {
  const root = fixture(t, true);
  const first = runExport(root);
  assert.equal(first.status, 1, first.stderr);
  assert.match(first.stdout, /1 files, 1 changed/);

  const recipe = join(root, output);
  const generated = readFileSync(recipe, "utf8");
  const entries = JSON.parse(generated);
  assert.ok(entries.length > 0);
  for (const { url, command } of entries) {
    assert.match(url, /^https:\/\/docs\.sglang\.io\/cookbook\/autoregressive\/ThinkingMachines\/Inkling-2-Mini#/);
    assert.match(command, /--model-path thinkingmachines\/Inkling-2-Mini/);
  }

  const crlf = generated.replaceAll("\n", "\r\n");
  writeFileSync(recipe, crlf);
  const second = runExport(root);
  assert.equal(second.status, 0, second.stderr);
  assert.match(second.stdout, /1 files, 0 changed/);
  assert.equal(readFileSync(recipe, "utf8"), crlf);
});

test("an empty export leaves existing CLI recipes intact", (t) => {
  const root = fixture(t, false);
  const recipe = join(root, output);
  mkdirSync(dirname(recipe), { recursive: true });
  writeFileSync(recipe, "[{}]\n");

  const result = runExport(root);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /No Cookbook recipes were exported/);
  assert.equal(readFileSync(recipe, "utf8"), "[{}]\n");
});
