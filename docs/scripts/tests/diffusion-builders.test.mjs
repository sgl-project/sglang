import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import test from "node:test";

const docs = fileURLToPath(new URL("../..", import.meta.url));
const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
  );
const pages = walk(join(docs, "cookbook/diffusion")).filter((path) => path.endsWith(".mdx"));
const configs = new Map();
for (const page of pages) {
  const content = readFileSync(page, "utf8");
  const path = content.match(/import\s+\{\s*config\s*\}\s+from\s+["']([^"']+)["']/)?.[1];
  if (!path) continue;
  const source = readFileSync(join(docs, path), "utf8");
  configs.set(page, (await import("data:text/javascript," + encodeURIComponent(source))).config);
}
const get = (suffix) => [...configs].find(([path]) => path.endsWith(join(...suffix.split("/"))))[1];
const selection = (config, overrides = {}) => ({
  ...config.commandBuilder.defaultSelection,
  ...Object.fromEntries(config.overlayDims.map((dim) => [dim.id, dim.default])),
  ...overrides,
});
const resolve = (config, overrides = {}) => config.commandBuilder.resolveDeployment(selection(config, overrides));
const curl = (config, overrides = {}) =>
  typeof config.curl === "function" ? config.curl(selection(config, overrides)) : config.curl;
const body = (config, overrides = {}) => JSON.parse(curl(config, overrides).match(/-d '([\s\S]+)'$/)[1]);
const migrated = [...configs.values()].filter((config) =>
  config.commandBuilder.resource.verifiedRecipes.some((recipe) => recipe.id.endsWith("-documented")),
);

test("every diffusion model uses one shared builder before model capabilities", () => {
  const models = pages.filter((path) => readFileSync(path, "utf8").includes("<DiffusionModelTags"));
  assert.ok(models.length >= 29);
  for (const page of models) {
    const content = readFileSync(page, "utf8");
    const quick = content.indexOf("## 1. Quick start");
    const builder = content.indexOf("<Deployment config={config} />");
    const capabilities = content.indexOf("## 2. Model capabilities");
    assert.ok(quick >= 0 && quick < builder && builder < capabilities, page);
    assert.equal(content.match(/<Deployment\b/g)?.length, 1, page);
    assert.ok(configs.get(page)?.commandBuilder, page);
    assert.doesNotMatch(content, /snippets\/diffusion\/[^'"\n]*-deployment\.jsx|Interactive Command Generator/, page);
  }
  assert.ok(!walk(join(docs, "src/snippets/diffusion")).some((path) => path.endsWith("-deployment.jsx")));
});

test("server builder has a footer link to the CLI reference", () => {
  const shared = readFileSync(join(docs, "src/snippets/_deployment.jsx"), "utf8");
  assert.match(shared, /builderScope === "serve" && \(\s*<p className="sgd-builder-docs-tip">/);
  const tip = shared.indexOf('className="sgd-builder-docs-tip"');
  assert.ok(tip > shared.indexOf('className="sgd-builder-output-rail"'));
  const href = shared.slice(tip).match(/href="([^"]+)"/)[1];
  assert.equal(href, "/docs/sglang-diffusion/api/cli");
  assert.match(readFileSync(join(docs, `${href}.mdx`), "utf8"), /title: CLI reference/);
});

test("migrated recipes retain unverified status, valid scopes and editable ports", () => {
  assert.equal(migrated.length, 22);
  const shared = readFileSync(join(docs, "src/snippets/_deployment.jsx"), "utf8");
  const builtInIds = new Set([...shared.matchAll(/\bid: "([^"]+)"/g)].map((match) => match[1]));
  for (const config of migrated) {
    for (const hw of config.supportedHardware) {
      assert.ok(builtInIds.has(hw) || config.hardware.some((entry) => entry.id === hw), `${config.modelName}: hidden ${hw}`);
    }
    for (const dim of config.overlayDims) assert.ok(["base", "serve", "request"].includes(dim.scope));
    for (const recipe of config.commandBuilder.resource.verifiedRecipes) {
      const result = resolve(config, { ...recipe, topology_mode: "manual" });
      assert.deepEqual(result.builder.errors, [], config.modelName + " " + recipe.hw);
      assert.equal(recipe.unverified, true);
      assert.equal(result.verified, false);
      assert.deepEqual(result.builder.verification, { serve: "unverified", request: "unverified" });
      assert.equal(result.flags.filter((flag) => flag.startsWith("--port ")).length, 1);
      assert.ok(result.flags.includes("--port {{PORT}}"));
      assert.equal(config.placeholders.PORT.default, config.placeholders.CURL_PORT.default);
    }
  }
});

test("migrated builders reject invalid resources and undocumented topology", () => {
  for (const config of migrated) {
    for (const overrides of [
      { hw: "unknown" },
      { nodes: 2 },
      { gpus_per_node: 0 },
      { gpus_per_node: 1.5 },
      { gpus_per_node: config.commandBuilder.resource.limits.gpus_per_node.max + 1 },
      { topology_mode: "manual", tp_size: 16, ulysses_degree: 16, ring_degree: 8 },
    ]) {
      const result = resolve(config, overrides);
      assert.ok(result.builder.errors.length, config.modelName + JSON.stringify(overrides));
      assert.deepEqual(result.builder.verification, { serve: "error", request: "error" });
      assert.equal(result.verified, false);
    }
  }
});

test("all reachable migrated options produce valid shell requests and no unresolved model", () => {
  let checked = 0;
  for (const config of migrated) {
    let states = config.supportedHardware.map((hw) => selection(config, { hw }));
    for (const dim of config.overlayDims) {
      states = states.flatMap((s) =>
        dim.options
          .filter((option) => !option.showWhen || option.showWhen(s))
          .map((option) => ({ ...s, [dim.id]: option.id })),
      );
    }
    for (const s of states) {
      const result = config.commandBuilder.resolveDeployment(s);
      assert.doesNotMatch(result.flags.join(" "), /undefined|NaN/);
      const command = typeof config.curl === "function" ? config.curl(s) : config.curl;
      assert.doesNotMatch(command, /undefined|\\n/);
      const shell = spawnSync("bash", ["-n"], { input: command, encoding: "utf8" });
      assert.equal(shell.status, 0, config.modelName + shell.stderr);
      const json = command.match(/-d '([\s\S]+)'$/)?.[1];
      if (json) assert.doesNotThrow(() => JSON.parse(json));
      checked++;
    }
  }
  assert.ok(checked > 300);
});

test("legacy image hardware, precision and checkpoint choices remain available", () => {
  const flux = get("/FLUX/FLUX.mdx");
  const qwen = get("/Qwen-Image/Qwen-Image.mdx");
  const edit = get("/Qwen-Image/Qwen-Image-Edit.mdx");
  const zimage = get("/Z-Image/Z-Image-Turbo.mdx");
  assert.equal(flux.supportedHardware.length, 11);
  assert.equal(qwen.supportedHardware.length, 10);
  assert.equal(edit.supportedHardware.length, 8);
  assert.equal(zimage.supportedHardware.length, 10);
  assert.ok(resolve(flux, { hw: "arc_b", version: "flux2-dev", gpus_per_node: 4 }).flags.includes("--tp-size 4"));
  assert.ok(resolve(flux, { hw: "a2", version: "flux2-dev", gpus_per_node: 2 }).flags.includes("--tp-size 2"));
  assert.ok(
    resolve(qwen, { hw: "b200", precision: "nvfp4" }).flags.includes(
      "--model-path lmsys/qwen-image-2512-modelopt-nvfp4-sglang",
    ),
  );
  assert.equal(qwen.overlayDims[0].options[1].showWhen({ hw: "h100" }), false);
  assert.ok(resolve(zimage, { hw: "a3", gpus_per_node: 2 }).flags.includes("--tp-size 2"));
  assert.match(curl(edit), /\/v1\/images\/edits/);
  assert.match(curl(edit), /image=@\{\{INPUT_IMAGE\}\}/);
});

test("H3-Image recommends the measured topology within the selected capacity", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const resource = config.commandBuilder.resource;
  for (const [gpus, ulysses] of [[1, 1], [2, 2]]) {
    const s = selection(config, { hw: "h200", gpus_per_node: gpus, size: "4736x3552" });
    const recipe = resource.recommendedRecipe(s);
    assert.equal(recipe.gpus_per_node, gpus);
    assert.equal(recipe.ulysses_degree, ulysses);
    assert.equal(recipe.verifiedWhen(s), true);
    assert.equal(recipe.verifiedWhen({ ...s, size: "5472x3072" }), false);
    assert.equal(recipe.verifiedWhen({ ...s, mode: "edit" }), false);
  }
  const four = resource.recommendedRecipe(selection(config, { hw: "h200", gpus_per_node: 4 }));
  assert.equal(four.gpus_per_node, 4);
  assert.equal(four.ulysses_degree, 4);
  assert.equal(four.unverified, true);
  assert.equal(four.verifiedWhen(selection(config, { hw: "h200", gpus_per_node: 4 })), false);
});

test("H3-Image derives capacity-aware commands without borrowing video-only features", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const resource = config.commandBuilder.resource;
  for (const [hw, gpus, tp, ulysses, placement] of [
    ["b200", 1, 1, 1, "auto"], ["gb300", 4, 1, 4, "auto"],
    ["h100", 1, 1, 1, "offload"], ["h100", 2, 1, 2, "resident"],
    ["h800", 2, 1, 2, "resident"],
    ["rtx3060", 1, 1, 1, "layerwise"], ["rtx4090", 1, 1, 1, "layerwise"],
    ["rtx5090", 1, 1, 1, "layerwise"], ["rtx6000ada", 1, 1, 1, "layerwise"],
    ["rtx3090", 2, 1, 2, "layerwise"],
  ]) {
    const selected = selection(config, { hw, gpus_per_node: gpus });
    const recipe = resource.recommendedRecipe(selected);
    assert.ok(recipe, hw);
    assert.equal(recipe.gpus_per_node, gpus);
    assert.equal(recipe.tp_size, tp);
    assert.equal(recipe.ulysses_degree, ulysses);
    assert.equal(recipe.placement, placement);
    assert.equal(recipe.encoder, gpus > 1 ? "fold" : "auto");
    assert.equal(recipe.adaln, "online");
    assert.equal(recipe.unverified, true);
    const result = resolve(config, { ...recipe, topology_mode: "manual" });
    assert.deepEqual(result.builder.errors, []);
    assert.deepEqual(result.builder.verification, { serve: "unverified", request: "unverified" });
    assert.equal(recipe.verifiedWhen(selected), false);
    assert.match(result.builder.warnings.join(" "), /[Dd]erived/);
  }
  for (const hw of config.supportedHardware) {
    const recipes = resource.verifiedRecipes.filter((recipe) => recipe.hw === hw);
    assert.ok(recipes.length, hw);
    const starting = recipes.find((recipe) => recipe.default) || recipes[0];
    assert.deepEqual(resolve(config, { ...starting, topology_mode: "manual" }).builder.errors, []);
  }
  const small = resolve(config, { hw: "rtx5090", gpus_per_node: 1 });
  assert.match(small.builder.errors.join(" "), /Layerwise offload/);
  assert.equal(small.verified, false);
  assert.equal(small.builder.verification.serve, "error");
  const flags = config.overlayDims.filter((dim) => dim.scope === "serve")
    .flatMap((dim) => dim.options.flatMap((option) => option.flags || []));
  assert.match(flags.join(" "), /all=layerwise-offload/);
  assert.match(flags.join(" "), /--layerwise-prefetch-size transformer=1,text_encoder=1,video_vae=1/);
});

test("H3-Image omits multi-GPU recipes without NVLink and streams small GPUs", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const resource = config.commandBuilder.resource;
  for (const hw of config.supportedHardware.filter((hw) => hw.startsWith("rtx") && hw !== "rtx3090")) {
    assert.equal(resource.limits.gpus_per_node.disabledWhen({ hw }), true);
    assert.ok(resource.verifiedRecipes.filter((recipe) => recipe.hw === hw).every((recipe) => recipe.gpus_per_node === 1));
    assert.match(resolve(config, { hw, gpus_per_node: 2 }).builder.errors.join(" "), /require NVLink/);
  }
  for (const gpus_per_node of [1, 2]) {
    const recipe = resource.recommendedRecipe(selection(config, { hw: "rtx3090", gpus_per_node }));
    assert.equal(recipe.placement, "layerwise");
    assert.deepEqual(resolve(config, recipe).builder.errors, []);
  }
  assert.match(resolve(config, { hw: "rtx3090", gpus_per_node: 4 }).builder.errors.join(" "), /two-card NVLink pair/);
});

test("H3-Image RTX PRO single-card coverage requires AdaLN precomputation", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const s = selection(config, { hw: "rtxpro6000" });
  const recipe = config.commandBuilder.resource.recommendedRecipe(s);
  assert.equal(recipe.placement, "resident");
  assert.equal(recipe.adaln, "auto");
  assert.equal(recipe.unverified, undefined);
  for (const placement of ["resident", "offload"]) {
    const selected = { ...s, placement, size: "4736x3552" };
    assert.equal(resolve(config, selected).verified, true);
    assert.equal(resolve(config, { ...selected, adaln: "resident" }).verified, false);
    assert.equal(resolve(config, { ...selected, gpus_per_node: 2, encoder: "fold" }).verified, false);
    assert.equal(resolve(config, { ...selected, mode: "edit", references: "two" }).verified, false);
    assert.equal(resolve(config, { ...selected, mode: "edit", size: "2048x2048", references: "two", quality: "lossless" }).verified, true);
    assert.deepEqual(resolve(config, { ...selected, mode: "edit", size: "2048x2048", references: "two", quality: "exact" }).builder.verification,
      { serve: "verified", request: "unverified" });
  }
});

test("H3-Image 4K verification stays within measured tasks and server settings", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  for (const topology of [
    { gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto" },
    { gpus_per_node: 2, tp_size: 2, ulysses_degree: 1, ring_degree: 1, encoder: "fold" },
    { gpus_per_node: 2, tp_size: 1, ulysses_degree: 2, ring_degree: 1, encoder: "fold" },
    { gpus_per_node: 2, tp_size: 1, ulysses_degree: 1, ring_degree: 2, encoder: "fold" },
  ]) {
    const base = { hw: "h200", topology_mode: "manual", ...topology };
    for (const quality of ["exact", "lossless"]) {
      for (const task of [
        { mode: "t2i", size: "4736x3552" },
        { mode: "t2i", size: "6272x2688" },
        { mode: "edit", size: "4096x4096" },
      ]) {
        assert.equal(resolve(config, { ...base, ...task, quality }).verified, true);
        for (const override of [{ placement: "resident" }, { placement: "offload" }, { adaln: "resident" }, { warmup: "off" }]) {
          const result = resolve(config, { ...base, ...task, quality, ...override });
          assert.equal(result.verified, false, JSON.stringify({ ...base, ...task, ...override }));
          assert.deepEqual(result.builder.errors, []);
        }
        assert.equal(resolve(config, { ...base, ...task, quality, hw: "b300", placement: "resident" }).verified, true);
      }
    }
    assert.equal(resolve(config, { ...base, mode: "edit", size: "4096x4096", references: "two" }).verified, false);
  }
  const invalid = resolve(config, { hw: "h200", gpus_per_node: 3 });
  assert.ok(invalid.builder.errors.length);
  assert.equal(invalid.verified, false);
  assert.equal(invalid.builder.verification.request, "error");
});

test("H3-Image request controls preserve reference order, seed and native canvas", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const command = curl(config, { mode: "edit", references: "two", size: "4096x4096", seed: 1234 });
  assert.equal(command.match(/-F 'image=@/g).length, 2);
  assert.ok(command.indexOf("{{IMAGE_PATH}}") < command.indexOf("{{IMAGE_PATH_2}}"));
  assert.match(command, /seed=1234/);
  assert.match(command, /<Picture 1>.*<Picture 2>/);
  assert.equal(spawnSync("bash", ["-n"], { input: command }).status, 0);
  const json = JSON.parse(curl(config, { seed: 1234, size: "4736x3552" }).match(/-d '([\s\S]+)' \|/)[1]);
  assert.equal(json.seed, 1234);
  assert.equal(json.size, "4736x3552");
  assert.equal(json.generator_device, "cpu");
  assert.equal(json.min_generation_area, undefined);
  assert.deepEqual(resolve(config, { seed: 1234, size: "4736x3552" }).flags, resolve(config).flags);
});

test("H3-Image graph warmup matches the generated request and server references", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  for (const mode of ["t2i", "edit"]) {
    const s = { mode, references: "two", size: "4096x4096", execution: "bcg", placement: "resident", adaln: "online" };
    const result = resolve(config, s);
    const flag = result.flags.find((value) => value.startsWith("--warmup-sampling-params "));
    const warmup = JSON.parse(flag.match(/'(.+)'$/)[1]);
    assert.equal(warmup.width, 4096);
    assert.equal(warmup.height, 4096);
    assert.equal(warmup.quality, "lossless");
    if (mode === "edit") {
      assert.deepEqual(warmup.image_path, ["{{WARMUP_IMAGE_PATH}}", "{{WARMUP_IMAGE_PATH_2}}"]);
      assert.ok(curl(config, s).includes(warmup.prompt));
    } else {
      assert.equal(warmup.image_path, undefined);
      assert.equal(warmup.prompt, JSON.parse(curl(config, s).match(/-d '([\s\S]+)' \|/)[1]).prompt);
    }
    assert.equal(spawnSync("bash", ["-n"], { input: "sglang serve " + result.flags.join(" ") }).status, 0);
  }
  assert.match(resolve(config, { execution: "bcg", warmup: "off" }).builder.errors.join(" "), /capture CUDA graphs/);
});

test("H3-Image quantization remains opt-in with separate hardware and placement coverage", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const dim = config.overlayDims.find((item) => item.id === "quantization");
  assert.equal(dim.default, "native");
  assert.ok(dim.options.find((option) => option.id === "mxfp8-all").flags.includes("--component-quantizations.text_encoder mxfp8"));
  assert.equal(resolve(config, { quantization: "fp8-all", placement: "layerwise", adaln: "online" }).builder.verification.serve, "verified");
  assert.equal(resolve(config, { quantization: "fp8-dit", placement: "layerwise", adaln: "online" }).builder.verification.serve, "unverified");
  assert.match(resolve(config, { hw: "rtx4090", quantization: "mxfp8-all", placement: "layerwise", adaln: "online" }).builder.errors.join(" "), /Hopper or newer/);
  assert.equal(resolve(config, { hw: "rtxpro6000", quantization: "mxfp8-all", placement: "resident" }).builder.verification.serve, "verified");
  assert.equal(resolve(config, { hw: "rtxpro6000", quantization: "mxfp8-dit", placement: "resident" }).builder.verification.serve, "unverified");
  assert.deepEqual(resolve(config, { quantization: "fp8-all", placement: "resident", adaln: "resident" }).builder.errors, []);
  assert.equal(resolve(config, { execution: "compile", placement: "resident" }).builder.verification.serve, "verified");
  assert.equal(resolve(config, { execution: "compile", placement: "layerwise", adaln: "online" }).builder.verification.serve, "unverified");
  assert.doesNotMatch(curl(config, { quantization: "fp8-all" }), /--quantization|component-quantizations/);
});

test("H3-Image file URLs couple persistent output with a separate image download", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const base = { hw: "b300", gpus_per_node: 2, encoder: "fold", size: "4736x3552" };
  const temporary = resolve(config, base);
  const persistent = resolve(config, { ...base, delivery: "url" });
  assert.ok(!temporary.flags.some((flag) => flag.startsWith("--output-path")));
  assert.ok(persistent.flags.includes("--output-path {{OUTPUT_DIR}}"));
  assert.deepEqual(persistent.flags.filter((flag) => !flag.startsWith("--output-path")), temporary.flags);
  assert.equal(persistent.verified, true);
  assert.equal(resolve(config, { ...base, delivery: "url", png: "none" }).verified, true);
  for (const override of [
    { hw: "h200" }, { gpus_per_node: 1, encoder: "auto" }, { quality: "exact" },
    { size: "2048x2048" }, { topology_mode: "manual", tp_size: 2, ulysses_degree: 1 },
    { mode: "edit", size: "4096x4096", references: "two" },
  ]) assert.equal(resolve(config, { ...base, delivery: "url", ...override }).verified, false);
  assert.match(config.overlayDims.find((dim) => dim.id === "delivery").description, /restart.*server process/);
  for (const mode of ["t2i", "edit"]) {
    const command = curl(config, { ...base, mode, delivery: "url" });
    assert.match(command, /urllib\.request\.urlretrieve/);
    assert.match(command, /urllib\.parse\.urljoin/);
    assert.doesNotMatch(command, /base64\.b64decode/);
    assert.equal(spawnSync("bash", ["-n"], { input: command }).status, 0);
    if (mode === "edit") assert.match(command, /response_format=url/);
    else assert.equal(JSON.parse(command.match(/-d '([\s\S]+)' \|/)[1]).response_format, "url");
  }
  assert.equal(JSON.parse(curl(config).match(/-d '([\s\S]+)' \|/)[1]).response_format, "b64_json");
});

test("H3-Image request coverage does not change measured server coverage", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const server = { hw: "h200", gpus_per_node: 2, encoder: "fold" };
  const baseline = resolve(config, server);
  assert.equal(baseline.verified, true);
  for (const request of [
    { size: "1024x1024" },
    { size: "5472x3072" },
    { quality: "exact" },
    { mode: "edit", size: "4096x4096", references: "two" },
  ]) {
    const result = resolve(config, { ...server, ...request });
    assert.deepEqual(result.flags, baseline.flags);
    assert.deepEqual(result.builder.verification, { serve: "verified", request: "unverified" });
  }
  assert.equal(resolve(config, { ...server, warmup: "off" }).builder.verification.serve, "unverified");
});

test("H3-Image Spark verifies native single-GB10 Auto recipes within measured scope", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const resource = config.commandBuilder.resource;
  const s = selection(config, { hw: "dgx-spark" });
  const recipe = resource.recommendedRecipe(s);
  assert.equal(recipe.unverified, undefined);
  assert.equal(recipe.adaln, "auto");
  assert.equal(recipe.encoder, "auto");
  assert.equal(recipe.placement, "auto");
  assert.equal(resource.limits.gpus_per_node.disabledWhen(s), true);
  assert.equal(resource.limits.gpus_per_node.disabledValue, 1);
  for (const task of [
    { size: "2048x2048" }, { mode: "edit", size: "2048x2048" },
    { size: "4736x3552" }, { mode: "edit", size: "4096x4096" }, { size: "6272x2688" },
  ]) {
    const result = resolve(config, { ...recipe, ...task });
    assert.deepEqual(result.builder.errors, []);
    assert.equal(result.builder.resolvedSettings.adaln, "Precompute");
    assert.deepEqual(result.builder.verification, { serve: "verified", request: "verified" });
    assert.equal(result.verified, true);
  }
  for (const override of [
    { size: "3552x4736" }, { size: "4096x4096" }, { delivery: "url" }, { png: "none" },
    { mode: "edit", size: "4096x4096", references: "two" }, { placement: "resident" },
    { placement: "offload" }, { warmup: "off" }, { conditioning: "off" },
    { mode: "edit", references: "two", quality: "exact" },
  ]) assert.equal(resolve(config, { ...recipe, ...override }).verified, false);
  assert.ok(resolve(config, { ...recipe, gpus_per_node: 2 }).builder.errors.length);
  assert.match(resolve(config, { ...recipe, adaln: "resident" }).builder.errors.join(" "), /requires native AdaLN precomputation/);
});

test("H3-Image conditioning reuse is a server choice with separate coverage", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const dim = config.overlayDims.find((entry) => entry.id === "conditioning");
  assert.equal(dim.scope, "serve");
  assert.deepEqual(dim.options.find((option) => option.id === "off").flags, ["--disable-conditioning-cache"]);
  const base = { hw: "h200", mode: "edit", size: "2048x2048" };
  assert.equal(resolve(config, base).verified, true);
  const off = resolve(config, { ...base, conditioning: "off" });
  assert.deepEqual(off.builder.errors, []);
  assert.deepEqual(off.builder.verification, { serve: "verified", request: "verified" });
  for (const request of [{ mode: "t2i" }, { size: "4096x4096" }, { references: "two" }, { quality: "exact" }]) {
    assert.deepEqual(resolve(config, { ...base, conditioning: "off", ...request }).builder.verification, { serve: "verified", request: "unverified" });
  }
  for (const server of [{ hw: "b300" }, { placement: "offload" }, { adaln: "online" }, { warmup: "off" }]) {
    assert.equal(resolve(config, { ...base, conditioning: "off", ...server }).verified, false);
  }
  assert.equal(curl(config, { ...base, conditioning: "off" }), curl(config, base));
  const recipe = config.commandBuilder.resource.recommendedRecipe(selection(config, { ...base, conditioning: "off" }));
  assert.equal(recipe.conditioning, "on");
  assert.equal(recipe.verifiedWhen(selection(config, { ...base, conditioning: "off" })), true);
});

test("H3-Image VAE scheduling keeps local coverage separate from request choices", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const dim = config.overlayDims.find((entry) => entry.id === "vae");
  assert.equal(dim.scope, "serve");
  assert.deepEqual(dim.options.find((option) => option.id === "local").flags, ["--vae-config.use-parallel-tiling false"]);
  const base = { hw: "h200", gpus_per_node: 2, encoder: "fold", size: "4736x3552" };
  assert.equal(resolve(config, { ...base, vae: "local" }).verified, true);
  assert.equal(resolve(config, { ...base, hw: "b300", vae: "local" }).verified, true);
  assert.deepEqual(resolve(config, { ...base, vae: "local", size: "2048x2048" }).builder.verification, { serve: "verified", request: "unverified" });
  for (const server of [{ conditioning: "off" }, { placement: "resident" }, { adaln: "online" }, { gpus_per_node: 1 }]) {
    assert.equal(resolve(config, { ...base, vae: "local", ...server }).verified, false);
  }
  assert.equal(curl(config, { ...base, vae: "local" }), curl(config, base));
  const recipe = config.commandBuilder.resource.recommendedRecipe(selection(config, { ...base, vae: "local" }));
  assert.equal(recipe.vae, "auto");
});

test("H3-Image PNG compression stays request-scoped and preserves the default payload", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  assert.equal(config.overlayDims.find((dim) => dim.id === "png").scope, "request");
  assert.doesNotMatch(curl(config), /output_compression/);
  const server = { hw: "h200", gpus_per_node: 2, encoder: "fold", size: "4736x3552" };
  const uncompressed = resolve(config, { ...server, png: "none" });
  assert.deepEqual(uncompressed.flags, resolve(config, server).flags);
  assert.deepEqual(uncompressed.builder.verification, { serve: "verified", request: "verified" });
  assert.deepEqual(resolve(config, { ...server, hw: "b300", png: "none" }).builder.verification, { serve: "verified", request: "verified" });
  const payload = JSON.parse(curl(config, { ...server, png: "none" }).match(/-d '([\s\S]+)' \|/)[1]);
  assert.equal(payload.output_compression, 0);
  assert.equal(payload.output_format, "png");
  assert.equal(payload.quality, "lossless");
  const edit = curl(config, { ...server, mode: "edit", size: "4096x4096", png: "none" });
  assert.match(edit, /-F 'output_compression=0'/);
  assert.equal(spawnSync("bash", ["-n"], { input: edit }).status, 0);
  for (const request of [{ size: "2048x2048" }, { quality: "exact" }, { mode: "edit", size: "4096x4096", references: "two" }, { vae: "local" }]) {
    assert.equal(resolve(config, { ...server, png: "none", ...request }).builder.verification.request, "unverified");
  }
});

test("H3-Image AMD recipes use AITER Auto placement and their measured 4K scope", () => {
  const config = get("/MiniMax/MiniMax-H3-Image.mdx");
  const resource = config.commandBuilder.resource;
  for (const hw of ["mi355x", "mi350x", "mi325x", "mi300x"]) {
    assert.ok(config.supportedHardware.includes(hw), hw);
    assert.equal(config.hardware.find((entry) => entry.id === hw).vendor, "amd");
    for (const [gpus_per_node, tp_size, ulysses_degree] of [[1, 1, 1], [2, 2, 1], [2, 1, 2], [4, 4, 1], [4, 1, 4], [8, 1, 8], [8, 8, 1]]) {
      const base = { hw, gpus_per_node, topology_mode: "manual", tp_size, ulysses_degree, ring_degree: 1 };
      for (const task of [
        { size: "4736x3552" }, { size: "6272x2688" }, { mode: "edit", size: "4096x4096" },
        { size: "2048x2048" }, { mode: "edit", size: "2048x2048" },
      ]) {
        const result = resolve(config, { ...base, ...task });
        assert.deepEqual(result.builder.errors, []);
        assert.equal(result.verified, true, JSON.stringify({ ...base, ...task }));
        assert.equal(result.builder.resolvedSettings.attention, "AITER (auto)");
        assert.equal(result.builder.resolvedSettings.adaln, "Keep weights");
        assert.ok(!result.flags.some((flag) => flag.startsWith("--attention-backend")));
      }
      for (const override of [
        { size: "5472x3072" }, { quality: "exact" }, { placement: "resident" }, { adaln: "online" },
        { warmup: "off" }, { quantization: "fp8-all" }, { mode: "edit", size: "4096x4096", references: "two" },
      ]) assert.equal(resolve(config, { ...base, size: "4736x3552", ...override }).verified, false);
    }
    for (const gpus_per_node of [1, 2, 4, 8]) {
      const recipe = resource.recommendedRecipe(selection(config, { hw, gpus_per_node }));
      assert.equal(recipe.ulysses_degree, gpus_per_node);
      assert.equal(recipe.unverified, undefined);
    }
    const ring = resolve(config, { hw, gpus_per_node: 2, topology_mode: "manual", tp_size: 1, ulysses_degree: 1, ring_degree: 2 });
    assert.match(ring.builder.errors.join(" "), /Ring parallelism requires FlashAttention/);
  }
  assert.equal(resolve(config, { hw: "h200", size: "4736x3552" }).builder.resolvedSettings.attention, undefined);
});

test("Wan preserves standard, multi-GPU, Ascend and LoRA recipes", () => {
  for (const version of ["1", "2"]) {
    const config = get(`/Wan/Wan2.${version}.mdx`);
    assert.equal(config.overlayDims.find((dim) => dim.id === "variant").options.length, 3);
    for (const hw of config.supportedHardware.filter((id) => !["a2", "a3", "xeon"].includes(id))) {
      const gpus = version === "2" && hw === "b300" ? 8 : 4;
      const result = resolve(config, { hw, gpus_per_node: gpus });
      assert.deepEqual(result.builder.errors, []);
      for (const flag of [
        "--dit-layerwise-offload true",
        `--num-gpus ${gpus}`,
        "--ulysses-degree 2",
        "--enable-cfg-parallel",
      ])
        assert.ok(result.flags.includes(flag));
    }
    const ascend = resolve(config, { hw: "a3", gpus_per_node: 8 });
    for (const flag of ["--tp-size 2", "--sp-degree 4", "--num-gpus 8", "--attention-backend laser_attn"])
      assert.ok(ascend.flags.includes(flag));
    assert.equal(body(config, { variant: "i2v-14b" }).input_reference, "{{INPUT_IMAGE}}");
  }
  assert.ok(resolve(get("/Wan/Wan2.1.mdx")).flags.includes("--lora-path NIVEDAN/wan2.1-lora"));
  assert.ok(!resolve(get("/Wan/Wan2.2.mdx")).flags.some((flag) => flag.startsWith("--lora-path")));
});

test("LTX request controls do not leak into startup and decoder loading is explicit", () => {
  const config = get("/LTX/LTX2.5.mdx");
  const startup = resolve(config, { decoder_weights: "loaded" });
  const request = resolve(config, { decoder_weights: "loaded", decoder: "diffusion", duration: "auto" });
  assert.deepEqual(request.flags, startup.flags);
  assert.ok(request.flags.includes("--load-diffusion-decoder"));
  assert.ok(resolve(config, { decoder: "diffusion" }).builder.errors.length);
  assert.equal(body(config, { decoder: "diffusion", duration: "auto" }).use_diffusion_decoder, true);
  assert.equal(body(config, { duration: "auto" }).auto_duration, true);
  assert.equal(body(config, { duration: "auto" }).num_frames, undefined);
  assert.equal(body(config).num_frames, 121);
  assert.equal(body(config, { pipeline: "two-stage", weights: "dev" }).size, "1920x1088");
  assert.equal(body(config, { weights: "dev" }).guidance_scale, 3);
  const previous = get("/LTX/LTX2 & LTX2.3.mdx");
  assert.equal(
    previous.commandBuilder.resource.verifiedRecipes.find((recipe) => recipe.hw === "cuda").device,
    "original",
  );
  assert.ok(
    resolve(previous, { gpus_per_node: 4, lora: "transition" }).flags.includes(
      "--lora-weight-name ltx2.3-transition.safetensors",
    ),
  );
});

test("structured prompts and realtime protocols remain model-specific", () => {
  const ideogram = get("/Ideogram/Ideogram4.mdx");
  const fal = body(ideogram, { checkpoint: "instant" });
  assert.ok(JSON.parse(fal.prompt).compositional_deconstruction.elements.length);
  assert.equal(fal.preset, undefined);
  assert.equal(body(ideogram).preset, "V4_QUALITY_48");
  const lingbot = JSON.parse(body(get("/LingBot-Video/LingBot-Video-MoE.mdx")).prompt);
  assert.ok(lingbot.prominent_elements[0].actions.length);
  for (const suffix of ["/LingBot-World/LingBot-World.mdx", "/LingBot-World/LingBot-World-2.0.mdx"])
    assert.match(curl(get(suffix)), /\/v1\/models$/);
  const sana = get("/SANA-WM/SANA-WM.mdx");
  assert.match(curl(sana, { checkpoint: "streaming", mode: "realtime" }), /\/v1\/models$/);
  assert.equal(body(sana).diffusers_kwargs.action, "w-80,wl-80,l-80,wj-80");
  assert.ok(resolve(sana, { checkpoint: "dense", mode: "realtime" }).builder.errors.length);
});
