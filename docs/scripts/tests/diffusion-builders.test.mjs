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
const get = (suffix) => [...configs].find(([path]) => path.endsWith(suffix))[1];
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
  assert.equal(flux.supportedHardware.length, 10);
  assert.equal(qwen.supportedHardware.length, 9);
  assert.equal(edit.supportedHardware.length, 7);
  assert.equal(zimage.supportedHardware.length, 9);
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

test("Wan preserves standard, multi-GPU, Ascend and LoRA recipes", () => {
  for (const version of ["1", "2"]) {
    const config = get(`/Wan/Wan2.${version}.mdx`);
    assert.equal(config.overlayDims.find((dim) => dim.id === "variant").options.length, 3);
    for (const hw of config.supportedHardware.filter((id) => !["a2", "a3"].includes(id))) {
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
