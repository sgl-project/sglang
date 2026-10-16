import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/snippets/configs/Wan/wan-animate-2.jsx", import.meta.url), "utf8");
const { config } = await import("data:text/javascript," + encodeURIComponent(source));
const selection = (overrides = {}) => ({
  ...config.commandBuilder.defaultSelection,
  ...Object.fromEntries(config.overlayDims.map((dim) => [dim.id, dim.default])),
  ...overrides,
});
const resolve = (overrides = {}) => config.commandBuilder.resolveDeployment(selection(overrides));

// Independent expectations from the legacy picker's seven deployment presets.
const presets = [
  ["single", 1, "off", 1, 1, []],
  ["cfg2", 2, "on", 1, 1, ["--enable-cfg-parallel"]],
  ["sp2", 2, "off", 1, 2, ["--ulysses-degree 2"]],
  ["tp2", 2, "off", 2, 1, ["--tp-size 2"]],
  ["cfg-sp4", 4, "on", 1, 2, ["--enable-cfg-parallel", "--ulysses-degree 2"]],
  ["tp-sp4", 4, "off", 2, 2, ["--tp-size 2", "--ulysses-degree 2"]],
  ["cfg-tp-sp8", 8, "on", 2, 2, ["--enable-cfg-parallel", "--tp-size 2", "--ulysses-degree 2"]],
];
const placementFlags = {
  auto: [],
  speed: ["--performance-mode speed"],
  memory: ["--performance-mode memory"],
  offload: ["--dit-layerwise-offload", "--layerwise-offload-components transformer"],
  "cfg-regression": ["--dit-layerwise-offload", "--layerwise-offload-components transformer",
    "--encoder-parallel replicate", "--vae-config.use-parallel-decode false"],
};

test("all 233 legacy selections preserve flags, environment and verification", () => {
  let combinations = 0;
  let verified = 0;
  for (const hw of ["h200", "b300", "rtx4090", "gb200", "h100"]) {
    for (const [preset, gpus_per_node, cfg, tp_size, ulysses_degree, flags] of presets) {
      for (const placement of Object.keys(placementFlags)) {
        for (const encoder of ["auto", "offload"]) {
          if (placement === "cfg-regression" && preset !== "cfg2") continue;
          if (hw === "rtx4090" && (preset !== "single" || placement !== "memory" || encoder !== "auto")) continue;
          const s = { hw, gpus_per_node, cfg, tp_size, ulysses_degree, placement, encoder, topology_mode: "manual" };
          const result = resolve(s);
          const expectedFlags = ["--model-path {{MODEL_NAME}}", "--port {{PORT}}", `--num-gpus ${gpus_per_node}`,
            ...flags, ...placementFlags[placement], ...(encoder === "offload" ? ["--text-encoder-cpu-offload"] : [])];
          const expectedEnv = hw === "rtx4090" ? ["PYTORCH_ALLOC_CONF=expandable_segments:True"]
            : placement === "cfg-regression" ? ["SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=1"] : [];
          const isVerified = encoder === "auto" && (
            (["h200", "b300"].includes(hw) && preset === "single" && ["auto", "memory"].includes(placement))
            || (hw === "b300" && preset === "cfg2" && ["auto", "cfg-regression"].includes(placement))
            || hw === "rtx4090"
          );
          const label = JSON.stringify(s);
          assert.deepEqual(result.builder.errors, [], label);
          assert.deepEqual([...result.flags].sort(), expectedFlags.sort(), label);
          assert.deepEqual(result.env, expectedEnv, label);
          assert.equal(result.verified, isVerified, label);
          assert.deepEqual(result.builder.verification, {
            serve: isVerified ? "verified" : "unverified",
            request: isVerified ? "verified" : "unverified",
          }, label);
          combinations++;
          verified += Number(isVerified);
        }
      }
    }
  }
  assert.equal(combinations, 233);
  assert.equal(verified, 7);
});

test("H3 builder uses scoped settings and preserves the default HTTP workload", () => {
  assert.equal(config.cells, undefined);
  assert.deepEqual(config.overlayDims.filter((d) => d.scope === "serve").map((d) => d.id), ["placement", "encoder", "cfg"]);
  assert.deepEqual(config.overlayDims.filter((d) => d.scope === "request").map((d) => d.id), ["resolution", "clip_len", "steps", "audio"]);
  assert.equal(config.overlayDims.find((d) => d.id === "steps").unit, "steps");
  assert.equal(config.curl(selection()), [
    "curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos",
    '--form-string "prompt=a person dancing"',
    '--form-string "video_path={{INPUT_VIDEO}}"',
    '--form-string "clip_len=37"',
    '--form-string "size=640x800"',
    '--form-string "num_inference_steps=40"',
    '--form-string "guidance_scale=3.0"',
    '--form-string "fps=16"',
    '--form-string "seed=42"',
    '--form-string "enable_audio=false"',
    '--form "input_reference=@{{INPUT_IMAGE}};type=image/png"',
  ].join(" \\\n  "));
});

test("recommended recipes have accurate verification and consumer memory settings", () => {
  const recipes = config.commandBuilder.resource.verifiedRecipes;
  assert.equal(recipes.filter((r) => !r.unverified).length, 7);
  for (const hw of config.supportedHardware) {
    assert.equal(recipes.filter((r) => r.hw === hw && r.default).length, 1);
  }
  for (const recipe of recipes) {
    const result = resolve({ ...recipe, topology_mode: "manual" });
    assert.deepEqual(result.builder.errors, []);
    assert.equal(result.verified, !recipe.unverified);
  }
  const consumer = resolve({ hw: "rtx4090" });
  assert.ok(consumer.flags.includes("--performance-mode memory"));
  assert.deepEqual(consumer.env, ["PYTORCH_ALLOC_CONF=expandable_segments:True"]);
  assert.equal(consumer.verified, true);
  assert.equal(resolve({ hw: "h100", placement: "offload" }).verified, false);
  assert.equal(resolve({ hw: "gb200" }).verified, false);
});

test("request changes affect only the payload and request verification", () => {
  const baseline = resolve();
  for (const [overrides, expected] of [
    [{ steps: 20 }, "num_inference_steps=20"],
    [{ clip_len: "65" }, "clip_len=65"],
    [{ resolution: "720x1280" }, "size=720x1280"],
    [{ audio: "true" }, "enable_audio=true"],
  ]) {
    const result = resolve(overrides);
    assert.deepEqual(result.flags, baseline.flags);
    assert.deepEqual(result.env, baseline.env);
    assert.deepEqual(result.builder.verification, { serve: "verified", request: "unverified" });
    assert.ok(config.curl(selection(overrides)).includes(expected));
  }
});

test("automatic topology accounts for CFG ranks", () => {
  for (const [gpus_per_node, cfg, tp, ulysses] of [
    [1, "auto", 1, 1], [2, "auto", 1, 1], [4, "auto", 1, 2], [8, "auto", 2, 2],
    [2, "off", 1, 2], [4, "off", 2, 2],
  ]) {
    const result = resolve({ gpus_per_node, cfg });
    assert.deepEqual(result.builder.errors, []);
    assert.deepEqual(result.builder.topology, { tp_size: tp, ulysses_degree: ulysses, ring_degree: 1 });
  }
  assert.equal(resolve({ gpus_per_node: 4 }).builder.verification.serve, "unverified");
});

test("invalid resource and topology combinations cannot be verified", () => {
  for (const overrides of [
    { nodes: 2 }, { gpus_per_node: 0 }, { gpus_per_node: 9 }, { gpus_per_node: 1.5 },
    { cfg: "on" }, { gpus_per_node: 3, cfg: "on" }, { gpus_per_node: 6, cfg: "off" },
    { topology_mode: "manual", tp_size: 0 },
    { topology_mode: "manual", gpus_per_node: 2, cfg: "off" },
    { topology_mode: "manual", gpus_per_node: 2, cfg: "off", ring_degree: 2 },
    { placement: "cfg-regression" },
    { placement: "cfg-regression", gpus_per_node: 2, cfg: "off" },
    { hw: "rtx4090", placement: "speed" }, { hw: "rtx4090", encoder: "offload" },
  ]) {
    const result = resolve(overrides);
    assert.ok(result.builder.errors.length, JSON.stringify(overrides));
    assert.equal(result.verified, false);
    assert.deepEqual(result.builder.verification, { serve: "error", request: "error" });
  }
});
