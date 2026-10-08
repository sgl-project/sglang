import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path) => readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");
const load = async (name) => (await import("data:text/javascript," + encodeURIComponent(
  read(`src/snippets/configs/Kandinsky/${name}.jsx`),
))).config;
const generation = await load("kandinsky6");
const sr = await load("kandinsky6-sr");
const selection = (config, overrides = {}) => ({
  ...config.commandBuilder.defaultSelection,
  ...Object.fromEntries(config.overlayDims.map((dim) => [dim.id, dim.default])),
  ...overrides,
});
const flags = (config, s) => [
  ...config.commandBuilder.resolveDeployment(s).flags,
  ...config.overlayDims.flatMap((dim) => {
    if (dim.showWhen && !dim.showWhen(s)) return [];
    const option = dim.options.find((option) => option.id === s[dim.id]);
    if (!option || (typeof option.disabled === "function" ? option.disabled(s) : option.disabled)) return [];
    return (typeof option.flags === "function" ? option.flags(s) : option.flags) || [];
  }),
];

test("RTX recipe has a renderable hardware entry and uses SDPA", () => {
  assert.deepEqual(generation.hardware.find((hw) => hw.id === "rtxpro6000"), {
    id: "rtxpro6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell",
  });
  assert.ok(flags(generation, selection(generation, { hw: "rtxpro6000" }))
    .includes("--attention-backend torch_sdpa"));
  const invalid = generation.commandBuilder.resolveDeployment(selection(generation, {
    hw: "rtxpro6000", gpus_per_node: 2, topology_mode: "manual",
    tp_size: 1, ulysses_degree: 1, ring_degree: 2,
  })).builder;
  assert.equal(invalid.verification.serve, "error");
  assert.match(invalid.errors.join(" "), /SDPA.*Ring/);
});

test("placement verification is scoped; single-GPU FSDP is invalid", () => {
  const builder = generation.commandBuilder;
  assert.ok(builder.resolveDeployment(selection(generation, { placement: "fsdp" })).builder.errors.length);
  const s = selection(generation, { hw: "gb300", gpus_per_node: 4, placement: "fsdp" });
  assert.equal(builder.resolveDeployment(s).builder.verification.serve, "verified");
  assert.equal(builder.resolveDeployment({ ...s, placement: "offload" }).builder.verification.serve, "unverified");
  assert.ok(flags(generation, s).includes("--use-fsdp-inference true"));
});

test("checkpoint and conditioning choices change the correct command", () => {
  const s = selection(generation, { weights: "pro", mode: "image", workload: "smoke" });
  assert.ok(flags(generation, s).some((flag) => flag.endsWith("Kandinsky-6.0-Pro-5s-Diffusers")));
  assert.match(generation.curl(s), /input_reference=@/);
  assert.match(generation.curl(s), /num_inference_steps=2/);
  assert.doesNotMatch(generation.curl({ ...s, mode: "text" }), /input_reference/);
});

for (const config of [generation, sr]) {
  test(`${config.modelName}: invalid and custom topologies are not verified`, () => {
    const s = selection(config, { gpus_per_node: 3 });
    const invalid = config.commandBuilder.resolveDeployment(s).builder;
    assert.ok(invalid.errors.length);
    assert.equal(invalid.verification.serve, "error");
    const custom = config.commandBuilder.resolveDeployment(selection(config, {
      gpus_per_node: 4, topology_mode: "manual", tp_size: 4, ulysses_degree: 1, ring_degree: 1,
    })).builder;
    assert.equal(custom.errors.length, 0);
    assert.equal(custom.verification.serve, "unverified");
    assert.match(custom.warnings.join(" "), /TP/);
  });

  test(`${config.modelName}: dense backend overrides and Ring incompatibility`, () => {
    for (const attention of ["platform", "fa", "torch_sdpa"]) {
      const s = selection(config, { attention });
      const command = flags(config, s);
      assert.equal(command.filter((flag) => flag.startsWith("--attention-backend ")).length, 1);
      assert.ok(command.includes(`--attention-backend ${attention === "platform" ? "fa" : attention}`));
    }
    const invalid = config.commandBuilder.resolveDeployment(selection(config, {
      attention: "torch_sdpa", topology_mode: "manual", gpus_per_node: 2,
      tp_size: 1, ulysses_degree: 1, ring_degree: 2,
    })).builder;
    assert.equal(invalid.verification.serve, "error");
    assert.match(invalid.errors.join(" "), /SDPA.*Ring/);
  });

  test(`${config.modelName}: each component selection maps to one offload policy`, () => {
    const dim = config.overlayDims.find((dim) => dim.id === "offload_components");
    for (const placement of ["component", "offload"]) {
      for (const { id: offload_components } of dim.options) {
        const s = selection(config, { placement, offload_components });
        const command = flags(config, s);
        assert.ok(dim.showWhen(s));
        const prefix = placement === "component" ? "--cpu-offload-components" : "--layerwise-offload-components";
        assert.deepEqual(command.filter((flag) => /--(cpu|layerwise)-offload-components/.test(flag)), [
          `${prefix} ${offload_components}`,
        ]);
        assert.ok(!command.includes("--use-fsdp-inference true"));
        assert.equal(config.commandBuilder.resolveDeployment(s).builder.verification.serve, "unverified");
      }
    }
  });

  test(`${config.modelName}: DiT tuning is visible and emitted only when applicable`, () => {
    const s = selection(config, { placement: "offload", offload_components: "transformer", prefetch: 2, resident_layers: 4 });
    const tuning = config.overlayDims.filter((dim) => ["prefetch", "resident_layers"].includes(dim.id));
    assert.equal(tuning.length, 2);
    assert.ok(tuning.every((dim) => dim.showWhen(s) && !dim.verifiedWhen(s)));
    assert.ok(flags(config, s).includes("--dit-offload-prefetch-size 2"));
    assert.ok(flags(config, s).includes("--dit-layerwise-resident-layers 4"));
    assert.equal(config.commandBuilder.resolveDeployment(s).builder.resolvedSettings.resident_layers, "4 layers");
    for (const overrides of [
      { placement: "resident" }, { placement: "fsdp" }, { placement: "component" },
      { offload_components: config === sr ? "vae latent_upscaler" : "text_encoder" },
    ]) {
      const hidden = { ...s, ...overrides };
      assert.ok(tuning.every((dim) => !dim.showWhen(hidden)));
      assert.ok(!flags(config, hidden).some((flag) => /--dit-(offload-prefetch-size|layerwise-resident-layers)/.test(flag)));
    }
    const defaults = selection(config, { placement: "offload" });
    assert.ok(tuning.every((dim) => dim.verifiedWhen(defaults)));
    assert.ok(!flags(config, defaults).some((flag) => /--dit-(offload-prefetch-size|layerwise-resident-layers)/.test(flag)));
  });
}

test("Sage3 is a DiT-only, approximate RTX override, not a verified recipe", () => {
  const s = selection(generation, { hw: "rtxpro6000", attention: "sage_attn_3" });
  const command = flags(generation, s);
  assert.ok(command.includes("--attention-backend torch_sdpa"));
  assert.ok(command.includes("--component-attention-backends transformer=sage_attn_3"));
  const builder = generation.commandBuilder.resolveDeployment(s).builder;
  assert.equal(builder.errors.length, 0);
  assert.equal(builder.verification.serve, "unverified");
  assert.match(builder.warnings.join(" "), /approximate.*PSNR 11.97.*SNR -2.21/);
  for (const hw of ["b200", "gb300"]) {
    const invalid = generation.commandBuilder.resolveDeployment({ ...s, hw }).builder;
    assert.equal(invalid.verification.serve, "error");
    assert.match(invalid.errors.join(" "), /SM120/);
  }
  assert.match(generation.commandBuilder.resolveDeployment({
    ...s, topology_mode: "manual", gpus_per_node: 2, ulysses_degree: 1, ring_degree: 2,
  }).builder.errors.join(" "), /Sage3.*Ring/);
  assert.match(generation.commandBuilder.resolveDeployment({ ...s, attention: "fa" })
    .builder.errors.join(" "), /resolves to SDPA/);
});

test("Cache-DiT changes only the request and never inherits uncached verification", () => {
  const s = selection(generation, { workload: "smoke" });
  assert.equal(generation.commandBuilder.resolveDeployment(s).builder.verification.request, "verified");
  for (const mode of ["text", "image"]) {
    const uncached = { ...s, mode };
    const cached = { ...uncached, cache: "on" };
    assert.deepEqual(flags(generation, cached), flags(generation, uncached));
    assert.match(generation.curl(cached), mode === "text" ? /"enable_cache_dit": true/ : /enable_cache_dit=true/);
    assert.match(generation.curl(uncached), mode === "text" ? /"enable_cache_dit": false/ : /enable_cache_dit=false/);
    const builder = generation.commandBuilder.resolveDeployment(cached).builder;
    assert.equal(builder.verification.serve, "verified");
    assert.equal(builder.verification.request, "unverified");
    assert.match(builder.warnings.join(" "), /Cache-DiT is approximate/);
  }
  const fsdp = generation.commandBuilder.resolveDeployment(selection(generation, {
    hw: "gb300", gpus_per_node: 4, placement: "fsdp", cache: "on",
  })).builder;
  assert.equal(fsdp.errors.length, 0);
});

test("SR FSDP is selectable on multiple GPUs without claiming E2E verification", () => {
  const s = selection(sr, { placement: "fsdp" });
  const builder = sr.commandBuilder.resolveDeployment(s).builder;
  assert.ok(flags(sr, s).includes("--use-fsdp-inference true"));
  assert.equal(builder.errors.length, 0);
  assert.equal(builder.verification.serve, "unverified");
  assert.equal(sr.commandBuilder.resolveDeployment({ ...s, gpus_per_node: 1 }).builder.verification.serve, "error");
});

test("SR non-default attention and layer counts do not inherit verified HTTP recipes", () => {
  const s = selection(sr, {
    placement: "offload", topology_mode: "manual", gpus_per_node: 2,
    tp_size: 1, ulysses_degree: 1, ring_degree: 2,
  });
  assert.equal(sr.commandBuilder.resolveDeployment(s).builder.verification.serve, "verified");
  for (const overrides of [{ prefetch: 2 }, { resident_layers: 1 }, { offload_components: "transformer" }]) {
    assert.equal(sr.commandBuilder.resolveDeployment({ ...s, ...overrides }).builder.verification.serve, "unverified");
  }
  const sdpa = sr.commandBuilder.resolveDeployment(selection(sr, {
    attention: "torch_sdpa", topology_mode: "manual", tp_size: 2, ulysses_degree: 2,
  })).builder;
  assert.equal(sdpa.errors.length, 0);
  assert.equal(sdpa.verification.serve, "unverified");
});

test("SR decode modes emit mutually exclusive flags and explicit numerical boundaries", () => {
  for (const decode of ["serial", "tiles", "spatial"]) {
    const s = selection(sr, { decode });
    const command = flags(sr, s);
    assert.ok(command.includes(`--vae-config.use-parallel-tiling ${decode === "tiles"}`));
    assert.ok(command.includes(`--vae-config.use-parallel-decode ${decode === "spatial"}`));
    if (decode === "spatial") {
      assert.ok(command.includes("--vae-config.parallel-decode-mode spatial_shard"));
      const warnings = sr.commandBuilder.resolveDeployment(s).builder.warnings.join(" ");
      assert.match(warnings, /Not bitwise lossless/);
      assert.match(warnings, /57.51/);
      assert.match(warnings, /4\/255/);
      const option = sr.overlayDims.find((dim) => dim.id === "decode").options.find((entry) => entry.id === decode);
      assert.match(option.hints.join(" "), /Not bitwise lossless/);
    }
  }
});

test("SR topology uses the released checkpoint's 28 heads, not the class default", () => {
  const resource = sr.commandBuilder.resource;
  assert.ok(resource.validateTopology({ nodes: 1, gpus_per_node: 8 }, {
    tp_size: 1, ulysses_degree: 8, ring_degree: 1,
  }).some((message) => message.includes("28 DiT heads")));
});

test("SR hardware extrapolation never inherits GB300 execution verification", () => {
  for (const hw of ["b200", "h200", "h100", "rtx5090"]) {
    assert.ok(sr.supportedHardware.includes(hw));
    for (const gpus_per_node of [1, 2, 4]) {
      const s = selection(sr, { hw, gpus_per_node });
      const builder = sr.commandBuilder.resolveDeployment(s).builder;
      assert.equal(builder.errors.length, 0);
      assert.equal(builder.verification.serve, "unverified");
      assert.equal(builder.verification.request, "unverified");
      assert.match(builder.warnings.join(" "), /Extrapolated hardware recipe/);
      assert.ok(flags(sr, s).includes(`--attention-backend ${hw === "rtx5090" ? "torch_sdpa" : "fa"}`));
    }
  }
  const consumer = sr.commandBuilder.resource.verifiedRecipes.find((r) => r.hw === "rtx5090");
  assert.equal(consumer.gpus_per_node, 1);
  assert.equal(consumer.placement, "offload");
  assert.equal(consumer.unverified, true);
  assert.match(sr.commandBuilder.resolveDeployment(selection(sr, { hw: "rtx5090" }))
    .builder.warnings.join(" "), /may still OOM/);
});

test("SR eight-GPU auto topology respects heads and backend Ring capability", () => {
  assert.equal(sr.commandBuilder.resource.limits.gpus_per_node.max, 8);
  for (const hw of ["gb300", "b200", "h200", "h100"]) {
    const s = selection(sr, { hw, gpus_per_node: 8 });
    const builder = sr.commandBuilder.resolveDeployment(s).builder;
    assert.deepEqual(builder.topology, { tp_size: 1, ulysses_degree: 4, ring_degree: 2 });
    assert.equal(builder.errors.length, 0);
    assert.equal(builder.verification.serve, "unverified");
    assert.match(builder.warnings.join(" "), /Eight GPUs.*Ulysses4 x Ring2/);
  }
  for (const hw of ["rtxpro6000", "rtx5090"]) {
    const builder = sr.commandBuilder.resolveDeployment(selection(sr, { hw, gpus_per_node: 8 })).builder;
    assert.equal(builder.verification.serve, "error");
    assert.match(builder.errors.join(" "), /SDPA.*Ring/);
    assert.match(sr.commandBuilder.resolveDeployment(selection(sr, { hw, attention: "fa" }))
      .builder.errors.join(" "), /resolves to SDPA/);
  }
});

test("SR RTX HTTP verification covers only the executed placements and topologies", () => {
  for (const [gpus_per_node, decode] of [[1, "serial"], [2, "tiles"]]) {
    const s = selection(sr, { hw: "rtxpro6000", gpus_per_node, decode });
    assert.equal(sr.commandBuilder.resolveDeployment(s).builder.verification.serve, "verified");
    assert.equal(sr.commandBuilder.resolveDeployment(s).builder.verification.request, "unverified");
    for (const overrides of [{ placement: "offload" }, { decode: "spatial" }, { gpus_per_node: 4 }]) {
      assert.equal(sr.commandBuilder.resolveDeployment({ ...s, ...overrides }).builder.verification.serve, "unverified");
    }
  }
  const recipe = sr.commandBuilder.resource.verifiedRecipes.find((r) => r.hw === "rtxpro6000" && r.default);
  assert.equal(recipe.decode, "serial");
  assert.equal(recipe.unverified, false);
});

test("SR tile batching changes only the request and warns about changed output", () => {
  const s = selection(sr);
  const batched = { ...s, tile_batch: "2", scale: "2.25" };
  assert.deepEqual(flags(sr, s), flags(sr, batched));
  assert.match(sr.curl(batched), /sr_tiles_batch_size=2/);
  assert.match(sr.curl(batched), /sr_resolution_scale=2.25/);
  assert.match(sr.curl(batched), /video_reference=@/);
  assert.match(sr.commandBuilder.resolveDeployment(batched).builder.warnings.join(" "), /not lossless/);
});

test("SR execution verification does not promise quality or arbitrary-video verification", () => {
  const s = selection(sr, { topology_mode: "manual", tp_size: 2, ulysses_degree: 2 });
  const builder = sr.commandBuilder.resolveDeployment(s).builder;
  assert.equal(builder.verification.serve, "verified");
  assert.equal(builder.verification.request, "unverified");
  assert.match(builder.warnings.join(" "), /not lossless output/);
  assert.match(sr.commandBuilder.resolveDeployment(selection(sr, { gpus_per_node: 1 })).builder.warnings.join(" "), /no parallel speedup/);
});

test("both cookbook pages use the shared builder and consistent ports", () => {
  for (const name of ["Kandinsky6", "Kandinsky6-SR"]) {
    const page = read(`cookbook/diffusion/Kandinsky/${name}.mdx`);
    assert.match(page, /<Deployment config=\{config\} \/>/);
    assert.ok(page.indexOf("## 1. Quick start") < page.indexOf("## 2. Model capabilities"));
    assert.match(page, /HF_TOKEN/);
  }
  for (const config of [generation, sr]) {
    assert.equal(config.placeholders.PORT.default, "30000");
    assert.equal(config.placeholders.CURL_PORT.default, "30000");
  }
  const page = read("cookbook/diffusion/Kandinsky/Kandinsky6-SR.mdx");
  assert.match(page, /Spatial shard is not bitwise lossless/);
  assert.match(page, /57.51 dB/);
  assert.match(page, /0.99921/);
  assert.match(page, /4\/255/);
});
