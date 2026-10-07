export const config = {
  modelName: "Ovis-Image 7B",
  supportedHardware: ["b200"],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    { id: "weights", title: "Checkpoint", scope: "base", default: "default", options: [{ id: "default", label: "Diffusers BF16" }] },
    { id: "mode", title: "Request mode", scope: "base", default: "text", options: [{ id: "text", label: "Text to image" }] },
    {
      id: "placement", title: "Placement", scope: "serve", default: "resident",
      description: "Offload reduces resident weight memory and adds CPU transfers.",
      options: [
        { id: "resident", label: "Resident", flags: ["--performance-mode speed"], recommended: true },
        { id: "component", label: "Component offload", flags: ["--cpu-offload-components transformer,text_encoder,vae"] },
        { id: "layerwise", label: "Layerwise offload", flags: ["--layerwise-offload-components transformer,text_encoder,vae"] },
      ],
    },
    {
      id: "attention", title: "Attention", scope: "serve", default: "sdpa",
      description: "Ring parallelism requires FlashAttention.",
      options: [
        { id: "sdpa", label: "Torch SDPA", flags: ["--attention-backend torch_sdpa"], recommended: true },
        { id: "fa", label: "FlashAttention", flags: ["--attention-backend fa"] },
      ],
    },
    {
      id: "cfg", title: "CFG parallelism", scope: "serve", default: "off",
      description: "Split positive and negative denoising across two groups; guidance must exceed one.",
      options: [{ id: "off", label: "Off" }, { id: "on", label: "On" }],
    },
    {
      id: "vae", title: "VAE decode", scope: "serve", default: "full",
      description: "Tiled and spatial examples request 1536 × 1024 to exceed the native 1024-pixel tiling threshold.",
      options: [
        { id: "full", label: "Untiled", flags: ["--vae-tiling false", "--vae-sp false"] },
        { id: "tiled", label: "Tiled", flags: ["--vae-tiling true", "--vae-sp false"] },
        { id: "spatial", label: "Spatial parallel", flags: ["--vae-tiling true", "--vae-sp true", "--vae-config.parallel-decode-mode spatial_shard"] },
      ],
    },
    { id: "outputs", title: "Outputs", scope: "request", kind: "number", default: 1, min: 1, max: 4, options: [] },
  ],
  commandBuilder: {
    defaultSelection: { hw: "b200", nodes: 1, gpus_per_node: 1, topology_mode: "auto", tp_size: 1, ulysses_degree: 1, ring_degree: 1 },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 4 } },
      // Populate only with measured HTTP recipes, never inferred hardware coverage.
      verifiedRecipes: [],
      autoTopology: (s) => ({ tp_size: Number(s.gpus_per_node) / (s.cfg === "on" ? 2 : 1), ulysses_degree: 1, ring_degree: 1 }),
      validateTopology: (s, t) => {
        const cfg = s.cfg === "on" ? 2 : 1;
        const errors = [];
        if (Number(s.nodes) !== 1 || Number(s.gpus_per_node) !== cfg * t.tp_size * t.ulysses_degree * t.ring_degree) errors.push("GPU count must equal CFG × TP × Ulysses × Ring on one node.");
        if (!Number.isInteger(t.tp_size) || t.tp_size < 1 || 24 % (t.tp_size * t.ulysses_degree)) errors.push("TP × Ulysses must divide 24 attention heads.");
        if (t.ring_degree > 1 && s.attention !== "fa") errors.push("Ring requires FlashAttention.");
        if (s.vae === "spatial" && Number(s.gpus_per_node) < 2) errors.push("Spatial parallel decode requires at least two GPUs.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const r = config.commandBuilder.resource;
      const t = s.topology_mode === "manual" ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) } : r.autoTopology(s);
      const errors = r.validateTopology(s, t);
      const recipe = r.verifiedRecipes.find((v) => v.hw === s.hw && v.gpus_per_node === Number(s.gpus_per_node) && v.tp_size === t.tp_size && v.ulysses_degree === t.ulysses_degree && v.ring_degree === t.ring_degree && v.placement === s.placement && v.attention === s.attention && v.cfg === s.cfg && v.vae === s.vae);
      const flags = ["--model-path {{MODEL_NAME}}", `--num-gpus ${s.gpus_per_node}`, `--tp-size ${t.tp_size}`, `--ulysses-degree ${t.ulysses_degree}`, `--ring-degree ${t.ring_degree}`];
      if (s.cfg === "on") flags.push("--enable-cfg-parallel");
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      const verified = !!recipe && errors.length === 0;
      return { match: { hw: s.hw }, nnodes: 1, verified, flags, builder: { topology: t, topologySummary: `CFG ${s.cfg === "on" ? 2 : 1} / TP ${t.tp_size} / Ulysses ${t.ulysses_degree} / Ring ${t.ring_degree}`, errors, warnings: verified ? [] : ["This exact HTTP recipe has not been verified."], verification: { serve: verified ? "verified" : "unverified", request: verified ? "verified" : "unverified" } } };
    },
  },
  modelNames: { default: "ATH-MaaS/Ovis-Image-7B" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: (s) => `curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt: "A shop sign reading HELLO, beside a red bicycle.", size: s.vae === "tiled" || s.vae === "spatial" ? "1536x1024" : "1024x1024", n: Number(s.outputs), num_inference_steps: 50, guidance_scale: 5, negative_prompt: "", seed: 42, generator_device: "cpu", response_format: "b64_json", output_format: "png" }, null, 2)}'`,
  cells: [],
};
