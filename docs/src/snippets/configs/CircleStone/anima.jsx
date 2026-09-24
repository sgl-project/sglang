export const config = {
  modelName: "Anima Base v1.0",
  supportedHardware: ["rtx5090", "dgx-spark", "h200"],
  hardware: [
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "consumer" },
    { id: "dgx-spark", label: "DGX Spark", vram: "128GB unified", vendor: "blackwell" },
  ],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "weights", title: "Checkpoint", scope: "base", default: "default",
      description: "Official self-contained Diffusers checkpoint.",
      options: [{ id: "default", label: "Base v1.0" }],
    },
    {
      id: "mode", title: "Request mode", scope: "base", default: "text",
      options: [{ id: "text", label: "Text to image" }],
    },
    {
      id: "placement", title: "Placement", scope: "serve", default: "resident",
      description: "Keep the pipeline resident when it fits. Use offload for a smaller memory budget.",
      learnMore: "#4-runtime-features",
      options: [
        { id: "resident", label: "Resident", flags: ["--performance-mode speed"], recommended: true },
        { id: "offload", label: "Layerwise offload", flags: ["--performance-mode memory", "--dit-layerwise-offload true"], soft: true, softReason: "This exact HTTP recipe is unverified." },
      ],
    },
    {
      id: "attention", title: "Attention", scope: "serve", default: "platform",
      description: "Exact attention backends can differ in floating-point reduction order.",
      options: [
        { id: "platform", label: "Platform default", flags: (s) => [`--attention-backend ${s.hw === "h200" ? "fa" : "torch_sdpa"}`], recommended: true },
        { id: "fa", label: "FlashAttention", flags: ["--attention-backend fa"], soft: (s) => s.hw !== "h200", softReason: "On RTX 5090 and Spark, this selector falls back to Torch SDPA. Select Torch SDPA explicitly." },
        { id: "sdpa", label: "Torch SDPA", flags: ["--attention-backend torch_sdpa"], soft: (s) => s.hw === "h200", softReason: "Offline generation verified; this HTTP recipe is unverified." },
      ],
    },
    {
      id: "outputs", title: "Outputs", scope: "request", kind: "number",
      default: 1, min: 1, max: 4, options: [],
    },
  ],
  commandBuilder: {
    defaultSelection: { hw: "rtx5090", nodes: 1, gpus_per_node: 1, topology_mode: "auto", tp_size: 1, ulysses_degree: 1, ring_degree: 1 },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
      verifiedRecipes: [
        { id: "rtx5090-1gpu-resident-sdpa", hw: "rtx5090", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", attention: "sdpa" },
        { id: "dgx-spark-1gpu-resident-sdpa", hw: "dgx-spark", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", attention: "sdpa" },
        { id: "h200-1gpu-resident-fa", hw: "h200", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", attention: "fa" },
      ],
      autoTopology: (s) => ({ tp_size: Number(s.gpus_per_node), ulysses_degree: 1, ring_degree: 1 }),
      validateTopology: (s, t) => {
        const errors = [];
        if (Number(s.nodes) !== 1) errors.push("This picker covers single-node deployment only.");
        if (s.hw === "dgx-spark" && Number(s.gpus_per_node) !== 1) errors.push("DGX Spark has one GPU per node.");
        if (Number(s.nodes) * Number(s.gpus_per_node) !== t.tp_size * t.ulysses_degree * t.ring_degree) errors.push("GPU count must equal TP * Ulysses * Ring.");
        if (16 % (t.tp_size * t.ulysses_degree)) errors.push("TP * Ulysses must divide Anima's 16 attention heads.");
        if (t.ring_degree > 1 && (s.hw !== "h200" || s.attention === "sdpa")) errors.push("Ring requires FlashAttention; this hardware/backend selection uses Torch SDPA.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const r = config.commandBuilder.resource;
      const attention = s.attention === "platform" ? (s.hw === "h200" ? "fa" : "sdpa") : s.attention;
      const t = s.topology_mode === "manual"
        ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
        : r.autoTopology(s);
      const errors = r.validateTopology(s, t);
      const recipe = r.verifiedRecipes.find((v) => v.hw === s.hw && v.gpus_per_node === Number(s.gpus_per_node) && v.tp_size === t.tp_size && v.ulysses_degree === t.ulysses_degree && v.ring_degree === t.ring_degree && v.placement === s.placement && v.attention === attention);
      const verified = !!recipe && Number(s.outputs) === 1 && errors.length === 0;
      const flags = ["--model-path {{MODEL_NAME}}"];
      const world = Number(s.nodes) * Number(s.gpus_per_node);
      if (world > 1) flags.push(`--num-gpus ${world}`, `--tp-size ${t.tp_size}`, `--ulysses-degree ${t.ulysses_degree}`, `--ring-degree ${t.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      return {
        match: { hw: s.hw }, nnodes: Number(s.nodes), verified, flags,
        builder: {
          topology: t, topologySummary: `TP ${t.tp_size} / Ulysses ${t.ulysses_degree} / Ring ${t.ring_degree}`,
          errors, warnings: verified ? [] : ["This exact HTTP recipe has not been verified."],
          verification: { serve: verified ? "verified" : "unverified", request: verified ? "verified" : "unverified" },
          resolvedSettings: { attention: s.attention === "platform" ? (attention === "fa" ? "FlashAttention" : "Torch SDPA") : undefined },
        },
      };
    },
  },
  modelNames: { default: "circlestone-labs/Anima-Base-v1.0-Diffusers" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: (s) => `curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt: "masterpiece, best quality, safe, watercolor landscape, a quiet seaside village at sunset", size: "1024x1024", n: Number(s.outputs), seed: 42, generator_device: "cpu", output_format: "png", response_format: "b64_json" }, null, 2)}'`,
  cells: [],
};
