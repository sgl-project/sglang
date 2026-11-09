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
      id: "execution", title: "Execution", scope: "serve", default: "compile",
      description: "Compile for repeated requests. Eager avoids compilation at startup and for new shapes.",
      learnMore: "#5-measured-tuning",
      options: [
        { id: "compile", label: "Compiled", flags: ["--enable-torch-compile true"], recommended: true },
        { id: "eager", label: "Eager", flags: [] },
      ],
    },
    {
      id: "attention", title: "Attention", scope: "serve", default: "platform",
      description: "Exact attention backends can differ in floating-point reduction order.",
      options: [
        { id: "platform", label: "Platform default", flags: (s) => [`--attention-backend ${s.hw === "h200" ? "fa" : s.hw === "rtx5090" ? "torch_cudnn_sdpa" : "torch_sdpa"}`], recommended: true },
        { id: "fa", label: "FlashAttention", flags: ["--attention-backend fa"], soft: (s) => s.hw !== "h200", softReason: "On RTX 5090 and Spark, this selector falls back to Torch SDPA. Select Torch SDPA explicitly." },
        { id: "sdpa", label: "Torch SDPA", flags: ["--attention-backend torch_sdpa"] },
        { id: "cudnn", label: "cuDNN SDPA", flags: ["--attention-backend torch_cudnn_sdpa"] },
      ],
    },
    {
      id: "cfg", title: "CFG parallelism", scope: "serve", default: "auto",
      description: "Auto uses CFG parallelism on two H200 GPUs. Manual TP/SP overrides disable automatic CFG splitting.",
      learnMore: "#5-measured-tuning",
      options: [
        { id: "auto", label: "Auto", flags: [] },
        { id: "off", label: "Off", flags: [] },
        { id: "on", label: "On", flags: [], disabled: (s) => Number(s.gpus_per_node) % 2 !== 0, disableReason: "CFG parallelism requires an even number of GPUs and guidance greater than 1." },
      ],
    },
    {
      id: "vae", title: "VAE decoding", scope: "serve", default: "auto",
      description: "Auto disables tiling for compiled H200 recipes. Untiled decoding uses about 4-5 GiB more memory at 1024 x 1024.",
      learnMore: "#5-measured-tuning",
      options: [
        { id: "auto", label: "Auto", flags: [] },
        { id: "tiled", label: "Tiled", flags: [] },
        { id: "full", label: "Untiled", flags: [] },
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
        { id: "rtx5090-1gpu-resident-cudnn", hw: "rtx5090", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", attention: "cudnn", execution: "compile", cfg: "auto", vae: "auto" },
        { id: "dgx-spark-1gpu-resident-sdpa", hw: "dgx-spark", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", attention: "sdpa", execution: "compile", cfg: "auto", vae: "auto" },
        { id: "h200-1gpu-resident-fa", hw: "h200", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", attention: "fa", execution: "compile", cfg: "auto", vae: "auto" },
        { id: "h200-2gpu-resident-cfg", hw: "h200", nodes: 1, gpus_per_node: 2, tp_size: 1, ulysses_degree: 1, ring_degree: 1, placement: "resident", attention: "fa", execution: "compile", cfg: "auto", vae: "auto" },
      ],
      cfgDegree: (s) => s.cfg === "on" || (s.cfg === "auto" && s.topology_mode !== "manual" && s.hw === "h200" && Number(s.gpus_per_node) === 2) ? 2 : 1,
      vaeTiling: (s) => s.vae === "tiled" || (s.vae === "auto" && (s.hw !== "h200" || s.execution !== "compile")),
      autoTopology: (s) => ({ tp_size: Number(s.gpus_per_node) / config.commandBuilder.resource.cfgDegree(s), ulysses_degree: 1, ring_degree: 1 }),
      validateTopology: (s, t) => {
        const errors = [];
        if (Number(s.nodes) !== 1) errors.push("This picker covers single-node deployment only.");
        if (s.hw === "dgx-spark" && Number(s.gpus_per_node) !== 1) errors.push("DGX Spark has one GPU per node.");
        const cfg = config.commandBuilder.resource.cfgDegree(s);
        if (!Number.isInteger(t.tp_size) || t.tp_size < 1 || Number(s.nodes) * Number(s.gpus_per_node) !== cfg * t.tp_size * t.ulysses_degree * t.ring_degree) errors.push("GPU count must equal CFG * TP * Ulysses * Ring.");
        if (16 % (t.tp_size * t.ulysses_degree)) errors.push("TP * Ulysses must divide Anima's 16 attention heads.");
        if (t.ring_degree > 1 && (s.hw !== "h200" || ["sdpa", "cudnn"].includes(s.attention))) errors.push("Ring requires FlashAttention.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const r = config.commandBuilder.resource;
      const attention = s.attention === "platform" ? (s.hw === "h200" ? "fa" : s.hw === "rtx5090" ? "cudnn" : "sdpa") : s.attention;
      const t = s.topology_mode === "manual"
        ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
        : r.autoTopology(s);
      const errors = r.validateTopology(s, t);
      const cfg = r.cfgDegree(s);
      const tiled = r.vaeTiling(s);
      const world = Number(s.nodes) * Number(s.gpus_per_node);
      const recipe = r.verifiedRecipes.find((v) => v.hw === s.hw && v.gpus_per_node === Number(s.gpus_per_node) && v.tp_size === t.tp_size && v.ulysses_degree === t.ulysses_degree && v.ring_degree === t.ring_degree && v.placement === s.placement && r.cfgDegree(v) === cfg);
      const attentions = world > 1 ? ["fa"] : s.execution === "eager"
        ? ["sdpa", "cudnn", ...(s.hw === "h200" ? ["fa"] : [])]
        : s.hw === "rtx5090" ? ["sdpa", "cudnn"] : [s.hw === "h200" ? "fa" : "sdpa"];
      const multiOutput = tiled && (s.execution === "compile" || attention === "fa" || (attention === "sdpa" && s.hw !== "h200"));
      const vaeVerified = tiled || (s.execution === "compile" && attention === (s.hw === "h200" ? "fa" : "sdpa"));
      const verified = !!recipe && attentions.includes(attention) && vaeVerified && Number(s.outputs) <= (multiOutput ? 2 : 1) && errors.length === 0;
      const flags = ["--model-path {{MODEL_NAME}}"];
      if (world > 1) flags.push(`--num-gpus ${world}`, `--tp-size ${t.tp_size}`, `--ulysses-degree ${t.ulysses_degree}`, `--ring-degree ${t.ring_degree}`);
      if (cfg > 1) flags.push("--enable-cfg-parallel");
      if (!tiled) flags.push("--vae-tiling false", "--vae-sp false");
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      return {
        match: { hw: s.hw }, nnodes: Number(s.nodes), verified, flags,
        builder: {
          topology: t, topologySummary: `CFG ${cfg} / TP ${t.tp_size} / Ulysses ${t.ulysses_degree} / Ring ${t.ring_degree}`,
          errors, warnings: verified ? [] : ["This exact HTTP recipe has not been verified."],
          verification: { serve: verified ? "verified" : "unverified", request: verified ? "verified" : "unverified" },
          resolvedSettings: { attention: s.attention === "platform" ? ({ fa: "FlashAttention", sdpa: "Torch SDPA", cudnn: "cuDNN SDPA" }[attention]) : undefined, cfg: cfg > 1 ? "2-way CFG" : "Off", vae: tiled ? "Tiled" : "Untiled" },
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
