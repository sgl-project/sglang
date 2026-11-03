export const config = {
  modelName: "MOSS-Image",
  supportedHardware: ["h200", "h100", "rtx5090", "rtx6000pro"],
  hardware: [
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "consumer" },
    { id: "rtx6000pro", label: "RTX PRO 6000 Blackwell", vram: "96GB", vendor: "consumer" },
  ],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "placement", title: "Placement", scope: "serve", default: "auto",
      description: "Auto uses the model's memory policy. Resident weights need room for activations too.",
      options: [
        { id: "auto", label: "Auto", recommended: true },
        { id: "resident", label: "Resident", flags: ["--component-residency all=resident"] },
        { id: "offload", label: "Layerwise offload", flags: ["--component-residency transformer=layerwise-offload,text_encoder=layerwise-offload"] },
      ],
    },
    {
      id: "attention", title: "Attention", scope: "serve", default: "platform",
      description: "Different attention kernels can change accumulated denoising results.",
      options: [
        { id: "platform", label: "Platform default", recommended: true },
        { id: "sdpa", label: "Torch SDPA", flags: ["--attention-backend torch_sdpa"] },
      ],
    },
    {
      id: "quality", title: "Quality", scope: "request", default: "lossless",
      description: "Exact preserves reference rounding; lossless can enable additional VAE fast paths.",
      options: [{ id: "lossless", label: "Lossless" }, { id: "exact", label: "Exact" }],
    },
    {
      id: "guidance", title: "Guidance recipe", scope: "request", default: "recommended",
      description: "Skipping changes the guidance algorithm, not just execution speed.",
      options: [
        { id: "recommended", label: "Two-pass skip", recommended: true },
        { id: "ordinary", label: "Ordinary CFG" },
        { id: "three", label: "Three-pass SLG" },
      ],
    },
  ],
  commandBuilder: {
    defaultSelection: { hw: "h200", nodes: 1, gpus_per_node: 1, topology_mode: "auto", tp_size: 1, ulysses_degree: 1, ring_degree: 1 },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 4 } },
      verifiedRecipes: [],
      autoTopology: (s) => ({ tp_size: 1, ulysses_degree: Number(s.gpus_per_node), ring_degree: 1 }),
      validateTopology: (s, t) => {
        const errors = [];
        if (![t.tp_size, t.ulysses_degree, t.ring_degree].every((n) => Number.isInteger(n) && n > 0)) errors.push("Parallel degrees must be positive integers.");
        if (t.tp_size !== 1) errors.push("Tensor parallelism is not implemented for this model.");
        if (Number(s.nodes) !== 1 || Number(s.gpus_per_node) !== t.tp_size * t.ulysses_degree * t.ring_degree) errors.push("GPU count must match TP times Ulysses times Ring on one node.");
        if (24 % t.ulysses_degree) errors.push("Ulysses degree must divide the 24 attention heads.");
        if (t.ring_degree > 1 && s.attention === "sdpa") errors.push("Ring requires an LSE-capable backend, not Torch SDPA.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
        : resource.autoTopology(s);
      const errors = resource.validateTopology(s, topology);
      const flags = ["--model-path {{MODEL_NAME}}"];
      if (Number(s.gpus_per_node) > 1) flags.push(`--num-gpus ${s.gpus_per_node}`, "--tp-size 1", `--ulysses-degree ${topology.ulysses_degree}`, `--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      return {
        match: { hw: s.hw }, nnodes: 1, verified: false, flags,
        builder: {
          topology, errors,
          topologySummary: `TP ${topology.tp_size}, Ulysses ${topology.ulysses_degree}, Ring ${topology.ring_degree}`,
          warnings: ["This exact hardware and workload recipe is not end-to-end verified."],
          verification: { serve: "unverified", request: "unverified" },
        },
      };
    },
  },
  modelNames: { default: "OpenMOSS-Team/MOSS-Image-Base" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: (s) => {
    const guidance = s.guidance === "ordinary" ? { uncond_skip_layers: [] }
      : s.guidance === "three" ? { guidance_scale: 5, uncond_skip_layers: [], slg_layers: [22], slg_scale: 3 } : {};
    return `curl --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt: "A quiet coastal town at sunrise, cinematic photography", size: "1024x1024", seed: 42, quality: s.quality, response_format: "b64_json", output_format: "png", ...guidance }, null, 2)}'`;
  },
  cells: [],
};
