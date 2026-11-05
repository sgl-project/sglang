export const config = {
  modelName: "HiDream-O1-Image",
  supportedHardware: ["h200"],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "steps", title: "Recipe", scope: "request", default: "full",
      description: "The reference full recipe uses 50 steps; fewer steps trade quality for latency.",
      options: [
        { id: "full", label: "Full (50 steps)", recommended: true },
        { id: "fast", label: "28 steps" },
      ],
    },
    {
      id: "guidance", title: "Guidance", scope: "request", default: "cfg",
      description: "CFG runs a second forward pass per step.",
      options: [
        { id: "cfg", label: "CFG 5.0", recommended: true },
        { id: "off", label: "Off (1.0)" },
      ],
    },
  ],
  commandBuilder: {
    defaultSelection: {
      hw: "h200", nodes: 1, gpus_per_node: 1, topology_mode: "auto",
      tp_size: 1, ulysses_degree: 1, ring_degree: 1,
    },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
      verifiedRecipes: [
        { id: "h200-tp1", hw: "h200", nodes: 1, gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1, default: true },
        { id: "h200-tp2", hw: "h200", nodes: 1, gpus_per_node: 2, tp_size: 2, ulysses_degree: 1, ring_degree: 1 },
      ],
      autoTopology: (s) => ({ tp_size: Number(s.gpus_per_node), ulysses_degree: 1, ring_degree: 1 }),
      validateTopology: (s, topology) => {
        const errors = [];
        if (![topology.tp_size, topology.ulysses_degree, topology.ring_degree].every((n) => Number.isInteger(n) && n > 0)) {
          errors.push("Parallel degrees must be positive integers.");
        }
        if (Number(s.nodes) !== 1 || topology.tp_size !== Number(s.gpus_per_node)) {
          errors.push("Use a single node with TP equal to the number of GPUs.");
        }
        if (8 % topology.tp_size) errors.push("TP must divide the 8 KV heads.");
        if (topology.ulysses_degree !== 1 || topology.ring_degree !== 1) {
          errors.push("Sequence parallelism is unsupported: the dense hybrid attention mask cannot be sharded.");
        }
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
        : resource.autoTopology(s);
      const errors = resource.validateTopology(s, topology);
      const recipe = resource.verifiedRecipes.find((entry) => entry.hw === s.hw
        && entry.nodes === Number(s.nodes)
        && entry.gpus_per_node === Number(s.gpus_per_node)
        && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree
        && entry.ring_degree === topology.ring_degree);
      const verified = !!recipe && errors.length === 0;
      const flags = ["--model-path {{MODEL_NAME}}", `--num-gpus ${Number(s.gpus_per_node)}`];
      if (topology.tp_size > 1) flags.push(`--tp-size ${topology.tp_size}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      return {
        match: { hw: s.hw }, nnodes: 1, verified, flags,
        builder: {
          topology, errors,
          topologySummary: `TP ${topology.tp_size}; no sequence parallelism`,
          warnings: verified ? [] : ["Valid custom topology; exact end-to-end verification is pending."],
          verification: { serve: verified ? "verified" : "unverified", request: verified ? "verified" : "unverified" },
        },
      };
    },
  },
  modelNames: { default: "HiDream-ai/HiDream-O1-Image" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30010" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30010" },
  },
  curl: (s) => `curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt: "A cat is sitting next to a sign that says HiDream-O1", size: "2048x2048", num_inference_steps: s.steps === "fast" ? 28 : 50, guidance_scale: s.guidance === "off" ? 1.0 : 5.0, seed: 42, response_format: "b64_json" }, null, 2)}'`,
  cells: [],
};
