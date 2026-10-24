export const config = {
  modelName: "HunyuanImage-3.0-Instruct",
  supportedHardware: ["h200"],
  groupHardware: false,
  runModes: ["python"],
  showPlaygroundLink: false,
  matchDims: [],
  overlayDims: [
    {
      id: "weights", title: "Checkpoint", scope: "base", default: "default",
      description: "The Instruct checkpoint; the distilled release is not covered here.",
      options: [{ id: "default", label: "Instruct" }],
    },
    {
      id: "execution", title: "Execution", scope: "serve", default: "eager",
      description: "Start without compilation or approximate caching.",
      options: [{ id: "eager", label: "Eager" }],
    },
    {
      id: "outputs", title: "Outputs", scope: "request", kind: "number",
      description: "Additional outputs increase activation memory.",
      min: 1, max: 4, default: 1, options: [],
    },
  ],
  commandBuilder: {
    defaultSelection: {
      hw: "h200", nodes: 1, gpus_per_node: 2, topology_mode: "auto",
      tp_size: 2, ulysses_degree: 1, ring_degree: 1,
    },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 2, max: 8 } },
      verifiedRecipes: [],
      autoTopology: (s) => ({ tp_size: Number(s.gpus_per_node), ulysses_degree: 1, ring_degree: 1 }),
      validateTopology: (s, topology) => {
        const errors = [];
        if (Number(s.nodes) !== 1 || topology.tp_size !== Number(s.gpus_per_node)) {
          errors.push("Use a single node with TP equal to the number of GPUs.");
        }
        if (![2, 4, 8].includes(topology.tp_size)) errors.push("This recipe supports TP 2, 4, or 8.");
        if (topology.ulysses_degree !== 1 || topology.ring_degree !== 1) {
          errors.push("HunyuanImage-3 does not implement sequence-parallel attention.");
        }
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
        : resource.autoTopology(s);
      return {
        match: { hw: s.hw }, nnodes: 1, verified: false,
        flags: ["--model-path {{MODEL_NAME}}", `--num-gpus ${Number(s.gpus_per_node)}`,
          `--tp-size ${topology.tp_size}`, "--ulysses-degree 1", "--ring-degree 1",
          "--host {{HOST_IP}}", "--port {{PORT}}"],
        builder: {
          topology,
          topologySummary: `TP ${topology.tp_size}; replicated vision encoder and VAE`,
          errors: resource.validateTopology(s, topology),
          warnings: ["Full-checkpoint end-to-end verification of this CUDA recipe is pending."],
          verification: { serve: "unverified", request: "unverified" },
        },
      };
    },
  },
  modelNames: { default: "tencent/HunyuanImage-3.0-Instruct" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30010" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30010" },
  },
  curl: (s) => `curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt: "A blue ceramic teapot on a wooden table, soft daylight.", n: Number(s.outputs), num_inference_steps: 50, guidance_scale: 2.5, seed: 42, width: 1024, height: 1024, response_format: "b64_json" }, null, 2)}'`,
  cells: [],
};
