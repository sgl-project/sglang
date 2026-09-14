export const config = {
  modelName: "Qwen-Image 2.1",
  supportedHardware: ["b200"],
  groupHardware: false,
  matchDims: [],

  overlayDims: [
    {
      id: "weights",
      title: "Checkpoint weights",
      scope: "base",
      description: "Choose the checkpoint partition required by this request mode.",
      default: "default",
      options: [{ id: "default", label: "Default", flags: [] }],
    },
    {
      id: "mode",
      title: "Request mode",
      scope: "base",
      default: "text",
      options: [{ id: "text", label: "Text" }],
    },
    {
      id: "placement",
      title: "Placement",
      scope: "serve",
      description: "Keep weights resident unless a verified capacity path requires sharding or offload.",
      default: "resident",
      options: [
        { id: "resident", label: "Resident", flags: ["--performance-mode speed"], recommended: true },
        { id: "offload", label: "Layerwise offload", flags: ["--dit-layerwise-offload true"], soft: true, softReason: "Full-checkpoint validation is pending." },
      ],
    },
    {
      id: "attention",
      title: "Attention",
      scope: "serve",
      description: "Use the platform default unless another backend was measured end to end.",
      default: "platform",
      options: [{ id: "platform", label: "Platform default", recommended: true }],
    },
    {
      id: "precision",
      title: "Precision",
      scope: "serve",
      default: "native",
      options: [{ id: "native", label: "Native mixed precision", recommended: true }],
    },
    {
      id: "encoder",
      title: "Encoder",
      scope: "serve",
      default: "auto",
      options: [{ id: "auto", label: "Auto", flags: ["--encoder-parallel auto"], recommended: true }],
    },
    {
      id: "execution",
      title: "Execution",
      scope: "serve",
      default: "eager",
      options: [{ id: "eager", label: "Eager", recommended: true }],
    },
    {
      id: "quality",
      title: "Quality",
      scope: "request",
      default: "lossless",
      options: [{ id: "lossless", label: "Lossless", recommended: true }],
    },
    {
      id: "outputs",
      title: "Outputs",
      scope: "request",
      kind: "number",
      min: 1,
      max: 10,
      unit: "outputs per prompt",
      default: 1,
      options: [],
    },
  ],

  commandBuilder: {
    defaultSelection: {
      hw: "b200",
      nodes: 1,
      gpus_per_node: 1,
      topology_mode: "auto",
      tp_size: 1,
      ulysses_degree: 1,
      ring_degree: 1,
    },
    resource: {
      limits: {
        nodes: { min: 1, max: 1 },
        gpus_per_node: { min: 1, max: 2 },
      },
      verifiedRecipes: [],
      autoTopology: (s) => ({
        tp_size: 1,
        ulysses_degree: Number(s.gpus_per_node),
        ring_degree: Number(s.nodes),
      }),
      validateTopology: (s, topology) => {
        const world = Number(s.nodes) * Number(s.gpus_per_node);
        if (32 % (Number(topology.tp_size) * Number(topology.ulysses_degree)) !== 0) return ["32 attention heads must be divisible by TP × Ulysses."];
        const product = Number(topology.tp_size) * Number(topology.ulysses_degree) * Number(topology.ring_degree);
        return world === product ? [] : [`World size ${world} must equal TP × Ulysses × Ring (${product}).`];
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
        && entry.placement === s.placement
        && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree
        && entry.ring_degree === topology.ring_degree);
      const world = Number(s.nodes) * Number(s.gpus_per_node);
      const flags = ["--model-path {{MODEL_NAME}}", "--model-id Qwen-Image-2.1", `--num-gpus ${world}`];
      if (topology.tp_size > 1) flags.push(`--tp-size ${topology.tp_size}`);
      flags.push(`--ulysses-degree ${topology.ulysses_degree}`);
      if (topology.ring_degree > 1) flags.push(`--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      const verified = !!recipe && errors.length === 0;
      return {
        match: { hw: s.hw },
        nnodes: Number(s.nodes),
        verified,
        flags,
        builder: {
          topology,
          topologySummary: `TP ${topology.tp_size} · Ulysses ${topology.ulysses_degree} · Ring ${topology.ring_degree}`,
          errors,
          warnings: verified ? [] : ["Valid custom topology; exact end-to-end verification is pending."],
          verification: { serve: verified ? "verified" : "unverified", request: verified ? "verified" : "unverified" },
        },
      };
    },
  },

  modelNames: { default: "/models/qwen-image-2.1" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30010" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30010" },
  },
  curl: (s) => `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ model: "{{MODEL_NAME}}", prompt: "A capybara reading a book by candlelight", n: Number(s.outputs), size: "1024x1024", num_inference_steps: 40, guidance_scale: 1, enable_cache_dit: false }, null, 2)}'`,
  cells: [],
};
