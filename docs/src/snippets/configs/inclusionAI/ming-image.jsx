export const config = {
  modelName: "Ming-Image",
  supportedHardware: ["h200"],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "mode",
      title: "Checkpoint and request",
      scope: "base",
      default: "text",
      description:
        "Design serves generation and editing. Decomposition uses the separate Design-Layer checkpoint.",
      options: [
        { id: "text", label: "Design: generation" },
        { id: "edit", label: "Design: edit" },
        { id: "layer", label: "Design-Layer: decompose" },
      ],
    },
    {
      id: "placement",
      title: "Placement",
      scope: "serve",
      default: "resident",
      description:
        "Resident weights avoid transfers. Offload requires additional host RAM.",
      options: [
        {
          id: "resident",
          label: "Resident",
          recommended: true,
          flags: ["--performance-mode speed"],
        },
        {
          id: "offload",
          label: "Encoder layerwise",
          flags: [
            "--performance-mode manual",
            "--component-residency text_encoder=layerwise-offload",
          ],
          soft: true,
          softReason: "Validate memory and latency for the selected workload.",
        },
      ],
    },
    {
      id: "attention",
      title: "Attention",
      scope: "serve",
      default: "platform",
      description:
        "Exact attention is the baseline; changing kernels can change floating-point rounding.",
      options: [
        { id: "platform", label: "Automatic", recommended: true },
        {
          id: "sdpa",
          label: "Torch SDPA",
          flags: ["--attention-backend torch_sdpa"],
        },
      ],
    },
    {
      id: "resolution",
      title: "Resolution",
      scope: "request",
      default: "1024",
      description:
        "Reference images use the closest official aspect bucket. 2048 applies to generation only.",
      options: [
        { id: "512", label: "512" },
        { id: "1024", label: "1024", recommended: true },
        {
          id: "2048",
          label: "2048",
          disabled: (s) => s.mode !== "text",
          disableReason:
            "Reference-image requests use the 512 or 1024 bucket families.",
        },
      ],
    },
    {
      id: "layers",
      title: "Layers",
      scope: "request",
      kind: "number",
      min: 1,
      max: 8,
      default: 4,
      description:
        "Number of returned layers for Design-Layer; ignored by Design requests.",
      options: [],
    },
  ],
  commandBuilder: {
    defaultSelection: {
      hw: "h200",
      nodes: 1,
      gpus_per_node: 1,
      topology_mode: "auto",
      tp_size: 1,
      ulysses_degree: 1,
      ring_degree: 1,
    },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 2 } },
      verifiedRecipes: [],
      autoTopology: (s) => ({
        tp_size: 1,
        ulysses_degree: Number(s.gpus_per_node),
        ring_degree: 1,
      }),
      validateTopology: (s, topology) => {
        const errors = [];
        const { tp_size: tp, ulysses_degree: sp, ring_degree: ring } = topology;
        if (Number(s.nodes) !== 1)
          errors.push("This picker covers a single node.");
        if (![tp, sp, ring].every((n) => Number.isInteger(n) && n >= 1))
          errors.push("Parallel degrees must be positive integers.");
        if (Number(s.nodes) * Number(s.gpus_per_node) !== tp * sp * ring)
          errors.push("GPU count must equal TP times Ulysses times Ring.");
        if (30 % (tp * sp))
          errors.push("TP times Ulysses must divide the 30 DiT heads.");
        if (tp > 2)
          errors.push(
            "This recipe covers encoder tensor parallelism up to two GPUs.",
          );
        if (ring > 1 && s.attention === "sdpa")
          errors.push("Ring requires FlashAttention.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology =
        s.topology_mode === "manual"
          ? {
              tp_size: Number(s.tp_size),
              ulysses_degree: Number(s.ulysses_degree),
              ring_degree: Number(s.ring_degree),
            }
          : resource.autoTopology(s);
      const errors = resource.validateTopology(s, topology);
      const world = Number(s.nodes) * Number(s.gpus_per_node);
      const verifiedServe =
        s.hw === "h200" &&
        world === 1 &&
        s.placement === "resident" &&
        s.attention === "platform";
      const verifiedRequest =
        verifiedServe &&
        s.resolution === "512" &&
        (s.mode !== "layer" || Number(s.layers) === 2);
      const model =
        s.mode === "layer"
          ? "inclusionAI/Ming-Image-0.1-Design-Layer"
          : "inclusionAI/Ming-Image-0.1-Design";
      const flags = [`--model-path ${model}`];
      if (world > 1)
        flags.push(
          `--num-gpus ${world}`,
          `--tp-size ${topology.tp_size}`,
          `--ulysses-degree ${topology.ulysses_degree}`,
        );
      if (topology.ring_degree > 1)
        flags.push(`--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      return {
        match: { hw: s.hw },
        nnodes: Number(s.nodes),
        verified: verifiedRequest,
        flags,
        builder: {
          topology,
          topologySummary: `TP ${topology.tp_size}, Ulysses ${topology.ulysses_degree}, Ring ${topology.ring_degree}`,
          errors,
          warnings: verifiedRequest
            ? []
            : ["This exact HTTP recipe has not completed verification."],
          verification: {
            serve: errors.length
              ? "error"
              : verifiedServe
                ? "verified"
                : "unverified",
            request: errors.length
              ? "error"
              : verifiedRequest
                ? "verified"
                : "unverified",
          },
        },
      };
    },
  },
  modelNames: { default: "inclusionAI/Ming-Image-0.1-Design" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    INPUT_IMAGE: {
      target: "curl",
      label: "Reference PNG on the client",
      default: "/path/to/input.png",
    },
  },
  curl: (s) => {
    const body = {
      prompt:
        s.mode === "layer"
          ? `Decompose this image into ${Number(s.layers)} layers.`
          : s.mode === "edit"
            ? "Change the teapot to blue, preserving its shape."
            : "RGBA, 4-channel, transparent background. A red enamel teapot, product photography.",
      size: `${s.resolution}x${s.resolution}`,
      output_format: "png",
      response_format: "b64_json",
    };
    if (s.mode === "text")
      return `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify(body, null, 2)}'`;
    if (s.mode === "layer") body.num_layers = Number(s.layers);
    const fields = Object.entries(body).map(
      ([key, value]) => `  --form-string '${key}=${value}'`,
    );
    fields.push('  -F "image=@{{INPUT_IMAGE}};type=image/png"');
    return `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits \\
${fields.join(" \\\n")}`;
  },
  runModes: () => ["python"],
  showPlaygroundLink: false,
  cells: [],
};
