export const config = {
  modelName: "Qwen-Image 2.1",
  supportedHardware: ["h200"],
  groupHardware: false,
  matchDims: [],

  overlayDims: [
    {
      id: "weights",
      title: "Checkpoint weights",
      scope: "base",
      description: "One checkpoint serves generation and editing. Set its authorized local path under Variables.",
      default: "default",
      options: [{ id: "default", label: "Qwen-Image 2.1", flags: [] }],
    },
    {
      id: "mode",
      title: "Request mode",
      scope: "base",
      description: "Switch between JSON generation and PNG uploads to the image-edit endpoint.",
      default: "text",
      options: [
        { id: "text", label: "Text to image" },
        { id: "edit", label: "Image edit", description: "Upload one reference PNG, preserving its alpha channel." },
        { id: "multi", label: "Multi-image edit", description: "Upload two ordered references; Picture 1 and Picture 2 follow this order." },
      ],
    },
    {
      id: "placement",
      title: "Placement",
      scope: "serve",
      description: "Keep weights resident on H200, or stream DiT layers to reduce device memory.",
      learnMore: "#5-runtime-features",
      default: "resident",
      options: [
        {
          id: "resident", label: "Resident", recommended: true,
          flags: (s) => [Number(s.gpus_per_node) === 1 ? "--performance-mode speed" : "--performance-mode manual"],
          description: "Single-H200 serving is verified. Custom topologies keep the selected placement explicit.",
        },
        {
          id: "offload", label: "Layerwise offload",
          flags: ["--performance-mode manual", "--dit-layerwise-offload true"],
          soft: true, softReason: "CLI offload passed; this server recipe has not been verified.",
          description: "Trades host-to-device transfers for lower DiT residency; requires sufficient host RAM.",
        },
      ],
    },
    {
      id: "attention",
      title: "Attention",
      scope: "serve",
      description: "Choose the target-image attention kernel. Text attention retains its causal mask.",
      learnMore: "#5-runtime-features",
      default: "platform",
      options: [
        { id: "platform", label: "Automatic", recommended: true, description: "FlashAttention on the verified H200 server." },
        { id: "fa", label: "FlashAttention", flags: ["--attention-backend fa"], description: "Select the H200 default explicitly." },
        {
          id: "sdpa", label: "Torch SDPA", flags: ["--attention-backend torch_sdpa"],
          soft: true, softReason: "Full generation/edit precision comparisons used CLI SDPA; this server variant is unverified.",
          description: "Use for reference comparisons. Floating-point reduction order can differ from FlashAttention.",
        },
        {
          id: "sage", label: "SageAttention", flags: ["--attention-backend sage_attn"],
          soft: true, softReason: "CLI smoke test passed; image and alpha quality need workload-specific validation.",
          description: "Approximate attention; requires the SageAttention dependency.",
        },
      ],
    },
    {
      id: "precision",
      title: "Precision",
      scope: "serve",
      description: "Use the checkpoint's native BF16 / FP32 computation. Quantized variants are not validated.",
      default: "native",
      options: [{ id: "native", label: "Native BF16 / FP32", recommended: true }],
    },
    {
      id: "encoder",
      title: "Encoder",
      scope: "serve",
      description: "Schedule Qwen3-VL independently of target-image attention.",
      learnMore: "#5-runtime-features",
      default: "auto",
      options: [
        { id: "auto", label: "Auto", flags: ["--encoder-parallel auto"], recommended: true },
        { id: "replicate", label: "Replicate", flags: ["--encoder-parallel replicate"], soft: true, softReason: "Explicit replication has not been verified for this server recipe." },
        { id: "fold", label: "Fold", flags: ["--encoder-parallel fold"], soft: true, softReason: "Requires node-local P2P access; this explicit server setting is unverified." },
      ],
    },
    {
      id: "vae",
      title: "VAE decoding",
      scope: "serve",
      description: "Decode RGBA in full, in tiles, or with tiles distributed across GPUs.",
      learnMore: "#5-runtime-features",
      default: "full",
      options: [
        { id: "full", label: "Full image", recommended: true, description: "Default for generation and condition-image encoding." },
        { id: "tiled", label: "Tiled", flags: ["--vae-tiling true"], soft: true, softReason: "Repeated 512px HTTP edits passed; other tiled workloads remain unverified.", description: "Reduces activation memory; can change pixels near tile boundaries." },
        {
          id: "parallel", label: "Parallel tiles", flags: ["--vae-tiling true", "--vae-sp true"],
          disabled: (s) => Number(s.gpus_per_node) < 2,
          disableReason: "Select two GPUs before distributing VAE tiles.",
          soft: true, softReason: "Two-H200 CLI decoding passed; this HTTP recipe is unverified.",
        },
      ],
    },
    {
      id: "execution",
      title: "Execution",
      scope: "serve",
      description: "Graph replay requires matching resolution and condition-prefix length.",
      learnMore: "#5-runtime-features",
      default: "eager",
      options: [
        { id: "eager", label: "Eager", recommended: true },
        {
          id: "bcg", label: "Breakable CUDA Graph",
          flags: ["--enable-breakable-cuda-graph true", "--warmup-resolutions 512x512", "--bcg-text-buckets 64"],
          soft: true, softReason: "Only a matching 512px CLI warmup was verified. Other prompts or image prefixes can fall back to eager.",
          description: "Captures a 512px warmup. Text buckets do not pad condition KV; this is not a guaranteed replay recipe.",
        },
      ],
    },
    {
      id: "background",
      title: "Background",
      scope: "request",
      description: "Both choices save PNG. Transparency is requested in the prompt, not imposed by postprocessing.",
      learnMore: "#transparent-png-output",
      default: "scene",
      options: [
        { id: "scene", label: "Scene", recommended: true },
        { id: "transparent", label: "Transparent / alpha", description: "Generate an isolated subject, or preserve the reference image's transparent background." },
      ],
    },
    {
      id: "resolution",
      title: "Resolution",
      scope: "request",
      description: "Square output canvas; reference images keep their own aspect ratios.",
      default: "1024",
      options: [{ id: "512", label: "512 × 512" }, { id: "1024", label: "1024 × 1024", recommended: true }],
    },
    {
      id: "steps",
      title: "Denoising steps",
      scope: "request",
      description: "40 is the checkpoint default. Fewer steps trade detail for latency.",
      kind: "number", min: 1, max: 100, unit: "steps", default: 40, options: [],
    },
    {
      id: "outputs",
      title: "Outputs",
      scope: "request",
      description: "Generate independent images for the same prompt.",
      kind: "number", min: 1, max: 10, unit: "outputs per prompt", default: 1, options: [],
    },
  ],

  commandBuilder: {
    defaultSelection: {
      hw: "h200", nodes: 1, gpus_per_node: 1, topology_mode: "auto",
      tp_size: 1, ulysses_degree: 1, ring_degree: 1,
    },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 2 } },
      verifiedRecipes: [
        { id: "h200-1-resident", hw: "h200", nodes: 1, gpus_per_node: 1, placement: "resident", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
      ],
      autoTopology: (s) => ({ tp_size: 1, ulysses_degree: Number(s.gpus_per_node), ring_degree: 1 }),
      validateTopology: (s, topology) => {
        const errors = [];
        const nodes = Number(s.nodes);
        const perNode = Number(s.gpus_per_node);
        const { tp_size: tp, ulysses_degree: ulysses, ring_degree: ring } = topology;
        if (nodes !== 1) errors.push("This picker covers single-node deployment only.");
        if (![1, 2].includes(perNode)) errors.push("Select one or two GPUs per node.");
        if (![tp, ulysses, ring].every((n) => [1, 2].includes(n))) errors.push("TP, Ulysses and Ring must each be 1 or 2.");
        if (nodes * perNode !== tp * ulysses * ring) errors.push(`World size ${nodes * perNode} must equal TP × Ulysses × Ring (${tp * ulysses * ring}).`);
        if (32 % (tp * ulysses) !== 0) errors.push("32 attention heads must be divisible by TP × Ulysses.");
        if (ring > 1 && s.attention === "sdpa") errors.push("Ring requires FlashAttention or SageAttention; Torch SDPA is unsupported.");
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
        && entry.nodes === Number(s.nodes) && entry.gpus_per_node === Number(s.gpus_per_node)
        && entry.placement === s.placement && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree && entry.ring_degree === topology.ring_degree);
      const serveVerified = !!recipe && errors.length === 0 && s.encoder === "auto"
        && ["platform", "fa"].includes(s.attention) && s.precision === "native"
        && s.execution === "eager" && s.vae === "full";
      // Exact HTTP workloads from the validation matrix, not blanket quality coverage.
      const requestVerified = serveVerified
        && ((["text", "edit"].includes(s.mode) && s.resolution === "1024" && Number(s.steps) === 40 && Number(s.outputs) === 1)
          || (s.background === "scene" && s.mode === "text" && s.resolution === "512" && Number(s.steps) === 4 && Number(s.outputs) === 2)
          || (s.background === "scene" && s.mode === "multi" && s.resolution === "512" && Number(s.steps) === 4 && Number(s.outputs) === 1));
      const world = Number(s.nodes) * Number(s.gpus_per_node);
      const flags = ['--model-path "{{MODEL_PATH}}"', "--model-id Qwen-Image-2.1", `--num-gpus ${world}`];
      if (topology.tp_size > 1) flags.push(`--tp-size ${topology.tp_size}`);
      flags.push(`--ulysses-degree ${topology.ulysses_degree}`);
      if (topology.ring_degree > 1) flags.push(`--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
      const warnings = [];
      if (!serveVerified && !errors.length) warnings.push("This server combination has not completed an exact HTTP verification run.");
      if (!requestVerified && !errors.length) warnings.push("This request shape is outside the verified HTTP matrix.");
      return {
        match: { hw: s.hw }, nnodes: Number(s.nodes), verified: serveVerified, flags,
        builder: {
          topology,
          topologySummary: `TP ${topology.tp_size} · Ulysses ${topology.ulysses_degree} · Ring ${topology.ring_degree}`,
          errors, warnings,
          verification: {
            serve: errors.length ? "error" : serveVerified ? "verified" : "unverified",
            request: errors.length ? "error" : requestVerified ? "verified" : "unverified",
          },
          resolvedSettings: {
            attention: s.attention === "platform" ? "FlashAttention (auto)" : undefined,
            encoder: s.encoder === "auto" && world === 1 ? "Single GPU (auto)" : undefined,
          },
        },
      };
    },
  },

  modelNames: { default: "Qwen-Image-2.1" },
  placeholders: {
    MODEL_PATH: { target: "command", label: "Authorized checkpoint directory", default: "/models/qwen-image-2.1" },
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30010" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30010" },
    INPUT_IMAGE: { target: "curl", label: "First reference PNG (client path)", default: "/path/to/input.png" },
    SECOND_IMAGE: { target: "curl", label: "Second reference PNG (client path)", default: "/path/to/reference.png" },
  },
  curl: (s) => {
    const transparent = s.background === "transparent";
    const prompts = {
      text: transparent
        ? "A single fluffy orange cat sitting, full body, isolated on a transparent background. A clean cutout with an alpha channel, transparent outside the cat, no floor, no shadow, no background."
        : "A capybara reading a book by candlelight",
      edit: transparent
        ? "Change the orange fur of the cat to gray, keeping its pose, shape and fur detail unchanged. Preserve the transparent background and alpha channel. No floor, no shadow, no background."
        : "Change the red teapot to blue, keeping its shape, table, window, and lighting unchanged.",
      multi: transparent
        ? "Combine the subjects from Picture 1 and Picture 2 into one composition on a transparent background. Preserve an alpha channel outside the subjects."
        : "Combine the subjects from Picture 1 and Picture 2 into one coherent scene, preserving their appearance.",
    };
    const request = {
      model: "{{MODEL_NAME}}", prompt: prompts[s.mode], n: Number(s.outputs),
      size: `${s.resolution}x${s.resolution}`, num_inference_steps: Number(s.steps),
      guidance_scale: 1, seed: 42, generator_device: "cpu",
      output_format: "png", response_format: "b64_json",
      background: transparent ? "transparent" : "auto",
    };
    if (s.mode === "text") {
      return `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify({ ...request, enable_cache_dit: false }, null, 2)}'`;
    }
    const fields = Object.entries(request).map(([key, value]) => `  --form-string '${key}=${value}'`);
    fields.push('  -F "image[]=@{{INPUT_IMAGE}};type=image/png"');
    if (s.mode === "multi") fields.push('  -F "image[]=@{{SECOND_IMAGE}};type=image/png"');
    return `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits \\
${fields.join(" \\\n")}`;
  },
  // The integration is installed from source; no published Docker image is verified.
  runModes: () => ["python"],
  showPlaygroundLink: false,
  cells: [],
};
