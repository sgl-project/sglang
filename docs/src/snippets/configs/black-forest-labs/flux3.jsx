// Image-only FLUX.3 diffusion command builder. Consumed by _deployment.jsx.

export const config = (() => {
const modelOf = (s) => s.variant === "distilled"
  ? "black-forest-labs/flux-3-image-klein"
  : "black-forest-labs/flux-3-image-klein-base";
const hardware = [
  { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "consumer" },
  { id: "rtx6000pro", label: "RTX PRO 6000 Blackwell", vram: "96GB", vendor: "consumer" },
];
const hardwareLabels = { rtx5090: "RTX 5090", rtx6000pro: "RTX PRO 6000", h100: "H100", h200: "H200", b200: "B200", b300: "B300", gb200: "GB200", gb300: "GB300" };

return {
  modelName: "FLUX.3",
  supportedHardware: Object.keys(hardwareLabels),
  hardware,
  groupHardware: true,
  matchDims: [],
  overlayDims: [
    // Request-scope rows only shape the cURL payload: `sglang serve` rejects
    // sampling flags, and one server serves both modes.
    {
      id: "mode",
      title: "Request mode",
      scope: "request",
      description: "One server serves both; a request with a reference image is an edit.",
      default: "t2i",
      options: [
        { id: "t2i", label: "Text to image", flags: [] },
        { id: "i2i", label: "Image to image", flags: [] },
      ],
    },
    {
      id: "variant",
      title: "Model",
      scope: "serve",
      description:
        "Base and distilled are self-contained repositories. Base uses CFG; distilled is a flow map without CFG.",
      default: "base",
      options: [
        {
          id: "distilled",
          label: "Distilled",
          subtitle: "16-step flow map, no CFG",
          flags: [],
        },
        { id: "base", label: "Base", subtitle: "50-step CFG", flags: [] },
      ],
    },
    {
      id: "quant",
      title: "Precision",
      scope: "serve",
      description:
        "FP8R and NVFP4 load a quantized DiT from variants/. Text encoder and VAE precision follow the package configuration.",
      default: "bf16",
      options: [
        { id: "bf16", label: "BF16", subtitle: "Reference", flags: [] },
        {
          id: "fp8r",
          label: "FP8R",
          subtitle: "Rowwise FP8 DiT",
          flags: [],
        },
        {
          id: "nvfp4",
          label: "NVFP4",
          subtitle: "Blackwell, FP4 DiT",
          flags: [],
          disabled: (s) => ["h100", "h200"].includes(s.hw),
          disableReason: "NVFP4 GEMMs need a Blackwell GPU.",
          soft: true,
          softReason: "Verify quantized-output fidelity for your workload.",
        },
      ],
    },
    {
      id: "placement",
      title: "Placement",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/api/cli#component-residency",
      description: "A 1 MP, batch-1 workload fits resident on RTX 5090. Larger references or batches need additional headroom; use auto or offload when necessary.",
      default: "auto",
      options: [
        { id: "auto", label: "Auto", recommended: true, flags: [] },
        {
          id: "offload",
          label: "CPU offload",
          flags: [
            "--component-residency transformer=layerwise-offload,text_encoder=component-offload,vae=component-offload",
          ],
          recommendedWhen: (s) => s.hw === "rtx5090" && Number(s.batch_size) > 1,
        },
        {
          id: "resident",
          label: "GPU resident",
          flags: ["--component-residency all=resident"],
          recommendedWhen: (s) => s.hw !== "rtx5090" || Number(s.batch_size) === 1,
        },
      ],
    },
    {
      id: "attention", title: "Attention", scope: "serve", default: "auto",
      description: "Keep the backend fixed for numerical comparisons.",
      options: [
        { id: "auto", label: "Platform default", flags: [], recommended: true },
        { id: "sdpa", label: "Torch SDPA", flags: ["--attention-backend torch_sdpa"] },
        { id: "fa", label: "FlashAttention", flags: ["--attention-backend fa"] },
      ],
    },
    {
      id: "execution", title: "Execution", scope: "serve", default: "eager",
      description: "CUDA graphs can reduce launch overhead but reserve additional memory.",
      options: [
        { id: "eager", label: "Eager", flags: [], recommended: true },
        { id: "bcg", label: "Breakable CUDA graphs", flags: ["--enable-breakable-cuda-graph"], soft: true, softReason: "Not verified for every hardware and topology." },
      ],
    },
    {
      id: "batch_size", title: "Max batch size", scope: "serve", kind: "number",
      min: 1, max: 8, default: 1, unit: "requests", options: [],
      description: "Larger batches trade activation memory and request latency for throughput.",
    },
    {
      id: "batch_delay", title: "Batch window", scope: "serve", kind: "number",
      min: 0, max: 100, default: 5, unit: "ms", options: [],
      showWhen: (s) => Number(s.batch_size) > 1,
    },
    {
      id: "quality", title: "Quality", scope: "request", default: "lossless",
      description: "Quantization and Cache-DiT remain separate, potentially lossy choices.",
      options: [{ id: "lossless", label: "Lossless" }, { id: "exact", label: "Exact" }],
    },
    {
      id: "resolution",
      title: "Resolution",
      scope: "request",
      // With a reference the output takes the reference's size.
      showWhen: (s) => s.mode !== "i2i",
      default: "1024",
      options: [
        { id: "1024", label: "1024×1024", flags: [], recommended: true },
      ],
    },
    {
      id: "schedule",
      title: "Sampling",
      scope: "request",
      // The distilled flow map takes its 16 steps from the package and has no CFG.
      showWhen: (s) => s.variant !== "distilled",
      default: "quality",
      options: [
        { id: "quality", label: "Quality · 50-step, CFG 3.5", flags: [], recommended: true },
      ],
    },
    {
      id: "outputs",
      title: "Outputs",
      scope: "request",
      kind: "number",
      min: 1,
      max: 4,
      unit: "outputs per prompt",
      default: 1,
      options: [],
    },
  ],

  commandBuilder: {
    defaultSelection: {
      hw: "rtx5090",
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
        gpus_per_node: { min: 1, max: 8 },
      },
      verifiedRecipes: [{
        id: "gb300-base-bf16", hw: "gb300", nodes: 1,
        gpus_per_node: 1, tp_size: 1, ulysses_degree: 1, ring_degree: 1,
        variant: "base", quant: "bf16", placement: "resident", attention: "fa",
        execution: "eager", quality: "exact", batch_size: 1, outputs: 1,
        default: true,
      }],
      autoTopology: (s) => ({
        tp_size: Number(s.gpus_per_node) || 1,
        ulysses_degree: 1,
        ring_degree: 1,
      }),
      validateTopology: (s, t) => {
        const errors = [];
        const gpus = Number(s.gpus_per_node);
        if (Number(s.nodes) !== 1 || !Number.isInteger(gpus) || gpus < 1 || gpus > 8) {
          errors.push("Choose one node with one to eight GPUs.");
        }
        if (![t.tp_size, t.ulysses_degree, t.ring_degree].every((n) => Number.isInteger(n) && n > 0)) {
          errors.push("Parallel degrees must be positive integers.");
        }
        if (24 % Number(t.tp_size) || (24 / Number(t.tp_size)) % Number(t.ulysses_degree)) {
          errors.push("TP and Ulysses must partition the 24 attention heads evenly.");
        }
        if (Number(t.ring_degree) !== 1) {
          errors.push("FLUX.3 sequence parallel uses Ulysses only (ring degree 1).");
        }
        if (Number(t.tp_size) * Number(t.ulysses_degree) !== gpus) {
          errors.push("TP size x Ulysses degree must equal the GPU count.");
        }
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
      const gpus = Number(s.gpus_per_node) || 1;
      const parallel = [];
      if (gpus > 1) parallel.push(`--num-gpus ${gpus}`, `--tp-size ${topology.tp_size}`);
      if (topology.ulysses_degree > 1) parallel.push(`--ulysses-degree ${topology.ulysses_degree}`);
      if (s.quant !== "bf16") parallel.push(`--model-variant ${s.quant}`);
      if (Number(s.batch_size) > 1) {
        parallel.push(`--batching-max-size ${Number(s.batch_size)}`, `--batching-delay-ms ${Number(s.batch_delay) || 0}`);
      }
      const testedProfile = (["gb300", "h200"].includes(s.hw) && s.attention === "fa"
          && ["base", "distilled"].includes(s.variant)
          && (s.quant === "bf16" || s.quality === "exact")
          && (s.hw === "gb300" || s.quant !== "nvfp4"))
        || (s.hw === "rtx5090" && s.attention === "sdpa"
          && s.variant === "base" && s.quant === "bf16");
      const verified = testedProfile && gpus === 1
        && s.placement === "resident"
        && s.execution === "eager" && ["exact", "lossless"].includes(s.quality)
        && Number(s.batch_size) === 1 && Number(s.outputs) === 1;
      const warnings = verified ? [] : ["This exact hardware, checkpoint and topology combination is not yet end-to-end verified."];
      if (s.hw === "rtx5090" && s.placement === "resident" && Number(s.batch_size) > 1) {
        warnings.push("Full-resolution batching needs additional activation memory. Use offload or batch 1 if the resident batch does not fit.");
      }
      return {
        match: { hw: s.hw },
        nnodes: 1,
        verified: verified && !errors.length,
        verificationStatus: errors.length ? "error" : verified ? "verified" : "unverified",
        env: [],
        flags: [
          `--model-path ${modelOf(s)}`,
          ...parallel,
          "--host {{HOST_IP}}",
          "--port {{PORT}}",
        ],
        builder: {
          topology,
          topologySummary: `${Number(s.gpus_per_node) || 1}× ${hardwareLabels[s.hw] || s.hw}`,
          errors,
          warnings,
          verification: {
            serve: errors.length ? "error" : verified ? "verified" : "unverified",
            request: errors.length ? "error" : verified ? "verified" : "unverified",
          },
          resolvedSettings: {
            placement: s.placement === "resident" ? "GPU resident" : s.placement === "offload" ? "Layerwise offload" : "Auto",
          },
        },
      };
    },
  },

  modelNames: {
    ...Object.fromEntries(["base", "distilled"].flatMap((variant) =>
      ["bf16", "fp8r", "nvfp4"].map((quant) => [`${variant}|${quant}`, modelOf({ variant })]))),
    default: modelOf({ variant: "base" }),
  },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30010" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30010" },
    INPUT_IMAGE: {
      target: "curl",
      label: "I2I input image",
      default: "/data/flux3/reference.png",
    },
  },

  curl: (s) => {
    const common = {
      prompt:
        s.mode === "i2i"
          ? "Render the reference as a winter wildlife photograph"
          : "A red fox standing in fresh snow",
      // With a reference the output takes the reference's size.
      ...(s.mode === "i2i" ? {} : { size: "1024x1024" }),
      ...(s.variant === "base" ? { num_inference_steps: 50, guidance_scale: 3.5 } : {}),
      seed: 0,
      quality: s.quality,
      n: Number(s.outputs) || 1,
      response_format: "b64_json",
    };
    if (s.mode === "i2i") {
      // The edit endpoint takes the reference upload; a reference selects i2i.
      const fields = Object.entries(common)
        .map(([key, value]) => `  -F "${key}=${value}" \\`)
        .join("\n");
      return `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits \\
  -F "image=@{{INPUT_IMAGE}}" \\
${fields}
  -o flux3_i2i.json`;
    }
    const request = common;
    return `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify(request, null, 2)}' \\
  -o flux3_t2i.json`;
  },

  runModes: () => ["python"],
  showPlaygroundLink: false,
  cells: [],
};
})();
