// Image-only FLUX.3 diffusion command builder. Consumed by _deployment.jsx.

export const config = (() => {
// B200 and H200 come from the shared catalog; only RTX 5090 is added here.
const hardware = [
  { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "consumer" },
];
const hardwareLabels = { b200: "B200", h200: "H200", rtx5090: "RTX 5090" };

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
      id: "model",
      title: "Model",
      scope: "serve",
      description:
        "Distilled (flux-3-image-klein) samples 16 steps without CFG; Base (flux-3-image-klein-base) samples 50 steps with CFG.",
      default: "distilled",
      options: [
        {
          id: "distilled",
          label: "Distilled",
          subtitle: "16-step flow map, no CFG",
          // A separate repo: resolveDeployment picks the model path.
          flags: [],
        },
        { id: "base", label: "Base", subtitle: "50-step CFG", flags: [] },
      ],
    },
    {
      id: "precision",
      title: "Precision",
      scope: "serve",
      description:
        "FP8R and NVFP4 load the quantized DiT from the repo's variants/; the text encoder and VAE stay BF16.",
      default: "bf16",
      options: [
        { id: "bf16", label: "BF16", subtitle: "Reference", flags: [] },
        {
          id: "fp8r",
          label: "FP8R",
          subtitle: "Rowwise FP8 DiT",
          flags: ["--model-variant fp8r"],
        },
        {
          id: "nvfp4",
          label: "NVFP4",
          subtitle: "Blackwell, FP4 DiT",
          flags: ["--model-variant nvfp4"],
          disabled: (s) => s.hw === "h200",
          disableReason: "NVFP4 GEMMs need a Blackwell GPU.",
          soft: (s) => s.hw !== "b200",
          softReason: "NVFP4 is verified on B200 only.",
        },
      ],
    },
    {
      id: "placement",
      title: "Placement",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/api/cli#component-residency",
      default: "offload",
      options: [
        {
          id: "offload",
          label: "CPU offload",
          flags: [
            "--dit-layerwise-offload True",
            "--dit-cpu-offload True",
            "--text-encoder-cpu-offload True",
            "--vae-cpu-offload True",
            "--flux3-dit-keep-non-layer true",
            "--flux3-keep-visual-vae-resident true",
          ],
          recommendedWhen: (s) => s.hw === "rtx5090",
        },
        {
          id: "resident",
          label: "GPU resident",
          flags: [],
          recommendedWhen: (s) => s.hw !== "rtx5090",
        },
      ],
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
      showWhen: (s) => s.model !== "distilled",
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
        gpus_per_node: { min: 1, max: 2 },
      },
      verifiedRecipes: [],
      // Two GPUs default to Ulysses sequence parallel; tensor parallel is the
      // manual alternative. Both are verified on 2x H200 and 2x B200.
      autoTopology: (s) => ({
        tp_size: 1,
        ulysses_degree: Number(s.gpus_per_node) || 1,
        ring_degree: 1,
      }),
      validateTopology: (s, t) => {
        const errors = [];
        const gpus = Number(s.gpus_per_node);
        if (Number(s.nodes) !== 1 || ![1, 2].includes(gpus)) {
          errors.push("The FLUX.3 image release is configured for one node with one or two GPUs.");
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
      if (gpus > 1) parallel.push(`--num-gpus ${gpus}`);
      if (topology.tp_size > 1) parallel.push(`--tp-size ${topology.tp_size}`);
      if (topology.ulysses_degree > 1) parallel.push(`--ulysses-degree ${topology.ulysses_degree}`);
      return {
        match: { hw: s.hw },
        nnodes: 1,
        verified: false,
        verificationStatus: errors.length ? "error" : "unverified",
        env: [],
        flags: [
          `--model-path ${s.model === "base" ? "{{WEIGHTS_ROOT}}" : "{{DISTILLED_ROOT}}"}`,
          ...parallel,
          "--host {{HOST_IP}}",
          "--port {{PORT}}",
        ],
        builder: {
          topology,
          topologySummary: `${Number(s.gpus_per_node) || 1}× ${hardwareLabels[s.hw] || s.hw}`,
          errors,
          warnings: ["No production benchmark is published for this image checkpoint yet."],
          verification: {
            serve: errors.length ? "error" : "unverified",
            request: errors.length ? "error" : "unverified",
          },
          resolvedSettings: {
            placement: s.placement === "resident" ? "GPU resident" : "CPU offload",
          },
        },
      };
    },
  },

  modelNames: { default: "FLUX.3" },
  placeholders: {
    WEIGHTS_ROOT: {
      target: "command",
      label: "Base model (Hub id or local repo)",
      default: "black-forest-labs/flux-3-image-klein-base",
    },
    DISTILLED_ROOT: {
      target: "command",
      label: "Distilled model (Hub id or local repo)",
      default: "black-forest-labs/flux-3-image-klein",
    },
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
      ...(s.model === "base" ? { num_inference_steps: 50, guidance_scale: 3.5 } : {}),
      seed: 0,
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
