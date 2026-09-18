export const config = (() => {
const sm120Hardware = ["rtx5090", "rtxpro6000"];
const platformAttention = (s) => sm120Hardware.includes(s.hw) ? "sdpa" : "fa";
const effectiveAttention = (s) => s.attention === "platform" || (sm120Hardware.includes(s.hw) && s.attention === "fa") ? platformAttention(s) : s.attention;

const config = {
  modelName: "Qwen-Image 2.1",
  supportedHardware: ["h200", "b200", "rtxpro6000", "rtx5090", "rtx4090"],
  hardware: [
    { id: "rtxpro6000", label: "RTX PRO 6000", vram: "96GB", vendor: "consumer" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "consumer" },
    { id: "rtx4090", label: "RTX 4090", vram: "24GB", vendor: "consumer" },
  ],
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
      description: "Hardware selection applies its recommended placement. Stream DiT layers when the full pipeline exceeds device memory.",
      learnMore: "#5-runtime-features",
      default: "resident",
      options: [
        {
          id: "resident", label: "Resident",
          recommendedWhen: (s) => ["h200", "b200", "rtxpro6000"].includes(s.hw),
          disabled: (s) => ["rtx5090", "rtx4090"].includes(s.hw) && Number(s.gpus_per_node) === 1,
          disableReason: "The full resident pipeline exceeds one consumer GPU's memory. Select CPU offload.",
          flags: (s) => [Number(s.gpus_per_node) === 1 ? "--performance-mode speed" : "--performance-mode manual"],
          description: "Keep all components on the GPU. Recommended for H200, B200, and RTX PRO 6000 96GB. RTX 5090 and RTX 4090 need offload.",
        },
        {
          id: "offload", label: "CPU offload",
          flags: (s) => ["--performance-mode manual", "--dit-layerwise-offload true", ...(s.hw === "rtx4090" ? ["--text-encoder-cpu-offload true"] : [])],
          recommendedWhen: (s) => ["rtx5090", "rtx4090"].includes(s.hw),
          soft: (s) => !["rtxpro6000", "rtx5090", "rtx4090"].includes(s.hw) || Number(s.gpus_per_node) !== 1,
          softReason: "This offload topology has not completed an HTTP verification run.",
          description: "Streams DiT layers. RTX 4090 also offloads the encoder between requests to leave room for image editing. Requires sufficient host RAM.",
        },
        {
          id: "all_offload", label: "All components layerwise",
          flags: ["--performance-mode manual", "--layerwise-offload-components all"],
          soft: true, softReason: "Full-checkpoint 512px editing passed on B200, including TP2 with spatial VAE decode; this HTTP recipe is unverified.",
          description: "Streams repeated blocks in the DiT, Qwen3-VL, and VAE. Uses more host-device transfers to reduce device memory.",
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
        {
          id: "platform", label: "Automatic", recommended: true,
          flags: (s) => [`--attention-backend ${platformAttention(s) === "sdpa" ? "torch_sdpa" : "fa"}`],
          description: "Uses SDPA on RTX PRO 6000 and RTX 5090, and FlashAttention on the other listed GPUs.",
        },
        { id: "fa", label: "FlashAttention", flags: ["--attention-backend fa"], description: "Exact attention with a fused kernel. This runtime falls back to Torch SDPA on RTX PRO 6000 and RTX 5090." },
        {
          id: "sdpa", label: "Torch SDPA", flags: ["--attention-backend torch_sdpa"],
          soft: (s) => !config.commandBuilder.resource.verifiedRecipes.some((r) => r.hw === s.hw && r.placement === s.placement && r.attentions.includes("sdpa") && Number(s.gpus_per_node) === r.gpus_per_node),
          softReason: "This hardware and placement combination has not completed HTTP verification with SDPA.",
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
      description: "Native precision is the default. Quantization changes image and alpha values. Set compatible FP8/NVFP4 directories or GGUF files under Variables.",
      default: "native",
      options: [
        { id: "native", label: "Native BF16 / FP32", recommended: true },
        {
          id: "fp8_dit", label: "Online FP8 DiT", flags: ["--component-quantizations.transformer fp8"],
          soft: true, softReason: "Online FP8 passed 1024px/40-step generation and editing on one resident B200. Other hardware, alpha, and feature combinations remain unverified.",
        },
        {
          id: "fp8_encoder", label: "Online FP8 encoder", flags: ["--component-quantizations.text_encoder fp8"],
          soft: true, softReason: "Online encoder FP8 passed 1024px/40-step generation and editing on one resident B200. It changes conditioning and output pixels.",
        },
        {
          id: "fp8_both", label: "Online FP8 DiT + encoder", flags: ["--component-quantizations.transformer fp8", "--component-quantizations.text_encoder fp8"],
          soft: true, softReason: "Online FP8 for both components passed generation, editing, and transparent output on one resident B200. Quality depends on the workload.",
        },
        {
          id: "serialized_fp8_dit", label: "Serialized FP8 DiT", flags: ['--component-paths.transformer "{{FP8_DIT_PATH}}"'],
          soft: true, softReason: "A tensorwise E4M3FN component export passed 1024px/40-step generation, editing, and transparent output on B200. Validate your exported checkpoint's quality.",
        },
        {
          id: "serialized_fp8_encoder", label: "Serialized FP8 encoder", flags: ['--component-paths.text_encoder "{{FP8_ENCODER_PATH}}"'],
          soft: true, softReason: "A tensorwise E4M3FN language encoder export passed generation, editing, and transparent output on B200; vision weights retain native precision.",
        },
        {
          id: "serialized_fp8_both", label: "Serialized FP8 DiT + encoder", flags: ['--component-paths.transformer "{{FP8_DIT_PATH}}"', '--component-paths.text_encoder "{{FP8_ENCODER_PATH}}"'],
          soft: true, softReason: "Exported components passed 1024px/40-step generation, editing, and transparent output on B200. All-component offload matched resident pixels after the vision RoPE fix; TP2 changes numerical results. Validate your exported checkpoint's quality.",
        },
        {
          id: "gguf_dit", label: "GGUF DiT", flags: ['--component-weights-paths.transformer "{{GGUF_DIT_PATH}}"'],
          soft: true, softReason: "A Q4_0 DiT export passed 1024px/40-step generation, editing, and transparent output on B200. Other exports and hardware need validation.",
        },
        {
          id: "gguf_encoder", label: "GGUF encoder", flags: ['--component-weights-paths.text_encoder "{{GGUF_ENCODER_PATH}}"'],
          soft: true, softReason: "A native-name Q4_0 language encoder export passed generation, editing, and transparent output on B200; vision weights retain native precision.",
        },
        {
          id: "gguf_both", label: "GGUF DiT + encoder", flags: ['--component-weights-paths.transformer "{{GGUF_DIT_PATH}}"', '--component-weights-paths.text_encoder "{{GGUF_ENCODER_PATH}}"'],
          soft: true, softReason: "Combined Q4_0 exports passed 1024px/40-step generation, editing, and transparent output on B200. GGUF reduces weight memory; output quality and speed depend on the export and workload.",
        },
        {
          id: "nvfp4_dit", label: "NVFP4 DiT", flags: ['--component-paths.transformer "{{NVFP4_DIT_PATH}}"'],
          disabled: (s) => !["b200", "rtxpro6000", "rtx5090"].includes(s.hw),
          disableReason: "Native NVFP4 requires a Blackwell GPU (compute capability 10.0 or newer).",
          soft: true, softReason: "A calibrated ModelOpt-format DiT export passed 1024px/40-step generation, editing, and transparent output on B200. Other exports, RTX PRO 6000, and RTX 5090 need validation.",
        },
        {
          id: "nvfp4_encoder", label: "NVFP4 encoder", flags: ['--component-paths.text_encoder "{{NVFP4_ENCODER_PATH}}"'],
          disabled: (s) => !["b200", "rtxpro6000", "rtx5090"].includes(s.hw),
          disableReason: "Native NVFP4 requires a Blackwell GPU (compute capability 10.0 or newer).",
          soft: true, softReason: "A calibrated language-encoder export passed generation, editing, and transparent output on B200; vision weights retain native precision. Output quality requires validation.",
        },
        {
          id: "nvfp4_both", label: "NVFP4 DiT + encoder", flags: ['--component-paths.transformer "{{NVFP4_DIT_PATH}}"', '--component-paths.text_encoder "{{NVFP4_ENCODER_PATH}}"'],
          disabled: (s) => !["b200", "rtxpro6000", "rtx5090"].includes(s.hw),
          disableReason: "Native NVFP4 requires a Blackwell GPU (compute capability 10.0 or newer).",
          soft: true, softReason: "Combined exports passed generation, editing, transparent output, offload, and TP2 on B200. The small max-calibration sample changes image and alpha values; validate your exported checkpoint.",
        },
      ],
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
        { id: "fold", label: "Fold", flags: ["--encoder-parallel fold"], soft: true, softReason: "Native encoder TP and full-checkpoint TP2 × SP2 editing passed on B200. Requires node-local P2P; this HTTP recipe is unverified." },
      ],
    },
    {
      id: "vae",
      title: "VAE decoding",
      scope: "serve",
      description: "Decode RGBA in full, in tiles, or with spatial work distributed across GPUs.",
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
        {
          id: "spatial", label: "Spatial shard", flags: ["--vae-config.parallel-decode-mode spatial_shard"],
          disabled: (s) => Number(s.gpus_per_node) < 2,
          disableReason: "Select at least two GPUs for spatial VAE decode.",
          soft: true, softReason: "Two-B200 full-checkpoint decoding passed with TP, CFG parallelism, and all-component offload; this HTTP recipe is unverified.",
          description: "Splits feature-map height and exchanges convolution halos. Preserves full-image attention; floating-point rounding can change pixels.",
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
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 4 } },
      verifiedRecipes: [
        { id: "h200-1-resident", hw: "h200", nodes: 1, gpus_per_node: 1, placement: "resident", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", attentions: ["fa"], default: true },
        { id: "b200-1-resident", hw: "b200", nodes: 1, gpus_per_node: 1, placement: "resident", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", attentions: ["fa", "sdpa"], default: true },
        { id: "rtxpro6000-1-resident", hw: "rtxpro6000", nodes: 1, gpus_per_node: 1, placement: "resident", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", attentions: ["sdpa"], default: true },
        { id: "rtxpro6000-1-offload", hw: "rtxpro6000", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", attentions: ["sdpa"] },
        { id: "rtx5090-1-offload", hw: "rtx5090", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", attentions: ["sdpa"], default: true },
        { id: "rtx4090-1-offload", hw: "rtx4090", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", attentions: ["fa"], default: true },
      ],
      autoTopology: (s) => ({ tp_size: 1, ulysses_degree: Number(s.gpus_per_node), ring_degree: 1 }),
      validateTopology: (s, topology) => {
        const errors = [];
        const nodes = Number(s.nodes);
        const perNode = Number(s.gpus_per_node);
        const { tp_size: tp, ulysses_degree: ulysses, ring_degree: ring } = topology;
        if (nodes !== 1) errors.push("This picker covers single-node deployment only.");
        if (![1, 2, 4].includes(perNode)) errors.push("Select one, two, or four GPUs per node.");
        if (![tp, ulysses, ring].every((n) => [1, 2, 4].includes(n))) errors.push("TP, Ulysses and Ring must each be 1, 2, or 4.");
        if (nodes * perNode !== tp * ulysses * ring) errors.push(`World size ${nodes * perNode} must equal TP × Ulysses × Ring (${tp * ulysses * ring}).`);
        if (32 % (tp * ulysses) !== 0) errors.push("32 attention heads must be divisible by TP × Ulysses.");
        if (ring > 1 && effectiveAttention(s) === "sdpa") errors.push("Ring requires FlashAttention or SageAttention; Torch SDPA is unsupported.");
        if (s.precision?.startsWith("nvfp4_") && !["b200", "rtxpro6000", "rtx5090"].includes(s.hw)) errors.push("Native NVFP4 requires a Blackwell GPU. Select B200, RTX PRO 6000, or RTX 5090.");
        if (perNode === 1 && ["rtx5090", "rtx4090"].includes(s.hw) && s.placement === "resident") errors.push("The full resident pipeline exceeds this GPU's memory. Select CPU offload.");
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
        && recipe.attentions.includes(effectiveAttention(s)) && s.precision === "native"
        && s.execution === "eager" && s.vae === "full";
      // Exact HTTP workloads from the validation matrix, not blanket quality coverage.
      const requestVerified = serveVerified
        && ((["text", "edit"].includes(s.mode) && s.resolution === "1024" && Number(s.steps) === 40 && Number(s.outputs) === 1
            && (["h200", "rtxpro6000"].includes(s.hw) || s.mode === "text" || s.background === "scene"))
          || (s.hw === "h200" && s.background === "scene" && s.mode === "text" && s.resolution === "512" && Number(s.steps) === 4 && Number(s.outputs) === 2)
          || (s.hw === "h200" && s.background === "scene" && s.mode === "multi" && s.resolution === "512" && Number(s.steps) === 4 && Number(s.outputs) === 1));
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
            attention: s.attention === "platform" ? `${platformAttention(s) === "sdpa" ? "Torch SDPA" : "FlashAttention"} (auto)`
              : sm120Hardware.includes(s.hw) && s.attention === "fa" ? "Torch SDPA (FA fallback)" : undefined,
            encoder: s.encoder === "auto" && world === 1 ? "Single GPU (auto)" : undefined,
          },
        },
      };
    },
  },

  modelNames: { default: "Qwen-Image-2.1" },
  placeholders: {
    MODEL_PATH: { target: "command", label: "Authorized checkpoint directory", default: "/models/qwen-image-2.1" },
    FP8_DIT_PATH: { target: "command", label: "Serialized FP8 DiT directory", default: "/models/qwen-image-2.1-fp8/transformer" },
    FP8_ENCODER_PATH: { target: "command", label: "Serialized FP8 encoder directory", default: "/models/qwen-image-2.1-fp8/text_encoder" },
    GGUF_DIT_PATH: { target: "command", label: "GGUF DiT file", default: "/models/qwen-image-2.1-gguf/transformer-Q4_0.gguf" },
    GGUF_ENCODER_PATH: { target: "command", label: "GGUF encoder file", default: "/models/qwen-image-2.1-gguf/text_encoder-Q4_0.gguf" },
    NVFP4_DIT_PATH: { target: "command", label: "NVFP4 DiT directory", default: "/models/qwen-image-2.1-nvfp4/transformer" },
    NVFP4_ENCODER_PATH: { target: "command", label: "NVFP4 encoder directory", default: "/models/qwen-image-2.1-nvfp4/text_encoder" },
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

return config;
})();
