export const config = {
  modelName: "Kandinsky 6.0 SR",
  supportedHardware: ["gb300", "b200", "h200", "h100", "rtxpro6000", "rtx5090"],
  hardware: [
    { id: "rtxpro6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "blackwell" },
  ],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "attention", title: "Attention", scope: "serve", default: "platform",
      description: "Dense attention. Switching kernels can change rounding; neither choice enables NABLA.",
      learnMore: "#5-memory-and-parallelism",
      options: [
        { id: "platform", label: "Recipe default", recommended: true,
          description: "SDPA on RTX; FA on datacenter Blackwell and Hopper." },
        { id: "fa", label: "FlashAttention",
          disabled: (s) => ["rtxpro6000", "rtx5090"].includes(s.hw),
          disableReason: "This SM120 platform maps FA to SDPA; select SDPA explicitly instead." },
        { id: "torch_sdpa", label: "PyTorch SDPA",
          soft: (s) => s.hw !== "rtxpro6000",
          softReason: "SDPA measurements apply to RTX PRO 6000, not every GPU or workload.",
          disabled: (s) => s.topology_mode === "manual" && Number(s.ring_degree) > 1,
          disableReason: "SDPA does not support Ring rotation." },
      ],
    },
    {
      id: "decode", title: "VAE decode", scope: "serve", default: "tiles",
      description: "Whole-tile parallelism preserves the tested serial output; spatial sharding is not bitwise lossless.",
      learnMore: "#4-decode-speed-memory-and-numerics",
      options: [
        { id: "serial", label: "Serial",
          description: "Reference decode. Each rank processes all tiles; single-GPU and single-tile baseline.",
          flags: ["--vae-config.use-parallel-tiling false", "--vae-config.use-parallel-decode false"] },
        { id: "tiles", label: "Whole-tile parallel", recommended: true,
          description: "Distributes complete tiles. Tested RGB/audio are exact versus serial on the same topology. Single-tile inputs fall back to serial.",
          hints: ["Whole-tile decode matched serial RGB/audio in tested same-topology runs."],
          flags: ["--vae-config.use-parallel-tiling true", "--vae-config.use-parallel-decode false"] },
        { id: "spatial", label: "Spatial shard (rounding changes)",
          description: "Lower activation memory, not bitwise lossless. GB300 measurement: PSNR 57.51 dB, min sampled SSIM 0.99921, max pixel difference 4/255; audio exact.",
          hints: ["Not bitwise lossless: spatial decode changes rounding; GB300 PSNR 57.51 dB, max pixel difference 4/255."],
          flags: ["--vae-config.use-parallel-tiling false", "--vae-config.use-parallel-decode true", "--vae-config.parallel-decode-mode spatial_shard"] },
      ],
    },
    {
      id: "placement", title: "Weight placement", scope: "serve", default: "resident",
      description: "Offload reduces GPU weight memory, but adds host memory and transfers.",
      learnMore: "#5-memory-and-parallelism",
      options: [
        { id: "resident", label: "Resident", recommended: true },
        { id: "component", label: "Component offload", soft: true,
          description: "Transfers each selected component whole around its use. This exact SR recipe is unverified.",
          flags: (s) => [`--cpu-offload-components ${s.offload_components}`] },
        { id: "offload", label: "Layerwise offload",
          flags: (s) => [`--layerwise-offload-components ${s.offload_components}`],
          description: "Streams supported DiT, KVAE and latent-upscaler blocks. It can be slower than resident execution." },
        { id: "fsdp", label: "FSDP", soft: true,
          softReason: "SR weight sharding has not completed full-checkpoint E2E validation.",
          description: "Shards DiT weights across GPUs. Selectable for validation, not a verified memory or latency recommendation.",
          flags: ["--use-fsdp-inference true"],
          disabled: (s) => Number(s.gpus_per_node) < 2,
          disableReason: "Weight sharding requires multiple GPUs." },
      ],
    },
    {
      id: "offload_components", title: "Offload components", scope: "serve", default: "all",
      showWhen: (s) => ["component", "offload"].includes(s.placement),
      description: "Choose which weights leave GPU memory. Partial selections are unverified recipes.",
      learnMore: "#5-memory-and-parallelism",
      options: [
        { id: "all", label: "All components" },
        { id: "transformer", label: "DiT", soft: true },
        { id: "vae", label: "VAE", soft: true },
        { id: "latent_upscaler", label: "Latent upscaler", soft: true },
        { id: "transformer vae", label: "DiT + VAE", soft: true },
        { id: "transformer latent_upscaler", label: "DiT + upscaler", soft: true },
        { id: "vae latent_upscaler", label: "VAE + upscaler", soft: true },
      ],
    },
    {
      id: "prefetch", title: "DiT prefetch", scope: "serve", kind: "number",
      min: 1, max: 4, default: 1, unit: "layers", options: [],
      showWhen: (s) => s.placement === "offload" && (s.offload_components === "all" || s.offload_components.includes("transformer")),
      verifiedWhen: (s) => Number(s.prefetch) === 1,
      description: "More prefetched layers use more VRAM and may overlap transfers better. Values above 1 are unverified here.",
      learnMore: "#5-memory-and-parallelism",
    },
    {
      id: "resident_layers", title: "DiT resident layers", scope: "serve", kind: "number",
      min: 0, max: 8, default: 0, unit: "layers", options: [],
      showWhen: (s) => s.placement === "offload" && (s.offload_components === "all" || s.offload_components.includes("transformer")),
      verifiedWhen: (s) => Number(s.resident_layers) === 0,
      description: "Keep leading DiT layers on GPU during the request. Trades VRAM for fewer transfers; nonzero recipes are unverified.",
      learnMore: "#5-memory-and-parallelism",
    },
    {
      id: "scale", title: "Upscale factor", scope: "request", default: "2",
      description: "Choose output scale, not quality. All use the two-step distilled checkpoint.",
      learnMore: "#request-parameters",
      options: [{ id: "2", label: "2x" }, { id: "4", label: "4x" }, { id: "2.25", label: "2.25x" }],
    },
    {
      id: "tile_batch", title: "Tile batch", scope: "request", default: "1",
      description: "Changing tile batching changes seeded noise and therefore the output.",
      options: [
        { id: "1", label: "1", recommended: true },
        { id: "2", label: "2 (different output)",
          description: "Uses more activation memory. Not an output-preserving acceleration." },
      ],
    },
  ],
  commandBuilder: {
    defaultSelection: {
      hw: "gb300", nodes: 1, gpus_per_node: 4, topology_mode: "auto",
      tp_size: 1, ulysses_degree: 4, ring_degree: 1,
    },
    resource: {
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
      verifiedRecipes: [
        { id: "gb300-ulysses4-offline", hw: "gb300", nodes: 1, gpus_per_node: 4,
          tp_size: 1, ulysses_degree: 4, ring_degree: 1, placement: "resident",
          decode: "tiles", default: true, unverified: true },
        ...["b200", "h200", "h100", "rtxpro6000", "rtx5090"].map((hw) => ({
          id: `${hw}-single-gpu`, hw, nodes: 1, gpus_per_node: 1,
          tp_size: 1, ulysses_degree: 1, ring_degree: 1,
          placement: hw === "rtx5090" ? "offload" : "resident",
          decode: "serial", default: true, unverified: hw !== "rtxpro6000",
        })),
        { id: "rtxpro6000-ulysses2-tiles", hw: "rtxpro6000", nodes: 1, gpus_per_node: 2,
          tp_size: 1, ulysses_degree: 2, ring_degree: 1, placement: "resident", decode: "tiles" },
      ],
      autoTopology: (s) => ({
        tp_size: 1,
        ulysses_degree: Number(s.gpus_per_node) === 8 ? 4 : Number(s.gpus_per_node),
        ring_degree: Number(s.gpus_per_node) === 8 ? 2 : 1,
      }),
      validateTopology: (s, t) => {
        const errors = [];
        const degrees = [t.tp_size, t.ulysses_degree, t.ring_degree];
        if (!degrees.every((n) => Number.isInteger(n) && n >= 1))
          errors.push("Parallel degrees must be positive integers.");
        if (Number(s.nodes) !== 1) errors.push("This picker covers one node.");
        if (degrees.reduce((a, b) => a * b, 1) !== Number(s.gpus_per_node))
          errors.push("GPU count must equal TP times Ulysses times Ring.");
        if (28 % (t.tp_size * t.ulysses_degree))
          errors.push("TP times Ulysses must divide the 28 DiT heads.");
        if (s.placement === "fsdp" && Number(s.gpus_per_node) < 2)
          errors.push("FSDP requires multiple GPUs.");
        if ((s.attention === "torch_sdpa" || ["rtxpro6000", "rtx5090"].includes(s.hw)) && t.ring_degree > 1)
          errors.push("SDPA does not support Ring rotation.");
        if (s.attention === "fa" && ["rtxpro6000", "rtx5090"].includes(s.hw))
          errors.push("FA resolves to SDPA on this SM120 platform; select SDPA instead.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
        : resource.autoTopology(s);
      const errors = resource.validateTopology(s, topology);
      const platformAttention = ["rtxpro6000", "rtx5090"].includes(s.hw) ? "torch_sdpa" : "fa";
      const attention = s.attention === "platform" ? platformAttention : s.attention;
      const streamingDit = s.placement === "offload" && (s.offload_components === "all" || s.offload_components.includes("transformer"));
      const gb300Served = s.hw === "gb300" && attention === "fa" && (
        (topology.tp_size === 2 && topology.ulysses_degree === 2 && topology.ring_degree === 1
          && s.placement === "resident" && s.decode !== "spatial")
        || (topology.tp_size === 1 && topology.ulysses_degree === 1 && topology.ring_degree === 2
          && s.placement === "offload" && s.offload_components === "all"
          && Number(s.prefetch) === 1 && Number(s.resident_layers) === 0 && s.decode !== "spatial")
      );
      const rtxServed = s.hw === "rtxpro6000" && attention === "torch_sdpa"
        && s.placement === "resident" && topology.tp_size === 1 && topology.ring_degree === 1
        && ((topology.ulysses_degree === 1 && s.decode === "serial")
          || (topology.ulysses_degree === 2 && s.decode === "tiles"));
      const served = errors.length === 0 && (gb300Served || rtxServed);
      const warnings = [
        "Verification describes execution, not lossless output. Arbitrary uploaded videos are outside the fixed HTTP test workload.",
      ];
      if (!served) warnings.push("This exact HTTP recipe is unverified. Offline timing does not verify every server and uploaded-video combination.");
      if (!["gb300", "rtxpro6000"].includes(s.hw))
        warnings.push("Extrapolated hardware recipe: no SR latency or peak-memory measurement on this GPU. Validate a short clip before scaling up.");
      if (attention !== platformAttention)
        warnings.push("This backend override is a different numerical path; its full-checkpoint output and performance are unverified.");
      if (s.hw === "rtx5090")
        warnings.push("32GB VRAM: start with layerwise offload, tile batch 1 and a short 384 x 256 source. Offload does not bound activation memory; full-size video may still OOM.");
      if (Number(s.gpus_per_node) > 4)
        warnings.push("More than four GPUs is extrapolated. Eight GPUs use Ulysses4 x Ring2 with FA; Ulysses8 cannot divide 28 heads.");
      if (s.hw === "rtxpro6000" && Number(s.gpus_per_node) > 2)
        warnings.push("RTX PRO 6000 measurements cover one or two GPUs only; larger GPU counts are extrapolated.");
      if (["component", "offload"].includes(s.placement))
        warnings.push("Offload uses host RAM and transfer bandwidth; prefetch and resident layers additionally consume VRAM.");
      if (s.decode === "spatial") warnings.push("Not bitwise lossless: spatial decode changes rounding. Tested PSNR 57.51 dB, max pixel difference 4/255; validate your video.");
      if (s.decode === "tiles") warnings.push("Whole-tile decode matched serial RGB/audio exactly on tested workloads within each topology, not across different TP/Ring settings.");
      if (topology.tp_size > 1 || topology.ring_degree > 1)
        warnings.push("TP/Ring can change output numerics independently of the VAE decode mode.");
      if (s.tile_batch !== "1") warnings.push("Tile batch 2 changes seeded noise and output; it is not lossless.");
      if (Number(s.gpus_per_node) === 1 && s.decode !== "serial")
        warnings.push("One GPU uses serial decoding; this selection provides no parallel speedup.");
      return {
        match: { hw: s.hw }, nnodes: Number(s.nodes), verified: false,
        flags: [
          "--model-path {{MODEL_NAME}}", `--num-gpus ${Number(s.gpus_per_node)}`,
          `--tp-size ${topology.tp_size}`, `--ulysses-degree ${topology.ulysses_degree}`,
          `--ring-degree ${topology.ring_degree}`, "--performance-mode speed",
          `--attention-backend ${attention}`, "--warmup-mode off", "--host {{HOST_IP}}", "--port {{PORT}}",
          ...(streamingDit && Number(s.prefetch) !== 1 ? [`--dit-offload-prefetch-size ${s.prefetch}`] : []),
          ...(streamingDit && Number(s.resident_layers) !== 0 ? [`--dit-layerwise-resident-layers ${s.resident_layers}`] : []),
        ],
        builder: {
          topology,
          resolvedSettings: {
            attention: attention === "fa" ? "FlashAttention" : "PyTorch SDPA",
            prefetch: `${s.prefetch} ${Number(s.prefetch) === 1 ? "layer" : "layers"}`,
            resident_layers: `${s.resident_layers} ${Number(s.resident_layers) === 1 ? "layer" : "layers"}`,
          },
          topologySummary: `TP ${topology.tp_size}, Ulysses ${topology.ulysses_degree}, Ring ${topology.ring_degree}`,
          errors, warnings,
          verification: {
            serve: errors.length ? "error" : served ? "verified" : "unverified",
            request: errors.length ? "error" : "unverified",
          },
        },
      };
    },
  },
  modelNames: { default: "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    INPUT_VIDEO: { target: "curl", label: "Source video", default: "/path/to/input.mp4" },
  },
  curl: (s) => `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos \\
  -F "video_reference=@{{INPUT_VIDEO}};type=video/mp4" \\
  -F sr_resolution_scale=${s.scale} \\
  -F sr_tiles_batch_size=${s.tile_batch} \\
  -F num_inference_steps=2 \\
  -F seed=42`,
  runModes: () => ["python"],
  showPlaygroundLink: false,
  cells: [],
};
