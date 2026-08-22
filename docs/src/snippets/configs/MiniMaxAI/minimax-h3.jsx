// MiniMax-H3 diffusion deployment matrix. Consumed by _deployment.jsx.
//
// H3 is the first opt-in user of the scoped commandBuilder renderer. Topology,
// checkpoint, server overlays, and request fields share one semantic selection,
// while the UI presents them by lifecycle and composes them through the existing
// deployment command engine.


export const config = {
  modelName: "MiniMax-H3",

  supportedHardware: [
    "b200",
    "b300",
    "h200",
    "h100",
    "mi300x",
    "mi355x",
    "rtx5090",
  ],
  hardware: [
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "consumer" },
  ],
  groupHardware: false,

  matchDims: [],

  overlayDims: [
    {
      id: "weights",
      title: "Checkpoint Weights",
      scope: "base",
      description: "Choose the checkpoint partition required by the request mode.",
      default: "fl2va",
      options: [
        {
          id: "fl2va",
          label: "FL2VA",
          subtitle: "(First-and-Last-Frame-to-Video-and-Audio)",
          flags: ["--model-variant fl2va"],
        },
        {
          id: "ref2va",
          label: "Ref2VA",
          subtitle: "(Reference-to-Video-and-Audio)",
          flags: ["--model-variant ref2va"],
        },
      ],
    },
    {
      id: "mode",
      title: "Request Mode",
      scope: "base",
      description: "The visible modes follow the selected checkpoint.",
      default: "t2va",
      options: [
        {
          id: "t2va",
          label: "Text only",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "i2va",
          label: "First frame",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "l2va",
          label: "Last frame",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "fl2va",
          label: "First + last frames",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "ref_image",
          label: "Image reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "ref_image_audio",
          label: "Image + audio",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "v2v",
          label: "Video reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "video_audio",
          label: "Video + soundtrack",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "audio_only",
          label: "Audio reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "mixed_ref",
          label: "Mixed references",
          showWhen: (s) => s.weights === "ref2va",
        },
      ],
    },
    {
      id: "placement",
      title: "Placement",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/api/cli#component-residency",
      description: "Keep weights resident for latency; shard or offload only when capacity requires it.",
      quality: "Memory policy",
      learnMore: "#7-feature-contracts-and-advanced-recipes",
      default: "resident",
      options: [
        {
          id: "auto",
          label: "Auto",
          flags: (s) => {
            const recipe = config.commandBuilder.resource.verifiedRecipes.find((entry) =>
              entry.hw === s.hw && entry.nodes === Number(s.nodes)
              && entry.gpus_per_node === Number(s.gpus_per_node));
            const placement = recipe?.placement || (s.hw === "rtx5090" ? "offload" : "resident");
            if (placement === "fsdp") return ["--performance-mode speed", "--use-fsdp-inference true"];
            return placement === "offload" ? [
                "--performance-mode memory",
                "--layerwise-offload-components dit,text_encoder,vae",
                "--dit-offload-prefetch-size 1",
                "--dit-layerwise-resident-layers 20",
                "--enable-torch-compile false",
              ] : ["--performance-mode speed"];
          },
          description: "Use the recommended placement for the selected hardware and resource shape.",
        },
        {
          id: "resident",
          label: "Resident",
          flags: ["--performance-mode speed"],
          recommendedWhen: (s) => s.hw !== "rtx5090",
          description: "Lowest-latency path when the full pipeline fits in aggregate GPU memory.",
        },
        {
          id: "fsdp",
          label: "FSDP",
          flags: ["--performance-mode speed", "--use-fsdp-inference true"],
          soft: (s) => !["b200", "b300", "h200", "h100"].includes(s.hw) || s.nodes > 1,
          softReason: "Verified on single-node B200/B300/H200/H100. Other hardware and multi-node runs take the same flags but have not been through a verification round.",
          description: "Reduces resident DiT memory but adds parameter collectives on every block.",
        },
        {
          id: "offload",
          label: "Layerwise offload",
          flags: [
            "--performance-mode memory",
            "--layerwise-offload-components dit,text_encoder,vae",
            "--dit-offload-prefetch-size 1",
            "--dit-layerwise-resident-layers 20",
            "--enable-torch-compile false",
          ],
          soft: (s) => s.hw !== "rtx5090",
          softReason: "Tuned and verified on RTX 5090. It runs on the datacenter GPUs too, where a resident recipe is simply faster.",
          recommendedWhen: (s) => s.hw === "rtx5090",
          description: "Capacity-first PCIe path. It is substantially slower than a resident datacenter recipe.",
        },
      ],
    },
    {
      id: "attention",
      title: "Attention",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/attention_backends",
      description: "Select the packed-attention kernel used by H3 transformer modules.",
      quality: "Kernel policy",
      learnMore: "#7-feature-contracts-and-advanced-recipes",
      default: "platform",
      options: [
        {
          id: "platform",
          label: "Automatic",
          flags: (s) => ["mi300x", "mi355x"].includes(s.hw) ? ["--attention-backend aiter"] : [],
          env: (s) => ["mi300x", "mi355x"].includes(s.hw) ? ["SGLANG_USE_AITER=1"] : [],
          recommended: true,
          description: "Applies the verified backend policy for the selected hardware.",
        },
        {
          id: "fa",
          label: "FlashAttention",
          flags: ["--attention-backend fa"],
          disabled: (s) => ["mi300x", "mi355x"].includes(s.hw),
          disableReason: "Use the verified AITER platform default on AMD.",
          description: "An explicit native-dtype CUDA comparison path; reduction ordering may still differ.",
        },
        {
          id: "sage",
          label: "SageAttention",
          flags: ["--attention-backend sage_attn"],
          disabled: (s) => ["mi300x", "mi355x"].includes(s.hw),
          disableReason: "SageAttention is not exposed for the AMD recipes.",
          description: "Approximate attention math. Install its packed-varlen dependency and inspect video and audio quality.",
        },
      ],
    },
    {
      id: "precision",
      title: "Precision",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/quantization",
      description: "Choose native mixed precision or a validated online weight quantization path.",
      quality: "Weight precision",
      learnMore: "#7-feature-contracts-and-advanced-recipes",
      default: "native",
      options: [
        {
          id: "native",
          label: "BF16 / FP32",
          recommended: true,
          description: "Reference mixed precision: BF16 transformer weights with required projections retained in FP32.",
        },
        {
          id: "fp8",
          label: "Online FP8",
          flags: ["--quantization fp8"],
          soft: (s) => !["b200", "b300"].includes(s.hw)
            || !["auto", "resident"].includes(s.placement)
            || s.nodes !== 1,
          softReason: "Verified for resident single-node B200/B300. Other hardware and placements take the same flag, but those recipes have not been verified yet.",
          description: "Approximate transformer weight quantization with the required H3 projections protected.",
        },
      ],
    },
    {
      id: "encoder",
      title: "Encoder",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/encoder_parallel",
      description: "Control text-encoder work placement independently from DiT topology.",
      quality: "Parallel policy",
      learnMore: "#7-feature-contracts-and-advanced-recipes",
      default: "auto",
      options: [
        {
          id: "auto",
          label: "Auto",
          flags: (s) => [`--encoder-parallel ${s.nodes > 1 ? "replicate" : "auto"}`],
          recommended: true,
          description: "Folds on verified single-host P2P systems and resolves to replicate across nodes.",
        },
        {
          id: "dp",
          label: "Data parallel",
          flags: ["--encoder-parallel dp"],
          disabled: (s) => (s.topology_mode === "manual"
            ? Number(s.tp_size)
            : config.commandBuilder.resource.autoTopology(s).tp_size) > 1,
          disableReason: "The server rejects encoder DP with TP > 1 (encoder_parallel=dp requires tp_size=1).",
          soft: (s) => s.nodes > 1,
          softReason: "Runs across nodes, but the measured 1.9× encode speedup comes from a single-node 2× H100 run; cross-node encoder DP is unverified.",
          description: "Useful for a real request batch; it is not bitwise-identical to fold scheduling.",
        },
        {
          id: "fold",
          label: "Fold",
          flags: ["--encoder-parallel fold"],
          disabled: (s) => s.nodes > 1,
          disableReason: "Fold assumes fast node-local peer-to-peer access.",
          description: "Uses one folded encoder copy across a node-local group and preserves native weights.",
        },
        {
          id: "replicate",
          label: "Replicate",
          flags: ["--encoder-parallel replicate"],
          recommendedWhen: (s) => s.nodes > 1,
          description: "The safe cross-node default because encoder auto is not node-boundary aware.",
        },
      ],
    },
    {
      id: "execution",
      title: "Execution",
      scope: "serve",
      description: "Choose eager execution or the measured breakable CUDA graph path.",
      quality: "Graph policy",
      learnMore: "#7-feature-contracts-and-advanced-recipes",
      default: "eager",
      options: [
        {
          id: "eager",
          label: "Eager",
          recommended: true,
          description: "Reference execution and the consistency baseline.",
        },
        {
          id: "bcg",
          label: "Compatible BCG",
          flags: [
            "--enable-breakable-cuda-graph true",
            "--warmup-resolutions 1344x768",
            "--bcg-text-buckets 5504",
          ],
          soft: (s) => !(["b200", "h200"].includes(s.hw) && s.weights === "ref2va"),
          softReason: "Verified for B200/H200 Ref2VA; BCG runs on the other recipes but they have not been through a verification round yet.",
          description: "Reuses matching execution signatures and reserves capture memory; it takes precedence over Cache-DiT.",
        },
      ],
    },
    {
      id: "quality",
      title: "Quality",
      scope: "request",
      docsHref: "/docs/sglang-diffusion/cache_dit",
      description: "Reference execution or the audited Cache-DiT acceleration preset.",
      quality: "Sampling policy",
      learnMore: "#choose-the-quality-level",
      default: "lossless",
      options: [
        {
          id: "lossless",
          label: "Lossless",
          recommended: true,
          description: "Reference-exact denoising without Cache-DiT approximation.",
        },
        {
          id: "high",
          label: "Audited high",
          disabled: (s) => s.execution !== "eager",
          disableReason: "BCG supersedes Cache-DiT, so the preset would have no effect — switch Execution to Eager to use it.",
          soft: (s) => !(s.hw === "h200" && s.nodes === 1 && s.gpus_per_node === 4
            && ["auto", "resident"].includes(s.placement)),
          softReason: "The 1.40× / SSIM 0.931 audit covers the resident eager 4× H200 workload; elsewhere the preset runs but its quality figures are unaudited.",
          description: "Measured 1.40× with SSIM 0.931 and PSNR 28.16 dB on the audited workload.",
        },
      ],
    },
    {
      id: "outputs",
      title: "Outputs",
      scope: "request",
      description: "Generate independent variants from one request.",
      quality: "1–10",
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
      gpus_per_node: 8,
      topology_mode: "auto",
      tp_size: 1,
      ulysses_degree: 8,
      ring_degree: 1,
    },
    resource: {
      limits: {
        nodes: { min: 1, max: 8 },
        gpus_per_node: { min: 1, max: 8 },
      },
      verifiedRecipes: [
        { id: "b200-resident-8", hw: "b200", nodes: 1, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto", default: true },
        { id: "b200-fsdp-4", hw: "b200", nodes: 1, gpus_per_node: 4, placement: "fsdp", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "b300-resident-8", hw: "b300", nodes: 1, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto", default: true },
        { id: "b300-fsdp-8", hw: "b300", nodes: 1, gpus_per_node: 8, placement: "fsdp", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto" },
        { id: "h200-resident-4", hw: "h200", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto", default: true },
        { id: "h200-fsdp-4", hw: "h200", nodes: 1, gpus_per_node: 4, placement: "fsdp", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "h200-cross-node-16", hw: "h200", nodes: 2, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 2, encoder: "replicate" },
        { id: "h100-resident-4", hw: "h100", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 2, ulysses_degree: 2, ring_degree: 1, encoder: "auto", default: true },
        { id: "h100-fsdp-4", hw: "h100", nodes: 1, gpus_per_node: 4, placement: "fsdp", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "mi300x-resident-1", hw: "mi300x", nodes: 1, gpus_per_node: 1, placement: "resident", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto" },
        { id: "mi300x-resident-2", hw: "mi300x", nodes: 1, gpus_per_node: 2, placement: "resident", tp_size: 1, ulysses_degree: 2, ring_degree: 1, encoder: "auto" },
        { id: "mi300x-resident-4", hw: "mi300x", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "mi300x-resident-8", hw: "mi300x", nodes: 1, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto", default: true },
        { id: "mi355x-resident-1", hw: "mi355x", nodes: 1, gpus_per_node: 1, placement: "resident", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto" },
        { id: "mi355x-resident-2", hw: "mi355x", nodes: 1, gpus_per_node: 2, placement: "resident", tp_size: 1, ulysses_degree: 2, ring_degree: 1, encoder: "auto" },
        { id: "mi355x-resident-4", hw: "mi355x", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "mi355x-resident-8", hw: "mi355x", nodes: 1, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx5090-offload-2", hw: "rtx5090", nodes: 1, gpus_per_node: 2, placement: "offload", tp_size: 2, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
      ],
      autoTopology: (s) => {
        const recipes = config.commandBuilder.resource.verifiedRecipes;
        const exact = recipes.find((recipe) => recipe.hw === s.hw
          && recipe.nodes === Number(s.nodes)
          && recipe.gpus_per_node === Number(s.gpus_per_node)
          && (s.placement === "auto" || recipe.placement === s.placement));
        if (exact) {
          return {
            tp_size: exact.tp_size,
            ulysses_degree: exact.ulysses_degree,
            ring_degree: exact.ring_degree,
          };
        }
        return {
          tp_size: 1,
          ulysses_degree: Number(s.gpus_per_node),
          ring_degree: Number(s.nodes),
        };
      },
      validateTopology: (s, topology) => {
        const errors = [];
        const nodes = Number(s.nodes);
        const perNode = Number(s.gpus_per_node);
        const world = nodes * perNode;
        const tp = Number(topology.tp_size);
        const ulysses = Number(topology.ulysses_degree);
        const ring = Number(topology.ring_degree);
        if (!Number.isInteger(nodes) || nodes < 1 || nodes > 8) errors.push("H3 supports 1–8 nodes.");
        if (!Number.isInteger(perNode) || perNode < 1 || perNode > 8) errors.push("H3 supports 1–8 GPUs per node.");
        if (![1, 2, 4, 8].includes(tp)) errors.push("Tensor parallel size must be one of 1, 2, 4, or 8.");
        if (world !== tp * ulysses * ring) errors.push(`World size ${world} must equal TP × Ulysses × Ring (${tp * ulysses * ring}).`);
        if (56 % tp !== 0 || (56 / tp) % ulysses !== 0) errors.push("H3's 56 attention heads must divide evenly across TP and Ulysses.");
        if (64 % (ulysses * ring) !== 0) errors.push("Ulysses × Ring must divide the 64 packed sequence partitions.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? {
            tp_size: Number(s.tp_size),
            ulysses_degree: Number(s.ulysses_degree),
            ring_degree: Number(s.ring_degree),
          }
        : resource.autoTopology(s);
      const errors = resource.validateTopology(s, topology);
      const automaticRecipe = resource.verifiedRecipes.find((entry) => entry.hw === s.hw
        && entry.nodes === Number(s.nodes)
        && entry.gpus_per_node === Number(s.gpus_per_node)
        && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree
        && entry.ring_degree === topology.ring_degree);
      const resolvedPlacement = s.placement === "auto"
        ? (automaticRecipe?.placement || (s.hw === "rtx5090" ? "offload" : "resident"))
        : s.placement;
      const coverageWarnings = [];
      if (resolvedPlacement === "offload" && s.hw !== "rtx5090") {
        coverageWarnings.push("Layerwise offload is tuned and verified on RTX 5090; on this hardware it runs unverified and a resident recipe is faster.");
      }
      if (resolvedPlacement === "fsdp" && (s.nodes !== 1 || !["b200", "b300", "h200", "h100"].includes(s.hw))) {
        coverageWarnings.push("FSDP outside the single-node NVIDIA recipes runs unverified.");
      }
      if (s.precision === "fp8" && (! ["b200", "b300"].includes(s.hw)
        || resolvedPlacement !== "resident" || s.nodes !== 1)) {
        coverageWarnings.push("Online FP8 outside resident single-node B200/B300 runs unverified.");
      }
      const highAudited = s.hw === "h200" && s.nodes === 1
        && s.gpus_per_node === 4 && resolvedPlacement === "resident";
      if (s.quality === "high" && s.execution !== "eager") {
        coverageWarnings.push("BCG supersedes Cache-DiT, so the high preset has no effect under this execution mode.");
      } else if (s.quality === "high" && !highAudited) {
        coverageWarnings.push("The high preset's 1.40× / SSIM 0.931 figures were audited on resident eager 4× H200; this workload is unaudited.");
      }

      const recipe = resource.verifiedRecipes.find((entry) => entry.hw === s.hw
        && entry.nodes === Number(s.nodes)
        && entry.gpus_per_node === Number(s.gpus_per_node)
        && entry.placement === resolvedPlacement
        && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree
        && entry.ring_degree === topology.ring_degree);
      const topologyVerified = !!recipe && errors.length === 0;
      const encoderVerified = s.encoder === "auto"
        || s.encoder === recipe?.encoder
        || (s.nodes > 1 && s.encoder === "replicate");
      const attentionVerified = s.attention === "platform";
      const precisionVerified = s.precision === "native"
        || (s.precision === "fp8" && ["b200", "b300"].includes(s.hw));
      const executionVerified = s.execution === "eager"
        || (s.execution === "bcg" && ["b200", "h200"].includes(s.hw) && s.weights === "ref2va");
      const serveVerified = topologyVerified && encoderVerified && attentionVerified
        && precisionVerified && executionVerified;
      const requestVerified = topologyVerified && (s.quality === "lossless"
        || (s.quality === "high" && highAudited && s.execution === "eager"));

      const topologyParts = [];
      if (topology.tp_size > 1) topologyParts.push(`TP ${topology.tp_size}`);
      if (Number(s.nodes) > 1) {
        topologyParts.push(`Ulysses ${topology.ulysses_degree} inside each node`);
        topologyParts.push(`Ring ${topology.ring_degree} across nodes`);
      } else {
        topologyParts.push(`Ulysses ${topology.ulysses_degree}`);
      }
      topologyParts.push({ resident: "Resident", fsdp: "FSDP", offload: "Layerwise offload" }[resolvedPlacement]);
      topologyParts.push(Number(s.nodes) > 1 ? `${s.nodes} nodes` : "Single node");

      const world = Number(s.nodes) * Number(s.gpus_per_node);
      const flags = ["--model-path {{MODEL_NAME}}", `--num-gpus ${world}`];
      if (topology.ring_degree > 1) flags.push(`--sp-degree ${world}`);
      if (topology.tp_size > 1) flags.push(`--tp-size ${topology.tp_size}`);
      flags.push(`--ulysses-degree ${topology.ulysses_degree}`);
      if (topology.ring_degree > 1) flags.push(`--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");

      const warnings = [...coverageWarnings];
      if (!topologyVerified && errors.length === 0) {
        warnings.push("This topology satisfies H3's static constraints but has not completed an exact end-to-end verification run.");
      }
      if (resolvedPlacement === "fsdp") {
        warnings.push("FSDP lowers resident DiT memory but adds per-block parameter collectives; prefer Resident when the pipeline fits.");
      }
      if (s.hw === "rtx5090") {
        warnings.push("The 2× RTX 5090 path requires a 384 GiB-class host and prioritizes capacity over latency.");
      }

      let automaticAttention = "FlashAttention (auto)";
      if (["mi300x", "mi355x"].includes(s.hw)) {
        automaticAttention = "AITER (auto)";
      } else if (topology.ring_degree === 1 && ["b200", "b300"].includes(s.hw)) {
        automaticAttention = "Dynamic cuDNN / FA (auto)";
      } else if (topology.ring_degree === 1 && s.hw === "rtx5090") {
        automaticAttention = "Torch SDPA (auto)";
      }

      return {
        match: { hw: s.hw },
        nnodes: Number(s.nodes),
        verified: serveVerified,
        verificationStatus: serveVerified ? "verified" : "unverified",
        flags,
        builder: {
          topology,
          topologySummary: topologyParts.filter(Boolean).join(" · "),
          errors,
          warnings,
          verification: {
            serve: errors.length ? "error" : (serveVerified ? "verified" : "unverified"),
            request: errors.length ? "error" : (requestVerified ? "verified" : "unverified"),
          },
          resolvedSettings: {
            placement: { resident: "Resident", fsdp: "FSDP", offload: "Layerwise offload" }[resolvedPlacement],
            attention: s.attention === "platform" ? automaticAttention : undefined,
            encoder: s.encoder === "auto" ? (s.nodes > 1 ? "Replicate (auto)" : "Auto") : undefined,
          },
        },
      };
    },
  },

  modelNames: {
    default: "MiniMaxAI/MiniMax-H3",
  },

  placeholders: {
    HOST_IP: {
      target: "command",
      label: "Bind host",
      default: "0.0.0.0",
    },
    PORT: {
      target: "command",
      label: "Bind port",
      default: "30010",
    },
    HF_TOKEN: {
      target: "command",
      label: "HF token (Docker)",
      default: "<your-hf-token>",
    },
    MEDIA_DIR: {
      target: "command",
      label: "Host media directory (Docker)",
      default: "/data/minimax-h3",
    },
    CURL_HOST: {
      target: "curl",
      label: "Server host",
      default: "localhost",
    },
    CURL_PORT: {
      target: "curl",
      label: "Server port",
      default: "30010",
    },
    DURATION_SECONDS: {
      target: "curl",
      label: "Duration (seconds, 4-15)",
      default: "5",
    },
    FIRST_FRAME: {
      target: "curl",
      label: "FL2VA first frame URI",
      default: "file:///data/minimax-h3/first-frame.png",
    },
    LAST_FRAME: {
      target: "curl",
      label: "FL2VA last frame URI",
      default: "file:///data/minimax-h3/last-frame.png",
    },
    INPUT_VIDEO: {
      target: "curl",
      label: "First video URI",
      default: "file:///data/minimax-h3/video-1.mp4",
    },
    INPUT_VIDEO_START_SECONDS: {
      target: "curl",
      label: "First video start (seconds)",
      default: "0",
    },
    SECOND_INPUT_VIDEO: {
      target: "curl",
      label: "Second video URI (mixed ref)",
      default: "file:///data/minimax-h3/video-2.mp4",
    },
    SECOND_INPUT_VIDEO_START_SECONDS: {
      target: "curl",
      label: "Second video start (seconds)",
      default: "0",
    },
    REFERENCE_IMAGE: {
      target: "curl",
      label: "First reference image URI",
      default: "file:///data/minimax-h3/reference-1.png",
    },
    SECOND_REFERENCE_IMAGE: {
      target: "curl",
      label: "Second reference image URI",
      default: "file:///data/minimax-h3/reference-2.png",
    },
    REFERENCE_AUDIO: {
      target: "curl",
      label: "First reference audio URI",
      default: "file:///data/minimax-h3/reference-1.mp3",
    },
    SECOND_REFERENCE_AUDIO: {
      target: "curl",
      label: "Second reference audio URI",
      default: "file:///data/minimax-h3/reference-2.mp3",
    },
  },

  curl: (s) => {
    const request = {
      model: "{{MODEL_NAME}}",
      prompt:
        "Night-vision bedroom footage: while the owner sleeps, three cats burst in playing tiny brass instruments at full volume, freeze, then march out as if nothing happened.",
      seconds: "{{DURATION_SECONDS}}",
      task: "t2va",
      conditions: [],
      target: {
        short_edge: 768,
        aspect_ratio: "16:9",
        duration_seconds: "{{DURATION_SECONDS}}",
      },
      quality: s.quality,
      num_outputs_per_prompt: Number(s.outputs),
      num_inference_steps: 50,
      flow_shift: 12.0,
      audio_flow_shift: 3.0,
      seed: 1101,
    };
    const imageReference = (uri) => ({
      type: "image",
      uri,
      role: "reference",
    });
    const audioReference = (uri) => ({
      type: "audio",
      uri,
      role: "reference",
    });
    const videoReference = (uri, start, type = "video") => ({
      type,
      uri,
      role: "reference",
      start_time_seconds: start,
    });

    if (["i2va", "l2va", "fl2va"].includes(s.mode)) {
      request.task = "fl2va";
      request.prompt =
        "Continue naturally between the supplied endpoint frame or frames, with synchronized ambient sound.";
      request.target.aspect_ratio = "auto";
      request.seed = 2101;
      request.conditions = [];
      if (s.mode !== "l2va") {
        request.conditions.push({
          type: "image",
          uri: "{{FIRST_FRAME}}",
          role: "keyframe",
          frame_index: 0,
        });
      }
      if (s.mode !== "i2va") {
        request.conditions.push({
          type: "image",
          uri: "{{LAST_FRAME}}",
          role: "keyframe",
          frame_index: -1,
        });
      }
    } else if (s.mode === "ref_image") {
      request.task = "ref2va";
      request.prompt = "Use <Picture 1> as the visual subject and style reference.";
      request.target.aspect_ratio = "auto";
      request.conditions = [imageReference("{{REFERENCE_IMAGE}}")];
      request.seed = 3101;
    } else if (s.mode === "ref_image_audio") {
      request.task = "ref2va";
      request.prompt =
        "Use <Picture 1> as the visual subject and <Audio 1> as the sound reference.";
      request.target.aspect_ratio = "auto";
      request.conditions = [
        imageReference("{{REFERENCE_IMAGE}}"),
        audioReference("{{REFERENCE_AUDIO}}"),
      ];
      request.seed = 3102;
    } else if (s.mode === "v2v" || s.mode === "video_audio") {
      request.task = "ref2va";
      request.prompt =
        s.mode === "video_audio"
          ? "Follow <Video 1> and its required <Audio 1> soundtrack with coherent synchronized motion."
          : "Follow the appearance and motion of <Video 1>; use its soundtrack when present.";
      request.conditions = [
        videoReference(
          "{{INPUT_VIDEO}}",
          "{{INPUT_VIDEO_START_SECONDS}}",
          s.mode === "video_audio" ? "video_audio" : "video",
        ),
      ];
      request.seed = s.mode === "video_audio" ? 4102 : 4101;
    } else if (s.mode === "audio_only") {
      request.task = "ref2va";
      request.prompt = "Build a coherent visual scene around <Audio 1>.";
      request.conditions = [audioReference("{{REFERENCE_AUDIO}}")];
      request.seed = 3103;
    } else if (s.mode === "mixed_ref") {
      request.task = "ref2va";
      request.prompt =
        "Combine <Picture 1>, <Picture 2>, <Audio 1>, <Audio 2>, <Video 1>, and <Video 2> in their one-based modality order.";
      request.conditions = [
        imageReference("{{REFERENCE_IMAGE}}"),
        imageReference("{{SECOND_REFERENCE_IMAGE}}"),
        audioReference("{{REFERENCE_AUDIO}}"),
        audioReference("{{SECOND_REFERENCE_AUDIO}}"),
        videoReference("{{INPUT_VIDEO}}", "{{INPUT_VIDEO_START_SECONDS}}"),
        videoReference(
          "{{SECOND_INPUT_VIDEO}}",
          "{{SECOND_INPUT_VIDEO_START_SECONDS}}",
        ),
      ];
      request.seed = 3104;
    }

    const body = JSON.stringify(request, null, 2).replace(
      /"{{(DURATION_SECONDS|INPUT_VIDEO_START_SECONDS|SECOND_INPUT_VIDEO_START_SECONDS)}}"/g,
      "{{$1}}",
    );
    return `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos \\
  -H 'Content-Type: application/json' \\
  -d '${body}'`;
  },

  dockerMounts: ["{{MEDIA_DIR}}:/data/minimax-h3:ro"],

  dockerRunCommand: (s) =>
    ["mi300x", "mi355x"].includes(s.hw)
      ? `bash -lc 'python -m pip install -e "/sgl-workspace/sglang/python[diffusion_hip]" && exec sglang serve "$@"' --`
      : `bash -lc 'python -m pip install -e "/sgl-workspace/sglang/python[diffusion]" && exec sglang serve "$@"' --`,

  // Publish AMD Docker only after an H3-capable ROCm image has been validated.
  runModes: (s) =>
    ["mi300x", "mi355x"].includes(s.hw)
      ? ["python"]
      : ["python", "docker"],

  dockerImages: {
    b200: "lmsysorg/sglang:dev",
    b300: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
    h100: "lmsysorg/sglang:dev",
  },

  showPlaygroundLink: false,

  cells: [],
};
