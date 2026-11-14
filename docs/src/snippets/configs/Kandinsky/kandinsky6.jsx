export const config = {
  modelName: "Kandinsky 6.0",
  supportedHardware: ["b200", "gb300", "rtxpro6000"],
  hardware: [
    { id: "rtxpro6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
  ],
  groupHardware: false,
  matchDims: [],
  overlayDims: [
    {
      id: "weights",
      title: "Checkpoint",
      scope: "base",
      default: "distill",
      description:
        "Pro-distill uses 10 steps without CFG; Pro uses 50 steps with CFG.",
      options: [
        { id: "distill", label: "Pro-distill", recommended: true },
        { id: "pro", label: "Pro" },
      ],
    },
    {
      id: "mode",
      title: "Conditioning",
      scope: "base",
      default: "text",
      description:
        "Both modes generate video and audio through the same server.",
      options: [
        { id: "text", label: "Text" },
        { id: "image", label: "Text and image" },
      ],
    },
    {
      id: "attention", title: "Attention", scope: "serve", default: "platform",
      description: "The recipe uses FA on B200/GB300 and SDPA on RTX PRO 6000. Sage3 overrides only the DiT; kernel changes can alter both outputs.",
      learnMore: "#rtx-pro-6000-and-sageattention3",
      options: [
        { id: "platform", label: "Recipe default", recommended: true },
        { id: "fa", label: "FlashAttention",
          disabled: (s) => s.hw === "rtxpro6000",
          disableReason: "This SM120 platform maps FA to SDPA; select SDPA explicitly instead." },
        { id: "torch_sdpa", label: "PyTorch SDPA",
          soft: (s) => s.hw !== "rtxpro6000",
          softReason: "Full-checkpoint SDPA validation used RTX PRO 6000.",
          disabled: (s) => s.topology_mode === "manual" && Number(s.ring_degree) > 1,
          disableReason: "SDPA does not support Ring rotation." },
        { id: "sage_attn_3", label: "Sage3 (approximate)", soft: true,
          disabled: (s) => s.hw !== "rtxpro6000",
          disableReason: "The tested Sage3 build requires SM120/121; it rejects B200/GB300.",
          softReason: "Offline benchmark only; this HTTP recipe is unverified. Requires a separate SageAttention3 installation.",
          description: "RTX: 21.3% lower latency, but video PSNR 11.97 dB and audio SNR -2.21 dB versus SDPA. Not lossless or quality-certified.",
          hints: ["Sage3 is approximate and changes video/audio substantially. Install the documented SM120 SageAttention3 build first."] },
      ],
    },
    {
      id: "placement",
      title: "Weight placement",
      scope: "serve",
      default: "resident",
      description: "Keep weights resident when they fit; trade transfer time for lower GPU memory otherwise.",
      learnMore: "#choosing-a-topology",
      options: [
        { id: "resident", label: "Resident", recommended: true,
          description: "Baseline arithmetic, without weight transfers between stages." },
        { id: "fsdp", label: "FSDP",
          description: "Shards DiT weights. GB300 Ulysses4 matched resident RGB/audio, using less GPU memory at higher latency.",
          flags: ["--use-fsdp-inference true"],
          disabled: (s) => Number(s.gpus_per_node) < 2,
          disableReason: "Weight sharding requires multiple GPUs." },
        { id: "component", label: "Component offload", soft: true,
          description: "Transfers selected components whole around their use. Partial recipes need their own memory and latency validation.",
          flags: (s) => [`--cpu-offload-components ${s.offload_components}`] },
        { id: "offload", label: "Layerwise offload",
          description: "Streams blocks from host RAM. This adds transfers and pinned host memory; not a free speedup.",
          flags: (s) => [`--layerwise-offload-components ${s.offload_components}`] },
      ],
    },
    {
      id: "offload_components", title: "Offload components", scope: "serve", default: "all",
      showWhen: (s) => ["component", "offload"].includes(s.placement),
      description: "Text encoders includes both encoders; VAEs includes video and audio. Partial selections are unverified recipes.",
      learnMore: "#choosing-a-topology",
      options: [
        { id: "all", label: "All components" },
        { id: "transformer", label: "DiT", soft: true },
        { id: "text_encoder", label: "Text encoders", soft: true },
        { id: "vae audio_vae", label: "VAEs", soft: true },
        { id: "audio_vae", label: "Audio VAE", soft: true },
        { id: "text_encoder vae audio_vae", label: "Encoders + VAEs", soft: true },
      ],
    },
    {
      id: "prefetch", title: "DiT prefetch", scope: "serve", kind: "number",
      min: 1, max: 4, default: 1, unit: "layers", options: [],
      showWhen: (s) => s.placement === "offload" && ["all", "transformer"].includes(s.offload_components),
      verifiedWhen: (s) => Number(s.prefetch) === 1,
      description: "More prefetched layers use more VRAM and may overlap transfers better. Values above 1 are unverified here.",
      learnMore: "#choosing-a-topology",
    },
    {
      id: "resident_layers", title: "DiT resident layers", scope: "serve", kind: "number",
      min: 0, max: 8, default: 0, unit: "layers", options: [],
      showWhen: (s) => s.placement === "offload" && ["all", "transformer"].includes(s.offload_components),
      verifiedWhen: (s) => Number(s.resident_layers) === 0,
      description: "Keep leading DiT layers on GPU during the request. Trades VRAM for fewer transfers; nonzero recipes are unverified.",
      learnMore: "#choosing-a-topology",
    },
    {
      id: "workload",
      title: "Workload",
      scope: "request",
      default: "full",
      description:
        "Use the lightweight request to check deployment, not output quality.",
      options: [
        { id: "full", label: "768 x 512, 121 frames", recommended: true },
        { id: "smoke", label: "384 x 256, 17 frames, 2 steps" },
      ],
    },
    {
      id: "cache", title: "Cache-DiT", scope: "request", default: "off",
      description: "Request-level block reuse can change both video and audio. The default cache policy is not a validated quality preset.",
      learnMore: "#cache-dit",
      options: [
        { id: "off", label: "Off", recommended: true },
        { id: "on", label: "On (approximate)", soft: true,
          description: "Uses the runtime cache defaults. Short requests may finish before cache warmup ends, with no acceleration." },
      ],
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
      limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 4 } },
      verifiedRecipes: [
        ["b200", 1, 1],
        ["b200", 2, 1],
        ["gb300", 4, 1],
        ["gb300", 4, 2],
        ["rtxpro6000", 1, 1],
      ].map(([hw, gpus, tp]) => ({
        id: `${hw}-distill-${gpus}-tp${tp}`,
        hw,
        weights: "distill",
        placement: "resident",
        nodes: 1,
        gpus_per_node: gpus,
        tp_size: tp,
        ulysses_degree: gpus / tp,
        ring_degree: 1,
        default: tp === 1 && (hw === "gb300" ? gpus === 4 : gpus === 1),
      })).concat([{
        id: "gb300-distill-fsdp",
        hw: "gb300", weights: "distill", placement: "fsdp",
        nodes: 1, gpus_per_node: 4, tp_size: 1, ulysses_degree: 4, ring_degree: 1,
      }]),
      autoTopology: (s) => ({
        tp_size: 1,
        ulysses_degree: Number(s.gpus_per_node),
        ring_degree: 1,
      }),
      validateTopology: (s, t) => {
        const errors = [];
        const degrees = [t.tp_size, t.ulysses_degree, t.ring_degree];
        if (!degrees.every((n) => Number.isInteger(n) && n >= 1))
          errors.push("Parallel degrees must be positive integers.");
        if (Number(s.nodes) !== 1) errors.push("This picker covers one node.");
        if (degrees.reduce((a, b) => a * b, 1) !== Number(s.gpus_per_node))
          errors.push("GPU count must equal TP times Ulysses times Ring.");
        if (32 % (t.tp_size * t.ulysses_degree))
          errors.push("TP times Ulysses must divide the 32 DiT heads.");
        if (s.placement === "fsdp" && Number(s.gpus_per_node) < 2)
          errors.push("FSDP requires multiple GPUs.");
        if ((["torch_sdpa", "sage_attn_3"].includes(s.attention) || s.hw === "rtxpro6000") && t.ring_degree > 1)
          errors.push("SDPA and Sage3 do not support Ring rotation; use FlashAttention on B200/GB300.");
        if (s.attention === "fa" && s.hw === "rtxpro6000")
          errors.push("FA resolves to SDPA on this SM120 platform; select SDPA instead.");
        if (s.attention === "sage_attn_3" && s.hw !== "rtxpro6000")
          errors.push("The tested Sage3 build requires SM120/121.");
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
      const platformAttention = s.hw === "rtxpro6000" ? "torch_sdpa" : "fa";
      const attention = s.attention === "platform" ? platformAttention : s.attention;
      const streamingDit = s.placement === "offload" && ["all", "transformer"].includes(s.offload_components);
      const recipe = resource.verifiedRecipes.find(
        (r) =>
          r.hw === s.hw &&
          r.weights === s.weights &&
          r.placement === s.placement &&
          r.nodes === Number(s.nodes) &&
          r.gpus_per_node === Number(s.gpus_per_node) &&
          r.tp_size === topology.tp_size &&
          r.ulysses_degree === topology.ulysses_degree &&
          r.ring_degree === topology.ring_degree,
      );
      const served = !!recipe && errors.length === 0 && attention === platformAttention;
      const requested = served && s.workload === "smoke" && s.cache === "off";
      const warnings = [];
      if (!requested)
        warnings.push(
          served
            ? "Only the uncached lightweight workload is verified for this HTTP recipe; full-size generation is checked offline."
            : "This exact checkpoint, placement and topology combination has not completed HTTP verification.",
        );
      if (topology.tp_size > 1)
        warnings.push(
          "TP changes video and audio outputs. Prefer Ulysses when memory permits; validate quality for your workload.",
        );
      if (topology.ring_degree > 1)
        warnings.push("Ring changes floating-point reductions; bitwise video/audio parity is not established.");
      if (["component", "offload"].includes(s.placement))
        warnings.push("Offload needs additional host RAM and transfer bandwidth; DiT prefetch and resident layers also consume VRAM.");
      if (attention !== platformAttention)
        warnings.push("This attention override has not completed the exact HTTP recipe; changing kernels can change video and audio.");
      if (attention === "sage_attn_3")
        warnings.push("Sage3 is approximate: measured video PSNR 11.97 dB and audio SNR -2.21 dB versus SDPA; no quality guarantee.");
      if (s.cache === "on")
        warnings.push("Cache-DiT is approximate and affects both streams; validate video and audio. Short requests may not skip any blocks.");
      const model =
        s.weights === "pro"
          ? "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
          : "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers";
      return {
        match: { hw: s.hw },
        nnodes: Number(s.nodes),
        verified: requested,
        flags: [
          `--model-path ${model}`,
          `--num-gpus ${Number(s.gpus_per_node)}`,
          `--tp-size ${topology.tp_size}`,
          `--ulysses-degree ${topology.ulysses_degree}`,
          `--ring-degree ${topology.ring_degree}`,
          "--performance-mode speed",
          `--attention-backend ${attention === "sage_attn_3" ? platformAttention : attention}`,
          ...(attention === "sage_attn_3" ? ["--component-attention-backends transformer=sage_attn_3"] : []),
          "--warmup-mode off",
          "--host {{HOST_IP}}",
          "--port {{PORT}}",
          ...(streamingDit && Number(s.prefetch) !== 1 ? [`--dit-offload-prefetch-size ${s.prefetch}`] : []),
          ...(streamingDit && Number(s.resident_layers) !== 0 ? [`--dit-layerwise-resident-layers ${s.resident_layers}`] : []),
        ],
        builder: {
          topology,
          resolvedSettings: {
            attention: { fa: "FlashAttention", torch_sdpa: "PyTorch SDPA", sage_attn_3: "Sage3 (approximate)" }[attention],
            prefetch: `${s.prefetch} ${Number(s.prefetch) === 1 ? "layer" : "layers"}`,
            resident_layers: `${s.resident_layers} ${Number(s.resident_layers) === 1 ? "layer" : "layers"}`,
          },
          topologySummary: `TP ${topology.tp_size}, Ulysses ${topology.ulysses_degree}, Ring ${topology.ring_degree}`,
          errors,
          warnings,
          verification: {
            serve: errors.length ? "error" : served ? "verified" : "unverified",
            request: errors.length
              ? "error"
              : requested
                ? "verified"
                : "unverified",
          },
        },
      };
    },
  },
  modelNames: {
    default: "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
  },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    INPUT_IMAGE: {
      target: "curl",
      label: "Reference PNG",
      default: "/path/to/reference.png",
    },
  },
  curl: (s) => {
    const smoke = s.workload === "smoke";
    const body = {
      prompt:
        "A woman chops vegetables in a sunlit kitchen, with soft jazz playing.",
      size: smoke ? "384x256" : "768x512",
      num_frames: smoke ? 17 : 121,
      fps: 24,
      seed: 42,
      enable_cache_dit: s.cache === "on",
    };
    if (smoke) body.num_inference_steps = 2;
    if (s.mode === "text")
      return `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify(body, null, 2)}'`;
    const fields = Object.entries(body).map(
      ([key, value]) => `  --form-string '${key}=${value}'`,
    );
    fields.push('  -F "input_reference=@{{INPUT_IMAGE}};type=image/png"');
    return `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos \\
${fields.join(" \\\n")}`;
  },
  runModes: () => ["python"],
  showPlaygroundLink: false,
  cells: [],
};
