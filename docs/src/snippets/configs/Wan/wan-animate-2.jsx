export const config = (() => {
  const cfgDegree = (s) => s.cfg === "on" || (s.cfg === "auto" && Number(s.gpus_per_node) % 2 === 0) ? 2 : 1;
  const placementOf = (s) => s.hw === "rtx4090" && s.placement === "auto" ? "memory" : s.placement;
  const recipes = [
    ["h200", 1, "auto"],
    ["h200", 1, "memory"],
    ["b300", 1, "auto"],
    ["b300", 1, "memory"],
    ["b300", 2, "auto"],
    ["b300", 2, "cfg-regression"],
    ["rtx4090", 1, "memory"],
    ["gb200", 1, "auto", true],
    ["h100", 1, "offload", true],
  ].map(([hw, gpus, placement, unverified = false]) => ({
    id: [hw, gpus, placement].join("-"), hw, nodes: 1, gpus_per_node: gpus,
    placement, encoder: "auto", cfg: "auto",
    tp_size: 1, ulysses_degree: 1, ring_degree: 1,
    default: gpus === 1 && (placement !== "memory" || hw === "rtx4090"),
    unverified,
  }));

  const config = {
    modelName: "Wan-Animate-2",
    supportedHardware: ["h200", "b300", "rtx4090", "gb200", "h100"],
    hardware: [
      { id: "rtx4090", label: "RTX 4090", vram: "24GB", vendor: "consumer" },
    ],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "placement", title: "Memory policy", scope: "serve", default: "auto",
        description: "CFG parity preserves the measured single-B300 output using offload, replicated encoders and serial VAE decode.",
        learnMore: "#5-1-hardware-comparison",
        options: [
          { id: "auto", label: "Auto", showWhen: (s) => s.hw !== "rtx4090" },
          { id: "speed", label: "Speed", showWhen: (s) => s.hw !== "rtx4090" },
          { id: "offload", label: "DiT layerwise offload", showWhen: (s) => s.hw !== "rtx4090" },
          { id: "memory", label: "Memory" },
          { id: "cfg-regression", label: "CFG parity", showWhen: (s) => s.hw !== "rtx4090",
            disabled: (s) => Number(s.gpus_per_node) !== 2 || cfgDegree(s) !== 2,
            disableReason: "CFG parity requires 2 GPUs with CFG parallelism." },
        ],
      },
      {
        id: "encoder", title: "Text encoder", scope: "serve", default: "auto",
        showWhen: (s) => s.hw !== "rtx4090",
        options: [
          { id: "auto", label: "Auto" },
          { id: "offload", label: "CPU offload", showWhen: (s) => s.hw !== "rtx4090" },
        ],
      },
      {
        id: "cfg", title: "CFG parallelism", scope: "serve", default: "auto",
        description: "Auto uses CFG parallelism for even GPU counts. TP and Ulysses are editable under Setup's advanced topology.",
        options: [
          { id: "auto", label: "Auto" },
          { id: "off", label: "Disabled" },
          { id: "on", label: "Enabled", disabled: (s) => Number(s.gpus_per_node) % 2 !== 0,
            disableReason: "CFG parallelism requires an even GPU count." },
        ],
      },
      {
        id: "resolution", title: "Pixel-area budget", scope: "request", default: "640x800",
        description: "The output keeps the reference image's aspect ratio; this is not a fixed output size.",
        learnMore: "#2-2-defaults",
        options: [
          { id: "640x800", label: "640 x 800" },
          { id: "720x1280", label: "720 x 1280" },
        ],
      },
      {
        id: "clip_len", title: "Clip length", scope: "request", default: "37",
        description: "Frames per denoising chunk, not total video length. Longer chunks require more VRAM.",
        options: ["17", "37", "65", "81"].map((id) => ({ id, label: id + " frames" })),
      },
      {
        id: "steps", title: "Denoising steps", scope: "request", kind: "number",
        default: 40, min: 1, max: 100, unit: "steps", options: [],
      },
      {
        id: "audio", title: "Reference audio", scope: "request", default: "false",
        options: [
          { id: "false", label: "Silent" },
          { id: "true", label: "Keep audio" },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: {
        hw: "h200", nodes: 1, gpus_per_node: 1, topology_mode: "auto",
        tp_size: 1, ulysses_degree: 1, ring_degree: 1,
      },
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
        verifiedRecipes: recipes,
        autoTopology: (s) => {
          const ranks = Number(s.gpus_per_node) / cfgDegree(s);
          const tp = ranks >= 4 && ranks % 2 === 0 ? 2 : 1;
          return { tp_size: tp, ulysses_degree: ranks / tp, ring_degree: 1 };
        },
        validateTopology: (s, t) => {
          const errors = [];
          const gpus = Number(s.gpus_per_node);
          if (Number(s.nodes) !== 1) errors.push("This cookbook covers single-node deployment.");
          if (!Number.isInteger(gpus) || gpus < 1 || gpus > 8) errors.push("Choose 1-8 GPUs per node.");
          if (![t.tp_size, t.ulysses_degree, t.ring_degree].every((n) => Number.isInteger(n) && n >= 1))
            errors.push("Parallel degrees must be positive integers.");
          if (gpus !== cfgDegree(s) * t.tp_size * t.ulysses_degree * t.ring_degree)
            errors.push("GPU count must equal CFG times TP times Ulysses times Ring.");
          if (40 % (t.tp_size * t.ulysses_degree))
            errors.push("TP times Ulysses must divide the 40 DiT attention heads.");
          if (t.ring_degree !== 1) errors.push("Wan-Animate-2 does not support Ring parallelism; keep Ring at 1.");
          if (s.placement === "cfg-regression" && (gpus !== 2 || cfgDegree(s) !== 2))
            errors.push("CFG parity requires 2 GPUs with CFG parallelism.");
          if (s.hw === "rtx4090" && (placementOf(s) !== "memory" || s.encoder !== "auto"))
            errors.push("The RTX 4090 recipe requires Memory policy with the default encoder policy.");
          return errors;
        },
      },
      resolveDeployment: (s) => {
        const resource = config.commandBuilder.resource;
        const topology = s.topology_mode === "manual"
          ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
          : resource.autoTopology(s);
        const errors = resource.validateTopology(s, topology);
        const placement = placementOf(s);
        const recipe = recipes.find((r) => r.hw === s.hw && r.nodes === Number(s.nodes)
          && r.gpus_per_node === Number(s.gpus_per_node) && r.placement === placement
          && r.tp_size === topology.tp_size && r.ulysses_degree === topology.ulysses_degree
          && r.ring_degree === topology.ring_degree && cfgDegree(r) === cfgDegree(s));
        const served = !!recipe && !recipe.unverified && s.encoder === "auto" && errors.length === 0;
        const requested = served && s.resolution === "640x800" && Number(s.clip_len) === 37
          && Number(s.steps) === 40 && s.audio === "false";
        const flags = ["--model-path {{MODEL_NAME}}", "--port {{PORT}}", "--num-gpus " + s.gpus_per_node];
        const env = s.hw === "rtx4090" ? ["PYTORCH_ALLOC_CONF=expandable_segments:True"] : [];
        if (cfgDegree(s) === 2) flags.push("--enable-cfg-parallel");
        if (topology.tp_size > 1) flags.push("--tp-size " + topology.tp_size);
        if (topology.ulysses_degree > 1) flags.push("--ulysses-degree " + topology.ulysses_degree);
        if (placement === "speed" || placement === "memory") flags.push("--performance-mode " + placement);
        if (placement === "offload" || placement === "cfg-regression")
          flags.push("--dit-layerwise-offload", "--layerwise-offload-components transformer");
        if (placement === "cfg-regression") {
          env.push("SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=1");
          flags.push("--encoder-parallel replicate", "--vae-config.use-parallel-decode false");
        }
        if (s.encoder === "offload") flags.push("--text-encoder-cpu-offload");

        const warnings = [];
        if (!served && !errors.length) warnings.push("This exact deployment has not completed HTTP verification; see the benchmark for offline coverage.");
        if (served && !requested) warnings.push("These request settings are outside the verified HTTP workload.");
        if (s.hw === "rtx4090") warnings.push("24 GB recipe: memory mode plus an expandable allocator. The 24-frame benchmark uses nearly all VRAM; longer clips or larger pixel budgets may OOM. Offload also needs sufficient host RAM.");
        if (s.hw === "h100") warnings.push("Use DiT layerwise offload for the documented 80 GB memory preset. H100 recipes have not been measured in this cookbook.");
        const placementLabel = { auto: "Auto", speed: "Speed", memory: "Memory", offload: "DiT layerwise offload", "cfg-regression": "CFG parity" }[placement];
        return {
          match: { hw: s.hw }, nnodes: Number(s.nodes), verified: served, flags, env,
          builder: {
            topology,
            topologySummary: ["CFG " + cfgDegree(s), "TP " + topology.tp_size, "Ulysses " + topology.ulysses_degree, placementLabel].join(" / "),
            errors, warnings,
            verification: {
              serve: errors.length ? "error" : served ? "verified" : "unverified",
              request: errors.length ? "error" : requested ? "verified" : "unverified",
            },
            resolvedSettings: {
              placement: placementLabel,
              cfg: cfgDegree(s) === 2 ? "Enabled" : "Disabled",
            },
          },
        };
      },
    },
    modelNames: { default: "Wan-AI/Wan2.2-Animate-2-14B-Diffusers" },
    placeholders: {
      PORT: { target: "command", label: "Server port", default: "30010" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Request port", default: "30010" },
      INPUT_IMAGE: { target: "curl", label: "Reference PNG on the client", default: "/path/to/reference.png" },
      INPUT_VIDEO: { target: "curl", label: "Reference video on the server", default: "/path/to/reference_video.mp4" },
    },
    curl: (s) => [
      "curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos",
      '--form-string "prompt=a person dancing"',
      '--form-string "video_path={{INPUT_VIDEO}}"',
      '--form-string "clip_len=' + s.clip_len + '"',
      '--form-string "size=' + s.resolution + '"',
      '--form-string "num_inference_steps=' + s.steps + '"',
      '--form-string "guidance_scale=3.0"',
      '--form-string "fps=16"',
      '--form-string "seed=42"',
      '--form-string "enable_audio=' + s.audio + '"',
      '--form "input_reference=@{{INPUT_IMAGE}};type=image/png"',
    ].join(" \\\n  "),
    runModes: ["python"],
    showPlaygroundLink: false,
    github: { cookbookModel: "Wan-AI/Wan2.2-Animate-2-14B-Diffusers" },
  };
  return config;
})();
