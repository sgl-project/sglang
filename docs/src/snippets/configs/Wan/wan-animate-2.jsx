export const config = (() => {
  const distilled = (s) => s.weights === "distilled";
  // The distilled checkpoint runs one branch per step and rejects --enable-cfg-parallel.
  const cfgDegree = (s) => !distilled(s) && (s.cfg === "on" || (s.cfg === "auto" && Number(s.gpus_per_node) % 2 === 0)) ? 2 : 1;
  const placementOf = (s) => s.hw === "rtx4090" && s.placement === "auto" ? "memory" : s.placement;
  // The distilled recipes were measured with the text encoder on the CPU, so Auto resolves to that policy for them.
  const encoderOf = (s) => distilled(s) && s.encoder === "auto" ? "offload" : s.encoder;
  const recipes = [
    ["h200", 1, "auto"],
    ["h200", 1, "memory"],
    ["b300", 1, "auto"],
    ["b300", 1, "memory"],
    ["b300", 2, "auto"],
    ["b300", 2, "cfg-regression"],
    ["rtx4090", 1, "memory"],
    ["h100", 1, "offload", true],
  ].map(([hw, gpus, placement, unverified = false]) => ({
    id: [hw, gpus, placement].join("-"), hw, nodes: 1, gpus_per_node: gpus,
    weights: "default", placement, encoder: "auto", cfg: "auto",
    tp_size: 1, ulysses_degree: 1, ring_degree: 1,
    default: gpus === 1 && (placement !== "memory" || hw === "rtx4090"),
    unverified,
    verifiedWhen: (s) => !distilled(s),
  }));
  // Distilled checkpoint: GB200 HTTP runs, two identical /v1/videos requests per
  // server, text encoder on the CPU. The 1-GPU resident entry is also GB200's
  // derived shape for the base checkpoint.
  recipes.push(...[
    [1, "auto", 1, 1],
    [1, "offload", 1, 1],
    [2, "auto", 1, 2],
    [2, "offload", 1, 2],
    [2, "auto", 2, 1],
    [4, "auto", 1, 4],
    [4, "offload", 1, 4],
    [4, "auto", 2, 2],
    [4, "auto", 4, 1],
  ].map(([gpus, placement, tp, ulysses]) => ({
    id: ["gb200", gpus, placement, "tp" + tp, "sp" + ulysses].join("-"),
    hw: "gb200", nodes: 1, gpus_per_node: gpus,
    weights: "distilled", placement, encoder: "auto", cfg: "auto",
    tp_size: tp, ulysses_degree: ulysses, ring_degree: 1,
    default: gpus === 1 && placement === "auto",
    unverified: false,
    verifiedWhen: distilled,
  })));

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
        id: "weights", title: "Checkpoint weights", scope: "base", default: "default",
        description: "Both checkpoints run the native WanAnimate2Pipeline with the same inputs. Restart the server when changing weights.",
        learnMore: "#2-model-capabilities",
        options: [
          { id: "default", label: "Wan-Animate-2", description: "Base checkpoint: 40 steps at guidance 3.0 by default, CFG parallelism available." },
          { id: "distilled", label: "Wan-Animate-2 Distilled", description: "Distilled checkpoint: 10 steps at guidance 1.0 (the only accepted value), one branch per step; measured on GB200." },
        ],
      },
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
            disableReason: "CFG parity requires 2 GPUs with CFG parallelism (base checkpoint)." },
        ],
      },
      {
        id: "encoder", title: "Text encoder", scope: "serve", default: "auto",
        description: "For the distilled checkpoint, Auto resolves to CPU offload, the policy of the measured GB200 recipes.",
        showWhen: (s) => s.hw !== "rtx4090",
        options: [
          { id: "auto", label: "Auto" },
          { id: "offload", label: "CPU offload", showWhen: (s) => s.hw !== "rtx4090" },
        ],
      },
      {
        id: "cfg", title: "CFG parallelism", scope: "serve", default: "auto",
        description: "Auto uses CFG parallelism for even GPU counts with the base checkpoint; the distilled checkpoint always runs without it. TP and Ulysses are editable under Setup's advanced topology.",
        options: [
          { id: "auto", label: "Auto" },
          { id: "off", label: "Disabled" },
          { id: "on", label: "Enabled", disabled: (s) => distilled(s) || Number(s.gpus_per_node) % 2 !== 0,
            disableReason: "CFG parallelism requires an even GPU count and the base checkpoint; the distilled checkpoint rejects it." },
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
        showWhen: (s) => !distilled(s),
        default: 40, min: 1, max: 100, unit: "steps", options: [],
      },
      {
        id: "distilled_steps", title: "Denoising steps", scope: "request", kind: "number",
        showWhen: distilled,
        description: "10 is the distilled checkpoint's default; the request keeps guidance at 1.0, the only value the checkpoint accepts.",
        default: 10, min: 1, max: 100, unit: "steps", options: [],
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
          // Without a CFG split, Ulysses across the ranks is the distilled checkpoint's fastest measured layout.
          if (distilled(s)) return { tp_size: 1, ulysses_degree: ranks, ring_degree: 1 };
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
          if (distilled(s) && s.cfg === "on")
            errors.push("The distilled checkpoint runs one branch per step and does not support CFG parallelism.");
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
        const encoder = encoderOf(s);
        const recipe = recipes.find((r) => r.hw === s.hw && r.nodes === Number(s.nodes)
          && r.gpus_per_node === Number(s.gpus_per_node) && r.placement === placement
          && r.tp_size === topology.tp_size && r.ulysses_degree === topology.ulysses_degree
          && r.ring_degree === topology.ring_degree && cfgDegree(r) === cfgDegree(s));
        const encoderVerified = distilled(s) ? encoder === "offload" : s.encoder === "auto";
        const served = !!recipe && !recipe.unverified && recipe.verifiedWhen(s) && encoderVerified && errors.length === 0;
        const steps = Number(distilled(s) ? s.distilled_steps : s.steps);
        const requested = served && s.resolution === "640x800" && Number(s.clip_len) === 37
          && steps === (distilled(s) ? 10 : 40) && s.audio === "false";
        const flags = [
          distilled(s) ? "--model-path {{DISTILLED_MODEL_PATH}}" : "--model-path {{MODEL_NAME}}",
          "--port {{PORT}}", "--num-gpus " + s.gpus_per_node,
        ];
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
        if (encoder === "offload") flags.push("--text-encoder-cpu-offload");

        const warnings = [];
        if (!served && !errors.length) warnings.push("This exact deployment has not completed HTTP verification; see the benchmark for offline coverage.");
        if (served && !requested) warnings.push("These request settings are outside the verified HTTP workload.");
        if (s.hw === "rtx4090") warnings.push("24 GB recipe: memory mode plus an expandable allocator. The 24-frame benchmark uses nearly all VRAM; longer clips or larger pixel budgets may OOM. Offload also needs sufficient host RAM.");
        if (s.hw === "h100") warnings.push("Use DiT layerwise offload for the documented 80 GB memory preset. H100 recipes have not been measured in this cookbook.");
        if (distilled(s) && s.hw !== "gb200") warnings.push("The distilled checkpoint was measured on GB200 only; see the distilled benchmark for its memory peaks.");
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
              encoder: encoder === "offload" ? "CPU offload" : "Auto",
              cfg: cfgDegree(s) === 2 ? "Enabled" : "Disabled",
            },
          },
        };
      },
    },
    modelNames: { default: "Wan-AI/Wan2.2-Animate-2-14B-Diffusers" },
    placeholders: {
      DISTILLED_MODEL_PATH: { target: "command", label: "Distilled checkpoint repository or directory", default: "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers" },
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
      '--form-string "num_inference_steps=' + (distilled(s) ? s.distilled_steps : s.steps) + '"',
      '--form-string "guidance_scale=' + (distilled(s) ? "1.0" : "3.0") + '"',
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
