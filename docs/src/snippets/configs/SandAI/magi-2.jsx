export const config = (() => {
  const MODEL = "sand-ai/MAGI-2-preview";
  // A rank owns whole heads on every Ulysses axis, so num_gpus must divide each.
  const ULYSSES_AXES = [
    ["preview attention heads", 24],
    ["refiner attention heads", 32],
    ["refiner key/value heads", 8],
  ];
  const MOE_HEADS = 12;
  const REJECTED_AXES = {
    tp_size: "MAGI-2 has no tensor-parallel layers; put the whole degree in Ulysses.",
    ring_degree: "MAGI-2 exchanges heads over the Ulysses group only; ring degree must be 1.",
    cfg_degree: "MAGI-2 runs both CFG branches in its own denoise loop; CFG parallel is rejected.",
    dp_size: "Data parallelism (--dp-size > 1) is rejected at startup.",
  };
  // 8 does not divide the 12 MoE heads, so 8 ranks run two expert-parallel groups of 4.
  const epSizeOf = (gpus) => (gpus === 8 ? 4 : gpus);

  const extraFlagsOf = (gpus) => {
    if (gpus === 8)
      return ["--pipeline-config-path {{PIPELINE_CONFIG}}", "--performance-mode speed", "--enable-torch-compile false"];
    if (gpus === 2) return ["--performance-mode memory", "--dit-layerwise-offload", "--dit-offload-prefetch-size 1"];
    return ["--performance-mode speed"];
  };

  const warningsOf = (gpus) => {
    if (gpus === 8)
      return [
        'Write {"ep_size": 4} to the pipeline config file; 8 GPUs do not divide the 12 MoE heads.',
        "Runs eager: torch.compile regresses the preview pass at 8 ranks (335 s compiled vs 184 s eager).",
      ];
    if (gpus === 2)
      return [
        "Two GPUs need DiT layerwise offload; the expert weights do not fit without it.",
        "The measured 689 s clip used --enable-torch-compile --regional-compile.",
      ];
    if (gpus === 4) return ["The measured 348 s clip used --enable-torch-compile --regional-compile."];
    return [];
  };

  const plan = (s) => {
    const gpus = Number(s.gpus_per_node);
    const flags = ["--model-path " + MODEL, "--num-gpus " + gpus, "--ulysses-degree " + gpus, ...extraFlagsOf(gpus)];
    const topology = { tp_size: 1, ulysses_degree: gpus, ring_degree: 1 };
    const summary = "Ulysses " + gpus + " / EP " + epSizeOf(gpus);
    // Both DiTs stay resident through the 1080p decode, which needs one ~9 GiB block.
    const env = ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"];
    return { flags, gpus, topology, env, summary, warnings: warningsOf(gpus), errors: [] };
  };

  const headErrorsOf = (gpus) => {
    const errors = ULYSSES_AXES.filter(([, count]) => count % gpus).map(
      ([name, count]) => "num_gpus (" + gpus + ") must divide the " + count + " " + name + ".",
    );
    const ep = epSizeOf(gpus);
    if (gpus % ep || MOE_HEADS % ep)
      errors.push("ep_size (" + ep + ") must divide num_gpus and the " + MOE_HEADS + " MoE heads.");
    return errors;
  };

  const defaults = {
    hw: "h200",
    nodes: 1,
    gpus_per_node: 4,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 4,
    ring_degree: 1,
  };

  const config = {
    modelName: "MAGI-2-preview",
    supportedHardware: ["h200"],
    hardware: [{ id: "h200", label: "H200", vram: "143GB", vendor: "nvidia" }],
    groupHardware: false,
    matchDims: [],
    overlayDims: [],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
        verifiedRecipes: [4, 8, 2].map((gpus) => ({
          ...defaults,
          id: "h200-ulysses-" + gpus,
          gpus_per_node: gpus,
          ...plan({ ...defaults, gpus_per_node: gpus }).topology,
          default: gpus === 4,
          unverified: true,
        })),
        autoTopology: (s) => plan(s).topology,
        validateTopology: (s, t) => {
          const errors = [];
          if (!config.supportedHardware.includes(s.hw)) errors.push("Choose a listed hardware platform.");
          if (Number(s.nodes) !== 1) errors.push("This recipe covers one node.");
          const gpus = Number(s.gpus_per_node);
          if (!Number.isInteger(gpus) || gpus < 1 || gpus > 8) {
            errors.push("Choose an integer GPU count between 1 and 8.");
            return errors;
          }
          for (const [key, message] of Object.entries(REJECTED_AXES)) {
            if (Number(t[key] ?? 1) !== 1) errors.push(message);
          }
          if (Number(t.ulysses_degree) !== gpus) errors.push("Ulysses must span all " + gpus + " GPUs.");
          errors.push(...headErrorsOf(gpus));
          if (gpus === 1) errors.push("One GPU runs out of memory while loading weights.");
          return errors;
        },
      },
      resolveDeployment: (s) => {
        const p = plan(s);
        const topology =
          s.topology_mode === "manual"
            ? {
                tp_size: Number(s.tp_size),
                ulysses_degree: Number(s.ulysses_degree),
                ring_degree: Number(s.ring_degree),
              }
            : p.topology;
        const errors = config.commandBuilder.resource.validateTopology(s, topology);
        const status = errors.length ? "error" : "unverified";
        return {
          match: { hw: s.hw },
          nnodes: 1,
          verified: false,
          flags: [...p.flags, "--port {{PORT}}"],
          env: p.env,
          builder: {
            topology,
            topologySummary: p.summary,
            errors,
            warnings: [
              ...p.warnings,
              "Measured with sglang generate; this exact server and request combination has not been HTTP-verified.",
            ],
            verification: { serve: status, request: status },
          },
        };
      },
    },
    modelNames: { default: MODEL },
    placeholders: {
      PORT: { target: "command", label: "Server port", default: "30010" },
      PIPELINE_CONFIG: { target: "command", label: "Pipeline config JSON (8 GPUs)", default: "magi2-ep4.json" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Request port", default: "30010" },
    },
    curl: () =>
      [
        "curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos",
        "-H 'Content-Type: application/json'",
        "-d '" +
          JSON.stringify(
            {
              model: MODEL,
              prompt:
                "A cellist tunes up alone in a wood-panelled rehearsal room at dusk, then draws one long low note; dust turns in the last of the window light and the room's natural reverb settles around the sound.",
              seed: 1101,
            },
            null,
            2,
          ) +
          "'",
      ].join(" \\\n  "),
    runModes: ["python"],
    showPlaygroundLink: false,
    github: { cookbookModel: MODEL },
  };
  return config;
})();
