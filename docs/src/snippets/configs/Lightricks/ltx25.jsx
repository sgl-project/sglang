export const config = (() => {
  const modelOf = () => "Lightricks/LTX-2.5-Diffusers";
  const plan = (s) => {
    const gpus = Number(s.gpus_per_node);
    const flags = [
      "--model-path " + modelOf(s),
      "--pipeline-class-name " + (s.pipeline === "two-stage" ? "LTX2TwoStagePipeline" : "LTX2Pipeline"),
    ];
    if (s.weights === "dev") flags.push("--model-variant dev");
    if (s.precision === "fp8") flags.push("--quantization fp8");
    if (s.placement === "offload") flags.push("--dit-layerwise-offload");
    if (s.decoder_weights === "loaded") flags.push("--load-diffusion-decoder");
    if (gpus === 2)
      flags.push(
        "--num-gpus 2",
        s.parallel === "tp" ? "--tp-size 2" : s.parallel === "cfg" ? "--enable-cfg-parallel" : "--ulysses-degree 2",
      );
    flags.push("--port {{PORT}}");
    return {
      flags,
      gpus,
      topology: {
        tp_size: gpus === 2 && s.parallel === "tp" ? 2 : 1,
        ulysses_degree: gpus === 2 && s.parallel === "sp" ? 2 : 1,
        ring_degree: 1,
      },
      warnings: [
        ...(s.parallel === "cfg" && gpus === 2 && s.weights !== "dev"
          ? ["Distilled weights run unguided; CFG parallelism does not accelerate them."]
          : []),
        ...(s.precision === "fp8" ? ["FP8 is approximate; evaluate output quality against BF16."] : []),
      ],
      errors:
        s.decoder === "diffusion" && s.decoder_weights !== "loaded"
          ? ["Load diffusion decoder weights in Server before requesting diffusion decode."]
          : [],
    };
  };
  const defaults = {
    hw: "h200",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
    weights: "distilled",
    pipeline: "one-stage",
    precision: "bf16",
    placement: "auto",
    parallel: "sp",
    decoder_weights: "off",
    decoder: "vae",
    duration: "fixed",
  };
  const config = {
    modelName: "LTX-2.5",
    supportedHardware: ["h200", "cuda"],
    hardware: [{ id: "cuda", label: "NVIDIA CUDA", vram: "", vendor: "nvidia" }],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "weights",
        title: "Weights",
        scope: "base",
        default: "distilled",
        options: [
          { id: "distilled", label: "Distilled" },
          { id: "dev", label: "Dev / SFT" },
        ],
      },
      {
        id: "pipeline",
        title: "Pipeline",
        scope: "serve",
        default: "one-stage",
        options: [
          { id: "one-stage", label: "One stage" },
          { id: "two-stage", label: "Two stage" },
        ],
      },
      {
        id: "precision",
        title: "Precision",
        scope: "serve",
        default: "bf16",
        options: [
          { id: "bf16", label: "BF16" },
          { id: "fp8", label: "FP8" },
        ],
      },
      {
        id: "placement",
        title: "Memory policy",
        scope: "serve",
        default: "auto",
        options: [
          { id: "auto", label: "Auto" },
          { id: "offload", label: "DiT layerwise offload" },
        ],
      },
      {
        id: "parallel",
        title: "Parallelism",
        scope: "serve",
        default: "sp",
        options: [
          { id: "sp", label: "Ulysses" },
          { id: "tp", label: "Tensor parallel" },
          { id: "cfg", label: "CFG parallel" },
        ],
      },
      {
        id: "decoder_weights",
        title: "Diffusion decoder weights",
        scope: "serve",
        default: "off",
        options: [
          { id: "off", label: "Not loaded" },
          { id: "loaded", label: "Loaded" },
        ],
      },
      {
        id: "decoder",
        title: "Decoder",
        scope: "request",
        default: "vae",
        options: [
          { id: "vae", label: "VAE" },
          { id: "diffusion", label: "Diffusion", showWhen: (s) => s.decoder_weights === "loaded" },
        ],
      },
      {
        id: "duration",
        title: "Duration",
        scope: "request",
        default: "fixed",
        options: [
          { id: "fixed", label: "Fixed" },
          { id: "auto", label: "Auto" },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 2 } },
        verifiedRecipes: ["h200", "cuda"].map((hw) => {
          const s = { ...defaults, hw };
          const p = plan(s);
          return {
            ...s,
            id: hw + "-documented",
            gpus_per_node: p.gpus,
            ...p.topology,
            default: true,
            unverified: true,
          };
        }),
        autoTopology: (s) => plan(s).topology,
        validateTopology: (s, t) => {
          const p = plan(s);
          const errors = [...(p.errors || [])];
          if (!config.supportedHardware.includes(s.hw)) errors.push("Choose a listed hardware platform.");
          if (Number(s.nodes) !== 1) errors.push("This recipe covers one node.");
          const count = Number(s.gpus_per_node);
          if (!Number.isInteger(count) || count < 1 || count > 2)
            errors.push("Choose an integer GPU count within the documented limits.");
          if (Number(s.gpus_per_node) !== p.gpus) errors.push("This recipe requires " + p.gpus + " GPUs / node.");
          if (["tp_size", "ulysses_degree", "ring_degree"].some((key) => Number(t[key]) !== p.topology[key]))
            errors.push("This topology is not a documented recipe for the selected model and hardware.");
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
        const flags = [...p.flags];
        if (!flags.some((flag) => flag.startsWith("--port "))) flags.push("--port {{PORT}}");
        const automaticSP =
          flags.find((flag) => flag.startsWith("--sp-degree ")) &&
          !flags.some((flag) => flag.startsWith("--ulysses-degree"));
        const summary = automaticSP
          ? ["TP " + topology.tp_size, "SP " + topology.ulysses_degree + " (runtime-selected exchange)"]
          : ["TP " + topology.tp_size, "Ulysses " + topology.ulysses_degree, "Ring " + topology.ring_degree];
        if (flags.includes("--enable-cfg-parallel")) summary.push("CFG 2");
        return {
          match: { hw: s.hw },
          nnodes: 1,
          verified: false,
          flags,
          env: p.env || [],
          builder: {
            topology,
            topologySummary: p.summary || summary.join(" / "),
            errors,
            warnings: [
              ...(p.warnings || []),
              "Documented recipe; this exact server and request combination has not been HTTP-verified.",
            ],
            verification: { serve: status, request: status },
          },
        };
      },
    },
    modelNames: { default: modelOf(defaults) },
    placeholders: {
      PORT: { target: "command", label: "Server port", default: "30000" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Request port", default: "30000" },
    },
    curl: (s) =>
      [
        "curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos",
        "-H 'Content-Type: application/json'",
        "-d '" +
          JSON.stringify(
            {
              model: modelOf(s),
              prompt: "A quiet street at dusk",
              ...{
                size: s.pipeline === "two-stage" ? "1920x1088" : "960x544",
                fps: 24,
                ...(s.weights === "dev" ? { num_inference_steps: 30, guidance_scale: 3.0 } : {}),
                ...(s.duration === "auto" ? { auto_duration: true } : { num_frames: 121 }),
                ...(s.decoder === "diffusion" ? { use_diffusion_decoder: true } : {}),
              },
            },
            null,
            2,
          ) +
          "'",
      ].join(" \\\n  "),
    runModes: ["python"],
    showPlaygroundLink: false,
    github: { cookbookModel: modelOf(defaults) },
  };
  return config;
})();
