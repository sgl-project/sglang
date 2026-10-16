export const config = (() => {
  const modelOf = (s) => "Lightricks/" + (s.model === "ltx2" ? "LTX-2" : "LTX-2.3");
  const plan = (s) => {
    const gpus = Number(s.gpus_per_node);
    const tp = gpus === 4 ? 2 : 1;
    const flags = [
      "--model-path " + modelOf(s),
      "--pipeline-class-name " +
        { "one-stage": "LTX2Pipeline", "two-stage": "LTX2TwoStagePipeline", "two-stage-hq": "LTX2TwoStageHQPipeline" }[
          s.pipeline
        ],
    ];
    if (gpus === 2) flags.push("--num-gpus 2", "--enable-cfg-parallel");
    if (gpus === 4) flags.push("--num-gpus 4", "--tp-size 2", "--enable-cfg-parallel");
    if (s.model === "ltx23" && s.pipeline !== "one-stage") flags.push("--ltx2-two-stage-device-mode " + s.device);
    if (s.lora === "transition" && s.model === "ltx23" && s.pipeline !== "one-stage")
      flags.push("--lora-path valiantcat/LTX-2.3-Transition-LORA", "--lora-weight-name ltx2.3-transition.safetensors");
    flags.push("--port {{PORT}}");
    return {
      flags,
      gpus,
      topology: { tp_size: tp, ulysses_degree: 1, ring_degree: 1 },
      errors: [
        ...(![1, 2, 4].includes(gpus) ? ["Choose 1, 2 or 4 GPUs."] : []),
        ...(s.model === "ltx2" && s.pipeline === "two-stage-hq" ? ["HQ requires LTX-2.3."] : []),
      ],
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
    model: "ltx23",
    pipeline: "two-stage",
    device: "resident",
    lora: "none",
  };
  const config = {
    modelName: "LTX-2 / LTX-2.3",
    supportedHardware: ["h200", "cuda"],
    hardware: [{ id: "cuda", label: "NVIDIA CUDA", vram: "", vendor: "nvidia" }],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "model",
        title: "Checkpoint",
        scope: "base",
        default: "ltx23",
        options: [
          { id: "ltx23", label: "LTX-2.3" },
          { id: "ltx2", label: "LTX-2" },
        ],
      },
      {
        id: "pipeline",
        title: "Pipeline",
        scope: "serve",
        default: "two-stage",
        options: [
          { id: "two-stage", label: "Two stage" },
          { id: "two-stage-hq", label: "Two stage HQ", showWhen: (s) => s.model === "ltx23" },
          { id: "one-stage", label: "One stage" },
        ],
      },
      {
        id: "device",
        title: "Device mode",
        scope: "serve",
        default: "resident",
        options: [
          { id: "resident", label: "Resident" },
          { id: "original", label: "Original" },
        ],
      },
      {
        id: "lora",
        title: "LoRA",
        scope: "serve",
        default: "none",
        options: [
          { id: "none", label: "None" },
          {
            id: "transition",
            label: "Transition LoRA",
            showWhen: (s) => s.model === "ltx23" && s.pipeline !== "one-stage",
          },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 4 } },
        verifiedRecipes: ["h200", "cuda"].map((hw) => {
          const s = { ...defaults, hw, device: hw === "cuda" ? "original" : "resident" };
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
          if (!Number.isInteger(count) || count < 1 || count > 4)
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
        "-d '" + JSON.stringify({ model: modelOf(s), prompt: "A quiet street at dusk", ...{} }, null, 2) + "'",
      ].join(" \\\n  "),
    runModes: ["python"],
    showPlaygroundLink: false,
    github: { cookbookModel: modelOf(defaults) },
  };
  return config;
})();
