export const config = (() => {
  const modelOf = (s) =>
    ({
      turbo: "krea/Krea-2-Turbo",
      raw: "krea/Krea-2-Raw",
    })[s.checkpoint || "turbo"];
  const plan = (s) => {
    const flags = ["--model-path " + modelOf(s), ...["--port {{PORT}}"]];
    let gpus = 1;
    let t = { tp_size: 1, ulysses_degree: 1, ring_degree: 1 };
    const env = [];
    const warnings = [];
    const errors = [];
    gpus = Number(s.gpus_per_node);
    flags.splice(1, 0, "--num-gpus " + gpus);
    if (gpus === 2) {
      t = { tp_size: s.parallel === "tp" ? 2 : 1, ulysses_degree: s.parallel === "sp" ? 2 : 1, ring_degree: 1 };
      flags.push(s.parallel === "tp" ? "--tp-size 2" : "--ulysses-degree 2");
    }
    if (gpus === 4) {
      t = { tp_size: 2, ulysses_degree: 2, ring_degree: 1 };
      flags.push("--tp-size 2", "--ulysses-degree 2");
    }
    if (![1, 2, 4].includes(gpus)) errors.push("Choose 1, 2 or 4 GPUs.");
    return { flags, gpus, topology: t, env, warnings, errors };
  };
  const defaults = {
    hw: "h200",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
    checkpoint: "turbo",
    parallel: "sp",
  };
  const config = {
    modelName: "Krea-2",
    supportedHardware: ["h200", "cuda"],
    hardware: [{ id: "cuda", label: "NVIDIA CUDA", vram: "", vendor: "nvidia" }],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "checkpoint",
        title: "Checkpoint",
        scope: "base",
        default: "turbo",
        options: [
          { id: "turbo", label: "Krea-2-Turbo" },
          { id: "raw", label: "Krea-2-Raw" },
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
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 4 } },
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
        "curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations",
        "-H 'Content-Type: application/json'",
        "-d '" +
          JSON.stringify(
            {
              model: modelOf(s),
              prompt: "A quiet lakeside cabin at sunrise",
              n: 1,
              response_format: "b64_json",
              ...{ size: "1024x1024" },
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
