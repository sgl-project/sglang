export const config = (() => {
  const modelOf = (s) =>
    ({
      nano: "nvidia/Cosmos3-Nano",
      super: "nvidia/Cosmos3-Super",
      t2i: "nvidia/Cosmos3-Super-Text2Image",
      i2v: "nvidia/Cosmos3-Super-Image2Video",
      droid: "nvidia/Cosmos3-Nano-Policy-DROID",
      edge: "nvidia/Cosmos3-Edge",
      edge_droid: "nvidia/Cosmos3-Edge-Policy-DROID",
      t2i4: "nvidia/Cosmos3-Super-Text2Image-4Step",
      i2v4: "nvidia/Cosmos3-Super-Image2Video-4Step",
    })[s.checkpoint || "nano"];
  const plan = (s) => {
    const flags = ["--model-path " + modelOf(s), ...[]];
    let gpus = 1;
    let t = { tp_size: 1, ulysses_degree: 1, ring_degree: 1 };
    const env = [];
    const warnings = [];
    const errors = [];
    gpus = s.checkpoint.startsWith("super") || ["t2i", "i2v", "t2i4", "i2v4"].includes(s.checkpoint) ? 4 : 1;
    t.ulysses_degree = gpus;
    flags.push("--num-gpus " + gpus);
    return { flags, gpus, topology: t, env, warnings, errors, summary: gpus + " GPUs / runtime auto parallelism" };
  };
  const defaults = {
    hw: "cuda",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
    checkpoint: "nano",
  };
  const config = {
    modelName: "Cosmos3",
    supportedHardware: ["cuda"],
    hardware: [{ id: "cuda", label: "NVIDIA CUDA", vram: "", vendor: "nvidia" }],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "checkpoint",
        title: "Checkpoint",
        scope: "base",
        default: "nano",
        options: [
          { id: "nano", label: "Cosmos3-Nano" },
          { id: "super", label: "Cosmos3-Super" },
          { id: "t2i", label: "Cosmos3-Super-Text2Image" },
          { id: "i2v", label: "Cosmos3-Super-Image2Video" },
          { id: "droid", label: "Cosmos3-Nano-Policy-DROID" },
          { id: "edge", label: "Cosmos3-Edge" },
          { id: "edge_droid", label: "Cosmos3-Edge-Policy-DROID" },
          { id: "t2i4", label: "Cosmos3-Super-Text2Image-4Step" },
          { id: "i2v4", label: "Cosmos3-Super-Image2Video-4Step" },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 4 } },
        verifiedRecipes: ["cuda"].map((hw) => {
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
    curl: "curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/models",
    runModes: ["python"],
    showPlaygroundLink: false,
    github: { cookbookModel: modelOf(defaults) },
  };
  return config;
})();
