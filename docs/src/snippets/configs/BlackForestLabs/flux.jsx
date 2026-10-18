export const config = (() => {
  const modelOf = (s) =>
    "black-forest-labs/" + (s.version === "flux2-dev" || s.hw === "arc_b" ? "FLUX.2-dev" : "FLUX.1-dev");
  const plan = (s) => {
    const flags = ["--model-path " + modelOf(s)];
    let gpus = 1;
    let tp = 1;
    if (s.hw === "a3" || (s.hw === "a2" && s.version === "flux2-dev")) {
      gpus = 2;
      tp = 2;
      flags.push("--tp-size 2", "--num-gpus 2");
    } else if (s.hw === "a2") flags.push("--num-gpus 1");
    else if (s.hw === "arc_b") {
      gpus = 4;
      tp = 4;
      flags.push(
        "--num-gpus 4",
        "--tp-size 4",
        "--component-residency dit=resident,text_encoder=layerwise-offload",
        "--dit-cpu-offload False",
      );
    } else flags.push("--ulysses-degree=1", "--ring-degree=1");
    return { flags, gpus, topology: { tp_size: tp, ulysses_degree: 1, ring_degree: 1 } };
  };
  const defaults = {
    hw: "b200",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
    version: "flux1-dev",
  };
  const config = {
    modelName: "FLUX",
    supportedHardware: ["b200", "b300", "h200", "h100", "mi300x", "mi325x", "mi355x", "a2", "a3", "arc_b"],
    hardware: [
      { id: "a2", label: "Ascend A2", vram: "", vendor: "npu" },
      { id: "a3", label: "Ascend A3", vram: "2 NPUs / card", vendor: "npu" },
      { id: "arc_b", label: "Intel BMG", vram: "", vendor: "intel" },
    ],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "version",
        title: "Checkpoint",
        scope: "base",
        default: "flux1-dev",
        options: [
          { id: "flux1-dev", label: "FLUX.1-dev", showWhen: (s) => s.hw !== "arc_b" },
          { id: "flux2-dev", label: "FLUX.2-dev" },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
        verifiedRecipes: ["b200", "b300", "h200", "h100", "mi300x", "mi325x", "mi355x", "a2", "a3", "arc_b"].map(
          (hw) => {
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
          },
        ),
        autoTopology: (s) => plan(s).topology,
        validateTopology: (s, t) => {
          const p = plan(s);
          const errors = [...(p.errors || [])];
          if (!config.supportedHardware.includes(s.hw)) errors.push("Choose a listed hardware platform.");
          if (Number(s.nodes) !== 1) errors.push("This recipe covers one node.");
          const count = Number(s.gpus_per_node);
          if (!Number.isInteger(count) || count < 1 || count > 8)
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
              ...{},
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
