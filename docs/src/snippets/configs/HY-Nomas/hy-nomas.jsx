export const config = (() => {
  // Real-weight H200 runs of sglang serve: HTTP /v1/videos and the realtime WebSocket
  // on one GPU, Ulysses SP=2 and DiT layerwise offload, all pixel-identical to one GPU.
  const recipes = [
    [1, 1, "resident"],
    [2, 2, "resident"],
    [1, 1, "offload"],
  ].map(([gpus, ulysses, placement]) => ({
    id: ["h200", gpus, placement, "sp" + ulysses].join("-"),
    hw: "h200",
    nodes: 1,
    gpus_per_node: gpus,
    placement,
    tp_size: 1,
    ulysses_degree: ulysses,
    ring_degree: 1,
    default: gpus === 1 && placement === "resident",
    unverified: false,
  }));

  const config = {
    modelName: "HY-Nomas",
    supportedHardware: ["h200"],
    hardware: [],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "placement",
        title: "Memory policy",
        scope: "serve",
        default: "resident",
        description: "Layerwise offload streams the DiT blocks from host memory; the small camera modules stay resident.",
        learnMore: "#3-deployment",
        options: [
          { id: "resident", label: "Resident" },
          { id: "offload", label: "DiT layerwise offload" },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: {
        hw: "h200",
        nodes: 1,
        gpus_per_node: 1,
        topology_mode: "auto",
        tp_size: 1,
        ulysses_degree: 1,
        ring_degree: 1,
      },
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
        verifiedRecipes: recipes,
        // Ulysses SP is the measured multi-GPU layout; TP=2 runs but is not output-equivalent.
        autoTopology: (s) => ({ tp_size: 1, ulysses_degree: Number(s.gpus_per_node), ring_degree: 1 }),
        validateTopology: (s, t) => {
          const errors = [];
          const gpus = Number(s.gpus_per_node);
          if (!config.supportedHardware.includes(s.hw)) errors.push("Choose a listed hardware platform.");
          if (Number(s.nodes) !== 1) errors.push("This recipe covers one node.");
          if (!Number.isInteger(gpus) || gpus < 1 || gpus > 8) errors.push("Choose 1-8 GPUs per node.");
          if (![t.tp_size, t.ulysses_degree, t.ring_degree].every((n) => Number.isInteger(n) && n >= 1))
            errors.push("Parallel degrees must be positive integers.");
          if (t.ring_degree !== 1) errors.push("HY-Nomas does not support Ring attention; keep Ring at 1.");
          if (gpus !== t.tp_size * t.ulysses_degree * t.ring_degree)
            errors.push("GPU count must equal TP times Ulysses.");
          if (16 % (t.tp_size * t.ulysses_degree))
            errors.push("TP times Ulysses must divide the 16 DiT attention heads.");
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
        const recipe = recipes.find(
          (r) =>
            r.hw === s.hw &&
            r.gpus_per_node === Number(s.gpus_per_node) &&
            r.placement === s.placement &&
            r.tp_size === topology.tp_size &&
            r.ulysses_degree === topology.ulysses_degree,
        );
        const served = !!recipe && errors.length === 0;
        const flags = ["--model-path {{MODEL_NAME}}", "--port {{PORT}}", "--num-gpus " + s.gpus_per_node];
        if (topology.tp_size > 1) flags.push("--tp-size " + topology.tp_size);
        if (topology.ulysses_degree > 1) flags.push("--sp-degree " + topology.ulysses_degree);
        if (s.placement === "offload") flags.push("--dit-layerwise-offload true");

        const warnings = [];
        if (topology.tp_size > 1)
          warnings.push("TP completes requests, but trained-weight output differs from one GPU; prefer Ulysses SP.");
        if (!served && !errors.length)
          warnings.push("This exact deployment has not been run with trained weights.");
        const verification = errors.length ? "error" : served ? "verified" : "unverified";
        return {
          match: { hw: s.hw },
          nnodes: 1,
          verified: served,
          flags,
          env: [],
          builder: {
            topology,
            topologySummary: [
              "TP " + topology.tp_size,
              "Ulysses " + topology.ulysses_degree,
              s.placement === "offload" ? "DiT layerwise offload" : "Resident",
            ].join(" / "),
            errors,
            warnings,
            verification: { serve: verification, request: verification },
          },
        };
      },
    },
    modelNames: { default: "tencent/Hy-Nomas" },
    placeholders: {
      PORT: { target: "command", label: "Server port", default: "30000" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Request port", default: "30000" },
    },
    curl: [
      "curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos",
      "-H 'Content-Type: application/json'",
      "--data-binary @request.json",
    ].join(" \\\n  "),
    runModes: ["python"],
    showPlaygroundLink: false,
    github: { cookbookModel: "tencent/Hy-Nomas" },
  };
  return config;
})();
