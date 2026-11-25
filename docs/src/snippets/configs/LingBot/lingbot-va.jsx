export const config = (() => {
  const ids = {
    libero: "robbyant/lingbot-va-posttrain-libero-long",
    robotwin: "robbyant/lingbot-va-posttrain-robotwin",
  };
  const topology = { tp_size: 1, ulysses_degree: 1, ring_degree: 1 };
  const defaults = { hw: "h200", nodes: 1, gpus_per_node: 1, topology_mode: "auto", checkpoint: "libero", ...topology };
  function validate(s, t) {
    return s.hw === "h200" && Number(s.nodes) === 1 && Number(s.gpus_per_node) === 1 &&
      Object.keys(topology).every((key) => Number(t[key]) === 1)
      ? [] : ["Choose one H200 and a single-rank topology."];
  }
  return {
    modelName: "LingBot-VA",
    supportedHardware: ["h200"],
    groupHardware: false,
    matchDims: [],
    overlayDims: [{
      id: "checkpoint", title: "Policy checkpoint", scope: "base", default: "libero",
      options: [{ id: "libero", label: "LIBERO-Long" }, { id: "robotwin", label: "RoboTwin" }],
    }],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 1 } },
        verifiedRecipes: [{ ...defaults, id: "h200", default: true, checkpoints: Object.keys(ids) }],
        autoTopology: () => topology,
        validateTopology: validate,
      },
      resolveDeployment: (s) => {
        const t = s.topology_mode === "manual" ? s : topology;
        const errors = validate(s, t);
        const status = errors.length ? "error" : "verified";
        return { match: { hw: s.hw }, nnodes: 1, verified: !errors.length,
          flags: ["--model-path " + ids[s.checkpoint || "libero"], "--backend sglang", "--warmup-mode off", "--port {{PORT}}"],
          builder: { topology, topologySummary: "One policy worker", errors, warnings: [],
            verification: { serve: status, request: status } } };
      },
    },
    modelNames: { default: ids.libero },
    placeholders: {
      PORT: { target: "command", label: "Server port", default: "30000" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Request port", default: "30000" },
    },
    curl: "curl -fsS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/actions/metadata",
    runModes: ["python"], showPlaygroundLink: false,
  };
})();
