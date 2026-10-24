export const config = (() => {
  const modelOf = () => "stabilityai/stable-diffusion-xl-base-1.0";
  const plan = () => ({
    flags: ["--model-path " + modelOf()],
    gpus: 1,
    topology: { tp_size: 1, ulysses_degree: 1, ring_degree: 1 },
  });
  const defaults = {
    hw: "arc_b",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
  };
  const config = {
    modelName: "Stable Diffusion XL",
    supportedHardware: ["arc_b", "a100"],
    hardware: [
      { id: "arc_b", label: "Intel Arc Pro B60", vram: "24GB", vendor: "intel" },
      { id: "a100", label: "A100", vram: "40GB", vendor: "nvidia" },
    ],
    groupHardware: false,
    matchDims: [],
    overlayDims: [],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 1 } },
        // arc_b: CI server test sdxl_base_image_t2i; a100: serve + curl on A100-SXM4-40GB.
        verifiedRecipes: ["arc_b", "a100"].map((hw) => ({
          ...defaults,
          hw,
          id: hw + "-verified",
          ...plan().topology,
          default: true,
        })),
        autoTopology: () => plan().topology,
        validateTopology: (s, t) => {
          const p = plan();
          const errors = [];
          if (!config.supportedHardware.includes(s.hw)) errors.push("Choose a listed hardware platform.");
          if (Number(s.nodes) !== 1) errors.push("This recipe covers one node.");
          if (Number(s.gpus_per_node) !== p.gpus) errors.push("This recipe requires " + p.gpus + " GPU / node.");
          if (["tp_size", "ulysses_degree", "ring_degree"].some((key) => Number(t[key]) !== p.topology[key]))
            errors.push("This topology is not a documented recipe for the selected model and hardware.");
          return errors;
        },
      },
      resolveDeployment: (s) => {
        const p = plan();
        const topology =
          s.topology_mode === "manual"
            ? {
                tp_size: Number(s.tp_size),
                ulysses_degree: Number(s.ulysses_degree),
                ring_degree: Number(s.ring_degree),
              }
            : p.topology;
        const errors = config.commandBuilder.resource.validateTopology(s, topology);
        const verified = errors.length === 0;
        const status = verified ? "verified" : "error";
        return {
          match: { hw: s.hw },
          nnodes: 1,
          verified,
          flags: [...p.flags, "--port {{PORT}}"],
          env: [],
          builder: {
            topology,
            topologySummary: "TP 1 / Ulysses 1 / Ring 1",
            errors,
            warnings: [],
            verification: { serve: status, request: status },
          },
        };
      },
    },
    modelNames: { default: modelOf() },
    placeholders: {
      PORT: { target: "command", label: "Server port", default: "30000" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Request port", default: "30000" },
    },
    curl: () =>
      [
        "curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations",
        "-H 'Content-Type: application/json'",
        "-d '" +
          JSON.stringify(
            {
              model: modelOf(),
              prompt: "A beautiful sunset over the ocean, photorealistic",
              n: 1,
              response_format: "b64_json",
            },
            null,
            2,
          ) +
          "'",
      ].join(" \\\n  "),
    runModes: ["python"],
    showPlaygroundLink: false,
    github: { cookbookModel: modelOf() },
  };
  return config;
})();
