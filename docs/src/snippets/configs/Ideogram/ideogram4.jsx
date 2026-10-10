export const config = (() => {
  const modelOf = (s) =>
    ({
      nf4: "ideogram-ai/ideogram-4-nf4",
      fp8: "ideogram-ai/ideogram-4-fp8",
      nvfp4: "Comfy-Org/Ideogram-4",
      fast: "fal/ideogram-v4-fast",
      instant: "fal/ideogram-v4-instant",
    })[s.checkpoint || "nf4"];
  const plan = (s) => {
    const flags = ["--model-path " + modelOf(s), ...["--num-gpus 1", "--performance-mode auto", "--port {{PORT}}"]];
    let gpus = 1;
    let t = { tp_size: 1, ulysses_degree: 1, ring_degree: 1 };
    const env = [];
    const warnings = [];
    const errors = [];
    env.push("HF_TOKEN=$HF_TOKEN");
    if (s.checkpoint === "nvfp4" && s.hw !== "b200")
      errors.push("The documented NVFP4 recipe requires Blackwell; select B200.");
    if (s.checkpoint === "fast")
      warnings.push(
        "Floating-point Fast inference bypasses its intended NVFP4 path and may degrade quality; use Instant for local inference.",
      );
    return { flags, gpus, topology: t, env, warnings, errors };
  };
  const defaults = {
    hw: "cuda",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
    checkpoint: "nf4",
  };
  const config = {
    modelName: "Ideogram 4",
    supportedHardware: ["cuda", "b200"],
    hardware: [{ id: "cuda", label: "NVIDIA CUDA", vram: "", vendor: "nvidia" }],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "checkpoint",
        title: "Checkpoint",
        scope: "base",
        default: "nf4",
        options: [
          { id: "nf4", label: "ideogram-4-nf4" },
          { id: "fp8", label: "ideogram-4-fp8" },
          { id: "nvfp4", label: "Ideogram-4" },
          { id: "fast", label: "ideogram-v4-fast" },
          { id: "instant", label: "ideogram-v4-instant" },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 1 } },
        verifiedRecipes: ["cuda", "b200"].map((hw) => {
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
          if (!Number.isInteger(count) || count < 1 || count > 1)
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
      PORT: { target: "command", label: "Server port", default: "30010" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Request port", default: "30010" },
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
              ...{
                prompt: ["fast", "instant"].includes(s.checkpoint)
                  ? JSON.stringify({
                      high_level_description:
                        "A bold typographic poster centered on the exact words INSTANT BY FAL, printed in black and electric orange on warm white paper.",
                      compositional_deconstruction: {
                        background: "Warm white textured paper with generous negative space.",
                        elements: [
                          {
                            type: "text",
                            text: "INSTANT BY FAL",
                            desc: "Large uppercase geometric sans-serif lettering, precisely centered.",
                          },
                        ],
                      },
                    })
                  : "A cinematic poster of a quiet bookstore at dusk with elegant hand-lettered signage",
                size: "1024x1024",
                seed: 0,
                ...(!["fast", "instant"].includes(s.checkpoint) ? { preset: "V4_QUALITY_48" } : {}),
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
