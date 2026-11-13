export const config = (() => {
  const modelOf = (s) =>
    ({
      dense: "Efficient-Large-Model/SANA-WM_bidirectional",
      streaming: "Efficient-Large-Model/SANA-WM_streaming",
    })[s.checkpoint || "dense"];
  const plan = (s) => {
    const flags = ["--model-path " + modelOf(s), ...["--host 127.0.0.1", "--port {{PORT}}"]];
    let gpus = 1;
    let t = { tp_size: 1, ulysses_degree: 1, ring_degree: 1 };
    const env = [];
    const warnings = [];
    const errors = [];
    flags.splice(
      1,
      0,
      "--pipeline-class-name " + (s.mode === "realtime" ? "SanaWMRealtimePipeline" : "SanaWMTwoStagePipeline"),
    );
    if (s.checkpoint === "streaming" && s.mode !== "realtime") flags.push("--streaming", "--refiner-chunked");
    if (s.mode === "realtime" && s.checkpoint !== "streaming")
      errors.push("Realtime requires the streaming checkpoint.");
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
    checkpoint: "dense",
    mode: "batch",
  };
  const config = {
    modelName: "SANA-WM",
    supportedHardware: ["cuda"],
    hardware: [{ id: "cuda", label: "NVIDIA CUDA", vram: "", vendor: "nvidia" }],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "checkpoint",
        title: "Checkpoint",
        scope: "base",
        default: "dense",
        options: [
          { id: "dense", label: "SANA-WM_bidirectional" },
          { id: "streaming", label: "SANA-WM_streaming" },
        ],
      },
      {
        id: "mode",
        title: "Serving mode",
        scope: "serve",
        default: "batch",
        options: [
          { id: "batch", label: "Batch" },
          { id: "realtime", label: "Realtime", showWhen: (s) => s.checkpoint === "streaming" },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 1 } },
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
      PORT: { target: "command", label: "Server port", default: "30000" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Request port", default: "30000" },
    },
    curl: (s) =>
      s.mode === "realtime"
        ? "curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/models"
        : ((s) =>
            [
              "curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos",
              "-H 'Content-Type: application/json'",
              "-d '" +
                JSON.stringify(
                  {
                    prompt: "a camera moving forward and turning left",
                    input_reference: "/path/to/first_frame.png",
                    num_frames: 321,
                    seed: 42,
                    fps: 16,
                    ...(s.checkpoint === "dense" ? { num_inference_steps: 60, guidance_scale: 5.0 } : {}),
                    diffusers_kwargs: { action: "w-80,wl-80,l-80,wj-80", intrinsics: "/path/to/intrinsics.npy" },
                  },
                  null,
                  2,
                ) +
                "'",
            ].join(" \\\n  "))(s),
    runModes: ["python"],
    showPlaygroundLink: false,
    github: { cookbookModel: modelOf(defaults) },
  };
  return config;
})();
