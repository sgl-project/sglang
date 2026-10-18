export const config = (() => {
  const modelOf = (s) =>
    ({
      "t2v-14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
      "t2v-1_3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
      "i2v-14b": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    })[s.variant];
  const plan = (s) => {
    const small = !s.variant.endsWith("14b");
    const ascend = ["a2", "a3"].includes(s.hw);
    const fast = ascend ? Number(s.gpus_per_node) === 8 : Number(s.gpus_per_node) > 1;
    let gpus = 1,
      tp = 1,
      sp = 1;
    const flags = ["--model-path " + modelOf(s)];
    if (ascend) {
      if (small && s.hw === "a2" && !fast) flags.push("--num-gpus 1");
      else {
        tp = small ? (fast ? 4 : 1) : 2;
        sp = !small && fast ? 4 : 1;
        gpus = fast ? 8 : s.hw === "a3" ? 2 : 4;
        flags.push("--tp-size " + tp, "--sp-degree " + sp, "--num-gpus " + gpus);
      }
      if (fast) flags.push("--attention-backend laser_attn");
    } else {
      flags.push("--dit-layerwise-offload true");
      if (fast) {
        gpus = 4;
        sp = 2;
        flags.push("--num-gpus 4", "--ulysses-degree 2", "--enable-cfg-parallel");
      }
    }
    const lora = {
      "t2v-14b": "NIVEDAN/wan2.1-lora",
      "i2v-14b": "valiantcat/Wan2.1-Fight-LoRA",
    }[s.variant];
    if (s.lora === s.variant && lora) flags.push("--lora-path " + lora);
    return {
      flags,
      gpus,
      topology: { tp_size: tp, ulysses_degree: sp, ring_degree: 1 },
      warnings: ascend ? ["Ascend recipes count NPU chips; one A3 card has two chips."] : [],
    };
  };
  const defaults = {
    hw: "b200",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
    variant: "t2v-14b",
    lora: "t2v-14b",
  };
  const config = {
    modelName: "Wan2.1",
    supportedHardware: ["b200", "b300", "h200", "h100", "mi300x", "mi325x", "mi355x", "a2", "a3"],
    hardware: [
      { id: "a2", label: "Ascend A2", vram: "", vendor: "npu" },
      { id: "a3", label: "Ascend A3", vram: "2 NPUs / card", vendor: "npu" },
    ],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "variant",
        title: "Checkpoint / task",
        scope: "base",
        default: "t2v-14b",
        options: [
          { id: "t2v-14b", label: "T2V 14B" },
          { id: "t2v-1_3b", label: "T2V 1.3B" },
          { id: "i2v-14b", label: "I2V 14B" },
        ],
      },
      {
        id: "lora",
        title: "LoRA",
        scope: "serve",
        default: "t2v-14b",
        options: [
          { id: "none", label: "None" },
          {
            id: "t2v-14b",
            label: "NIVEDAN/wan2.1-lora",
            showWhen: function anonymous(s) {
              return s.variant === "t2v-14b";
            },
          },
          {
            id: "i2v-14b",
            label: "valiantcat/Wan2.1-Fight-LoRA",
            showWhen: function anonymous(s) {
              return s.variant === "i2v-14b";
            },
          },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 8 } },
        verifiedRecipes: ["b200", "b300", "h200", "h100", "mi300x", "mi325x", "mi355x", "a2", "a3"].map((hw) => {
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
      INPUT_IMAGE: { target: "curl", label: "Image path on the server", default: "/path/to/input.png" },
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
              ...(s.variant.startsWith("i2v") ? { input_reference: "{{INPUT_IMAGE}}" } : {}),
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
