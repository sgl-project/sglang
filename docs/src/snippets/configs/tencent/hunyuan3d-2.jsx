export const config = (() => {
  const MODEL = "tencent/Hunyuan3D-2";
  // Exact E2E runs (serve + /v1/meshes) per hardware and output mode.
  const VERIFIED = [
    { hw: "h200", output: "textured" },
    { hw: "h200", output: "shape" },
  ];
  const defaults = {
    hw: "h200",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
  };
  const config = {
    modelName: "Hunyuan3D-2",
    supportedHardware: ["h200", "h100", "b200"],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "output",
        title: "Output",
        scope: "base",
        default: "textured",
        description: "Textured output returns a GLB; shape-only skips the delight and paint models and returns an untextured OBJ.",
        options: [
          { id: "textured", label: "Textured mesh (shape + paint)", recommended: true },
          {
            id: "shape",
            label: "Shape only",
            flags: ["--pipeline-config-path shape_only.json"],
            hints: ["Create shape_only.json first: echo '{\"paint_enable\": false}' > shape_only.json"],
          },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 1 } },
        verifiedRecipes: [{ ...defaults, id: "h200-1gpu", default: true }],
        autoTopology: () => ({ tp_size: 1, ulysses_degree: 1, ring_degree: 1 }),
        validateTopology: (s, t) => {
          const errors = [];
          if (Number(s.nodes) !== 1 || Number(s.gpus_per_node) !== 1)
            errors.push("Hunyuan3D-2 is served on one GPU.");
          if (t.tp_size !== 1 || t.ulysses_degree !== 1 || t.ring_degree !== 1)
            errors.push("Hunyuan3D-2 does not support tensor or sequence parallelism.");
          return errors;
        },
      },
      resolveDeployment: (s) => {
        const topology =
          s.topology_mode === "manual"
            ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
            : { tp_size: 1, ulysses_degree: 1, ring_degree: 1 };
        const errors = config.commandBuilder.resource.validateTopology(s, topology);
        const verified = !errors.length && VERIFIED.some((r) => r.hw === s.hw && r.output === s.output);
        const status = errors.length ? "error" : verified ? "verified" : "unverified";
        return {
          match: { hw: s.hw },
          nnodes: 1,
          verified,
          flags: [`--model-path ${MODEL}`, "--host {{HOST_IP}}", "--port {{PORT}}"],
          builder: {
            topology,
            topologySummary: "1 GPU",
            errors,
            warnings: verified || errors.length ? [] : ["This hardware has not been run end to end."],
            verification: { serve: status, request: status },
          },
        };
      },
    },
    modelNames: { default: MODEL },
    placeholders: {
      HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
      PORT: { target: "command", label: "Bind port", default: "30000" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
      INPUT_IMAGE: { target: "curl", label: "Input image on the client", default: "input.png" },
    },
    curl: (s) =>
      [
        "BASE=http://{{CURL_HOST}}:{{CURL_PORT}}/v1/meshes",
        'ID=$(curl -sS --fail-with-body $BASE -F "image=@{{INPUT_IMAGE}}" --form-string seed=0 | python3 -c "import json,sys; print(json.load(sys.stdin)[\'id\'])")',
        'until [ "$(curl -sS $BASE/$ID | python3 -c "import json,sys; print(json.load(sys.stdin)[\'status\'])")" = completed ]; do sleep 2; done',
        `curl -sS --fail-with-body $BASE/$ID/content -o output.${s.output === "shape" ? "obj" : "glb"}`,
      ].join("\n"),
    runModes: () => ["python"],
    showPlaygroundLink: false,
    cells: [],
  };
  return config;
})();
