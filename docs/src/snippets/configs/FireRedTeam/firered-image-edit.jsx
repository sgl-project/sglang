export const config = (() => {
  const modelOf = (s) => `FireRedTeam/FireRed-Image-Edit-${s.version}`;
  // Exact E2E runs (serve + /v1/images/edits) on this hardware and topology.
  const VERIFIED = [
    { hw: "h200", version: "1.1", gpus: 1, cfg: false, tp_size: 1, ulysses_degree: 1 },
    { hw: "h200", version: "1.1", gpus: 2, cfg: true, tp_size: 1, ulysses_degree: 1 },
    { hw: "h200", version: "1.1", gpus: 2, cfg: false, tp_size: 1, ulysses_degree: 2 },
    { hw: "h200", version: "1.0", gpus: 1, cfg: false, tp_size: 1, ulysses_degree: 1 },
    { hw: "h200", version: "1.0", gpus: 2, cfg: true, tp_size: 1, ulysses_degree: 1 },
  ];
  const topologyOf = (s) =>
    s.topology_mode === "manual"
      ? { tp_size: Number(s.tp_size), ulysses_degree: Number(s.ulysses_degree), ring_degree: Number(s.ring_degree) }
      : { tp_size: 1, ulysses_degree: 1, ring_degree: 1 };
  // The second GPU runs the unconditional CFG branch unless the manual topology already uses it.
  const cfgOf = (s, t) => Number(s.gpus_per_node) === 2 * t.tp_size * t.ulysses_degree * t.ring_degree;
  const defaults = {
    hw: "h200",
    nodes: 1,
    gpus_per_node: 2,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
    version: "1.1",
  };
  const config = {
    modelName: "FireRed-Image-Edit",
    supportedHardware: ["h200", "h100", "b200"],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "version",
        title: "Checkpoint",
        scope: "base",
        default: "1.1",
        description: "1.1 improves identity consistency and multi-image conditioning; both use the same pipeline.",
        options: [
          { id: "1.1", label: "FireRed-Image-Edit-1.1", recommended: true },
          { id: "1.0", label: "FireRed-Image-Edit-1.0" },
        ],
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: { nodes: { min: 1, max: 1 }, gpus_per_node: { min: 1, max: 2 } },
        // One card per topology; the 1.0 runs are matched in resolveDeployment.
        verifiedRecipes: VERIFIED.filter((r) => r.version === defaults.version && (r.gpus === 1 || r.cfg)).map((r) => ({
          ...defaults,
          id: `${r.hw}-${r.version}-${r.gpus}gpu${r.cfg ? "-cfg" : ""}`,
          version: r.version,
          hw: r.hw,
          gpus_per_node: r.gpus,
          tp_size: r.tp_size,
          ulysses_degree: r.ulysses_degree,
          ring_degree: 1,
          default: r.gpus === defaults.gpus_per_node && r.version === defaults.version,
        })),
        autoTopology: () => ({ tp_size: 1, ulysses_degree: 1, ring_degree: 1 }),
        validateTopology: (s, t) => {
          const errors = [];
          if (Number(s.nodes) !== 1) errors.push("This model is served on a single node.");
          if (t.tp_size !== 1 || t.ring_degree !== 1)
            errors.push("Only Ulysses sequence parallelism and CFG parallelism are documented for this model.");
          const product = t.tp_size * t.ulysses_degree * t.ring_degree;
          const world = Number(s.gpus_per_node);
          if (world !== product && world !== 2 * product)
            errors.push(`GPU count must equal Ulysses (${product}) or twice it with CFG parallelism.`);
          return errors;
        },
      },
      resolveDeployment: (s) => {
        const topology = topologyOf(s);
        const errors = config.commandBuilder.resource.validateTopology(s, topology);
        const cfg = cfgOf(s, topology);
        const world = Number(s.gpus_per_node);
        const flags = [`--model-path ${modelOf(s)}`];
        if (world > 1) flags.push(`--num-gpus ${world}`);
        if (cfg) flags.push("--enable-cfg-parallel");
        if (topology.ulysses_degree > 1) flags.push(`--ulysses-degree ${topology.ulysses_degree}`);
        flags.push("--host {{HOST_IP}}", "--port {{PORT}}");
        const verified =
          !errors.length &&
          VERIFIED.some(
            (r) =>
              r.hw === s.hw &&
              r.version === s.version &&
              r.gpus === world &&
              r.cfg === cfg &&
              r.tp_size === topology.tp_size &&
              r.ulysses_degree === topology.ulysses_degree,
          );
        const status = errors.length ? "error" : verified ? "verified" : "unverified";
        const summary = [`Ulysses ${topology.ulysses_degree}`];
        if (cfg) summary.push("CFG 2");
        return {
          match: { hw: s.hw },
          nnodes: 1,
          verified,
          flags,
          builder: {
            topology,
            topologySummary: summary.join(" / "),
            errors,
            warnings: verified || errors.length ? [] : ["Valid topology; this exact hardware and GPU count has not been run end to end."],
            verification: { serve: status, request: status },
          },
        };
      },
    },
    modelNames: { default: modelOf(defaults) },
    placeholders: {
      HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
      PORT: { target: "command", label: "Bind port", default: "30000" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
      INPUT_IMAGE: { target: "curl", label: "Input image on the client", default: "input.png" },
    },
    curl: () =>
      [
        "curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits",
        '-F "image=@{{INPUT_IMAGE}}"',
        "--form-string 'prompt=Restore this old photo and colorize it naturally.'",
        "--form-string seed=0",
        "--form-string response_format=b64_json",
        "| python3 -c \"import base64,json,sys; open('output.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))\"",
      ].join(" \\\n  "),
    runModes: () => ["python"],
    showPlaygroundLink: false,
    cells: [],
  };
  return config;
})();
