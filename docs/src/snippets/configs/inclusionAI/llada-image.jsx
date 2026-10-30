// LLaDA-Image command builder for SGLang Diffusion.
//
// Both checkpoints load remote configuration code, so every generated command
// keeps --trust-remote-code beside the commit-pinned --revision that was
// validated end to end on one H200.

export const config = {
  modelName: "LLaDA-Image",
  supportedHardware: ["h200"],
  groupHardware: false,
  matchDims: [],

  checkpoints: {
    base: {
      model: "inclusionAI/LLaDA-Image",
      revision: "e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039",
    },
    turbo: {
      model: "inclusionAI/LLaDA-Image-Turbo",
      revision: "f4afc52d925bbac4e22a1c947111fc1f127e37e5",
    },
  },

  overlayDims: [
    {
      id: "weights",
      title: "Checkpoint",
      scope: "base",
      description:
        "Base favors fidelity with 50 steps and CFG 5. Turbo is distilled to 4 steps without CFG. Each checkpoint applies its own sampling defaults.",
      default: "turbo",
      learnMore: "#checkpoint-selection",
      options: [
        { id: "base", label: "Base", subtitle: "50 steps, CFG 5" },
        { id: "turbo", label: "Turbo", subtitle: "4 steps, CFG 1", recommended: true },
      ],
    },
    {
      id: "mode",
      title: "Request mode",
      scope: "base",
      description:
        "One server handles both endpoints. Editing takes exactly one source image.",
      default: "generate",
      options: [
        { id: "generate", label: "Text to image", recommended: true },
        { id: "edit", label: "Image editing" },
      ],
    },
    {
      id: "placement",
      title: "Placement",
      scope: "serve",
      description:
        "The embedded LLaDA 2.0 MoE text encoder stays resident on the GPU.",
      default: "resident",
      learnMore: "#unsupported-controls",
      options: [
        { id: "resident", label: "Resident", recommended: true },
        {
          id: "encoder-offload",
          label: "Text encoder offload",
          disabled: true,
          disableReason:
            "LLaDA-Image refuses text encoder offload at startup because the embedded encoder must stay resident.",
        },
      ],
    },
    {
      id: "outputs",
      title: "Outputs",
      scope: "request",
      description:
        "Outputs of one request are generated in sequence. The verified request returns one image.",
      kind: "number",
      min: 1,
      max: 10,
      unit: "outputs per prompt",
      default: 1,
      options: [],
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
      limits: {
        nodes: { min: 1, max: 1 },
        gpus_per_node: { min: 1, max: 1 },
      },
      verifiedRecipes: [
        {
          id: "h200-1gpu",
          hw: "h200",
          nodes: 1,
          gpus_per_node: 1,
          placement: "resident",
          tp_size: 1,
          ulysses_degree: 1,
          ring_degree: 1,
          default: true,
        },
      ],
      autoTopology: () => ({ tp_size: 1, ulysses_degree: 1, ring_degree: 1 }),
      validateTopology: (s, topology) => {
        const errors = []
        if (Number(s.nodes) !== 1 || Number(s.gpus_per_node) !== 1) {
          errors.push("LLaDA-Image serves on one GPU and refuses multi-GPU launches at startup.")
        }
        if ([topology.tp_size, topology.ulysses_degree, topology.ring_degree].some((n) => Number(n) !== 1)) {
          errors.push("LLaDA-Image requires TP 1, Ulysses 1, and Ring 1.")
        }
        return errors
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource
      const topology = s.topology_mode === "manual"
        ? {
            tp_size: Number(s.tp_size),
            ulysses_degree: Number(s.ulysses_degree),
            ring_degree: Number(s.ring_degree),
          }
        : resource.autoTopology(s)
      const errors = resource.validateTopology(s, topology)
      const checkpoint = config.checkpoints[s.weights] || config.checkpoints.turbo
      const recipe = resource.verifiedRecipes.find((entry) => entry.hw === s.hw
        && entry.nodes === Number(s.nodes)
        && entry.gpus_per_node === Number(s.gpus_per_node)
        && entry.placement === s.placement
        && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree
        && entry.ring_degree === topology.ring_degree)
      const serveVerified = !!recipe && errors.length === 0
      const requestVerified = serveVerified && Number(s.outputs) === 1
      const flags = [
        "--trust-remote-code",
        `--model-path ${checkpoint.model}`,
        `--revision ${checkpoint.revision}`,
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ]
      return {
        match: { hw: s.hw },
        nnodes: 1,
        verified: requestVerified,
        flags,
        builder: {
          topology,
          topologySummary: "TP 1, Ulysses 1, Ring 1, one GPU",
          errors,
          warnings: errors.length || requestVerified
            ? []
            : ["Only one output per request was measured end to end."],
          verification: {
            serve: errors.length ? "error" : (serveVerified ? "verified" : "unverified"),
            request: errors.length ? "error" : (requestVerified ? "verified" : "unverified"),
          },
          resolvedSettings: {
            placement: "Resident",
            precision: "BF16",
            encoder: "LLaDA 2.0 MoE, resident",
          },
        },
      }
    },
  },

  modelNames: { default: "inclusionAI/LLaDA-Image-Turbo" },

  // No published image carries this integration yet, so only the Python run
  // mode is shown. After an end-to-end Docker run: drop `runModes` and point
  // dockerImages at that tag. Editing uploads the source image in the request
  // body, so the server container needs no extra mount.
  runModes: ["python"],
  dockerImages: {
    h200: "lmsysorg/sglang:dev",
  },

  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    INPUT_IMAGE: { target: "curl", label: "Source image on the client", default: "input.png" },
  },

  curl: (s) => {
    const checkpoint = config.checkpoints[s.weights] || config.checkpoints.turbo
    if (s.mode === "edit") {
      return `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits \\
  -F 'image=@{{INPUT_IMAGE}}' \\
  -F 'model=${checkpoint.model}' \\
  -F 'prompt=Place the subject on a beach at golden hour.' \\
  -F 'size=1024x1024' \\
  -F 'n=${Number(s.outputs)}' \\
  -F 'response_format=b64_json'`
    }
    const request = {
      model: checkpoint.model,
      prompt: "A cinematic photograph of a red fox standing in fresh snow at golden hour, detailed fur, shallow depth of field",
      size: "1024x1024",
      n: Number(s.outputs),
      response_format: "b64_json",
    }
    return `curl -sS --fail-with-body http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify(request, null, 2)}'`
  },
  showPlaygroundLink: false,
  cells: [],
}
