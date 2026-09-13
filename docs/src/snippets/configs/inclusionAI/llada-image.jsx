// LLaDA-Image deployment and request matrix for the scoped diffusion builder.

export const config = (() => {
const CHECKPOINTS = {
  "base-bf16": {
    model: "inclusionAI/LLaDA-Image",
    revision: "e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039",
    steps: 50,
    guidance: 5.0,
  },
  "base-fp8": {
    model: "inclusionAI/LLaDA-Image-FP8",
    revision: "7678c6071b139e1989565db41122b16d767cc585",
    steps: 50,
    guidance: 5.0,
  },
  "turbo-bf16": {
    model: "inclusionAI/LLaDA-Image-Turbo",
    revision: "033d068d0702ed81c45ec1bfbe61e318d843d948",
    steps: 4,
    guidance: 1.0,
  },
  "turbo-fp8": {
    model: "inclusionAI/LLaDA-Image-Turbo-FP8",
    revision: "2b822499be3f33a6f83c988dafab7986ebd1f733",
    steps: 4,
    guidance: 1.0,
  },
};

return {
  modelName: "LLaDA-Image",
  supportedHardware: ["h200"],
  groupHardware: false,
  matchDims: [],

  overlayDims: [
    {
      id: "weights",
      title: "Checkpoint weights",
      scope: "base",
      description: "Base prioritizes fidelity; Turbo uses a distilled four-step schedule. FP8 changes the checkpoint, not a server-side quantization flag.",
      default: "turbo-bf16",
      options: [
        { id: "base-bf16", label: "Base BF16", subtitle: "50 steps · CFG 5" },
        {
          id: "base-fp8",
          label: "Base FP8",
          subtitle: "50 steps · CFG 5",
        },
        { id: "turbo-bf16", label: "Turbo BF16", subtitle: "4 steps · CFG 1", recommended: true },
        {
          id: "turbo-fp8",
          label: "Turbo FP8",
          subtitle: "4 steps · CFG 1",
        },
      ],
    },
    {
      id: "mode",
      title: "Request mode",
      scope: "base",
      description: "One server handles both endpoints; editing requires exactly one source image.",
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
      description: "The embedded LLaDA 2.0 16B MoE text encoder must remain resident.",
      default: "resident",
      options: [
        { id: "resident", label: "Resident", recommended: true },
      ],
    },
    {
      id: "attention",
      title: "Attention",
      scope: "serve",
      description: "Use the platform-selected backend for the verified H200 recipes.",
      default: "platform",
      options: [
        { id: "platform", label: "Platform default", recommended: true },
      ],
    },
    {
      id: "precision",
      title: "Precision",
      scope: "serve",
      description: "Precision is encoded by the selected BF16 or FP8 checkpoint.",
      default: "native",
      options: [
        { id: "native", label: "Checkpoint native", recommended: true },
      ],
    },
    {
      id: "encoder",
      title: "Text encoder",
      scope: "serve",
      description: "LLaDA 2.0 16B MoE (LLaDA2MoeModelLM) runs as an embedded SRT worker on every SP rank and cannot use generic CPU offload.",
      default: "auto",
      options: [
        { id: "auto", label: "Resident per rank", recommended: true },
      ],
    },
    {
      id: "execution",
      title: "Execution",
      scope: "serve",
      default: "eager",
      options: [
        { id: "eager", label: "Eager", recommended: true },
      ],
    },
    {
      id: "outputs",
      title: "Outputs",
      scope: "request",
      description: "The verified request shape uses one output. SP2 rejects n greater than one.",
      kind: "number",
      min: 1,
      max: 1,
      unit: "output per prompt",
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
        gpus_per_node: { min: 1, max: 2 },
      },
      verifiedRecipes: [
        {
          id: "h200-sp1",
          hw: "h200",
          nodes: 1,
          gpus_per_node: 1,
          placement: "resident",
          tp_size: 1,
          ulysses_degree: 1,
          ring_degree: 1,
          encoder: "auto",
          default: true,
        },
        {
          id: "h200-sp2",
          hw: "h200",
          nodes: 1,
          gpus_per_node: 2,
          placement: "resident",
          tp_size: 1,
          ulysses_degree: 2,
          ring_degree: 1,
          encoder: "auto",
        },
      ],
      autoTopology: (s) => ({
        tp_size: 1,
        ulysses_degree: Number(s.gpus_per_node),
        ring_degree: 1,
      }),
      validateTopology: (s, topology) => {
        const errors = [];
        const nodes = Number(s.nodes);
        const gpus = Number(s.gpus_per_node);
        if (nodes !== 1) errors.push("LLaDA-Image currently supports single-node deployment only.");
        if (![1, 2].includes(gpus)) errors.push("LLaDA-Image supports one or two GPUs per server.");
        if (Number(topology.tp_size) !== 1) errors.push("LLaDA-Image requires tensor parallel size 1.");
        if (Number(topology.ulysses_degree) !== gpus) errors.push("Ulysses degree must equal the GPU count.");
        if (Number(topology.ring_degree) !== 1) errors.push("LLaDA-Image requires ring degree 1.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? {
            tp_size: Number(s.tp_size),
            ulysses_degree: Number(s.ulysses_degree),
            ring_degree: Number(s.ring_degree),
          }
        : resource.autoTopology(s);
      const errors = resource.validateTopology(s, topology);
      const checkpoint = CHECKPOINTS[s.weights] || CHECKPOINTS["turbo-bf16"];
      const recipe = resource.verifiedRecipes.find((entry) => entry.hw === s.hw
        && entry.nodes === Number(s.nodes)
        && entry.gpus_per_node === Number(s.gpus_per_node)
        && entry.placement === s.placement
        && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree
        && entry.ring_degree === topology.ring_degree);
      const topologyVerified = !!recipe && errors.length === 0;
      const serveVerified = topologyVerified;
      const requestVerified = topologyVerified;

      const flags = [
        `--model-path ${checkpoint.model}`,
        `--revision ${checkpoint.revision}`,
        "--trust-remote-code",
      ];
      if (Number(s.gpus_per_node) === 2) {
        flags.push(
          "--num-gpus 2",
          "--sp-degree 2",
          "--ulysses-degree 2",
          "--ring-degree 1",
        );
      }
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");

      return {
        match: { hw: s.hw },
        nnodes: 1,
        verified: serveVerified,
        verificationStatus: serveVerified ? "verified" : "unverified",
        flags,
        builder: {
          topology,
          topologySummary: Number(s.gpus_per_node) === 2
            ? "TP 1 · Ulysses 2 · Ring 1 · SP2 · Single node"
            : "TP 1 · Ulysses 1 · Ring 1 · SP1 · Single node",
          errors,
          warnings: [],
          verification: {
            serve: errors.length ? "error" : (serveVerified ? "verified" : "unverified"),
            request: errors.length ? "error" : (requestVerified ? "verified" : "unverified"),
          },
          resolvedSettings: {
            placement: "Resident",
            attention: "Platform default",
            precision: s.weights.endsWith("fp8") ? "Published FP8 checkpoint" : "BF16",
            encoder: "LLaDA 2.0 16B MoE, resident per SP rank",
            execution: "Eager",
          },
        },
      };
    },
  },

  modelNames: {
    default: "inclusionAI/LLaDA-Image-Turbo",
  },
  dockerImages: {
    h200: "lmsysorg/sglang:dev",
  },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    INPUT_IMAGE: { target: "curl", label: "Source image", default: "input.png" },
  },
  curl: (s) => {
    const checkpoint = CHECKPOINTS[s.weights] || CHECKPOINTS["turbo-bf16"];
    if (s.mode === "edit") {
      return `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/edits \\
  -F 'image=@{{INPUT_IMAGE}}' \\
  -F 'model=${checkpoint.model}' \\
  -F 'prompt=Place the subject on a beach at golden hour.' \\
  -F 'size=1024x1024' \\
  -F 'n=${Number(s.outputs)}' \\
  -F 'response_format=b64_json' \\
  -F 'num_inference_steps=${checkpoint.steps}' \\
  -F 'guidance_scale=${checkpoint.guidance}' \\
  -F 'seed=42'`;
    }
    const request = {
      model: checkpoint.model,
      prompt: "A cinematic photograph of a red fox standing in fresh snow at golden hour, detailed fur, shallow depth of field",
      size: "1024x1024",
      n: Number(s.outputs),
      response_format: "b64_json",
      num_inference_steps: checkpoint.steps,
      guidance_scale: checkpoint.guidance,
      seed: 42,
    };
    return `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify(request, null, 2)}'`;
  },
  cells: [],
  showPlaygroundLink: false,
};
})();
