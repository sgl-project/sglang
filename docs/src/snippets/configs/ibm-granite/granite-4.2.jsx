export const config = {
  modelName: "Granite 4.2",

  latencyPercentile: "P50",

  supportedHardware: ["h200", "b200", "xeon"],
  hardware: [
    { id: "xeon", label: "XEON", vram: "CPU", vendor: "cpu" },
  ],
  runModes: (base) => base.hw === "xeon" ? ["python"] : ["python", "docker"],

  variants: [
    { id: "3b", label: "3B", subtitle: "Dense" },
    { id: "8b", label: "8B", subtitle: "Dense" },
    { id: "30b", label: "30B", subtitle: "Dense" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "3b|bf16": "ibm-granite/granite-4.2-3b",
    "8b|bf16": "ibm-granite/granite-4.2-8b",
    "30b|bf16": "ibm-granite/granite-4.2-30b",
  },

  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  benchmarkCommands: {
    speed:
`python -m sglang.benchmark.serving \\
  --backend sglang-oai \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} --tokenizer {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 8 --flush-cache \\
  --temperature 0.0 --top-p 1.0 \\
  --seed 123 --disable-tqdm --output-details`,
    numPromptsByConc: { 1: 80, 16: 80 },
  },

  dockerImages: {
    h200: "lmsysorg/sglang:dev",
    b200: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "ibm-granite/granite-4.2-3b",
  },

  playgroundFeatures: {
    normalizeDeltas: (deltas, derived, base) => base.hw === "xeon" ? {} : deltas,
    attention: {
      showWhen: (base) => base.hw !== "xeon",
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },
    parsers: {
      showWhen: (base) => base.hw !== "xeon",
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },
    pdDisagg: {
      showWhen: (base) => base.hw !== "xeon",
      modes: [
        { id: "off", label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode", label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl", label: "NiXL" },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
    },
    hicache: {
      showWhen: (base) => base.hw !== "xeon",
      backends: [
        { id: null, label: "Auto" },
        { id: "file", label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "hf3fs", label: "HF3FS" },
        { id: "nixl", label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto", label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back", label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "xeon", variant: "3b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      warn: "Before serving on CPU, export SGLANG_USE_CPU_ENGINE=1 in your shell. Only basic text generation has been checked; full context, Playground overrides, dataset accuracy, and performance are unverified.",
      env: [],
      omitDefaultServerAddress: true,
      flags: [
        "--model {{MODEL_NAME}}",
        "--device cpu",
        "--tp 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "xeon", variant: "8b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      warn: "Before serving on CPU, export SGLANG_USE_CPU_ENGINE=1 in your shell. Only basic text generation has been checked; full context, Playground overrides, dataset accuracy, and performance are unverified.",
      env: [],
      omitDefaultServerAddress: true,
      flags: [
        "--model {{MODEL_NAME}}",
        "--device cpu",
        "--tp 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "xeon", variant: "30b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      warn: "Before serving on CPU, export SGLANG_USE_CPU_ENGINE=1 in your shell. Only basic text generation has been checked; full context, Playground overrides, dataset accuracy, and performance are unverified.",
      env: [],
      omitDefaultServerAddress: true,
      flags: [
        "--model {{MODEL_NAME}}",
        "--device cpu",
        "--tp 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "3b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "8b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "30b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "3b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "8b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "30b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
