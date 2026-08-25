export const config = {
  modelName: "Ling-3.0-tiny",

  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300"],
  groupHardware: false,

  variants: [
    { id: "default", label: "Ling-3.0-tiny" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
    { id: "fp8", label: "FP8" },
    { id: "int4", label: "INT4" },
  ],
  strategies: [
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "inclusionAI/Ling-3.0-tiny",
    "default|fp8": "inclusionAI/Ling-3.0-tiny-fp8",
    "default|int4": "inclusionAI/Ling-3.0-tiny-int4",
  },

  placeholders: {
    HOST_IP:      { target: "command", label: "Bind host",          default: "0.0.0.0"              },
    PORT:         { target: "command", label: "Bind port",          default: "30000"                },
    HF_TOKEN:     { target: "command", label: "HF token (Docker)",  default: "<your-hf-token>"      },
    CURL_HOST:    { target: "curl",    label: "Server host",        default: "localhost"            },
    CURL_PORT:    { target: "curl",    label: "Server port",        default: "30000"                },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"What is the capital of France?"}] }'`,

  dockerImages: {
    "h20-3e": "lmsysorg/sglang:dev-Ling-3.0-tiny",
    "h200": "lmsysorg/sglang:dev-Ling-3.0-tiny",
    "h800": "lmsysorg/sglang:dev-Ling-3.0-tiny",
    "h100": "lmsysorg/sglang:dev-Ling-3.0-tiny",
    "b200": "lmsysorg/sglang:dev-Ling-3.0-tiny",
    "gb300": "lmsysorg/sglang:dev-Ling-3.0-tiny",
  },

  benchmarkCommands: {
    speed: `python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct: `# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32 \\
  --temperature 1.0 --top-p 0.95 \\
  --thinking`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  github: {
    cookbookModel: "inclusionAI/Ling-3.0-tiny",
  },

  playgroundFeatures: {
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser deepseek-r1" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser glm45" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
