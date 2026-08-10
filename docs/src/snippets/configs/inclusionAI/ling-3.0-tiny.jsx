export const config = {
  modelName: "Ling-3.0-tiny",

  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300"],
  groupHardware: false,

  hardware: [
    { id: "h20-3e", label: "H20-3e", vram: "141GB", vendor: "nvidia" },
    { id: "h800", label: "H800", vram: "80GB", vendor: "nvidia" },
    { id: "gb300", label: "GB300", vram: "288GB", vendor: "nvidia" },
  ],

  variants: [
    { id: "default", label: "Ling-3.0-tiny" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "inclusionAI/Ling-3.0-tiny",
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
    "h20-3e": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "h200": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "h800": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "h100": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "b200": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "gb300": "lmsysorg/sglang:dev-Ling-3.0-flash",
  },

  benchmarkCommands: {
    speed: `python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct: `# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  github: {
    cookbookModel: "inclusionAI/Ling-3.0-tiny",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4] },
      ],
    },
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser deepseek-r1" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser glm45" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "h20-3e", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--context-length 131072",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--context-length 131072",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--context-length 131072",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--context-length 131072",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--context-length 131072",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--context-length 131072",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
