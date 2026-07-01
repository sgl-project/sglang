// Ornith-1.0 (DeepReinforce) — config-driven cookbook page.
// The launch flags follow the SGLang quickstarts published on the model cards; the 9B recipe makes the single-GPU default explicit with --tp 1.
// FP8 quantization cells use their FP8 repo ids from the collection with the same flags; their README quickstarts currently point to the non-FP8 repos.
// Cells remain unverified until exact recipes are run and signed off.

export const config = {
  modelName: "Ornith-1.0",

  supportedHardware: ["h100", "h200"],

  showVerificationBadge: false,

  variants: [
    { id: "397b", label: "397B", subtitle: "MoE" },
    { id: "35b",  label: "35B",  subtitle: "MoE" },
    { id: "9b",       label: "9B",       subtitle: "dense" },
  ],

  quantizations: [
    { id: "bf16",  label: "BF16" },
    { id: "fp8",   label: "FP8" },
  ],

  strategies: [
    { id: "default", label: "Default" },
  ],

  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "397b|bf16":     "Ornith-1.0-397B",
    "397b|fp8":      "Ornith-1.0-397B-FP8",
    "35b|bf16":      "Ornith-1.0-35B",
    "35b|fp8":       "Ornith-1.0-35B-FP8",
    "9b|bf16":       "Ornith-1.0-9B",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"         },
    PORT:      { target: "command", label: "Bind port",         default: "30000"           },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost"       },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"           },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Write a compact Python is_prime function."}], "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 1024 }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name random \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 128: 256, 256: 512 },
  },

  dockerImages: {
    h100: "lmsysorg/sglang:latest",
    h200: "lmsysorg/sglang:latest",
  },

  github: {
    cookbookModel: "deepreinforce-ai/Ornith-1.0",
  },

  cells: [
    {
      match: { hw: "h200", variant: "397b", quant: "bf16", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-397B",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 262144",
        "--mem-fraction-static 0.8",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "397b", quant: "fp8", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-397B-FP8",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 262144",
        "--mem-fraction-static 0.8",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "35b", quant: "bf16", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-35B",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 2",
        "--context-length 262144",
        "--mem-fraction-static 0.85",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "35b", quant: "fp8", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-35B-FP8",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 2",
        "--context-length 262144",
        "--mem-fraction-static 0.85",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "9b", quant: "bf16", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-9B",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 1",
        "--context-length 262144",
        "--mem-fraction-static 0.85",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "397b", quant: "fp8", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-397B-FP8",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 262144",
        "--mem-fraction-static 0.8",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "35b", quant: "bf16", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-35B",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 2",
        "--context-length 262144",
        "--mem-fraction-static 0.85",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "35b", quant: "fp8", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-35B-FP8",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 2",
        "--context-length 262144",
        "--mem-fraction-static 0.85",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "9b", quant: "bf16", strategy: "default", nodes: "single" },
      env: [],
      flags: [
        "--model-path deepreinforce-ai/Ornith-1.0-9B",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 1",
        "--context-length 262144",
        "--mem-fraction-static 0.85",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
