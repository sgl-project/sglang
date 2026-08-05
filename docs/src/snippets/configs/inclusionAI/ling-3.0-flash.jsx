export const config = {
  modelName: "Ling-3.0-flash",

  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300"],
  groupHardware: false,

  hardware: [
    { id: "h20-3e", label: "H20-3e", vram: "141GB", vendor: "nvidia" },
    { id: "h800", label: "H800", vram: "80GB", vendor: "nvidia" },
    { id: "gb300", label: "GB300", vram: "288GB", vendor: "nvidia" },
  ],

  variants: [
    { id: "default", label: "Ling-3.0-flash" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
    { id: "fp8", label: "FP8" },
  ],
  strategies: [
    { id: "low-latency", label: "Low-Latency" },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "inclusionAI/Ling-3.0-flash",
    "default|fp8": "inclusionAI/Ling-3.0-flash-fp8",
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
    cookbookModel: "inclusionAI/Ling-3.0-flash",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 4, 8] },
      ],
    },
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser ling3" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser ling3" },
      ],
    },
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off", label: "Off (greedy)" },
        { id: "nextn", label: "NEXTN (built-in MTP)", flags: ["--speculative-algorithm NEXTN"] },
      ],
    },
  },

  cells: [
    {
      match: { hw: "h20-3e", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--context-length 262144",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
