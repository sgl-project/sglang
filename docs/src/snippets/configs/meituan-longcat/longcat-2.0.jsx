// Single `export const config` literal - no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr` literals - engine injects them.

export const config = {
  modelName: "LongCat-2.0",

  supportedHardware: ["b300", "b200", "h200", "h20"],

  // Model-specific GPUs the shared HARDWARE_CATALOG does not carry.
  hardware: [
    { id: "h20", label: "H20", vram: "96GB", vendor: "nvidia" },
  ],

  variants: [
    { id: "default", label: "LongCat-2.0", subtitle: "1.6T MoE · LSA" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
  ],
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  modelNames: {
    "default|fp8": "meituan-longcat/LongCat-2.0-FP8",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    NODE0_IP:  { target: "command", label: "Head node IP",      default: "<node0-ip>"   },
    NODE_RANK: { target: "command", label: "This node rank",    default: "<node-rank>"  },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // Reproduce commands for the Benchmark card's "Reproduce" modal.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
    numPromptsByConc: { 1: 8, 16: 64, 64: 128, 256: 512, 1024: 2048 },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    b300: "lmsysorg/sglang:dev-cu13",
    b200: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
    h20:  "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "meituan-longcat/LongCat-2.0-FP8",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null,
          8,
          { value: 16, disable: { nodes: ["single"] },
            disableReason: "TP=16 requires 16 ranks - switch the Deploy panel's Nodes to Multi-Nodes first." },
        ]},
      ],
    },

    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [
        null,
        8,
        { value: 16, disable: { nodes: ["single"] },
          disableReason: "EP=16 requires 16 ranks - switch the Deploy panel's Nodes to Multi-Nodes first." },
      ]},
    },

    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
      ],
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--max-running-requests 64",
        "--mem-fraction-static 0.92",
        "--chunked-prefill-size 2048",
        "--nsa-prefill-backend fa3",
        "--kv-cache-dtype bfloat16",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":12}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--ep 16",
        "--max-running-requests 64",
        "--mem-fraction-static 0.92",
        "--chunked-prefill-size 2048",
        "--nsa-prefill-backend fa3",
        "--kv-cache-dtype bfloat16",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":12}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--ep 16",
        "--max-running-requests 64",
        "--mem-fraction-static 0.92",
        "--chunked-prefill-size 2048",
        "--nsa-prefill-backend fa3",
        "--kv-cache-dtype bfloat16",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":12}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--ep 16",
        "--max-running-requests 64",
        "--mem-fraction-static 0.92",
        "--chunked-prefill-size 2048",
        "--nsa-prefill-backend fa3",
        "--kv-cache-dtype bfloat16",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":12}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
