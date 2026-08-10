// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// `{{MODEL_NAME}}` resolves to `modelNames` below (HF repos under the nvidia org).

export const config = {
  modelName: "Nemotron 3.5 Lightning",

  // Two validated single-GPU platforms, both at TP1/EP1.
  supportedHardware: ["h100", "dgx-spark"],

  variants: [{ id: "default", label: "Default" }],

  quantizations: [{ id: "nvfp4", label: "NVFP4" }],

  // The base serving recipe plus the three validated speculative decoders.
  strategies: [
    { id: "balanced", label: "Balanced" },
    { id: "mtp",      label: "MTP"      },
    { id: "dflash",   label: "DFlash"   },
    { id: "dspark",   label: "DSpark"   },
  ],

  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|nvfp4": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"         },
    PORT:      { target: "command", label: "Bind port",         default: "30000"           },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost"       },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"           },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  accuracyLabels: [["gsm8k_pct", "GSM8K", "%"]],

  dockerImages: {
    // Multi-arch index (amd64 + arm64), so one tag covers H100 and GB10.
    // Equivalent to dev-cu13-nemotron3-5-lighting.
    h100:        "lmsysorg/sglang:dev-nemotron3-5-lighting",
    "dgx-spark": "lmsysorg/sglang:dev-nemotron3-5-lighting",
  },

  github: {
    cookbookModel: "nvidia/nemotron-3.5-lightning",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },

    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "marlin", label: "Marlin (W4A16)", flags: ["--moe-runner-backend marlin"] },
          { id: "deepep", label: "DeepEP",         flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser nemotron_3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE",
                  "--speculative-draft-model-path {{MODEL_NAME}}",
                  "--speculative-num-steps 5",
                  "--speculative-eagle-topk 1",
                  "--speculative-num-draft-tokens 6",
                  "--speculative-draft-attention-backend flashinfer",
                  "--speculative-moe-runner-backend marlin"] },
        { id: "dflash",  label: "DFlash",
          flags: ["--speculative-algorithm DFLASH",
                  "--speculative-draft-model-path nvidia/nemotron-3.5-dflash-w4a16-preview",
                  "--speculative-dflash-block-size 6",
                  "--speculative-draft-attention-backend flashinfer",
                  "--speculative-moe-runner-backend marlin"] },
        { id: "dspark",  label: "DSpark",
          flags: ["--speculative-algorithm DSPARK",
                  "--speculative-draft-model-path nvidia/nemotron-3.5-dspark-w4a16-preview",
                  "--speculative-dspark-block-size 7",
                  "--speculative-draft-attention-backend flashinfer",
                  "--speculative-moe-runner-backend marlin"] },
      ],
    },
  },

  cells: [
    // ==== NVIDIA Hopper (SM90) + NVFP4, single GPU ====
    // fa3 attention + Triton Mamba, extra_buffer radix strategy.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 1",
        "--ep-size 1",
        "--context-length 28672",
        "--attention-backend fa3",
        "--mamba-backend triton",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-track-interval 256",
        "--mamba-cache-philox-rounds 0",
        "--moe-runner-backend marlin",
        "--kv-cache-dtype auto",
        "--max-total-tokens 32768",
        "--max-running-requests 16",
        "--max-prefill-tokens 10240",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--num-continuous-decode-steps 1",
        "--cuda-graph-max-bs-decode 16",
        "--watchdog-timeout 1800",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MTP: the draft head is embedded in the target checkpoint, so SGLang's
    // EAGLE path points --speculative-draft-model-path back at the target.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "mtp", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 1",
        "--ep-size 1",
        "--context-length 28672",
        "--attention-backend fa3",
        "--mamba-backend triton",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-track-interval 256",
        "--mamba-cache-philox-rounds 0",
        "--moe-runner-backend marlin",
        "--kv-cache-dtype auto",
        "--max-total-tokens 32768",
        "--max-running-requests 16",
        "--max-prefill-tokens 10240",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--num-continuous-decode-steps 1",
        "--cuda-graph-max-bs-decode 16",
        "--watchdog-timeout 1800",
        "--speculative-moe-runner-backend marlin",
        "--speculative-attention-mode prefill",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash: separate draft model; depth five -> block/verify width six.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 1",
        "--ep-size 1",
        "--context-length 28672",
        "--attention-backend fa3",
        "--mamba-backend triton",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-track-interval 256",
        "--mamba-cache-philox-rounds 0",
        "--moe-runner-backend marlin",
        "--kv-cache-dtype auto",
        "--max-total-tokens 32768",
        "--max-running-requests 16",
        "--max-prefill-tokens 10240",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--num-continuous-decode-steps 1",
        "--cuda-graph-max-bs-decode 16",
        "--watchdog-timeout 1800",
        "--speculative-moe-runner-backend marlin",
        "--speculative-attention-mode prefill",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/nemotron-3.5-dflash-w4a16-preview",
        "--speculative-dflash-block-size 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DSpark: separate draft model; gamma seven -> verify width eight.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" },
      env: [
        "SGLANG_RAGGED_VERIFY_MODE=static",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 1",
        "--ep-size 1",
        "--context-length 28672",
        "--attention-backend fa3",
        "--mamba-backend triton",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-track-interval 256",
        "--mamba-cache-philox-rounds 0",
        "--moe-runner-backend marlin",
        "--kv-cache-dtype auto",
        "--max-total-tokens 32768",
        "--max-running-requests 16",
        "--max-prefill-tokens 10240",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--num-continuous-decode-steps 1",
        "--cuda-graph-max-bs-decode 16",
        "--watchdog-timeout 1800",
        "--speculative-moe-runner-backend marlin",
        "--speculative-attention-mode prefill",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/nemotron-3.5-dspark-w4a16-preview",
        "--speculative-dspark-block-size 7",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== NVIDIA DGX Spark (GB10 / SM121) + NVFP4, single GPU ====
    // 128GB coherent unified memory. Same profile as H100 apart from
    // flashinfer attention, a decode CUDA-graph batch cap of 4, and a
    // shorter watchdog timeout.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 1",
        "--ep-size 1",
        "--context-length 28672",
        "--attention-backend flashinfer",
        "--mamba-backend triton",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-track-interval 256",
        "--mamba-cache-philox-rounds 0",
        "--moe-runner-backend marlin",
        "--kv-cache-dtype auto",
        "--max-total-tokens 32768",
        "--max-running-requests 16",
        "--max-prefill-tokens 10240",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--num-continuous-decode-steps 1",
        "--cuda-graph-max-bs-decode 4",
        "--watchdog-timeout 300",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MTP: the draft head is embedded in the target checkpoint, so SGLang's
    // EAGLE path points --speculative-draft-model-path back at the target.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "mtp", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 1",
        "--ep-size 1",
        "--context-length 28672",
        "--attention-backend flashinfer",
        "--mamba-backend triton",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-track-interval 256",
        "--mamba-cache-philox-rounds 0",
        "--moe-runner-backend marlin",
        "--kv-cache-dtype auto",
        "--max-total-tokens 32768",
        "--max-running-requests 16",
        "--max-prefill-tokens 10240",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--num-continuous-decode-steps 1",
        "--cuda-graph-max-bs-decode 4",
        "--watchdog-timeout 300",
        "--speculative-moe-runner-backend marlin",
        "--speculative-attention-mode prefill",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash: separate draft model; depth five -> block/verify width six.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 1",
        "--ep-size 1",
        "--context-length 28672",
        "--attention-backend flashinfer",
        "--mamba-backend triton",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-track-interval 256",
        "--mamba-cache-philox-rounds 0",
        "--moe-runner-backend marlin",
        "--kv-cache-dtype auto",
        "--max-total-tokens 32768",
        "--max-running-requests 16",
        "--max-prefill-tokens 10240",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--num-continuous-decode-steps 1",
        "--cuda-graph-max-bs-decode 4",
        "--watchdog-timeout 300",
        "--speculative-moe-runner-backend marlin",
        "--speculative-attention-mode prefill",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/nemotron-3.5-dflash-w4a16-preview",
        "--speculative-dflash-block-size 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DSpark: separate draft model; gamma seven -> verify width eight.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" },
      env: [
        "SGLANG_RAGGED_VERIFY_MODE=static",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 1",
        "--ep-size 1",
        "--context-length 28672",
        "--attention-backend flashinfer",
        "--mamba-backend triton",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-track-interval 256",
        "--mamba-cache-philox-rounds 0",
        "--moe-runner-backend marlin",
        "--kv-cache-dtype auto",
        "--max-total-tokens 32768",
        "--max-running-requests 16",
        "--max-prefill-tokens 10240",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--num-continuous-decode-steps 1",
        "--cuda-graph-max-bs-decode 4",
        "--watchdog-timeout 300",
        "--speculative-moe-runner-backend marlin",
        "--speculative-attention-mode prefill",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/nemotron-3.5-dspark-w4a16-preview",
        "--speculative-dspark-block-size 7",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
