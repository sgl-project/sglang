// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// `{{MODEL_NAME}}` resolves to `modelNames` below (HF repos under the nvidia org).

export const config = {
  modelName: "Nemotron 3.5 Lightning",

  // Two validated single-GPU platforms. B200 publishes the TP1/EP1 profile;
  // the four-GPU TP4/EP4 profile is reachable through the Playground TP knob.
  supportedHardware: ["b200", "h100"],

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
    b200: "lmsysorg/sglang:nightly-dev-20260719-99f5a6f4",
    h100: "lmsysorg/sglang:nightly-dev-20260719-99f5a6f4",
  },

  github: {
    cookbookModel: "nvidia/nemotron-3.5-lightning",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        // TP4/EP4 is the validated four-B200 profile; pair it with
        // --max-total-tokens 524288 and --chunked-prefill-size 16384.
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
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--enable-metrics",
        "--context-length 28672",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.85",
        "--max-total-tokens 32768",
        "--max-prefill-tokens 10240",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MTP: the draft head is built into the target checkpoint, so SGLang's
    // EAGLE path points --speculative-draft-model-path back at the target.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "mtp", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--enable-metrics",
        "--context-length 28672",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.85",
        "--max-total-tokens 32768",
        "--max-prefill-tokens 10240",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-moe-runner-backend marlin",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash: separate draft model; depth five maps to block/verify width six.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--enable-metrics",
        "--context-length 28672",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.85",
        "--max-total-tokens 32768",
        "--max-prefill-tokens 10240",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/nemotron-3.5-dflash-w4a16-preview",
        "--speculative-dflash-block-size 6",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-moe-runner-backend marlin",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DSpark: separate draft model proposing a whole block at once; gamma seven
    // maps to verify width eight. SGLANG_RAGGED_VERIFY_MODE=static is required —
    // without it the ragged verify path picks its own mode and the block size
    // below is not honored.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" },
      env: [
        "SGLANG_RAGGED_VERIFY_MODE=static",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--enable-metrics",
        "--context-length 28672",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.85",
        "--max-total-tokens 32768",
        "--max-prefill-tokens 10240",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/nemotron-3.5-dspark-w4a16-preview",
        "--speculative-dspark-block-size 7",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-moe-runner-backend marlin",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVIDIA Blackwell (SM100) + NVFP4, single GPU ====
    // flashinfer Mamba with no_buffer radix caching and stochastic cache
    // rounding; overlap scheduling is off on this platform.
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--enable-metrics",
        "--context-length 28672",
        "--attention-backend flashinfer",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy no_buffer",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--disable-overlap-schedule",
        "--mem-fraction-static 0.85",
        "--max-total-tokens 32768",
        "--max-prefill-tokens 10240",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "mtp", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--enable-metrics",
        "--context-length 28672",
        "--attention-backend flashinfer",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy no_buffer",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--disable-overlap-schedule",
        "--mem-fraction-static 0.85",
        "--max-total-tokens 32768",
        "--max-prefill-tokens 10240",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-moe-runner-backend marlin",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--enable-metrics",
        "--context-length 28672",
        "--attention-backend flashinfer",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy no_buffer",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--disable-overlap-schedule",
        "--mem-fraction-static 0.85",
        "--max-total-tokens 32768",
        "--max-prefill-tokens 10240",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/nemotron-3.5-dflash-w4a16-preview",
        "--speculative-dflash-block-size 6",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-moe-runner-backend marlin",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" },
      env: [
        "SGLANG_RAGGED_VERIFY_MODE=static",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--enable-metrics",
        "--context-length 28672",
        "--attention-backend flashinfer",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--mamba-radix-cache-strategy no_buffer",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--disable-overlap-schedule",
        "--mem-fraction-static 0.85",
        "--max-total-tokens 32768",
        "--max-prefill-tokens 10240",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/nemotron-3.5-dspark-w4a16-preview",
        "--speculative-dspark-block-size 7",
        "--speculative-draft-attention-backend flashinfer",
        "--speculative-moe-runner-backend marlin",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

  ],
};
