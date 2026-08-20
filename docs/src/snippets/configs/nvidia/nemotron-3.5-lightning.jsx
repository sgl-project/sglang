// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// `{{MODEL_NAME}}` resolves to `modelNames` below (HF repos under the nvidia org).

export const config = {
  modelName: "Nemotron 3.5 Lightning",

  // Three validated single-GPU platforms, all at TP1/EP1.
  supportedHardware: ["b200", "h100", "dgx-spark"],

  variants: [{ id: "default", label: "Default" }],

  quantizations: [{ id: "nvfp4", label: "NVFP4" }, { id: "bf16", label: "BF16" }],

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
    "default|bf16": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
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
    // Multi-arch index (amd64 + arm64), so one tag covers H100, B200, and GB10.
    // Equivalent to dev-cu13-nemotron3-5-lightning.
    b200:        "lmsysorg/sglang:dev-nemotron3-5-lightning",
    h100:        "lmsysorg/sglang:dev-nemotron3-5-lightning",
    "dgx-spark": "lmsysorg/sglang:dev-nemotron3-5-lightning",
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
                  "--speculative-num-draft-tokens 6"] },
        { id: "dflash",  label: "DFlash",
          flags: ["--speculative-algorithm DFLASH",
                  "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
                  "--speculative-dflash-block-size 4"] },
        { id: "dspark",  label: "DSpark",
          flags: ["--speculative-algorithm DSPARK",
                  "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
                  "--speculative-dspark-block-size 3"] },
      ],
    },
  },

  cells: [
    // ==== NVIDIA B200 (SM100) + NVFP4, single GPU ====
    // The Nemotron-H resolver selects FlashInfer target attention without
    // speculation and TRT-LLM MHA target/eligible-draft attention with it.
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--mem-fraction-static 0.85",
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
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash uses depth five on B200; its full-attention draft resolves to
    // FlashInfer while target verification remains TRT-LLM MHA.
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
        "--speculative-dflash-block-size 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
        "--speculative-dspark-block-size 3",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== NVIDIA Hopper (SM90) + NVFP4, single GPU ====
    // FA3 target attention is selected by default and inherited by the draft.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
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
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash: separate draft model; depth three -> block/verify width four.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
        "--speculative-dflash-block-size 4",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DSpark: separate draft model; gamma three.
    {
      match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
        "--speculative-dspark-block-size 3",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== NVIDIA DGX Spark (GB10 / SM121) + NVFP4, single GPU ====
    // The Nemotron-H resolver selects Triton target attention plus FlashInfer
    // draft attention for speculative decoding on SM121.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 4",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
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
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash: separate draft model; depth three -> block/verify width four.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
        "--speculative-dflash-block-size 4",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DSpark: separate draft model; gamma three.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
        "--speculative-dspark-block-size 3",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== NVIDIA B200 (SM100) + BF16, single GPU ====
    // The Nemotron-H resolver selects FlashInfer target attention without
    // speculation and TRT-LLM MHA target/eligible-draft attention with it.
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "mtp", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash uses depth five on B200; its full-attention draft resolves to
    // FlashInfer while target verification remains TRT-LLM MHA.
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
        "--speculative-dflash-block-size 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "dspark", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds 5",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
        "--speculative-dspark-block-size 3",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== NVIDIA Hopper (SM90) + BF16, single GPU ====
    // FA3 target attention is selected by default and inherited by the draft.
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MTP: the draft head is embedded in the target checkpoint, so SGLang's
    // EAGLE path points --speculative-draft-model-path back at the target.
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "mtp", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash: separate draft model; depth three -> block/verify width four.
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
        "--speculative-dflash-block-size 4",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DSpark: separate draft model; gamma three.
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "dspark", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
        "--speculative-dspark-block-size 3",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== NVIDIA DGX Spark (GB10 / SM121) + BF16, single GPU ====
    // The Nemotron-H resolver selects Triton target attention plus FlashInfer
    // draft attention for speculative decoding on SM121.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 4",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MTP: the draft head is embedded in the target checkpoint, so SGLang's
    // EAGLE path points --speculative-draft-model-path back at the target.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", strategy: "mtp", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DFlash: separate draft model; depth three -> block/verify width four.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", strategy: "dflash", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
        "--speculative-dflash-block-size 4",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DSpark: separate draft model; gamma three.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", strategy: "dspark", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mamba-ssm-dtype float16",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
        "--speculative-dspark-block-size 3",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
