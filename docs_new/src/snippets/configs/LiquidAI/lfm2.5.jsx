// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// LFM2.5 note: every variant runs on ONE GPU (TP=1), so the matrix is
// hw × variant only (single quant / strategy / nodes). The reasoning parser is
// variant-intrinsic (`qwen3` for 8B-A1B, `qwen3-thinking` for 1.2B-Thinking) and
// the `lfm2` tool-call parser is part of the recommended launch, so both are baked
// into the verified cells rather than exposed as a Parsers playground axis (a
// single axis item cannot carry a per-variant flag).

export const config = {
  modelName: "LFM2.5",

  supportedHardware: ["h100", "h200", "b200"],

  variants: [
    { id: "8b-a1b",   label: "8B-A1B",        subtitle: "8.3B MoE · reasoning" },
    { id: "instruct", label: "1.2B Instruct", subtitle: "1.17B dense" },
    { id: "thinking", label: "1.2B Thinking", subtitle: "1.17B · reasoning" },
    { id: "350m",     label: "350M",          subtitle: "dense" },
    { id: "jp",       label: "1.2B JP",       subtitle: "Japanese" },
    { id: "vl",       label: "VL 1.6B",       subtitle: "vision" },
    { id: "vl-450m",  label: "VL 450M",       subtitle: "vision · compact" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "default", label: "Default" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "8b-a1b|bf16":   "LiquidAI/LFM2.5-8B-A1B",
    "instruct|bf16": "LiquidAI/LFM2.5-1.2B-Instruct",
    "thinking|bf16": "LiquidAI/LFM2.5-1.2B-Thinking",
    "350m|bf16":     "LiquidAI/LFM2.5-350M",
    "jp|bf16":       "LiquidAI/LFM2.5-1.2B-JP-202606",
    "vl|bf16":       "LiquidAI/LFM2.5-VL-1.6B",
    "vl-450m|bf16":  "LiquidAI/LFM2.5-VL-450M",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // Reproduce commands for the Benchmark card's "⚡ Reproduce" modal.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --num-threads 128`,
      gpqa_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
# GPQA's HF dataset (Idavidrein/gpqa) is gated — accept its terms with your HF account first.
sgl-eval run gpqa \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --num-threads 128`,
      mmlu_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run mmlu \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --num-threads 128`,
      aime25_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --num-threads 128`,
      mmmu_pct:
`python3 -m sglang.test.run_eval --eval-name mmmu \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --num-examples 900 --num-threads 128 --max-tokens 2048 \\
  --temperature 0.1 --min-p 0.15`,
    },
    numPromptsByConc: { 1: 10, 16: 32, 64: 128, 100: 1000, 256: 512 },
  },

  // The eval set rendered in the benchmark card + "⚡ Reproduce" (the engine
  // ships no default — every config declares its own).
  accuracyLabels: [
    ["gpqa_pct",   "GPQA Diamond",   "%"],
    ["aime25_pct", "AIME25",         "%"],
    ["gsm8k_pct",  "GSM8K (1-shot)", "%"],
    ["mmlu_pct",   "MMLU",           "%"],
    ["mmmu_pct",   "MMMU (val)",     "%"],
  ],

  // Per-variant accuracy applied to every cell. ALL values are MEASURED through
  // SGLang (B200, dev-cu13) with the exact commands in the Reproduce modal:
  // gsm8k / gpqa / aime25 / mmlu via sgl-eval (registry defaults — gpqa pass@1
  // avg-of-8, aime25 avg-of-16, gsm8k + mmlu single-shot), mmmu via
  // sglang.test.run_eval (900 examples, card sampling). They agree with the
  // LiquidAI model-card numbers within a few points where both exist; see the
  // model cards for Liquid's own reported suite (IFEval / MATH500 / BFCL / ...).
  defaultAccuracy: {
    "8b-a1b":  { mmlu_pct: 76.61, gsm8k_pct: 91.96, gpqa_pct: 52.27, aime25_pct: 45.21 },
    thinking:  { mmlu_pct: 63.2, gsm8k_pct: 86.35, gpqa_pct: 39.08, aime25_pct: 27.08 },
    instruct:  { mmlu_pct: 60.33, gsm8k_pct: 75.13, gpqa_pct: 34.41, aime25_pct: 9.58 },
    "350m":    { mmlu_pct: 40.69, gsm8k_pct: 30.63, gpqa_pct: 28.35 },
    vl:        { mmmu_pct: 39.12 },
    "vl-450m": { mmmu_pct: 30.56 },
  },

  // LFM2.5 support (model classes + the `lfm2` tool-call parser) ships in the
  // SGLang dev image; not yet in a tagged release.
  dockerImages: {
    h100: "lmsysorg/sglang:dev-cu13",
    h200: "lmsysorg/sglang:dev-cu13",
    b200: "lmsysorg/sglang:dev-cu13",
  },

  // Pre-selects the issue template's `model` dropdown on "Submit verified cell".
  github: {
    cookbookModel: "LiquidAI/lfm2.5",
  },

  playgroundFeatures: {
    // TP override only: every variant fits on (and is verified at) TP=1; TP=2 is
    // exposed for experimentation on the larger checkpoints. No Parsers axis —
    // see the header note (parsers are variant-intrinsic and live in the cells).
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2] },
      ],
    },
  },

  cells: [
    // ====================================================================
    // H100 (sm90) — default attention backend; parsers per variant
    // ====================================================================
    {
      match: { hw: "h100", variant: "8b-a1b", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--reasoning-parser qwen3",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "instruct", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "thinking", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--reasoning-parser qwen3-thinking",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "350m", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "jp", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "vl", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "vl-450m", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H200 (sm90) — same hopper recipes as H100
    // ====================================================================
    {
      match: { hw: "h200", variant: "8b-a1b", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--reasoning-parser qwen3",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "instruct", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "thinking", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--reasoning-parser qwen3-thinking",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "350m", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "jp", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "vl", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "vl-450m", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--tool-call-parser lfm2",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 (sm100) — explicit attention backend per variant:
    // dense text → trtllm_mha; 8B-A1B + VL use a mamba-style conv state cache
    // that needs a page-size-1 backend → flashinfer (VL adds fa4 vision tower)
    // ====================================================================
    {
      match: { hw: "b200", variant: "8b-a1b", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--attention-backend flashinfer",
        "--reasoning-parser qwen3",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "instruct", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--attention-backend trtllm_mha",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "thinking", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--attention-backend trtllm_mha",
        "--reasoning-parser qwen3-thinking",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "350m", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--attention-backend trtllm_mha",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "jp", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--attention-backend trtllm_mha",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "vl", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--attention-backend flashinfer",
        "--mm-attention-backend fa4",
        "--tool-call-parser lfm2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "vl-450m", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--attention-backend flashinfer",
        "--mm-attention-backend fa4",
        "--tool-call-parser lfm2",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
