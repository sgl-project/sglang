// Laguna-S-2.1 (poolside) — config-driven cookbook page.
// Consumed by the shared _deployment.jsx + _playground.jsx engines (no model code there).
//
// Architecture: 118B total / ~8B active per token (top-10 routing over 256 routed experts
// + 1 shared expert). 48 layers: 12 full-attention + 36 sliding-window (window 512).
// 1,048,576-token context window. Per-head sigmoid output gating on attention.
//
// Build: uses the same `laguna` model type as Laguna-XS-2.1. All support is in SGLang main.
// The model ships custom config code on the Hub, so --trust-remote-code is required.
//
// Attention backend (same rules as Laguna-XS-2.1 — hybrid-SWA is backend-sensitive):
//   - Dense (High-Throughput): leave --attention-backend UNSET. Auto-select is correct:
//     fa3 on Hopper (H200), trtllm_mha on Blackwell (B300/GB300).
//   - DFlash (Low-Latency): MUST pin the target backend explicitly — with a speculative
//     algorithm active, the resolver falls back to flashinfer, which breaks this
//     hybrid-SWA model at tp≥4 on Blackwell (reproduced on Laguna-XS-2.1 at GB300,
//     GSM8K 76%→28%). Pin fa3 on H200, trtllm_mha on B300/GB300.
//   - NEVER use --attention-backend triton for Laguna (broken SWA handling).
//
// TP sizing (118B BF16 ≈ 236 GB; FP8 ≈ 118 GB; NVFP4/INT4 ≈ 59 GB):
//   - H200 8-GPU (8×141 GB): --tp 8 for all quants (29.5 GB weights/GPU for BF16).
//     Unlike Laguna-XS-2.1, the much larger moe_intermediate_size of this 118B model
//     is expected to allow plain TP=8 for FP8 and INT4 without --ep-size (the XS-2.1
//     constraint was 512/8=64 < block_size=128; that does not hold at the larger expert
//     width in S-2.1). FP8 cells include SGLANG_SHARED_EXPERT_TP1=1 as a precaution
//     (unverified whether the shared expert needs replication here).
//   - B300 8-GPU (8×288 GB): same --tp 8 reasoning.
//   - GB300 4-GPU (4×288 GB): --tp 4 throughout (59 GB/GPU for BF16, ≥229 GB KV budget).
//
// NVFP4 is Blackwell-only → no h200×nvfp4 cells.
//
// DFlash memory: Low-Latency cells carry --mem-fraction-static 0.7 (consistent with
// Laguna-XS-2.1; the larger model makes draft-vocab all-gather OOM more likely).
//
// verified:false on all cells — commands follow the XS-2.1 verified shape + the TP
// scaling above but have not yet been run against Laguna-S-2.1. Replace with
// verified:true after confirming each cell with a full GSM8K pass.
//
// Draft/target precision ALWAYS matches: each quantized target pairs with the DFlash
// draft calibrated for it (poolside/Laguna-S-2.1-DFlash[-FP8/-NVFP4/-INT4]).

export const config = {
  modelName: "Laguna-S-2.1",

  supportedHardware: ["h200", "b300", "gb300"],

  variants: [
    { id: "default", label: "Default" },
  ],

  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
    { id: "int4",  label: "INT4"  },
  ],

  // Two operating points:
  //   low-latency  = DFlash speculative decoding (matched-precision draft) — interactive /
  //                  few-stream serving.
  //   high-throughput = plain serving (no speculation) — batch-saturated workloads, where
  //                  speculation's draft+rejection overhead costs more than it saves.
  strategies: [
    { id: "low-latency",     label: "Low-latency" },
    { id: "high-throughput", label: "High-throughput" },
  ],

  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16":  "poolside/Laguna-S-2.1",
    "default|fp8":   "poolside/Laguna-S-2.1-FP8",
    "default|nvfp4": "poolside/Laguna-S-2.1-NVFP4",
    "default|int4":  "poolside/Laguna-S-2.1-INT4",
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
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    // GSM8K is the required accuracy sanity on every verified cell, via sgl-eval.
    accuracy: {
      gsm8k_pct:
`# pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 128`,
      // Laguna's template gates on enable_thinking (not the generic 'thinking' key).
      // To run AIME25 with thinking, serve with a copy of the model's chat template
      // whose enable_thinking default is flipped to true, passed via --chat-template.
      aime25_pct:
`# pip install git+https://github.com/sgl-project/sgl-eval
# Serve with an enable_thinking=true chat template (see Configuration Tips: Thinking).
sgl-eval run aime25 \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --n-repeats 16 --max-tokens 64000 \\
  --temperature 1.0 --top-p 0.95 --thinking \\
  --num-threads 128`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 128: 256, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // No variant-wide accuracy default; real numbers are per-cell in laguna-s21-benchmarks.jsx.
  defaultAccuracy: {
    default: { gsm8k_pct: null, aime25_pct: null },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
    ["aime25_pct", "AIME25", "%"],
  ],

  dockerImages: {
    h200:  "lmsysorg/sglang:dev",
    b300:  "lmsysorg/sglang:dev",
    gb300: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "poolside/Laguna-S-2.1",
  },

  playgroundFeatures: {

    // Hybrid-SWA GQA model (8 KV heads) — TP shards cleanly at 1/2/4/8.
    // No DP-Attention / CP knobs: unvalidated on this model family.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },

    // Reasoning + tool-call parsers (poolside_v1, same family as Laguna-XS-2.1).
    // Baked into every Deploy cell. The chat template auto-detects both on
    // transformers ≥ 5.10 (`Auto-detected template features: reasoning_parser=poolside_v1,
    // tool_call_parser=poolside_v1`), so these are explicit-but-redundant there.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser poolside_v1" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser poolside_v1" },
      ],
    },
  },

  // Cells: (h200 × {bf16,fp8,int4} + b300/gb300 × {bf16,fp8,nvfp4,int4}) × {low-latency, high-throughput}.
  // All verified:false — not yet measured on Laguna-S-2.1.
  // NVFP4 is Blackwell-only: no h200×nvfp4 cells.
  cells: [

    // ══════════════ NVIDIA Hopper H200 (8-GPU HGX, --tp 8) ══════════════
    // Dense auto-selects fa3 on Hopper (no flag needed). LL pins fa3 (DFlash-safe on
    // Hopper; with a spec algorithm active, auto would fall back to flashinfer).
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend fa3",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // SGLANG_SHARED_EXPERT_TP1=1 included as precaution — the shared expert is likely
      // also block-FP8-quantized (same architecture family as XS-2.1). Unverified.
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend fa3",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-FP8",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend fa3",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-INT4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ══════════════ NVIDIA Blackwell Ultra B300 (8-GPU HGX) ══════════════
    // Dense auto-selects trtllm_mha on Blackwell (no flag). LL MUST pin trtllm_mha —
    // with DFlash active, auto falls back to flashinfer, which is broken for this
    // hybrid-SWA model at tp≥4 (reproduced + bisected on GB300 with Laguna-XS-2.1).
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-FP8",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-NVFP4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-INT4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ══════════════ NVIDIA Grace-Blackwell GB300 (4-GPU single node, --tp 4) ══════════════
    // BF16 118B at tp 4 = 59 GB/GPU (out of 288 GB): ample KV budget.
    // Dense auto-selects trtllm_mha on Blackwell. LL pins trtllm_mha (same flashinfer
    // concern as B300 above).
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-FP8",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-NVFP4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-INT4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
