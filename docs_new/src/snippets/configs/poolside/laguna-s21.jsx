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
// --sampling-defaults openai (ALL cells):
//   generation_config.json ships with top_k=0. With --sampling-defaults model (the
//   default), sglang applies top_k=0 at serving time and rejects any non-greedy sampling
//   request with HTTP 400. --sampling-defaults openai uses OpenAI API defaults instead,
//   ignoring the model-level top_k=0. Greedy (GSM8K) is unaffected; non-greedy
//   (AIME25, chat sampling) requires this flag. Verified 2026-07-20 on 8×H200.
//
// TP sizing (118B BF16 ≈ 236 GB; FP8 ≈ 118 GB; NVFP4/INT4 ≈ 59 GB):
//   - H200 8-GPU (8×141 GB): --tp 8 for all quants (29.5 GB weights/GPU for BF16).
//     Unlike Laguna-XS-2.1, the much larger moe_intermediate_size of this 118B model
//     is expected to allow plain TP=8 for FP8 and INT4 without --ep-size (the XS-2.1
//     constraint was 512/8=64 < block_size=128; that does not hold at the larger expert
//     width in S-2.1). Confirmed on H200 for BF16, FP8, INT4 (2026-07-20).
//   - B300 8-GPU (8×288 GB): same --tp 8 reasoning.
//   - GB300 4-GPU (4×288 GB): --tp 4 throughout (59 GB/GPU for BF16, ≥229 GB KV budget).
//
// BF16 memory on H200:
//   BF16 (29.5 GB/GPU) consumes ~2× more weight memory than FP8 on H200, leaving less
//   headroom for CUDA-graph capture and NCCL allocs. Default mem-fraction 0.893 → OOM
//   at cuda-graph capture; 0.88 → NCCL crash at high concurrency. --mem-fraction-static
//   0.80 is required for BF16 high-throughput on H200. Low-latency already carries 0.7.
//   B300/GB300 (288 GB/GPU) are unaffected and use the default heuristic.
//
// SGLANG_SHARED_EXPERT_TP1=1 (FP8 cells only):
//   Confirmed required: the FP8 checkpoint block-quantizes the shared expert (128×128
//   scales, can't TP-shard cleanly). INT4 keeps the shared expert in BF16 (no flag
//   needed). BF16 is unquantized. Verified 2026-07-20 on 8×H200.
//
// NVFP4 is Blackwell-only → no h200×nvfp4 cells.
//
// DFlash memory: Low-Latency cells carry --mem-fraction-static 0.7.
//
// FP8 DFlash draft rope issue:
//   poolside/Laguna-S-2.1-DFlash-FP8 (as of 2026-07-20) is missing rope_theta in
//   rope_parameters. If the server crashes at draft model load, patch the draft config
//   locally: add "rope_theta" to the rope_parameters block. INT4 draft is unaffected.
//
// verified:true = command ran and passed full GSM8K on that cell. H200 cells verified
// 2026-07-20 (sglang 0.5.15.post1). B300/GB300 cells are unverified pending measurement.
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
      // Serve with a copy of the model's chat template whose enable_thinking default
      // is flipped to true, passed via --chat-template.
      // Note: BF16 reasons ~2× longer than FP8/INT4 (median 34.8k vs 16.9k tokens)
      // and routinely truncates at 64k — use max_tokens ≥ 128k for a valid BF16 score.
      aime25_pct:
`# pip install git+https://github.com/sgl-project/sgl-eval
# Serve with an enable_thinking=true chat template (see Configuration Tips: Thinking).
# For BF16: use --max-tokens 131072 (see Configuration Tips: BF16 reasoning length).
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

  // H200: lmsysorg/sglang:latest (sglang 0.5.15.post1, verified 2026-07-20).
  // B300/GB300: unverified, using dev (nightly) as default.
  dockerImages: {
    h200:  "lmsysorg/sglang:latest",
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
  // NVFP4 is Blackwell-only: no h200×nvfp4 cells.
  // H200 cells: verified:true (measured 2026-07-20, sglang 0.5.15.post1).
  // B300/GB300 cells: verified:false — commands follow the H200 verified shape + TP scaling.
  cells: [

    // ══════════════ NVIDIA Hopper H200 (8-GPU HGX, --tp 8) — VERIFIED ══════════════
    // Dense auto-selects fa3 on Hopper (no flag needed). LL pins fa3 (DFlash-safe on
    // Hopper; with a spec algorithm active, auto would fall back to flashinfer).
    // BF16 high-throughput requires --mem-fraction-static 0.80 on H200: default 0.893
    // OOMs at cuda-graph capture; 0.88 crashes at high concurrency under NCCL alloc.
    // BF16 low-latency already carries 0.7 (sufficient).
    // FP8/INT4 use default mem-fraction (their smaller weight footprint leaves enough
    // headroom).
    {
      // VERIFIED 8×H200 tp8 (2026-07-20): GSM8K 93.18%.
      // --mem-fraction-static 0.80 required — default/0.88 OOMs (see header comment).
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--mem-fraction-static 0.80",
        "--sampling-defaults openai",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED 8×H200 tp8 (2026-07-20): GSM8K 93.33%.
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
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
        "--sampling-defaults openai",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED 8×H200 tp8 (2026-07-20): GSM8K 94.24%, AIME25 67.29%.
      // SGLANG_SHARED_EXPERT_TP1=1 confirmed required (FP8 block-quantizes shared expert).
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--sampling-defaults openai",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED 8×H200 tp8 (2026-07-20): GSM8K 94.47%.
      // Note: DFlash-FP8 draft may crash at load (rope_theta missing) — see header.
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
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
        "--sampling-defaults openai",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED 8×H200 tp8 (2026-07-20): GSM8K 95.00%, AIME25 67.71%.
      // INT4 shared expert stays bf16 — no SGLANG_SHARED_EXPERT_TP1 needed.
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--sampling-defaults openai",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED 8×H200 tp8 (2026-07-20): GSM8K 94.24%.
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      verified: true,
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
        "--sampling-defaults openai",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ══════════════ NVIDIA Blackwell Ultra B300 (8-GPU HGX) — UNVERIFIED ══════════════
    // Dense auto-selects trtllm_mha on Blackwell (no flag). LL MUST pin trtllm_mha —
    // with DFlash active, auto falls back to flashinfer, which is broken for this
    // hybrid-SWA model at tp≥4 (reproduced + bisected on GB300 with Laguna-XS-2.1).
    // B300 has 288 GB/GPU: BF16 29.5 GB/GPU leaves ≥258 GB for KV — no mem-fraction
    // adjustment needed for dense cells (unlike H200).
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ══════════════ NVIDIA Grace-Blackwell GB300 (4-GPU single node, --tp 4) — UNVERIFIED ══════════════
    // BF16 118B at tp 4 = 59 GB/GPU (out of 288 GB): ample KV budget, no mem-fraction
    // adjustment needed for dense cells. Dense auto-selects trtllm_mha on Blackwell.
    // LL pins trtllm_mha (same flashinfer concern as B300 above).
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
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
        "--sampling-defaults openai",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
