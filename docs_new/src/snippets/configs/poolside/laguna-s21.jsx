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
//   (AIME25, chat sampling) requires this flag. Verified 2026-07-20 on 8×H200 and
//   4×GB300.
//
// TP sizing (118B BF16 ≈ 236 GB; FP8 ≈ 118 GB; NVFP4/INT4 ≈ 59 GB):
//   - H200 8-GPU (8×141 GB): --tp 8 for all quants (29.5 GB weights/GPU for BF16).
//     Unlike Laguna-XS-2.1, the much larger moe_intermediate_size of this 118B model
//     allows plain TP=8 for FP8 and INT4 without --ep-size. Confirmed on H200 (2026-07-20).
//   - B300 8-GPU (8×288 GB): same --tp 8 reasoning.
//   - GB300 4-GPU (4×288 GB): --tp 4 throughout (59 GB/GPU for BF16, ≥229 GB KV budget).
//     Confirmed on GB300 (2026-07-20).
//
// BF16 memory on H200:
//   BF16 (29.5 GB/GPU) leaves less headroom on H200 for CUDA-graph capture and NCCL
//   allocs than FP8/INT4. Default mem-fraction 0.893 → OOM at cuda-graph capture; 0.88
//   → NCCL crash at high concurrency. --mem-fraction-static 0.80 is required for BF16
//   high-throughput on H200. Low-latency already carries 0.7.
//   GB300 (288 GB/GPU) and B300 (288 GB/GPU) are unaffected — default heuristic suffices.
//
// SGLANG_SHARED_EXPERT_TP1=1 (FP8 cells, ALL hardware):
//   Confirmed required on H200 (TP=8) and GB300 (TP=4): the FP8 checkpoint
//   block-quantizes the shared expert (128×128 scales), which cannot TP-shard cleanly at
//   either TP degree on S-2.1. INT4 keeps the shared expert in BF16 (no flag needed);
//   BF16 is unquantized. Note: Laguna-XS-2.1 at TP=4 does NOT need this flag — the
//   difference is architecture-specific (XS-2.1 shared expert shards cleanly at TP=4).
//
// NVFP4 on GB300 — BROKEN (2026-07-20):
//   Both NVFP4 cells produce degenerate output (repetition loop, never emits EOS) on
//   4×GB300 — verified identical behaviour with and without spec-decode. Perf numbers
//   exist (mechanically valid) but quality is unusable. Root cause: likely a checkpoint
//   or sglang sm103 kernel issue; needs poolside/sglang follow-up. NVFP4 cells on GB300
//   remain verified:false until resolved. NVFP4 on B300 is unverified.
//
// NVFP4 is Blackwell-only → no h200×nvfp4 cells.
//
// DFlash memory: Low-Latency cells carry --mem-fraction-static 0.7.
//
// FP8 / NVFP4 DFlash draft rope issue:
//   poolside/Laguna-S-2.1-DFlash-FP8 and poolside/Laguna-S-2.1-DFlash-NVFP4 (as of
//   2026-07-20) are missing rope_theta in rope_parameters. If the server crashes at
//   draft model load, patch the draft config locally by adding "rope_theta": 500000.0
//   to the rope_parameters block. The BF16 and INT4 draft models are unaffected.
//
// verified:true = command ran and passed full GSM8K on that cell.
//   H200 cells: verified 2026-07-20, sglang 0.5.15.post1.
//   GB300 cells (BF16/FP8/INT4): verified 2026-07-20, sglang dev/custom build.
//   B300 cells: unverified pending measurement.
//   GB300 NVFP4 cells: verified:false — quality broken, see note above.
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
      // For BF16: use --max-tokens 131072 (see Configuration Tips: BF16 reasoning length).
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
  // H200 cells: verified (2026-07-20, sglang 0.5.15.post1).
  // GB300 BF16/FP8/INT4 cells: verified (2026-07-20, sglang dev/custom build).
  // GB300 NVFP4 cells: verified:false — output quality broken (see header).
  // B300 cells: verified:false — pending measurement.
  cells: [

    // ══════════════ NVIDIA Hopper H200 (8-GPU HGX, --tp 8) — VERIFIED ══════════════
    // Dense auto-selects fa3 on Hopper (no flag needed). LL pins fa3.
    // BF16 high-throughput requires --mem-fraction-static 0.80 on H200 (see header).
    // BF16 low-latency already carries 0.7; FP8/INT4 use default.
    {
      // VERIFIED 8×H200 tp8 (2026-07-20): GSM8K 93.18%.
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
      // SGLANG_SHARED_EXPERT_TP1=1 confirmed required (block-FP8 shared expert).
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
      // DFlash-FP8 draft: see rope_theta note in header.
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
    // Dense auto-selects trtllm_mha on Blackwell. LL MUST pin trtllm_mha (flashinfer
    // breaks hybrid-SWA at tp≥4 with DFlash active — reproduced on Laguna-XS-2.1).
    // B300 (288 GB/GPU): BF16 29.5 GB/GPU leaves ≥258 GB for KV — no mem-fraction needed.
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

    // ══════════════ NVIDIA Grace-Blackwell GB300 (4-GPU single node, --tp 4) ══════════════
    // BF16/FP8/INT4 verified 2026-07-20 (sglang dev/custom build).
    // NVFP4: verified:false — degenerate output (repetition loop, never terminates);
    //   see header comment. Perf is mechanically valid but quality is broken.
    // Dense auto-selects trtllm_mha on Blackwell. LL pins trtllm_mha.
    // BF16 59 GB/GPU (out of 288 GB): ample KV budget, default mem-fraction suffices.
    // FP8 cells require SGLANG_SHARED_EXPERT_TP1=1 — confirmed on GB300 at TP=4
    //   (S-2.1's FP8 shared expert cannot shard 4-way; this differs from XS-2.1
    //   where TP=4 works without the flag due to different expert widths).
    {
      // VERIFIED 4×GB300 tp4 (2026-07-20): GSM8K 93.33%.
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
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
      // VERIFIED 4×GB300 tp4 (2026-07-20): GSM8K 93.86%.
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
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
      // VERIFIED 4×GB300 tp4 (2026-07-20): GSM8K 94.77%.
      // SGLANG_SHARED_EXPERT_TP1=1 confirmed required at TP=4 on S-2.1 (see header).
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
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
      // VERIFIED 4×GB300 tp4 (2026-07-20): GSM8K 94.54%.
      // DFlash-FP8 draft: see rope_theta note in header.
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
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
      // verified:false — NVFP4 output quality broken on GB300 (degenerate loop).
      // See header comment. Perf kernel runs but model quality is unusable.
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
      // verified:false — NVFP4 output quality broken on GB300 (degenerate loop).
      // DFlash-NVFP4 draft also has rope_theta issue (see header).
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
      // VERIFIED 4×GB300 tp4 (2026-07-20): GSM8K 94.77%.
      // INT4 shared expert stays bf16 — no SGLANG_SHARED_EXPERT_TP1 needed.
      match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: true,
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
      // VERIFIED 4×GB300 tp4 (2026-07-20): GSM8K 95.15%.
      match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      verified: true,
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
