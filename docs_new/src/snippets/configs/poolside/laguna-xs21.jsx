// Laguna-XS-2.1 (poolside) — config-driven cookbook page.
// Consumed by the shared _deployment.jsx + _playground.jsx engines (no model code there).
//
// Build: the `laguna` model type (hybrid SWA + MoE) is on SGLang main. Two extra pieces:
//   - INT4: poolside/Laguna-XS-2.1-INT4 is a MIXED-precision compressed-tensors MoE
//     (4-bit + 8-bit config groups, regex targets, no "Linear" group) — needs PR #29761
//     (MERGED on main) or it crashes at load with KeyError: 'Linear'.
//   - Low-Latency (DFlash speculative decoding): needs PR #29446 (OPEN at authoring time) —
//     install from that branch until it merges. High-Throughput cells run on plain main.
//
// Attention backend (IMPORTANT — Laguna is hybrid-SWA and backend-sensitive):
//   - Dense (High-Throughput): leave --attention-backend UNSET. Auto-select is correct:
//     fa3 on Hopper (H200), trtllm_mha on Blackwell (B200/GB300).
//   - DFlash (Low-Latency): auto-select is NOT safe — with a speculative algorithm active
//     the resolver falls back to flashinfer, which on Blackwell HALVES greedy GSM8K at
//     tp=4 (76.2% -> 28%, reproduced+bisected on GB300). Every LL cell therefore PINS the
//     target backend explicitly: fa3 on H200, trtllm_mha on Blackwell. The draft worker
//     cannot run trtllm_mha and auto-falls-back to flashinfer — measured identical to a
//     forced fa4 draft (82.5% vs 81.5% holdout, accept-len 4.63 both), so it is left auto.
//   - NEVER use --attention-backend triton for Laguna: 13.2% GSM8K (broken SWA handling)
//     plus a CUBLAS crash at tp=4 CUDA-graph capture.
//
// Draft/target precision ALWAYS matches: each quantized target pairs with the DFlash draft
// calibrated for it (…-DFlash, …-DFlash-FP8, …-DFlash-NVFP4, …-DFlash-INT4). The drafts
// themselves are small bf16 5-layer models (~0.9 GB) — the suffix is the calibration target.
//
// Memory: DFlash cells carry --mem-fraction-static 0.7 — at tp=4 on GB300 the default
// fraction OOMs in the draft vocab all-gather ("Failed to CUDA calloc"); 0.7 is validated.
// Dense cells use the default heuristic (validated at defaults on GB300).
//
// TP: 8-GPU HGX nodes (H200/B200) → --tp 8; GB300 (Grace-Blackwell, 4-GPU single node) → --tp 4.
// Adjust --tp to your node size (48 attn / 8 KV heads shard cleanly at 1/2/4/8).
//
// NVFP4 is Blackwell-only → no h200×nvfp4 cells (same rule as Laguna-M.1).
//
// verified:true = ran that command shape on that hardware and it served correctly + passed
// full GSM8K (see laguna-xs21-benchmarks.jsx). GB300 cells are verified (4×GB300, tp 4);
// H200/B200 cells are command-correct but pending measurement.

export const config = {
  modelName: "Laguna-XS-2.1",

  supportedHardware: ["h200", "b200", "gb300"],

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
  //                  few-stream serving; measured accept-length ~3.8–4.2 at tp=4 (~5.7–6.8 at tp=1).
  //   high-throughput = plain serving (no speculation) — batch-saturated workloads, where
  //                  speculation's draft+rejection overhead costs more than it saves.
  strategies: [
    { id: "low-latency",     label: "Low-Latency (DFlash)" },
    { id: "high-throughput", label: "High-Throughput" },
  ],

  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16":  "poolside/Laguna-XS-2.1",
    "default|fp8":   "poolside/Laguna-XS-2.1-FP8",
    "default|nvfp4": "poolside/Laguna-XS-2.1-NVFP4",
    "default|int4":  "poolside/Laguna-XS-2.1-INT4",
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
    // GSM8K is the required accuracy sanity on every verified cell (cookbook_guide §3), via sgl-eval.
    accuracy: {
      gsm8k_pct:
`# pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 128`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 128: 256, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // No variant-wide accuracy default; real numbers are per-cell in laguna-xs21-benchmarks.jsx.
  defaultAccuracy: {
    default: { gsm8k_pct: null },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  // High-Throughput (dense) runs on any current nightly (INT4 needs a build ≥ PR #29761's merge).
  // Low-Latency (DFlash) needs PR #29446 — build from that branch until it merges, then any
  // nightly at or past the merge commit.
  dockerImages: {
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
  },

  github: {
    cookbookModel: "poolside/Laguna-XS-2.1",
  },

  playgroundFeatures: {

    // Hybrid-SWA GQA model (48 Q / 8 KV heads) — TP shards cleanly at 1/2/4/8.
    // Accuracy verified TP-independent on the trtllm_mha backend (tp1 == tp4 on GB300).
    // No DP-Attention / CP knobs: unvalidated on this model family — not exposed.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },

    // Reasoning + tool-call parsers (poolside_v1, same family as Laguna-M.1 / XS.2).
    // ALSO baked into every Deploy cell below. The chat template auto-detects both
    // (`Auto-detected template features: reasoning_parser=poolside_v1, tool_call_parser=poolside_v1`),
    // so these are explicit-but-redundant on transformers ≥ 5.10.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser poolside_v1" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser poolside_v1" },
      ],
    },
  },

  // Cells: (h200 × {bf16,fp8,int4} + b200/gb300 × {bf16,fp8,nvfp4,int4}) × {low-latency, high-throughput}.
  // Draft model precision always matches the target's.
  cells: [

    // ══════════════ NVIDIA Hopper H200 (8-GPU HGX) — BF16 / FP8 / INT4 ══════════════
    // Dense auto-selects fa3 on Hopper (no flag). LL pins fa3 (DFlash-safe on Hopper;
    // with a spec algorithm active, auto would fall back to flashinfer).
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
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
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend fa3",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend fa3",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash-FP8",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // INT4 (mixed 4/8-bit compressed-tensors MoE) — needs a build ≥ PR #29761 (merged).
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
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
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend fa3",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash-INT4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ══════════════ NVIDIA Blackwell B200 (8-GPU HGX) — BF16 / FP8 / NVFP4 / INT4 ══════════════
    // Dense auto-selects trtllm_mha on Blackwell (no flag). LL MUST pin trtllm_mha —
    // with DFlash active, auto falls back to flashinfer, which is broken for this
    // hybrid-SWA model at tp≥4 (GSM8K 28% vs 76%; reproduced + bisected on GB300).
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash-FP8",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash-NVFP4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // INT4 (mixed 4/8-bit compressed-tensors MoE) — needs a build ≥ PR #29761 (merged).
      match: { hw: "b200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "b200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash-INT4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ══════════════ NVIDIA Grace-Blackwell GB300 (4-GPU single node) — VERIFIED ══════════════
    // All 8 cells ran on 4×GB300 (tp 4) with full-GSM8K accuracy (laguna-xs21-benchmarks.jsx):
    // dense via backend auto-select (resolves trtllm_mha), DFlash with trtllm_mha pinned.
    {
      // VERIFIED 4×GB300 tp4: GSM8K 75.66% (full 1319, greedy).
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
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
      // VERIFIED 4×GB300 tp4: GSM8K 76.19%, accept-length 4.17 (matched bf16 draft).
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
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED 4×GB300 tp4: GSM8K 71.87%.
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
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
      // VERIFIED 4×GB300 tp4: GSM8K 72.02%, accept-length 4.05 (matched fp8-calibrated draft).
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
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
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash-FP8",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED 4×GB300 tp4: GSM8K 78.39%.
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
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
      // VERIFIED 4×GB300 tp4: GSM8K 74.53%, accept-length 4.02 (matched nvfp4-calibrated draft).
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
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
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash-NVFP4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED 4×GB300 tp4: GSM8K 66.79%. Mixed 4/8-bit MoE — needs a build ≥ PR #29761 (merged).
      match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      verified: true,
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
      // VERIFIED 4×GB300 tp4: GSM8K 67.02%, accept-length 3.80 (matched int4-calibrated draft).
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
        "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash-INT4",
        "--page-size 1",
        "--mem-fraction-static 0.7",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
