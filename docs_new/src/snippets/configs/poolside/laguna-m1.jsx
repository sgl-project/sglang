// Laguna-M.1 (poolside) — config-driven cookbook page.
// Consumed by the shared _deployment.jsx + _playground.jsx engines (no model code there).
//
// Build: M.1 needs SGLang PR #28400 (softplus per-element output gating, MERGED) AND PR #28604
// (global-attention SWA fix — M.1 is sliding_window=0 / all-global; without it M.1 crashes ~1s
// into any concurrent batch with AssertionError: ... swa_lock_ref=0). Both are merged on main
// (verified on a 3f668733 build). The shipped recipe carries NO workaround flag, but the pinned
// build MUST contain BOTH; the #28400-merge wheel 0.5.14.dev20260618+g343aeeef39 is #28400-ONLY
// and crashes under load. Pin dockerImages + benchmarks.sglang_version to a build at a commit
// ≥ #28604. See /sgl-workspace/laguna-m1-day0-checklist.md (step 2) + laguna-m1-results.md.
//
// Model is now natively supported (#28400) → NO --trust-remote-code needed.
//
// Hardware: H200 (Hopper) + B200/B300/GB200/GB300 (Blackwell).
//   - BF16 runs everywhere.
//   - FP8 is HOPPER-ONLY — not compatible with Blackwell (use NVFP4 for low-precision there).
//   - NVFP4 is Blackwell-only.
// So the only quant×hw combos with a cell are: H200×{BF16,FP8} and each Blackwell×{BF16,NVFP4}.
// TP: 8-GPU HGX nodes (H200/B200/B300) → --tp 8 (the maintainer's baseline); GB200/GB300
// (Grace-Blackwell, typically 4-GPU single node) → --tp 4. Adjust --tp to your node size.
//
// Strategy: a SINGLE "Balanced" operating point (maintainer decision — no LL/HT split; an
// earlier TP=8+DP-Attention "high-throughput" idea was dropped: DP-Attention is ~15% SLOWER
// on this GQA model, see laguna-m1-results.md).

export const config = {
  modelName: "Laguna-M.1",

  supportedHardware: ["h200", "b200", "b300", "gb200", "gb300"],

  variants: [
    { id: "default", label: "Default" },
  ],

  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
  ],

  // Single balanced operating point (maintainer decision — no LL/HT split).
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],

  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16":  "poolside/Laguna-M.1",
    "default|fp8":   "poolside/Laguna-M.1-FP8",
    "default|nvfp4": "poolside/Laguna-M.1-NVFP4",
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
    // GSM8K sanity is the REQUIRED minimum on every verified cell (cookbook_guide §3); AIME 25
    // (thinking ON) is the harder accuracy check (model_support_guide). All via sgl-eval.
    // NOTE: M.1 needs enable_thinking, not sgl-eval's --thinking key (which the template ignores)
    // — thinking evals were run via the enable_thinking wrapper (laguna-m1-results.md).
    accuracy: {
      gsm8k_pct:
`# pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 128`,
      aime25_pct:
`# pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 --thinking \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 128: 256, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // Accuracy here varies by QUANT (BF16 vs NVFP4), so real numbers live per-cell in
  // laguna-m1-benchmarks.jsx (which overrides this). Keep null = no variant-wide default.
  defaultAccuracy: {
    default: { gsm8k_pct: null, aime25_pct: null },
  },

  accuracyLabels: [
    ["gsm8k_pct",  "GSM8K",   "%"],
    ["aime25_pct", "AIME 25", "%"],
  ],

  // Pinned nightly with the Laguna-M.1 build (PR #28400 + #28604 + #28649; cu13 covers H200 + all Blackwell).
  // dev-cu13-618-nightly was generated after the FP8 g_proj fix (#28649) landed, so it serves H200 FP8 as
  // well as BF16/NVFP4. (Equivalent pip nightly: 0.5.14.dev20260618+g97e3b8998d.)
  dockerImages: {
    h200:  "lmsysorg/sglang:dev-cu13-618-nightly",
    b200:  "lmsysorg/sglang:dev-cu13-618-nightly",
    b300:  "lmsysorg/sglang:dev-cu13-618-nightly",
    gb200: "lmsysorg/sglang:dev-cu13-618-nightly",
    gb300: "lmsysorg/sglang:dev-cu13-618-nightly",
  },

  github: {
    cookbookModel: "poolside/Laguna-M.1",
  },

  playgroundFeatures: {

    // M.1 is global-attention (no SWA); expose TP + DP-Attention. No CP.
    // DP-Attention is a Playground experiment only — ~15% slower than plain TP on this GQA
    // model (8 KV heads), so it is NOT in the shipped Balanced recipe.
    attention: {
      knobs: [
        { id: "tp",     label: "TP",           values: [null, 1, 2, 4, 8] },
        { id: "dpAttn", label: "DP-Attention", values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // 256-expert top-16 MoE. DeepEP all-to-all + EP degree.
    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    // Reasoning + tool-call parsers (poolside_v1, same family as Laguna-XS.2). ALSO baked into
    // every Deploy cell below (the maintainer's baseline carries them).
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser poolside_v1" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser poolside_v1" },
      ],
    },
  },

  // One Balanced cell per valid (hw × quant): H200×{BF16,FP8}, each Blackwell×{BF16,NVFP4}.
  // FP8 is Hopper-only (no Blackwell FP8 cell — those combos grey out). Baseline recipe
  // (parsers poolside_v1, NO --trust-remote-code) baked into every cell.
  // TP: H200/B200/B300 = --tp 8; GB200/GB300 = --tp 4 (4-GPU single node).
  // verified:true = ran that exact command on that hardware and it served correctly + passed a
  // GSM8K-class eval. Absent verified = yellow/unverified badge.
  cells: [
    // ===== NVIDIA Hopper (H200) — BF16 / FP8 (NVFP4 is Blackwell-only) =====
    {
      // VERIFIED on 8xH200 (BF16, tp8): GSM8K 93.02% + perf (laguna-m1 H200 results).
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED on 8xH200 (FP8, tp8): GSM8K 93.25 + AIME25 0.50; g_proj FP8 quant fix validated.
      // FP8 needs the g_proj fix (PR #28649, MERGED) on top of #28400+#28604 — the pinned
      // dev-cu13-618-nightly image includes all three.
      // FP8 is Hopper-only — it is NOT compatible with Blackwell, so there is no Blackwell FP8 cell.
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Blackwell B200 (8-GPU HGX) — BF16 / NVFP4 (FP8 is Hopper-only) =====
    {
      // VERIFIED on 8xB200 (BF16, tp8): served clean under batched shared-prefix load,
      // GSM8K 91.88% + AIME25 66.88% (laguna-m1-results.md).
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED on 8xB200 (NVFP4, tp8): GSM8K 89.38% (laguna-m1-results.md). tp8 now matches
      // the shipped recipe.
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Blackwell B300 (8-GPU HGX) — BF16 / NVFP4 (UNVERIFIED; FP8 is Hopper-only) =====
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Grace-Blackwell GB200 (4-GPU single node) — BF16 / NVFP4 (UNVERIFIED; FP8 Hopper-only) =====
    {
      match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Grace-Blackwell GB300 (4-GPU single node) — BF16 / NVFP4 (UNVERIFIED; FP8 Hopper-only) =====
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
