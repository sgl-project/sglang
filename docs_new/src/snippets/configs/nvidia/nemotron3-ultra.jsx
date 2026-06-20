// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Migrated faithfully from the legacy generator
// docs_new/src/snippets/autoregressive/ (NVIDIA Nemotron3-Ultra). All cells are
// UNVERIFIED (yellow): this is a transcription of the legacy interactive
// generator, not a re-measurement. See the PR body for the strategy
// tier-mapping table (needs maintainer sign-off) and the day-0 benchmark
// reproducibility decision.

export const config = {
  modelName: "NVIDIA Nemotron3-Ultra",

  // NVIDIA-only. NVFP4 is Blackwell-only (b200/b300/gb200/gb300) — never on
  // h100/h200. BF16 spans Hopper (16×, 2-node) + Blackwell (8×). Which (quant ×
  // hw) combos actually exist is governed by the legacy SUPPORT matrix → cell
  // presence (no cell == greyed out).
  supportedHardware: [
    "h100", "h200", "b200", "b300", "gb200", "gb300",
  ],

  // No deployable model-name variant axis — the legacy "Model" radio is the
  // QUANTIZATION dim (BF16 / NVFP4), handled below.
  variants: [
    { id: "default", label: "Default" },
  ],
  // 3rd dim — the legacy "Model" radio. NVFP4 is the legacy default (Blackwell only).
  quantizations: [
    { id: "nvfp4", label: "NVFP4" },
    { id: "bf16",  label: "BF16"  },
  ],
  // 4th dim. The legacy `dpattention` radio carries the page's own operating-point
  // subtitles ("Low latency" / "High throughput") → it drives the strategies dim.
  // Two named poles → low-latency + high-throughput (no `balanced`):
  //   low-latency     = dp-attention OFF + MTP on  + EP off  (the legacy default)
  //   high-throughput = dp-attention ON  + MTP off + EP on
  // PROPOSED — needs maintainer sign-off (three stacked perf controls; see PR body).
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  // `multi-N` id carries the node count for `--nnodes N`. BF16 on H100/H200 is
  // 16× across 2 nodes (legacy `tp:16, multinode:true`).
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  modelNames: {
    "default|nvfp4": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    "default|bf16":  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    NODE0_IP:  { target: "command", label: "Head node IP",     default: "<head-node-ip>" },
    NODE_RANK: { target: "command", label: "This node rank",   default: "<node-rank>"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost"      },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"          },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // Reproduce commands for the Benchmark card's "⚡ Reproduce" modal. Kept even
  // though no measured numbers survive (the legacy "main branch" result is
  // non-reproducible — dropped), so the modal still guides re-measurement
  // against a pinned build.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    accuracy: {
      gsm8k_pct:
`python3 benchmark/gsm8k/bench_sglang.py`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // The eval set the legacy page measured (GSM8K). Required for accuracy rows to
  // render. No numbers are shipped (the legacy result was dropped as
  // non-reproducible) — this only labels the row + the ⚡Reproduce accuracy chip.
  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  // Dedicated dev images — Nemotron3-Ultra support is "not in any stable
  // release". CUDA 13 default; CUDA 12 variant exists (see §Install). Verbatim
  // from the legacy install section.
  dockerImages: {
    h100:  "lmsysorg/sglang:dev-nemotron3-ultra",
    h200:  "lmsysorg/sglang:dev-nemotron3-ultra",
    b200:  "lmsysorg/sglang:dev-nemotron3-ultra",
    b300:  "lmsysorg/sglang:dev-nemotron3-ultra",
    gb200: "lmsysorg/sglang:dev-nemotron3-ultra",
    gb300: "lmsysorg/sglang:dev-nemotron3-ultra",
  },

  // Pre-selects the issue template's `model` field on "Submit verified cell".
  github: {
    cookbookModel: "nvidia/nemotron3-ultra",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // DP-Attention is a combined knob: value is the DP degree AND toggles
    // `--enable-dp-attention`. `--dp` must divide `--tp`. The legacy widget
    // verifies DP=2 for BF16 and DP∈{2,4,8} for NVFP4 (capped by TP); the
    // Playground exposes the full set for experimentation.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null, 4, 8,
          { value: 16, disable: { nodes: ["single"] },
            disableReason: "TP=16 is the 2-node BF16 setup — switch the Deploy panel's Nodes to Multi-Nodes first." },
        ]},
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null, false, 2, 4, 8,
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card: "MoE Parallelism" -----
    // Hybrid MoE. The legacy widget only supports ep_size == 1 (off) or
    // ep_size == tp_size (on) — surfaced here as the EP knob (set EP = TP).
    moe: {
      ep: { label: "EP", values: [
        null, 4, 8,
        { value: 16, disable: { nodes: ["single"] },
          disableReason: "EP=16 (= TP=16) is the 2-node BF16 setup — switch the Deploy panel's Nodes to Multi-Nodes first." },
      ]},
    },

    // ----- Card: "Parsers" -----
    // Reasoning + tool-call parsers. Add-only (never baked into Deploy cells).
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser nemotron_3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----
    // MTP via EAGLE (the legacy "Multi-token Prediction" toggle, default ON).
    // Baked into the low-latency cells; the high-throughput cells strip it.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
      ],
    },

    // ----- Card(s): "KV Cache DType" (generic flagSelects) -----
    // The legacy "KV Cache DType" radio (None / fp8_e4m3 / bf16). None is the
    // accuracy-safe default and emits NO flag (flagless option); fp8_e4m3 is
    // accuracy-degrading so it lives here as a Playground option, never in cells.
    flagSelects: [
      {
        id: "kvCacheDtype",
        title: "KV Cache DType",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "none",     label: "None" },
          { id: "fp8_e4m3", label: "fp8_e4m3", flags: ["--kv-cache-dtype fp8_e4m3"] },
          { id: "bf16",     label: "bf16",     flags: ["--kv-cache-dtype bf16"] },
        ],
      },
    ],
  },

  // ============================================================================
  // Cells — ALL UNVERIFIED (yellow): faithful transcription of the legacy
  // generator, parsers-OFF + KV-dtype=None baseline. 16 cells = 8 low-latency +
  // 8 high-throughput, over the 8 SUPPORT-matrix (quant × hw) combos. Token
  // audit: 16/16 identical to the legacy generator output. cells[0] mirrors the
  // legacy default selection (NVFP4 / B200 / MTP-on / dp-off).
  // ============================================================================
  cells: [
    // ==================== NVFP4 (Blackwell only) ====================
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--ep 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--ep 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--ep 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--ep 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==================== BF16 ====================
    // H100 / H200: 16× across 2 nodes (renderer injects --nnodes/--node-rank/
    // --dist-init-addr). B200 / B300: 8× single node.
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 2",
        "--enable-dp-attention",
        "--ep 16",
        "--mamba-scheduler-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 2",
        "--enable-dp-attention",
        "--ep 16",
        "--mamba-scheduler-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 2",
        "--enable-dp-attention",
        "--ep 8",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 2",
        "--enable-dp-attention",
        "--ep 8",
        "--mamba-scheduler-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
