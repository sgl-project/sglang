// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Faithful migration of the legacy GLM-5.1 generator (single source of truth).
// All cells are UNVERIFIED (yellow) — a migration is a faithful port, not a
// re-verification. The legacy `dpattention` radio is subtitled "Low Latency" /
// "High Throughput", so it maps to the strategies dimension (low-latency =
// DP-Attention off; high-throughput = DP-Attention on). Speculative decoding
// (EAGLE) was the legacy default-ON toggle, hidden on AMD — so it bakes into
// BOTH NVIDIA strategy tiers and is also a Playground `speculative` axis; AMD
// cells carry no spec flags (the legacy widget hid the toggle there).
// Reasoning / tool-call parsers are Playground-only (never in cells).

export const config = {
  modelName: "GLM-5.1",

  // Catalog ids as-is (all present in HARDWARE_CATALOG — no config.hardware needed).
  supportedHardware: [
    "h100", "h200", "b300", "gb300",
    "mi300x", "mi325x", "mi355x",
  ],

  // No variant axis on the legacy page — single deployable family.
  variants: [
    { id: "default", label: "Default" },
  ],
  // The legacy quant radio is hardware-pinned (the generator forces the
  // effective quant by hardware): H100/H200 → FP8, B300/GB300 → NVFP4,
  // AMD → BF16. Each hw therefore ships exactly one quant; per-hw greying
  // falls out of which cells exist.
  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
  ],
  // Two operating points per (hw × quant) — the legacy `dpattention` radio's
  // "Low Latency" / "High Throughput" subtitles.
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "zai-org/GLM-5.1",
    "default|fp8":  "zai-org/GLM-5.1-FP8",
    // NVFP4 ships under the nvidia org (B300/GB300 only).
    "default|nvfp4": "nvidia/GLM-5.1-NVFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"   },
    PORT:      { target: "command", label: "Bind port",        default: "30000"     },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"     },
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
`python3 benchmark/gsm8k/bench_sglang.py --port {{CURL_PORT}}`,
      mmlu_pct:
`python3 benchmark/mmlu/bench_sglang.py --port {{CURL_PORT}}`,
    },
    numPromptsByConc: { 1: 10, 100: 1000 },
  },

  // The eval set rendered in the benchmark card + "⚡ Reproduce" (the engine
  // ships no default — every config declares its own). GSM8K + MMLU are the
  // suites the legacy page reported; accuracy was measured on GLM-5 (see the
  // benchmarks file header) and applied per-entry there, so no defaultAccuracy.
  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
    ["mmlu_pct",  "MMLU",  "%"],
  ],

  dockerImages: {
    h100:  "lmsysorg/sglang:dev",
    h200:  "lmsysorg/sglang:dev",
    // B300 / GB300 require the CUDA 13 image variant (legacy §3.2 note).
    b300:  "lmsysorg/sglang:dev-cu130",
    gb300: "lmsysorg/sglang:dev-cu130",
    mi300x: "lmsysorg/sglang:dev-rocm720-mi30x",
    mi325x: "lmsysorg/sglang:dev-rocm720-mi30x",
    mi355x: "lmsysorg/sglang:dev-rocm720-mi35x",
  },

  // Pre-selects the issue template's `model` field on "Submit verified cell".
  github: {
    cookbookModel: "zai-org/glm-5.1",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // DP-Attention is a combined knob: value is the DP degree AND toggles `--enable-dp-attention`.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null, 1, 2, 4, 8,
          { value: 16, disable: { nodes: ["single"] },
            disableReason: "TP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
        ]},
        { id: "cp",     label: "CP", values: [null, 1, 2, 4] },
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null, false, 1, 2, 4, 8,
            { value: 16, disable: { nodes: ["single"] },
              disableReason: "DP-Attention=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card: "MoE Parallelism" -----  GLM-5.1 is a DSA + MoE model
    // (shares DeepSeek-V3.2 structure). MegaMoE is DSv4-Blackwell-only → omitted.
    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [
        null, 1, 2, 4, 8,
        { value: 16, disable: { nodes: ["single"] },
          disableReason: "EP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
      ]},
    },

    // ----- Card: "Parsers" -----  Reasoning + tool-call (never baked into cells).
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser glm45" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser glm47" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----  EAGLE is the legacy default
    // (baked into NVIDIA cells); the preset lets users re-apply or turn it off.
    // Not supported on AMD for GLM-5.1 (legacy §3.2).
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"],
          disable: { hw: ["mi300x", "mi325x", "mi355x"] },
          disableReason: "EAGLE speculative decoding is not currently supported on AMD for GLM-5.1." },
      ],
    },

    // ----- Card: "PD Disaggregation" -----
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl",     label: "NiXL" },
      ],
      // `auto` is a sentinel (emits no --disaggregation-ib-device flag).
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:30000 \\
  --decode http://<decode-host>:30001 \\
  --host 0.0.0.0 --port 8000 \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`,
      },
    },

    // ----- Card: "Hierarchical KV Cache" -----
    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "hf3fs",    label: "HF3FS" },
        { id: "nixl",     label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },

    // ----- Card: "HiSparse" -----  GLM-5.1 is a DSA model (shares DeepSeek-V3.2
    // structure; see the HiSparse guide). Decode-only: shown when live PD-Disagg
    // mode is `decode`.
    hisparse: {
      requiredFlags: ["--disable-radix-cache"],
      config: { top_k: 2048, device_buffer_size: 6144 },
      hostRatios: [
        { id: 5,  label: "5 (~1TB host)" },
        { id: 10, label: "10 (~2TB host)" },
      ],
      defaultHostRatio: 10,
    },
  },

  // 14 cells = 4 NVIDIA hw × 2 strategies + 3 AMD hw × 2 strategies.
  // All UNVERIFIED (faithful migration). Token sets equal the legacy generator's
  // output per (hw, quant, strategy) — re-sorted to canonical flag order, with
  // {{HOST_IP}}/{{PORT}} appended. NVIDIA cells bake the legacy default-ON
  // EAGLE spec flags into BOTH tiers; AMD cells omit them (legacy hid the toggle).
  cells: [
    // ====================================================================
    // H200 + FP8  (tp=8, mem 0.85) — page default hardware
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H100 + FP8  (tp=16, mem 0.85)
    // ====================================================================
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 16",
        "--enable-dp-attention",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 + NVFP4  (tp=8, mem 0.80) — modelopt_fp4 + trust-remote-code
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
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
        "--quantization modelopt_fp4",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 + NVFP4  (tp=4, mem 0.80) — modelopt_fp4 + trust-remote-code
    // ====================================================================
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
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
        "--quantization modelopt_fp4",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // AMD — MI300X + FP8  (tp=8, mem 0.80)
    // TileLang DSA backends; no speculative decoding (hidden on AMD).
    // FP8 weights (zai-org/GLM-5.1-FP8) verified and benchmarked on MI300X.
    // KV cache auto-set to bfloat16 by the DSA backend on SM9 device.
    // ====================================================================
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // AMD — MI300X / MI325X / MI355X + BF16  (tp=8, mem 0.80)
    // TileLang DSA backends; no speculative decoding (hidden on AMD).
    // ====================================================================
    {
      match: { hw: "mi300x", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
