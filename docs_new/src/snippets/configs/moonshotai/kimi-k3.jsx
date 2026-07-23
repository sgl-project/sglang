// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Recipes transcribed from the k3-track serving benchmark scripts
// (benchmark/H200/script/v1/launch-k3.sh, benchmark/B300/script/v1/launch-k3.sh).
// Kimi-K3 is a hybrid MoE VLM: 93 layers = 69 KDA (linear) + 24 MLA, 896 routed
// experts + 1 shared. Served today from the DarkSharpness/sglang-kimi fork.

export const config = {
  modelName: "Kimi-K3",

  // H200 (2×8 TP16/EP16), B300 (1×8 TP8) and GB300 (2×4 TP8 MNNVL) have validated
  // serving recipes in k3-track. B200 / GB200 / MI350X / MI355X are day-0 targets
  // whose recipes are pending — listed here (they grey out until a cell lands).
  supportedHardware: ["h200", "b300", "gb300", "b200", "gb200", "mi350x", "mi355x"],

  // No variant axis — single checkpoint.
  variants: [
    { id: "default", label: "Default" },
  ],
  // Weights ship MXFP4; served via the Marlin W4A16 runner (accuracy) or the
  // FlashInfer MXFP4 runner (throughput, Playground swap).
  quantizations: [
    { id: "mxfp4", label: "MXFP4" },
  ],
  // Three operating points: DSPARK/MTP speculative decoding for latency (spec ON),
  // the accuracy-preserving default (mem-frac 0.85), and the throughput-tuned config
  // (mem-frac 0.90 + extra_buffer_lazy, spec OFF).
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "balanced",        label: "Balanced"        },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  // `multi-N` id carries the node count for `--nnodes N`.
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  modelNames: {
    "default|mxfp4": "moonshotai/Kimi-K3",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    NODE0_IP:  { target: "command", label: "Head node IP",     default: "<node0-ip>"     },
    NODE_RANK: { target: "command", label: "This node rank",   default: "<node-rank>"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost"      },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"          },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // Reproduce command for the benchmark card's "⚡ Reproduce" modal. The k3-track
  // sweep is 8k-in / 1k-out random, num_prompts = 5 × concurrency, cache-cold.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    // num_prompts = 5 × concurrency (k3-track floor 16).
    numPromptsByConc: { 1: 16, 16: 80, 64: 320, 256: 1280, 1024: 5120 },
  },

  // K3 currently serves from the DarkSharpness/sglang-kimi fork; no public
  // lmsysorg tag carries it yet, so these are day-0 placeholders (nightly). GB300 /
  // GB200 ran on CUDA 13 kernels → -cu130; MI35x uses the ROCm image.
  dockerImages: {
    h200:   "lmsysorg/sglang:dev",
    b300:   "lmsysorg/sglang:dev",
    gb300:  "lmsysorg/sglang:dev-cu130",
    b200:   "lmsysorg/sglang:dev",
    gb200:  "lmsysorg/sglang:dev-cu130",
    mi350x: "lmsysorg/sglang:dev-rocm720-mi35x",
    mi355x: "lmsysorg/sglang:dev-rocm720-mi35x",
  },

  // Pre-selects the issue template's `model` field on "Submit verified cell".
  github: {
    cookbookModel: "moonshotai/kimi-k3",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // DP-Attention is a combined knob: value = DP degree AND toggles `--enable-dp-attention`.
    // K3's MLA latent KV is TP-replicated, so DP-attention RAISES per-GPU KV pressure —
    // dp=8/attn_tp=1 OOMs on a single node; use dp=2/attn_tp=4. No CP knob: K3 uses
    // decode context parallel (`--dcp-size`), a different lever from prefill `--attn-cp-size`.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null, 8,
          { value: 16, disable: { nodes: ["single"] },
            disableReason: "TP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
        ]},
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null, false, 2, 4,
            { value: 8, disable: { nodes: ["single"] },
              disableReason: "DP-Attention=8 with attn_tp=1 holds the full unsharded MLA KV per rank and OOMs on a single node — prefer dp=2/attn_tp=4." },
            { value: 16, disable: { nodes: ["single"] },
              disableReason: "DP-Attention=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card: "MoE Parallelism" -----
    // K3 = 896 routed experts + 1 shared. Marlin (W4A16) is the accuracy runner; the
    // MXFP4 / a2a runners (FlashInfer MXFP4, DeepEP, MegaMoE) are throughput levers.
    moe: {
      backend: {
        options: [
          { id: null,               label: "Inherited" },
          { id: "deepep",           label: "DeepEP",            flags: ["--moe-a2a-backend deepep"] },
          // Blackwell-only kernel-fusion path; selecting it reveals the Quantization sub-select.
          { id: "megamoe",          label: "MegaMoE",           flags: ["--moe-a2a-backend megamoe"],
            requiresHw: ["b200", "b300", "gb200", "gb300"] },
          { id: "flashinfer_mxfp4", label: "FlashInfer (MXFP4)", flags: ["--moe-runner-backend flashinfer_mxfp4"] },
          { id: "marlin",           label: "Marlin (W4A16)",    flags: ["--moe-runner-backend marlin"] },
        ],
      },
      // MegaMoE quantization sub-select — shown only when backend === "megamoe".
      megamoeQuant: {
        stripEnv: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
        options: [
          { id: "w4a8", label: "W4A8",
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
          { id: "w4a4", label: "W4A4",
            env: [
              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1",
              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1",
            ] },
        ],
      },
      ep: { label: "EP", values: [
        null, 1, 2, 4, 8,
        { value: 16, disable: { nodes: ["single"] },
          disableReason: "EP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
      ]},
    },

    // ----- Card: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser kimi_k3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser kimi_k3" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----
    // DSPARK is K3's MTP path (γ = block size; verify window = γ+1). The benchmark
    // recipes run NOSPEC; DSPARK here is an experimentation starting point.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "dspark",  label: "DSPARK (MTP)",
          flags: ["--speculative-algorithm DSPARK", "--speculative-dspark-block-size 7"] },
      ],
    },

    // ----- Card: "PD Disaggregation" ----- (validated functionally on B300×2 mooncake)
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake",
          env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1"],
          envWhen: { hw: ["gb200", "gb300"] } },
        { id: "nixl", label: "NiXL" },
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

    // ----- Card: "Hierarchical KV Cache" ----- (K3 hybrid L1/L2/L3, incl. KDA state)
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

    // ----- Axis: Flag Selects (K3 hybrid dual-pool knobs) -----
    // KDA state pool vs full-KV pool levers (k3-track cookbook/hybrid-ratio.md). The
    // flagless option is the accuracy-safe default; the others are capacity/long-ctx
    // levers whose accuracy A/B is workload-gated — Playground opt-ins, not cells.
    flagSelects: [
      {
        id: "kvCacheDtype", title: "KV Cache Precision",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "auto", label: "Auto (BF16)" },
          { id: "fp8",  label: "FP8 (E4M3) — long-ctx lever", flags: ["--kv-cache-dtype fp8_e4m3"] },
        ],
      },
      {
        id: "mambaSsmDtype", title: "Mamba SSM State Precision",
        stripPrefixes: ["--mamba-ssm-dtype"],
        options: [
          { id: "auto", label: "Auto (FP32)" },
          { id: "bf16", label: "BFloat16 — ~2× state slots", flags: ["--mamba-ssm-dtype bfloat16"] },
        ],
      },
      {
        id: "mambaRadix", title: "Mamba Radix Cache Strategy",
        stripPrefixes: ["--mamba-radix-cache-strategy"],
        options: [
          { id: "auto",  label: "Auto (extra_buffer)" },
          { id: "lazy",  label: "extra_buffer_lazy — +conc", flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
          { id: "nobuf", label: "no_buffer",                 flags: ["--mamba-radix-cache-strategy no_buffer"] },
        ],
      },
    ],
  },

  cells: [
    // ====================================================================
    // B300 — single node, TP8, MXFP4 Marlin (W4A16), trtllm_mla decode.
    // From benchmark/B300/script/v1/launch-k3.sh (commit d70f59487).
    // Keep the verified Balanced cell first: it is the command panel default.
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_MM_FEATURE_CACHE_MB=1024",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 131072",
        "--moe-runner-backend marlin",
        "--decode-attention-backend trtllm_mla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--mm-feature-transport cuda_ipc",
        "--mm-processor-worker-num 2",
        "--mm-io-worker-num 16",
        "--skip-server-warmup",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Low-latency: balanced recipe + DSPARK/MTP speculative decoding (γ=7).
      // MTP has not yet completed a B300 serving benchmark round, so this cell
      // remains unverified.
      match: { hw: "b300", variant: "default", quant: "mxfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [
        "SGLANG_MM_FEATURE_CACHE_MB=1024",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 131072",
        "--moe-runner-backend marlin",
        "--decode-attention-backend trtllm_mla",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path moonshotai/Kimi-K3-DSpark",
        "--speculative-dspark-block-size 7",
        "--speculative-draft-attention-backend flashinfer",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--mm-feature-transport cuda_ipc",
        "--mm-processor-worker-num 2",
        "--mm-io-worker-num 16",
        "--skip-server-warmup",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H200 — 2×8, TP16/EP16, MXFP4 Marlin, flashmla decode. From
    // benchmark/H200/script/v1/launch-k3.sh (commit 0dc8b030d).
    // Same block runs on both ranks; NIC + host-IP env go in multiNodeHints.
    // ====================================================================
    {
      // Low-latency: balanced recipe + DSPARK/MTP speculative decoding (γ=7, verify
      // window 8). Draft-attention flashinfer is the current correctness path. NOT yet
      // benchmarked as a serving round on H200 (the E01 sweep was NOSPEC) → unverified.
      match: { hw: "h200", variant: "default", quant: "mxfp4", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path moonshotai/Kimi-K3-DSpark",
        "--speculative-dspark-block-size 7",
        "--speculative-draft-attention-backend flashinfer",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--mamba-full-memory-ratio 0.9",
        "--cuda-graph-max-bs 64",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Accuracy-preserving default (eval-command: mem-frac 0.85, graph-bs 64).
      match: { hw: "h200", variant: "default", quant: "mxfp4", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--mamba-full-memory-ratio 0.9",
        "--cuda-graph-max-bs 64",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Throughput-tuned (perf-command: mem-frac 0.90, graph-bs 256, extra_buffer_lazy → max_running 98).
      match: { hw: "h200", variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.90",
        "--mamba-full-memory-ratio 0.9",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--cuda-graph-max-bs 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 — 2×4 (2 nodes), TP8 MNNVL. Baseline TP8/marlin/trtllm_mla from
    // p0-performance E02 (server_launch.sh, commit 23cadbc0); EP8+MegaMoE+SP-MoE
    // high-throughput from E05 (commit 4ec77e5ed, bs=32 tpt/GPU 856.9).
    // ====================================================================
    {
      // Low-latency: baseline TP8 + DSPARK/MTP (γ=7). Assembled from the baseline
      // recipe + spec flags; MTP serving round not committed as one script → unverified.
      match: { hw: "gb300", variant: "default", quant: "mxfp4", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--decode-attention-backend trtllm_mla",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path moonshotai/Kimi-K3-DSpark",
        "--speculative-dspark-block-size 7",
        "--speculative-draft-attention-backend flashinfer",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Balanced: baseline TP8 (marlin, trtllm_mla, mem-frac 0.85, max_running 128).
      match: { hw: "gb300", variant: "default", quant: "mxfp4", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--decode-attention-backend trtllm_mla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 128",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // High-throughput: EP8 + MegaMoE + SP-MoE (auto) + cutedsl_mla. bs=32 tpt/GPU
      // 856.9 (first > TP8 baseline), serving TTFT −43~71%, GSM8K 0.990 (E05).
      match: { hw: "gb300", variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--moe-a2a-backend megamoe",
        "--decode-attention-backend cutedsl_mla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--max-total-tokens 620000",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],

  // H200 cross-node fabric env (bond0 in the k3-track devbox — substitute your NIC).
  multiNodeHints: {
    h200: [
      "Multi-node K3 needs the cross-node NIC pinned on BOTH ranks:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # e.g. bond0",
      "  NCCL_SOCKET_IFNAME=<your-nic>   # force NCCL off kube-ipvs0",
      "  SGLANG_HOST_IP=<this-node-ip>",
    ],
  },
};
