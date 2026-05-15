// DeepSeek-V4 Playground.
//
// Inherits the verified base from §3 (read from URL hash) and lets users layer
// 3 feature axes on top: Attention+MoE Parallelism, Parsers, Speculative
// Decoding. Live command regeneration with inline diff styling against the
// verified base. Same Copy / cURL / Env action buttons as §3.
//
// === MINTLIFY CONSTRAINT ===
// Mintlify's snippet loader strips module-level state and refuses to share
// data across `.jsx` files. So the cell catalog + config object below is
// DUPLICATED from deepseek-v4-deployment.jsx — when you edit cells in one
// file, update the other. Both blocks are marked with "KEEP IN SYNC" sentinels.
export const DeepSeekV4Playground = () => {
  // ==========================================================================
  // === KEEP IN SYNC WITH deepseek-v4-deployment.jsx (begin) ===
  // The cell catalog and per-cookbook config below are duplicated verbatim
  // from the §3 deployment snippet. When changing one, update the other.
  // ==========================================================================
  const HARDWARE_CATALOG = {
    nvidia: [
      { id: "h100",  label: "H100",  vram: "80GB"  },
      { id: "h200",  label: "H200",  vram: "141GB" },
      { id: "b200",  label: "B200",  vram: "192GB" },
      { id: "b300",  label: "B300",  vram: "288GB" },
      { id: "gb200", label: "GB200", vram: "192GB" },
      { id: "gb300", label: "GB300", vram: "288GB" },
    ],
    amd: [
      { id: "mi300x", label: "MI300X", vram: "192GB" },
      { id: "mi325x", label: "MI325X", vram: "256GB" },
      { id: "mi350x", label: "MI350X", vram: "288GB" },
      { id: "mi355x", label: "MI355X", vram: "288GB" },
    ],
  };

  // ==========================================================================
  // 2. Cell catalog — the only section that differs per cookbook
  // ==========================================================================

  // ----- 2.a Shared flag fragments -----
  const HEAD = ["--trust-remote-code", "--model-path {{MODEL_NAME}}"];
  const TAIL = ["--host {{HOST_IP}}", "--port {{PORT}}"];

  const MTP_314 = [
    "--speculative-algo EAGLE",
    "--speculative-num-steps 3",
    "--speculative-eagle-topk 1",
    "--speculative-num-draft-tokens 4",
  ];
  const MTP_112 = [
    "--speculative-algo EAGLE",
    "--speculative-num-steps 1",
    "--speculative-eagle-topk 1",
    "--speculative-num-draft-tokens 2",
  ];
  // DeepEP large-SMS config (single-quoted JSON so users can copy-paste without escaping).
  const DEEPEP_LARGE_SMS =
    `--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'`;
  // B200/B300 Pro accuracy-verified env block (same 5 vars across all 3 strategies for b200|big).
  const B200_PRO_ACC_ENV = [
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0",
    "SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1",
    "SGLANG_OPT_USE_JIT_NORM=1",
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA=1",
    "SGLANG_OPT_USE_TOPK_V2=1",
  ];

  // ----- 2.b Cell factory -----
  const cellOf = (match, { verified = true, env = [], flags = [] } = {}) => ({
    match, verified, env, flags,
  });
  const cloneToHw = (cells, fromHw, toHw) =>
    cells.filter((c) => c.match.hw === fromHw)
         .map((c) => ({ ...c, match: { ...c.match, hw: toHw } }));

  // ----- 2.c Cells -----
  const baseCells = [
    // ----- B200 + FP4 -----
    cellOf({ hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" }, {
      env: [],
      flags: [...HEAD, "--tp 4", "--moe-runner-backend flashinfer_mxfp4", ...MTP_314,
              "--chunked-prefill-size 4096", "--disable-flashinfer-autotune",
              "--swa-full-tokens-ratio 0.1", ...TAIL],
    }),
    cellOf({ hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" }, {
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", ...MTP_112, DEEPEP_LARGE_SMS, ...TAIL],
    }),
    cellOf({ hw: "b200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" }, {
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", DEEPEP_LARGE_SMS, ...TAIL],
    }),
    cellOf({ hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" }, {
      env: [...B200_PRO_ACC_ENV],
      flags: [...HEAD, "--tp 8", "--moe-runner-backend flashinfer_mxfp4", ...MTP_314,
              "--chunked-prefill-size 8192", "--disable-flashinfer-autotune",
              "--swa-full-tokens-ratio 0.1", "--mem-fraction-static 0.90", ...TAIL],
    }),
    cellOf({ hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" }, {
      env: [...B200_PRO_ACC_ENV],
      flags: [...HEAD, "--tp 8", "--dp 8", "--enable-dp-attention",
              "--moe-runner-backend flashinfer_mxfp4", "--disable-flashinfer-autotune",
              "--chunked-prefill-size 32768", "--swa-full-tokens-ratio 0.1", ...MTP_112,
              "--mem-fraction-static 0.92", "--cuda-graph-max-bs 256",
              DEEPEP_LARGE_SMS, ...TAIL],
    }),
    cellOf({ hw: "b200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" }, {
      env: [...B200_PRO_ACC_ENV,
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0",
            "NVSHMEM_DISABLE_IB=1",
            "SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1",
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"],
      flags: [...HEAD, "--tp 8", "--dp 8", "--enable-dp-attention",
              "--moe-a2a-backend megamoe", "--mem-fraction-static 0.835",
              "--cuda-graph-max-bs 544", "--swa-full-tokens-ratio 0.075",
              "--chunked-prefill-size 65536", "--tokenizer-worker-num 8",
              "--enable-prefill-delayer", DEEPEP_LARGE_SMS, ...TAIL],
    }),

    // ----- GB200 + FP4 -----
    cellOf({ hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" }, {
      env: [],
      flags: [...HEAD, "--tp 4", "--moe-runner-backend flashinfer_mxfp4", ...MTP_314,
              "--chunked-prefill-size 4096", "--disable-flashinfer-autotune",
              "--swa-full-tokens-ratio 0.1", ...TAIL],
    }),
    cellOf({ hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" }, {
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", ...MTP_112, DEEPEP_LARGE_SMS, ...TAIL],
    }),
    cellOf({ hw: "gb200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" }, {
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", DEEPEP_LARGE_SMS, ...TAIL],
    }),
    // GB200 Pro requires 2 nodes; multi-node wiring (--nnodes / --node-rank /
    // --dist-init-addr) added by the renderer below.
    cellOf({ hw: "gb200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" }, {
      env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [...HEAD, "--tp 8", "--moe-runner-backend flashinfer_mxfp4", ...MTP_314,
              "--chunked-prefill-size 4096", "--disable-flashinfer-autotune",
              "--swa-full-tokens-ratio 0.1", "--mem-fraction-static 0.88", ...TAIL],
    }),
    cellOf({ hw: "gb200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" }, {
      env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [...HEAD, "--tp 8", "--dp 8", "--enable-dp-attention",
              "--moe-a2a-backend deepep", ...MTP_112,
              "--mem-fraction-static 0.78", "--cuda-graph-max-bs 64",
              "--max-running-requests 128", ...TAIL],
    }),
    cellOf({ hw: "gb200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "multi-2" }, {
      env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [...HEAD, "--tp 8", "--dp 8", "--enable-dp-attention",
              "--moe-a2a-backend deepep", "--mem-fraction-static 0.78",
              "--cuda-graph-max-bs 64", "--max-running-requests 256", ...TAIL],
    }),

    // ----- GB300 + FP4 -----
    cellOf({ hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" }, {
      env: [],
      flags: [...HEAD, "--tp 4", "--moe-runner-backend flashinfer_mxfp4", ...MTP_314,
              "--chunked-prefill-size 4096", "--disable-flashinfer-autotune",
              "--swa-full-tokens-ratio 0.1", ...TAIL],
    }),
    cellOf({ hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" }, {
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", ...MTP_112, DEEPEP_LARGE_SMS, ...TAIL],
    }),
    cellOf({ hw: "gb300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" }, {
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", DEEPEP_LARGE_SMS, ...TAIL],
    }),
    cellOf({ hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" }, {
      env: [],
      flags: [...HEAD, "--tp 4", "--moe-runner-backend flashinfer_mxfp4", ...MTP_314,
              "--chunked-prefill-size 4096", "--disable-flashinfer-autotune",
              "--swa-full-tokens-ratio 0.1", "--mem-fraction-static 0.88", ...TAIL],
    }),
    cellOf({ hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" }, {
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", ...MTP_112,
              "--mem-fraction-static 0.9", "--cuda-graph-max-bs 128",
              "--max-running-requests 256", DEEPEP_LARGE_SMS, ...TAIL],
    }),
    cellOf({ hw: "gb300", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" }, {
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", "--mem-fraction-static 0.9",
              "--cuda-graph-max-bs 128", "--max-running-requests 256",
              DEEPEP_LARGE_SMS, ...TAIL],
    }),

    // ----- H200 + FP8 -----
    cellOf({ hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" }, {
      env: ["SGLANG_DSV4_FP4_EXPERTS=0"],
      flags: [...HEAD, "--tp 4", ...MTP_314, ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" }, {
      env: ["SGLANG_DSV4_FP4_EXPERTS=0", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", ...MTP_112,
              "--cuda-graph-max-bs 128", "--max-running-requests 128",
              DEEPEP_LARGE_SMS, ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "flash", quant: "fp8", strategy: "max-throughput", nodes: "single" }, {
      env: ["SGLANG_DSV4_FP4_EXPERTS=0", "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [...HEAD, "--tp 4", "--dp 4", "--enable-dp-attention",
              "--moe-a2a-backend deepep", "--cuda-graph-max-bs 128",
              "--max-running-requests 256", DEEPEP_LARGE_SMS, ...TAIL],
    }),
    // H200 Pro FP8: low-latency exposes BOTH single-node (TP=8 Marlin) and
    // multi-2 (TP=16 DP-attn + DeepEP) — the old combined block, split.
    cellOf({ hw: "h200", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "single" }, {
      env: ["SGLANG_DSV4_FP4_EXPERTS=0"],
      flags: [...HEAD, "--tp 8", "--moe-runner-backend marlin", ...MTP_314,
              "--chunked-prefill-size 4096", "--disable-flashinfer-autotune",
              "--mem-fraction-static 0.88", ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "multi-2" }, {
      env: ["SGLANG_DSV4_FP4_EXPERTS=0", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128"],
      flags: [...HEAD, "--tp 16", "--dp 16", "--enable-dp-attention",
              "--moe-a2a-backend deepep", "--cuda-graph-max-bs 8",
              "--max-running-requests 32", ...MTP_314,
              "--mem-fraction-static 0.88", ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "pro", quant: "fp8", strategy: "balanced", nodes: "multi-2" }, {
      env: ["SGLANG_DSV4_FP4_EXPERTS=0", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128"],
      flags: [...HEAD, "--tp 16", "--dp 16", "--enable-dp-attention",
              "--moe-a2a-backend deepep", ...MTP_112,
              "--mem-fraction-static 0.88", "--cuda-graph-max-bs 8",
              "--max-running-requests 32", ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "pro", quant: "fp8", strategy: "max-throughput", nodes: "multi-2" }, {
      env: ["SGLANG_DSV4_FP4_EXPERTS=0", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128"],
      flags: [...HEAD, "--tp 16", "--dp 16", "--enable-dp-attention",
              "--moe-a2a-backend deepep", "--mem-fraction-static 0.88",
              "--cuda-graph-max-bs 128", "--max-running-requests 256", ...TAIL],
    }),

    // ----- H200 + FP4 (Marlin, single-node only) -----
    cellOf({ hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 4", "--moe-runner-backend marlin", ...MTP_314, ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 4", "--moe-runner-backend marlin", ...MTP_112, ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 4", "--moe-runner-backend marlin", ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 8", "--moe-runner-backend marlin", ...MTP_314,
                       "--mem-fraction-static 0.88", ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 8", "--moe-runner-backend marlin", ...MTP_112,
                       "--mem-fraction-static 0.88", ...TAIL],
    }),
    cellOf({ hw: "h200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 8", "--moe-runner-backend marlin",
                       "--mem-fraction-static 0.88", ...TAIL],
    }),

    // ----- H100 + FP4 (Marlin) -----
    cellOf({ hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 8", "--moe-runner-backend marlin", ...MTP_314, ...TAIL],
    }),
    cellOf({ hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 8", "--moe-runner-backend marlin", ...MTP_112, ...TAIL],
    }),
    cellOf({ hw: "h100", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" }, {
      env: [], flags: [...HEAD, "--tp 8", "--moe-runner-backend marlin", ...TAIL],
    }),
    cellOf({ hw: "h100", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" }, {
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [...HEAD, "--tp 16", "--moe-runner-backend marlin", ...MTP_314,
              "--mem-fraction-static 0.9", "--cuda-graph-max-bs 8",
              "--max-running-requests 32", ...TAIL],
    }),
    cellOf({ hw: "h100", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" }, {
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [...HEAD, "--tp 16", "--moe-runner-backend marlin", ...MTP_112,
              "--mem-fraction-static 0.9", "--cuda-graph-max-bs 8",
              "--max-running-requests 32", ...TAIL],
    }),
    cellOf({ hw: "h100", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "multi-2" }, {
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [...HEAD, "--tp 16", "--moe-runner-backend marlin",
              "--mem-fraction-static 0.9", ...TAIL],
    }),
  ];

  // B300 mirrors B200 in the original generator. Materialise the duplicates.
  const allCells = [...baseCells, ...cloneToHw(baseCells, "b200", "b300")];

  const MULTI_NODE_HINTS = {
    gb200: [
      "The following env vars may be needed depending on your cluster:",
      "  GLOO_SOCKET_IFNAME=<your-nic>",
      "  NVSHMEM_ENABLE_NIC_PE_MAPPING=1",
      "  NVSHMEM_HCA_LIST=<your-hca-list>",
    ],
  };

  // ==========================================================================
  const config = {
    modelName: "DeepSeek-V4",
    // `supportedHardware` is the catalog-visibility list, NOT the
    // has-runnable-cells list. Listing AMD ids here makes those buttons appear
    // in the UI; since no cell references them, `isOptionAvailable` will
    // return false for each one and they render greyed out automatically.
    // Drop an id from this list to hide it from the UI entirely.
    supportedHardware: [
      "h100", "h200", "b200", "b300", "gb200", "gb300",
      "mi300x", "mi325x", "mi350x", "mi355x",
    ],
    variants: [
      { id: "flash", label: "Flash", subtitle: "285B" },
      { id: "pro",   label: "Pro",   subtitle: "1.6T" },
    ],
    quantizations: [
      { id: "fp8", label: "FP8" },
      { id: "fp4", label: "FP4" },
    ],
    strategies: [
      { id: "low-latency",    label: "Low-Latency"    },
      { id: "balanced",       label: "Balanced"       },
      { id: "max-throughput", label: "Max-Throughput" },
    ],
    // Nodes dimension. The id format `multi-N` carries the node count so the
    // renderer can emit `--nnodes N` automatically. We expose just one
    // multi-node option here using 2 nodes as the canonical example; cookbooks
    // that need a different N (e.g. 4 nodes) can add `multi-4` here AND in the
    // matching cells without any renderer change.
    nodesOptions: [
      { id: "single",  label: "Single Node" },
      { id: "multi-2", label: "Multi-Nodes" },
    ],
    // HF slug: layered lookup. `${hw}|${variant}|${quant}` (override) → `${variant}|${quant}` (base).
    // Both `--model-path` (via {{MODEL_NAME}} in cell.flags) and the cURL body's
    // `"model"` field resolve from the same map.
    modelNames: {
      "flash|fp4": "deepseek-ai/DeepSeek-V4-Flash",
      "flash|fp8": "deepseek-ai/DeepSeek-V4-Flash",
      "pro|fp4":   "deepseek-ai/DeepSeek-V4-Pro",
      "pro|fp8":   "deepseek-ai/DeepSeek-V4-Pro",
      // H200 FP8 needs the sgl-project repackaging (Hopper can't run FP4-mixed Instruct).
      "h200|flash|fp8": "sgl-project/DeepSeek-V4-Flash-FP8",
      "h200|pro|fp8":   "sgl-project/DeepSeek-V4-Pro-FP8",
    },
    placeholders: {
      HOST_IP:   { target: "command", label: "Bind host",       default: "0.0.0.0"  },
      PORT:      { target: "command", label: "Bind port",       default: "30000"    },
      NODE0_IP:  { target: "command", label: "Head node IP",    default: "<node0-ip>"   },
      NODE_RANK: { target: "command", label: "This node rank",  default: "<node-rank>"  },
      CURL_HOST: { target: "curl",    label: "Server host",     default: "localhost" },
      CURL_PORT: { target: "curl",    label: "Server port",     default: "30000"     },
    },
    curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,
    cells: allCells,
    multiNodeHints: MULTI_NODE_HINTS,
  };

  // === KEEP IN SYNC WITH deepseek-v4-deployment.jsx (end) ===

  // ==========================================================================
  // Playground-specific feature axis definitions
  // ==========================================================================

  // Speculative-decoding presets. `current` keeps the base cell's flags as-is.
  // `off` strips them. Named presets re-emit a known-good combo.
  const SPEC_PRESETS = [
    { id: "current",   label: "Inherited from base" },
    { id: "off",       label: "Off (greedy)" },
    { id: "mtp-314",   label: "EAGLE / MTP 3-1-4",
      flags: ["--speculative-algo EAGLE", "--speculative-num-steps 3",
              "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
    { id: "mtp-112",   label: "EAGLE / MTP 1-1-2",
      flags: ["--speculative-algo EAGLE", "--speculative-num-steps 1",
              "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 2"] },
    // Placeholders for future algorithms — listed but disabled until implemented.
    { id: "draftflash", label: "DraftFlash",  disabled: true,
      disabledReason: "Coming soon — pending DraftFlash kernel integration." },
    { id: "nextn",      label: "NextN",       disabled: true,
      disabledReason: "Coming soon — pending NextN algorithm support." },
  ];

  // Parallelism chip options. `null` = "inherited from base" (no override).
  const TP_OPTS = [null, 1, 2, 4, 8, 16];
  const DP_OPTS = [null, 1, 2, 4, 8, 16];
  const CP_OPTS = [null, 1, 2, 4];

  // MoE backends. `null` = inherited.
  const MOE_OPTS = [
    { id: null,                label: "Inherited" },
    { id: "deepep",            label: "DeepEP",            kind: "a2a" },
    { id: "megamoe",           label: "MegaMoE",           kind: "a2a" },
    { id: "flashinfer_mxfp4",  label: "FlashInfer (MXFP4)", kind: "runner" },
    { id: "marlin",            label: "Marlin (W4A16)",    kind: "runner" },
  ];
  const EP_OPTS = [null, 1, 2, 4, 8, 16];

  // ==========================================================================
  // Pure helpers (shape-identical to §3 where they overlap)
  // ==========================================================================
  const DIMENSIONS = ["hw", "variant", "quant", "strategy", "nodes"];
  const findCell = (cells, sel) =>
    cells.find((c) => DIMENSIONS.every((d) => c.match[d] === sel[d]));

  const resolveModelName = (sel) => {
    const triple = `${sel.hw}|${sel.variant}|${sel.quant}`;
    const pair = `${sel.variant}|${sel.quant}`;
    return config.modelNames[triple] ?? config.modelNames[pair] ?? "";
  };

  const interpolate = (text, env, modelName) =>
    text.replace(/{{(\w+)}}/g, (_, key) =>
      key === "MODEL_NAME" ? modelName : (env[key] ?? `{{${key}}}`));

  const parseNnodes = (id) => {
    if (id === "single") return 1;
    const m = /^multi-(\d+)$/.exec(id);
    return m ? parseInt(m[1], 10) : 1;
  };

  // Strip any flag whose first whitespace-delimited token equals one of
  // `prefixes`. Used to remove the base's values for an axis before re-emitting
  // the playground's choice. We must match exactly the first token because
  // values may contain hyphens / equals (e.g. `--moe-runner-backend marlin`).
  const stripFlagsByFirstToken = (flags, prefixes) => {
    const set = new Set(prefixes);
    return flags.filter((f) => {
      const head = f.split(/[\s=]/)[0];
      return !set.has(head);
    });
  };

  // Insert a list of new flags just before the trailing --host/--port pair so
  // the diff stays visually grouped with the existing structure.
  const insertBeforeTail = (flags, additions) => {
    const idx = flags.findIndex((f) => f.startsWith("--host"));
    const at = idx === -1 ? flags.length : idx;
    const out = flags.slice();
    out.splice(at, 0, ...additions);
    return out;
  };

  // Apply playground deltas on top of the base cell's flags.
  const applyDeltas = (baseFlags, d) => {
    let flags = [...baseFlags];

    // --- Attention parallelism overrides ---
    const attnTouched =
      d.attn.tp !== null || d.attn.dp !== null || d.attn.cp !== null ||
      d.attn.dpAttn !== null;
    if (attnTouched) {
      flags = stripFlagsByFirstToken(flags, [
        "--tp", "--dp", "--enable-dp-attention",
        "--enable-nsa-prefill-context-parallel", "--nsa-prefill-cp-mode",
      ]);
      const add = [];
      if (d.attn.tp !== null) add.push(`--tp ${d.attn.tp}`);
      if (d.attn.dp !== null && d.attn.dp > 1) add.push(`--dp ${d.attn.dp}`);
      if (d.attn.dpAttn === true) add.push("--enable-dp-attention");
      if (d.attn.cp !== null && d.attn.cp > 1) {
        add.push("--enable-nsa-prefill-context-parallel");
        add.push("--nsa-prefill-cp-mode round-robin-split");
      }
      // Insert right after --model-path to mirror §3 ordering.
      const at = flags.findIndex((f) => f.startsWith("--model-path")) + 1;
      flags.splice(at, 0, ...add);
    }

    // --- MoE parallelism overrides ---
    const moeBackend = MOE_OPTS.find((o) => o.id === d.moe.backend);
    if (d.moe.backend !== null || d.moe.ep !== null) {
      flags = stripFlagsByFirstToken(flags, [
        "--moe-a2a-backend", "--moe-runner-backend", "--ep",
      ]);
      const add = [];
      if (moeBackend && moeBackend.kind === "a2a") {
        add.push(`--moe-a2a-backend ${moeBackend.id}`);
      } else if (moeBackend && moeBackend.kind === "runner") {
        add.push(`--moe-runner-backend ${moeBackend.id}`);
      }
      if (d.moe.ep !== null && d.moe.ep > 1) add.push(`--ep ${d.moe.ep}`);
      // Insert right after parallelism block (after --tp if present, else
      // after --model-path).
      let at = flags.findIndex((f) => f.startsWith("--enable-dp-attention"));
      if (at === -1) at = flags.findIndex((f) => f.startsWith("--tp"));
      if (at === -1) at = flags.findIndex((f) => f.startsWith("--model-path"));
      flags.splice(at + 1, 0, ...add);
    }

    // --- Speculative decoding ---
    if (d.spec !== "current") {
      flags = stripFlagsByFirstToken(flags, [
        "--speculative-algo", "--speculative-num-steps",
        "--speculative-eagle-topk", "--speculative-num-draft-tokens",
      ]);
      const preset = SPEC_PRESETS.find((p) => p.id === d.spec);
      if (preset && preset.flags && preset.flags.length) {
        flags = insertBeforeTail(flags, preset.flags);
      }
    }

    // --- Parsers (toggle) ---
    flags = stripFlagsByFirstToken(flags, ["--reasoning-parser", "--tool-call-parser"]);
    const parserAdds = [];
    if (d.parsers.reasoning) parserAdds.push("--reasoning-parser deepseek-v4");
    if (d.parsers.toolCall)  parserAdds.push("--tool-call-parser deepseekv4");
    if (parserAdds.length) flags = insertBeforeTail(flags, parserAdds);

    return flags;
  };

  // Renderer (same shape as §3 — multi-node prepending, env block, hints).
  const renderCommandLines = (cell, flags, sel, envValues) => {
    const modelName = resolveModelName(sel);
    const nnodes = parseNnodes(sel.nodes);
    const multinode = nnodes > 1;
    let f = [...flags];
    if (multinode && !f.some((x) => x.startsWith("--nnodes"))) {
      const at = f.findIndex((x) => x.startsWith("--model-path")) + 1;
      f.splice(at, 0,
        `--nnodes ${nnodes}`,
        `--node-rank {{NODE_RANK}}`,
        `--dist-init-addr {{NODE0_IP}}:20000`);
    }
    const flagBlock = f.map((x) => "  " + x).join(" \\\n");
    const envBlock = cell.env.length ? cell.env.join(" \\\n") + " \\\n" : "";
    let cmd = `${envBlock}sglang serve \\\n${flagBlock}`;
    if (multinode && config.multiNodeHints && config.multiNodeHints[sel.hw]) {
      const hint = config.multiNodeHints[sel.hw]
        .map((line) => (line.length ? "# " + line : "#")).join("\n");
      cmd = `${hint}\n${cmd}`;
    }
    cmd = interpolate(cmd, envValues, modelName);
    if (multinode) {
      const header =
        `# Multi-node (${nnodes} nodes). Run the same command on every node with:\n` +
        `#   <node-rank> = 0 on the head node, 1..${nnodes - 1} on the others\n` +
        `#   <node0-ip>  = IP of the head node (reachable from all others)`;
      cmd = `${header}\n${cmd}`;
    }
    return cmd;
  };

  // Line-level diff. Returns array of {line, kind: 'unchanged'|'added'|'removed'}.
  // Greedy LCS-like: walk both sides, emit `unchanged` when they agree,
  // `added` for playground-only, `removed` for base-only. This isn't optimal
  // but produces readable output when the two share most lines (the common case).
  const computeDiff = (baseStr, pgStr) => {
    const a = baseStr.split("\n");
    const b = pgStr.split("\n");
    const aSet = new Set(a);
    const bSet = new Set(b);
    let i = 0, j = 0;
    const out = [];
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j]) {
        out.push({ line: b[j], kind: "unchanged" });
        i++; j++;
      } else if (j < b.length && !aSet.has(b[j])) {
        out.push({ line: b[j], kind: "added" });
        j++;
      } else if (i < a.length && !bSet.has(a[i])) {
        out.push({ line: a[i], kind: "removed" });
        i++;
      } else if (i < a.length) {
        // Same line appears in both but at different positions — advance base
        // without emitting (the same line will be matched later as unchanged).
        i++;
      } else {
        j++;
      }
    }
    return out;
  };

  const placeholderDefaults = (schema) => {
    const out = {};
    for (const [k, v] of Object.entries(schema || {})) out[k] = v.default ?? "";
    return out;
  };

  // ==========================================================================
  // Style helper (mostly shared with §3 — adds diff-line colors)
  // ==========================================================================
  const makeStyles = (isDark) => ({
    container: { maxWidth: "900px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "8px" },
    card: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#A78BFA" : "#8B5CF6"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
    },
    cardRow: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#A78BFA" : "#8B5CF6"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      display: "flex", alignItems: "center", gap: "12px",
    },
    baseStrip: {
      padding: "8px 12px",
      borderRadius: "4px",
      background: isDark ? "#064e3b" : "#d1fae5",
      color: isDark ? "#a7f3d0" : "#065f46",
      fontSize: "12px",
      display: "flex", alignItems: "center", gap: "10px",
    },
    title: { fontSize: "13px", fontWeight: "600", color: isDark ? "#e5e7eb" : "inherit", marginBottom: "8px" },
    titleInline: { fontSize: "13px", fontWeight: "600", minWidth: "180px", flexShrink: 0, color: isDark ? "#e5e7eb" : "inherit" },
    rowFlex: { display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center", flex: 1 },
    // Stack multiple parameter sub-rows inside one card vertically.
    cardStack: { display: "flex", flexDirection: "column", gap: "6px" },
    // Each parameter sub-row: fixed-width label on the left, chips on the right.
    subRow: { display: "flex", alignItems: "center", gap: "10px" },
    subLabel: {
      fontSize: "11px",
      fontWeight: 600,
      color: isDark ? "#9ca3af" : "#6b7280",
      minWidth: "96px",
      flexShrink: 0,
      letterSpacing: "0.02em",
    },
    chipRow: { display: "flex", flexWrap: "wrap", gap: "6px", flex: 1 },
    chip: {
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px",
      cursor: "pointer",
      fontSize: "12px",
      userSelect: "none",
      background: isDark ? "#374151" : "#fff",
      color: isDark ? "#e5e7eb" : "inherit",
      minWidth: "44px",
      textAlign: "center",
    },
    chipChecked: { background: "#8B5CF6", color: "white", borderColor: "#8B5CF6" },
    chipDisabled: { cursor: "not-allowed", opacity: 0.4 },
    axisLabel: { fontSize: "11px", color: isDark ? "#9ca3af" : "#6b7280", marginRight: "6px" },
    commandWrap: {
      position: "relative",
      background: isDark ? "#111827" : "#f5f5f5",
      borderRadius: "6px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      overflow: "hidden",
    },
    commandHeader: {
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "6px 10px",
      borderBottom: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      background: isDark ? "#1f2937" : "#fafafa",
    },
    commandPre: {
      padding: "12px 16px",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontSize: "12px", lineHeight: "1.5",
      color: isDark ? "#e5e7eb" : "#374151",
      whiteSpace: "pre-wrap", overflowX: "auto", margin: 0,
    },
    diffLineUnchanged: { display: "block" },
    diffLineAdded: {
      display: "block",
      background: isDark ? "rgba(16,185,129,0.15)" : "rgba(16,185,129,0.18)",
      color: isDark ? "#a7f3d0" : "#065f46",
      borderLeft: `3px solid #10b981`,
      paddingLeft: "8px", marginLeft: "-8px",
    },
    diffLineRemoved: {
      display: "block",
      background: isDark ? "rgba(239,68,68,0.10)" : "rgba(239,68,68,0.10)",
      color: isDark ? "#fca5a5" : "#991b1b",
      textDecoration: "line-through",
      opacity: 0.7,
      borderLeft: `3px solid #ef4444`,
      paddingLeft: "8px", marginLeft: "-8px",
    },
    badge: {
      display: "inline-flex", alignItems: "center", gap: "6px",
      padding: "2px 8px", borderRadius: "10px",
      background: isDark ? "#78350f" : "#fef3c7",
      color: isDark ? "#fde68a" : "#92400e",
      fontSize: "11px", fontWeight: 600,
    },
    badgeDot: {
      width: "8px", height: "8px", borderRadius: "50%", background: "#f59e0b",
    },
    iconButton: {
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#374151",
      fontSize: "11px", fontWeight: 500, cursor: "pointer",
      display: "inline-flex", alignItems: "center", gap: "4px",
    },
    iconRow: { display: "inline-flex", gap: "6px" },
    modalBackdrop: {
      position: "fixed", inset: 0,
      background: "rgba(0,0,0,0.5)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 9999,
    },
    modalBox: {
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      borderRadius: "8px", padding: "20px",
      maxWidth: "720px", width: "92%", maxHeight: "85vh", overflowY: "auto",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
    },
    modalHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" },
    modalTitle: { fontSize: "15px", fontWeight: 600 },
    modalCloseBtn: {
      background: "transparent", border: "none", color: "inherit",
      fontSize: "20px", cursor: "pointer", padding: "0 6px", lineHeight: 1,
    },
    formField: { display: "flex", flexDirection: "column", gap: "4px", marginBottom: "10px" },
    formLabel: { fontSize: "12px", fontWeight: 500, color: isDark ? "#9ca3af" : "#4b5563" },
    formInput: {
      padding: "6px 10px", fontSize: "13px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#111827" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    },
    sectionHeading: {
      fontSize: "12px", fontWeight: 600, textTransform: "uppercase",
      letterSpacing: "0.04em",
      color: isDark ? "#9ca3af" : "#6b7280",
      margin: "12px 0 6px 0",
    },
    primaryBtn: {
      padding: "6px 14px", background: "#8B5CF6", color: "white",
      border: "none", borderRadius: "4px", cursor: "pointer",
      fontSize: "13px", fontWeight: 500,
    },
    resetBtn: {
      marginLeft: "auto",
      padding: "2px 8px",
      fontSize: "11px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "3px",
      background: "transparent",
      color: isDark ? "#9ca3af" : "#6b7280",
      cursor: "pointer",
    },
  });

  // ==========================================================================
  // State + effects
  // ==========================================================================
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const check = () => {
      const html = document.documentElement;
      setIsDark(
        html.classList.contains("dark") ||
          html.getAttribute("data-theme") === "dark" ||
          html.style.colorScheme === "dark"
      );
    };
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"],
    });
    return () => observer.disconnect();
  }, []);

  // Shared env store with §3 (same localStorage key) so HOST/PORT/etc. are
  // unified across the page.
  const STORAGE_KEY = "sglang-deploy-env";
  const [env, setEnv] = useState(() => placeholderDefaults(config.placeholders));
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        setEnv({ ...placeholderDefaults(config.placeholders), ...parsed });
      }
    } catch {}
  }, []);
  const saveEnv = (next) => {
    setEnv(next);
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch {}
  };

  // Base selection: live-link to §3. Two channels because of how browsers work:
  //   1. The `sglang-deploy-sel` custom event — §3 dispatches this every time
  //      its selection changes. This is the primary channel for in-page sync.
  //      We need it because §3 mirrors state via `history.replaceState`, which
  //      by spec does NOT fire `hashchange`.
  //   2. The standard `hashchange` event — covers the case where someone
  //      hand-edits the URL bar or follows a deep link in another tab.
  // We also snapshot from the URL on mount so deep-linking still works.
  const initialBaseFromHash = () => {
    const fallback = config.cells[0].match;
    if (typeof window === "undefined") return { ...fallback };
    const raw = window.location.hash.replace(/^#/, "");
    if (!raw) return { ...fallback };
    const params = new URLSearchParams(raw);
    const out = { ...fallback };
    params.forEach((value, key) => { if (key in out) out[key] = value; });
    return out;
  };
  const [base, setBase] = useState(() => initialBaseFromHash());
  useEffect(() => {
    const onHash = () => setBase(initialBaseFromHash());
    const onSelEvent = (e) => {
      // Trust the event payload over re-parsing the hash so we pick up changes
      // even before the browser has reflected the replaceState write.
      const fallback = config.cells[0].match;
      const incoming = (e && e.detail) || {};
      const next = { ...fallback };
      for (const k of Object.keys(next)) {
        if (incoming[k] !== undefined) next[k] = incoming[k];
      }
      setBase(next);
    };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("sglang-deploy-sel", onSelEvent);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("sglang-deploy-sel", onSelEvent);
    };
  }, []);

  // Playground deltas. `null` = inherit from base for that knob.
  const [deltas, setDeltas] = useState({
    attn: { tp: null, dp: null, cp: null, dpAttn: null },
    moe:  { backend: null, ep: null },
    spec: "current",
    parsers: { reasoning: false, toolCall: false },
  });

  const [modal, setModal] = useState(null);
  useEffect(() => {
    if (modal === null) return;
    const onKey = (e) => { if (e.key === "Escape") setModal(null); };
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [modal]);

  const [copied, setCopied] = useState(false);
  const [curlCopied, setCurlCopied] = useState(false);
  const [envDraft, setEnvDraft] = useState(env);
  useEffect(() => { if (modal === "env") setEnvDraft(env); }, [modal, env]);

  // ==========================================================================
  // Derived
  // ==========================================================================
  const s = makeStyles(isDark);
  const baseCell = findCell(config.cells, base);
  const modelName = resolveModelName(base);

  let baseCommand = "";
  let playgroundCommand = "";
  let diffLines = [];
  if (baseCell) {
    baseCommand = renderCommandLines(baseCell, baseCell.flags, base, env);
    const pgFlags = applyDeltas(baseCell.flags, deltas);
    playgroundCommand = renderCommandLines(baseCell, pgFlags, base, env);
    diffLines = computeDiff(baseCommand, playgroundCommand);
  }

  const curlText = interpolate(config.curl || "", env, modelName);

  const resetAll = () => setDeltas({
    attn: { tp: null, dp: null, cp: null, dpAttn: null },
    moe:  { backend: null, ep: null },
    spec: "current",
    parsers: { reasoning: false, toolCall: false },
  });

  const placeholderGroups = (() => {
    const out = { command: [], curl: [] };
    for (const [key, meta] of Object.entries(config.placeholders || {})) {
      (out[meta.target] || (out[meta.target] = [])).push({ key, ...meta });
    }
    return out;
  })();

  const handleCopy = () => {
    navigator.clipboard.writeText(playgroundCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  const copyCurl = () => {
    navigator.clipboard.writeText(curlText);
    setCurlCopied(true);
    setTimeout(() => setCurlCopied(false), 1200);
  };

  // ==========================================================================
  // JSX render
  // ==========================================================================
  // A row of chip selectors. `current` is the value bound to the chip group;
  // `onPick(v)` is called when the user clicks a chip. Disabled chips are
  // unclickable (used for placeholder spec algorithms).
  const renderChip = (label, current, value, onPick, opts = {}) => {
    const checked = current === value;
    const disabled = !!opts.disabled;
    return (
      <span
        key={`${label}-${value === null ? "auto" : value}`}
        style={{
          ...s.chip,
          ...(checked ? s.chipChecked : {}),
          ...(disabled ? s.chipDisabled : {}),
        }}
        title={disabled ? (opts.disabledReason || "Not available") : ""}
        onClick={() => { if (!disabled) onPick(value); }}
      >
        {label}
      </span>
    );
  };

  const setAttn = (k, v) => setDeltas((d) => ({ ...d, attn: { ...d.attn, [k]: v } }));
  const setMoe  = (k, v) => setDeltas((d) => ({ ...d, moe:  { ...d.moe,  [k]: v } }));
  const setParser = (k, v) => setDeltas((d) => ({ ...d, parsers: { ...d.parsers, [k]: v } }));

  // Format a hash-suffixed badge of the inherited base.
  const baseSummary = baseCell
    ? `${base.hw.toUpperCase()} · ${base.variant} · ${base.quant.toUpperCase()} · ${base.strategy} · ${base.nodes}`
    : "(no verified cell at current §3 selection — showing playground only)";

  return (
    <div style={s.container} className="not-prose">
      {/* Inherited base summary */}
      <div style={s.baseStrip}>
        <span style={{ fontWeight: 600 }}>Inherited base from §3:</span>
        <code style={{ fontFamily: "Menlo, monospace" }}>{baseSummary}</code>
        <button style={s.resetBtn} onClick={resetAll}>Reset all overrides</button>
      </div>

      {/* Axis 1: Attention Parallelism — one parameter per sub-row */}
      <div style={{ ...s.card, ...s.cardStack }}>
        <div style={s.title}>Attention Parallelism</div>
        <div style={s.subRow}>
          <span style={s.subLabel}>TP</span>
          <div style={s.chipRow}>
            {TP_OPTS.map((v) =>
              renderChip(v === null ? "auto" : String(v), deltas.attn.tp, v,
                (nv) => setAttn("tp", nv)))}
          </div>
        </div>
        <div style={s.subRow}>
          <span style={s.subLabel}>DP</span>
          <div style={s.chipRow}>
            {DP_OPTS.map((v) =>
              renderChip(v === null ? "auto" : String(v), deltas.attn.dp, v,
                (nv) => setAttn("dp", nv)))}
          </div>
        </div>
        <div style={s.subRow}>
          <span style={s.subLabel}>CP</span>
          <div style={s.chipRow}>
            {CP_OPTS.map((v) =>
              renderChip(v === null ? "auto" : String(v), deltas.attn.cp, v,
                (nv) => setAttn("cp", nv)))}
          </div>
        </div>
        <div style={s.subRow}>
          <span style={s.subLabel}>DP-Attention</span>
          <div style={s.chipRow}>
            {renderChip("auto", deltas.attn.dpAttn, null,  (v) => setAttn("dpAttn", v))}
            {renderChip("on",   deltas.attn.dpAttn, true,  (v) => setAttn("dpAttn", v))}
            {renderChip("off",  deltas.attn.dpAttn, false, (v) => setAttn("dpAttn", v))}
          </div>
        </div>
      </div>

      {/* Axis 2: MoE Parallelism — Backend row, EP row */}
      <div style={{ ...s.card, ...s.cardStack }}>
        <div style={s.title}>MoE Parallelism</div>
        <div style={s.subRow}>
          <span style={s.subLabel}>Backend</span>
          <div style={s.chipRow}>
            {MOE_OPTS.map((o) =>
              renderChip(o.label, deltas.moe.backend, o.id, (v) => setMoe("backend", v)))}
          </div>
        </div>
        <div style={s.subRow}>
          <span style={s.subLabel}>EP</span>
          <div style={s.chipRow}>
            {EP_OPTS.map((v) =>
              renderChip(v === null ? "auto" : String(v), deltas.moe.ep, v,
                (nv) => setMoe("ep", nv)))}
          </div>
        </div>
      </div>

      {/* Axis 3: Parsers — one toggle per parser. Clicking the chip flips its
          on/off state, no separate "off" button needed. */}
      <div style={{ ...s.card, ...s.cardStack }}>
        <div style={s.title}>Parsers</div>
        <div style={s.subRow}>
          <span style={s.subLabel}>Reasoning</span>
          <div style={s.chipRow}>
            {renderChip("deepseek-v4", deltas.parsers.reasoning, true,
              () => setParser("reasoning", !deltas.parsers.reasoning))}
          </div>
        </div>
        <div style={s.subRow}>
          <span style={s.subLabel}>Tool Call</span>
          <div style={s.chipRow}>
            {renderChip("deepseekv4", deltas.parsers.toolCall, true,
              () => setParser("toolCall", !deltas.parsers.toolCall))}
          </div>
        </div>
      </div>

      {/* Axis 4: Speculative Decoding */}
      <div style={s.card}>
        <div style={s.title}>Speculative Decoding</div>
        <div style={s.rowFlex}>
          {SPEC_PRESETS.map((p) =>
            renderChip(p.label, deltas.spec, p.id,
              (v) => setDeltas((d) => ({ ...d, spec: v })),
              { disabled: p.disabled, disabledReason: p.disabledReason }))}
        </div>
      </div>

      {/* Command box */}
      <div style={s.card}>
        <div style={s.title}>Playground Command (diff vs verified base)</div>
        <div style={s.commandWrap}>
          <div style={s.commandHeader}>
            <div style={s.badge}>
              <span style={s.badgeDot} />
              Auto-Estimated
            </div>
            <div style={s.iconRow}>
              <button style={s.iconButton} onClick={handleCopy}>
                {copied ? "✓ Copied" : "⧉ Copy"}
              </button>
              <button style={s.iconButton} onClick={() => setModal("curl")}>$ cURL</button>
              <button style={s.iconButton} onClick={() => setModal("env")}>⚙ Env</button>
            </div>
          </div>
          <pre style={s.commandPre}>
            {baseCell ? diffLines.map((d, i) => (
              <span
                key={i}
                style={
                  d.kind === "added" ? s.diffLineAdded :
                  d.kind === "removed" ? s.diffLineRemoved :
                  s.diffLineUnchanged
                }
              >
                {d.kind === "added" ? "+ " : d.kind === "removed" ? "- " : "  "}
                {d.line}{"\n"}
              </span>
            )) : "# No verified base cell at the current §3 selection.\n# Pick a supported hardware/variant in §3 to populate the playground base."}
          </pre>
        </div>
      </div>

      {/* cURL modal */}
      {modal === "curl" && (
        <div style={s.modalBackdrop} onClick={() => setModal(null)}>
          <div style={s.modalBox} onClick={(e) => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <div style={s.modalTitle}>cURL example</div>
              <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
            </div>
            <div style={s.commandWrap}>
              <div style={s.commandHeader}>
                <div style={{ fontSize: 11, opacity: 0.7 }}>
                  Model: <code>{modelName || "(unresolved)"}</code>
                </div>
                <button style={s.iconButton} onClick={copyCurl}>
                  {curlCopied ? "✓ Copied" : "⧉ Copy"}
                </button>
              </div>
              <pre style={s.commandPre}>{curlText}</pre>
            </div>
            <p style={{ fontSize: 11, opacity: 0.7, marginTop: 8 }}>
              Edit <code>CURL_HOST</code> / <code>CURL_PORT</code> in the Env panel.
            </p>
          </div>
        </div>
      )}

      {/* Env modal */}
      {modal === "env" && (
        <div style={s.modalBackdrop} onClick={() => setModal(null)}>
          <div style={s.modalBox} onClick={(e) => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <div style={s.modalTitle}>Env / placeholder values</div>
              <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
            </div>
            {placeholderGroups.curl.length > 0 && (
              <div>
                <div style={s.sectionHeading}>cURL placeholders</div>
                {placeholderGroups.curl.map(({ key, label }) => (
                  <div key={key} style={s.formField}>
                    <label style={s.formLabel}>
                      {label} <code style={{ opacity: 0.6 }}>{`{{${key}}}`}</code>
                    </label>
                    <input
                      style={s.formInput}
                      value={envDraft[key] ?? ""}
                      onChange={(e) => setEnvDraft({ ...envDraft, [key]: e.target.value })}
                    />
                  </div>
                ))}
              </div>
            )}
            {placeholderGroups.command.length > 0 && (
              <div>
                <div style={s.sectionHeading}>Command placeholders</div>
                {placeholderGroups.command.map(({ key, label }) => (
                  <div key={key} style={s.formField}>
                    <label style={s.formLabel}>
                      {label} <code style={{ opacity: 0.6 }}>{`{{${key}}}`}</code>
                    </label>
                    <input
                      style={s.formInput}
                      value={envDraft[key] ?? ""}
                      onChange={(e) => setEnvDraft({ ...envDraft, [key]: e.target.value })}
                    />
                  </div>
                ))}
              </div>
            )}
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 16 }}>
              <button style={{ ...s.iconButton, padding: "6px 14px" }} onClick={() => setModal(null)}>Cancel</button>
              <button style={s.primaryBtn} onClick={() => { saveEnv(envDraft); setModal(null); }}>Save</button>
            </div>
            <p style={{ fontSize: 11, opacity: 0.7, marginTop: 10 }}>
              Values persist in localStorage and are shared with §3.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
