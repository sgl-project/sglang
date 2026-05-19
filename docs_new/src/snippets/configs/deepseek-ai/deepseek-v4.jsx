// DeepSeek-V4 cookbook config — paired with /src/snippets/_deployment.jsx.
//
// SHAPE: must be a single literal `export const config = {...}` — no function
// calls, spreads, fragment refs, or IIFE. Mintlify re-evaluates this export at
// hydration time with module-level identifiers out of scope, so any non-literal
// value fails with `ReferenceError`.
//
// Cells are intentionally denormalized: each enumerates its full env + flags,
// so changing a common flag means sweeping all cells.
//
// `verified: true` → green "Verified" badge; absence/false → yellow "Auto-Estimated".
//
// Multi-node: do NOT put `--nnodes` / `--node-rank` / `--dist-init-addr` in
// cell.flags. The engine prepends them from `match.nodes` (`single` → 1,
// `multi-N` → N).

export const config = {
  modelName: "DeepSeek-V4",

  // `supportedHardware` is the catalog-visibility list, NOT the
  // has-runnable-cells list. Listing AMD ids here makes those buttons appear
  // in the UI; since no cell references them, the engine's grey-out logic
  // disables them automatically. Drop an id to hide it entirely.
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
  // renderer can emit `--nnodes N` automatically.
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  // HF slug: layered lookup. `${hw}|${variant}|${quant}` (override) →
  // `${variant}|${quant}` (base). Both `--model-path` (via {{MODEL_NAME}} in
  // cell.flags) and the cURL body's `"model"` field resolve from the same
  // map.
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
    // HuggingFace access token — used only by the Docker output for the
    // `--env "HF_TOKEN=..."` line; Python mode never injects it.
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",     default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",     default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  multiNodeHints: {
    h100: [
      "The following env vars may be needed depending on your cluster:",
      "  GLOO_SOCKET_IFNAME=<your-nic>",
      "  NVSHMEM_ENABLE_NIC_PE_MAPPING=1",
      "  NVSHMEM_HCA_LIST=<your-hca-list>",
    ],
    gb200: [
      "The following env vars may be needed depending on your cluster:",
      "  GLOO_SOCKET_IFNAME=<your-nic>",
      "  NVSHMEM_ENABLE_NIC_PE_MAPPING=1",
      "  NVSHMEM_HCA_LIST=<your-hca-list>",
    ],
  },

  // Per-hardware Docker image. Mirrors the "Docker Images by Hardware
  // Platform" table in §2 of DeepSeek-V4.mdx; if you change one, change both.
  dockerImages: {
    h100:  "lmsysorg/sglang:latest",
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    gb200: "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
  },

  // -----------------------------------------------------------------------
  // Playground feature axes (drives §3.3 Playground UI).
  //
  // OPT-IN MODEL: each axis is rendered ONLY if its key is present here.
  // Omit any axis (or comment its block) to hide it for this model. No
  // explicit `enabled` field — the presence of the key is the enable signal.
  //
  // 6 axes correspond 1:1 to the 6 cards in the old hand-written
  // deepseek-v4-playground.jsx. The engine (_playground.jsx) owns the widget
  // types and the strip/insert behaviour; this section just supplies the
  // labels, value ranges, and the actual SGLang flag strings to emit per
  // option (so model-specific things like the parser slug or the MTP_314
  // numbers live here, not in the engine).
  // -----------------------------------------------------------------------
  playgroundFeatures: {

    // ----- Card 1: "Attention Parallelism" — 4 knobs (TP/DP/CP/DP-Attention) -----
    //
    // Per-chip constraints (engine schema): an entry can be either a bare
    // value (`null`, `1`, ...) OR an object `{value, hide?, disable?,
    // disableReason?, label?}`. `hide` / `disable` are constraint objects —
    // keys are base-cell fields, values are arrays of allowed matches; the
    // constraint fires only when EVERY key in it matches the current base
    // (AND across keys).
    //
    //   hide    — chip omitted entirely (used for hard impossibilities like
    //             Pro variant on TP=1: 1.4T-param Pro can't fit on 1 GPU).
    //   disable — chip greyed out + tooltip (soft warning, user can still
    //             see the option exists but is unsupported here).
    //
    // DeepSeek-V4 specifics:
    //   - Pro variant won't fit on TP=1/2 regardless of hardware — hide.
    //   - TP=16 means 16 ranks → single-node has at most 8 GPUs, so
    //     TP=16 on single-node is impossible — disable (with a hint that
    //     they should switch §3.1's "Nodes" to multi-2 first).
    //   - Same constraints mirror onto DP.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null,
          { value: 1, hide: { variant: ["pro"] } },
          { value: 2, hide: { variant: ["pro"] } },
          4,
          8,
          { value: 16, disable: { nodes: ["single"] },
            disableReason: "TP=16 requires 16 ranks — switch §3.1's Nodes to multi-2 first." },
        ]},
        { id: "dp", label: "DP", values: [
          null,
          { value: 1, hide: { variant: ["pro"] } },
          { value: 2, hide: { variant: ["pro"] } },
          4,
          8,
          { value: 16, disable: { nodes: ["single"] },
            disableReason: "DP=16 requires 16 ranks — switch §3.1's Nodes to multi-2 first." },
        ]},
        { id: "cp",     label: "CP", values: [null, 1, 2, 4] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, true, false],
          labels: { "auto": "auto", "true": "on", "false": "off" } },
      ],
    },

    // ----- Card 2: "MoE Parallelism" — Backend select + EP knob -----
    // Each backend option's `flags` array is what the engine splices in
    // when the user picks it (after stripping the base's --moe-a2a-backend
    // / --moe-runner-backend flags).
    moe: {
      backend: {
        options: [
          { id: null,                label: "Inherited" },
          { id: "deepep",            label: "DeepEP",
            flags: ["--moe-a2a-backend deepep"] },
          { id: "megamoe",           label: "MegaMoE",
            flags: ["--moe-a2a-backend megamoe"] },
          { id: "flashinfer_mxfp4",  label: "FlashInfer (MXFP4)",
            flags: ["--moe-runner-backend flashinfer_mxfp4"] },
          { id: "marlin",            label: "Marlin (W4A16)",
            flags: ["--moe-runner-backend marlin"] },
        ],
      },
      // EP=16 is similarly bounded by total ranks — disable on single-node.
      ep: { label: "EP", values: [
        null,
        { value: 1, hide: { variant: ["pro"] } },
        { value: 2, hide: { variant: ["pro"] } },
        4,
        8,
        { value: 16, disable: { nodes: ["single"] },
          disableReason: "EP=16 requires 16 ranks — switch §3.1's Nodes to multi-2 first." },
      ]},
    },

    // ----- Card 3: "Parsers" — multi-toggle, one chip per item -----
    // Per-model `flag` because parser slugs differ (deepseek-v4 / qwen3 / ...).
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser deepseek-v4" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser deepseekv4" },
      ],
    },

    // ----- Card 4: "Speculative Decoding" — single-select chip group -----
    // `current` keeps the base cell's spec flags as-is; `off` strips them.
    // Other options have their own `flags` array that the engine splices in
    // after stripping the base's spec flags.
    speculative: {
      options: [
        { id: "current",    label: "Inherited from base" },
        { id: "off",        label: "Off (greedy)" },
        { id: "mtp-314",    label: "EAGLE / MTP 3-1-4",
          flags: ["--speculative-algo EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
        { id: "mtp-112",    label: "EAGLE / MTP 1-1-2",
          flags: ["--speculative-algo EAGLE", "--speculative-num-steps 1",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 2"] },
        { id: "draftflash", label: "DraftFlash", disabled: true,
          disabledReason: "Coming soon — pending DraftFlash kernel integration." },
        { id: "nextn",      label: "NextN",      disabled: true,
          disabledReason: "Coming soon — pending NextN algorithm support." },
      ],
    },

    // ----- Card 5: "PD Disaggregation" — Mode select + IB Device select -----
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      // IB device defaults differ per Blackwell variant: B200 typically uses
      // mlx5_7, H200 uses mlx5_0, GB300 uses NVLink (no IB). `auto` (default)
      // emits no --disaggregation-ib-device flag.
      ibDevices: ["auto", "mlx5_0", "mlx5_7"],
    },

    // ----- Card 6: "Hierarchical KV Cache" — Enable + Storage + Write Policy -----
    // `auto` (default) for backend emits no --hicache-storage-backend flag —
    // host RAM only. `auto` for writePolicy resolves to "write_through" in
    // the engine.
    hicache: {
      backends: [
        { id: null,        label: "auto" },
        { id: "file",      label: "file" },
        { id: "mooncake",  label: "mooncake" },
        { id: "hf3fs",     label: "hf3fs" },
        { id: "nixl",      label: "nixl" },
      ],
      writePolicies: ["auto", "write_through", "write_back", "write_through_selective"],
    },

    // ----- Card 7: "MegaMoE" — single-select chip group, Blackwell-only -----
    // MegaMoE fuses MoE dispatch + GEMM into a single kernel for higher MoE
    // throughput. Only runnable on Blackwell GPUs — `requiresHw` tells the
    // engine to hide this card when the base cell's hw is outside the list.
    //
    // Each option carries:
    //   flags        : spliced into the command (swaps the base's
    //                  --moe-a2a-backend if present, otherwise appends)
    //   env          : appended to the cell's env block
    //   stripEnv     : env prefixes to drop from the base cell's env before
    //                  applying this option (MegaMoE doesn't use the DeepEP
    //                  dispatch buffer, so the SGLANG_DEEPEP_NUM_MAX_*
    //                  budget shouldn't be set alongside it).
    //
    // Visibility gates (both checked at render time):
    //   requiresHw       : base cell's hw MUST be in this list
    //   excludesStrategy : hide when base cell's strategy is in this list
    //                      (MegaMoE's throughput optimization is incompatible
    //                      with low-latency mode)
    //
    // `disabled` is the no-op identity option; `w4a8` is the standard MegaMoE
    // kernel; `w4a4` additionally enables the FP4-acts variant.
    megamoe: {
      requiresHw: ["b200", "b300", "gb200", "gb300"],
      excludesStrategy: ["low-latency"],
      stripEnv: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
      options: [
        { id: "disabled", label: "Disabled" },
        { id: "w4a8",     label: "W4A8",
          flags: ["--moe-a2a-backend megamoe"],
          env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
        { id: "w4a4",     label: "W4A4",
          flags: ["--moe-a2a-backend megamoe"],
          env: [
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1",
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1",
          ] },
      ],
    },
  },

  // -----------------------------------------------------------------------
  // Cell catalog — one entry per supported
  // (hw × variant × quant × strategy × nodes) combination. Anything not
  // listed here is auto-greyed-out by the engine's `isOptionAvailable`.
  // -----------------------------------------------------------------------
  cells: [
    // ====================================================================
    // B200 + FP4
    // ====================================================================
    {
      match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 4096",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-runner-backend flashinfer_mxfp4",
        "--disable-flashinfer-autotune",
        "--chunked-prefill-size 32768",
        "--swa-full-tokens-ratio 0.1",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.92",
        "--cuda-graph-max-bs 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Note: this cell ships with `--moe-a2a-backend deepep` as the verified
      // default. To run B200/B300 Pro max-throughput with MegaMoE, flip the
      // MegaMoE chip in the §3.3 Playground — it strips the
      // SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK env and swaps the moe
      // backend automatically.
      match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.835",
        "--cuda-graph-max-bs 544",
        "--swa-full-tokens-ratio 0.075",
        "--chunked-prefill-size 65536",
        "--tokenizer-worker-num 8",
        "--enable-prefill-delayer",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 + FP4 — same launch recipes as B200, different docker image SKU.
    // The cells below are line-for-line copies of the B200 cells above with
    // only `match.hw` changed; keep them in sync when editing B200.
    // ====================================================================
    {
      match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 4096",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-runner-backend flashinfer_mxfp4",
        "--disable-flashinfer-autotune",
        "--chunked-prefill-size 32768",
        "--swa-full-tokens-ratio 0.1",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.92",
        "--cuda-graph-max-bs 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Mirrors B200 Pro max-throughput; default is deepep (MegaMoE off).
      // Use the §3.3 Playground MegaMoE chip to switch to megamoe.
      match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.835",
        "--cuda-graph-max-bs 544",
        "--swa-full-tokens-ratio 0.075",
        "--chunked-prefill-size 65536",
        "--tokenizer-worker-num 8",
        "--enable-prefill-delayer",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB200 + FP4
    // ====================================================================
    {
      match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 4096",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // GB200 Pro requires 2 nodes; multi-node wiring added by the renderer.
    {
      match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 4096",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs 64",
        "--max-running-requests 128",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs 64",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 + FP4
    // ====================================================================
    {
      match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 4096",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 4096",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H200 + FP8 (deepep, no Marlin)
    // ====================================================================
    {
      match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_DSV4_FP4_EXPERTS=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 128",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // H200 Pro FP8: low-latency exposes BOTH single-node (TP=8 Marlin) and
    // multi-2 (TP=16 DP-attn + DeepEP) — the old combined block, split.
    {
      match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_DSV4_FP4_EXPERTS=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 4096",
        "--disable-flashinfer-autotune",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 16",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--cuda-graph-max-bs 8",
        "--max-running-requests 32",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 16",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.88",
        "--cuda-graph-max-bs 8",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "max-throughput", nodes: "multi-2" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 16",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.88",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H200 + FP4 (Marlin runner, single-node only)
    // ====================================================================
    {
      match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend marlin",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H100 + FP4 (Marlin runner)
    // ====================================================================
    {
      match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs 8",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--speculative-algo EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs 8",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "multi-2" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
