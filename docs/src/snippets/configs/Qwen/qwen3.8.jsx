// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Qwen3.8-2.4T-A95B: 92 layers as 23 repeats of (3 x Gated DeltaNet -> MoE, then
// 1 x Gated Attention -> MoE), so 69 linear-attention layers to 23 full-attention
// ones; MoE with 512 experts, 10 routed + 1 shared active; 2.4T total / 95B
// active params. Text-only, and reasoning cannot be disabled.
//
// A hardware x quantization x strategy combination with no launch recipe has no
// cell, and the engine greys it out.

export const config = {
  modelName: "Qwen3.8",

  supportedHardware: ["h200", "b200", "b300", "gb300", "mi300x", "mi350x", "mi355x"],

  variants: [
    { id: "default", label: "Default" },
  ],
  // Checkpoint precisions. NVFP4 is NVIDIA's FP4 format (Blackwell only); MXFP4
  // is the OCP format AMD CDNA4 supports natively (mi350x/mi355x only). Not to be
  // confused with the Playground's "FlashInfer (MXFP4)" MoE runner chip, an
  // unrelated NVIDIA kernel that shares the name.
  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
    { id: "mxfp4", label: "MXFP4" },
  ],
  // Three operating points on the throughput/latency curve, plus `dspark`, which
  // swaps NEXTN for the trained DSpark draft model. Only GB300/FP8 carries the
  // full ladder; single-recipe hardware parks under `balanced`.
  strategies: [
    { id: "low-latency",     label: "Low Latency"     },
    { id: "balanced",        label: "Balanced"        },
    { id: "high-throughput", label: "High Throughput" },
    { id: "dspark",          label: "DSpark"          },
  ],
  // Node counts a recipe spans. GB300 hosts are 4 GPUs, so its TP16 shapes take
  // 4 nodes; everything else is an 8-GPU host.
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "2 Nodes"     },
    { id: "multi-4", label: "4 Nodes"     },
    { id: "multi-8", label: "8 Nodes"     },
  ],

  modelNames: {
    "default|bf16":  "Qwen/Qwen3.8-2.4T-A95B",
    // Separate repo, not a revision of the BF16 one.
    "default|fp8":   "Qwen/Qwen3.8-2.4T-A95B-FP8",
    "default|nvfp4": "RadixArk/Qwen3.8-2.4T-A95B-NVFP4",
    "default|mxfp4": "Qwen/Qwen3.8-2.4T-A95B-FP8-MXFP4",
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

  latencyPercentile: "Mean",

  // The "⚡ Reproduce" modal's benchmark command. --random-range-ratio 1 pins ISL
  // exactly rather than drawing a range, so runs stay comparable.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang-oai \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --request-rate inf \\
  --flush-cache`,
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // Per-hardware image for Docker mode. The two ROCm images are not
  // interchangeable: MI300X (gfx942) takes the mi30x build on ROCm 7.00,
  // MI350X/MI355X (gfx950) the mi35x build on ROCm 7.20.
  dockerImages: {
    h200:   "lmsysorg/sglang:qwen38",
    b200:   "lmsysorg/sglang:qwen38",
    b300:   "lmsysorg/sglang:qwen38",
    gb300:  "lmsysorg/sglang:qwen38",
    mi300x: "lmsysorg/sglang-rocm:v0.5.17-rocm700-mi30x-20260813",
    mi350x: "lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260812",
    mi355x: "lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260812",
  },

  // Per-hardware notes shown above a multi-node command. Every multi-node recipe
  // here crosses IB (only GB300 has rack-scale NVLink and needs no manual NIC
  // config); without a pinned socket interface and HCA list, RCCL/GLOO can pick a
  // non-routable NIC on a multi-homed host and the rendezvous stalls. The device
  // names below are examples — substitute your own (`ip -br addr`,
  // `ibv_devinfo`).
  multiNodeHints: {
    h200: [
      "TP8 x PP4 over IB NDR 400. Pin the rendezvous NIC and list your HCAs on every node:",
      "  export GLOO_SOCKET_IFNAME=bond0",
      "  export NCCL_SOCKET_IFNAME=bond0",
      "  export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
      "  export NCCL_IB_DISABLE=0",
      "  export SGLANG_HOST_IP=<this_node_ip>",
    ],
    b200: [
      "TP8 x PP2 over IB. Pin the rendezvous NIC and list your HCAs on BOTH nodes:",
      "  export GLOO_SOCKET_IFNAME=<iface>",
      "  export NCCL_SOCKET_IFNAME=<iface>",
      "  export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
      "  export NCCL_IB_DISABLE=0",
      "  export SGLANG_HOST_IP=<this_node_ip>",
    ],
    b300: [
      "TP8 x PP2 over IB. Pin the rendezvous NIC and list your HCAs on BOTH nodes:",
      "  export GLOO_SOCKET_IFNAME=<iface>",
      "  export NCCL_SOCKET_IFNAME=<iface>",
      "  export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
      "  export NCCL_IB_DISABLE=0",
      "  export SGLANG_HOST_IP=<this_node_ip>",
    ],
    mi300x: [
      "Pin the rendezvous NIC on BOTH nodes (replace eno8303 with your interface):",
      "  export GLOO_SOCKET_IFNAME=eno8303",
      "  export NCCL_SOCKET_IFNAME=eno8303",
      "  export RCCL_SOCKET_IFNAME=eno8303",
    ],
  },

  github: {
    cookbookModel: "Qwen/Qwen3.8-2.4T-A95B",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // 23 GQA full-attention layers expose the usual TP/DP-Attention knobs; the
    // range is widened past the template default given the model's scale.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8, 16, 32, 64] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 1, 2, 4, 8, 16, 32, 64],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card: "MoE Parallelism" -----
    // 512 routed experts + 1 shared, top-10 routing.
    moe: {
      backend: {
        options: [
          { id: null,               label: "Inherited" },
          { id: "deepep",           label: "DeepEP",            flags: ["--moe-a2a-backend deepep"] },
          { id: "megamoe",          label: "MegaMoE",           flags: ["--moe-a2a-backend megamoe"],
            requiresHw: ["b200", "b300", "gb300"] },
          // FlashInfer is CUDA-only — this is the NVIDIA MoE runner kernel, NOT
          // the AMD "mxfp4" checkpoint quantization above (unrelated despite
          // the shared name). Gate it off AMD so it can't be picked there.
          { id: "flashinfer_mxfp4", label: "FlashInfer (MXFP4)", flags: ["--moe-runner-backend flashinfer_mxfp4"],
            requiresHw: ["h200", "b200", "b300", "gb300"] },
          { id: "marlin",           label: "Marlin (W4A16)",    flags: ["--moe-runner-backend marlin"] },
        ],
      },
      megamoeQuant: {
        stripEnv: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
        options: [
          { id: "w4a8", label: "W4A8",
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
          { id: "w4a4", label: "W4A4",
            flags: ["--enable-w4a4-mxfp4-megamoe"],
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
        ],
      },
      // WideEP — the launch post's large-scale-EP claim — so the range goes past
      // the template default's 16.
      ep: { label: "EP", values: [null, 1, 2, 4, 8, 16, 32, 64] },
    },

    // ----- Card: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----
    // DSpark is the trained draft model from the launch post. ReplaySSM is a
    // separate opt-in (see the flagSelects row below); nothing turns it on
    // implicitly, and --enable-linear-replayssm-spec defaults off.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
        // _handle_dspark (arg_groups/speculative_hook.py) raises rather than
        // degrading: pp_size must be 1, and under DP-Attention it also needs
        // --enable-dp-lm-head, moe_a2a_backend none, and no context parallel.
        // Gate the chip on both so the panel can't emit a command that aborts.
        { id: "dspark",  label: "DSpark",
          flags: ["--speculative-algorithm DSPARK",
                  "--speculative-draft-model-path RadixArk/Qwen3.8-2.4T-A95B-DSpark"],
          disable: [
            { when: { dpAttnOn: [true] },
              reason: "DSpark with DP-Attention additionally requires --enable-dp-lm-head, the built-in TP MoE (--moe-a2a-backend none) and no context parallel. Turn DP-Attention off in the Attention card above, or pick a cell that doesn't use it." },
            { when: { hw: ["h200", "mi300x"] },
              reason: "DSpark requires pp_size == 1 and this recipe is pipelined (TP x PP across nodes)." },
            { when: { hw: ["b200", "b300"], quant: ["fp8"] },
              reason: "DSpark requires pp_size == 1 and the B200/B300 FP8 recipes are TP8 x PP2." },
          ] },
        { id: "ngram",   label: "NGRAM",
          flags: ["--speculative-algorithm NGRAM",
                  "--speculative-num-draft-tokens 16",
                  "--speculative-ngram-max-bfs-breadth 10"],
          disable: { dpAttnOn: [true] },
          disableReason: "NGRAM is incompatible with DP-Attention. Turn DP-Attention off in the Attention card above to use NGRAM." },
      ],
    },

    // ----- Card: "PD Disaggregation" -----
    // Role flags follow the P/D bundle's own prefill and decode workers. Two
    // flags those recipes carry are deliberately not emitted:
    // --prefill-round-robin-balance is a DeprecatedAction on current SGLang and
    // does nothing, and --mamba-track-interval is context-dependent (the source
    // recipes set it equal to their context cap) so a fixed value here would be
    // wrong for cells serving the native window.
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role",
          flags: ["--load-balance-method round_robin",
                  "--enable-symm-mem",
                  "--scheduler-recv-interval 1"] },
        { id: "decode",  label: "Decode role",
          flags: ["--enable-symm-mem",
                  "--scheduler-recv-interval 1",
                  "--disaggregation-decode-polling-interval 1",
                  "--skip-server-warmup"],
          env: ["SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000",
                "SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096"] },
      ],
      transferBackends: [
        // Shared by the P/D bundle's prefill and decode workers. MC_FORCE_MNNVL=1
        // is in those recipes too but is MNNVL-fabric only, so it is left to the
        // operator rather than emitted on hardware that has no such fabric.
        { id: "mooncake", label: "Mooncake",
          env: ["SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000",
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000",
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000",
                "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0",
                "SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=3600",
                "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
                "SGLANG_DISAGG_STAGING_BUFFER=1"] },
        { id: "nixl", label: "NiXL" },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      // Ports come from the engine's PD_PORTS, not literals — the decode role
      // serves on 30100, so a hardcoded target would not reach it.
      // In PD mode, --policy is the prefill fallback; keep decode explicit.
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --policy round_robin \\
  --decode-policy round_robin \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --worker-startup-timeout-secs 7200 \\
  --request-timeout-secs 6900 \\
  --pool-idle-timeout-secs 4 \\
  --disable-circuit-breaker`,
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

    // ----- Axis: Flag Selects (GDN state knobs) -----
    flagSelects: [
      {
        // Opt-in row for ReplaySSM on a speculative run. Bit-identical to the
        // recurrent baseline per the launch post, so there's no accuracy
        // tradeoff — only a memory one.
        id: "replaySsm", title: "ReplaySSM (spec)",
        showWhen: (b) => b.spec === "dspark",
        stripPrefixes: ["--enable-linear-replayssm-spec"],
        options: [
          { id: "off", label: "Off" },
          {
            id: "on", label: "On",
            disable: { pdMode: ["prefill"] },
            disableReason: "A PD prefill server never runs speculative verify, so --enable-linear-replayssm-spec is rejected at startup.",
            flags: ["--enable-linear-replayssm-spec"],
          },
        ],
      },
      {
        // Radix prefix caching over the GDN state — see "ReplaySSM and Overlap
        // for the GDN State" above for what extra_buffer buys.
        id: "mambaRadix", title: "GDN Radix Cache Strategy",
        stripPrefixes: ["--mamba-radix-cache-strategy"],
        options: [
          { id: "auto",  label: "Auto (extra_buffer)" },
          { id: "lazy",  label: "extra_buffer_lazy", flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
          { id: "nobuf", label: "no_buffer",         flags: ["--mamba-radix-cache-strategy no_buffer"] },
        ],
      },
      {
        id: "kvCacheDtype", title: "KV Cache Precision",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "auto", label: "Auto (BF16)" },
          { id: "fp8",  label: "FP8 (E4M3) — halves KV memory", flags: ["--kv-cache-dtype fp8_e4m3"] },
        ],
      },
    ],
  },

  // Ordering: the first cell seeds the Deploy panel's default selection.
  cells: [
    {
      // GB300 / FP8, balanced — DP4 attention with per-DP TP4, MoE EP16 over
      // DeepEP v2 hybrid, NEXTN 3+1 with ReplaySSM. The capacity set
      // (max-total-tokens / max-running-requests / mamba pool / decode graph
      // ladder) is tuned as a unit; retune the values together.
      //
      // The SGLANG_DEEPEP_V2_*_PER_RANK envs size the a2a buffers against
      // --chunked-prefill-size 32768 — the default 128 cap refuses to start.
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-4" },
      verified: true,
      env: [
        "SGLANG_DEEPEP_V2_EXPAND_PREFILL=1",
        "SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048",
        "SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK=384",
        "EP_DISABLE_GIN=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dp-size 4",
        "--ep-size 16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--enable-dp-attention-local-control-broadcast",
        "--moe-dense-tp-size 1",
        "--kv-cache-dtype fp8_e4m3",
        "--linear-attn-prefill-backend flashinfer",
        "--moe-a2a-backend deepep_v2",
        "--deepep-v2-mode hybrid",
        "--enable-eplb",
        "--mamba-ssm-dtype bfloat16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--enable-linear-replayssm-spec",
        "--mem-fraction-static 0.93",
        "--max-total-tokens 360448",
        "--max-running-requests 128",
        "--max-mamba-cache-size 132",
        "--cuda-graph-max-bs-decode 32",
        "--cuda-graph-bs-decode 1 2 4 8 16 32",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 / FP8, low-latency — TP16 across 4 nodes, narrow EP, NEXTN with
      // ReplaySSM, CuteDSL AllReduce fusion. --max-mamba-cache-size 80 is
      // 16 concurrent requests x the 5 GDN state slots extra_buffer budgets each;
      // retune the two together.
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "multi-4" },
      verified: true,
      env: [
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1",
        "NCCL_NVLS_ENABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend trtllm_mha",
        "--moe-runner-backend flashinfer_trtllm",
        "--mamba-ssm-dtype bfloat16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--speculative-algorithm NEXTN",
        "--enable-linear-replayssm-spec",
        "--mem-fraction-static 0.95",
        "--max-running-requests 16",
        "--max-mamba-cache-size 80",
        "--max-total-tokens 262144",
        "--cuda-graph-max-bs-decode 16",
        "--cuda-graph-bs-decode 1 2 4 8 16",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 / FP8, high-throughput — same wide-EP shape as balanced, MTP off.
      // MASKED=384 caps the decode slab, which would otherwise default to the
      // full 2048 and cost GiBs at graph capture. --max-total-tokens 2800000 is
      // the accuracy-oriented value; throughput runs use 2000000.
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "multi-4" },
      verified: true,
      env: [
        "SGLANG_DEEPEP_V2_EXPAND_PREFILL=1",
        "SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048",
        "SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK=384",
        "EP_DISABLE_GIN=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dp-size 4",
        "--ep-size 16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--enable-dp-attention-local-control-broadcast",
        "--moe-dense-tp-size 1",
        "--kv-cache-dtype fp8_e4m3",
        "--linear-attn-prefill-backend flashinfer",
        "--moe-a2a-backend deepep_v2",
        "--deepep-v2-mode hybrid",
        "--enable-eplb",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-ssm-dtype bfloat16",
        "--mem-fraction-static 0.93",
        "--max-total-tokens 2800000",
        "--cuda-graph-max-bs-decode 128",
        "--cuda-graph-bs-decode 1 2 4 8 16 32 64 96 128",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 / NVFP4, low-latency — TP8 narrow EP across 2 nodes, NEXTN 3+1 with
      // ReplaySSM, CuteDSL AllReduce fusion. --fp4-gemm-backend and
      // --moe-runner-backend are real overrides: auto picks the CuTe DSL FP4
      // kernels on SM100 and never enables the TRT-LLM fused NVFP4 MoE path.
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      env: [
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1",
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NVLS_ENABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend trtllm_mha",
        "--linear-attn-prefill-backend flashinfer",
        "--moe-runner-backend flashinfer_trtllm",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--enable-linear-replayssm-spec",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-backend-decode full",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 / NVFP4, high-throughput — full DP16 attention with EP16 MoE over
      // FlashInfer one-sided A2A, MTP off. flashinfer_trtllm_routed must stay:
      // with a2a=flashinfer the auto runner resolution aborts at startup.
      // SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK is load-bearing —
      // unset it falls back to 1024 and startup raises once 1024 x ep_size no
      // longer covers the largest CuteDSL MoE forward.
      // --disable-prefill-cuda-graph is required at chunk 131072, where graph
      // capture would OOM; --skip-server-warmup goes with it.
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "multi-4" },
      verified: true,
      env: [
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192",
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NET_GDR_C2C=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dp-size 16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--enable-dp-attention-local-control-broadcast",
        "--moe-dense-tp-size 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--kv-cache-dtype fp8_e4m3",
        "--linear-attn-prefill-backend flashinfer",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--moe-a2a-backend flashinfer",
        "--ep-dispatch-algorithm static",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-ssm-dtype bfloat16",
        "--mem-fraction-static 0.95",
        "--chunked-prefill-size 131072",
        "--max-prefill-tokens 8192",
        "--weight-loader-drop-cache-after-load",
        "--model-loader-extra-config '{\"enable_multithread_load\": false}'",
        "--disable-prefill-cuda-graph",
        "--cuda-graph-backend-decode full",
        "--skip-server-warmup",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 / BF16 — full-precision weights, TP32 across 8 nodes x 4 GPUs.
      // 4.8TB does not fit 16 GPUs, and GB300 is the one platform where a flat
      // TP32 stays on rack-scale NVLink instead of crossing IB. KV stays at model
      // precision (no --kv-cache-dtype), the highest-fidelity configuration here.
      // The sizing is derived, not measured.
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "multi-8" },
      env: [
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1",
        "NCCL_NVLS_ENABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-eagle-topk 1",
        "--enable-linear-replayssm-spec",
        "--mem-fraction-static 0.95",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // H200 / FP8 — 4 nodes x 8 GPUs, TP8 x PP4 over IB. The one Hopper recipe,
      // and the only cell where the flashinfer linear-attention backends are real
      // overrides: SM90 defaults to triton for both GDN halves. --page-size 64 is
      // likewise non-default on Hopper. mem-fraction is left to the auto
      // heuristic, which prices the graph set into its reserve.
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-4" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 4",
        "--dist-timeout 1800",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-full-memory-ratio 0.95",
        "--mamba-ssm-dtype bfloat16",
        "--max-prefill-tokens 8192",
        "--page-size 64",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // B300 / FP8 — 2 nodes x 8 GPUs, TP8 x PP2 over IB (~2.4TB of weights do not
      // fit one 2.30TB node). pp > 1 forbids speculative decoding and
      // auto-disables the overlap scheduler, so extra_buffer budgets 4 GDN slots
      // per request rather than 5. --context-length 262144 is the model's native
      // window written out, not a cap.
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--dist-timeout 1800",
        "--context-length 262144",
        "--attention-backend trtllm_mha",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--moe-runner-backend flashinfer_trtllm",
        "--moe-a2a-backend none",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-full-memory-ratio 0.95",
        "--mamba-ssm-dtype bfloat16",
        "--mem-fraction-static 0.95",
        "--max-running-requests 512",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--page-size 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 128",
        "--cuda-graph-bs-decode 1 2 4 8 16 32 64 128",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // B300 / NVFP4 — single node, TP8, NEXTN with ReplaySSM and the CuteDSL
      // AllReduce fusion. BF16 KV: the only NVFP4 cell serving KV at model
      // precision. --mamba-ssm-dtype bfloat16 is load-bearing — the SM100
      // flashinfer GDN decode default is gated on it, and without it decode falls
      // back to Triton.
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--moe-runner-backend flashinfer_trtllm",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--enable-linear-replayssm-spec",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // B200 / FP8 — 2 nodes x 8 GPUs, TP8 x PP2, same shape as the B300 FP8 cell
      // but leaner: ~0.6TB free after weights instead of ~2.1TB, so the
      // concurrency ceiling and mem-fraction are both left derived.
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--dist-timeout 1800",
        "--mamba-full-memory-ratio 0.95",
        "--mamba-ssm-dtype bfloat16",
        "--chunked-prefill-size 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // B200 / NVFP4 — 2 nodes x 8 GPUs, TP8 x PP2. Not the single-node shape the
      // weights would allow: at TP8 on one node ~153.6GB of weights leaves only
      // ~25GB per GPU for the pools, so this recipe pipelines two nodes and cuts
      // the per-GPU weight share to ~77GB. pp > 1 forbids speculative decoding.
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--dist-timeout 1800",
        "--attention-backend trtllm_mha",
        "--mamba-ssm-dtype bfloat16",
        "--mamba-radix-cache-strategy extra_buffer",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.88",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI355X / MXFP4 — single node, TP8. SGLANG_USE_AITER gates the AITER
      // MXFP4-MoE / GEMM / norm / rope kernels; the ROCm image sets it, a
      // bare-pip host does not. mem-fraction 0.9 is pre-scaling — aiter
      // multiplies it by 0.85 above 8K context, so ~0.765 effective.
      match: { hw: "mi355x", variant: "default", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_AITER=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.9",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI350X — identical command to MI355X (same CDNA4 gfx950, same 288GB, same
      // mi35x ROCm image).
      match: { hw: "mi350x", variant: "default", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_AITER=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.9",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI300X / FP8 — 2 nodes x 8 GPUs, TP8 x PP2 (gfx942 has no MXFP4 hardware,
      // and ~2.4TB of FP8 does not fit 1.5TB per node). --disable-custom-all-reduce
      // puts the intra-node all-reduce on RCCL and must be on every rank;
      // mem-fraction 1.0 is ~0.85 effective after the aiter scale.
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [
        "SGLANG_USE_AITER=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--disable-custom-all-reduce",
        "--dist-timeout 3600",
        "--page-size 16",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 1.0",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 / FP8, DSpark — the low-latency TP16 narrow-EP shape with the DSpark
      // draft model in place of NEXTN. The balanced tier is not a candidate: its
      // DeepEP v2 a2a rules DSpark out under DP-attention.
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "dspark", nodes: "multi-4" },
      env: [
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1",
        "NCCL_NVLS_ENABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--kv-cache-dtype fp8_e4m3",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path RadixArk/Qwen3.8-2.4T-A95B-DSpark",
        "--mem-fraction-static 0.95",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 / NVFP4, DSpark — the low-latency TP8 shape with the DSpark draft
      // model in place of NEXTN.
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "multi-2" },
      env: [
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1",
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NVLS_ENABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend trtllm_mha",
        "--linear-attn-prefill-backend flashinfer",
        "--moe-runner-backend flashinfer_trtllm",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path RadixArk/Qwen3.8-2.4T-A95B-DSpark",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-backend-decode full",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 / BF16, DSpark — the TP32 shape with the DSpark draft model in place
      // of NEXTN.
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "dspark", nodes: "multi-8" },
      env: [
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1",
        "NCCL_NVLS_ENABLE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path RadixArk/Qwen3.8-2.4T-A95B-DSpark",
        "--mem-fraction-static 0.95",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // B300 / NVFP4, DSpark — single-node TP8 with the DSpark draft model. The
      // draft needs its own weights and KV, so mem-fraction drops to 0.80 and
      // --context-length trims the native window to buy that room back.
      // SGLANG_ENABLE_MOE_DEFERRED_FINALIZE defers the MoE finalize so it fuses
      // into the CuteDSL AllReduce workspace.
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1",
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--context-length 200000",
        "--preferred-sampling-params '{\"top_k\": 20}'",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--moe-runner-backend flashinfer_trtllm",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm DSPARK",
        "--speculative-draft-model-path RadixArk/Qwen3.8-2.4T-A95B-DSpark",
        "--mem-fraction-static 0.80",
        "--max-running-requests 128",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--cuda-graph-backend-prefill breakable",
        "--cuda-graph-max-bs-prefill 8192",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 128",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
