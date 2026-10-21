// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.

export const config = {
  modelName: "GLM-5.3",

  supportedHardware: [
    "h200", "b200", "gb300", "b300",
    "mi355x", "mi325x", "mi300x", "a3", "a5"
  ],

  hardware: [
    { id: "a3", label: "A3 Series", vram: "64GB/die", vendor: "npu", npuDevices: 16 },
    { id: "a5", label: "950PR/DT series", vram: "128GB", vendor: "npu", npuDevices: 8 },
  ],

  // Single released checkpoint — no size/mode split.
  variants: [
    { id: "default", label: "GLM-5.3", subtitle: "MoE · DSA" },
  ],
  quantizations: [
    { showWhen: (s) => s.hw === "a5", id: "w8a8c8-mxfp8", label: "W8A8C8_MXFP8 (ModelSlim)", disabled: (s) => s.hw !== "a5", disableReason: "The 950PR/DT series unified and PD recipes use the W8A8C8_MXFP8 checkpoint." },
    { showWhen: (s) => s.hw === "a3", id: "w8a8", label: "W8A8 (ModelSlim)" },
    { showWhen: (s) => !["a3", "a5"].includes(s.hw), id: "fp8", label: "FP8" },
    { showWhen: (s) => !["a3", "a5"].includes(s.hw), id: "bf16", label: "BF16" },
    { showWhen: (s) => !["a3", "a5"].includes(s.hw), id: "nvfp4", label: "NVFP4 (Experimental)" },
  ],
  strategies: [
    { id: "low-latency",    label: "Low-Latency"    },
    { id: "balanced",       label: "Balanced"       },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  modelNames: {
    "a5|default|w8a8c8-mxfp8": "glm53",
    "a3|default|w8a8": "glm-5",
    "default|fp8": "zai-org/GLM-5.3",
    "default|bf16": "zai-org/GLM-5.3-BF16",
    "default|nvfp4": "RadixArk/GLM-5.3-NVFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    NODE0_IP:  { target: "command", label: "Head node IP",      default: "<node0-ip>"   },
    NODE_RANK: { target: "command", label: "This node rank",    default: "<node-rank>"  },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"     },
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
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
      aime26_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run aime26 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 64000 \\
  --temperature 1.0 --top-p 0.95 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
      aime25_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run aime25 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 64000 \\
  --temperature 1.0 --top-p 0.95 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
    },
    numPromptsByConc: { 1: 8, 16: 64, 64: 128, 256: 512, 1024: 2048, 4096: 8192 },
  },

  accuracyLabels: [
    ["aime26_pct", "AIME26",         "%"],
    ["aime25_pct", "AIME25",         "%"],
    ["gsm8k_pct", "GSM8K (1-shot)", "%"],
  ],

  dockerImages: {
    a3: "quay.io/ascend/sglang:cann9.1.0-a3-B080",
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    // >= v0.5.20 so `--*-parser auto` detects GLM-5.3 (#38297); the rocm700
    // line stopped at v0.5.19, so mi30x moves to the rocm720 build.
    mi355x: "lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260920",
    mi325x: "lmsysorg/sglang-rocm:v0.5.20-rocm720-mi30x-20260920",
    mi300x: "lmsysorg/sglang-rocm:v0.5.20-rocm720-mi30x-20260920",
  },

  runModes: (s) => ["a3", "a5"].includes(s.hw) ? ["python"] : ["python", "docker"],

  github: {
    cookbookModel: "zai-org/glm-5.3",
  },

  // Parser flags live in one overlay dim so every generated command gets them
  // without per-cell duplication; the Parsers card toggles derive on/off from
  // the composed flags. `auto` needs the GLM-5.3 template detection (v0.5.20+).
  overlayDims: [
    {
      id: "pdMode", title: "NPU PD mode", default: "unified",
      showWhen: (s) => ["a3", "a5"].includes(s.hw) && s.nodes === "multi-2",
      options: [
        { id: "unified", label: "Unified" },
        { id: "prefill", label: "Prefill" },
        { id: "decode", label: "Decode" },
        { id: "router", label: "Router" },
      ],
    },

    {
      id: "parsers",
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      title: "Parsers",
      default: "auto",
      options: [
        {
          id: "auto",
          label: "Auto (glm45 + glm47)",
          stripPrefixes: ["--reasoning-parser", "--tool-call-parser"],
          flags: [
            "--reasoning-parser auto",
            "--tool-call-parser auto",
          ],
        },
        {
          id: "off",
          label: "Off",
          stripPrefixes: ["--reasoning-parser", "--tool-call-parser"],
          flags: [],
        },
      ],
    },
  ],

  playgroundFeatures: {

    // ----- Card 1: "Attention Parallelism" -----
    // DSA prefill Context Parallelism (CP) splits the long-prefill attention across
    // `cp` ranks — runs on Hopper (H200) and Blackwell (B200/GB300/B300).
    // CP sizes auto-gate in the engine to the runtime derivation
    // attn_cp_size = tp/dp (a user-passed --attn-cp-size is overridden).
    // CP is single-machine only (tp_size <= 8). Interleave CP + DP-Attention
    // currently fails the runtime's attn_dp_size == 1 assert but is allowed here
    // with a warning (combined support is planned upstream).
    // Strategy knob: interleave (ex round-robin-split) is the default;
    // zigzag (ex in-seq-split) is exposed as an
    // experiment — the runtime auto-configures deepep + ep=tp for it and
    // restricts it to batch_size=1 (long-context single-request runs).
    attention: {
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      knobs: [
        { id: "tp", label: "TP", values: [null, 4, 8] },
        { id: "cp", label: "CP (DSA prefill)",
          values: [null, { value: 1, label: "Off" }, 4, 8],
          disable: [
            { when: { hw: ["mi355x", "mi325x", "mi300x"] },
              reason: "The ROCm DSA-CP path is not yet validated on AMD (MI300X/MI325X/MI355X) — keep CP off there for now." },
            { when: { nodes: ["multi-2"] },
              reason: "Prefill Context Parallel is single-machine only (SGLang asserts tp_size <= 8; cross-machine CP has precision issues)." },
          ] },
        { id: "cpStrategy", label: "CP Strategy",
          values: [
            null,
            "interleave",
            { value: "zigzag", label: "zigzag (experimental)" },
          ],
          disable: [
            { when: { hw: ["mi355x", "mi325x", "mi300x"] },
              reason: "The ROCm DSA-CP path is not yet validated on AMD (MI300X/MI325X/MI355X) — keep CP off there for now." },
            { when: { nodes: ["multi-2"] },
              reason: "Prefill Context Parallel is single-machine only (SGLang asserts tp_size <= 8; cross-machine CP has precision issues)." },
          ] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card 2: "MoE Parallelism" -----
    moe: {
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 4, 8] },
    },

    // ----- Card 3: "Parsers" -----
    parsers: {
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    // ----- Card 4: "Speculative Decoding" -----
    // GLM-5.3 ships a single MTP (nextn) layer; index_share_for_mtp_iteration reuses the
    // DSA indexer topk across draft steps (topk==1 only). DFlash2 is the one
    // algorithm no Deploy cell ships: its draft is a separate checkpoint.
    speculative: {
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp-516", label: "EAGLE / MTP 5-1-6 (low-latency)",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 5",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 6"],
          disable: { hw: ["mi355x", "mi325x", "mi300x"] },
          disableReason: "MTP/EAGLE speculative decoding is not yet validated on AMD ROCm (MI300X/MI325X/MI355X): the gfx950 spec-decode draft kernel is not yet validated and at --speculative-num-steps > 3 hits a separate build issue; the DSA nextn draft path is CUDA-only." },
        { id: "mtp-112", label: "EAGLE / MTP 1-1-2 (balanced)",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 1",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 2"],
          disable: { hw: ["mi355x", "mi325x", "mi300x"] },
          disableReason: "MTP/EAGLE speculative decoding is not yet validated on AMD ROCm (MI300X/MI325X/MI355X): the gfx950 spec-decode draft kernel is not yet validated and at --speculative-num-steps > 3 hits a separate build issue; the DSA nextn draft path is CUDA-only." },
        { id: "dflash", label: "DFlash2 (block diffusion)",
          // Block-wise draft from a separate checkpoint, not the in-checkpoint
          // MTP layer. The block size (8) comes from the draft's own
          // dflash_config, so no --speculative-num-draft-tokens here. The draft
          // is a small dense model and does not run on the target's DSA
          // backends, hence the explicit draft attention backend.
          flags: ["--speculative-algorithm DFLASH",
                  "--speculative-draft-model-path incoai/GLM-5.3-DFlash2",
                  "--speculative-draft-attention-backend fa4"],
          note: "⚠️ The draft is a separate checkpoint: fetch incoai/GLM-5.3-DFlash2 alongside the target. It is public but licensed CC BY-NC-ND 4.0 for research and evaluation.",
          disable: [
            { when: { dpAttnOn: [true] },
              reason: "DFLASH speculative decoding does not support DP-Attention — the server rejects the combination at startup. Turn DP-Attention off in the Attention card above (the high-throughput recipes enable it)." },
            { when: { hw: ["mi355x", "mi325x", "mi300x"] },
              reason: "DFLASH speculative decoding only supports CUDA and NPU devices; the server rejects it on ROCm at startup." },
          ] },
      ],
    },

    // ----- Card 5: "PD Disaggregation" -----
    // GLM-5.3 is a DSA model (same family as DeepSeek-V3.2/V4) and supports
    // prefill/decode disaggregation. Owns the `--disaggregation-*` flags; the
    // engine also pins role-specific serving ports (spaced apart) so prefill +
    // decode don't collide on one host.
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        { id: "ascend", label: "Ascend", hide: { hw: ["h200", "b200", "b300", "gb300", "mi355x", "mi325x", "mi300x"] } },
        // Mooncake (recommended). The NCCL/MNNVL env is only needed on the
        // NVLink-multinode Grace-Blackwell platform (GB300 here).
        { id: "mooncake", label: "Mooncake", hide: { hw: ["a3", "a5"] },
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb300"] } },
        { id: "nixl",     label: "NiXL", hide: { hw: ["a3", "a5"] } },
      ],
      // No IB-device knob: mooncake auto-detects the HCA. Pass
      // --disaggregation-ib-device only if discovery picks the wrong NIC
      // (see Configuration Tips).
      // Router fronting the prefill + decode roles; substitute <prefill-host>/<decode-host>.
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`,
      },
    },

    // ----- Card 6: "Hierarchical KV Cache" -----
    hicache: {
      excludesHw: ["a3", "a5"],
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
      ],
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },
  },

  // Ascend recipes are transcribed from the linked tutorial. PD roles own their
  // worker topology; changing a GPU overlay must never rewrite these commands.
  resolveRecipe: (cell, s) => {
    if (!(["a3", "a5"].includes(s.hw) && s.nodes === "multi-2")) return cell;
    const role = s.pdMode;
    if (!["prefill", "decode", "router"].includes(role)) return cell;
    const router = { port: 30000, command: s.hw === "a5" ? config.npuRouterCommand.replace("--policy round_robin", "--policy cache_aware") : config.npuRouterCommand };
    if (role === "router") return { ...cell, pd: true, pdMode: "router",
      nnodes: 1, verified: false, env: [], flags: [], warn: undefined,
      commands: { python: router.command, docker: `docker run --network host --rm quay.io/ascend/sglang:cann9.1.0-a3-B080 ${config.npuRouterCommand}` }, router };
    const recipe = config.npuPdRecipes.find((r) => r.hw === s.hw && r.pdMode === role);
    if (!recipe) return cell;
    const flags = recipe.flags.map((flag) => flag.startsWith("--port ")
      ? "--port " + (recipe.pdMode === "decode" ? 30100 : 30000) : flag);
    if (recipe.nnodes > 1) flags.push("--nnodes " + recipe.nnodes,
      "--node-rank {{NODE_RANK}}", "--dist-init-addr {{NODE0_IP}}:20000");
    const command = [...recipe.env, "sglang serve", ...flags.map((flag) => "  " + flag)].join(" \\\n");
    return { ...cell, ...recipe, pd: true, verified: false, warn: undefined, router,
      commands: { python: command } };
  },

  npuRouterCommand: "python3 -m sglang_router.launch_router \\\n  --pd-disaggregation \\\n  --prefill http://<prefill-head-ip>:30000 8998 \\\n  --decode http://<decode-head-ip>:30100 \\\n  --host {{HOST_IP}} --port {{PORT}} \\\n  --policy round_robin",

  npuPdRecipes: [
    {
      hw: "a3",
      pdMode: "prefill",
      nnodes: 2,
      env: [
        "SGLANG_SET_CPU_AFFINITY=1",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "STREAMS_PER_DEVICE=32",
        "ASCEND_MF_LOG_LEVEL=2",
        "ASCEND_MF_STORE_URL=\"tcp://<prefill-head-ip>:24707\"",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200",
        "HCCL_SOCKET_IFNAME=\"<network-interface>\"",
        "GLOO_SOCKET_IFNAME=\"<network-interface>\"",
        "HCCL_BUFFSIZE=1200",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=72",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024",
        "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
        "TASK_QUEUE_ENABLE=2",
        "ENABLE_PROFILING=0",
        "SGLANG_PP_LAYER_PARTITION=18,20,24,16"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3\"",
        "--trust-remote-code",
        "--tp-size 8",
        "--pp-size 4",
        "--dp-size 1",
        "--attn-cp-size 8",
        "--disaggregation-mode prefill",
        "--disaggregation-bootstrap-port 8998",
        "--mem-fraction-static 0.72",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--enable-prefill-cp",
        "--cp-strategy interleave",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--moe-dense-tp 1",
        "--load-balance-method round_robin",
        "--disaggregation-transfer-backend ascend",
        "--max-running-requests 64",
        "--served-model-name glm-5",
        "--chunked-prefill-size 4096",
        "--max-prefill-tokens 180000",
        "--moe-a2a-backend deepep",
        "--deepep-mode normal",
        "--disable-shared-experts-fusion",
        "--disable-cuda-graph",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      hw: "a3",
      pdMode: "decode",
      nnodes: 2,
      env: [
        "SGLANG_SET_CPU_AFFINITY=1",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "STREAMS_PER_DEVICE=32",
        "ASCEND_MF_LOG_LEVEL=2",
        "ASCEND_MF_STORE_URL=\"tcp://<prefill-head-ip>:24707\"",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200",
        "HCCL_SOCKET_IFNAME=\"<network-interface>\"",
        "GLOO_SOCKET_IFNAME=\"<network-interface>\"",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "DEEPEP_HCCL_BUFFSIZE=200",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16",
        "TASK_QUEUE_ENABLE=0",
        "SGLANG_NPU_USE_MULTI_STREAM=1"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3\"",
        "--trust-remote-code",
        "--tp-size 32",
        "--dp-size 32",
        "--ep-size 32",
        "--disaggregation-mode decode",
        "--enable-dp-attention",
        "--mem-fraction-static 0.97",
        "--max-running-requests 32",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--served-model-name glm-5",
        "--moe-a2a-backend deepep",
        "--deepep-mode low_latency",
        "--cuda-graph-bs-decode 1 2 3",
        "--disaggregation-transfer-backend ascend",
        "--watchdog-timeout 9000",
        "--context-length 180000",
        "--tokenizer-worker-num 4",
        "--disable-shared-experts-fusion",
        "--load-balance-method round_robin",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      hw: "a5",
      pdMode: "prefill",
      nnodes: 2,
      env: [
        "HCCL_CONNECT_TIMEOUT=300",
        "HCCL_EXEC_TIMEOUT=68",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "ACL_DEVICE_SYNC_TIMEOUT=60",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "MF_HYBM_USE_VMM_SEGMENT=1",
        "STREAMS_PER_DEVICE=32",
        "ASCEND_USE_FIA=1",
        "SGLANG_NPU_USE_MLAPO=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35",
        "TRANSFORMERS_VERBOSITY=error",
        "HCCL_HOST_SOCKET_PORT_RANGE=auto",
        "GLOO_SOCKET_IFNAME=\"<gloo-interface>\"",
        "ASCEND_MF_TRANSFER_PROTOCOL=device_urma",
        "ASCEND_MF_STORE_URL=\"tcp://<memfabric-store-host>:31001\"",
        "ASCEND_MF_LOG_LEVEL=2",
        "HCCL_SOCKET_IFNAME=\"<network-interface>\"",
        "DEEPEP_HCCL_BUFFSIZE=2048",
        "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1",
        "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3\"",
        "--trust-remote-code",
        "--tp-size 16",
        "--dp 4",
        "--served-model-name glm53",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--load-balance-method round_robin",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend ascend",
        "--disaggregation-bootstrap-port 8998",
        "--attention-backend ascend",
        "--device npu",
        "--watchdog-timeout 9000",
        "--max-running-requests 64",
        "--mem-fraction-static 0.82",
        "--quantization modelslim",
        "--max-prefill-tokens 2048000",
        "--chunked-prefill-size 16384",
        "--kv-cache-dtype \"fp8_e4m3\"",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--disable-cuda-graph",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      hw: "a5",
      pdMode: "decode",
      nnodes: 2,
      env: [
        "HCCL_CONNECT_TIMEOUT=300",
        "HCCL_EXEC_TIMEOUT=68",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "ACL_DEVICE_SYNC_TIMEOUT=60",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "MF_HYBM_USE_VMM_SEGMENT=1",
        "STREAMS_PER_DEVICE=32",
        "ASCEND_USE_FIA=1",
        "SGLANG_NPU_USE_MLAPO=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35",
        "TRANSFORMERS_VERBOSITY=error",
        "HCCL_HOST_SOCKET_PORT_RANGE=auto",
        "GLOO_SOCKET_IFNAME=\"<gloo-interface>\"",
        "ASCEND_MF_TRANSFER_PROTOCOL=device_urma",
        "ASCEND_MF_STORE_URL=\"tcp://<memfabric-store-host>:31001\"",
        "ASCEND_MF_LOG_LEVEL=2",
        "HCCL_SOCKET_IFNAME=\"<network-interface>\"",
        "DEEPEP_HCCL_BUFFSIZE=400",
        "SGLANG_ENABLE_SPEC_V2=1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3\"",
        "--trust-remote-code",
        "--tp-size 16",
        "--dp 4",
        "--served-model-name glm53",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--load-balance-method round_robin",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend ascend",
        "--attention-backend ascend",
        "--device npu",
        "--watchdog-timeout 9000",
        "--max-running-requests 64",
        "--mem-fraction-static 0.86",
        "--quantization modelslim",
        "--max-prefill-tokens 2048000",
        "--chunked-prefill-size 16384",
        "--kv-cache-dtype \"fp8_e4m3\"",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--cuda-graph-bs-decode 16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 4",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 5",
        "--speculative-draft-kv-cache-dtype bf16",
        "--enable-metrics",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    }
  ],

  cells: [
    // ====================================================================
    // H200 + FP8 (Hopper) — TP8.
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attn-dp-size 8",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        // Large chunked-prefill is the main high-throughput tuning lever;
        // max-running should track the available KV capacity.
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 + FP8 (Blackwell) — TP8.
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attn-dp-size 8",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        // Large chunked-prefill is the main high-throughput tuning lever;
        // max-running should track the available KV capacity.
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 + FP8 (Grace-Blackwell, 4-GPU single node) — TP4.
    // The flags follow the B200 recipe with TP4/DP4 for a 4-GPU node.
    // Stage the weights on node-local NVMe first — shared cluster-storage reads are slow.
    // ====================================================================
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--attn-dp-size 4",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        // Same prefill lever as H200/B200; max-running tracks the TP4 KV capacity.
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 + FP8 (Blackwell Ultra, 8-GPU single node) — TP8.
    // The recipe follows the B200 FP8 path.
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attn-dp-size 8",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 + BF16 (Blackwell Ultra, 8-GPU single node) — TP8.
    // The unquantized GLM-5.3 (~700B, ~1.51 TB) only fits single-node on 8xB300
    // (~2.1 TB HBM); smaller GPUs need multi-node (e.g. 2x 8xH200). Balanced/HT run plain TP8
    // without DP-Attention or DeepEP.
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.9",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--mem-fraction-static 0.9",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // BF16 multi-node (inferred) — the 1.51 TB checkpoint spread over 2 nodes.
    // 2x 8xH200 / 2x 8xB200 at TP16, 2x 4xGB300 at TP8. The engine injects
    // --nnodes / --node-rank / --dist-init-addr from the Multi-Nodes selector.
    // Recipes reuse the single-node B300 flags and remain unverified.
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--mem-fraction-static 0.85",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--mem-fraction-static 0.85",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--mem-fraction-static 0.85",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },


    // ====================================================================
    // NVFP4 — RadixArk/GLM-5.3-NVFP4 (Model Optimizer, experts-only W4A4).
    // Same runtime contract as nvidia/GLM-5.2-NVFP4 (experts-only NVFP4, KV FP8,
    // no per-tensor k/v scale tensors), so the GLM-5.2 NVFP4 recipes carry over:
    // TP8 on B200/B300, TP4 on GB300; low-latency MTP 5-1-6, balanced MTP 2-1-3.
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--attn-dp-size 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.92",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--attn-dp-size 8",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.92",
        "--max-running-requests 512",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--max-prefill-tokens 8192",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--attn-dp-size 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.92",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--quantization modelopt_fp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--max-prefill-tokens 8192",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--quantization modelopt_fp4",
        "--attn-dp-size 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.92",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // AMD MI300X / MI325X / MI355X (ROCm) — TP8, DSA tilelang backend.
    // No MTP: disabled in the Speculative card for AMD (the gfx950 spec-decode
    // draft kernel is not yet validated, and num-steps>3 hits a separate build
    // issue). Strategies differ only by batch-shaping levers
    // (cuda-graph-max-bs / max-running-requests / chunked-prefill):
    //   low-latency      — large chunked-prefill, default bs.
    //   balanced         — chunked-prefill 32768 + bs128, max-running 80.
    //   high-throughput  — bs256, max-running 256.
    // BF16 (~1.51 TB) only fits single-node on MI325X (2 TB) / MI355X (2.3 TB);
    // MI300X (1.5 TB) needs multi-node, so its BF16 cells are omitted.
    // ====================================================================
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // Ascend NPU: final verification of the tutorial recipes is in progress.
    {
      match: {
        hw: "a3",
        variant: "default",
        quant: "w8a8",
        strategy: "low-latency",
        nodes: "single"
      },
      verified: false,
      verificationStatus: "in-progress",
      warn: "Reference adapted from GLM-5.2; not yet verified on GLM-5.3 hardware. See [the Ascend tutorial](/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_3#single-node-deployment).",
      env: [
        "SGLANG_SET_CPU_AFFINITY=1",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "STREAMS_PER_DEVICE=32",
        "HCCL_BUFFSIZE=1000",
        "HCCL_SOCKET_IFNAME=lo",
        "GLOO_SOCKET_IFNAME=lo",
        "SGLANG_NPU_USE_MULTI_STREAM=1"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3\"",
        "--trust-remote-code",
        "--tp-size 16",
        "--attention-backend ascend",
        "--device npu",
        "--chunked-prefill-size 4096",
        "--max-prefill-tokens 8192",
        "--mem-fraction-static 0.75",
        "--max-running-requests 32",
        "--served-model-name glm-5",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--disable-shared-experts-fusion",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      match: {
        hw: "a3",
        variant: "default",
        quant: "w8a8",
        strategy: "low-latency",
        nodes: "multi-2"
      },
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_SET_CPU_AFFINITY=1",
        "STREAMS_PER_DEVICE=32",
        "HCCL_SOCKET_IFNAME=\"<network-interface>\"",
        "GLOO_SOCKET_IFNAME=\"<network-interface>\"",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=40",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
        "HCCL_BUFFSIZE=2000"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3\"",
        "--trust-remote-code",
        "--tp-size 32",
        "--dp-size 16",
        "--attention-backend ascend",
        "--device npu",
        "--enable-dp-attention",
        "--chunked-prefill-size 65536",
        "--max-prefill-tokens 280000",
        "--mem-fraction-static 0.68",
        "--served-model-name glm-5",
        "--cuda-graph-bs-decode 2",
        "--max-running-requests 32",
        "--quantization modelslim",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--load-balance-method round_robin",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      match: {
        hw: "a5",
        variant: "default",
        quant: "w8a8c8-mxfp8",
        strategy: "low-latency",
        nodes: "multi-2"
      },
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "DEEPEP_HCCL_BUFFSIZE=1024",
        "HCCL_CONNECT_TIMEOUT=300",
        "HCCL_EXEC_TIMEOUT=68",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "ACL_DEVICE_SYNC_TIMEOUT=60",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "STREAMS_PER_DEVICE=32",
        "ASCEND_USE_FIA=1",
        "SGLANG_NPU_USE_MLAPO=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35",
        "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1",
        "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200",
        "SGLANG_ENABLE_SPEC_V2=1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "TRANSFORMERS_VERBOSITY=error",
        "HCCL_HOST_SOCKET_PORT_RANGE=auto",
        "GLOO_SOCKET_IFNAME=\"<gloo-interface>\"",
        "HCCL_SOCKET_IFNAME=\"<network-interface>\""
      ],
      flags: [
        "--model-path \"/models/GLM-5.3\"",
        "--trust-remote-code",
        "--tp-size 16",
        "--dp 1",
        "--served-model-name glm53",
        "--attention-backend ascend",
        "--device npu",
        "--watchdog-timeout 9000",
        "--max-running-requests 8",
        "--mem-fraction-static 0.8",
        "--quantization modelslim",
        "--max-prefill-tokens 2048000",
        "--chunked-prefill-size 16384",
        "--kv-cache-dtype fp8_e4m3",
        "--speculative-draft-kv-cache-dtype bf16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--load-balance-method round_robin",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--enable-metrics",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 4",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 5",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
  ],
};
