// Muse Spark Open Deployment recipes preserve the original default commands.
// Deployment selects hardware, quantization, and strategy; further tuning lives
// in Playground. NVIDIA uses FA4, MI355X uses Triton for DFlash draft attention.
// Xeon keeps its CPU-engine INT4 recipe. H200 runs NVFP4 on one node with
// Marlin W4A16 experts and BF16 at TP16 across two nodes; it has no
// SP/MegaMoE (SM100-only kernels).
export const config = {
  modelName: "Muse Spark Open",
  supportedHardware: ["b300", "gb300", "b200", "h200", "mi355x", "xeon"],
  hardware: [
    { id: "xeon", label: "Xeon", vendor: "cpu", vram: "CPU" },
  ],
  showPlaygroundLink: false,

  matchDims: [
    {
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency", label: "Low Latency", subtitle: "Current defaults" },
        { id: "high-throughput", label: "High Throughput", subtitle: "Spec decode off" },
      ],
    },
    {
      id: "quant",
      title: "Quantization",
      options: [
        {
          id: "nvfp4",
          label: "NVFP4",
          showWhen: (s) => s.hw !== "xeon",
        },
        {
          id: "mxfp4",
          label: "MXFP4",
          showWhen: (s) => s.hw !== "xeon",
          disabled: (s) => s.hw !== "mi355x",
          disableReason: "Quark MXFP4 is the MI355X recipe; NVIDIA cells use NVFP4 or BF16.",
        },
        { id: "bf16", label: "BF16", showWhen: (s) => s.hw !== "xeon" },
        // INT4 weights, published for the CPU engine.
        { id: "int4", label: "INT4", showWhen: (s) => s.hw === "xeon" },
      ],
    },
  ],

  modelNames: {
    nvfp4: "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    mxfp4: "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4",
    bf16: "meta-models/Muse-Spark-Open-816B-A42B",
    int4: "intel-ai/Muse-Spark-Open-816B-A42B-INT4",
  },
  dockerImages: {
    b300: "lmsysorg/sglang:dev-muse-spark-open",
    gb300: "lmsysorg/sglang:dev-muse-spark-open",
    b200: "lmsysorg/sglang:dev-muse-spark-open",
    h200: "lmsysorg/sglang:dev-muse-spark-open",
    mi355x: "lmsysorg/sglang-rocm:muse-spark-open",
    xeon: "lmsysorg/sglang:muse-spark-1.2-xeon",
  },
  dockerHostNetworkWhen: (s) => ["prefill", "decode"].includes(s.pdMode),
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    NODE0_IP: { target: "command", label: "Head node IP", default: "<node0-ip>" },
    NODE_RANK: { target: "command", label: "This node rank", default: "<node-rank>" },
    DRAFT_MODEL_ID: { target: "command", label: "Dflash V2 draft model", default: "RadixArk/Muse-Spark-Open-DFlash2" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    CURL_DECODE_PORT: { target: "curl", label: "Decode server port", default: "30100" },
  },
  curl: (s) => `curl http://{{CURL_HOST}}:${s.pdMode === "decode" ? "{{CURL_DECODE_PORT}}" : "{{CURL_PORT}}"}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,
  github: {
    cookbookModel: "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
  },

  playgroundFeatures: {
    baseSummaryDims: ["hw", "quant"],
    normalizeDeltas: (deltas, derived, base, changedAxis) => {
      const next = {
        ...deltas,
        attention: { ...deltas.attention },
        moe: { ...deltas.moe },
        flagSelects: { ...deltas.flagSelects },
      };
      // Either selector drives the same degree. Keep nulls on initial load
      // so the inherited recipe is unchanged. Xeon has no EP setting.
      if (base.hw !== "xeon") {
        const size = changedAxis === "moe"
          ? next.moe.ep : (next.attention.tp ?? next.moe.ep);
        if (size != null) {
          next.attention.tp = size;
          next.moe.ep = size;
        }
      } else {
        next.moe.ep = null;
      }
      const sp = next.flagSelects.prefillSp ?? derived.flagSelects?.prefillSp;
      if (sp !== "on") next.flagSelects.prefillMegaMoe = "off";
      return next;
    },
    attention: {
      knobs: [{
        id: "tp", label: "TP",
        values: [
          null, 4, 8, 16,
          // Keep the CPU recipe's inherited degree visible and selectable.
          { value: 6, label: "Inherited (6)", hide: { hw: ["b300", "gb300", "b200", "h200", "mi355x"] } },
        ],
      }],
    },
    moe: {
      showWhen: (s) => s.hw !== "xeon",
      ep: { label: "EP", values: [null, 4, 8, 16] },
    },
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser muse", default: true },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser muse", default: true },
      ],
    },
    pdDisagg: {
      exportEnv: true,
      stripEnv: [
        "MC_INTRANODE_NVLINK", "MC_TCP_ENABLE_CONNECTION_POOL",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "CUDA_VISIBLE_DEVICES",
        "NCCL_NVLS_ENABLE", "NCCL_MNNVL_ENABLE", "NCCL_CUMEM_ENABLE", "MC_FORCE_MNNVL",
      ],
      ports: { prefill: 30001, decode: 30002 },
      modes: [
        { id: "off", label: "Off" },
        { id: "prefill", label: "Prefill" },
        {
          id: "decode", label: "Decode",
          stripPrefixes: [
            "--enable-layernorm-sp", "--enable-megamoe-for-layernorm-sp",
            "--enable-hierarchical-cache", "--hicache-size",
          ],
        },
      ],
      roleOverrides: [
        {
          mode: "prefill", when: { hw: ["b300", "b200", "h200"] },
          env: [
            "MC_INTRANODE_NVLINK=true",
            "MC_TCP_ENABLE_CONNECTION_POOL=true",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK",
            "CUDA_VISIBLE_DEVICES=0,1,2,3",
          ],
        },
        {
          mode: "decode", when: { hw: ["b300", "b200", "h200"] },
          env: [
            "MC_INTRANODE_NVLINK=true",
            "MC_TCP_ENABLE_CONNECTION_POOL=true",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK",
            "CUDA_VISIBLE_DEVICES=4,5,6,7",
          ],
        },
        {
          mode: "prefill", when: { hw: ["gb300"] },
          env: [
            "NCCL_NVLS_ENABLE=1",
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK",
            "MC_FORCE_MNNVL=1",
          ],
        },
        {
          mode: "decode", when: { hw: ["gb300"] },
          env: [
            "NCCL_NVLS_ENABLE=1",
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK",
            "MC_FORCE_MNNVL=1",
          ],
        },
      ],
      transferBackends: [{ id: "mooncake", label: "Mooncake" }],
      router: {
        port: 8000,
        command:
`sglang-router launch \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}}`,
      },
    },
    flagSelects: [
      {
        id: "prefillSp", title: "Prefill Sequence Parallel",
        showWhen: (s) => s.pdMode !== "decode",
        stripPrefixes: ["--enable-layernorm-sp"],
        options: [
          {
            id: "on", label: "On", flags: ["--enable-layernorm-sp"],
            disable: { hw: ["h200"] },
            disableReason: "Layernorm SP needs SM100 collective GEMM kernels; unavailable on H200.",
          },
          { id: "off", label: "Off" },
        ],
      },
      {
        id: "prefillMegaMoe", title: "Prefill MegaMoE",
        showWhen: (s) => s.pdMode !== "decode",
        stripPrefixes: ["--enable-megamoe-for-layernorm-sp"],
        options: [
          {
            id: "on", label: "On", flags: ["--enable-megamoe-for-layernorm-sp"],
            disable: [
              { when: { hw: ["h200"] }, reason: "Prefill MegaMoE needs SM100 kernels; unavailable on H200." },
              { when: { "flagSelect.prefillSp": ["off"] }, reason: "Enable Prefill Sequence Parallel first." },
            ],
          },
          { id: "off", label: "Off" },
        ],
      },
      {
        id: "speculative", title: "Speculative Decoding",
        default: (s) => s.hw === "xeon" || s.strategy === "high-throughput" ? "off" : "dflash-v2",
        stripPrefixes: [
          "--speculative-algorithm", "--speculative-draft-model-path",
          "--speculative-draft-attention-backend", "--speculative-num-draft-tokens",
          "--speculative-use-block-verification",
        ],
        options: [
          { id: "off", label: "Off" },
          {
            id: "dflash-v2", label: "DFlash V2",
            flags: (values, s) => [
              "--speculative-algorithm DFLASH",
              "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
              `--speculative-draft-attention-backend ${values.draftAttentionBackend || (s.hw === "mi355x" ? "triton" : "fa4")}`,
              "--speculative-num-draft-tokens 8",
            ],
          },
        ],
      },
      {
        id: "draftAttentionBackend", title: "Draft attention backend",
        default: (s) => s.hw === "mi355x" ? "triton" : "fa4",
        showWhen: (s, values) => values.speculative === "dflash-v2",
        stripPrefixes: ["--speculative-draft-attention-backend"],
        options: [
          { id: "fa4", label: "fa4", flags: ["--speculative-draft-attention-backend fa4"] },
          { id: "triton", label: "triton", flags: ["--speculative-draft-attention-backend triton"] },
        ],
      },
      {
        id: "draftTokens", title: "Num draft tokens", control: "slider", default: "8",
        showWhen: (s, values) => values.speculative === "dflash-v2",
        stripPrefixes: ["--speculative-num-draft-tokens"],
        range: { min: 1, max: 8, flag: "--speculative-num-draft-tokens" },
      },
      {
        id: "blockVerification", title: "Block Verification", control: "toggle", default: "off",
        showWhen: (s, values) => values.speculative === "dflash-v2",
        stripPrefixes: ["--speculative-use-block-verification"],
        options: [
          { id: "off", label: "Off" },
          { id: "on", label: "On", flags: ["--speculative-use-block-verification"] },
        ],
      },
      {
        id: "mmTransport", title: "Multimodal Feature Transport", default: "cpu",
        stripPrefixes: ["--mm-feature-transport"],
        options: [
          // CPU preserves the inherited command without adding an override.
          { id: "cpu", label: "cpu", flags: () => null },
          { id: "cuda_ipc", label: "cuda ipc", flags: ["--mm-feature-transport cuda_ipc"] },
        ],
      },
      {
        id: "kvCache", title: "KV cache",
        default: (s) => s.kvCache === "bfloat16" ? "bf16" : "fp8_e4m3",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "fp8_e4m3", label: "fp8_e4m3", flags: ["--kv-cache-dtype fp8_e4m3"] },
          { id: "bf16", label: "bf16", flags: ["--kv-cache-dtype bfloat16"] },
        ],
      },
      {
        id: "hicache", title: "HiCache",
        default: (s) => ["xeon", "mi355x"].includes(s.hw) ? "off" : "l2",
        showWhen: (s) => s.pdMode !== "decode",
        stripPrefixes: ["--enable-hierarchical-cache", "--hicache-size"],
        options: [
          { id: "off", label: "Off" },
          { id: "l2", label: "L1+L2(Host)", flags: ["--enable-hierarchical-cache", "--hicache-size 200"] },
        ],
      },
      {
        id: "hicacheSize", title: "HiCache size", control: "slider", default: "200",
        showWhen: (s, values) => s.pdMode !== "decode" && values.hicache === "l2",
        stripPrefixes: ["--hicache-size"],
        range: { min: 0, max: 400, flag: "--hicache-size" },
      },
    ],
  },

  // One recipe per hardware/quantization pair, with the former Deployment
  // defaults baked in. Playground starts from these same flags and environment.
  cells: [
    {
      match: { hw: "b300", quant: "nvfp4" },
      verified: true,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--quantization modelopt_fp4",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--enable-layernorm-sp",
        "--enable-megamoe-for-layernorm-sp",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend fa4",
        "--speculative-num-draft-tokens 8",
        "--mm-feature-transport cpu",
        "--enable-hierarchical-cache",
        "--hicache-size 200",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "b300", quant: "bf16" },
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dtype bfloat16",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--enable-layernorm-sp",
        "--enable-megamoe-for-layernorm-sp",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend fa4",
        "--speculative-num-draft-tokens 8",
        "--mm-feature-transport cpu",
        "--enable-hierarchical-cache",
        "--hicache-size 200",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "gb300", quant: "nvfp4" },
      verified: true,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--quantization modelopt_fp4",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--enable-layernorm-sp",
        "--enable-megamoe-for-layernorm-sp",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend fa4",
        "--speculative-num-draft-tokens 8",
        "--mm-feature-transport cpu",
        "--enable-hierarchical-cache",
        "--hicache-size 200",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "gb300", quant: "bf16" },
      nnodes: 2,
      // Expandable segments admit the two-node NVL72 TP8 group to custom
      // all-reduce v2; without them decode all-reduce falls back to NCCL.
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dtype bfloat16",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--enable-layernorm-sp",
        "--enable-megamoe-for-layernorm-sp",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend fa4",
        "--speculative-num-draft-tokens 8",
        "--mm-feature-transport cpu",
        "--enable-hierarchical-cache",
        "--hicache-size 200",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "b200", quant: "nvfp4" },
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--quantization modelopt_fp4",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--enable-layernorm-sp",
        "--enable-megamoe-for-layernorm-sp",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend fa4",
        "--speculative-num-draft-tokens 8",
        "--mm-feature-transport cpu",
        "--enable-hierarchical-cache",
        "--hicache-size 200",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "b200", quant: "bf16" },
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dtype bfloat16",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--enable-layernorm-sp",
        "--enable-megamoe-for-layernorm-sp",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend fa4",
        "--speculative-num-draft-tokens 8",
        "--mm-feature-transport cpu",
        "--enable-hierarchical-cache",
        "--hicache-size 200",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    // H200: Hopper has no TRT-LLM FP4 MoE, so NVFP4 experts run as Marlin
    // W4A16. BF16 leaves ~3.6 GB/GPU after weights, and the 1M-context verify
    // graphs take ~2.9 GB of it, so it runs at 0.88 to keep a usable KV pool.
    {
      match: { hw: "h200", quant: "nvfp4" },
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--quantization modelopt_fp4",
        "--moe-runner-backend marlin",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend fa4",
        "--speculative-num-draft-tokens 8",
        "--mm-feature-transport cpu",
        "--enable-hierarchical-cache",
        "--hicache-size 200",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "h200", quant: "bf16" },
      nnodes: 2,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--dtype bfloat16",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend fa4",
        "--speculative-num-draft-tokens 8",
        "--mm-feature-transport cpu",
        "--enable-hierarchical-cache",
        "--hicache-size 200",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "mi355x", quant: "mxfp4" },
      nnodes: 1,
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_USE_AITER_UNIFIED_ATTN=1",
        "MUSE_MOE_BACKEND=flydsl",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--attention-backend aiter",
        "--page-size 16",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend triton",
        "--speculative-num-draft-tokens 8",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "mi355x", quant: "nvfp4" },
      nnodes: 1,
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_USE_AITER_UNIFIED_ATTN=1",
        "MUSE_MOE_BACKEND=flydsl",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--quantization quark_mxfp4",
        "--attention-backend aiter",
        "--page-size 16",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend triton",
        "--speculative-num-draft-tokens 8",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "mi355x", quant: "bf16" },
      nnodes: 1,
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_USE_AITER_UNIFIED_ATTN=1",
        "MUSE_MOE_BACKEND=flydsl",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dtype bfloat16",
        "--attention-backend aiter",
        "--page-size 16",
        "--mem-fraction-static 0.92",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
        "--speculative-draft-attention-backend triton",
        "--speculative-num-draft-tokens 8",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
    {
      match: { hw: "xeon", quant: "int4" },
      nnodes: 1,
      env: [
        "SGLANG_USE_CPU_ENGINE=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 6",
        "--page-size 1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
      ],
    },
  ].flatMap((cell) => [
    {
      ...cell,
      match: { ...cell.match, strategy: "low-latency" },
    },
    {
      ...cell,
      match: { ...cell.match, strategy: "high-throughput" },
      verified: false,
      flags: cell.flags.filter((flag) => !flag.startsWith("--speculative-")),
    },
  ]),
};
