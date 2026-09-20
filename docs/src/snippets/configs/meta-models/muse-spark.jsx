// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// NVIDIA recipes follow the Muse Spark 1.2 deployment specification.
// MI355X cells are the ROCm aiter + FlyDSL profile (MXFP4 TEP 4/8, BF16
// TEP 8). Xeon is the CPU-engine profile: the INT4 checkpoint and TP 6
// (tensor parallel only — the CPU engine has no expert-parallel degree), with
// KV Cache Precision and Parsers as the only overlays that apply there.
// H200 remains a placeholder with no cells.
export const config = {
  modelName: "Muse Spark 1.2",
  supportedHardware: ["b300", "gb300", "b200", "h200", "mi355x", "xeon"],
  hardware: [
    { id: "xeon", label: "Xeon", vendor: "cpu", vram: "CPU" },
  ],
  showPlaygroundLink: false,

  matchDims: [
    {
      id: "quant",
      title: "Quantization",
      options: [
        {
          id: "nvfp4",
          label: "NVFP4",
          showWhen: (s) => s.hw !== "xeon",
          disabled: (s) => s.hw === "mi355x",
          disableReason: "NVFP4 is the NVIDIA recipe; MI355X uses Quark MXFP4 or BF16.",
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
    {
      id: "tep",
      title: "TEP Size",
      // Xeon has no size to pick — the CPU recipe is tensor parallel only and
      // pins --tp-size 6 — so the whole row hides there.
      showWhen: (s) => s.hw !== "xeon",
      options: [
        {
          id: "4",
          label: "4",
          showWhen: (s) => s.hw !== "xeon",
          disabled: (s) => s.quant === "bf16" || s.hw === "h200",
          disableReason: "TEP 4 is unavailable for BF16 checkpoints and Hopper GPUs; use TEP 8.",
        },
        // The Xeon cell still keys on a value of this dim; the option is
        // never rendered because the row above hides on that platform.
        { id: "6", label: "6", showWhen: (s) => s.hw === "xeon" },
        { id: "8", label: "8", showWhen: (s) => s.hw !== "xeon" },
      ],
    },
  ],

  overlayDims: [
    {
      id: "sequenceParallel",
      title: "Sequence Parallel + Prefill MegaMoE",
      showWhen: (s) => s.hw !== "xeon",
      default: "on",
      options: [
        {
          id: "on",
          label: "On",
          flags: ["--enable-layernorm-sp", "--enable-megamoe-for-layernorm-sp"],
          disabled: (s) => s.hw === "mi355x",
          disableReason: "Layernorm SP + Prefill MegaMoE is a NVIDIA recipe; it is not part of the MI355X launch.",
        },
        { id: "off", label: "Off" },
      ],
    },
    {
      id: "kvCache",
      title: "KV Cache Precision",
      default: "fp8_e4m3",
      options: [
        { id: "fp8_e4m3", label: "fp8_e4m3" },
        { id: "bfloat16", label: "bfloat16", flags: ["--kv-cache-dtype bfloat16"] },
      ],
    },
    {
      id: "speculative",
      title: "Speculative Decoding",
      showWhen: (s) => s.hw !== "xeon",
      default: "dflash-v2",
      options: [
        {
          id: "dflash-v2",
          label: "Dflash V2",
          flags: [
            "--speculative-algorithm DFLASH",
            "--speculative-draft-model-path {{DRAFT_MODEL_ID}}",
            "--speculative-draft-attention-backend fa4",
            "--speculative-num-draft-tokens 6",
          ],
          disabled: (s) => s.hw === "mi355x",
          disableReason: "Dflash V2 with fa4 draft attention is not validated on MI355X.",
        },
        { id: "off", label: "Off" },
      ],
    },
    {
      id: "mmTransport",
      title: "Multimodal Feature Transport",
      showWhen: (s) => s.hw !== "xeon" && s.hw !== "mi355x",
      default: "cpu",
      options: [
        { id: "cpu", label: "CPU", flags: ["--mm-feature-transport cpu"] },
        {
          id: "cuda_ipc",
          label: "Cuda IPC",
          flags: ["--mm-feature-transport cuda_ipc"],
          disabled: (s) => s.hw === "mi355x",
          disableReason: "CUDA IPC feature transport is NVIDIA-only; keep CPU on MI355X.",
        },
      ],
    },
    {
      id: "pdMode",
      title: "PD Disaggregation",
      showWhen: (s) => s.hw !== "xeon",
      default: "collocated",
      options: [
        { id: "collocated", label: "Collocated" },
        {
          id: "prefill",
          label: "Prefill Node",
          flags: ["--disaggregation-mode prefill", "--disaggregation-transfer-backend mooncake"],
          disabled: (s) => s.hw === "mi355x",
          disableReason: "Prefill/decode disaggregation is not part of the validated MI355X recipe.",
          env: (s) => s.hw === "gb300" ? [
            "NCCL_NVLS_ENABLE=1",
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK",
            "MC_FORCE_MNNVL=1",
          ] : ["b300", "b200", "h200"].includes(s.hw) ? [
            "MC_INTRANODE_NVLINK=true",
            "MC_TCP_ENABLE_CONNECTION_POOL=true",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK",
          ].concat(s.tep === "4" ? ["CUDA_VISIBLE_DEVICES=0,1,2,3"] : []) : [],
        },
        {
          id: "decode",
          label: "Decode Node",
          flags: ["--disaggregation-mode decode", "--disaggregation-transfer-backend mooncake"],
          disabled: (s) => s.hw === "mi355x",
          disableReason: "Prefill/decode disaggregation is not part of the validated MI355X recipe.",
          env: (s) => s.hw === "gb300" ? [
            "NCCL_NVLS_ENABLE=1",
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK",
            "MC_FORCE_MNNVL=1",
          ] : ["b300", "b200", "h200"].includes(s.hw) ? [
            "MC_INTRANODE_NVLINK=true",
            "MC_TCP_ENABLE_CONNECTION_POOL=true",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK",
          ].concat(s.tep === "4" ? ["CUDA_VISIBLE_DEVICES=4,5,6,7"] : []) : [],
        },
      ],
    },
    {
      id: "hicache",
      title: "HiCache",
      showWhen: (s) => s.hw !== "xeon",
      default: "l2",
      options: [
        {
          id: "l2",
          label: "L1+L2(Host)",
          flags: (s) => s.pdMode === "decode" ? [] : [
            "--enable-hierarchical-cache", "--hicache-size 200",
          ],
          disabled: (s) => s.hw === "mi355x",
          disableReason: "HiCache is not part of the validated MI355X recipe.",
        },
        { id: "off", label: "Off" },
      ],
    },
    {
      // Hardware-independent (server-side text handling), so it stays on Xeon.
      id: "parsers",
      title: "Parsers",
      default: "on",
      options: [
        {
          id: "on",
          label: "Reasoning/Tool",
          flags: ["--reasoning-parser muse", "--tool-call-parser muse"],
        },
        { id: "off", label: "Off" },
      ],
    },
  ],

  modelNames: {
    nvfp4: "meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open",
    mxfp4: "amd/Muse-Spark-1.2-816B-A42B-Quark-MXFP4",
    bf16: "meta-models/Muse-Spark-1.2-816B-A42B-open",
    int4: "intel-ai/Muse-Spark-1.2-816B-A42B-open-W4A16-RTN-AutoRound",
  },
  dockerImages: {
    b300: "lmsysorg/sglang:muse-spark-1.2",
    gb300: "lmsysorg/sglang:muse-spark-1.2",
    b200: "lmsysorg/sglang:muse-spark-1.2",
    mi355x: "lmsysorg/sglang-rocm:muse-spark-1.2",
    xeon: "lmsysorg/sglang:muse-spark-1.2-xeon",
  },
  dockerHostNetworkWhen: (s) => s.pdMode !== "collocated",
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    NODE0_IP: { target: "command", label: "Head node IP", default: "<node0-ip>" },
    NODE_RANK: { target: "command", label: "This node rank", default: "<node-rank>" },
    DRAFT_MODEL_ID: { target: "command", label: "Dflash V2 draft model", default: "<draft_model_id>" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
    CURL_DECODE_PORT: { target: "curl", label: "Decode server port", default: "30100" },
  },
  curl: (s) => `curl http://{{CURL_HOST}}:${s.pdMode === "decode" ? "{{CURL_DECODE_PORT}}" : "{{CURL_PORT}}"}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,
  github: {
    cookbookModel: "meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open",
  },

  // No verification badge is claimed before these recipes are measured.
  // The first cell supplies the B300 / NVFP4 / TEP 4 default selection.
  cells: [
    {
      match: { hw: "b300", quant: "nvfp4", tep: "4" },
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
      ],
    },
    {
      match: { hw: "b300", quant: "nvfp4", tep: "8" },
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--quantization modelopt_fp4",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", quant: "bf16", tep: "8" },
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", quant: "nvfp4", tep: "4" },
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
      ],
    },
    {
      match: { hw: "gb300", quant: "nvfp4", tep: "8" },
      nnodes: 2,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--quantization modelopt_fp4",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", quant: "bf16", tep: "8" },
      nnodes: 2,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "nvfp4", tep: "4" },
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
      ],
    },
    {
      match: { hw: "b200", quant: "nvfp4", tep: "8" },
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--quantization modelopt_fp4",
        "--moe-runner-backend flashinfer_trtllm",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "bf16", tep: "8" },
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MI355X ROCm — page-16 aiter attention + FlyDSL MoE. MXFP4 runs TEP 4 or
    // TEP 8; BF16 is TEP 8 only. Do not copy NVIDIA overlays (Dflash V2 /
    // MegaMoE / CUDA IPC / Mooncake PD / HiCache) onto these cells.
    {
      match: { hw: "mi355x", quant: "mxfp4", tep: "4" },
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
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "mxfp4", tep: "8" },
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
        "--attention-backend aiter",
        "--page-size 16",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "bf16", tep: "8" },
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
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // Intel Xeon (CPU engine), INT4 weights. SGLANG_USE_CPU_ENGINE=1 selects
    // the CPU runtime; --page-size 1 is what the CPU attention path expects.
    {
      match: { hw: "xeon", quant: "int4", tep: "6" },
      nnodes: 1,
      env: ["SGLANG_USE_CPU_ENGINE=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 6",
        "--page-size 1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
