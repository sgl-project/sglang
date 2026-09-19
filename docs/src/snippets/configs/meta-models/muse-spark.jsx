// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// Recipes follow the Muse Spark 1.2 deployment specification; hardware
// placeholders intentionally have no cells until launch recipes are available.
export const config = {
  modelName: "Muse Spark 1.2",
  supportedHardware: ["b300", "gb300", "b200", "h200", "mi355x", "xeon"],
  hardware: [
    { id: "xeon", label: "Xeon", vendor: "intel", vram: "Coming soon" },
  ],
  showPlaygroundLink: false,

  matchDims: [
    {
      id: "quant",
      title: "Quantization",
      options: [
        { id: "nvfp4", label: "NVFP4" },
        { id: "bf16", label: "BF16" },
      ],
    },
    {
      id: "tep",
      title: "TEP Size",
      options: [
        {
          id: "4",
          label: "4",
          disabled: (s) => s.quant === "bf16" || s.hw === "h200",
          disableReason: "TEP 4 is unavailable for BF16 checkpoints and Hopper GPUs; use TEP 8.",
        },
        { id: "8", label: "8" },
      ],
    },
  ],

  overlayDims: [
    {
      id: "sequenceParallel",
      title: "Sequence Parallel + Prefill MegaMoE",
      default: "on",
      options: [
        {
          id: "on",
          label: "On",
          flags: ["--enable-layernorm-sp", "--enable-megamoe-for-layernorm-sp"],
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
        },
        { id: "off", label: "Off" },
      ],
    },
    {
      id: "mmTransport",
      title: "Multimodal Feature Transport",
      default: "cpu",
      options: [
        { id: "cpu", label: "CPU", flags: ["--mm-feature-transport cpu"] },
        { id: "cuda_ipc", label: "Cuda IPC", flags: ["--mm-feature-transport cuda_ipc"] },
      ],
    },
    {
      id: "pdMode",
      title: "PD Disaggregation",
      default: "collocated",
      options: [
        { id: "collocated", label: "Collocated" },
        {
          id: "prefill",
          label: "Prefill Node",
          flags: ["--disaggregation-mode prefill", "--disaggregation-transfer-backend mooncake"],
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
      default: "l2",
      options: [
        {
          id: "l2",
          label: "L1+L2(Host)",
          flags: (s) => s.pdMode === "decode" ? [] : [
            "--enable-hierarchical-cache", "--hicache-size 200",
          ],
        },
        { id: "off", label: "Off" },
      ],
    },
    {
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
    bf16: "meta-models/Muse-Spark-1.2-816B-A42B-open",
  },
  dockerImages: {
    b300: "lmsysorg/sglang:muse-spark-1.2",
    gb300: "lmsysorg/sglang:muse-spark-1.2",
    b200: "lmsysorg/sglang:muse-spark-1.2",
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
  ],
};
