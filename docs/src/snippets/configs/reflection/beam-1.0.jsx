export const config = {
  modelName: "Beam 1.0",
  supportedHardware: ["b300", "gb300", "h200"],
  groupHardware: false,
  runModes: ["docker"],
  showPlaygroundLink: false,
  accuracyLabels: [["gsm8k_pct", "GSM8K", "%"]],
  matchDims: [
    {
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency", label: "Low-Latency" },
        { id: "high-throughput", label: "High-Throughput" },
      ],
    },
    {
      id: "quant",
      title: "Quantization",
      options: [
        {
          id: "nvfp4", label: "NVFP4",
          disabled: (sel) => sel.hw === "h200",
          disableReason: "The NVFP4 recipe requires Blackwell; use FP8 or BF16 on H200.",
        },
        { id: "fp8", label: "FP8" },
        { id: "bf16", label: "BF16" },
      ],
    },
  ],
  modelNames: {
    nvfp4: "reflection/Beam-1.0-501B-A23B-NVFP4",
    fp8: "reflection/Beam-1.0-501B-A23B-FP8",
    bf16: "reflection/Beam-1.0-501B-A23B",
  },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    NODE0_IP: { target: "command", label: "Head node IP", default: "<node0-ip>" },
    NODE_RANK: { target: "command", label: "This node rank", default: "<node-rank>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"beam","messages":[{"role":"user","content":"Hello"}]}'`,
  dockerImages: {
    b300: "lmsysorg/sglang:beam",
    gb300: "lmsysorg/sglang:beam",
    h200: "lmsysorg/sglang:beam",
  },
  github: {
    cookbookModel: "reflection/Beam-1.0-501B-A23B",
  },
  cells: [
    {
      "match": {
        "hw": "b300",
        "strategy": "low-latency",
        "quant": "nvfp4"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2",
        "--speculative-dflash-block-size 8",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "b300",
        "strategy": "low-latency",
        "quant": "fp8"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2",
        "--speculative-dflash-block-size 8",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "b300",
        "strategy": "low-latency",
        "quant": "bf16"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2",
        "--speculative-dflash-block-size 8",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "b300",
        "strategy": "high-throughput",
        "quant": "nvfp4"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 4",
        "--attn-dp-size 4",
        "--enable-dp-lm-head",
        "--ep-size 4",
        "--quantization modelopt_fp4",
        "--moe-a2a-backend flashinfer_megamoe",
        "--moe-runner-backend flashinfer_megamoe",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "b300",
        "strategy": "high-throughput",
        "quant": "fp8"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 4",
        "--attn-dp-size 4",
        "--enable-dp-lm-head",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "b300",
        "strategy": "high-throughput",
        "quant": "bf16"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 8",
        "--attn-dp-size 8",
        "--enable-dp-lm-head",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "gb300",
        "strategy": "low-latency",
        "quant": "nvfp4"
      },
      "nnodes": 1,
      "verified": true,
      "env": [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NVLS_ENABLE=1"
      ],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2",
        "--speculative-dflash-block-size 8",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "gb300",
        "strategy": "low-latency",
        "quant": "fp8"
      },
      "nnodes": 1,
      "verified": true,
      "env": [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NVLS_ENABLE=1"
      ],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2",
        "--speculative-dflash-block-size 8",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "gb300",
        "strategy": "low-latency",
        "quant": "bf16"
      },
      "nnodes": 2,
      "verified": true,
      "env": [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NVLS_ENABLE=1"
      ],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2",
        "--speculative-dflash-block-size 8",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "gb300",
        "strategy": "high-throughput",
        "quant": "nvfp4"
      },
      "nnodes": 1,
      "verified": true,
      "env": [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NVLS_ENABLE=1"
      ],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 4",
        "--attn-dp-size 4",
        "--enable-dp-lm-head",
        "--ep-size 4",
        "--quantization modelopt_fp4",
        "--moe-a2a-backend flashinfer_megamoe",
        "--moe-runner-backend flashinfer_megamoe",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "gb300",
        "strategy": "high-throughput",
        "quant": "fp8"
      },
      "nnodes": 1,
      "verified": true,
      "env": [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NVLS_ENABLE=1"
      ],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 4",
        "--attn-dp-size 4",
        "--enable-dp-lm-head",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "gb300",
        "strategy": "high-throughput",
        "quant": "bf16"
      },
      "nnodes": 2,
      "verified": true,
      "env": [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "NCCL_NVLS_ENABLE=1"
      ],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 8",
        "--attn-dp-size 8",
        "--enable-dp-lm-head",
        "--attention-backend trtllm_mha",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "h200",
        "strategy": "low-latency",
        "quant": "fp8"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 8",
        "--attention-backend fa3",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2",
        "--speculative-dflash-block-size 8",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "h200",
        "strategy": "low-latency",
        "quant": "bf16"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 8",
        "--attention-backend fa3",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.90",
        "--enable-unified-memory",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2",
        "--speculative-dflash-block-size 8",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "h200",
        "strategy": "high-throughput",
        "quant": "fp8"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 8",
        "--attn-dp-size 8",
        "--enable-dp-lm-head",
        "--attention-backend fa3",
        "--page-size 64",
        "--context-length 1048576",
        "--mem-fraction-static 0.85",
        "--enable-unified-memory",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "h200",
        "strategy": "high-throughput",
        "quant": "bf16"
      },
      "nnodes": 1,
      "verified": true,
      "env": [],
      "flags": [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name beam",
        "--tp 8",
        "--attention-backend fa3",
        "--page-size 64",
        "--context-length 1048576",
        "--enable-unified-memory",
        "--tool-call-parser beam",
        "--reasoning-parser beam",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    }
  ]
};
