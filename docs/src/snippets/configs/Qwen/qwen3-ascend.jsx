// A3 measured recipes and A2 validation candidates; status is recorded per cell.
export const config = {
  "modelName": "Qwen3 on Ascend",
  "supportedHardware": [
    "a3",
    "a2"
  ],
  "groupHardware": false,
  "matchDims": [
    {
      "id": "variant",
      "title": "Model",
      "options": [
        {
          "id": "8b",
          "label": "Qwen3-8B"
        },
        {
          "id": "30b-a3b",
          "label": "Qwen3-30B-A3B"
        },
        {
          "id": "32b",
          "label": "Qwen3-32B"
        },
        {
          "id": "235b-a22b",
          "label": "Qwen3-235B-A22B"
        }
      ]
    },
    {
      "id": "quant",
      "title": "Quantization",
      "options": [
        {
          "id": "w8a8",
          "label": "W8A8 (ModelSlim)"
        },
        {
          "id": "bf16",
          "label": "BF16"
        }
      ]
    },
    {
      "id": "strategy",
      "title": "Workload",
      "options": [
        {
          "id": "low-latency",
          "label": "Low latency"
        },
        {
          "id": "high-throughput",
          "label": "High throughput"
        }
      ]
    },
    {
      "id": "nodes",
      "title": "Deployment",
      "options": [
        {
          "id": "single",
          "label": "Single node · PD co-located"
        }
      ]
    }
  ],
  "modelNames": {
    "8b|w8a8": "vllm-ascend/Qwen3-8B-w8a8",
    "30b-a3b|w8a8": "Eco-Tech/Qwen3-30B-A3B-w8a8",
    "32b|bf16": "Qwen/Qwen3-32B",
    "32b|w8a8": "vllm-ascend/Qwen3-32B-W8A8",
    "235b-a22b|bf16": "Qwen/Qwen3-235B-A22B",
    "235b-a22b|w8a8": "vllm-ascend/Qwen3-235B-A22B-W8A8"
  },
  "runModes": [
    "python"
  ],
  "showPlaygroundLink": false,
  "latencyPercentile": "Mean",
  "benchmarkDeviceLabel": "NPU device",
  "accuracyLabels": [
    [
      "gpqa_diamond_pct",
      "GPQA Diamond (198 questions)",
      "%"
    ],
    [
      "aime25_pct",
      "AIME 2025 (30 questions)",
      "%"
    ],
    [
      "gsm8k_pct",
      "GSM8K (1314 questions, 5-shot)",
      "%"
    ]
  ],
  "benchmarkCommands": {
    "speed": "python -m sglang.benchmark.serving \\\n  --backend sglang \\\n  --host {{CURL_HOST}} --port {{CURL_PORT}} \\\n  --model \"{{MODEL_PATH}}\" \\\n  --dataset-name {{DATASET}} \\\n  --random-input-len {{ISL}} --random-output-len {{OSL}} \\\n  --max-concurrency {{MAX_CONCURRENCY}} --num-prompts {{NUM_PROMPTS}} \\\n  --random-range-ratio 1 --seed 1 \\\n  --flush-cache --output-details \\\n  --output-file qwen3-ascend-results.jsonl"
  },
  "placeholders": {
    "MODEL_PATH": {
      "target": "command",
      "label": "Local main checkpoint",
      "default": "/models/main-checkpoint"
    },
    "DRAFT_MODEL_PATH": {
      "target": "command",
      "label": "Matching EAGLE3 checkpoint",
      "default": "/models/eagle3-checkpoint"
    },
    "HCCL_IFNAME": {
      "target": "command",
      "label": "HCCL network interface",
      "default": "eth0"
    },
    "GLOO_IFNAME": {
      "target": "command",
      "label": "Gloo network interface",
      "default": "eth0"
    },
    "HOST_IP": {
      "target": "command",
      "label": "Bind host",
      "default": "127.0.0.1"
    },
    "PORT": {
      "target": "command",
      "label": "Bind port",
      "default": "6688"
    },
    "CURL_HOST": {
      "target": "curl",
      "label": "Service host",
      "default": "127.0.0.1"
    },
    "CURL_PORT": {
      "target": "curl",
      "label": "Service port",
      "default": "6688"
    }
  },
  "curl": "curl --fail http://{{CURL_HOST}}:{{CURL_PORT}}/generate -H 'Content-Type: application/json' -d '{\"text\":\"The capital of France is\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":64}}'",
  "github": {
    "owner": "sgl-project",
    "repo": "sglang",
    "cookbookModel": "Qwen/Qwen3"
  },
  "cells": [
    {
      "match": {
        "hw": "a3",
        "variant": "8b",
        "quant": "w8a8",
        "strategy": "low-latency",
        "nodes": "single"
      },
      "verified": true,
      "verificationStatus": "verified",
      "warn": "Requires 1 A3 physical card on one node; uses 2 logical NPU devices (TP=2). See [Ascend reference workloads](#ascend-reference-workloads). Validated with the checkpoint pair and runtime recorded below.",
      "env": [
        "GLOO_SOCKET_IFNAME={{GLOO_IFNAME}}",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "HCCL_SOCKET_IFNAME={{HCCL_IFNAME}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1"
      ],
      "flags": [
        "--model-path \"{{MODEL_PATH}}\"",
        "--trust-remote-code",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--max-running-requests 1",
        "--max-prefill-tokens 16384",
        "--disable-radix-cache",
        "--chunked-prefill-size -1",
        "--tp-size 2",
        "--mem-fraction-static 0.894",
        "--cuda-graph-bs-decode 1",
        "--dtype bfloat16",
        "--speculative-draft-model-quantization unquant",
        "--speculative-algorithm EAGLE3",
        "--speculative-draft-model-path \"{{DRAFT_MODEL_PATH}}\"",
        "--speculative-num-steps 4",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 5",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "a3",
        "variant": "8b",
        "quant": "w8a8",
        "strategy": "high-throughput",
        "nodes": "single"
      },
      "verified": false,
      "verificationStatus": "unverified",
      "warn": "Requires 1 A3 physical card on one node; uses 1 logical NPU device (TP=1). Performance is measured. The recorded GPQA score includes one context-limit rejection; a separate full evaluation with the corrected output budget remains pending on A3. See [Ascend reference workloads](#ascend-reference-workloads).",
      "env": [
        "GLOO_SOCKET_IFNAME={{GLOO_IFNAME}}",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "HCCL_SOCKET_IFNAME={{HCCL_IFNAME}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1"
      ],
      "flags": [
        "--model-path \"{{MODEL_PATH}}\"",
        "--trust-remote-code",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--max-running-requests 70",
        "--max-prefill-tokens 16384",
        "--disable-radix-cache",
        "--chunked-prefill-size 16384",
        "--tp-size 1",
        "--mem-fraction-static 0.85",
        "--cuda-graph-bs-decode 8 12 24 36 48 51 55 60 63 64 66 68 70",
        "--dtype bfloat16",
        "--speculative-draft-model-quantization unquant",
        "--speculative-algorithm EAGLE3",
        "--speculative-draft-model-path \"{{DRAFT_MODEL_PATH}}\"",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen",
        "--prefill-delayer-max-delay-passes 50",
        "--enable-prefill-delayer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "a3",
        "variant": "30b-a3b",
        "quant": "w8a8",
        "strategy": "low-latency",
        "nodes": "single"
      },
      "verified": true,
      "verificationStatus": "verified",
      "warn": "Requires 1 A3 physical card on one node; uses 2 logical NPU devices (TP=2). See [Ascend reference workloads](#ascend-reference-workloads). Validated with the checkpoint pair and runtime recorded below.",
      "env": [
        "ASCEND_LAUNCH_BLOCKING=0",
        "DEEPEP_HCCL_BUFFSIZE=400",
        "GLOO_SOCKET_IFNAME={{GLOO_IFNAME}}",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "HCCL_SOCKET_IFNAME={{HCCL_IFNAME}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1"
      ],
      "flags": [
        "--model-path \"{{MODEL_PATH}}\"",
        "--trust-remote-code",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--max-running-requests 162",
        "--disable-radix-cache",
        "--speculative-draft-model-quantization unquant",
        "--chunked-prefill-size -1",
        "--max-prefill-tokens 35000",
        "--speculative-algorithm EAGLE3",
        "--speculative-draft-model-path \"{{DRAFT_MODEL_PATH}}\"",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--tp-size 2",
        "--mem-fraction-static 0.87",
        "--cuda-graph-bs-decode 1 5 15 40 70 100 120 130 140 146 150 154 156 158 160 162",
        "--dtype bfloat16",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen",
        "--prefill-delayer-max-delay-passes 200",
        "--enable-prefill-delayer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "a3",
        "variant": "30b-a3b",
        "quant": "w8a8",
        "strategy": "high-throughput",
        "nodes": "single"
      },
      "verified": true,
      "verificationStatus": "verified",
      "warn": "Requires 1 A3 physical card on one node; uses 2 logical NPU devices (TP=2). See [Ascend reference workloads](#ascend-reference-workloads). Validated with the checkpoint pair and runtime recorded below.",
      "env": [
        "ASCEND_LAUNCH_BLOCKING=0",
        "DEEPEP_HCCL_BUFFSIZE=400",
        "GLOO_SOCKET_IFNAME={{GLOO_IFNAME}}",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "HCCL_SOCKET_IFNAME={{HCCL_IFNAME}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1"
      ],
      "flags": [
        "--model-path \"{{MODEL_PATH}}\"",
        "--trust-remote-code",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--max-running-requests 162",
        "--disable-radix-cache",
        "--speculative-draft-model-quantization unquant",
        "--chunked-prefill-size -1",
        "--max-prefill-tokens 35000",
        "--speculative-algorithm EAGLE3",
        "--speculative-draft-model-path \"{{DRAFT_MODEL_PATH}}\"",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--tp-size 2",
        "--mem-fraction-static 0.87",
        "--cuda-graph-bs-decode 1 5 15 40 70 100 120 130 140 146 150 154 156 158 160 162",
        "--dtype bfloat16",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen",
        "--prefill-delayer-max-delay-passes 200",
        "--enable-prefill-delayer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "a2",
        "variant": "32b",
        "quant": "bf16",
        "strategy": "low-latency",
        "nodes": "single"
      },
      "verified": true,
      "verificationStatus": "verified",
      "warn": "Requires 4 A2 cards with 64 GB each on one node (TP=4). Allows eight concurrent requests, uses chunked prefill and decode graphs for batch sizes 1, 2, 4, and 8, without EAGLE3. Performance uses client concurrency 1; GSM8K uses 8. Validated with the checkpoint and runtime recorded below; the A3 low-latency target does not apply.",
      "env": [
        "GLOO_SOCKET_IFNAME={{GLOO_IFNAME}}",
        "HCCL_SOCKET_IFNAME={{HCCL_IFNAME}}",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
      ],
      "flags": [
        "--model-path \"{{MODEL_PATH}}\"",
        "--trust-remote-code",
        "--attention-backend ascend",
        "--device npu",
        "--dtype bfloat16",
        "--tp-size 4",
        "--context-length 32768",
        "--max-running-requests 8",
        "--chunked-prefill-size 4096",
        "--mem-fraction-static 0.85",
        "--disable-radix-cache",
        "--cuda-graph-bs-decode 1 2 4 8",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "a2",
        "variant": "32b",
        "quant": "w8a8",
        "strategy": "high-throughput",
        "nodes": "single"
      },
      "verified": true,
      "verificationStatus": "verified",
      "warn": "Requires 4 A2 cards with 64 GB each on one node (4 logical NPU devices, TP=4). Validated on 910B3 with startup, temperature-0/1 inference, 400/400 performance requests, and a full 198-question GPQA run using a 38000-token output budget. The A2 reference's two-card heading conflicts with its TP4 command; use four devices. See [Ascend reference workloads](#ascend-reference-workloads).",
      "env": [
        "GLOO_SOCKET_IFNAME={{GLOO_IFNAME}}",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "HCCL_SOCKET_IFNAME={{HCCL_IFNAME}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1"
      ],
      "flags": [
        "--model-path \"{{MODEL_PATH}}\"",
        "--trust-remote-code",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--max-running-requests 101",
        "--disable-radix-cache",
        "--speculative-draft-model-quantization unquant",
        "--chunked-prefill-size -1",
        "--max-prefill-tokens 35000",
        "--speculative-algorithm EAGLE3",
        "--speculative-draft-model-path \"{{DRAFT_MODEL_PATH}}\"",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--tp-size 4",
        "--mem-fraction-static 0.845",
        "--cuda-graph-bs-decode 16 32 64 72 88 90 92 94 96 97 98 99 100 101",
        "--dtype bfloat16",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen",
        "--prefill-delayer-max-delay-passes 100",
        "--enable-prefill-delayer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      "match": {
        "hw": "a2",
        "variant": "235b-a22b",
        "quant": "w8a8",
        "strategy": "high-throughput",
        "nodes": "single"
      },
      "verified": false,
      "verificationStatus": "unverified",
      "warn": "A2 validation candidate: 8 cards with 64 GB each on one node (TP=8). Uses the tutorial's recommended W8A8 device count, pure tensor parallelism, eight concurrent requests, and no EAGLE3 or graph capture. This is an untuned starting point; runtime, accuracy, and performance validation are pending. BF16 needs separate two-node resource validation and is not offered by this single-node selector.",
      "env": [
        "GLOO_SOCKET_IFNAME={{GLOO_IFNAME}}",
        "HCCL_SOCKET_IFNAME={{HCCL_IFNAME}}",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
      ],
      "flags": [
        "--model-path \"{{MODEL_PATH}}\"",
        "--trust-remote-code",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--tp-size 8",
        "--context-length 8192",
        "--max-running-requests 8",
        "--chunked-prefill-size 4096",
        "--mem-fraction-static 0.90",
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    }
  ],
  "hardware": [
    {
      "id": "a2",
      "label": "A2 Series",
      "vram": "64GB/device",
      "vendor": "npu",
      "npuDevices": 8
    }
  ]
};
