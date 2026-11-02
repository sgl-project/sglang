export const b300Cells = [
  {
    "id": "qwen3.8next-fp4-b300-sglang-agentic-mtp#4683-tp4",
    "hw": "b300",
    "ckpt": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "label": "Agg TP4·EP4",
    "detail": "MTP 3-1-4",
    "gpus": 4,
    "configKey": "qwen3.8next-fp4-b300-sglang-agentic-mtp",
    "image": "lmsysorg/sglang:nightly-dev-cu13-20261028-586355a8@sha256:b4f09a2ccbcfcc262d49600c3cd7f8581a793ea7d104947257b6a95500953b90",
    "imageStatus": "ok",
    "imageCommit": null,
    "submitted": "sglang",
    "setup": [],
    "disclosures": [],
    "kv": {
      "available": ["none"],
      "unavailable": {
        "hicache": "Not qualified for this deployment.",
        "mooncake": "Not qualified for this deployment."
      }
    },
    "prov": {
      "sha": "d06564bc533cc54584ef6a35837336a2b4b84421",
      "recipe": "inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/b300-fp4-mtp/agentic.yaml",
      "run": "41117066222/attempts/1",
      "pr": "4683",
      "status": "verified-pr",
      "date": "2026-10-30"
    },
    "points": [
      {
        "concs": [1],
        "override": "override_tp4_c1",
        "kv": "none"
      },
      {
        "concs": [2],
        "override": "override_tp4_c2",
        "kv": "none",
        "patch": {
          "sglang": [
            {
              "args": {
                "--max-running-requests": ["4"]
              },
              "env": {}
            }
          ]
        }
      }
    ],
    "routers": {
      "sglang": {
        "install": [],
        "files": [],
        "blocks": [
          {
            "kind": "worker",
            "role": "agg",
            "workers": 1,
            "nodes_per_worker": 1,
            "cmd": "python3 -m sglang.launch_server",
            "env": {
              "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
              "SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE": "0",
              "SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING": "1"
            },
            "args": [
              ["--model-path", "nvidia/Qwen3.8-Flash-Next-NVFP4"],
              ["--host", "0.0.0.0"],
              ["--port", "30000"],
              ["--served-model-name", "nvidia/Qwen3.8-Flash-Next-NVFP4"],
              ["--revision", "fc694b54fb0174e0913e6adf86691ef85a4ead47"],
              ["--speculative-draft-model-revision", "fc694b54fb0174e0913e6adf86691ef85a4ead47"],
              ["--trust-remote-code"],
              ["--quantization", "modelopt_mixed"],
              ["--context-length", "262144"],
              ["--tensor-parallel-size", "4"],
              ["--pipeline-parallel-size", "1"],
              ["--data-parallel-size", "1"],
              ["--expert-parallel-size", "4"],
              ["--attention-backend", "trtllm_mha"],
              ["--linear-attn-prefill-backend", "flashinfer"],
              ["--linear-attn-decode-backend", "flashinfer"],
              ["--linear-attn-verify-backend", "triton"],
              ["--mamba-ssm-dtype", "bfloat16"],
              ["--mem-fraction-static", "0.88"],
              ["--max-running-requests", "2"],
              ["--page-size", "64"],
              ["--chunked-prefill-size", "16384"],
              ["--max-prefill-tokens", "16384"],
              ["--moe-runner-backend", "flashinfer_trtllm"],
              ["--moe-a2a-backend", "none"],
              ["--disable-shared-experts-fusion"],
              ["--speculative-algorithm", "NEXTN"],
              ["--speculative-num-steps", "3"],
              ["--speculative-eagle-topk", "1"],
              ["--speculative-num-draft-tokens", "4"],
              ["--speculative-moe-runner-backend", "flashinfer_trtllm"],
              ["--enable-linear-replayssm-spec"],
              ["--cuda-graph-backend-prefill", "breakable"],
              ["--cuda-graph-backend-decode", "full"],
              ["--cuda-graph-max-bs-decode", "96"],
              ["--cuda-graph-bs-decode", "1", "2", "3", "4", "5", "6", "7", "8", "10", "12", "14", "16", "24", "32", "40", "48", "56", "64", "72", "80", "88", "96"],
              ["--kv-cache-dtype", "bfloat16"],
              ["--speculative-draft-kv-cache-dtype", "bfloat16"],
              ["--ple-offload-embedding"],
              ["--ple-offload-backend", "pinned"],
              ["--reasoning-parser", "auto"],
              ["--tokenizer-worker-num", "6"],
              ["--scheduler-recv-interval", "1"],
              ["--stream-interval", "20"],
              ["--watchdog-timeout", "1800"],
              ["--flashinfer-allreduce-fusion-backend", "trtllm"]
            ]
          }
        ]
      }
    }
  },
  {
    "id": "qwen3.8next-fp4-b300-sglang-agentic-mtp#4683-tp1",
    "hw": "b300",
    "ckpt": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "label": "Agg TP1 · GPU KV",
    "detail": "MTP 3-1-4",
    "gpus": 1,
    "configKey": "qwen3.8next-fp4-b300-sglang-agentic-mtp",
    "image": "lmsysorg/sglang:nightly-dev-cu13-20261028-586355a8@sha256:b4f09a2ccbcfcc262d49600c3cd7f8581a793ea7d104947257b6a95500953b90",
    "imageStatus": "ok",
    "imageCommit": null,
    "submitted": "sglang",
    "setup": [],
    "disclosures": [],
    "kv": {
      "available": ["none"],
      "unavailable": {
        "hicache": "Not qualified for this deployment.",
        "mooncake": "Not qualified for this deployment."
      }
    },
    "prov": {
      "sha": "d06564bc533cc54584ef6a35837336a2b4b84421",
      "recipe": "inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/b300-fp4-mtp/agentic.yaml",
      "run": "41117066222/attempts/1",
      "pr": "4683",
      "status": "verified-pr",
      "date": "2026-10-30"
    },
    "points": [
      {
        "concs": [1],
        "override": "override_c1",
        "kv": "none"
      },
      {
        "concs": [2],
        "override": "override_c2",
        "kv": "none",
        "patch": {
          "sglang": [
            {
              "args": {},
              "env": {}
            }
          ]
        }
      },
      {
        "concs": [12],
        "override": "override_c12",
        "kv": "none",
        "patch": {
          "sglang": [
            {
              "args": {
                "--cuda-graph-bs-decode": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14", "16", "24", "32", "40", "48", "56", "64", "72", "80", "88", "96"],
                "--mamba-track-interval": ["262144"],
                "--kv-cache-dtype": ["fp8_e4m3"],
                "--speculative-draft-kv-cache-dtype": ["fp8_e4m3"]
              },
              "env": {
                "SGLANG_Q35MF_MKH_N3B": null,
                "SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH": null,
                "SGLANG_Q35MF_MKH_N3C": null,
                "SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH": null,
                "SGLANG_Q35MF_EDGE_SHM_SYNC": null,
                "SGLANG_Q35MF_BK_VEC_COPY": null,
                "SGLANG_Q35MF_EDGE_PACKED_VERIFY_BCAST": null
              }
            }
          ]
        }
      },
      {
        "concs": [24],
        "override": "override_c24",
        "kv": "none",
        "patch": {
          "sglang": [
            {
              "args": {
                "--cuda-graph-bs-decode": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "40", "48", "56", "64", "72", "80", "88", "96"],
                "--mamba-track-interval": ["262144"],
                "--max-mamba-cache-size": ["960"],
                "--kv-cache-dtype": ["fp8_e4m3"],
                "--speculative-draft-kv-cache-dtype": ["fp8_e4m3"]
              },
              "env": {
                "SGLANG_Q35MF_MKH_N3B": null,
                "SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH": null,
                "SGLANG_Q35MF_MKH_N3C": null,
                "SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH": null,
                "SGLANG_Q35MF_EDGE_SHM_SYNC": null,
                "SGLANG_Q35MF_BK_VEC_COPY": null,
                "SGLANG_Q35MF_EDGE_PACKED_VERIFY_BCAST": null
              }
            }
          ]
        }
      }
    ],
    "routers": {
      "sglang": {
        "install": [],
        "files": [],
        "blocks": [
          {
            "kind": "worker",
            "role": "agg",
            "workers": 1,
            "nodes_per_worker": 1,
            "cmd": "python3 -m sglang.launch_server",
            "env": {
              "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
              "SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE": "0",
              "SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING": "1",
              "SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH": "1",
              "SGLANG_Q35MF_MKH_N3B": "1",
              "SGLANG_Q35MF_MKH_N3C": "1",
              "SGLANG_Q35MF_BK_VEC_COPY": "1",
              "SGLANG_Q35MF_EDGE_SHM_SYNC": "1",
              "SGLANG_Q35MF_EDGE_PACKED_VERIFY_BCAST": "1",
              "SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH": "1"
            },
            "args": [
              ["--model-path", "nvidia/Qwen3.8-Flash-Next-NVFP4"],
              ["--host", "0.0.0.0"],
              ["--port", "30000"],
              ["--served-model-name", "nvidia/Qwen3.8-Flash-Next-NVFP4"],
              ["--revision", "fc694b54fb0174e0913e6adf86691ef85a4ead47"],
              ["--speculative-draft-model-revision", "fc694b54fb0174e0913e6adf86691ef85a4ead47"],
              ["--trust-remote-code"],
              ["--quantization", "modelopt_mixed"],
              ["--context-length", "262144"],
              ["--tensor-parallel-size", "1"],
              ["--pipeline-parallel-size", "1"],
              ["--data-parallel-size", "1"],
              ["--expert-parallel-size", "1"],
              ["--attention-backend", "trtllm_mha"],
              ["--linear-attn-prefill-backend", "flashinfer"],
              ["--linear-attn-decode-backend", "flashinfer"],
              ["--linear-attn-verify-backend", "triton"],
              ["--mamba-ssm-dtype", "bfloat16"],
              ["--mem-fraction-static", "0.88"],
              ["--max-running-requests", "96"],
              ["--page-size", "64"],
              ["--chunked-prefill-size", "16384"],
              ["--max-prefill-tokens", "16384"],
              ["--moe-runner-backend", "flashinfer_trtllm"],
              ["--moe-a2a-backend", "none"],
              ["--disable-shared-experts-fusion"],
              ["--speculative-algorithm", "NEXTN"],
              ["--speculative-num-steps", "3"],
              ["--speculative-eagle-topk", "1"],
              ["--speculative-num-draft-tokens", "4"],
              ["--speculative-moe-runner-backend", "flashinfer_trtllm"],
              ["--enable-linear-replayssm-spec"],
              ["--cuda-graph-backend-prefill", "breakable"],
              ["--cuda-graph-backend-decode", "full"],
              ["--cuda-graph-max-bs-decode", "96"],
              ["--cuda-graph-bs-decode", "1", "2", "3", "4", "5", "6", "7", "8", "10", "12", "14", "16", "24", "32", "40", "48", "56", "64", "72", "80", "88", "96"],
              ["--kv-cache-dtype", "bfloat16"],
              ["--speculative-draft-kv-cache-dtype", "bfloat16"],
              ["--ple-offload-embedding"],
              ["--ple-offload-backend", "pinned"],
              ["--reasoning-parser", "auto"],
              ["--tokenizer-worker-num", "6"],
              ["--scheduler-recv-interval", "1"],
              ["--stream-interval", "20"],
              ["--watchdog-timeout", "1800"]
            ]
          }
        ]
      }
    }
  },
  {
    "id": "qwen3.8next-fp4-b300-sglang-agentic-mtp#4683-tp1-hicache",
    "hw": "b300",
    "ckpt": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "label": "Agg TP1 · HiCache",
    "detail": "MTP 3-1-4",
    "gpus": 1,
    "configKey": "qwen3.8next-fp4-b300-sglang-agentic-mtp",
    "image": "lmsysorg/sglang:nightly-dev-cu13-20261028-586355a8@sha256:b4f09a2ccbcfcc262d49600c3cd7f8581a793ea7d104947257b6a95500953b90",
    "imageStatus": "ok",
    "imageCommit": null,
    "submitted": "sglang",
    "setup": [],
    "disclosures": [],
    "kv": {
      "available": ["hicache"],
      "unavailable": {
        "none": "Not qualified for this deployment.",
        "mooncake": "Not qualified for this deployment."
      }
    },
    "prov": {
      "sha": "d06564bc533cc54584ef6a35837336a2b4b84421",
      "recipe": "inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/b300-fp4-mtp/agentic.yaml",
      "run": "41117066222/attempts/1",
      "pr": "4683",
      "status": "verified-pr",
      "date": "2026-10-30"
    },
    "points": [
      {
        "concs": [48],
        "override": "override_c48",
        "kv": "hicache"
      },
      {
        "concs": [80],
        "override": "override_c80",
        "kv": "hicache",
        "patch": {
          "sglang": [
            {
              "args": {
                "--cuda-graph-bs-decode": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "88", "92", "96", "100", "104", "108", "112", "116", "120", "124", "128"],
                "--max-running-requests": ["128"],
                "--mamba-max-states-per-path": ["2"],
                "--max-mamba-cache-size": ["640"],
                "--cuda-graph-max-bs-decode": ["128"],
                "--strip-thinking-cache": []
              },
              "env": {}
            }
          ]
        }
      }
    ],
    "routers": {
      "sglang": {
        "install": [],
        "files": [],
        "blocks": [
          {
            "kind": "worker",
            "role": "agg",
            "workers": 1,
            "nodes_per_worker": 1,
            "cmd": "python3 -m sglang.launch_server",
            "env": {
              "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
              "SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE": "0",
              "SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING": "1"
            },
            "args": [
              ["--model-path", "nvidia/Qwen3.8-Flash-Next-NVFP4"],
              ["--host", "0.0.0.0"],
              ["--port", "30000"],
              ["--served-model-name", "nvidia/Qwen3.8-Flash-Next-NVFP4"],
              ["--revision", "fc694b54fb0174e0913e6adf86691ef85a4ead47"],
              ["--speculative-draft-model-revision", "fc694b54fb0174e0913e6adf86691ef85a4ead47"],
              ["--trust-remote-code"],
              ["--quantization", "modelopt_mixed"],
              ["--context-length", "262144"],
              ["--tensor-parallel-size", "1"],
              ["--pipeline-parallel-size", "1"],
              ["--data-parallel-size", "1"],
              ["--expert-parallel-size", "1"],
              ["--attention-backend", "trtllm_mha"],
              ["--linear-attn-prefill-backend", "flashinfer"],
              ["--linear-attn-decode-backend", "flashinfer"],
              ["--linear-attn-verify-backend", "triton"],
              ["--mamba-ssm-dtype", "bfloat16"],
              ["--mamba-radix-cache-strategy", "extra_buffer_lazy"],
              ["--mamba-track-interval", "262144"],
              ["--max-mamba-cache-size", "720"],
              ["--mem-fraction-static", "0.88"],
              ["--max-running-requests", "96"],
              ["--page-size", "64"],
              ["--chunked-prefill-size", "16384"],
              ["--max-prefill-tokens", "16384"],
              ["--moe-runner-backend", "flashinfer_trtllm"],
              ["--moe-a2a-backend", "none"],
              ["--disable-shared-experts-fusion"],
              ["--speculative-algorithm", "NEXTN"],
              ["--speculative-num-steps", "3"],
              ["--speculative-eagle-topk", "1"],
              ["--speculative-num-draft-tokens", "4"],
              ["--speculative-moe-runner-backend", "flashinfer_trtllm"],
              ["--enable-linear-replayssm-spec"],
              ["--cuda-graph-backend-prefill", "breakable"],
              ["--cuda-graph-backend-decode", "full"],
              ["--cuda-graph-max-bs-decode", "96"],
              ["--cuda-graph-bs-decode", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "20", "24", "28", "32", "36", "40", "44", "48", "52", "56", "60", "64", "72", "80", "88", "96"],
              ["--kv-cache-dtype", "fp8_e4m3"],
              ["--speculative-draft-kv-cache-dtype", "fp8_e4m3"],
              ["--enable-hierarchical-cache"],
              ["--hicache-size", "298"],
              ["--hicache-io-backend", "kernel"],
              ["--hicache-mem-layout", "page_first"],
              ["--hicache-write-policy", "write_back"],
              ["--ple-offload-embedding"],
              ["--ple-offload-backend", "pinned"],
              ["--reasoning-parser", "auto"],
              ["--tokenizer-worker-num", "1"],
              ["--scheduler-recv-interval", "1"],
              ["--stream-interval", "20"],
              ["--watchdog-timeout", "1800"]
            ]
          }
        ]
      }
    }
  }
];
