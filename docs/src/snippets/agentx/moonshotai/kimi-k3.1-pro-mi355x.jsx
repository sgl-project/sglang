// Hand-maintained: MI355X AgentX launches of Kimi-K3.1-Pro (SGLang only), rendered with the
// GB300 snapshot in one explorer. Same encoding as the generated agentx/ files (see _agentx.jsx).
// The commands are SemiAnalysisAI/InferenceX#4928's MI355X recipes: single node c1-c56 and the 1P1D
// at c128 and c144, without settings the image or SGLang applies by default.
export const agentxMi355x = {
 "model": "Kimi-K3.1-Pro",
 "generatedOn": "2026-11-10",
 "source": {
  "repo": "https://github.com/SemiAnalysisAI/InferenceX",
  "text": {
   "verified": "✓ Measured on MI355X (Kimi K3.1 Pro AgentX, 3600 s)",
   "derived": "Derived — not measured",
   "submitted": "as measured",
   "notRun": "this combination was not measured on MI355X",
   "dateLabel": "measured",
   "acceptance": "This point was measured with the DFlash acceptance length fixed at {al} (the draft's expected acceptance, via SGLANG_SIMULATE_ACC_LEN), which is not part of these commands. Real speed depends on your workload's acceptance."
  }
 },
 "hardware": [
  {
   "id": "mi355x",
   "label": "MI355X"
  }
 ],
 "checkpoints": [
  {
   "id": "moonshotai/Kimi-K3.1-Pro"
  }
 ],
 "envs": [],
 "kvMooncake": null,
 "files": {},
 "cells": [
  {
   "id": "kimik3.1-fp4-mi355x-sglang-agentic-dflash#0",
   "hw": "mi355x",
   "ckpt": "moonshotai/Kimi-K3.1-Pro",
   "label": "Agg TP8 · monokernel",
   "detail": "DFlash 8",
   "gpus": 8,
   "configKey": "kimik3.1-fp4-mi355x-sglang-agentic-dflash",
   "image": "lmsysorg/sglang:dev-kimi-k31-pro-mi35x@sha256:5901db1401ad7b8bc62ac67175f6f013b1b868e2a2dab063ee8848e7290fa9a7",
   "imageStatus": "ok",
   "imageCommit": null,
   "submitted": "sglang",
   "setup": [],
   "disclosures": ["synthetic-acceptance"],
   "kv": {
    "available": ["none"],
    "unavailable": {
     "hicache": "The monokernel does not accept KV offload.",
     "mooncake": "HiCache L3 is not part of the MI355X recipes."
    }
   },
   "prov": {
    "sha": "d96c510d0f1120180abb98704415254abbd092fd",
    "recipe": "inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/kimik3.1/sglang/mi355x-fp4-mtp/agentic.yaml",
    "pr": "4928",
    "status": "verified",
    "date": "2026-11-09"
   },
   "points": [
    {
     "concs": [1],
     "item": 0,
     "override": "override_tp8_c1",
     "goldenAL": "3.68",
     "kv": "none"
    },
    {
     "concs": [4],
     "item": 1,
     "override": "override_tp8_c4",
     "goldenAL": "3.68",
     "kv": "none",
     "patch": {
      "sglang": [
       {
        "args": {
         "--max-running-requests": ["4"]
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
        "AITER_FLYDSL_FORCE": "1",
        "AITER_SITUV2_A8W4": "1",
        "HSA_USERPTR_FOR_PAGED_MEM": "0",
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.95",
        "SGLANG_AITER_K3_OPT": "1",
        "SGLANG_AMD_MONO_K31_WIDE": "1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT": "1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM": "1",
        "SGLANG_OPT_DFLASH_RING_LOCS_FUSED": "1",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "6",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
        "SGLANG_USE_AITER": "1"
       },
       "args": [
        ["--model-path", "moonshotai/Kimi-K3.1-Pro"],
        ["--served-model-name", "moonshotai/Kimi-K3.1-Pro"],
        ["--host", "0.0.0.0"],
        ["--port", "30000"],
        ["--chunked-prefill-size", "16384"],
        ["--cuda-graph-backend-prefill", "breakable"],
        ["--cuda-graph-bs-prefill", "256", "512", "768", "1024", "1536", "2048", "3072", "4096"],
        ["--cuda-graph-max-bs-decode", "4"],
        ["--enable-amd-monokernel"],
        ["--enable-linear-replayssm-spec"],
        ["--enable-metrics"],
        ["--kv-cache-dtype", "fp8_e4m3"],
        ["--mamba-radix-cache-strategy", "extra_buffer_lazy"],
        ["--max-mamba-cache-size", "150"],
        ["--max-running-requests", "1"],
        ["--mem-fraction-static", "0.85"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--prefill-decode-interval", "16"],
        ["--random-seed", "42"],
        ["--reasoning-parser", "kimi_k3"],
        ["--schedule-policy", "lpm"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-kv-cache-dtype", "fp8_e4m3"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--speculative-draft-model-revision", "d0d8bc2e3c6d0a195d2e00a49017e71ca8d881f1"],
        ["--speculative-num-draft-tokens", "8"],
        ["--speculative-use-block-verification"],
        ["--tool-call-parser", "kimi_k3"],
        ["--tp-size", "8"],
        ["--trust-remote-code"],
        ["--weight-loader-prefetch-checkpoints"]
       ]
      }
     ]
    }
   }
  },
  {
   "id": "kimik3.1-fp4-mi355x-sglang-agentic-dflash#1",
   "hw": "mi355x",
   "ckpt": "moonshotai/Kimi-K3.1-Pro",
   "label": "Agg TP8·DCP8",
   "detail": "DFlash 6 / 5",
   "gpus": 8,
   "configKey": "kimik3.1-fp4-mi355x-sglang-agentic-dflash",
   "image": "lmsysorg/sglang:dev-kimi-k31-pro-mi35x@sha256:5901db1401ad7b8bc62ac67175f6f013b1b868e2a2dab063ee8848e7290fa9a7",
   "imageStatus": "ok",
   "imageCommit": null,
   "submitted": "sglang",
   "setup": [],
   "disclosures": ["synthetic-acceptance"],
   "kv": {
    "available": ["none", "hicache"],
    "unavailable": {
     "mooncake": "HiCache L3 is not part of the MI355X recipes."
    },
    "hicache": {
     "args": [
      ["--enable-hierarchical-cache"],
      ["--hicache-ratio", "2.5"],
      ["--hicache-write-policy", "write_back"]
     ],
     "donor": "kimik3.1-fp4-mi355x-sglang-agentic-dflash",
     "donorHardware": "mi355x"
    }
   },
   "prov": {
    "sha": "d96c510d0f1120180abb98704415254abbd092fd",
    "recipe": "inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/kimik3.1/sglang/mi355x-fp4-mtp/agentic.yaml",
    "pr": "4928",
    "status": "verified",
    "date": "2026-11-09"
   },
   "points": [
    {
     "concs": [8],
     "item": 0,
     "override": "override_tp8_dcp8_c8",
     "goldenAL": "3.43",
     "kv": "hicache"
    },
    {
     "concs": [16],
     "item": 1,
     "override": "override_tp8_dcp8_c16",
     "goldenAL": "3.25",
     "kv": "hicache",
     "patch": {
      "sglang": [
       {
        "args": {
         "--cuda-graph-max-bs-decode": ["8"],
         "--max-running-requests": ["8"],
         "--speculative-num-draft-tokens": ["5"]
        }
       }
      ]
     }
    },
    {
     "concs": [24],
     "item": 2,
     "override": "override_tp8_dcp8_c24",
     "goldenAL": "3.25",
     "kv": "hicache",
     "patch": {
      "sglang": [
       {
        "args": {
         "--cuda-graph-max-bs-decode": ["13"],
         "--max-running-requests": ["13"],
         "--speculative-num-draft-tokens": ["5"]
        }
       }
      ]
     }
    },
    {
     "concs": [32],
     "item": 3,
     "override": "override_tp8_dcp8_c32",
     "goldenAL": "3.25",
     "kv": "hicache",
     "patch": {
      "sglang": [
       {
        "args": {
         "--cuda-graph-max-bs-decode": ["20"],
         "--max-running-requests": ["20"],
         "--speculative-num-draft-tokens": ["5"]
        }
       }
      ]
     }
    },
    {
     "concs": [48],
     "item": 4,
     "override": "override_tp8_dcp8_c48",
     "goldenAL": "3.25",
     "kv": "hicache",
     "patch": {
      "sglang": [
       {
        "args": {
         "--chunked-prefill-size": ["32768"],
         "--cuda-graph-max-bs-decode": ["48"],
         "--max-running-requests": ["48"],
         "--speculative-num-draft-tokens": ["5"]
        }
       }
      ]
     }
    },
    {
     "concs": [56],
     "item": 5,
     "override": "override_tp8_dcp8_c56",
     "goldenAL": "3.25",
     "kv": "hicache",
     "patch": {
      "sglang": [
       {
        "args": {
         "--chunked-prefill-size": ["32768"],
         "--cuda-graph-max-bs-decode": ["56"],
         "--max-running-requests": ["56"],
         "--speculative-num-draft-tokens": ["5"]
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
        "AITER_FLYDSL_FORCE": "1",
        "AITER_SITUV2_A8W4": "1",
        "HSA_USERPTR_FOR_PAGED_MEM": "0",
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.95",
        "RUSTUP_TOOLCHAIN": "stable",
        "SGLANG_AITER_K3_OPT": "1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT": "1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM": "1",
        "SGLANG_HICACHE_MAMBA_HOST_RATIO": "10",
        "SGLANG_OPT_DFLASH_RING_LOCS_FUSED": "1",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "10",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
        "SGLANG_USE_AITER": "1"
       },
       "args": [
        ["--model-path", "moonshotai/Kimi-K3.1-Pro"],
        ["--served-model-name", "moonshotai/Kimi-K3.1-Pro"],
        ["--host", "0.0.0.0"],
        ["--port", "30000"],
        ["--chunked-prefill-size", "16384"],
        ["--cuda-graph-backend-prefill", "breakable"],
        ["--cuda-graph-bs-prefill", "256", "512", "768", "1024", "1536", "2048", "3072", "4096"],
        ["--cuda-graph-max-bs-decode", "6"],
        ["--dcp-comm-backend", "a2a"],
        ["--dcp-size", "8"],
        ["--enable-hierarchical-cache"],
        ["--enable-linear-replayssm-spec"],
        ["--enable-metrics"],
        ["--hicache-ratio", "2.5"],
        ["--hicache-write-policy", "write_back"],
        ["--kv-cache-dtype", "fp8_e4m3"],
        ["--mamba-full-memory-ratio", "0.65"],
        ["--mamba-radix-cache-strategy", "extra_buffer_lazy"],
        ["--max-running-requests", "6"],
        ["--mem-fraction-static", "0.85"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--prefill-decode-interval", "32"],
        ["--random-seed", "42"],
        ["--reasoning-parser", "kimi_k3"],
        ["--schedule-policy", "lpm"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-kv-cache-dtype", "fp8_e4m3"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--speculative-draft-model-revision", "d0d8bc2e3c6d0a195d2e00a49017e71ca8d881f1"],
        ["--speculative-num-draft-tokens", "6"],
        ["--speculative-use-block-verification"],
        ["--tool-call-parser", "kimi_k3"],
        ["--tp-size", "8"],
        ["--trust-remote-code"],
        ["--weight-loader-prefetch-checkpoints"]
       ]
      }
     ]
    }
   }
  },
  {
   "id": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-disagg#0",
   "hw": "mi355x",
   "ckpt": "moonshotai/Kimi-K3.1-Pro",
   "label": "Disagg 1P1D · prefill TP8·DCP8 · decode TP8·DCP8",
   "detail": "DFlash 5",
   "gpus": 16,
   "configKey": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-disagg",
   "image": "lmsysorg/sglang:dev-kimi-k31-pro-mi35x@sha256:5901db1401ad7b8bc62ac67175f6f013b1b868e2a2dab063ee8848e7290fa9a7",
   "imageStatus": "ok",
   "imageCommit": null,
   "submitted": "sglang",
   "setup": [
    {
     "script": "-v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro",
     "reason": "docker run mounts on Ionic (AMD Pollara) NIC hosts. The image has no Ionic verbs provider; without the host's, Mooncake moves KV over TCP."
    }
   ],
   "disclosures": ["synthetic-acceptance"],
   "kv": {
    "available": ["none", "hicache"],
    "unavailable": {
     "mooncake": "HiCache L3 is not part of the MI355X recipes."
    },
    "hicache": {
     "args": [
      ["--enable-hierarchical-cache"],
      ["--hicache-ratio", "4.0"],
      ["--hicache-write-policy", "write_back"]
     ],
     "donor": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-disagg",
     "donorHardware": "mi355x"
    }
   },
   "prov": {
    "sha": "d96c510d0f1120180abb98704415254abbd092fd",
    "recipe": "inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/kimik3.1/sglang/mi355x-fp4/agentx/disagg-variants.yaml",
    "pr": "4928",
    "status": "verified",
    "date": "2026-11-09"
   },
   "points": [
    {
     "concs": [128],
     "item": 0,
     "override": "override_1p1d_c128",
     "goldenAL": "3.25",
     "kv": "hicache"
    },
    {
     "concs": [144],
     "item": 1,
     "override": "override_1p1d_c144",
     "goldenAL": "3.25",
     "kv": "hicache",
     "patch": {
      "sglang": [
       null,
       {
        "args": {
         "--cuda-graph-max-bs-decode": ["160"],
         "--max-running-requests": ["160"]
        }
       },
       null
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
       "role": "prefill",
       "workers": 1,
       "nodes_per_worker": 1,
       "cmd": "python3 -m sglang.launch_server",
       "env": {
        "AITER_FLYDSL_FORCE": "1",
        "AITER_SITUV2_A8W4": "1",
        "HSA_USERPTR_FOR_PAGED_MEM": "0",
        "MC_GID_INDEX": "1",
        "MC_MAX_CONCURRENT_REG_MR": "8",
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.95",
        "RUSTUP_TOOLCHAIN": "stable",
        "SGLANG_AITER_K3_OPT": "1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
        "SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION": "0.1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT": "1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM": "1",
        "SGLANG_HICACHE_MAMBA_HOST_RATIO": "5",
        "SGLANG_OPT_DFLASH_RING_LOCS_FUSED": "1",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "10",
        "SGLANG_SPEC_TP_SYNC": "init,dflash-accept-greedy",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "7200",
        "SGLANG_USE_AITER": "1"
       },
       "args": [
        ["--model-path", "moonshotai/Kimi-K3.1-Pro"],
        ["--served-model-name", "moonshotai/Kimi-K3.1-Pro"],
        ["--host", "0.0.0.0"],
        ["--port", "30000"],
        ["--disaggregation-mode", "prefill"],
        ["--disaggregation-bootstrap-port", "8998"],
        ["--chunked-prefill-size", "16384"],
        ["--cuda-graph-backend-prefill", "breakable"],
        ["--cuda-graph-bs-prefill", "256", "512", "768", "1024", "1536", "2048", "3072", "4096"],
        ["--dcp-comm-backend", "a2a"],
        ["--dcp-size", "8"],
        ["--disaggregation-decode-allocation-policy", "early"],
        ["--disaggregation-transfer-backend", "mooncake"],
        ["--enable-dsa-fp4-indexer"],
        ["--enable-hierarchical-cache"],
        ["--enable-metrics"],
        ["--hicache-ratio", "4.0"],
        ["--hicache-write-policy", "write_back"],
        ["--kv-cache-dtype", "fp8_e4m3"],
        ["--mamba-full-memory-ratio", "0.65"],
        ["--mamba-max-states-per-path", "1"],
        ["--mamba-radix-cache-strategy", "extra_buffer"],
        ["--mamba-track-interval", "64"],
        ["--max-prefill-tokens", "16384"],
        ["--mem-fraction-static", "0.85"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--random-seed", "42"],
        ["--reasoning-parser", "kimi_k3"],
        ["--schedule-policy", "shortest-prefill-first"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-kv-cache-dtype", "fp8_e4m3"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--speculative-draft-model-revision", "d0d8bc2e3c6d0a195d2e00a49017e71ca8d881f1"],
        ["--speculative-num-draft-tokens", "5"],
        ["--speculative-use-block-verification"],
        ["--tool-call-parser", "kimi_k3"],
        ["--tp-size", "8"],
        ["--trust-remote-code"],
        ["--weight-loader-prefetch-checkpoints"]
       ]
      },
      {
       "kind": "worker",
       "role": "decode",
       "workers": 1,
       "nodes_per_worker": 1,
       "cmd": "python3 -m sglang.launch_server",
       "env": {
        "AITER_FLYDSL_FORCE": "1",
        "AITER_SITUV2_A8W4": "1",
        "HSA_USERPTR_FOR_PAGED_MEM": "0",
        "MC_GID_INDEX": "1",
        "MC_MAX_CONCURRENT_REG_MR": "8",
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.99",
        "RUSTUP_TOOLCHAIN": "stable",
        "SGLANG_AITER_K3_OPT": "1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT": "1",
        "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM": "1",
        "SGLANG_OPT_DFLASH_RING_LOCS_FUSED": "1",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "10",
        "SGLANG_SPEC_TP_SYNC": "init,dflash-accept-greedy",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "7200",
        "SGLANG_USE_AITER": "1"
       },
       "args": [
        ["--model-path", "moonshotai/Kimi-K3.1-Pro"],
        ["--served-model-name", "moonshotai/Kimi-K3.1-Pro"],
        ["--host", "0.0.0.0"],
        ["--port", "30000"],
        ["--disaggregation-mode", "decode"],
        ["--cuda-graph-max-bs-decode", "144"],
        ["--dcp-comm-backend", "a2a"],
        ["--dcp-size", "8"],
        ["--disaggregation-decode-enable-radix-cache"],
        ["--disaggregation-decode-extra-slots", "16"],
        ["--disaggregation-transfer-backend", "mooncake"],
        ["--enable-dsa-fp4-indexer"],
        ["--enable-linear-replayssm-spec"],
        ["--enable-metrics"],
        ["--kv-cache-dtype", "fp8_e4m3"],
        ["--mamba-radix-cache-strategy", "extra_buffer"],
        ["--max-running-requests", "144"],
        ["--mem-fraction-static", "0.92"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--random-seed", "42"],
        ["--reasoning-parser", "kimi_k3"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-kv-cache-dtype", "fp8_e4m3"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--speculative-draft-model-revision", "d0d8bc2e3c6d0a195d2e00a49017e71ca8d881f1"],
        ["--speculative-num-draft-tokens", "5"],
        ["--speculative-use-block-verification"],
        ["--tool-call-parser", "kimi_k3"],
        ["--tp-size", "8"],
        ["--trust-remote-code"],
        ["--weight-loader-prefetch-checkpoints"]
       ]
      },
      {
       "kind": "sglang-router",
       "cmd": "python3 -m sglang_router.launch_router",
       "env": {},
       "args": [
        ["--pd-disaggregation"],
        ["--prefill", "http://$PREFILL_HEAD_IP:30000", "8998"],
        ["--decode", "http://$DECODE_HEAD_IP:30000"],
        ["--host", "0.0.0.0"],
        ["--port", "8000"],
        ["--request-timeout-secs", "3600"]
       ]
      }
     ]
    }
   }
  }
 ]
};
