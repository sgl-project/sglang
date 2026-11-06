// Hand-maintained: MI355X AgentX launches of Kimi-K3.1-Pro (SGLang only), rendered with the
// GB300 snapshot in one explorer. Same encoding as the generated agentx/ files (see _agentx.jsx).
// Single node: c16-c56 are the 3600 s AgentX ladder of 2026-11-02; c1-c8 run the decode monokernel, which
// takes no KV offload, and are verified without a published acceptance length. The 1P1D (2x8, c92, 3600 s) is the Deploy panel's verified MI355X prefill / decode pair
// with its default overlays.
export const agentxMi355x = {
 "model": "Kimi-K3.1-Pro",
 "generatedOn": "2026-11-04",
 "source": {
  "repo": "",
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
   "id": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-tp8#0",
   "hw": "mi355x",
   "ckpt": "moonshotai/Kimi-K3.1-Pro",
   "label": "Agg TP8 · monokernel",
   "detail": "DFlash 6",
   "gpus": 8,
   "configKey": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-tp8",
   "image": "<K31_MI35X_IMAGE>",
   "imageStatus": "custom",
   "imageCommit": null,
   "submitted": "sglang",
   "setup": [],
   "disclosures": ["synthetic-acceptance"],
   "kv": {
    "available": ["none"],
    "unavailable": {
     "hicache": "The monokernel does not accept KV offload.",
     "mooncake": "HiCache L3 (Mooncake) has not run on MI355X."
    }
   },
   "prov": {
    "status": "verified"
   },
   "points": [
    {
     "concs": [1],
     "item": 0,
     "kv": "none"
    },
    {
     "concs": [4],
     "item": 1,
     "kv": "none",
     "patch": {
      "sglang": [
       {
        "args": {
         "--max-running-requests": ["4"],
         "--cuda-graph-max-bs-decode": ["4"],
         "--prefill-decode-interval": ["16"]
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
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "6",
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
        ["--cuda-graph-max-bs-decode", "1"],
        ["--dtype", "bfloat16"],
        ["--enable-amd-monokernel"],
        ["--enable-linear-replayssm-spec"],
        ["--enable-metrics"],
        ["--kv-cache-dtype", "fp8_e4m3"],
        ["--mamba-radix-cache-strategy", "extra_buffer_lazy"],
        ["--max-mamba-cache-size", "60"],
        ["--max-running-requests", "1"],
        ["--mem-fraction-static", "0.85"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--reasoning-parser", "kimi_k3"],
        ["--schedule-policy", "lpm"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--speculative-num-draft-tokens", "6"],
        ["--stream-interval", "1"],
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
   "id": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8-mono#0",
   "hw": "mi355x",
   "ckpt": "moonshotai/Kimi-K3.1-Pro",
   "label": "Agg TP8·DCP8 · monokernel",
   "detail": "DFlash 6",
   "gpus": 8,
   "configKey": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8-mono",
   "image": "<K31_MI35X_IMAGE>",
   "imageStatus": "custom",
   "imageCommit": null,
   "submitted": "sglang",
   "setup": [],
   "disclosures": ["synthetic-acceptance"],
   "kv": {
    "available": ["none"],
    "unavailable": {
     "hicache": "The monokernel does not accept KV offload.",
     "mooncake": "HiCache L3 (Mooncake) has not run on MI355X."
    }
   },
   "prov": {
    "status": "verified"
   },
   "points": [
    {
     "concs": [8],
     "item": 0,
     "kv": "none"
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
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "10",
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
        ["--cuda-graph-max-bs-decode", "8"],
        ["--dcp-comm-backend", "a2a"],
        ["--dcp-size", "8"],
        ["--dtype", "bfloat16"],
        ["--enable-amd-monokernel"],
        ["--enable-linear-replayssm-spec"],
        ["--enable-metrics"],
        ["--kv-cache-dtype", "fp8_e4m3"],
        ["--mamba-full-memory-ratio", "0.65"],
        ["--mamba-radix-cache-strategy", "extra_buffer_lazy"],
        ["--max-running-requests", "8"],
        ["--mem-fraction-static", "0.85"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--prefill-decode-interval", "16"],
        ["--reasoning-parser", "kimi_k3"],
        ["--schedule-policy", "lpm"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--speculative-num-draft-tokens", "6"],
        ["--stream-interval", "1"],
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
   "id": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8#0",
   "hw": "mi355x",
   "ckpt": "moonshotai/Kimi-K3.1-Pro",
   "label": "Agg TP8·DCP8",
   "detail": "DFlash 6",
   "gpus": 8,
   "configKey": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8",
   "image": "<K31_MI35X_IMAGE>",
   "imageStatus": "custom",
   "imageCommit": null,
   "submitted": "sglang",
   "setup": [],
   "disclosures": ["synthetic-acceptance"],
   "kv": {
    "available": ["none", "hicache"],
    "unavailable": {
     "mooncake": "HiCache L3 (Mooncake) has not run on MI355X."
    },
    "hicache": {
     "args": [
      ["--enable-hierarchical-cache"],
      ["--hicache-ratio", "2.5"],
      ["--hicache-write-policy", "write_back"]
     ],
     "donor": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8",
     "donorHardware": "mi355x"
    }
   },
   "prov": {
    "status": "verified",
    "date": "2026-11-02"
   },
   "points": [
    {
     "concs": [16],
     "item": 0,
     "goldenAL": "3.399",
     "kv": "hicache"
    },
    {
     "concs": [32],
     "item": 1,
     "goldenAL": "3.399",
     "kv": "hicache",
     "patch": {
      "sglang": [
       {
        "args": {
         "--max-running-requests": ["32"],
         "--cuda-graph-max-bs-decode": ["32"]
        }
       }
      ]
     }
    },
    {
     "concs": [48],
     "item": 2,
     "goldenAL": "3.399",
     "kv": "hicache",
     "patch": {
      "sglang": [
       {
        "args": {
         "--max-running-requests": ["48"],
         "--cuda-graph-max-bs-decode": ["48"]
        }
       }
      ]
     }
    },
    {
     "concs": [56],
     "item": 3,
     "goldenAL": "3.399",
     "kv": "hicache",
     "patch": {
      "sglang": [
       {
        "args": {
         "--max-running-requests": ["64"],
         "--cuda-graph-max-bs-decode": ["64"]
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
        "SGLANG_HICACHE_MAMBA_HOST_RATIO": "12",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "10",
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
        ["--cuda-graph-max-bs-decode", "16"],
        ["--dcp-comm-backend", "a2a"],
        ["--dcp-size", "8"],
        ["--dtype", "bfloat16"],
        ["--enable-hierarchical-cache"],
        ["--enable-linear-replayssm-spec"],
        ["--enable-metrics"],
        ["--hicache-ratio", "2.5"],
        ["--hicache-write-policy", "write_back"],
        ["--kv-cache-dtype", "fp8_e4m3"],
        ["--mamba-full-memory-ratio", "0.65"],
        ["--mamba-radix-cache-strategy", "extra_buffer_lazy"],
        ["--max-running-requests", "16"],
        ["--mem-fraction-static", "0.85"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--prefill-decode-interval", "16"],
        ["--reasoning-parser", "kimi_k3"],
        ["--schedule-policy", "lpm"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--speculative-num-draft-tokens", "6"],
        ["--stream-interval", "1"],
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
   "id": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-disagg-1p1d#0",
   "hw": "mi355x",
   "ckpt": "moonshotai/Kimi-K3.1-Pro",
   "label": "Disagg 1P1D · prefill TP8·DCP8 · decode TP8·DCP8",
   "detail": "DFlash 6",
   "gpus": 16,
   "configKey": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-disagg-1p1d",
   "image": "<K31_MI35X_IMAGE>",
   "imageStatus": "custom",
   "imageCommit": null,
   "submitted": "sglang",
   "setup": [],
   "disclosures": ["synthetic-acceptance"],
   "kv": {
    "available": ["none", "hicache"],
    "unavailable": {
     "mooncake": "HiCache L3 (Mooncake) has not run on MI355X."
    },
    "hicache": {
     "args": [
      ["--enable-hierarchical-cache"],
      ["--hicache-ratio", "2.5"],
      ["--hicache-write-policy", "write_back"]
     ],
     "donor": "kimik3.1-fp4-mi355x-sglang-agentic-dflash-disagg-1p1d",
     "donorHardware": "mi355x"
    }
   },
   "prov": {
    "status": "verified"
   },
   "points": [
    {
     "concs": [92],
     "item": 0,
     "goldenAL": "3.399",
     "kv": "hicache"
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
        "NCCL_IB_DISABLE": "1",
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.95",
        "SGLANG_AITER_K3_OPT": "1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
        "SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION": "0.1",
        "SGLANG_HICACHE_MAMBA_HOST_RATIO": "8",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "10",
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
        ["--cuda-graph-max-bs-decode", "16"],
        ["--dcp-comm-backend", "a2a"],
        ["--dcp-size", "8"],
        ["--disaggregation-decode-allocation-policy", "early"],
        ["--disaggregation-transfer-backend", "mooncake"],
        ["--enable-hierarchical-cache"],
        ["--hicache-ratio", "2.5"],
        ["--hicache-write-policy", "write_back"],
        ["--mamba-full-memory-ratio", "0.8"],
        ["--mamba-radix-cache-strategy", "extra_buffer"],
        ["--max-prefill-tokens", "16384"],
        ["--max-running-requests", "128"],
        ["--mem-fraction-static", "0.85"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--reasoning-parser", "kimi_k3"],
        ["--schedule-policy", "shortest-prefill-first"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-dflash-block-size", "6"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-kv-cache-dtype", "fp8_e4m3"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--tool-call-parser", "kimi_k3"],
        ["--tp-size", "8"],
        ["--trust-remote-code"],
        ["--watchdog-timeout", "3600"],
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
        "MC_GID_INDEX": "1",
        "MC_MAX_CONCURRENT_REG_MR": "8",
        "NCCL_IB_DISABLE": "1",
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.95",
        "SGLANG_AITER_K3_OPT": "1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1",
        "SGLANG_ROCM_RUNTIME_RESERVE_GB": "10",
        "SGLANG_USE_AITER": "1"
       },
       "args": [
        ["--model-path", "moonshotai/Kimi-K3.1-Pro"],
        ["--served-model-name", "moonshotai/Kimi-K3.1-Pro"],
        ["--host", "0.0.0.0"],
        ["--port", "30000"],
        ["--disaggregation-mode", "decode"],
        ["--cuda-graph-max-bs-decode", "96"],
        ["--dcp-comm-backend", "a2a"],
        ["--dcp-size", "8"],
        ["--disaggregation-decode-enable-radix-cache"],
        ["--disaggregation-decode-extra-slots", "16"],
        ["--disaggregation-transfer-backend", "mooncake"],
        ["--enable-linear-replayssm-spec"],
        ["--mamba-radix-cache-strategy", "extra_buffer"],
        ["--max-mamba-cache-size", "160"],
        ["--max-running-requests", "96"],
        ["--mem-fraction-static", "0.91"],
        ["--model-loader-extra-config", "{\"enable_multithread_load\": true}"],
        ["--reasoning-parser", "kimi_k3"],
        ["--speculative-algorithm", "DFLASH"],
        ["--speculative-dflash-block-size", "6"],
        ["--speculative-draft-attention-backend", "triton"],
        ["--speculative-draft-kv-cache-dtype", "fp8_e4m3"],
        ["--speculative-draft-model-path", "RadixArk/Kimi-K3.1-Dflash2-Preview"],
        ["--tool-call-parser", "kimi_k3"],
        ["--tp-size", "8"],
        ["--trust-remote-code"],
        ["--watchdog-timeout", "3600"],
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
