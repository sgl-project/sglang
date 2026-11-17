// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Ascend NPU (Atlas 800I A3) deployments for GLM-5.1. The GPU recipes on the
// same page still come from the legacy generator (glm-51-deployment.jsx); this
// config only carries the A3 cells, so it can fold into a config-driven
// GLM-5.1 page as-is. Recipes are transcribed from the Ascend GLM-5.1 best
// practice (/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/glm_5_1):
//   PD Mixed          -> "W4A8 16P IN3K5 OUT1K5 50ms" (2 A3 nodes, 32 dies).
//   Prefill / Decode  -> "W4A8 1P1D 32P IN16K OUT1K 50ms" (4 A3 nodes, 2 per role).
// Transcribed from the 2026-10-15 revision of that page (CANN 9.1.0 update, #44432).
// Deviations from the best-practice scripts, all forced by current main:
//   - --disable-cuda-graph -> --cuda-graph-backend-{decode,prefill} disabled,
//     --moe-dense-tp -> --moe-dense-tp-size (deprecated / abbreviated spellings).
//   - SGLANG_ENABLE_SPEC_V2 and SGLANG_SPEC_ENABLE_OVERLAP_REFLOW are dropped:
//     main no longer reads either (spec decoding always runs the V2 worker).
//   - --served-model-name glm-5 is dropped so requests use the checkpoint name.

export const config = {
  modelName: "GLM-5.1",

  // One A3 node is 8 cards x 2 dies = 16 devices; every recipe spans two nodes.
  supportedHardware: ["a3"],

  matchDims: [
    {
      id: "quant",
      title: "Quantization",
      options: [
        { id: "w4a8", label: "W4A8", subtitle: "msModelSlim INT8" },
      ],
    },
    {
      // `pdMode` is the id the engine keys its role-specific serving ports on
      // (prefill 30000, decode 30100); the router command in the page uses them.
      id: "pdMode",
      title: "Deployment",
      options: [
        { id: "unified", label: "PD Mixed",       subtitle: "2 x A3 · 32 dies" },
        { id: "prefill", label: "PD Disagg · Prefill", subtitle: "nodes 1-2 of 4" },
        { id: "decode",  label: "PD Disagg · Decode",  subtitle: "nodes 3-4 of 4" },
      ],
    },
  ],

  modelNames: {
    // ModelScope repo — pair with SGLANG_USE_MODELSCOPE=1 (set in every cell).
    w4a8: "Eco-Tech/GLM-5.1-w4a8",
  },

  placeholders: {
    HOST_IP:       { target: "command", label: "Bind host",                  default: "0.0.0.0"           },
    PORT:          { target: "command", label: "Bind port",                  default: "30000"             },
    NODE0_IP:      { target: "command", label: "Head node IP (of this role)", default: "<node0-ip>"         },
    NODE_RANK:     { target: "command", label: "This node rank",             default: "<node-rank>"       },
    NETWORK_IFACE: { target: "command", label: "Cross-node NIC",             default: "<your-nic>"        },
    PREFILL_IP:    { target: "command", label: "Prefill head node IP",       default: "<prefill-node0-ip>" },
    CURL_HOST:     { target: "curl",    label: "Server host",                default: "localhost"         },
    CURL_PORT:     { target: "curl",    label: "Server or router port",      default: "30000"             },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"What is the capital of France?"}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --seed 1 --flush-cache`,
  },

  // Pinned CANN 9.1.0 release image for Atlas 800I A3. v0.5.22 is the first
  // release that carries both features these recipes need: NPU prefill CP
  // (#39427) and --attn-dp-size (#41818). The rolling main-cann9.1.0-a3 tag
  // works too but moves; re-pin here if a verification run uses another build.
  dockerImages: {
    a3: "quay.io/ascend/sglang:v0.5.22-cann9.1.0-a3",
  },

  // This page renders the deployment matrix only; the GPU Playground is not wired up here.
  showPlaygroundLink: false,

  github: {
    cookbookModel: "zai-org/glm-5.1",
  },

  cells: [
    // PD mixed: prefill and decode share one server across two A3 nodes (TP32).
    {
      match: { hw: "a3", quant: "w4a8", pdMode: "unified" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_USE_MODELSCOPE=1",
        "SGLANG_SET_CPU_AFFINITY=1",
        "STREAMS_PER_DEVICE=32",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "HCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "DEEPEP_HCCL_BUFFSIZE=2500",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--device npu",
        "--attention-backend ascend",
        "--quantization modelslim",
        "--tp-size 32",
        "--attn-dp-size 16",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--load-balance-method round_robin",
        "--mem-fraction-static 0.65",
        "--chunked-prefill-size 65536",
        "--max-prefill-tokens 280000",
        "--max-running-requests 256",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        // The W4A8 checkpoint leaves the MTP (nextn) layer in BF16.
        "--speculative-draft-model-quantization unquant",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // PD disaggregation, prefill role: TP32 across the first two of four A3 nodes.
    {
      match: { hw: "a3", quant: "w4a8", pdMode: "prefill" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      // The engine renders `warn` as plain text plus [label](#anchor) links only.
      warn: "This is the prefill role of a 1P1D deployment on four A3 nodes: run it on two nodes, run PD Disagg · Decode on the other two, then front both roles with the router. All four nodes need the same ASCEND_MF_STORE_URL — see [Ascend NPU deployment](#ascend-npu-deployment).",
      env: [
        "SGLANG_USE_MODELSCOPE=1",
        "SGLANG_SET_CPU_AFFINITY=1",
        "STREAMS_PER_DEVICE=32",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        // KV-transfer rendezvous: the same value on all four nodes.
        "ASCEND_MF_STORE_URL=tcp://{{PREFILL_IP}}:24670",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "HCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "TASK_QUEUE_ENABLE=2",
        "DEEPEP_HCCL_BUFFSIZE=1200",
        "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=16",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--device npu",
        "--attention-backend ascend",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--tp-size 32",
        // The prefill role runs context parallelism across all 32 ranks, so it
        // has no attention DP (interleave CP and DP attention are exclusive).
        "--dp-size 1",
        "--enable-prefill-cp",
        "--cp-strategy interleave",
        "--attn-cp-size 32",
        "--moe-dense-tp-size 1",
        "--moe-a2a-backend deepep",
        "--deepep-mode normal",
        "--disable-shared-experts-fusion",
        "--cuda-graph-backend-decode disabled",
        "--cuda-graph-backend-prefill disabled",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend ascend",
        "--disaggregation-bootstrap-port 8998",
        "--mem-fraction-static 0.75",
        "--chunked-prefill-size 524288",
        "--max-prefill-tokens 180000",
        "--max-running-requests 64",
        "--load-balance-method round_robin",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // PD disaggregation, decode role: TP32 / DP32 / EP32 across the other two nodes.
    {
      match: { hw: "a3", quant: "w4a8", pdMode: "decode" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      warn: "This is the decode role of a 1P1D deployment on four A3 nodes: run it on the two nodes not used by PD Disagg · Prefill, start it after the prefill role, then front both roles with the router. All four nodes need the same ASCEND_MF_STORE_URL — see [Ascend NPU deployment](#ascend-npu-deployment).",
      env: [
        "SGLANG_USE_MODELSCOPE=1",
        "SGLANG_SET_CPU_AFFINITY=1",
        "STREAMS_PER_DEVICE=32",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "ASCEND_MF_STORE_URL=tcp://{{PREFILL_IP}}:24670",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "HCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "TASK_QUEUE_ENABLE=0",
        "DEEPEP_HCCL_BUFFSIZE=650",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--device npu",
        "--attention-backend ascend",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--tp-size 32",
        "--attn-dp-size 32",
        "--ep-size 32",
        "--moe-a2a-backend deepep",
        "--deepep-mode low_latency",
        "--disable-shared-experts-fusion",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend ascend",
        "--mem-fraction-static 0.87",
        "--max-running-requests 96",
        "--cuda-graph-bs-decode 1 2 3",
        "--context-length 180000",
        "--tokenizer-worker-num 4",
        "--load-balance-method round_robin",
        "--watchdog-timeout 9000",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--speculative-draft-model-quantization unquant",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
