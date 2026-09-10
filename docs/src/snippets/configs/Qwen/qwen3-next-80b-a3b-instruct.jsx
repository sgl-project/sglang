// Ascend NPU cookbook configuration for Qwen3-Next-80B-A3B-Instruct.
// The deployment recipe follows the validated 1-card A3 PD mixed reference.

export const config = {
  modelName: "Qwen3-Next-80B-A3B-Instruct",

  // One A3 card contains two dies, so the recipe uses TP=2.
  supportedHardware: ["a3"],
  hardware: [
    { id: "a3", label: "Atlas 800I A3 (1 card)", vram: "64GB/die", vendor: "npu" },
  ],

  matchDims: [
    {
      id: "variant",
      title: "Model variant",
      options: [{ id: "instruct", label: "Instruct" }],
    },
    {
      id: "quant",
      title: "Quantization",
      options: [{ id: "w8a8", label: "W8A8 INT8" }],
    },
    {
      id: "deployment",
      title: "Deployment mode",
      options: [
        { id: "pd-colocated", label: "PD co-located", subtitle: "Prefill + decode" },
      ],
    },
    {
      id: "nodes",
      title: "Nodes",
      options: [{ id: "single", label: "Single node" }],
    },
  ],

  modelNames: {
    "instruct|w8a8": "vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8",
  },

  placeholders: {
    MODEL_PATH: {
      target: "command",
      label: "Main model path",
      default: "/path/to/Qwen3-Next-80B-A3B-Instruct-W8A8",
    },
    DRAFT_MODEL_PATH: {
      target: "command",
      label: "Draft model path",
      default: "/path/to/Qwen3-Next-80B-A3B-Instruct",
    },
    NETWORK_IFACE: {
      target: "command",
      label: "HCCL/Gloo interface",
      default: "<network-interface>",
    },
    HOST_IP: { target: "command", label: "Bind host", default: "127.0.0.1" },
    PORT: { target: "command", label: "Bind port", default: "6688" },
    CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
    CURL_PORT: { target: "curl", label: "Server port", default: "6688" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/generate \\
  -H 'Content-Type: application/json' \\
  -d '{"text":"What is the capital of France?","sampling_params":{"temperature":0,"max_new_tokens":64}}'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.benchmark.serving \\
  --dataset-name random \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --max-concurrency {{MAX_CONCURRENCY}} \\
  --num-prompts {{NUM_PROMPTS}} \\
  --random-input-len {{ISL}} \\
  --random-output-len {{OSL}} \\
  --random-range-ratio 1 \\
  --seed 1`,
    numPromptsByConc: { 1: 1 },
    accuracy: {
      gsm8k_pct:
`python3 -m sglang.test.run_eval \\
  --eval-name gsm8k \\
  --api generate \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --num-examples 100 \\
  --num-shots 5 \\
  --num-threads 1 \\
  --max-tokens 512 \\
  --temperature 0`,
    },
  },

  // This recipe is validated as a host Python launch. Do not present a Docker
  // command until its image, device mapping, and host mounts are verified.
  runModes: ["python"],
  showPlaygroundLink: false,

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K (100, 5-shot)", "%"],
  ],

  github: {
    cookbookModel: "vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8",
  },

  cells: [
    {
      match: {
        hw: "a3",
        variant: "instruct",
        quant: "w8a8",
        deployment: "pd-colocated",
        nodes: "single",
      },
      verified: true,
      env: [
        "ASCEND_USE_FIA=1",
        "DEEPEP_HCCL_BUFFSIZE=2000",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=10",
        "FORCE_DRAFT_MODEL_NON_QUANT=1",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "HCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=400",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "SGLANG_ENABLE_SPEC_V2=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_NPU_USE_MULTI_STREAM=0",
        "SGLANG_SET_CPU_AFFINITY=1",
        "SGLANG_WARMUP_TIMEOUT=3600",
        "STREAMS_PER_DEVICE=32",
        "TASK_QUEUE_ENABLE=1",
        "ZBCCL_BOOTSTRAP_URL=tcp://127.0.0.1:24669",
        "ZBCCL_ENABLE_GRAPH=1",
        "ZBCCL_LOCAL_MEM_SIZE=60416",
        "ZBCCL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_PATH}}",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--page-size 128",
        "--tp-size 2",
        "--watchdog-timeout 9000",
        "--mem-fraction-static 0.85",
        "--disable-radix-cache",
        "--max-prefill-tokens 28672",
        "--context-length 26384",
        "--max-total-tokens 122304",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--speculative-draft-model-quantization unquant",
        "--speculative-draft-model-path {{DRAFT_MODEL_PATH}}",
        "--chunked-prefill-size -1",
        "--max-running-requests 2",
        "--cuda-graph-bs-decode 2",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
