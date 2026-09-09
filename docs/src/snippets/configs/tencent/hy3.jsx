// Hy3 cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.
//
// The shipping Hy3 tokenizer appends a shared suffix to every special token
// (e.g. <tool_calls:TAG>); SGLang's `hunyuan` reasoning/tool-call parsers
// resolve the real token strings from the vocab at runtime (PR #29920), so the
// same recipe serves both the preview (suffix-less) and the shipping (suffixed)
// tokenizer — no per-model hard-coding.
//
// BF16 weights are ~590GB. Single-node TP fits: H200 (141GB, TP8 = 74GB/GPU),
// B200 (180GB, TP4 = 148GB/GPU), B300/GB300 (272GB, TP4), GB200 (192GB, TP4).

export const config = {
  modelName: "Hy3",

  supportedHardware: ["h200", "b200", "b300", "gb200", "gb300", "a3"],

  variants: [
    { id: "default", label: "Default" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
    { id: "fp8",  label: "FP8"  },
  ],
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "balanced",        label: "Balanced"        },
  ],
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  modelNames: {
    "default|bf16": "tencent/Hy3",
    "default|fp8":  "tencent/Hy3-FP8",
  },

  placeholders: {
    DATASET_PATH: { target: "command", label: "Local ShareGPT dataset (Ascend)", default: "/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json" },
    MODEL_PATH: { target: "command", label: "Local Hy3 directory (Ascend)", default: "/models/Hy3" },
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    NODE0_IP:  { target: "command", label: "Head node IP",     default: "<node0-ip>"      },
    NODE_RANK: { target: "command", label: "This node rank",   default: "<node-rank>"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost"       },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"           },
  },

  curl: (sel) => sel.hw === "a3"
    ? `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"What is the capital of France?"}], "max_tokens": 128, "temperature": 0, "top_p": 1.0, "chat_template_kwargs": {"reasoning_effort":"no_think"} }'`
    : `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
      aime26_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run aime26 \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 1 --max-tokens 28672 \\
  --temperature 0.6 --top-p 0.95 --thinking \\
  --out-dir /sgl-workspace/logs`,
    },
    numPromptsByConc: { 1: 32, 16: 32, 64: 128, 256: 512, 1024: 2048 },
  },

  benchmarkCommandsByHardware: {
    a3: {
      speed: `export SGLANG_USE_CPU_ENGINE=1

python3 -m sglang.benchmark.serving \\
  --backend sglang-oai-chat \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model "{{MODEL_PATH}}" --served-model-name {{MODEL_NAME}} --tokenizer "{{MODEL_PATH}}" \\
  --dataset-name sharegpt --dataset-path "{{DATASET_PATH}}" \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --seed 42 --warmup-requests 4 --flush-cache \\
  --extra-request-body '{"chat_template_kwargs":{"reasoning_effort":"no_think"}}' \\
  --output-file hy3-a3-sharegpt.jsonl`,
      accuracy: {
        gsm8k_pct: `evalscope eval \\
  --model {{MODEL_NAME}} \\
  --api-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 --api-key EMPTY \\
  --eval-type openai_api --datasets gsm8k \\
  --dataset-args '{"gsm8k":{"few_shot_num":1,"few_shot_random":false}}' \\
  --generation-config '{"temperature":0,"max_tokens":4096,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \\
  --eval-batch-size 16 --seed 42 --timeout 600`,
      },
      numPromptsByConc: { 16: 128 },
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K (1-shot)", "%"],
    ["aime26_pct", "AIME26",         "%"],
  ],

  multiNodeHints: {
    gb200: [
      "The following env vars may be needed depending on your cluster:",
      "  GLOO_SOCKET_IFNAME=<your-nic>",
      "  NVSHMEM_ENABLE_NIC_PE_MAPPING=1",
      "  NVSHMEM_HCA_LIST=<your-hca-list>",
    ],
  },

  dockerImages: {
    // The dev image bundles the HYV3 model code + the suffix-aware `hunyuan`
    // parser. Switch to `:latest` once a tagged release picks it up.
    h200:  "lmsysorg/sglang:dev",
    b200:  "lmsysorg/sglang:dev",
    b300:  "lmsysorg/sglang:dev",
    gb200: "lmsysorg/sglang:dev",
    gb300: "lmsysorg/sglang:dev",
    a3: "quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16",
  },

  dockerShmSize: "64g",
  dockerHostNetworkWhen: (sel) => sel.hw === "a3",
  dockerMounts: (sel) => sel.hw === "a3"
    ? ['"{{MODEL_PATH}}:{{MODEL_PATH}}:ro"'] : [],
  dockerRunCommand: (sel) => sel.hw === "a3"
    ? "bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && exec sglang serve \"$@\"' --"
    : "sglang serve",

  github: {
    cookbookModel: "tencent/Hy3",
  },

  playgroundFeatures: {

    // ----- Card 1: "Attention Parallelism" -----
    // No CP knob: prefill Context Parallel needs model-side integration in
    // SGLang (DeepSeek-family / Qwen-MoE / Mellum have it) and HYV3ForCausalLM
    // has none — the engine's CP knob would emit --enable-prefill-cp flags
    // that don't work on this model.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null,
          { value: 1, disable: { hw: ["a3"] }, disableReason: "The A3 BF16 recipe uses tensor parallelism across 16 devices." },
          { value: 2, disable: { hw: ["a3"] }, disableReason: "The A3 BF16 recipe uses tensor parallelism across 16 devices." },
          { value: 4, disable: { hw: ["a3"] }, disableReason: "The A3 BF16 recipe uses tensor parallelism across 16 devices." },
          { value: 8, disable: { hw: ["a3"] }, disableReason: "The A3 BF16 recipe uses tensor parallelism across 16 devices." },
          { value: 16, disable: { hw: ["h200", "b200", "b300", "gb200", "gb300"], nodes: ["single"] },
            disableReason: "TP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
        ]},
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null,
            false,
            { value: 1, hide: { hw: ["a3"] } },
            { value: 2, hide: { hw: ["a3"] } },
            { value: 4, hide: { hw: ["a3"] } },
            { value: 8, hide: { hw: ["a3"] } },
            { value: 16, hide: { hw: ["a3"] }, disable: { nodes: ["single"] },
              disableReason: "DP-Attention=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card 2: "MoE Parallelism" -----
    moe: {
      showWhen: (base) => base.hw !== "a3",
      backend: {
        options: [
          { id: null,                label: "Inherited" },
          { id: "deepep",            label: "DeepEP",
            flags: ["--moe-a2a-backend deepep"] },
          { id: "megamoe",           label: "MegaMoE",
            flags: ["--moe-a2a-backend megamoe"],
            requiresHw: ["b200", "b300", "gb200", "gb300"] },
        ],
      },
      ep: { label: "EP", values: [
        null,
        1,
        2,
        4,
        8,
        { value: 16, disable: { nodes: ["single"] },
          disableReason: "EP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
      ]},
    },

    // ----- Card 3: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    // ----- Card 4: "Speculative Decoding" -----
    speculative: {
      showWhen: (base) => base.hw !== "a3",
      options: [
        { id: "current",    label: "Inherited from base" },
        { id: "off",        label: "Off (greedy)" },
        { id: "mtp-314",    label: "EAGLE / MTP 3-1-4",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
        { id: "mtp-112",    label: "EAGLE / MTP 1-1-2",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 1",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 2"] },
        { id: "ngram",      label: "NGRAM",
          flags: ["--speculative-algorithm NGRAM",
                  "--speculative-num-draft-tokens 16",
                  "--speculative-ngram-max-bfs-breadth 10"],
          disable: { dpAttnOn: [true] },
          disableReason: "NGRAM is incompatible with DP-Attention. Turn DP-Attention off in the Attention card above to use NGRAM." },
      ],
    },

    // ----- Card 5: "PD Disaggregation" -----
    pdDisagg: {
      showWhen: (base) => base.hw !== "a3",
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake",
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb200", "gb300"] } },
        { id: "nixl",     label: "NiXL" },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --policy round_robin \\
  --host 0.0.0.0 --port {{ROUTER_PORT}}`,
      },
    },

    // ----- Card 6: "Hierarchical KV Cache" -----
    hicache: {
      excludesHw: ["a3"],
      backends: [
        { id: "null_placeholder", label: "Auto" },
        { id: "file",      label: "File" },
        { id: "mooncake",  label: "Mooncake" },
        { id: "hf3fs",     label: "HF3FS" },
        { id: "nixl",      label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "a3", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      warn: "One Atlas 800I A3 server: 8 cards, 16 devices, 64 GB per device. Set MODEL_PATH to your local Hy3 directory. See [Ascend setup](#ascend-setup).",
      env: [
        "SGLANG_SET_CPU_AFFINITY=1",
        "ASCEND_USE_FIA=1",
        "STREAMS_PER_DEVICE=32",
        "HCCL_BUFFSIZE=3000",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "HCCL_SOCKET_IFNAME=lo",
        "GLOO_SOCKET_IFNAME=lo",
        "SGLANG_ENABLE_SPEC_V2=1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
      ],
      flags: [
        '--model-path "{{MODEL_PATH}}"',
        '--served-model-name "{{MODEL_NAME}}"',
        "--attention-backend ascend",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--device npu",
        "--tp-size 16",
        "--mem-fraction-static 0.84",
        "--dtype bfloat16",
        "--base-gpu-id 0",
        "--prefill-max-requests 40",
        "--max-running-requests 40",
        "--cuda-graph-bs 4 8 16 20 24 28 32 36 40",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // H200 (141GB) — TP=8 for BF16 (~590GB)
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 (180GB) — TP=4 (BF16 590GB → 148GB/GPU, fits with KV headroom)
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 (272GB) — TP=4
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 — TP=4 (inferred from B300, same sm_103 + aarch64)
    // ====================================================================
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB200 (sm_100 + aarch64) — TP=4 (single-node 4×192GB = 768GB fits BF16 590GB)
    // ====================================================================
    {
      match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // FP8 (~300GB) — TP=4 on H200/B200, TP=2 on B300/GB300/GB200
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
