// Hunyuan 3 (Hy3) cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.
//
// The shipping Hy3 tokenizer appends a shared suffix to every special token
// (e.g. <tool_calls:TAG>); SGLang's `hunyuan` reasoning/tool-call parsers
// resolve the real token strings from the vocab at runtime (PR #29920), so the
// same recipe serves both the preview (suffix-less) and the shipping (suffixed)
// tokenizer — no per-model hard-coding.
//
// BF16 weights are ~552GB, so single-node serving requires 8 GPUs (H200/B200)
// or 4 GPUs on B300/GB300 (272GB). H100/A100 80GB need multi-node TP=16+.

export const config = {
  modelName: "Hy3",

  supportedHardware: ["h200", "b200", "b300", "gb200", "gb300"],

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
    { id: "high-throughput", label: "High-Throughput" },
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
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    NODE0_IP:  { target: "command", label: "Head node IP",     default: "<node0-ip>"      },
    NODE_RANK: { target: "command", label: "This node rank",   default: "<node-rank>"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost"       },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"           },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
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
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
      aime26_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime26 \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 1 --max-tokens 28672 \\
  --temperature 0.6 --top-p 0.95 --thinking \\
  --out-dir /sgl-workspace/logs`,
    },
    numPromptsByConc: { 1: 32, 16: 32, 64: 128, 256: 512, 1024: 2048 },
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
  },

  github: {
    cookbookModel: "tencent/hy3",
  },

  playgroundFeatures: {

    // ----- Card 1: "Attention Parallelism" -----
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null,
          1,
          2,
          4,
          8,
          { value: 16, disable: { nodes: ["single"] },
            disableReason: "TP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
        ]},
        { id: "cp",     label: "CP", values: [null, 1, 2, 4] },
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null,
            false,
            1,
            2,
            4,
            8,
            { value: 16, disable: { nodes: ["single"] },
              disableReason: "DP-Attention=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card 2: "MoE Parallelism" -----
    moe: {
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
    // ====================================================================
    // H200 (141GB) — TP=8 for BF16 (~552GB)
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 (180GB) — TP=8 for BF16
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--attention-backend trtllm_mha",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--attention-backend trtllm_mha",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend trtllm_mha",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--attention-backend trtllm_mha",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend trtllm_mha",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB200 (sm_100 + aarch64) — TP=4 (inferred from B200)
    // ====================================================================
    {
      match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend trtllm_mha",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--attention-backend trtllm_mha",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--kv-cache-dtype fp8_e4m3",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--dp 2",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--dp 2",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
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
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 2",
        "--dp 2",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--attention-backend trtllm_mha",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
