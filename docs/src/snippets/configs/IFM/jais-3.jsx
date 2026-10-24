// Single `export const config` literal — Mintlify re-evaluates this module at
// hydration time, so keep the cookbook data self-contained.
//
// The base recipes below serve each repository's main branch (no pinned
// revision) on NVIDIA H100 GPUs, using SGLang's native K2 Horizon support
// (released in 0.5.20). Playground overrides remain separate from them.

export const config = {
  modelName: "Jais 3",

  latencyPercentile: "P50",

  supportedHardware: ["h100"],
  runModes: ["python"],

  variants: [
    { id: "0.9b", label: "0.9B", subtitle: "Dense" },
    { id: "3.7b", label: "3.7B", subtitle: "Dense" },
    { id: "7b", label: "7B", subtitle: "Dense" },
    { id: "32b", label: "32B", subtitle: "Dense" },
    { id: "36b", label: "MoVA-36B-A4B", subtitle: "MoE + MoVA" },
    { id: "375b", label: "375B-A23B", subtitle: "MoE" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
    { id: "multi-2", label: "2 Nodes" },
  ],

  modelNames: {
    "0.9b|bf16": "IFM/Jais-3-0.9B",
    "3.7b|bf16": "IFM/Jais-3-3.7B",
    "7b|bf16": "IFM/Jais-3-7B",
    "32b|bf16": "IFM/Jais-3-32B",
    "36b|bf16": "IFM/Jais-3-MoVA-36B-A4B",
    "375b|bf16": "IFM/Jais-3-375B-A23B",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",   default: "0.0.0.0"   },
    PORT:      { target: "command", label: "Bind port",   default: "30000"     },
    NODE0_IP:  { target: "command", label: "Head node IP",   default: "<node0-ip>"   },
    NODE_RANK: { target: "command", label: "This node rank", default: "<node-rank>"  },
    CURL_HOST: { target: "curl",    label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port", default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.benchmark.serving \\
  --backend sglang \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}} \\
  --model {{MODEL_NAME}} --served-model-name {{MODEL_NAME}} \\
  --tokenizer {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} --tokenize-prompt \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --request-rate inf --seed 20260901 \\
  --temperature 0.0 --top-p 1.0 \\
  --warmup-requests 64 --flush-cache \\
  --output-file benchmark.raw.jsonl --output-details`,
    accuracy: {
      gsm8k_pct:
`pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --num-examples 1319 --num-threads 32 --n-repeats 1 \\
  --max-tokens 32768 --temperature 0.0 --top-p 0.95 \\
  --seed 0 --reasoning-effort high --prompt math`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  // The page covers a family, so use the largest checkpoint as the canonical
  // issue-form model while each recipe still resolves its exact HF repository.
  github: {
    cookbookModel: "IFM/Jais-3-375B-A23B",
  },

  playgroundFeatures: {
    // CP is unsupported/unvalidated for the xLLM path. DP-attention and
    // alternate MoE backends are also intentionally omitted until validated.
    attention: {
      knobs: [
        {
          id: "tp",
          label: "TP",
          values: [
            null,
            { value: 1, disable: { variant: ["375b"] },
              disableReason: "375B-A23B BF16 does not fit on one H100 at TP=1." },
            { value: 2, disable: { variant: ["375b"] },
              disableReason: "375B-A23B BF16 does not fit on two H100 GPUs." },
            { value: 4, disable: { variant: ["375b"] },
              disableReason: "375B-A23B BF16 does not fit on four H100 GPUs." },
            { value: 8, disable: { variant: ["375b"] },
              disableReason: "375B-A23B BF16 does not fit on eight H100 GPUs; it needs TP=16 across two nodes." },
            { value: 16, disable: { variant: ["0.9b", "3.7b", "7b", "32b", "36b"] },
              disableReason: "TP=16 spans two eight-GPU nodes and applies only to 375B-A23B." },
          ],
        },
      ],
    },

    // Jais 3 MoVA-36B-A4B and 375B-A23B contain sparse MoE feed-forward layers. Keep
    // expert parallelism disabled on the dense variants.
    // The MoVA recipe runs with EP=2. The 375B recipe is two-node TP=16 with EP=16.
    moe: {
      ep: {
        label: "EP",
        values: [
          null,
          { value: 1,
            disable: { variant: ["0.9b", "3.7b", "7b", "32b"] },
            disableReason: "Expert parallelism applies only to the sparse MoVA-36B-A4B and 375B-A23B variants." },
          { value: 2,
            disable: [
              { when: { variant: ["375b"] },
                reason: "The two-node 375B-A23B recipe runs at TP=16, where SGLang only accepts EP=16." },
              { when: { variant: ["0.9b", "3.7b", "7b", "32b"] },
                reason: "Expert parallelism applies only to the sparse MoVA-36B-A4B and 375B-A23B variants." },
              { when: { effTp: [1] },
                reason: "EP=2 requires an effective TP degree of at least 2." },
            ] },
          { value: 4,
            disable: [
              { when: { variant: ["375b"] },
                reason: "The two-node 375B-A23B recipe runs at TP=16, where SGLang only accepts EP=16." },
              { when: { variant: ["0.9b", "3.7b", "7b", "32b"] },
                reason: "Expert parallelism applies only to the sparse MoVA-36B-A4B and 375B-A23B variants." },
              { when: { effTp: [1, 2] },
                reason: "EP=4 requires an effective TP degree of at least 4." },
            ] },
          { value: 8,
            disable: [
              { when: { variant: ["375b"] },
                reason: "The two-node 375B-A23B recipe runs at TP=16, where SGLang only accepts EP=16." },
              { when: { variant: ["0.9b", "3.7b", "7b", "32b"] },
                reason: "Expert parallelism applies only to the sparse MoVA-36B-A4B and 375B-A23B variants." },
              { when: { effTp: [1, 2, 4] },
                reason: "EP=8 requires an effective TP degree of at least 8." },
              { when: { variant: ["36b"] },
                reason: "MoVA-36B-A4B has 100 routed experts, which is not divisible by EP=8." },
            ] },
          { value: 16,
            disable: [
              { when: { variant: ["0.9b", "3.7b", "7b", "32b", "36b"] },
                reason: "EP=16 applies only to the two-node 375B-A23B recipe at TP=16." },
            ] },
        ],
      },
    },

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser k2_horizon" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser k2_horizon" },
      ],
    },

    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off", label: "Off" },
        { id: "ngram", label: "NGRAM",
          flags: ["--speculative-algorithm NGRAM",
                  "--speculative-num-draft-tokens 16",
                  "--speculative-ngram-max-bfs-breadth 10"] },
      ],
    },

    pdDisagg: {
      modes: [
        { id: "off", label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode", label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl", label: "NiXL" },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`,
      },
    },

    hicache: {
      backends: [
        { id: null, label: "Auto" },
        { id: "file", label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "hf3fs", label: "HF3FS" },
        { id: "nixl", label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto", label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back", label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "h100", variant: "0.9b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--dtype bfloat16",
        "--attention-backend fa3",
        "--reasoning-parser k2_horizon",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "3.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--dtype bfloat16",
        "--attention-backend fa3",
        "--reasoning-parser k2_horizon",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--dtype bfloat16",
        "--attention-backend fa3",
        "--reasoning-parser k2_horizon",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "32b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--dtype bfloat16",
        "--attention-backend fa3",
        "--reasoning-parser k2_horizon",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "36b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--ep 2",
        "--dtype bfloat16",
        "--attention-backend fa3",
        "--json-model-override-args '{\"xllm_source_router_gemm_partitions\":2}'",
        "--reasoning-parser k2_horizon",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // About 760 GB of BF16 weights: more than one eight-H100 node (640 GB)
      // holds, so this recipe spans two nodes. The benchmarks were measured at
      // TP=16 without EP; the --ep 16 variant below has not been launched yet.
      match: { hw: "h100", variant: "375b", quant: "bf16", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--ep 16",
        "--dtype bfloat16",
        "--attention-backend fa3",
        "--model-loader-extra-config '{\"enable_multithread_load\":false}'",
        "--reasoning-parser k2_horizon",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
