// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr` literals — engine injects them.
//
// `{{MODEL_NAME}}` resolves to `modelNames` below (HF repo under the ai-sage org).
// Sibling page: gigachat3.5-reasoning.jsx. Keep the shared recipe flags in sync between the two configs.

export const config = {
  modelName: "GigaChat 3.5",

  // FP8 only: 432B params → ~432 GB of weights → one 8-GPU Hopper node (BF16, 864 GB,
  // does not fit 8×H100). H100 is verified end-to-end; H200 is unverified.
  supportedHardware: ["h100", "h200"],

  // Single checkpoint (Instant, no thinking). The Reasoning checkpoint has its own page + config.
  variants: [
    { id: "default", label: "Default" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
  ],
  // Two recipes per hw: MTP on → low-latency, MTP off → high-throughput.
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  // Eval rendered in the benchmark card; numbers live in the sibling -benchmarks.jsx.
  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  modelNames: {
    "default|fp8": "ai-sage/GigaChat3.5-432B-A28B",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"         },
    PORT:      { target: "command", label: "Bind port",         default: "30000"           },
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
  --flush-cache`,
    accuracy: {
      gsm8k_pct:
`python3 -m sglang.test.run_eval --eval-name gsm8k --api chat \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}} \\
  --num-examples 1319 --num-threads 64 --max-tokens 8192`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // Merged to main in PR #29189 (b63f8416b3, 2026-09-21). No released image carries it
  // and the nightly :dev path is unverified, so only the Python run mode is shown.
  // After an end-to-end Docker run: drop `runModes`, point dockerImages at that tag.
  runModes: ["python"],
  dockerImages: {
    h100: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "ai-sage/GigaChat3.5-432B-A28B",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // TP=8 verified. TP=4 only fits on H200 (~108 GB of FP8 weights per GPU).
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null,
          { value: 4, disable: { hw: ["h100"] },
            disableReason: "FP8 weights are ~432 GB — TP=4 does not fit 4×80 GB. Pick H200 or stay on TP=8." },
          8,
        ]},
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card: "MoE Parallelism" ----- 256 routed + 1 shared expert.
    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    // ----- Card: "Parsers" -----
    // Tool calls only. No reasoning parser: the Instant checkpoint never emits `</think>`
    // and the `gigachat35` reasoning parser is a forced splitter that would route the
    // whole answer into reasoning_content.
    parsers: {
      items: [
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser gigachat35" },
      ],
    },

    // ----- Capacity levers (measured on 8×H100, this checkpoint; details below) -----
    flagSelects: [
      {
        id: "mambaSsmDtype", title: "GDN State Precision",
        stripPrefixes: ["--mamba-ssm-dtype"],
        options: [
          { id: "auto", label: "Auto (FP32)" },
          { id: "bf16", label: "BFloat16 — halves state memory", flags: ["--mamba-ssm-dtype bfloat16"] },
        ],
      },
      {
        id: "kvCacheDtype", title: "KV Cache Precision",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "auto", label: "Auto (BF16)" },
          { id: "fp8", label: "FP8 (E4M3) — doubles KV capacity, ~2× slower decode on H100",
            flags: ["--kv-cache-dtype fp8_e4m3"],
            disable: { strategy: ["low-latency"] },
            disableReason: "With MTP on, the draft-extend CUDA graph capture runs out of memory under fp8 KV on 80 GB." },
        ],
      },
    ],

    // ----- Card: "Speculative Decoding" -----
    // Stacked NextN heads run through the multi-layer EAGLE worker (auto-selected for
    // this arch). Steps = head count (2); draft tokens = steps + 1.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP 2-1-3",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 2",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 3"],
          note: "MTP needs the Mamba pool pinned (--max-mamba-cache-size) and --max-running-requests raised, or concurrency is clamped — see Configuration Tips." },
      ],
    },
  },

  cells: [
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MTP (2 draft heads). Pin the state pool (240 slots) and raise --max-running-requests,
    // otherwise concurrency is clamped.
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--max-running-requests 80",
        "--max-mamba-cache-size 240",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // H200 — same Hopper FP8 path, 141 GB/GPU. H100 recipes copied verbatim (the pin
    // could go higher); unverified until measured.
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--max-running-requests 80",
        "--max-mamba-cache-size 240",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
