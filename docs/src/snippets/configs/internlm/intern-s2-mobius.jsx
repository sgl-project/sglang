// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.

export const config = {
  modelName: "Intern-S2-Mobius",

  supportedHardware: ["h200", "b200"],

  variants: [
    { id: "default", label: "Intern-S2-Mobius", subtitle: "Mobius-v0" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
    { id: "fp8", label: "FP8" },
  ],
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "internlm/Intern-S2-Mobius",
    "default|fp8": "internlm/Intern-S2-Mobius-FP8",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // Reproduce commands for the Benchmark card's "⚡ Reproduce" modal.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 8 --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
      gpqa_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gpqa \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 16`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 1024 },
  },

  // Per-variant accuracy applied to every cell; per-cell `accuracy` overrides.
  // Measured on 2xH200 with the low-latency (EAGLE NEXTN 3-1-4) recipe, with no
  // server-side sampling override, so the checkpoint's generation_config.json
  // defaults apply throughout (temp=1.0, top_p=0.95, top_k=20). NOTE: the model
  // card separately recommends temp=0.8 / top_p=1.0 / top_k=50 / min_p=0.0 —
  // those are NOT what generation_config.json ships, so they only apply when the
  // client sends them explicitly.
  // gsm8k : full 1319-example test split.
  // gpqa  : Diamond, 198 problems × 8 repeats, pass@1 avg-of-8 = 79.23% ± 1.49,
  //         pass@8 = 88.38 %, majority@8 = 80.56 %, stop_rate = 100 %.
  defaultAccuracy: {
    default: { gsm8k_pct: 96.66, gpqa_pct: 79.23 },
  },

  accuracyLabels: [
    ["gpqa_pct",  "GPQA Diamond",   "%"],
    ["gsm8k_pct", "GSM8K (1-shot)", "%"],
  ],

  dockerImages: {
    h200: "lmsysorg/sglang:dev",
    b200: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "internlm/Intern-S2-Mobius",
  },

  playgroundFeatures: {
    // ----- Card: "Attention Parallelism" -----
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
        { id: "cp", label: "CP", values: [null, 1, 2, 4] },
      ],
    },

    // No "MoE Parallelism" card. The routed experts do not live per-layer: all 40
    // layers query 4 globally shared expert banks (`meta_mlp`, config `num_blocks: 4`
    // — models/interns2_mobius.py), so EP has nothing to shard. The runtime enforces
    // that: server_args._handle_model_specific_adjustments raises for this arch on
    // `--ep-size != 1` (and `--pp-size != 1`), so an EP chip would emit a command
    // that cannot start. `--moe-a2a-backend deepep` is out for the same reason.

    // ----- Card: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----  MTP (NEXTN) is the cook-worthy preset.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp-314", label: "MTP / NEXTN 3-1-4 (recommended)",
          flags: ["--speculative-algorithm NEXTN", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
        { id: "mtp-213", label: "MTP / NEXTN 2-1-3 (lighter draft)",
          flags: ["--speculative-algorithm NEXTN", "--speculative-num-steps 2",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 3"] },
      ],
    },
  },

  cells: [
    // ==== H200, 2 GPUs, BF16, low-latency (MTP NEXTN on) — VERIFIED ====
    // GSM8K 1319 leg: 96.66 % acc / 100 % stop. Bench 8K-in / 1K-out (see
    // intern-s2-mobius-benchmarks.jsx for the full 1/16/64 sweep; per conc=16
    // spec reaches 18029 total tok/s vs 9358 no-spec).
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mem-fraction-static 0.8",
        "--context-length 262144",
        "--reasoning-parser qwen3",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== H200, 2 GPUs, BF16, high-throughput (no speculative) — VERIFIED ====
    // GSM8K 1319 leg: 96.82 % acc / 100 % stop. Bench 8K-in / 1K-out — the
    // spec-off recipe scales cleanly to conc=256 (34786 tok/s total at
    // saturation), >1.3× the spec-on peak at conc=64. See benchmarks.jsx.
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mem-fraction-static 0.8",
        "--context-length 262144",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.6",
        "--context-length 262144",
        "--reasoning-parser qwen3",
        "--moe-runner-backend deep_gemm",
        "--disable-prefill-cuda-graph",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.6",
        "--context-length 262144",
        "--reasoning-parser qwen3",
        "--moe-runner-backend deep_gemm",
        "--disable-prefill-cuda-graph",
        "--cuda-graph-max-bs-decode 16",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== B200, 2 GPUs, BF16, low-latency (MTP NEXTN on) — INFERRED from H200 ====
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mem-fraction-static 0.8",
        "--context-length 262144",
        "--reasoning-parser qwen3",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== B200, 1 GPU, BF16, high-throughput — INFERRED (single 192 GB HBM fits 73 GB weights + KV) ====
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.8",
        "--context-length 262144",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.6",
        "--context-length 262144",
        "--reasoning-parser qwen3",
        "--moe-runner-backend deep_gemm",
        "--disable-prefill-cuda-graph",
        "--cuda-graph-max-bs-decode 16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mem-fraction-static 0.6",
        "--context-length 262144",
        "--reasoning-parser qwen3",
        "--moe-runner-backend deep_gemm",
        "--disable-prefill-cuda-graph",
        "--cuda-graph-max-bs-decode 16",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
