// Qwen3.6 cookbook config — consumed by /src/snippets/_deployment.jsx and
// /src/snippets/_playground.jsx. Pure data literal (no spreads/calls/IIFE).
//
// Ported verbatim from the legacy Qwen3.6 command generator (now removed in this
// migration). Two strategies map onto its speculative toggle: low-latency = MTP on
// (EAGLE + mamba radix cache v2), high-throughput = MTP off.
// Cells reproduce the generator's output verbatim EXCEPT the reasoning/tool-call parser
// flags are omitted (a Playground feature, not part of a Deployment/benchmark command —
// DSv4/Qwen3.5 convention). All variants tp=1 single-node. nvfp4 is Blackwell-only
// (B200/B300). Xeon (CPU) is supported via the balanced tier (Intel-vetted recipe,
// declared in config.hardware with vendor:"intel").

export const config = {
  modelName: "Qwen3.6",

  // TTFT/TPOT in the benchmarks file are P50 (median_ttft_ms / median_tpot_ms
  // from bench_serving), not means. Engine renders the "(P50)" label from this.
  latencyPercentile: "P50",

  supportedHardware: ["h100", "h200", "b200", "b300", "xeon"],

  // Xeon (CPU) isn't in the shared HARDWARE_CATALOG — declare it here so its
  // cells render. `vendor: "intel"` puts it in its own selector group.
  hardware: [
    { id: "xeon", label: "Xeon", vram: "host RAM", vendor: "intel" },
  ],

  variants: [
    { id: "35b-a3b", label: "35B-A3B", subtitle: "MoE A3B" },
    { id: "27b", label: "27B", subtitle: "Dense" },
  ],

  quantizations: [
    { id: "bf16",  label: "BF16" },
    { id: "fp8",   label: "FP8" },
    { id: "nvfp4", label: "NVFP4" },
  ],

  strategies: [
    { id: "low-latency",     label: "Low-Latency" },
    { id: "balanced",        label: "Balanced" },
    { id: "high-throughput", label: "High-Throughput" },
  ],

  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  multiNodeHints: {},

  modelNames: {
    "35b-a3b|bf16": "Qwen/Qwen3.6-35B-A3B",
    "35b-a3b|fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
    "35b-a3b|nvfp4": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "27b|bf16": "Qwen/Qwen3.6-27B",
    "27b|fp8": "Qwen/Qwen3.6-27B-FP8",
    "27b|nvfp4": "nvidia/Qwen3.6-27B-NVFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT:      { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl",    label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port", default: "30000" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"What is 15% of 240?"}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} \\
  --random-output-len {{OSL}} \\
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    numPromptsByConc: { 1: 8, 16: 64, 1024: 2048, 4096: 8192 },
  },

  dockerImages: {
    h100: "lmsysorg/sglang:latest",
    h200: "lmsysorg/sglang:latest",
    b200: "lmsysorg/sglang:latest",
    b300: "lmsysorg/sglang:latest",
    xeon: "lmsysorg/sglang:v0.5.13-xeon",
  },

  github: {
    cookbookModel: "qwen/qwen3.6",
  },

  // Playground axes on top of the selected Deploy cell. All variants tp=1;
  // the speculative select flips MTP (EAGLE) — the preset the LL cells bake in.
  playgroundFeatures: {

    // ----- Attention Parallelism -----
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- MoE Parallelism (35B-A3B is MoE; hidden on the dense 27B) -----
    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"],
            hide: { variant: ["27b"] } },
        ],
      },
      ep: { label: "EP", values: [
        null,
        { value: 1, hide: { variant: ["27b"] } },
        { value: 2, hide: { variant: ["27b"] } },
        { value: 4, hide: { variant: ["27b"] } },
        { value: 8, hide: { variant: ["27b"] } },
      ]},
    },

    // ----- Parsers (Qwen3.6: qwen3 reasoning + qwen3_coder tool-call; Playground-only) -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    // ----- Speculative Decoding (EAGLE / MTP); disabled on Xeon (unsupported) -----
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "eagle",   label: "EAGLE / MTP 3-1-4",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"],
          disable: { hw: ["xeon"] },
          disableReason: "Speculative decoding is not supported on Xeon (CPU)." },
      ],
    },

    // ----- PD Disaggregation -----
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl",     label: "NiXL" },
      ],
      // `auto` is a sentinel (emits no --disaggregation-ib-device flag).
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      // Router fronting the prefill + decode roles; substitute <prefill-host>/<decode-host>.
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

    // ----- Hierarchical KV Cache -----
    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "hf3fs",    label: "HF3FS" },
        { id: "nixl",     label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },

    // ----- Mamba Radix Cache (Qwen3.6 hybrid Gated-Delta-Net) -----
    // Coupled with MTP in the Deploy cells (LL bakes V2); exposed here to override.
    flagSelects: [
      {
        id: "mambaCache",
        title: "Mamba Radix Cache",
        stripPrefixes: ["--mamba-radix-cache-strategy"],
        options: [
          { id: "auto", label: "Inherited" },
          { id: "v1",   label: "V1 (off)" },
          { id: "v2",   label: "V2 (extra_buffer)",
            flags: ["--mamba-radix-cache-strategy extra_buffer"] },
        ],
      },
    ],
  },

  cells: [
    {
      // OOMs on a single 80GB H100: the mamba state cache can't fit alongside the
      // ~70GB bf16 weights + EAGLE draft at mem-fraction 0.8 (server crashes on
      // startup). Benchmark pending — needs a lower mem-fraction or multi-GPU tp
      // before it can be marked verified.
      match: { hw: "h100", variant: "35b-a3b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // HT conc-1024/4096 sweep is impractical on a single H100 node → not yet
      // benchmarked/verified (no benchmarks entry). Pending.
      match: { hw: "h100", variant: "35b-a3b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "35b-a3b", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // HT conc-1024/4096 sweep is impractical on a single H100 node → not yet
      // benchmarked/verified (no benchmarks entry). Pending.
      match: { hw: "h100", variant: "35b-a3b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // HT conc-1024/4096 sweep is impractical on a single H100 node → not yet
      // benchmarked/verified (no benchmarks entry). Pending.
      match: { hw: "h100", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // HT conc-1024/4096 sweep is impractical on a single H100 node → not yet
      // benchmarked/verified (no benchmarks entry). Pending.
      match: { hw: "h100", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "35b-a3b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // mem-fraction-static 0.92 (vs generator default 0.8): +13% throughput on this
      // KV-bound MoE cell. Re-benched at 0.92 (conc 1024+4096 on 0.5.16) → verified.
      match: { hw: "h200", variant: "35b-a3b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.92",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "35b-a3b", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "35b-a3b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "35b-a3b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "35b-a3b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "35b-a3b", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "35b-a3b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // NVFP4-MoE needs --moe-runner-backend flashinfer_cutlass (default crashes).
      // Re-benched on 0.5.16 with this flag (conc 1+16) → verified.
      match: { hw: "b200", variant: "35b-a3b", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend trtllm_mha",
        "--moe-runner-backend flashinfer_cutlass",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // flashinfer_cutlass moe runner (see LL note). Re-benched on 0.5.16 (conc 1024+4096) → verified.
      match: { hw: "b200", variant: "35b-a3b", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend trtllm_mha",
        "--moe-runner-backend flashinfer_cutlass",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "27b", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "27b", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "35b-a3b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "35b-a3b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "35b-a3b", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "35b-a3b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "35b-a3b", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend trtllm_mha",
        "--moe-runner-backend flashinfer_cutlass",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "35b-a3b", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend trtllm_mha",
        "--moe-runner-backend flashinfer_cutlass",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "27b", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend trtllm_mha",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "27b", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend trtllm_mha",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---- Xeon (CPU) — speculative unsupported -> single recipe -> balanced.
    // Ported verbatim from the legacy generator (--device cpu --disable-overlap-schedule
    // --tp {3 for 35B-A3B, 6 for 27B}; no spec/mamba/mem-fraction/trtllm_mha). NVFP4 is
    // Blackwell-only, so Xeon has bf16+fp8 only. verified:true — Intel-provided/vetted
    // recipe (no perf benchmark on our side, so these render with no speed row).
    {
      match: { hw: "xeon", variant: "35b-a3b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--device cpu",
        "--disable-overlap-schedule",
        "--tp 3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "xeon", variant: "35b-a3b", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--device cpu",
        "--disable-overlap-schedule",
        "--tp 3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "xeon", variant: "27b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--device cpu",
        "--disable-overlap-schedule",
        "--tp 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "xeon", variant: "27b", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--device cpu",
        "--disable-overlap-schedule",
        "--tp 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
