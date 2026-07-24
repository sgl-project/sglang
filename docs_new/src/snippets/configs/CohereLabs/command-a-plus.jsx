// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.

export const config = {
  modelName: "Command A+",

  // B300 + H200 are benchmark-verified. B200 (Blackwell — same recipe as B300) and
  // H100 (Hopper — same recipe as H200) are sanity-checked → unverified badge until
  // re-run. Other hw auto-greys-out.
  supportedHardware: ["b300", "b200", "h200", "h100"],

  // No size variants — the three HF repos differ only by precision (see quantizations).
  variants: [
    { id: "default", label: "Default" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
    { id: "fp8",  label: "FP8"  },
    { id: "w4a4", label: "W4A4" },
  ],
  // One verified operating point per precision → a single Balanced tier.
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "CohereLabs/command-a-plus-05-2026-bf16",
    "default|fp8":  "CohereLabs/command-a-plus-05-2026-fp8",
    "default|w4a4": "CohereLabs/command-a-plus-05-2026-w4a4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost"      },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"          },
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
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
    numPromptsByConc: { 16: 80, 64: 80, 80: 80, 256: 256 },
  },

  // Accuracy differs per precision, so it lives per-cell in the benchmarks file.
  defaultAccuracy: {
    default: { gsm8k_pct: null },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K (5-shot)", "%"],
  ],

  dockerImages: {
    b300: "lmsysorg/sglang:latest",
    b200: "lmsysorg/sglang:latest",
    h200: "lmsysorg/sglang:latest",
    h100: "lmsysorg/sglang:latest",
  },

  github: {
    cookbookModel: "CohereLabs/command-a-plus",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    attention: {
      knobs: [
        { id: "tp",     label: "TP", values: [null, 1, 2, 4, 8] },
        { id: "cp",     label: "CP", values: [null, 1, 2, 4] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card: "MoE Parallelism" -----  Command A+ is a 128-expert MoE; these are
    // the runner backends that apply to its cohere2_moe checkpoints (cutlass = BF16,
    // triton = default/FP8, flashinfer_trtllm = NVFP4 W4A4), plus DeepEP + the EP knob.
    moe: {
      backend: {
        options: [
          { id: null,                label: "Inherited" },
          { id: "deepep",            label: "DeepEP",
            flags: ["--moe-a2a-backend deepep"] },
          { id: "cutlass",           label: "CUTLASS",
            flags: ["--moe-runner-backend cutlass"] },
          { id: "triton",            label: "Triton",
            flags: ["--moe-runner-backend triton"] },
          { id: "flashinfer_trtllm", label: "FlashInfer TRT-LLM (NVFP4)",
            flags: ["--moe-runner-backend flashinfer_trtllm"] },
        ],
      },
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    // ----- Card: "Parsers" -----  Cohere Command-A family uses one detector for both
    // reasoning (<START_THINKING>/<END_THINKING>) and tool calls.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser cohere_command4" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser cohere_command4" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----  No EAGLE/MTP draft ships for Command A+;
    // NGRAM is the draft-free option that applies.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "ngram",   label: "NGRAM",
          flags: ["--speculative-algorithm NGRAM",
                  "--speculative-num-draft-tokens 16",
                  "--speculative-ngram-max-bfs-breadth 10"],
          disable: { dpAttnOn: [true] },
          disableReason: "NGRAM is incompatible with DP-Attention. Turn DP-Attention off in the Attention card above to use NGRAM." },
      ],
    },

    // ----- Card: "PD Disaggregation" -----
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
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:30000 \\
  --decode http://<decode-host>:30001 \\
  --host 0.0.0.0 --port 8000 \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`,
      },
    },

    // ----- Card: "Hierarchical KV Cache" -----  218B model — host-tier KV offload helps.
    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "hf3fs",    label: "HF3FS" },
        { id: "nixl",     label: "NiXL" },
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
    // B300 + BF16 (TP=4, CUTLASS MoE runner; attention auto-selects trtllm_mha)
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend cutlass",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // B300 + FP8 (TP=2, compressed-tensors W8A8 PTPC; default Triton MoE runner)
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // B300 + W4A4 (TP=1, NVFP4 experts; FlashInfer TRT-LLM MoE runner)
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "w4a4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // B200 (Blackwell) — same recipe as B300 (B200's 192 GB fits all three
    // precisions at these TPs). Not yet benchmarked on B200 → verified: false.
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend cutlass",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "w4a4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // H200 (Hopper) — plain TP, default Triton FusedMoE (no Blackwell CUTLASS
    // path). Sizes follow Cohere's official per-quant GPU guidance (BF16 TP=8,
    // FP8 TP=4). VERIFIED on an 8x H200 devbox (sglang main @ 20b2817). The
    // `--cuda-graph-backend-prefill disabled` flag is REQUIRED on current main:
    // the tc_piecewise prefill-graph compile crashes for cohere2_moe (see the
    // page §2 note). Decode CUDA graph stays full. W4A4 (NVFP4) is Blackwell-only
    // — the Hopper dequant path isn't validated, so no H200 W4A4 cell.
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--cuda-graph-backend-prefill disabled",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--cuda-graph-backend-prefill disabled",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // H100 (Hopper) — same plain-TP recipe as H200 (default Triton MoE) + the
    // same `--cuda-graph-backend-prefill disabled` workaround. Not yet
    // benchmarked → verified: false. W4A4 (NVFP4) is Blackwell-only (no H100 cell).
    // ====================================================================
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--cuda-graph-backend-prefill disabled",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--cuda-graph-backend-prefill disabled",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
