// MiniMax-M2.7 cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.
//
// Migrated faithfully from the legacy MiniMax-M2.7 generator widget. The legacy
// page is the single source of truth: flags, env prefixes, TP/EP values, and
// docker tags are copied verbatim — NOT modernized. All cells are left
// unverified (yellow): a faithful port carries no fresh measurement, so the
// re-verification track owns flipping any cell to green.
//
// 5-dim mapping (legacy -> template):
//   hardware radio  -> match.hw  (A100 + Xeon are off-catalog -> config.hardware)
//   gpuCount radio  -> strategies as GPU-budget tiers: 2 GPUs -> low-latency,
//                      4 GPUs -> balanced, 8 GPUs -> high-throughput. The legacy
//                      SUPPORT matrix (2 GPUs only on AMD/GB300; 8 GPUs hidden on
//                      GB300; Xeon fixed at TP=6) is preserved by which cells
//                      exist. Xeon's single fixed-TP recipe has no
//                      latency/throughput slant -> balanced (per-combination
//                      rule; Qwen3.5 Xeon precedent). [NAMING needs maintainer
//                      sign-off — see PR body.]
//   precision radio -> quantizations (fp8 / fp4). NVFP4 is Blackwell-only, so
//                      fp4 cells exist only on B200/GB300; the engine greys it
//                      elsewhere.
//   thinking radio  -> parsers axis (--reasoning-parser, Playground-only)
//   toolcall radio  -> parsers axis (--tool-call-parser, Playground-only)
//
// AMD recipes append --kv-cache-dtype fp8_e4m3 unconditionally in the legacy
// command ("for memory efficiency") — kept verbatim per the migration rule
// (baked into the default command, not a user-selectable toggle).

export const config = {
  modelName: "MiniMax-M2.7",

  supportedHardware: ["h200", "b200", "gb300", "a100", "h100", "mi300x", "mi325x", "mi355x", "xeon"],

  // Off-catalog hardware the shared HARDWARE_CATALOG doesn't carry. The engine
  // merges these in (A100 joins the NVIDIA row; Xeon renders a new INTEL row).
  hardware: [
    { id: "a100", label: "A100", vram: "80GB",    vendor: "nvidia" },
    { id: "xeon", label: "Xeon", vram: "host RAM", vendor: "intel"  },
  ],

  variants: [
    { id: "default", label: "Default" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
    { id: "fp4", label: "FP4" },
  ],
  // GPU-budget tiers: 2 GPUs -> low-latency, 4 -> balanced, 8 -> high-throughput.
  // The page ships the full trio; the engine greys unused chips per selection.
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "balanced",        label: "Balanced"        },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|fp8": "MiniMaxAI/MiniMax-M2.7",
    "default|fp4": "nvidia/MiniMax-M2.7-NVFP4",
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

  // Reproduce commands for the Benchmark card's "⚡ Reproduce" modal.
  // Speed mirrors the legacy §5.2 bench_serving runs (random isl/osl 1000/1000).
  // Accuracy mirrors the legacy §5.1 NVIDIA NeMo-Skills (`ns eval`) commands.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    accuracy: {
      gpqa_pct:
`ns prepare_data gpqa
ns eval \\
  --cluster=local --server_type=openai \\
  --model={{MODEL_NAME}} \\
  --server_address=http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --output_dir=./m2.7-eval/ \\
  --benchmarks=gpqa:8 \\
  ++prompt_config=eval/aai/mcq-4choices \\
  ++inference.tokens_to_generate=120000 \\
  ++inference.temperature=0.6 ++inference.top_p=0.95 \\
  ++parse_reasoning=True`,
      aime25_pct:
`ns prepare_data aime25
ns eval \\
  --cluster=local --server_type=openai \\
  --model={{MODEL_NAME}} \\
  --server_address=http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --output_dir=./m2.7-eval/ \\
  --benchmarks=aime25:8 \\
  ++inference.tokens_to_generate=120000 \\
  ++inference.temperature=0.6 ++inference.top_p=0.95 \\
  ++parse_reasoning=True`,
      mmlu_pro_pct:
`ns prepare_data mmlu-pro
ns eval \\
  --cluster=local --server_type=openai \\
  --model={{MODEL_NAME}} \\
  --server_address=http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --output_dir=./m2.7-eval/ \\
  --benchmarks=mmlu-pro \\
  ++prompt_config=eval/aai/mcq-10choices \\
  ++inference.tokens_to_generate=32768 \\
  ++inference.temperature=0.0 \\
  ++parse_reasoning=True`,
      gsm8k_pct:
`python3 benchmark/gsm8k/bench_sglang.py \\
  --num-questions 1319 --num-shots 8 \\
  --parallel 32 \\
  --host {{CURL_HOST}} --port {{CURL_PORT}}`,
    },
    numPromptsByConc: { 1: 10, 100: 500 },
  },

  accuracyLabels: [
    ["gpqa_pct",     "GPQA Diamond", "%"],
    ["aime25_pct",   "AIME 2025",    "%"],
    ["mmlu_pro_pct", "MMLU-Pro",     "%"],
    ["gsm8k_pct",    "GSM8K",        "%"],
  ],

  dockerImages: {
    // Pinned exactly as the legacy §2 install table — not upgraded.
    a100:   "lmsysorg/sglang:v0.5.10.post1",
    h100:   "lmsysorg/sglang:v0.5.10.post1",
    h200:   "lmsysorg/sglang:v0.5.10.post1",
    b200:   "lmsysorg/sglang:v0.5.10.post1",
    gb300:  "lmsysorg/sglang:v0.5.10.post1-cu130",
    mi300x: "lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x",
    mi325x: "lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x",
    mi355x: "lmsysorg/sglang:v0.5.10.post1-rocm720-mi35x",
    // xeon: no docker mapping — install from source (CPU build); falls back to :dev.
  },

  github: {
    cookbookModel: "MiniMaxAI/MiniMax-M2.7",
  },

  playgroundFeatures: {

    // ----- Attention Parallelism -----
    attention: {
      knobs: [
        { id: "tp",     label: "TP", values: [null, 2, 4, 6, 8] },
        { id: "cp",     label: "CP", values: [null, 1, 2, 4] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- MoE Parallelism -----
    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 2, 4, 8] },
    },

    // ----- Parsers (Playground-only; never baked into Deploy cells) -----
    // M2.7 uses an inline-tag reasoning parser (minimax-append-think: thinking is
    // wrapped in <think>...</think> inside message.content) + the minimax-m2
    // tool-call parser.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser minimax-append-think" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser minimax-m2" },
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
        { id: "mooncake", label: "Mooncake",
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb300"] } },
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
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--mem-fraction-static 0.85",
        "--fp8-gemm-backend flashinfer_trtllm",
        "--dtype bfloat16",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp4", strategy: "low-latency", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--mem-fraction-static 0.85",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--mem-fraction-static 0.85",
        "--fp8-gemm-backend flashinfer_trtllm",
        "--dtype bfloat16",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp4", strategy: "balanced", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--mem-fraction-static 0.85",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.85",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--mem-fraction-static 0.85",
        "--fp8-gemm-backend flashinfer_trtllm",
        "--dtype bfloat16",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp4", strategy: "balanced", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--mem-fraction-static 0.85",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--mem-fraction-static 0.85",
        "--fp8-gemm-backend flashinfer_trtllm",
        "--dtype bfloat16",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_USE_FUSED_PARALLEL_QKNORM=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--mem-fraction-static 0.85",
        "--enable-flashinfer-allreduce-fusion",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "a100", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "a100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--ep 2",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--ep 2",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--ep 2",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--mem-fraction-static 0.85",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "xeon", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
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
