// MiniMax-M3 cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.
//
// MXFP8 MoE: validated single-node tp4 on NVIDIA Blackwell — B200 (sm_100),
// B300 (sm_103), GB300 (sm_103, aarch64 Grace); GB200 (sm_100, aarch64) is
// inferred-supported (both axes validated above) but not directly benchmarked.
// AMD: validated single-node tp8 — MI355X (gfx950, CDNA4) serves MXFP8
// natively; MI300X (gfx942, CDNA3) auto-converts MXFP8 -> block-fp8 [128,128]
// at load and serves it with the tuned ROCm kernels. MI350X (gfx950) and
// MI325X (gfx942) are inferred-supported from their same-arch siblings.
// Hopper (H200) cannot run the MXFP8 kernels, so it serves the bf16 build
// (MiniMaxAI/MiniMax-M3) at tp8 — validated on 8xH200. See §2.4 on the page.

export const config = {
  modelName: "MiniMax-M3",

  // TTFT/TPOT were recorded as Mean (no percentile restated in the source runs).
  latencyPercentile: "Mean",

  supportedHardware: ["b200", "b300", "gb200", "gb300", "mi300x", "mi325x", "mi350x", "mi355x", "h200"],

  variants: [
    { id: "default", label: "Default" },
  ],
  quantizations: [
    { id: "mxfp8", label: "MXFP8" },
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|mxfp8": "MiniMaxAI/MiniMax-M3-MXFP8",
    "default|bf16": "MiniMaxAI/MiniMax-M3",
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

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    accuracy: {
      gsm8k_pct:
`pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --temperature 1.0 --top-p 0.95 \\
  --thinking`,
      gpqa_pct:
`pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gpqa \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --temperature 1.0 --top-p 0.95 \\
  --thinking --n-repeats 4 --max-tokens 40960`,
      mmmu_pro_pct:
`pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run mmmu_pro \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --temperature 1.0 --top-p 0.95 \\
  --thinking`,
    },
    numPromptsByConc: { 24: 24, 64: 128 },
  },

  accuracyLabels: [
    ["gpqa_pct", "GPQA Diamond", "%"],
    ["gsm8k_pct", "GSM8K", "%"],
    ["mmmu_pro_pct", "MMMU-Pro", "%"],
  ],

  dockerImages: {
    // M3-specific dev images (multi-arch amd64+arm64). cu13 carries the sm_103
    // (B300/GB300) + Grace arm64 builds; cu12 is the Hopper/CUDA-12 build;
    // dev-minimax-m3 is the rolling default.
    b200: "lmsysorg/sglang:dev-minimax-m3",
    b300: "lmsysorg/sglang:dev-cu13-minimax-m3",
    gb200: "lmsysorg/sglang:dev-cu13-minimax-m3",
    gb300: "lmsysorg/sglang:dev-cu13-minimax-m3",
    h200: "lmsysorg/sglang:dev-cu12-minimax-m3",
    // AMD ROCm images — published M3 builds, by arch (gfx942 -> mi30x, gfx950 -> mi35x).
    mi300x: "aigmkt/minimax-m3-sglang-rocm700-mi30x",
    mi325x: "aigmkt/minimax-m3-sglang-rocm700-mi30x",
    mi350x: "aigmkt/minimax-m3-sglang-rocm720-mi35x",
    mi355x: "aigmkt/minimax-m3-sglang-rocm720-mi35x",
  },

  github: {
    cookbookModel: "MiniMaxAI/MiniMax-M3-MXFP8",
  },

  playgroundFeatures: {

    // ----- Attention Parallelism -----
    attention: {
      knobs: [
        { id: "tp",     label: "TP", values: [null, 1, 2, 4, 8] },
        { id: "cp",     label: "CP", values: [null, 1, 2, 4] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 1, 2, 4, 8],
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

    // ----- Parsers -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
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

    // ----- Hierarchical KV Cache -----
    hicache: {
      backends: [
        { id: null,        label: "Auto" },
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

  // NVIDIA Blackwell: one validated single-node recipe per family — tp4 across
  // B300 / GB200 / GB300, tp8 on B200. fa4 + page 128 + deep_gemm are the M3
  // SM100 auto-defaults on current main, so this is also the bare-launch
  // behavior; they engage MiniMax's MSA sparse-attention kernel (fmha_sm100,
  // pre-installed in the dev-minimax-m3 images; see Configuration Tips), Triton
  // fallback otherwise.
  // AMD: tp8. MI350X/MI355X (gfx950) serve MXFP8 natively (backends auto). MI300X/
  // MI325X (gfx942) need --attention-backend aiter + --moe-runner-backend triton,
  // and the MXFP8 weights are auto-converted to block-fp8 at load; the cold-start
  // AITER JIT can exceed the default warmup window, hence the watchdog/skip flags.
  cells: [
    {
      match: { hw: "b200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--attention-backend fa4",
        "--moe-runner-backend deep_gemm",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.65",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend fa4",
        "--moe-runner-backend deep_gemm",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.75",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB200 (sm_100 + aarch64): inferred-supported (both axes validated on
      // B200 and GB300), not directly benchmarked. Same recipe as the others.
      match: { hw: "gb200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend fa4",
        "--moe-runner-backend deep_gemm",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.75",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 4",
        "--attention-backend fa4",
        "--moe-runner-backend deep_gemm",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.75",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI355X (gfx950, CDNA4): native MXFP8, backends auto-selected.
      match: { hw: "mi355x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--quantization mxfp8",
        "--dtype bfloat16",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI350X (gfx950, CDNA4): inferred-supported from MI355X (same arch),
      // not directly benchmarked. Same native-MXFP8 recipe.
      match: { hw: "mi350x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--quantization mxfp8",
        "--dtype bfloat16",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI300X (gfx942, CDNA3): no hardware MX matmul — SGLang converts MXFP8 ->
      // block-fp8 [128,128] at load. aiter attention + triton MoE runner are the
      // validated backends; watchdog/skip-warmup ride out the cold-start AITER JIT.
      match: { hw: "mi300x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--quantization mxfp8",
        "--dtype bfloat16",
        "--attention-backend aiter",
        "--moe-runner-backend triton",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 3600",
        "--skip-server-warmup",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI325X (gfx942, CDNA3): inferred-supported from MI300X (same arch),
      // not directly benchmarked. Same MXFP8 -> block-fp8 recipe.
      match: { hw: "mi325x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--quantization mxfp8",
        "--dtype bfloat16",
        "--attention-backend aiter",
        "--moe-runner-backend triton",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 3600",
        "--skip-server-warmup",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Hopper (H200): MXFP8 MoE kernels are Blackwell-only, so Hopper serves the
      // full-precision bf16 build (MiniMaxAI/MiniMax-M3) at tp8 — bf16 weights don't
      // fit a single 4-GPU node. Everything else auto-resolves for Hopper: fa3
      // attention, page_size 1, MoE auto-pins to Triton (the bf16 deep_gemm path is
      // not used), decode keeps full CUDA graph; MSA (§2.1) is Blackwell-only so the
      // sparse step runs on the built-in Triton fallback. See §2.4.
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--tp 8",
        "--mem-fraction-static 0.75",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
