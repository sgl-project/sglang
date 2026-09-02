// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.

export const config = {
  modelName: "DeepSeek-V4",

  latencyPercentile: "P50",

  supportedHardware: [
    "h100", "h200", "b200", "b300", "gb200", "gb300",
    "rtx6000", "rtx5090",
    // NVIDIA DGX Spark (GB10, SM121) — Flash Official FP4 only, as a 2-node
    // TP=2 pair over ConnectX-7 RoCE; the shared HARDWARE_CATALOG carries the
    // entry and its multi-node Docker flags.
    "dgx-spark",
    // AMD ROCm — MI300X (Flash FP8) + MI355X (Flash/Pro, FP4/FP8).
    "mi300x", "mi355x",
  ],

  // Model-specific GPUs the shared HARDWARE_CATALOG doesn't carry — the engine
  // merges these in, so a model-specific GPU is config data, not an engine edit.
  // RTX PRO 6000 and RTX 5090 (SM120 / Blackwell Desktop) are workstation and
  // consumer cards, not datacenter GPUs.
  hardware: [
    { id: "rtx6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "blackwell" },
  ],

  variants: [
    { id: "flash", label: "Flash", subtitle: "284B" },
    { id: "flash-official", label: "Flash Official", subtitle: "284B · 0731" },
    { id: "flash-vision", label: "Flash Vision", subtitle: "305B · Exp" },
    { id: "pro",   label: "Pro",   subtitle: "1.6T" },
    { id: "pro-official", label: "Pro Official", subtitle: "1.6T · 0813" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
    { id: "fp4", label: "FP4" },
    { id: "nvfp4", label: "NVFP4" },
  ],
  strategies: [
    { id: "low-latency",    label: "Low-Latency"    },
    { id: "balanced",       label: "Balanced"       },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  // `multi-N` id carries the node count for `--nnodes N`.
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  modelNames: {
    "flash|fp4": "deepseek-ai/DeepSeek-V4-Flash",
    "flash|fp8": "deepseek-ai/DeepSeek-V4-Flash",
    "flash|nvfp4": "nvidia/DeepSeek-V4-Flash-NVFP4",
    "flash-official|fp4": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "flash-official|nvfp4": "nvidia/DeepSeek-V4-Flash-0731-NVFP4",
    "flash-vision|fp4": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "pro|fp4":   "deepseek-ai/DeepSeek-V4-Pro",
    "pro|fp8":   "deepseek-ai/DeepSeek-V4-Pro",
    "pro|nvfp4": "nvidia/DeepSeek-V4-Pro-NVFP4",
    "pro-official|fp4": "deepseek-ai/DeepSeek-V4-Pro-0813",
    "pro-official|nvfp4": "nvidia/DeepSeek-V4-Pro-0813-NVFP4",
    // H200 FP8 needs the sgl-project repackaging (Hopper can't run FP4-mixed Instruct).
    "h200|flash|fp8": "sgl-project/DeepSeek-V4-Flash-FP8",
    "h200|pro|fp8":   "sgl-project/DeepSeek-V4-Pro-FP8",
    // AMD FP8 uses the sgl-project repackaging.
    "mi300x|flash|fp8": "sgl-project/DeepSeek-V4-Flash-FP8",
    "mi355x|flash|fp8": "sgl-project/DeepSeek-V4-Flash-FP8",
    "mi355x|pro|fp8":   "sgl-project/DeepSeek-V4-Pro-FP8",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",       default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",       default: "30000"    },
    NODE0_IP:  { target: "command", label: "Head node IP",    default: "<node0-ip>"   },
    NODE_RANK: { target: "command", label: "This node rank",  default: "<node-rank>"  },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",     default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",     default: "30000"     },
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
  --warmup-requests 64 --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
      gpqa_pct: {
        flash:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gpqa \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 200000 \\
  --temperature 1.0 --top-p 1.0 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
        "flash-official":
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gpqa \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 200000 \\
  --temperature 1.0 --top-p 1.0 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
        pro:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gpqa \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 400000 \\
  --temperature 1.0 --top-p 1.0 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
      },
      aime25_pct: {
        "flash-official":
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 200000 \\
  --temperature 1.0 --top-p 1.0 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
        flash:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 200000 \\
  --temperature 1.0 --top-p 1.0 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
        pro:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 400000 \\
  --temperature 1.0 --top-p 1.0 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
      },
      mmmu_pro_pct: {
        "flash-vision":
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run mmmu_pro \\
  --reasoning-effort max \\
  --temperature 1.0 --top-p 0.95 \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
      },
    },
    numPromptsByConc: { 1: 32, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // Per-variant accuracy applied to every cell; per-cell `accuracy` overrides.
  defaultAccuracy: {
    flash: { gpqa_pct: 88.1, aime25_pct: 95,   gsm8k_pct: 96.13 },
    pro:   { gpqa_pct: 90.1, aime25_pct: 97.5, gsm8k_pct: 96.13 },
  },

  // The eval set rendered in the benchmark card + "⚡ Reproduce" (the engine
  // ships no default — every config declares its own).
  accuracyLabels: [
    ["gpqa_pct",   "GPQA Diamond",   "%"],
    ["aime25_pct", "AIME25",         "%"],
    ["gsm8k_pct",  "GSM8K (1-shot)", "%"],
    ["mmmu_pro_pct", "MMMU-Pro (standard, 10-option)", "%"],
  ],

  // Prepended as `# ...` comments above multi-node commands.
  multiNodeHints: {
    gb200: [
      "The following env vars may be needed depending on your cluster:",
      "  GLOO_SOCKET_IFNAME=<your-nic>",
      "  NVSHMEM_ENABLE_NIC_PE_MAPPING=1",
      "  NVSHMEM_HCA_LIST=<your-hca-list>",
    ],
  },

  dockerImages: {
    // Flash Vision (Exp) support has not shipped in a release yet
    // (sgl-project/sglang#37253) — until it does, the variant needs this
    // preview build on every hardware.
    "flash-vision|fp4": "lmsysorg/sglang:dev-dsv4-flash-vision",
    // DGX Spark ONLY. A dedicated preview build for the 2x GB10 pair: it bakes
    // in the SM12x b12x MoE/attention kernels (sgl-project/sglang#34878,
    // #35899, #34018) and CuTeDSL/NCCL pins the GB10 recipe needs, none of
    // which are in `latest`. It is not built for, and must not be used on, any
    // other hardware — every other row keeps its own image.
    "dgx-spark|flash-official|fp4": "lmsysorg/sglang:dev-v4f-2dgx",
    // NVFP4 checkpoints crash at weight load on v0.5.18 (the MXFP4-packed MTP
    // layer's FP8 delegate needs the #36275 guard, merged 2026-08-26) — route
    // every NVFP4 cell to the nightly until a release contains that fix.
    "b200|nvfp4":  "lmsysorg/sglang:dev",
    "b300|nvfp4":  "lmsysorg/sglang:dev",
    "gb200|nvfp4": "lmsysorg/sglang:dev",
    "gb300|nvfp4": "lmsysorg/sglang:dev",
    h100:  "lmsysorg/sglang:latest",
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    gb200: "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
    // AMD daily-updated lmsysorg/sglang-rocm images. Bump the dated tag when you
    // re-verify on a newer build.
    mi300x: "lmsysorg/sglang-rocm:v0.5.18-rocm720-mi30x-20260829",
    mi355x: "lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260829",
  },

  // Pre-selects the issue template's `model` dropdown on "Submit verified cell".
  github: {
    cookbookModel: "deepseek-ai/deepseek-v4",
  },

  playgroundFeatures: {

    // ----- Card 1: "Attention Parallelism" -----
    // DP-Attention is a combined knob: value is the DP degree AND toggles `--enable-dp-attention`.
    // CP sizes auto-gate in the engine to the runtime derivation
    // attn_cp_size = tp/dp (a user-passed --attn-cp-size is overridden).
    // CP is single-machine only (tp_size <= 8). Interleave CP + DP-Attention
    // currently fails the runtime's dp_size == 1 assert but is allowed here
    // with a warning (combined support is planned upstream). No `cpStrategy`
    // knob: DeepSeek-V4 supports only interleave (the runtime rejects zigzag).
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null,
          { value: 1, hide: { variant: ["pro"] } },
          { value: 2, hide: { variant: ["pro"] } },
          4,
          8,
          { value: 16, disable: { nodes: ["single"] },
            disableReason: "TP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
        ]},
        { id: "cp", label: "CP",
          values: [null, { value: 1, label: "Off" }, 2, 4, 8],
          disable: [
            { when: { nodes: ["multi-2"] },
              reason: "Prefill Context Parallel is single-machine only (SGLang asserts tp_size <= 8; cross-machine CP has precision issues)." },
          ] },
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null,
            false,
            { value: 1, hide: { variant: ["pro"] } },
            { value: 2, hide: { variant: ["pro"] } },
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
          // Blackwell-only; no strategy gate — the Playground allows MegaMoE on any
          // strategy for experimentation (docs recommend it on high-throughput).
          { id: "megamoe",           label: "MegaMoE",
            flags: ["--moe-a2a-backend megamoe"],
            requiresHw: ["b200", "b300", "gb200", "gb300"] },
          { id: "flashinfer_mxfp4",  label: "FlashInfer (MXFP4)",
            flags: ["--moe-runner-backend flashinfer_mxfp4"] },
          { id: "marlin",            label: "Marlin (W4A16)",
            flags: ["--moe-runner-backend marlin"] },
        ],
      },
      megamoeQuant: {
        stripEnv: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
        options: [
          { id: "w4a8", label: "W4A8",
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
          { id: "w4a4", label: "W4A4",
            flags: ["--enable-w4a4-mxfp4-megamoe"],
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
        ],
      },
      ep: { label: "EP", values: [
        null,
        { value: 1, hide: { variant: ["pro"] } },
        { value: 2, hide: { variant: ["pro"] } },
        4,
        8,
        { value: 16, disable: { nodes: ["single"] },
          disableReason: "EP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
      ]},
    },

    // ----- Card 3: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser deepseek-v4" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser deepseekv4" },
      ],
    },

    // ----- Card 4: "Speculative Decoding" -----
    speculative: {
      options: [
        { id: "current",    label: "Inherited from base" },
        { id: "off",        label: "Off (greedy)" },
        { id: "mtp-314",    label: "EAGLE / MTP 3-1-4",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"],
          hide: { variant: ["flash-official", "flash-vision", "pro-official"] } },
        { id: "mtp-112",    label: "EAGLE / MTP 1-1-2",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 1",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 2"],
          hide: { variant: ["flash-official", "flash-vision", "pro-official"] } },
        { id: "dspark",     label: "DSpark",
          flags: ["--speculative-algorithm DSPARK"],
          hide: { variant: ["flash", "pro"] },
          disable: [
            { when: { dpAttnOn: [true] },
              reason: "DSpark is not compatible with DP Attention on the current release." },
            { when: { hw: ["mi300x", "mi355x"] },
              reason: "DSpark currently requires CUDA." },
          ] },
        { id: "ngram",      label: "NGRAM",
          flags: ["--speculative-algorithm NGRAM",
                  "--speculative-num-draft-tokens 16",
                  "--speculative-ngram-max-bfs-breadth 10"],
          disable: { dpAttnOn: [true] },
          disableReason: "NGRAM is incompatible with DP-Attention. Turn DP-Attention off in the Attention card above to use NGRAM." },
        { id: "dflash",     label: "DFlash", disabled: true,
          disableReason: "Coming soon — pending DFlash kernel integration." },
      ],
    },

    // ----- Card 5: "PD Disaggregation" -----
    pdDisagg: {
      incompatibleSpeculativeAlgorithms: ["DSPARK"],
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
        // MORI-IO transport is AMD-only — hidden on every non-ROCm platform.
        { id: "mori",     label: "MORI",
          hide: { hw: ["h100", "h200", "b200", "b300", "gb200", "gb300", "rtx6000"] } },
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

    // ----- Card 6: "Hierarchical KV Cache" -----
    hicache: {
      excludesHw: ["rtx6000"],
      // AMD ROCm (MI300X/MI325X/MI350X/MI355X): page_first_direct + direct io.
      amdIo: { memLayout: "page_first_direct", ioBackend: "direct", ratio: 4 },
      roleOverrides: [
        {
          when: {
            hw: ["mi355x"], variant: ["pro"], quant: ["fp4"],
            strategy: ["low-latency"], nodes: ["single"],
          },
          mode: "prefill",
          transferBackend: "mori",
          memLayout: "page_first",
          ioBackend: "direct",
          ratio: 5,
          writePolicy: "write_through",
          prefetchPolicy: "best_effort",
        },
      ],
      notices: [
        {
          when: {
            hw: ["mi355x"], variant: ["pro"], quant: ["fp4"],
            strategy: ["low-latency"], nodes: ["single"],
          },
          mode: "decode",
          transferBackend: "mori",
          text: "HiCache is not recommended on the decode role with MORI.",
        },
      ],
      amdStorageFileOnly: true,
      backends: [
        { id: null,        label: "Auto" },
        { id: "file",      label: "File" },
        { id: "mooncake",  label: "Mooncake",
          hide: { hw: ["mi300x", "mi355x"] } },
        { id: "hf3fs",     label: "HF3FS",
          hide: { hw: ["mi300x", "mi355x"] } },
        { id: "nixl",      label: "NiXL",
          hide: { hw: ["mi300x", "mi355x"] } },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },

    // ----- Card 7: "HiSparse" -----
    // Decode-only: shown/emitted only when the live PD-Disagg mode is `decode`.
    hisparse: {
      requiredFlags: [
        "--disable-radix-cache",
      ],
      config: { top_k: 2048, device_buffer_size: 6144 },
      hostRatios: [
        { id: 5,  label: "5 (~1TB host)" },
        { id: 10, label: "10 (~2TB host)" },
      ],
      defaultHostRatio: 10,
    },

    flagSelects: [
      {
        id: "dsparkDraftTokens",
        title: "DSpark Proposed Draft Tokens",
        showWhen: (base) =>
          (base.variant === "flash-official" || base.variant === "pro-official") &&
          base.specAlgorithm === "DSPARK",
        control: "slider",
        stripPrefixes: ["--speculative-dspark-block-size"],
        options: [
          { id: "auto", label: "Checkpoint default" },
          { id: "1", label: "1", flags: ["--speculative-dspark-block-size 1"] },
          { id: "2", label: "2", flags: ["--speculative-dspark-block-size 2"] },
          { id: "3", label: "3", flags: ["--speculative-dspark-block-size 3"] },
          { id: "4", label: "4", flags: ["--speculative-dspark-block-size 4"] },
          { id: "5", label: "5", flags: ["--speculative-dspark-block-size 5"] },
        ],
      },
    ],
  },

  cells: [
    // ====================================================================
    // B200 + FP4
    // ====================================================================
    {
      match: { hw: "b200", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--chunked-prefill-size 32768",
        "--swa-full-tokens-ratio 0.1",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.92",
        "--cuda-graph-max-bs-decode 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.835",
        "--cuda-graph-max-bs-decode 544",
        "--swa-full-tokens-ratio 0.075",
        "--chunked-prefill-size 65536",
        "--tokenizer-worker-num 8",
        "--enable-prefill-delayer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "b300", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-runner-backend flashinfer_mxfp4",
        "--disable-flashinfer-autotune",
        "--chunked-prefill-size 32768",
        "--swa-full-tokens-ratio 0.1",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.92",
        "--cuda-graph-max-bs-decode 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.835",
        "--cuda-graph-max-bs-decode 544",
        "--swa-full-tokens-ratio 0.075",
        "--chunked-prefill-size 65536",
        "--tokenizer-worker-num 8",
        "--enable-prefill-delayer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 + NVFP4
    // ====================================================================
    {
      match: { hw: "b200", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "b200", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // B200 + NVFP4 — Official (0731 / 0813)
    // Mirrors the Flash/Pro NVFP4 cells; the official checkpoints bundle a
    // DSpark draft head, so low-latency uses `--speculative-algorithm DSPARK`
    // instead of the EAGLE shape flags. Verified on 8xB200 (GSM8K + AIME25,
    // sgl-eval; see the benchmarks entries).
    // ====================================================================
    {
      match: { hw: "b200", variant: "flash-official", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "pro-official", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // B300 + NVFP4
    // ====================================================================
    {
      match: { hw: "b300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // B300 + NVFP4 — Official (0731 / 0813)
    // Mirrors the Flash/Pro NVFP4 cells; the official checkpoints bundle a
    // DSpark draft head, so low-latency uses `--speculative-algorithm DSPARK`
    // instead of the EAGLE shape flags. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      match: { hw: "b300", variant: "flash-official", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro-official", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB200 + FP4
    // ====================================================================
    {
      match: { hw: "gb200", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 64",
        "--max-running-requests 128",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 64",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB200 + NVFP4
    // ====================================================================
    {
      match: { hw: "gb200", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // GB200 + NVFP4 — Official (0731 / 0813)
    // Mirrors the Flash/Pro NVFP4 cells; the official checkpoints bundle a
    // DSpark draft head, so low-latency uses `--speculative-algorithm DSPARK`
    // instead of the EAGLE shape flags. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      match: { hw: "gb200", variant: "flash-official", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "pro-official", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 + FP4
    // ====================================================================
    {
      match: { hw: "gb300", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // GB300 + FP4 — Pro Official (0813)
    //
    // The 0813 checkpoint bundles a DSpark draft head, so the low-latency
    // recipe uses `--speculative-algorithm DSPARK` and omits the EAGLE shape
    // flags (SGLang reads gamma from the checkpoint). EAGLE loads on this
    // checkpoint without erroring but accepts no draft tokens.
    // ====================================================================
    {
      match: { hw: "gb300", variant: "pro-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // --max-running-requests is server-wide and floor-divided by attn_dp_size,
      // so 512 gives 128 running slots per DP rank. That is the point where both
      // the slot budget and the KV pool run full on this topology; the three
      // memory flags together are what keep the KV pool large enough to reach it.
      match: { hw: "gb300", variant: "pro-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 512",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 + FP4 — Pro Official (0813)
    // Mirrors the verified Pro cells; speculative decoding re-fitted to the
    // bundled DSpark head. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      match: { hw: "b200", variant: "pro-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // DSpark is incompatible with DP attention -> target-only.
      match: { hw: "b200", variant: "pro-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--chunked-prefill-size 32768",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.92",
        "--cuda-graph-max-bs-decode 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "pro-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.835",
        "--cuda-graph-max-bs-decode 544",
        "--swa-full-tokens-ratio 0.075",
        "--chunked-prefill-size 65536",
        "--tokenizer-worker-num 8",
        "--enable-prefill-delayer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // B300 + FP4 — Pro Official (0813)
    // Mirrors the verified Pro cells; speculative decoding re-fitted to the
    // bundled DSpark head. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      match: { hw: "b300", variant: "pro-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // DSpark is incompatible with DP attention -> target-only.
      match: { hw: "b300", variant: "pro-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-runner-backend flashinfer_mxfp4",
        "--disable-flashinfer-autotune",
        "--chunked-prefill-size 32768",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.92",
        "--cuda-graph-max-bs-decode 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.835",
        "--cuda-graph-max-bs-decode 544",
        "--swa-full-tokens-ratio 0.075",
        "--chunked-prefill-size 65536",
        "--tokenizer-worker-num 8",
        "--enable-prefill-delayer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // GB200 + FP4 — Pro Official (0813)
    // Mirrors the verified Pro cells; speculative decoding re-fitted to the
    // bundled DSpark head. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      match: { hw: "gb200", variant: "pro-official", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // DSpark is incompatible with DP attention -> target-only.
      match: { hw: "gb200", variant: "pro-official", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 64",
        "--max-running-requests 128",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "pro-official", quant: "fp4", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1", "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.78",
        "--cuda-graph-max-bs-decode 64",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // H200 + FP4 — Pro Official (0813)
    // Mirrors the verified Pro cells; speculative decoding re-fitted to the
    // bundled DSpark head. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      match: { hw: "h200", variant: "pro-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // H100 + FP4 — Pro Official (0813)
    // Mirrors the verified Pro cells; speculative decoding re-fitted to the
    // bundled DSpark head. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      match: { hw: "h100", variant: "pro-official", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs-decode 8",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "pro-official", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs-decode 8",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "pro-official", quant: "fp4", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // MI355X + FP4 — Pro Official (0813)
    // Mirrors the verified Pro cells; speculative decoding re-fitted to the
    // bundled DSpark head. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      // DSpark requires CUDA; EAGLE binds a head that accepts nothing on 0813 -> target-only.
      match: { hw: "mi355x", variant: "pro-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_ROCM700A=0", "TORCH_BLAS_PREFER_HIPBLASLT=1", "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton", "AITER_BF16_FP8_MOE_BOUND=0", "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 16384",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // DSpark requires CUDA; EAGLE binds a head that accepts nothing on 0813 -> target-only.
      match: { hw: "mi355x", variant: "pro-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_ROCM700A=0", "TORCH_BLAS_PREFER_HIPBLASLT=1", "SGLANG_SHARED_EXPERT_TP1=1", "SGLANG_DP_SHARED_EXPERT_LOCAL=1", "SGLANG_DP_USE_GATHERV=1", "SGLANG_DP_USE_REDUCE_SCATTER=1", "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton", "AITER_BF16_FP8_MOE_BOUND=0", "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // DSpark requires CUDA; EAGLE binds a head that accepts nothing on 0813 -> target-only.
      match: { hw: "mi355x", variant: "pro-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_ROCM700A=0", "TORCH_BLAS_PREFER_HIPBLASLT=1", "SGLANG_SHARED_EXPERT_TP1=1", "SGLANG_DP_SHARED_EXPERT_LOCAL=1", "SGLANG_DP_USE_GATHERV=1", "SGLANG_DP_USE_REDUCE_SCATTER=1", "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton", "AITER_BF16_FP8_MOE_BOUND=0", "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 + NVFP4
    // ====================================================================
    {
      match: { hw: "gb300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // GB300 + NVFP4 — Official (0731 / 0813)
    // Mirrors the Flash/Pro NVFP4 cells; the official checkpoints bundle a
    // DSpark draft head, so low-latency uses `--speculative-algorithm DSPARK`
    // instead of the EAGLE shape flags. NOT yet run end-to-end on this hardware.
    // ====================================================================
    {
      match: { hw: "gb300", variant: "flash-official", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro-official", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio 0.1",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H200 + FP8 (deepep, no Marlin)
    // ====================================================================
    {
      match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_DSV4_FP4_EXPERTS=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
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
      match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 128",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 256",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 16",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--cuda-graph-max-bs-decode 8",
        "--max-running-requests 32",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 16",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.88",
        "--cuda-graph-max-bs-decode 8",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      env: [
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--dp 16",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.88",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "h200", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      // W4A8 (MXFP4 weights x FP8 activations, FlashInfer Humming kernels);
      // requires FlashInfer >= 0.6.18. Falls back: drop the precision flag
      // for the W4A16 path, or use --moe-runner-backend marlin.
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--flashinfer-mxfp4-moe-precision fp8",
        "--speculative-algorithm DSPARK",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      // W4A8 Humming path -- see the flash-official cell above.
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--flashinfer-mxfp4-moe-precision fp8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend marlin",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend marlin",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      // W4A8 Humming path -- see the flash-official cell above.
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--flashinfer-mxfp4-moe-precision fp8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.90",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--mem-fraction-static 0.88",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H100 + FP4 (Marlin runner)
    // ====================================================================
    {
      match: { hw: "h100", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algorithm DSPARK",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "flash-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algorithm DSPARK",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "flash-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs-decode 8",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs-decode 8",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      env: ["SGLANG_SHARED_EXPERT_TP1=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--moe-runner-backend marlin",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // RTX PRO 6000 (SM120 / Blackwell Desktop) — Flash + low-latency only
    // ====================================================================
    {
      match: { hw: "rtx6000", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_mxfp4",
        "--mem-fraction-static 0.92",
        "--cuda-graph-max-bs-decode 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_mxfp4",
        "--mem-fraction-static 0.92",
        "--cuda-graph-max-bs-decode 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // RTX 5090 (SM120 / Blackwell Desktop) — Flash Official + low-latency
    // ====================================================================
    {
      match: { hw: "rtx5090", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--mem-fraction-static 0.90",
        "--cuda-graph-max-bs-decode 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // AMD ROCm (MI300X / MI355X)
    // --------------------------------------------------------------------

    // ---------- MI300X (192GB) — Flash FP8 ----------
    {
      match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.1",
        "--disable-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 16384",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-prefill-delayer",
        "--prefill-delayer-max-delay-ms 5000",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.1",
        "--disable-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-prefill-delayer",
        "--prefill-delayer-max-delay-ms 5000",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.1",
        "--disable-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- MI355X (288GB) — Flash FP4 ----------
    {
      match: { hw: "mi355x", variant: "flash-official", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 16384",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 16384",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash-official", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash-official", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- MI355X (288GB) — Flash FP8 ----------
    {
      match: { hw: "mi355x", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 16384",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- MI355X (288GB) — Pro FP4 ----------
    {
      match: { hw: "mi355x", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 16384",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- MI355X (288GB) — Pro FP8 ----------
    {
      match: { hw: "mi355x", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 16384",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "pro", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "pro", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_SHARED_EXPERT_TP1=1",
        "SGLANG_DP_SHARED_EXPERT_LOCAL=1",
        "SGLANG_DP_USE_GATHERV=1",
        "SGLANG_DP_USE_REDUCE_SCATTER=1",
        "SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton",
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_OPT_USE_AITER_BATCHED_GEMM=true",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--enable-two-batch-overlap",
        "--attention-backend dsv4",
        "--page-size 256",
        "--mem-fraction-static 0.90",
        "--swa-full-tokens-ratio 0.15",
        "--enforce-shared-experts-fusion",
        "--kv-cache-dtype fp8_e4m3",
        "--chunked-prefill-size 65536",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // DGX Spark (GB10 / SM121) — Flash Official FP4, 2-node TP=2, Balanced
    // ====================================================================
    // One cell only: the verified GB10 recipe. It runs the SM12x b12x MoE
    // (W4A8) + b12x compressed-MLA attention with DSpark, split TP=2 across two
    // DGX Sparks over ConnectX-7 RoCE. Verified end to end on 2x DGX Spark with
    // the `lmsysorg/sglang:dev-v4f-2dgx` image (GSM8K 96%, AgentX c1/c2 clean).
    // Every other DGX Spark combination (other variants / quants / strategies /
    // single node) is intentionally absent and greys out: a single 128GB GB10
    // cannot hold the checkpoint, and the b12x kernels are text-only today
    // (image prefill for Flash Vision is unsupported on SM12x).
    // Env: b12x attention + FP8 wo_a opt-in + MHC post/pre fusion are the GB10
    // tuning knobs; SGLANG_B12X_MAX_TOKENS must track --chunked-prefill-size;
    // expandable_segments avoids unified-memory fragmentation OOMs.
    {
      match: { hw: "dgx-spark", variant: "flash-official", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
      verified: true,
      warn: "The Docker image lmsysorg/sglang:dev-v4f-2dgx is a DGX Spark-only preview build (2x GB10, TP=2 over ConnectX-7) — do not use it on other hardware. Use Docker mode: the bare Python command needs the b12x kernel package this image ships. See [DGX Spark notes](#spark-note).",
      env: [
        "SGLANG_SM120_FLASHMLA_BACKEND=b12x",
        "B12X_MLA_SM120_DSV4_H16_NATIVE=1",
        "SGLANG_OPT_FUSE_MHC_POST_PRE=1",
        "SGLANG_OPT_FP8_WO_A_GEMM=1",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1",
        "SGLANG_B12X_MAX_TOKENS=8192",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend b12x",
        "--speculative-algorithm DSPARK",
        "--chunked-prefill-size 8192",
        "--context-length 327680",
        "--mem-fraction-static 0.80",
        "--swa-full-tokens-ratio 0.2",
        "--cuda-graph-max-bs-decode 32",
        "--max-running-requests 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 + FP4 — Flash Vision (Exp)
    //
    // DeepSeek-V4-Flash-Vision-Exp (sgl-project/sglang#37253): the 0731
    // Flash base plus a vision encoder + aligner. The checkpoint bundles a
    // DSpark head; low-latency recipes enable it (--speculative-algorithm
    // DSPARK, no other spec flags — the draft ships in the main checkpoint),
    // verified on B200 via the MMMU-Pro round (4×B200, image batches).
    // Balanced / high-throughput stay target-only: those recipes run DP
    // attention, which DSpark is incompatible with on the current release.
    // GB300 verified via the same MMMU-Pro round (4×GB300); B300 /
    // GB200 / H200 / H100 — final verification in progress.
    // ====================================================================
    {
      match: { hw: "b200", variant: "flash-vision", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash-vision", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash-vision", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 / GB200 / GB300 + FP4 — Flash Vision (Exp)
    // ====================================================================
    {
      match: { hw: "b300", variant: "flash-vision", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash-vision", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash-vision", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash-vision", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash-vision", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "flash-vision", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash-vision", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash-vision", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: true,
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
        "--deepep-config '{\"normal_dispatch\":{\"num_sms\":96},\"normal_combine\":{\"num_sms\":96}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash-vision", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend megamoe",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H200 + FP4 — Flash Vision (Exp)
    // ====================================================================
    {
      match: { hw: "h200", variant: "flash-vision", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend marlin",
        "--speculative-algorithm DSPARK",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash-vision", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H100 + FP4 — Flash Vision (Exp)
    // ====================================================================
    {
      match: { hw: "h100", variant: "flash-vision", quant: "fp4", strategy: "balanced", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "DeepSeek-V4-Flash-Vision-Exp support has not shipped in an SGLang release yet (sglang PR 37253): Docker mode already points at the preview image; for Python mode install SGLang from that PR. See [Flash Vision notes](#vision-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
