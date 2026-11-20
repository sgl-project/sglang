// Hy4-flash cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.
//
// Deploy dims (matchDims): Hardware × Model Variant × Quantization × Strategy.
// No nodes dim — every recipe is a single host.
//
// Sizing (drives the TP choices below):
// ~220B total / ~12B active MoE (512 routed + 1 shared experts, top-16).
// About 63B of the total are the Engram n-gram hash tables (4 layers). The
// runtime re-quantizes them to FP8 on load and shards them across TP ranks,
// so on-GPU weights come to ≈ 378GB (BF16 checkpoint) and ≈ 224GB (FP8
// checkpoint), not the 440GB / 287GB on disk.
//   B300 / GB300 (288GB): TP4 for both checkpoints.
//   H200 (141GB): FP8 → TP4 (~56GB/rank); BF16 → TP8 (~47GB/rank). BF16 at
//   TP4 would be ~95GB/rank and leave almost no KV pool.
//
// Both checkpoints also carry the vision tower and the DFly speculative-
// decoding draft weights. The recipes serve text + image input (the vision
// tower loads by default). Prefill CUDA graphs stay off
// (`--cuda-graph-backend-prefill disabled`).
//
// Strategies: Low-Latency = DFly speculative decoding on (block size 6);
// Balanced = DFly off. Same TP per hardware × quant for both.
//
// GB300 BF16 was verified with `--language-model-only` before VL support
// landed; its recipe now drops that flag and was not re-run.
//
// Recipes carry `verificationStatus: "in-progress"` until each one has been
// run end-to-end — when it lands, REPLACE that line with `verified: true`
// (`verificationStatus` takes precedence over `verified` in the engine).
// Verified so far: Balanced on GB300 BF16, B300 FP8, B300 BF16, H200 FP8,
// GB300 FP8; Low-Latency (DFly) on B300 FP8, B300 BF16 (GSM8K only, no 8k1k),
// GB300 FP8. GB300 FP8 (both strategies) was a startup check only — server
// up, text + image requests answered; no benchmark numbers yet.

export const config = {
  modelName: "Hy4-flash",

  supportedHardware: ["h200", "b300", "gb300"],

  matchDims: [
    {
      id: "variant",
      title: "Model Variant",
      options: [
        { id: "hy4-flash", label: "Hy4-flash" },
      ],
    },
    {
      id: "quant",
      title: "Quantization",
      options: [
        { id: "bf16", label: "BF16" },
        { id: "fp8",  label: "FP8"  },
      ],
    },
    {
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency", label: "Low-Latency" },
        { id: "balanced",    label: "Balanced"    },
      ],
    },
  ],

  modelNames: {
    "hy4-flash|bf16": "tencent/Hy4-flash",
    "hy4-flash|fp8":  "tencent/Hy4-flash-FP8",
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
`python3 -m sglang.benchmark.serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
    // 10 prompts at C=1, 5 × C otherwise.
    numPromptsByConc: { 1: 10, 64: 320 },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    h200:  "lmsysorg/sglang:hy4-flash",
    b300:  "lmsysorg/sglang:hy4-flash",
    gb300: "lmsysorg/sglang:hy4-flash",
  },

  github: {
    cookbookModel: "tencent/Hy4-flash",
  },

  playgroundFeatures: {

    // ----- Card 1: "PD Disaggregation" -----
    pdDisagg: {
      // DFly rejects PD disaggregation at startup, so the roles are greyed out
      // on the Low-Latency (DFly) recipes.
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role",
          disable: [{ when: { strategy: ["low-latency"] },
                      reason: "DFly speculative decoding does not support PD disaggregation — switch the Deploy panel's Strategy to Balanced." }] },
        { id: "decode",  label: "Decode role",
          disable: [{ when: { strategy: ["low-latency"] },
                      reason: "DFly speculative decoding does not support PD disaggregation — switch the Deploy panel's Strategy to Balanced." }] },
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
      // `auto` is a sentinel (emits no --disaggregation-ib-device flag).
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      // Router fronting the prefill + decode roles; substitute <prefill-host>/<decode-host>.
      router: {
        port: 8000,
        command:
`sgl-router \\
  --model-id {{MODEL_NAME}} \\
  --no-tokenizer \\
  --worker-urls http://<prefill-host>:{{PREFILL_PORT}} http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --disable-circuit-breaker`,
      },
    },

    // ----- Card 2: "Hierarchical KV Cache" -----
    // Host (L2) and storage (L3) tiers; the hybrid-cache linkers restore the
    // DSA indexer and sliding-window state alongside the MLA KV.
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
    // NOTE: the engine defaults a hash-less visit to cells[0].

    // ====================================================================
    // B300 (288GB) — TP4 for both checkpoints.
    // ====================================================================
    {
      match: { hw: "b300", variant: "hy4-flash", quant: "fp8", strategy: "balanced" },
      nnodes: 1,
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
    {
      match: { hw: "b300", variant: "hy4-flash", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--cuda-graph-backend-prefill disabled",
        "--speculative-algorithm DFLY",
        "--speculative-dfly-block-size 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "hy4-flash", quant: "bf16", strategy: "balanced" },
      nnodes: 1,
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
    {
      match: { hw: "b300", variant: "hy4-flash", quant: "bf16", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--cuda-graph-backend-prefill disabled",
        "--speculative-algorithm DFLY",
        "--speculative-dfly-block-size 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 (288GB, 4-GPU hosts) — TP4 for both checkpoints.
    // ====================================================================
    {
      match: { hw: "gb300", variant: "hy4-flash", quant: "fp8", strategy: "balanced" },
      nnodes: 1,
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
    {
      match: { hw: "gb300", variant: "hy4-flash", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--cuda-graph-backend-prefill disabled",
        "--speculative-algorithm DFLY",
        "--speculative-dfly-block-size 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "hy4-flash", quant: "bf16", strategy: "balanced" },
      nnodes: 1,
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
    {
      match: { hw: "gb300", variant: "hy4-flash", quant: "bf16", strategy: "low-latency" },
      nnodes: 1,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--cuda-graph-backend-prefill disabled",
        "--speculative-algorithm DFLY",
        "--speculative-dfly-block-size 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H200 (141GB) — FP8 TP4 (~56GB/rank); BF16 TP8 (~47GB/rank).
    // ====================================================================
    {
      match: { hw: "h200", variant: "hy4-flash", quant: "fp8", strategy: "balanced" },
      nnodes: 1,
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
    {
      match: { hw: "h200", variant: "hy4-flash", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--cuda-graph-backend-prefill disabled",
        "--speculative-algorithm DFLY",
        "--speculative-dfly-block-size 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "hy4-flash", quant: "bf16", strategy: "balanced" },
      nnodes: 1,
      verificationStatus: "in-progress",
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
      match: { hw: "h200", variant: "hy4-flash", quant: "bf16", strategy: "low-latency" },
      nnodes: 1,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--cuda-graph-backend-prefill disabled",
        "--speculative-algorithm DFLY",
        "--speculative-dfly-block-size 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
