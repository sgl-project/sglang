export const config = {
  modelName: "Ling-3.0-flash",

  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300"],
  groupHardware: false,

  variants: [{ id: "default", label: "Ling-3.0-flash" }],
  quantizations: [
    { id: "bf16", label: "BF16" },
    { id: "fp8", label: "FP8" },
    { id: "int4", label: "INT4" },
    { id: "mxfp4", label: "MXFP4" },
  ],
  strategies: [
    { id: "low-latency", label: "Low-Latency" },
    { id: "high-throughput", label: "High-Throughput" },
    { id: "hicache", label: "HiCache + Mooncake" },
  ],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|bf16": "inclusionAI/Ling-3.0-flash",
    "default|fp8": "inclusionAI/Ling-3.0-flash-fp8",
    "default|int4": "inclusionAI/Ling-3.0-flash-int4",
    "default|mxfp4": "inclusionAI/Ling-3.0-flash-fp4",
  },

  placeholders: {
    HOST_IP:      { target: "command", label: "Bind host",          default: "0.0.0.0"              },
    PORT:         { target: "command", label: "Bind port",          default: "30000"                },
    HF_TOKEN:     { target: "command", label: "HF token (Docker)",  default: "<your-hf-token>"      },
    MOONCAKE_MASTER: { target: "command", label: "Mooncake master", default: "127.0.0.1:50171"      },
    MOONCAKE_METADATA_SERVER: { target: "command", label: "Mooncake metadata", default: "http://127.0.0.1:8290/metadata" },
    CURL_HOST:    { target: "curl",    label: "Server host",        default: "localhost"            },
    CURL_PORT:    { target: "curl",    label: "Server port",        default: "30000"                },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"What is the capital of France?"}] }'`,

  dockerImages: {
    "h20-3e": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "h200": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "h800": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "h100": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "b200": "lmsysorg/sglang:dev-Ling-3.0-flash",
    "gb300": "lmsysorg/sglang:dev-Ling-3.0-flash",
  },

  dockerHostNetworkWhen: (_sel, { flags }) =>
    flags.some((flag) => flag === "--hicache-storage-backend mooncake"),

  benchmarkCommands: {
    speed: `python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct: `# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  github: {
    cookbookModel: "inclusionAI/Ling-3.0-flash",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 4, 8] },
      ],
    },
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser ling3" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser ling3" },
      ],
    },
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off", label: "Off (greedy)" },
        { id: "nextn", label: "NEXTN (built-in MTP)", flags: ["--speculative-algorithm NEXTN"] },
        {
          id: "dspark",
          label: "DSPARK (draft model)",
          // --linear-replayssm-cache-len 32: the draft's block size 8 makes the
          // verify window 9 tokens, and the KDA ReplaySSM ring must be a power
          // of two >= 2x the window — the 16 default is too small and the
          // server refuses to start.
          flags: [
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path inclusionAI/Ling-3.0-flash-dspark",
            "--enable-linear-replayssm-spec",
            "--linear-replayssm-cache-len 32",
          ],
        },
      ],
    },
    hicache: {
      defaultBackend: "mooncake",
      requiredFlags: [
        "--mamba-scheduler-strategy extra_buffer",
        "--enable-cache-report",
      ],
      backends: [
        {
          id: "mooncake",
          label: "Mooncake",
          flags: [
            "--hicache-storage-backend-extra-config '{\"hicache_storage_pass_prefix_keys\":true}'",
          ],
          env: [
            "MOONCAKE_MASTER={{MOONCAKE_MASTER}}",
            "MOONCAKE_PROTOCOL=tcp",
            "MC_MS_AUTO_DISC=0",
            "MOONCAKE_DEVICE=",
            "MOONCAKE_TE_META_DATA_SERVER={{MOONCAKE_METADATA_SERVER}}",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE=0",
          ],
        },
      ],
    },
  },

  matchDims: [
    { id: "variant", title: "Model Variant", options: [{ id: "default", label: "Ling-3.0-flash" }] },
    {
      id: "quant",
      title: "Quantization",
      options: [
        { id: "bf16", label: "BF16" },
        { id: "fp8", label: "FP8" },
        { id: "int4", label: "INT4" },
        { id: "mxfp4", label: "MXFP4" },
      ],
    },
    {
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency", label: "Low-Latency" },
        { id: "high-throughput", label: "High-Throughput" },
        { id: "hicache", label: "HiCache + Mooncake" },
      ],
    },
    {
      id: "spec",
      title: "Spec Decode",
      options: [
        { id: "nextn", label: "NEXTN (built-in MTP)" },
        { id: "dspark", label: "DSPARK (draft model)" },
        { id: "off", label: "Off (greedy)" },
      ],
    },
    { id: "nodes", title: "Nodes", options: [{ id: "single", label: "Single Node" }] },
  ],

  // DSPARK twins of the low-latency cells: same shape, NEXTN swapped for the
  // external-draft path. --linear-replayssm-cache-len 32 because the draft's
  // block size 8 needs a power-of-two ring >= 2x the 9-token verify window.
  // Everything lives inside this IIFE because Mintlify's snippet compiler only
  // evaluates the exported expression — top-level module code is dropped.
  cells: (() => {
    const DSPARK_FLAGS = [
      "--speculative-algorithm DSPARK",
      "--speculative-draft-model-path inclusionAI/Ling-3.0-flash-dspark",
      "--enable-linear-replayssm-spec",
      "--linear-replayssm-cache-len 32",
    ];
    const dsparkTwin = (cell, verified) => ({
      ...cell,
      verified,
      match: { ...cell.match, spec: "dspark" },
      flags: cell.flags.flatMap((f) => (f === "--speculative-algorithm NEXTN" ? DSPARK_FLAGS : [f])),
    });
    const lowLatencyCells = [
    {
      match: { hw: "h20-3e", variant: "default", quant: "bf16", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "bf16", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "fp8", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "fp8", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", spec: "nextn", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
];
    return [
      ...lowLatencyCells.flatMap((c) => [
        c,
        dsparkTwin(c, c.match.hw === "b200" && c.match.quant === "bf16"),
      ]),
    {
      match: { hw: "h20-3e", variant: "default", quant: "bf16", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "bf16", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "fp8", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "fp8", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mem-fraction-static 0.85",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "int4", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mem-fraction-static 0.85",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "mxfp4", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_mxfp4",
        "--mem-fraction-static 0.85",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "mxfp4", strategy: "high-throughput", spec: "off", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_mxfp4",
        "--fp8-gemm-backend triton",
        "--mem-fraction-static 0.85",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // Hybrid KDA must pass prefix keys to Mooncake; otherwise storage writes are empty.
    // Cold uncached extends above chunked_prefill_size skip write-through for that influx.
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "hicache", spec: "nextn", nodes: "single" },
      verified: true,
      env: [
        "MOONCAKE_MASTER={{MOONCAKE_MASTER}}",
        "MOONCAKE_PROTOCOL=tcp",
        "MC_MS_AUTO_DISC=0",
        "MOONCAKE_DEVICE=",
        "MOONCAKE_TE_META_DATA_SERVER={{MOONCAKE_METADATA_SERVER}}",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE=0",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--enable-hierarchical-cache",
        "--hicache-storage-backend mooncake",
        "--hicache-io-backend direct",
        "--hicache-mem-layout page_first_direct",
        "--mamba-scheduler-strategy extra_buffer",
        "--enable-cache-report",
        "--hicache-storage-prefetch-policy wait_complete",
        "--hicache-storage-backend-extra-config '{\"hicache_storage_pass_prefix_keys\":true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "hicache", spec: "nextn", nodes: "single" },
      verified: false,
      env: [
        "MOONCAKE_MASTER={{MOONCAKE_MASTER}}",
        "MOONCAKE_PROTOCOL=tcp",
        "MC_MS_AUTO_DISC=0",
        "MOONCAKE_DEVICE=",
        "MOONCAKE_TE_META_DATA_SERVER={{MOONCAKE_METADATA_SERVER}}",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE=0",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--enable-hierarchical-cache",
        "--hicache-storage-backend mooncake",
        "--hicache-io-backend direct",
        "--hicache-mem-layout page_first_direct",
        "--mamba-scheduler-strategy extra_buffer",
        "--enable-cache-report",
        "--hicache-storage-prefetch-policy wait_complete",
        "--hicache-storage-backend-extra-config '{\"hicache_storage_pass_prefix_keys\":true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "hicache", spec: "nextn", nodes: "single" },
      verified: false,
      env: [
        "MOONCAKE_MASTER={{MOONCAKE_MASTER}}",
        "MOONCAKE_PROTOCOL=tcp",
        "MC_MS_AUTO_DISC=0",
        "MOONCAKE_DEVICE=",
        "MOONCAKE_TE_META_DATA_SERVER={{MOONCAKE_METADATA_SERVER}}",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE=0",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--enable-hierarchical-cache",
        "--hicache-storage-backend mooncake",
        "--hicache-io-backend direct",
        "--hicache-mem-layout page_first_direct",
        "--mamba-scheduler-strategy extra_buffer",
        "--enable-cache-report",
        "--hicache-storage-prefetch-policy wait_complete",
        "--hicache-storage-backend-extra-config '{\"hicache_storage_pass_prefix_keys\":true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "hicache", spec: "nextn", nodes: "single" },
      verified: false,
      env: [
        "MOONCAKE_MASTER={{MOONCAKE_MASTER}}",
        "MOONCAKE_PROTOCOL=tcp",
        "MC_MS_AUTO_DISC=0",
        "MOONCAKE_DEVICE=",
        "MOONCAKE_TE_META_DATA_SERVER={{MOONCAKE_METADATA_SERVER}}",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE=0",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--speculative-algorithm NEXTN",
        "--mem-fraction-static 0.8",
        "--enable-hierarchical-cache",
        "--hicache-storage-backend mooncake",
        "--hicache-io-backend direct",
        "--hicache-mem-layout page_first_direct",
        "--mamba-scheduler-strategy extra_buffer",
        "--enable-cache-report",
        "--hicache-storage-prefetch-policy wait_complete",
        "--hicache-storage-backend-extra-config '{\"hicache_storage_pass_prefix_keys\":true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    ];
  })(),
};
