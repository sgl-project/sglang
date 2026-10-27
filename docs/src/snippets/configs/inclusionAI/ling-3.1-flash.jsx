export const config = {
  modelName: "Ling-3.1-flash",
  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200", "gb300"],
  groupHardware: false,
  variants: [{ id: "default", label: "Ling-3.1-flash" }],
  quantizations: [{ id: "fp8", label: "FP8" }],
  strategies: [{ id: "low-latency", label: "Low-Latency" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],
  modelNames: { "default|fp8": "inclusionAI/Ling-3.1-flash" },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    MOONCAKE_MASTER: { target: "command", label: "Mooncake master", default: "127.0.0.1:50171" },
    MOONCAKE_METADATA_SERVER: { target: "command", label: "Mooncake metadata", default: "http://127.0.0.1:8290/metadata" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: "curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"model\":\"{{MODEL_NAME}}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 15% of 240?\"}]}'",
  dockerImages: { "h20-3e": "lmsysorg/sglang:dev-ling-3.1-flash", h200: "lmsysorg/sglang:dev-ling-3.1-flash", h800: "lmsysorg/sglang:dev-ling-3.1-flash", h100: "lmsysorg/sglang:dev-ling-3.1-flash", b200: "lmsysorg/sglang:dev-ling-3.1-flash", gb300: "lmsysorg/sglang:dev-ling-3.1-flash" },
  dockerHostNetworkWhen: (_sel, { flags }) =>
    flags.some((flag) => flag === "--hicache-storage-backend mooncake"),
  benchmarkCommands: {
    accuracy: { gsm8k_pct: "sgl-eval run gsm8k \\\n  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\\n  --num-threads 32" },
  },
  accuracyLabels: [["gsm8k_pct", "GSM8K (1319 questions)", "%"], ["stop_rate_pct", "Stop rate", "%"]],
  github: { cookbookModel: "inclusionAI/Ling-3.1-flash" },
  playgroundFeatures: {
    attention: {
      knobs: [
        {
          id: "tp",
          label: "TP",
          values: [
            null,
            {
              value: 4,
              disable: { hw: ["h200", "h20-3e", "h800", "h100"] },
              disableReason: "This FP8 model requires eight GPUs on 80 GB-class hardware for model capacity.",
            },
            { value: 8, disable: { hw: ["gb300"] }, disableReason: "This single-node GB300 recipe has four GPUs." },
          ],
        },
        {
          id: "cp",
          label: "CP",
          values: [
            null,
            {
              value: 2,
              disable: true,
              disableReason: "BailingKDA does not implement the sequence gather/state handling required for prefill context parallelism.",
            },
            {
              value: 4,
              disable: true,
              disableReason: "BailingKDA does not implement the sequence gather/state handling required for prefill context parallelism.",
            },
          ],
        },
        {
          id: "dpAttn",
          label: "DP-Attention",
          values: [
            null,
            false,
            1,
            2,
            4,
            {
              value: 8,
              disable: { hw: ["gb300"] },
              disableReason: "DP-Attention size must divide TP; the GB300 recipe uses TP4.",
            },
          ],
          labels: { auto: "Auto", false: "Off" },
        },
      ],
    },
    moe: {
      backend: {
        options: [
          { id: null, label: "Inherited" },
          { id: "auto", label: "Auto", flags: ["--moe-runner-backend auto"] },
          { id: "deep_gemm", label: "DeepGEMM", flags: ["--moe-runner-backend deep_gemm"] },
          { id: "triton", label: "Triton", flags: ["--moe-runner-backend triton"] },
        ],
      },
      ep: {
        label: "EP",
        values: [
          null,
          4,
          {
            value: 8,
            disable: { hw: ["gb300"] },
            disableReason: "EP cannot exceed the four TP ranks in the GB300 recipe.",
          },
        ],
      },
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
        { id: "off", label: "Off" },
        { id: "nextn", label: "NEXTN (built-in MTP)", flags: ["--speculative-algorithm NEXTN"] },
      ],
    },
    hicache: {
      defaultBackend: "mooncake",
      requiredFlags: ["--mamba-radix-cache-strategy extra_buffer", "--enable-cache-report"],
      backends: [
        {
          id: "mooncake",
          label: "Mooncake",
          flags: ["--hicache-storage-backend-extra-config '{\"hicache_storage_pass_prefix_keys\":true}'"],
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
    flagSelects: [
      {
        id: "prefixCache",
        title: "Multi-turn prefix cache",
        stripPrefixes: ["--mamba-radix-cache-strategy", "--chunked-prefill-size"],
        options: [
          { id: null, label: "Inherited from base" },
          {
            id: "split",
            label: "Cache fork and tail",
            flags: ["--mamba-radix-cache-strategy extra_buffer", "--chunked-prefill-size 4096"],
            env: ["SGLANG_SPLIT_PREFILL_AT_MAMBA_FORK=1"],
          },
        ],
      },
    ],
    pdDisagg: {
      modes: [{ id: "off", label: "Off" }, { id: "prefill", label: "Prefill role" }, { id: "decode", label: "Decode role" }],
      transferBackends: [{ id: "mooncake", label: "Mooncake" }],
      ibDevices: [{ id: "auto", label: "Auto" }],
    },
  },
  cells: [
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_ENABLE_FP8_BLOCK_GEMV=1", "SGLANG_ENABLE_FUSED_KDA_OUTPUT_QUANT=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--speculative-algorithm NEXTN",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ENABLE_FP8_BLOCK_GEMV=1", "SGLANG_ENABLE_FUSED_KDA_OUTPUT_QUANT=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--speculative-algorithm NEXTN",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ENABLE_FP8_BLOCK_GEMV=1", "SGLANG_ENABLE_FUSED_KDA_OUTPUT_QUANT=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--speculative-algorithm NEXTN",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: ["SGLANG_ENABLE_FP8_BLOCK_GEMV=1", "SGLANG_ENABLE_FUSED_KDA_OUTPUT_QUANT=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--speculative-algorithm NEXTN",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--speculative-algorithm NEXTN",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--speculative-algorithm NEXTN",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
