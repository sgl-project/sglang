export const config = {
  modelName: "Step-5-Preview",

  supportedHardware: ["b300", "h200"],

  variants: [
    { id: "default", label: "Default" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
  ],
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "balanced",        label: "Balanced"        },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|fp8": "stepfun-ai/Step-5-Preview",
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

  dockerImages: {
    b300: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "stepfun-ai/Step-5-Preview",
  },

  playgroundFeatures: {

    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 4, 8] },
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null, false,
            { value: 2, disable: { hw: ["h200"] },
              disableReason: "DP-Attention has only been validated on Blackwell for Step-5-Preview." },
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    moe: {
      backend: {
        options: [
          { id: null, label: "Inherited" },
          { id: "megamoe", label: "MegaMoE", flags: ["--moe-a2a-backend megamoe"],
            requiresHw: ["b300"] },
          { id: "flashinfer_trtllm", label: "FlashInfer TRT-LLM", flags: ["--moe-runner-backend flashinfer_trtllm"],
            requiresHw: ["b300"] },
          { id: "triton", label: "Triton", flags: ["--moe-runner-backend triton"] },
        ],
      },
      ep: { label: "EP", values: [null, 1, 4, 8] },
    },

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser step5" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser step5" },
      ],
    },

    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4",
                  "--enable-multi-layer-eagle"] },
      ],
    },

    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl", label: "NiXL" },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
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

    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
      ],
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },

    flagSelects: [
      {
        id: "kvCacheDtype", title: "KV Cache Precision",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "bf16", label: "BF16", flags: ["--kv-cache-dtype bfloat16"] },
          { id: "fp8", label: "FP8 (E4M3) — full-attention layers only", flags: ["--kv-cache-dtype fp8_e4m3"],
            disable: { hw: ["h200"] },
            disableReason: "The SM90 sparse decode kernel only supports BF16 KV cache." },
        ],
      },
    ],
  },

  cells: [
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm",
        "--attention-backend fa4",
        "--page-size 64",
        "--kv-cache-dtype bfloat16",
        "--mem-fraction-static 0.9",
        "--swa-full-tokens-ratio 0.03",
        "--chunked-prefill-size 16384",
        "--max-running-requests 64",
        "--cuda-graph-backend-prefill breakable",
        "--flashinfer-allreduce-fusion-backend trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--enable-multi-layer-eagle",
        "--reasoning-parser step5",
        "--tool-call-parser step5",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-a2a-backend megamoe",
        "--attention-backend fa4",
        "--page-size 64",
        "--kv-cache-dtype bfloat16",
        "--mem-fraction-static 0.9",
        "--swa-full-tokens-ratio 0.03",
        "--chunked-prefill-size 16384",
        "--max-running-requests 64",
        "--cuda-graph-backend-prefill breakable",
        "--flashinfer-allreduce-fusion-backend trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--enable-multi-layer-eagle",
        "--reasoning-parser step5",
        "--tool-call-parser step5",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attn-dp-size 2",
        "--ep-size 8",
        "--moe-runner-backend flashinfer_trtllm",
        "--attention-backend fa4",
        "--page-size 64",
        "--kv-cache-dtype bfloat16",
        "--mem-fraction-static 0.9",
        "--swa-full-tokens-ratio 0.03",
        "--chunked-prefill-size 16384",
        "--max-running-requests 64",
        "--cuda-graph-backend-prefill disabled",
        "--flashinfer-allreduce-fusion-backend trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--enable-multi-layer-eagle",
        "--reasoning-parser step5",
        "--tool-call-parser step5",
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
        "--tp 8",
        "--ep-size 8",
        "--moe-runner-backend triton",
        "--attention-backend fa3",
        "--page-size 64",
        "--mem-fraction-static 0.9",
        "--swa-full-tokens-ratio 0.03",
        "--chunked-prefill-size 16384",
        "--max-running-requests 64",
        "--cuda-graph-backend-prefill breakable",
        "--flashinfer-allreduce-fusion-backend trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--enable-multi-layer-eagle",
        "--reasoning-parser step5",
        "--tool-call-parser step5",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
