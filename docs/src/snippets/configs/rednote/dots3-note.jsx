// Dots3-Note cookbook config. Consumed by _deployment.jsx + _playground.jsx.
// Single `export const config` literal - no spreads/calls/IIFE (Mintlify re-evals at hydration).

export const config = {
  modelName: "Dots3-Note",

  // No Playground on this page — the only extra knob (the dots tool-call parser)
  // is already baked into the cells.
  showPlaygroundLink: false,

  // Hopper only for now — no Blackwell support.
  supportedHardware: ["h200"],

  // One model and one node shape — only the checkpoint precision is a real choice.
  matchDims: [
    {
      id: "quant",
      title: "Checkpoint Precision",
      options: [
        { id: "bf16", label: "BF16" },
        { id: "fp8", label: "FP8" },
      ],
    },
  ],

  modelNames: {
    // TODO: replace with the public repo id once the checkpoint is released.
    default: "<dots-note-checkpoint>",
  },

  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: {
      target: "command",
      label: "HF token (Docker)",
      default: "<your-hf-token>",
    },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{
  "model": "{{MODEL_NAME}}",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "video_url", "video_url": {"url": "https://example.com/sample.mp4"}},
      {"type": "text", "text": "Summarize what happens in this video."}
    ]
  }]
}'`,

  dockerImages: {
    h200: "lmsysorg/sglang:dev",
  },


  cells: [
    {
      match: { hw: "h200", quant: "bf16" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1",
        "SGLANG_ENABLE_JIT_DEEPGEMM=1",
        "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=8192",
        "SGLANG_MAX_KV_CHUNK_CAPACITY=8192",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
        "SGLANG_WARMUP_TIMEOUT=1800",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 524288",
        "--enable-dp-attention",
        "--dp-size 8",
        "--tp-size 8",
        "--ep-size 8",
        "--mem-fraction-static 0.87",
        "--max-running-requests 256",
        "--chunked-prefill-size 16384",
        "--trust-remote-code",
        "--swa-full-tokens-ratio 0.03",
        "--prefill-attention-backend fa3",
        "--decode-attention-backend fa3",
        "--page-size 64",
        "--moe-dense-tp-size 1",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-backend-prefill disabled",
        "--cuda-graph-max-bs-decode 32",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-draft-attention-backend fa3",
        "--moe-a2a-backend deepep",
        "--moe-runner-backend deep_gemm",
        "--deepep-dispatcher-output-dtype bf16",
        "--deepep-mode auto",
        "--enable-nccl-nvls",
        "--enable-multimodal",
        "--enable-metrics",
        "--tool-call-parser dots",
        "--reasoning-parser qwen3",
        "--watchdog-timeout 1800",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", quant: "fp8" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1",
        "SGLANG_ENABLE_JIT_DEEPGEMM=1",
        "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=8192",
        "SGLANG_MAX_KV_CHUNK_CAPACITY=8192",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
        "SGLANG_WARMUP_TIMEOUT=1800",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 524288",
        "--enable-dp-attention",
        "--dp-size 8",
        "--tp-size 8",
        "--ep-size 8",
        "--mem-fraction-static 0.87",
        "--max-running-requests 256",
        "--chunked-prefill-size 16384",
        "--trust-remote-code",
        "--swa-full-tokens-ratio 0.03",
        "--prefill-attention-backend fa3",
        "--decode-attention-backend fa3",
        "--page-size 64",
        "--moe-dense-tp-size 1",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-backend-prefill disabled",
        "--cuda-graph-max-bs-decode 32",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-draft-attention-backend fa3",
        "--moe-a2a-backend deepep",
        "--moe-runner-backend auto",
        "--deepep-dispatcher-output-dtype auto",
        "--deepep-mode auto",
        "--enable-nccl-nvls",
        "--enable-multimodal",
        "--enable-metrics",
        "--reasoning-parser qwen3",
        "--tool-call-parser dots",
        "--watchdog-timeout 1800",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
