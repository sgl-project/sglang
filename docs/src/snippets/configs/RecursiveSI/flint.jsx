export const config = {
  modelName: "Flint",

  supportedHardware: ["h200", "b200", "b300", "gb300", "dgx-spark"],

  // A single hidden dim: the engine renders nothing when matchDims is empty.
  matchDims: [
    { id: "variant", title: "Model Variant", showWhen: () => false, options: [{ id: "default", label: "Default" }] },
  ],

  modelNames: {
    default: "RecursiveSI/flint",
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
-d '{ "model": "flint", "messages": [{"role":"user","content":"Hello"}] }'`,

  showPlaygroundLink: false,

  dockerImages: {
    h200:        "lmsysorg/sglang:dev-flint",
    b200:        "lmsysorg/sglang:dev-flint",
    b300:        "lmsysorg/sglang:dev-flint",
    gb300:       "lmsysorg/sglang:dev-flint",
    "dgx-spark": "lmsysorg/sglang:dev-flint",
  },

  github: {
    cookbookModel: "RecursiveSI/flint",
  },

  cells: [
    {
      match: { hw: "h200", variant: "default" },
      verified: true,
      nnodes: 1,
      env: [],
      // No backend flags: FA4 and the trtllm MoE are Blackwell-only, and on Hopper Flint resolves to FA3 and Triton MoE.
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name flint",
        "--tp-size 1",
        "--enable-fp32-lm-head",
        "--context-length 262144",
        "--chunked-prefill-size 4096",
        "--enable-unified-memory",
        "--mem-fraction-static 0.8",
        "--tool-call-parser flint",
        "--reasoning-parser flint",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default" },
      verified: true,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name flint",
        "--tp-size 1",
        "--attention-backend fa4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--enable-fp32-lm-head",
        "--context-length 262144",
        "--chunked-prefill-size 4096",
        "--swa-full-tokens-ratio 0.05",
        "--mem-fraction-static 0.8",
        "--tool-call-parser flint",
        "--reasoning-parser flint",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default" },
      verified: true,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name flint",
        "--tp-size 1",
        "--attention-backend fa4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--enable-fp32-lm-head",
        "--context-length 262144",
        "--chunked-prefill-size 4096",
        "--swa-full-tokens-ratio 0.05",
        "--mem-fraction-static 0.8",
        "--tool-call-parser flint",
        "--reasoning-parser flint",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default" },
      verified: true,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name flint",
        "--tp-size 1",
        "--attention-backend fa4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--enable-fp32-lm-head",
        "--context-length 262144",
        "--chunked-prefill-size 4096",
        "--swa-full-tokens-ratio 0.05",
        "--mem-fraction-static 0.8",
        "--tool-call-parser flint",
        "--reasoning-parser flint",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default" },
      verified: true,
      nnodes: 1,
      env: [],
      // No backend flags: on GB10 Flint resolves to Triton attention and Triton MoE.
      // GB10 counts page cache as used GPU memory, so evict each weight shard after loading it.
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name flint",
        "--tp-size 1",
        "--enable-fp32-lm-head",
        "--context-length 262144",
        "--chunked-prefill-size 4096",
        "--enable-unified-memory",
        "--mem-fraction-static 0.8",
        "--weight-loader-drop-cache-after-load",
        "--tool-call-parser flint",
        "--reasoning-parser flint",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
