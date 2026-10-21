export const config = {
  modelName: "Step 5 Preview",

  supportedHardware: ["b300", "h200"],

  hardware: [
    { id: "b300", label: "Blackwell", vendor: "blackwell" },
    { id: "h200", label: "Hopper", vendor: "hopper" },
  ],

  groupHardware: false,

  matchDims: [
    {
      id: "parallel",
      title: "Parallelism",
      options: [
        { id: "tp4", label: "TP4", showWhen: (s) => s.hw === "b300" },
        { id: "ep4", label: "EP4", showWhen: (s) => s.hw === "b300" },
        { id: "ep8", label: "EP8", showWhen: (s) => s.hw === "h200" },
        { id: "dp2", label: "ATTN DP2" },
      ],
    },
  ],

  overlayDims: [
    {
      id: "quant",
      title: "Model",
      default: "fp8",
      options: [
        { id: "fp8", label: "FP8", subtitle: "Step-5-Preview" },
        { id: "nvfp4", label: "NVFP4", subtitle: "Step-5-Preview-NVFP4",
          disabled: (s) => s.hw === "h200",
          disableReason: "The NVFP4 checkpoint needs Blackwell: its routed experts run on FlashInfer FP4 MoE kernels." },
      ],
    },
    {
      id: "moe",
      title: "MoE",
      default: "trtllm",
      options: [
        { id: "trtllm", label: "FlashInfer TRT-LLM", showWhen: (s) => s.hw === "b300" },
        { id: "megamoe", label: "MegaMoE", showWhen: (s) => s.hw === "b300",
          flags: ["--moe-a2a-backend megamoe"],
          disabled: (s) => s.quant === "nvfp4",
          disableReason: "MegaMoE supports the FP8 checkpoint only. Pick FP8 in the Model row to use MegaMoE." },
        { id: "triton", label: "Triton", showWhen: (s) => s.hw === "h200" },
      ],
    },
    {
      id: "hicache",
      title: "HiCache",
      default: "off",
      options: [
        { id: "off", label: "Off" },
        { id: "on", label: "On (host memory)",
          flags: (s) => (s.hw === "b300" && s.parallel === "dp2"
            ? ["--enable-hierarchical-cache", "--hicache-ratio 1.5"]
            : ["--enable-hierarchical-cache", "--hicache-ratio 3.5"]),
          hints: ["Size --hicache-ratio to the host memory available per rank."] },
      ],
    },
    {
      id: "kvDtype",
      title: "KV Cache",
      default: "bf16",
      options: [
        { id: "bf16", label: "BF16",
          stripPrefixes: ["--kv-cache-dtype"], flags: ["--kv-cache-dtype bfloat16"] },
        { id: "fp8", label: "FP8",
          stripPrefixes: ["--kv-cache-dtype"], flags: ["--kv-cache-dtype fp8_e4m3"],
          soft: true,
          softReason: "FP8 KV applies to full-attention layers only and is lossy; re-verify accuracy on your workload." },
      ],
    },
    {
      id: "spec",
      title: "Spec Decode",
      default: "mtp",
      options: [
        { id: "mtp", label: "MTP" },
        { id: "off", label: "Off",
          stripPrefixes: ["--speculative-algorithm", "--speculative-num-steps", "--speculative-eagle-topk",
                          "--speculative-num-draft-tokens", "--enable-multi-layer-eagle"] },
      ],
    },
  ],

  modelNames: {
    default: "stepfun-ai/Step-5-Preview",
    nvfp4: "stepfun-ai/Step-5-Preview-NVFP4",
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
    b300: "lmsysorg/sglang:step-5-preview",
    h200: "lmsysorg/sglang:step-5-preview",
  },

  github: {
    cookbookModel: "stepfun-ai/Step-5-Preview",
  },

  playgroundFeatures: {

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser step5" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser step5" },
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
  },

  cells: [
    {
      match: { hw: "b300", parallel: "tp4" },
      nnodes: 1,
      verificationStatus: "verified",
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
      match: { hw: "b300", parallel: "ep4" },
      nnodes: 1,
      verificationStatus: "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--moe-runner-backend flashinfer_trtllm",
        "--attention-backend fa4",
        "--page-size 64",
        "--kv-cache-dtype bfloat16",
        "--mem-fraction-static 0.9",
        "--swa-full-tokens-ratio 0.03",
        "--chunked-prefill-size 16384",
        "--max-running-requests 64",
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
      match: { hw: "b300", parallel: "dp2" },
      nnodes: 1,
      verificationStatus: "verified",
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
      match: { hw: "h200", parallel: "ep8" },
      nnodes: 1,
      verificationStatus: "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--moe-runner-backend triton",
        "--attention-backend fa3",
        "--page-size 64",
        "--kv-cache-dtype bfloat16",
        "--mem-fraction-static 0.9",
        "--swa-full-tokens-ratio 0.03",
        "--chunked-prefill-size 16384",
        "--max-running-requests 64",
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
      match: { hw: "h200", parallel: "dp2" },
      nnodes: 1,
      verificationStatus: "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attn-dp-size 2",
        "--ep-size 8",
        "--moe-runner-backend triton",
        "--attention-backend fa3",
        "--page-size 64",
        "--kv-cache-dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--swa-full-tokens-ratio 0.03",
        "--chunked-prefill-size 16384",
        "--max-running-requests 64",
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
