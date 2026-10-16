// RTX Spark recipe reported by a user; instantiated from cookbook-add-model.
// Keep this scoped recipe separate from the legacy Qwen3.6 deployment selector.
export const config = {
  modelName: "Qwen3.6-35B-A3B-NVFP4",
  showPlaygroundLink: false,
  supportedHardware: ["rtx-spark"],
  hardware: [{ id: "rtx-spark", label: "RTX Spark", vram: "Up to 128GB unified", vendor: "blackwell" }],
  variants: [{ id: "default", label: "35B-A3B" }],
  quantizations: [{ id: "nvfp4", label: "NVFP4" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],
  modelNames: {
    "default|nvfp4": "nvidia/Qwen3.6-35B-A3B-NVFP4",
  },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions -H 'Content-Type: application/json' -d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,
  // Nightly fallback, not the image used for the reported measurement.
  dockerImages: { "rtx-spark": "lmsysorg/sglang:dev" },
  github: { cookbookModel: "nvidia/Qwen3.6-35B-A3B-NVFP4" },
  cells: [
    {
      match: { hw: "rtx-spark", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--mamba-radix-cache-strategy extra_buffer",
        "--mem-fraction-static 0.8",
        "--moe-runner-backend flashinfer_cutlass",
      ],
    },
  ],
};
