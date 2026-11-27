export const config = {
  modelName: "NVIDIA Nemotron 3.5 Decision 8B",
  supportedHardware: ["h100", "b200", "gb300", "rtx-pro-6000", "a100", "dgx-spark"],
  hardware: [
    { id: "a100", label: "A100", vram: "80GB", vendor: "nvidia" },
    { id: "rtx-pro-6000", label: "RTX PRO 6000 Blackwell Server Edition", vram: "96GB", vendor: "blackwell" },
  ],
  variants: [{ id: "default", label: "8B", subtitle: "1 GPU" }],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [{ id: "balanced", label: "Default" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],
  modelNames: {
    "default|bf16": "nvidia/NVIDIA-Nemotron-3.5-Decision-8B-BF16",
  },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/systemone \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"{{MODEL_NAME}}","state":{"ticket":"The service is down."},"questions":{"team":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":{"billing":"Payments and invoices","operations":"Service outages"}}}}'`,
  github: { cookbookModel: "nvidia/NVIDIA-Nemotron-3.5-Decision-8B-BF16" },
  dockerImages: {
    h100: "lmsysorg/sglang:dev",
    b200: "lmsysorg/sglang:dev",
    gb300: "lmsysorg/sglang:dev",
    "rtx-pro-6000": "lmsysorg/sglang:dev",
    a100: "lmsysorg/sglang:dev",
    "dgx-spark": "lmsysorg/sglang:dev",
  },
  playgroundFeatures: {
    attention: {
      knobs: [{ id: "tp", label: "TP", values: [null, 1, 2] }],
    },
  },
  cells: [
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: ["--model-path {{MODEL_NAME}}", "--host {{HOST_IP}}", "--port {{PORT}}"],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: ["--model-path {{MODEL_NAME}}", "--host {{HOST_IP}}", "--port {{PORT}}"],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: ["--model-path {{MODEL_NAME}}", "--host {{HOST_IP}}", "--port {{PORT}}"],
    },
    {
      match: { hw: "rtx-pro-6000", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: ["--model-path {{MODEL_NAME}}", "--host {{HOST_IP}}", "--port {{PORT}}"],
    },
    {
      match: { hw: "a100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: ["--model-path {{MODEL_NAME}}", "--host {{HOST_IP}}", "--port {{PORT}}"],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: ["--model-path {{MODEL_NAME}}", "--host {{HOST_IP}}", "--port {{PORT}}"],
    },
  ],
}
