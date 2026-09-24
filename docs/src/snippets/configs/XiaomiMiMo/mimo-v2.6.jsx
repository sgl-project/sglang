// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// Recipes: https://github.com/sgl-project/sglang/pull/40448
// SGLang: 983e643854f15cf9ef4370a49dfd74b6af54c3e3.
// B300 validation reported by the model team. H200 cells reuse the B300
// recipes with `--moe-runner-backend marlin` in place of deep_gemm. Public V2.6 checkpoints are
// XiaomiMiMo/MiMo-V2.6-{Flash,Pro}-RL (MXFP4 experts, bf16 router, bundled
// dflash/ drafter); modelNames are served aliases, and the checkpoint paths
// stay editable so a local copy can be used instead.
export const config = {
  modelName: "MiMo-V2.6",
  supportedHardware: ["h200", "b200", "b300", "gb300"],
  variants: [
    { id: "flash", label: "Flash", subtitle: "309B / 15B active · 4 GPUs" },
    { id: "pro", label: "Pro", subtitle: "1.02T / 42B active · 8 GPUs" },
  ],
  quantizations: [{ id: "mxfp4", label: "MXFP4" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [
    { id: "single", label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes", showWhen: (s) => s.hw === "gb300" && s.variant === "pro" },
  ],
  modelNames: {
    "flash|mxfp4": "mimo-v2.6-flash",
    "pro|mxfp4": "mimo-v2.6-pro",
  },
  placeholders: {
    FLASH_MODEL_PATH: { target: "command", label: "Flash checkpoint path", default: "/model/MiMo-V2.6-Flash" },
    FLASH_DRAFT_PATH: { target: "command", label: "Flash DFlash checkpoint path", default: "/model/MiMo-V2.6-Flash/dflash" },
    PRO_MODEL_PATH: { target: "command", label: "Pro checkpoint path", default: "/model/MiMo-V2.6-Pro" },
    PRO_DRAFT_PATH: { target: "command", label: "Pro DFlash checkpoint path", default: "/model/MiMo-V2.6-Pro/dflash" },
    MODEL_ROOT: { target: "command", label: "Host model directory (Docker)", default: "/model" },
    NODE0_IP: { target: "command", label: "Head node IP", default: "<node0-ip>" },
    NODE_RANK: { target: "command", label: "This node rank", default: "<node-rank>" },
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"{{MODEL_NAME}}","messages":[{"role":"user","content":"What is 15% of 240?"}],"chat_template_kwargs":{"enable_thinking":true}}'`,
  // The MiMo-V2.6 support (PR #40448) is on main, so the nightly tag carries it.
  dockerImages: {
    h200: "lmsysorg/sglang:dev",
    b200: "lmsysorg/sglang:dev",
    b300: "lmsysorg/sglang:dev",
    gb300: "lmsysorg/sglang:dev",
  },
  dockerMounts: ["\"{{MODEL_ROOT}}:/model:ro\""],
  github: { cookbookModel: "MiMo-V2.6 (Flash / Pro)" },
  playgroundFeatures: {
    // Keep the validated TP/EP topology. DFlash on CUDA rejects DP-attention.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser mimo" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser mimo" },
      ],
    },
    speculative: {
      options: [
        { id: "current", label: "Inherited DFlash", note: "Use the matching DFlash checkpoint for the selected model. The base recipe drafts 8 tokens." },
        { id: "off", label: "Off", note: "Disabling DFlash is an unverified override; remeasure latency and throughput for your workload." },
      ],
    },
  },
  cells: [
    {
      match: { hw: "b300", variant: "flash", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path '{{FLASH_MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--moe-runner-backend deep_gemm",
        "--trust-remote-code",
        "--reasoning-parser mimo",
        "--tool-call-parser mimo",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "pro", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path '{{PRO_MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--moe-runner-backend deep_gemm",
        "--trust-remote-code",
        "--reasoning-parser mimo",
        "--tool-call-parser mimo",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path '{{FLASH_MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend marlin",
        "--trust-remote-code",
        "--reasoning-parser mimo",
        "--tool-call-parser mimo",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "pro", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path '{{PRO_MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend marlin",
        "--trust-remote-code",
        "--reasoning-parser mimo",
        "--tool-call-parser mimo",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path '{{FLASH_MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--moe-runner-backend deep_gemm",
        "--trust-remote-code",
        "--reasoning-parser mimo",
        "--tool-call-parser mimo",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "pro", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path '{{PRO_MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--moe-runner-backend deep_gemm",
        "--trust-remote-code",
        "--reasoning-parser mimo",
        "--tool-call-parser mimo",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path '{{FLASH_MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--moe-runner-backend deep_gemm",
        "--trust-remote-code",
        "--reasoning-parser mimo",
        "--tool-call-parser mimo",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "pro", quant: "mxfp4", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path '{{PRO_MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--moe-runner-backend deep_gemm",
        "--trust-remote-code",
        "--reasoning-parser mimo",
        "--tool-call-parser mimo",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
