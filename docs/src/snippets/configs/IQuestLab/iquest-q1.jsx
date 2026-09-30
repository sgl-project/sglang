// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// Recipes follow the IQuestLab/IQuest-Q1 model card. The checkpoint ships the
// MTP draft in its mtp/ subdirectory, so both paths point at a local copy.
// IQuest Q1 requires FA3 attention, which SGLang builds for Hopper (SM90) and
// earlier only; Blackwell is not listed.
export const config = {
  modelName: "IQuest-Q1",
  supportedHardware: ["h200"],
  variants: [
    { id: "default", label: "Default", subtitle: "320B / 15B active · 8 GPUs" },
  ],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [
    { id: "low-latency", label: "Low-Latency" },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [{ id: "single", label: "Single Node" }],
  modelNames: {
    "default|bf16": "IQuest-Q1",
  },
  placeholders: {
    MODEL_PATH: { target: "command", label: "IQuest-Q1 checkpoint path", default: "/model/IQuest-Q1" },
    DRAFT_PATH: { target: "command", label: "MTP draft path", default: "/model/IQuest-Q1/mtp" },
    MODEL_ROOT: { target: "command", label: "Host model directory (Docker)", default: "/model" },
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"{{MODEL_NAME}}","messages":[{"role":"user","content":"What is 15% of 240?"}]}'`,
  dockerImages: { h200: "lmsysorg/sglang:dev" },
  dockerMounts: ["\"{{MODEL_ROOT}}:/model:ro\""],
  github: { cookbookModel: "IQuestLab/IQuest-Q1" },
  cells: [
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path '{{MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend fa3",
        "--mem-fraction-static 0.85",
        "--disable-prefill-cuda-graph",
        "--enable-torch-compile",
        "--reasoning-parser iquest_q1",
        "--tool-call-parser iquest_q1",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-draft-model-path '{{DRAFT_PATH}}'",
        "--speculative-draft-attention-backend fa3",
        "--speculative-use-rejection-sampling",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path '{{MODEL_PATH}}'",
        "--served-model-name {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend fa3",
        "--mem-fraction-static 0.85",
        "--disable-prefill-cuda-graph",
        "--enable-torch-compile",
        "--reasoning-parser iquest_q1",
        "--tool-call-parser iquest_q1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
