// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// BF16, TP=1 recipes tested on B200 and H200 at PR #42721 / 4704c2424a81.
// Each model passed text and image requests, including 16,384-token prompts.
export const config = {
  modelName: "Clef",
  showPlaygroundLink: false,
  supportedHardware: ["h200", "b200"],
  variants: [
    { id: "clef", label: "Clef", subtitle: "27B dense, 1 GPU" },
    { id: "flash", label: "Clef Flash", subtitle: "9B dense, 1 GPU" },
  ],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],
  modelNames: {
    "clef|bf16": "Cloudflare/clef",
    "flash|bf16": "Cloudflare/clef-flash",
  },
  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },
  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/systemone \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"{{MODEL_NAME}}","state":"My Stripe integration keeps failing. Please help ASAP.","questions":{"urgency":{"type":"noul","instructions":"Does this message express urgency?"}}}'`,
  dockerImages: { h200: "lmsysorg/sglang:dev", b200: "lmsysorg/sglang:dev" },
  github: { cookbookModel: "Cloudflare/clef" },
  cells: [
    {
      match: { hw: "h200", variant: "clef", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name {{MODEL_NAME}}",
        "--tp-size 1",
        "--dtype bfloat16",
        "--context-length 32768",
        "--max-total-tokens 32768",
        "--max-running-requests 1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name {{MODEL_NAME}}",
        "--tp-size 1",
        "--dtype bfloat16",
        "--context-length 32768",
        "--max-total-tokens 32768",
        "--max-running-requests 1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "clef", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name {{MODEL_NAME}}",
        "--tp-size 1",
        "--dtype bfloat16",
        "--context-length 32768",
        "--max-total-tokens 32768",
        "--max-running-requests 1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--served-model-name {{MODEL_NAME}}",
        "--tp-size 1",
        "--dtype bfloat16",
        "--context-length 32768",
        "--max-total-tokens 32768",
        "--max-running-requests 1",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
}
