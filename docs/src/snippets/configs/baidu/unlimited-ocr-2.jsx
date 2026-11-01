// Unlimited-OCR-2 cookbook config. Consumed by _deployment.jsx + _playground.jsx.

export const config = {
  modelName: "Unlimited-OCR-2",

  supportedHardware: ["l20"],

  // L20 is not in the shared HARDWARE_CATALOG.
  hardware: [
    { id: "l20", label: "L20", vram: "48GB", vendor: "nvidia" },
  ],

  variants: [{ id: "default", label: "Default" }],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|bf16": "baidu/Unlimited-OCR-2",
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
      {"type": "image_url", "image_url": {"url": "https://example.com/page_1.png"}},
      {"type": "image_url", "image_url": {"url": "https://example.com/page_2.png"}},
      {"type": "text", "text": "<PAGE><image><PAGE><image>Multi page parsing."}
    ]
  }],
  "images_config": {"image_mode": "base"},
  "skip_special_tokens": false,
  "max_tokens": 16000
}'`,

  dockerImages: {
    l20: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "baidu/Unlimited-OCR-2",
  },

  // Speculative decoding, PD disaggregation and pipeline parallelism are
  // rejected at launch; the radix cache is always off, so HiCache has no use.
  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2] },
      ],
    },
  },

  cells: [
    {
      match: {
        hw: "l20",
        variant: "default",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--chat-template unlimited-ocr",
        "--uocr-doc-split",
        "--enable-streaming-session",
        "--uocr-max-page-tokens 8000",
        "--cuda-graph-backend-decode full",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
