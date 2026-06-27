// Unlimited-OCR cookbook config. Consumed by _deployment.jsx + _playground.jsx.

export const config = {
  modelName: "Unlimited-OCR",

  supportedHardware: ["h100", "h200", "b200", "b300", "gb200", "gb300"],

  variants: [{ id: "default", label: "Default" }],
  quantizations: [{ id: "default", label: "Default" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|default": "baidu/Unlimited-OCR",
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
      {"type": "text", "text": "document parsing."},
      {"type": "image_url", "image_url": {"url": "https://example.com/your_document.png"}}
    ]
  }],
  "images_config": {"image_mode": "gundam"},
  "temperature": 0,
  "max_tokens": 2048
}'`,

  dockerImages: {
    h100: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
    b200: "lmsysorg/sglang:dev",
    b300: "lmsysorg/sglang:dev",
    gb200: "lmsysorg/sglang:dev",
    gb300: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "baidu/Unlimited-OCR",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },
  },

  cells: [
    {
      match: {
        hw: "h100",
        variant: "default",
        quant: "default",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend fa3",
        "--page-size 1",
        "--context-length 32768",
        "--enable-custom-logit-processor",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "h200",
        variant: "default",
        quant: "default",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend fa3",
        "--page-size 1",
        "--context-length 32768",
        "--enable-custom-logit-processor",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "b200",
        variant: "default",
        quant: "default",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend fa3",
        "--page-size 1",
        "--context-length 32768",
        "--enable-custom-logit-processor",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "b300",
        variant: "default",
        quant: "default",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend fa3",
        "--page-size 1",
        "--context-length 32768",
        "--enable-custom-logit-processor",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "gb200",
        variant: "default",
        quant: "default",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend fa3",
        "--page-size 1",
        "--context-length 32768",
        "--enable-custom-logit-processor",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "gb300",
        variant: "default",
        quant: "default",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--attention-backend fa3",
        "--page-size 1",
        "--context-length 32768",
        "--enable-custom-logit-processor",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
