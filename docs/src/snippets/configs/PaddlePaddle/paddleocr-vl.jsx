// PaddleOCR-VL cookbook config. Consumed by _deployment.jsx + _playground.jsx.
//
// All three releases (0.9B / 1.5 / 1.6) ship an identical `config.json` — same
// PaddleOCRVLForConditionalGeneration architecture, same SigLIP tower and
// ERNIE-4.5-0.3B backbone — so one recipe serves every variant and only the HF
// slug changes.

export const config = {
  modelName: "PaddleOCR-VL",

  supportedHardware: ["h100", "h200", "b200"],

  variants: [
    { id: "v16", label: "1.6", subtitle: "Latest" },
    { id: "v15", label: "1.5" },
    { id: "v09", label: "0.9B", subtitle: "Original" },
  ],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "v16|bf16": "PaddlePaddle/PaddleOCR-VL-1.6",
    "v15|bf16": "PaddlePaddle/PaddleOCR-VL-1.5",
    "v09|bf16": "PaddlePaddle/PaddleOCR-VL",
  },

  // Page resolution is the dominant cost knob: the ViT and the prefill both
  // scale with the patch count, and `max_pixels` is expressed in 28x28 units
  // (patch 14 x 2x2 merge), so the value divided by 784 is the image-token
  // budget per page. 1280 is the checkpoint's own preprocessor default.
  overlayDims: [
    {
      id: "pageRes",
      title: "Page Resolution",
      default: "default",
      options: [
        {
          id: "fast",
          label: "Fast (768 tok)",
          flags: [
            "--mm-process-config '{\"image\": {\"max_pixels\": 602112}}'",
          ],
        },
        { id: "default", label: "Default (1280 tok)", flags: [] },
        {
          id: "detail",
          label: "High detail (2048 tok)",
          flags: [
            "--mm-process-config '{\"image\": {\"max_pixels\": 1605632}}'",
          ],
        },
      ],
    },
  ],

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
      {"type": "text", "text": "OCR:"},
      {"type": "image_url", "image_url": {"url": "https://example.com/your_document.png"}}
    ]
  }],
  "temperature": 0,
  "max_tokens": 2048
}'`,

  dockerImages: {
    h100: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
    b200: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "PaddlePaddle/PaddleOCR-VL",
  },

  playgroundFeatures: {
    attention: {
      knobs: [{ id: "tp", label: "TP", values: [null, 1, 2, 4] }],
    },
  },

  cells: [
    // ==== 1.6 ====
    {
      match: {
        hw: "h100",
        variant: "v16",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "h200",
        variant: "v16",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "b200",
        variant: "v16",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== 1.5 ====
    {
      match: {
        hw: "h100",
        variant: "v15",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "h200",
        variant: "v15",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "b200",
        variant: "v15",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ==== 0.9B ====
    {
      match: {
        hw: "h100",
        variant: "v09",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "h200",
        variant: "v09",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "b200",
        variant: "v09",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--context-length 16384",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 32768",
        "--enable-mixed-chunk",
        "--num-continuous-decode-steps 2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
