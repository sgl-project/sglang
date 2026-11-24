export const config = {
  modelName: "Ling-3.2-air",
  supportedHardware: ["b200", "gb300", "b300", "h200", "h20-3e", "h100", "h800"],

  // One measured or model-card layout per GPU; other TP sizes are Playground overrides.
  matchDims: [],

  overlayDims: [
    {
      id: "mtp",
      title: "Spec Decode",
      default: "off",
      options: [
        { id: "off", label: "Off" },
        { id: "nextn", label: "MTP (NEXTN)", flags: ["--speculative-algorithm NEXTN"] },
      ],
    },
  ],

  modelNames: { default: "inclusionAI/Ling-3.2-air" },

  dockerImages: {
    b200: "lmsysorg/sglang:ling-3.2-air",
    gb300: "lmsysorg/sglang:ling-3.2-air",
    b300: "lmsysorg/sglang:ling-3.2-air",
    h200: "lmsysorg/sglang:ling-3.2-air",
    "h20-3e": "lmsysorg/sglang:ling-3.2-air",
    h100: "lmsysorg/sglang:ling-3.2-air",
    h800: "lmsysorg/sglang:ling-3.2-air",
  },

  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"{{MODEL_NAME}}","messages":[{"role":"user","content":"What is 15% of 240?"}]}'`,

  github: { cookbookModel: "inclusionAI/Ling-3.2-air" },

  benchmarkCommands: {
    speed:
`python3 -m sglang.benchmark.serving \\
  --backend sglang-oai \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name random \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --temperature 1.0 --top-p 0.95 --max-tokens 65536`,
    },
    numPromptsByConc: { 1: 32, 16: 64 },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K (1319 questions)", "%"],
    ["stop_rate_pct", "Stop rate", "%"],
  ],

  playgroundFeatures: {
    attention: {
      knobs: [
        {
          id: "tp",
          label: "TP",
          values: [
            null,
            {
              value: 2,
              disable: { hw: ["h100", "h800"] },
              disableReason: "At TP2 the FP8 weights take about 60 GB per GPU, which leaves too little memory on an 80 GB GPU.",
            },
            4,
            { value: 8, disable: { hw: ["gb300"] }, disableReason: "A GB300 node has four GPUs." },
          ],
        },
      ],
    },
  },

  cells: [
    {
      match: { hw: "b200" },
      nnodes: 1,
      verificationStatus: "verified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 2",
        "--trust-remote-code",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300" },
      nnodes: 1,
      verificationStatus: "verified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--trust-remote-code",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300" },
      nnodes: 1,
      verificationStatus: "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--trust-remote-code",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200" },
      nnodes: 1,
      verificationStatus: "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--trust-remote-code",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h20-3e" },
      nnodes: 1,
      verificationStatus: "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--trust-remote-code",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100" },
      nnodes: 1,
      verificationStatus: "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--trust-remote-code",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800" },
      nnodes: 1,
      verificationStatus: "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--trust-remote-code",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
