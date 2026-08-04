export const config = {
  modelName: "Nemotron 3.5 Lightning",

  // Only 1x H100 is validated today (the NVFP4 W4A4 checkpoint recipe).
  supportedHardware: ["h100"],

  variants: [{ id: "default", label: "Default" }],

  quantizations: [{ id: "fp4", label: "NVFP4" }],

  strategies: [{ id: "balanced", label: "Balanced" }],

  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|fp4": "nvidia/nemotron-nano-3.5-ea2-W4A4-PTQ-20260723",
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

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  accuracyLabels: [["gsm8k_pct", "GSM8K", "%"]],

  dockerImages: {
    h100: "lmsysorg/sglang:nightly-dev-20260719-99f5a6f4",
  },

  github: {
    cookbookModel: "nvidia/nemotron-nano-3.5-ea2-W4A4-PTQ-20260723",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },

    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "marlin", label: "Marlin (W4A16)", flags: ["--moe-runner-backend marlin"] },
          { id: "deepep", label: "DeepEP",         flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser nemotron_3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
      ],
    },
  },

  cells: [
    // ==== NVIDIA Hopper + NVFP4 W4A4 (single node, TP=1) ====
    {
      // Not marked verified: the recipe ran on the 20260719 nightly, but no
      // GSM8K-class eval has been measured yet (see the pending benchmarks file).
      match: { hw: "h100", variant: "default", quant: "fp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--reasoning-parser nemotron_3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
