// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Every cell carries `--mamba-radix-cache-strategy extra_buffer` for the GDN
// hybrid backbone. FP8 / INT4 are 35B-A3B checkpoints only. INT4 stays at TP=1:
// the bf16 vision tower's W4A16 group-128 divisibility fails at TP=2.
// DSpark is an overlay; its draft path follows the variant.

export const config = {
  modelName: "MiniCPM-V 4.7",

  supportedHardware: ["gb300"],

  matchDims: [
    { id: "variant", title: "Model Variant", options: [
      { id: "1b",      label: "1B",      subtitle: "Dense" },
      { id: "35b-a3b", label: "35B-A3B", subtitle: "MoE" },
    ] },
    { id: "quant", title: "Quantization", options: [
      { id: "bf16", label: "BF16" },
      { id: "fp8",  label: "FP8",  showWhen: (s) => s.variant === "35b-a3b" },
      { id: "int4", label: "INT4", showWhen: (s) => s.variant === "35b-a3b" },
    ] },
    { id: "nodes", title: "Nodes", options: [
      { id: "single", label: "Single Node" },
    ] },
  ],

  overlayDims: [
    {
      id: "spec",
      title: "Speculative Decoding",
      default: "none",
      options: [
        { id: "none", label: "None" },
        {
          id: "dspark", label: "DSPARK",
          flags: (s) => [
            "--speculative-algorithm DSPARK",
            s.variant === "1b"
              ? "--speculative-draft-model-path openbmb/MiniCPM-V-4.7-1B-DSpark"
              : "--speculative-draft-model-path openbmb/MiniCPM-V-4.7-35B-A3B-DSpark",
          ],
        },
      ],
    },
  ],

  modelNames: {
    "1b|bf16":      "openbmb/MiniCPM-V-4.7-1B",
    "35b-a3b|bf16": "openbmb/MiniCPM-V-4.7-35B-A3B",
    "35b-a3b|fp8":  "openbmb/MiniCPM-V-4.7-35B-A3B-FP8",
    "35b-a3b|int4": "openbmb/MiniCPM-V-4.7-35B-A3B-INT4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://www.ilankelman.org/stopsigns/australia.jpg"}},{"type":"text","text":"Describe this image in one sentence."}]}] }'`,

  benchmarkCommands: {
    accuracy: {
      gsm8k_pct: `# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --temperature 1.0 --top-p 0.95`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    gb300: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "openbmb/MiniCPM-V-4.7-35B-A3B",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4] },
      ],
    },

    // Opt-in: the cells run without a parser, so the thinking trace stays in
    // `content`; `qwen3` matches the chat template's <think> format.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "gb300", variant: "1b", quant: "bf16", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mamba-radix-cache-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "35b-a3b", quant: "bf16", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mamba-radix-cache-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "35b-a3b", quant: "fp8", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mamba-radix-cache-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "35b-a3b", quant: "int4", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--mamba-radix-cache-strategy extra_buffer",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
