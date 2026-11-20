// Hy4 cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.
//
// Sizing: the FP8 checkpoint holds ~786GB of weights → TP8 on 8x H200
// or B200 (~98GB/rank), or TP4 on 4x B300/GB300 (~197GB/rank).
// The BF16 checkpoint (~1.55TB) exceeds both layouts, so it has no recipe.

export const config = {
  modelName: "Hy4",

  supportedHardware: ["b300", "b200", "h200", "gb300"],

  matchDims: [
    {
      id: "quant",
      title: "Quantization",
      options: [
        { id: "fp8",  label: "FP8"  },
        { id: "bf16", label: "BF16", disabled: true,
          disableReason: "The BF16 checkpoint (~1.55TB of weights) exceeds TP8 on H200/B200 and TP4 on B300/GB300; use the FP8 checkpoint." },
      ],
    },
    {
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency",    label: "Low-Latency"    },
        { id: "high-throughput", label: "High-Throughput" },
      ],
    },
  ],

  modelNames: {
    "fp8":  "tencent/Hy4-FP8",
    "bf16": "tencent/Hy4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",   default: "0.0.0.0"   },
    PORT:      { target: "command", label: "Bind port",   default: "30000"     },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port", default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"}},{"type":"text","text":"Describe this image."}]}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.benchmark.serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} --tokenizer {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1.0 --tokenize-prompt \\
  --temperature 0 --num-prompts {{NUM_PROMPTS}} \\
  --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests {{MAX_CONCURRENCY}} --flush-cache`,
    accuracy: {
      gsm8k_pct:
`sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} --num-threads 64 \\
  --max-tokens 32768 --temperature 0 --top-p 0.95`,
    },
    numPromptsByConc: { 1: 16, 64: 256 },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    b300:  "lmsysorg/sglang:hy4",
    b200:  "lmsysorg/sglang:hy4",
    h200:  "lmsysorg/sglang:hy4",
    gb300: "lmsysorg/sglang:hy4",
  },

  runModes: ["python", "docker"],

  github: {
    cookbookModel: "tencent/Hy4",
  },

  playgroundFeatures: {

    attention: {
      knobs: [
        { id: "cp", label: "CP (DSA prefill)", freeSize: true,
          values: [null, { value: 1, label: "Off" },
            { value: 2, disable: [{ when: { dpAttnOn: [true] },
              reason: "Prefill CP is refused with DP-Attention for Hy4; turn DP-Attention off." }] }],
          disable: [{ when: { specAlgorithm: ["DFLY"] },
            reason: "Prefill CP is refused with DFly2; use the High-Throughput recipe or set Speculative to Off." }] },
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null,
            false,
            { value: 2, disable: [{ when: { cpOn: [true] },
                  reason: "DP-Attention is refused with prefill CP for Hy4; turn CP off." }] },
            { value: 4, disable: [{ when: { cpOn: [true] },
                  reason: "DP-Attention is refused with prefill CP for Hy4; turn CP off." }] },
            { value: 8,
              disable: [
                { when: { cpOn: [true] },
                  reason: "DP-Attention is refused with prefill CP for Hy4; turn CP off." },
                { when: { effTp: [4] },
                  reason: "DP-Attention=8 needs TP 8; the selected recipe runs TP 4." },
              ] },
          ],
          disable: [{ when: { specAlgorithm: ["DFLY"] },
            reason: "DP-Attention is refused with DFly2; use the High-Throughput recipe or set Speculative to Off." }],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off" },
        { id: "dfly",    label: "DFly2",
          flags: ["--speculative-algorithm DFLY"],
          disable: [
            { when: { strategy: ["high-throughput"] },
              reason: "Start from the DFly2 recipe: switch the Deploy panel's Strategy to Low-Latency." },
            { when: { cpOn: [true] },
              reason: "DFly2 is refused with prefill CP; turn CP off in the Attention card." },
            { when: { dpAttnOn: [true] },
              reason: "DFly2 is refused with DP-Attention; turn DP-Attention off in the Attention card." },
          ] },
      ],
    },

    hicache: {
      backends: [
        { id: null,       label: "None (host memory)" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl",     label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "h200", quant: "fp8", strategy: "low-latency" },
      verified: true,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm DFLY",
        "--served-model-name {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", quant: "fp8", strategy: "high-throughput" },
      verified: false,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--served-model-name {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", quant: "fp8", strategy: "low-latency" },
      verified: true,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm DFLY",
        "--served-model-name {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", quant: "fp8", strategy: "high-throughput" },
      verified: false,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--served-model-name {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", quant: "fp8", strategy: "low-latency" },
      verified: false,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm DFLY",
        "--served-model-name {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", quant: "fp8", strategy: "high-throughput" },
      verified: false,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--served-model-name {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "fp8", strategy: "low-latency" },
      verified: false,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm DFLY",
        "--served-model-name {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "fp8", strategy: "high-throughput" },
      verified: false,
      nnodes: 1,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--served-model-name {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
