// LFM2.5-VL-3B launch recipe. Every green cell below was exercised with
// BF16 on the named hardware and passed its downstream evaluation gate.

export const config = {
  modelName: "LFM2.5-VL-3B",
  latencyPercentile: "Mean",
  supportedHardware: ["h100", "h200", "b200"],

  variants: [
    { id: "default", label: "3B", subtitle: "LFM2 hybrid LM · SigLIP2 So400M" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "default", label: "Default" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "LiquidAI/LFM2.5-VL-3B",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT:      { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
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
      {"type": "image_url", "image_url": {"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}},
      {"type": "text", "text": "What is in this image?"}
    ]
  }],
  "temperature": 0,
  "max_tokens": 64
}'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name random \\
  --random-input-len 1024 --random-output-len 1024 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    accuracy: {
      countbench_pct:
`# VLMEvalKit (github.com/open-compass/VLMEvalKit) against the OpenAI-compatible API
python3 run.py --data CountBenchQA --model LFM2-VL-3B \\
  --use-vllm --api-nproc 32 --mode all`,
      docvqa_pct:
`# VLMEvalKit (github.com/open-compass/VLMEvalKit) against the OpenAI-compatible API
python3 run.py --data DocVQA_VAL --model LFM2-VL-3B \\
  --use-vllm --api-nproc 32 --mode all`,
      mmmu_pro_pct:
`# VLMEvalKit (github.com/open-compass/VLMEvalKit) against the OpenAI-compatible API
python3 run.py --data MMMU_Pro_10c --model LFM2-VL-3B \\
  --use-vllm --api-nproc 32 --mode all`,
    },
    numPromptsByConc: { 1: 10, 100: 1000 },
  },

  accuracyLabels: [
    ["countbench_pct", "CountBenchQA", "%"],
    ["docvqa_pct", "DocVQA", "%"],
    ["mmmu_pro_pct", "MMMU-Pro", "%"],
  ],

  dockerImages: {
    h100: "lmsysorg/sglang:nightly-dev-20260729-16a52bff",
    h200: "lmsysorg/sglang:nightly-dev-20260729-16a52bff",
    b200: "lmsysorg/sglang:nightly-dev-20260729-16a52bff",
  },

  github: {
    cookbookModel: "LiquidAI/LFM2.5-VL-3B",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1] },
      ],
    },
    parsers: {
      items: [
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser lfm2" },
      ],
    },
  },

  cells: [
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--dtype bfloat16",
        "--mem-fraction-static 0.48",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--dtype bfloat16",
        "--mem-fraction-static 0.48",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "default", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_USE_CUDA_IPC_TRANSPORT=1",
        "SGLANG_USE_IPC_POOL_HANDLE_CACHE=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--dtype bfloat16",
        "--mem-fraction-static 0.48",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
