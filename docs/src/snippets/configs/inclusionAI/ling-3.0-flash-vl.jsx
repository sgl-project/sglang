export const config = {
  modelName: "Ling-3.0-flash-VL",

  supportedHardware: ["gb300", "b300", "b200", "h200", "h100"],
  groupHardware: false,

  variants: [{ id: "default", label: "Ling-3.0-flash-VL" }],
  quantizations: [
    { id: "bf16", label: "BF16" },
    { id: "fp8", label: "FP8" },
  ],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|bf16": "inclusionAI/Ling-3.0-flash-VL",
    "default|fp8": "inclusionAI/Ling-3.0-flash-VL-FP8",
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
-d '{
  "model": "{{MODEL_NAME}}",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"}},
      {"type": "text", "text": "Describe this image in one sentence."}
    ]
  }]
}'`,

  dockerImages: {
    gb300: "lmsysorg/sglang:dev-Ling-3.0-flash-VL",
    b300:  "lmsysorg/sglang:dev-Ling-3.0-flash-VL",
    b200:  "lmsysorg/sglang:dev-Ling-3.0-flash-VL",
    h200:  "lmsysorg/sglang:dev-Ling-3.0-flash-VL",
    h100:  "lmsysorg/sglang:dev-Ling-3.0-flash-VL",
  },

  benchmarkCommands: {
    speed: `python3 -m sglang.bench_serving \\
  --backend sglang-oai-chat \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --image-count 1 --image-resolution 720p \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    accuracy: {
      mmmu_pro_pct: `pip install sgl-eval
sgl-eval run mmmu_pro \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --temperature 0.0 --top-p 0.95 \\
  --num-threads 64`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128 },
  },

  accuracyLabels: [
    ["mmmu_pro_pct", "MMMU-Pro", "%"],
  ],

  github: {
    cookbookModel: "inclusionAI/Ling-3.0-flash-VL",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 2, 4, 8] },
      ],
    },
  },

  cells: [
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Verified with online dynamic FP8 (--quantization fp8 on the BF16
      // checkpoint), the same serving path the FP8 repo uses natively.
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--context-length 262144",
        '--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}',
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
