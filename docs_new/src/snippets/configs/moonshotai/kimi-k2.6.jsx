// Kimi-K2.6 cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.
//
// Migrated faithfully from the legacy command generator (no recipe
// changes). The legacy generator's option space was:
//   hardware  x  quantization(int4/nvfp4)  x  reasoning  x  toolcall
//   x  dpattention(Low Latency / High Throughput)  x  speculative(NVIDIA-only)
//
// Mapping to the 5-dim matrix + Playground:
//   - dpattention -> `strategies` (its options are subtitled "Low Latency" /
//     "High Throughput" — the page's own named operating-point split):
//     disabled -> low-latency, enabled -> high-throughput (adds --dp N
//     --enable-dp-attention with N == --tp).
//   - reasoning / toolcall -> `parsers` Playground axis ONLY; per the cookbook
//     convention --reasoning-parser / --tool-call-parser are NEVER baked into
//     Deployment cells (the legacy generator defaulted them ON — that default
//     lives in the Playground, not the cells).
//   - speculative (EAGLE3, NVIDIA-only, legacy default OFF) -> `speculative`
//     Playground axis preset; not baked (default off).
//   - quantization int4 (native checkpoint) on every platform; nvfp4 only on
//     NVIDIA Blackwell (b300/gb300) -> absent cells elsewhere.
//
// TP coupling (verbatim from the generator): h200/b300 INT4 tp8, gb300 tp4,
// all AMD tp4; NVFP4 b300 tp8, gb300 tp4.
//
// NOTHING is marked verified: the maintainer has not personally validated
// Kimi-K2.6's deploy commands, so every cell stays unverified (yellow). The
// measured benchmark blocks (kimi-k2.6-benchmarks.jsx) are independent of the
// verified badge — they transcribe the legacy page's real numbers.
//
// --kv-cache-dtype fp8_e4m3 is baked into the AMD cells because the legacy
// recipe ships it unconditionally ("for memory efficiency", paired with
// --mem-fraction-static 0.8 to fit INT4 weights + KV); kept verbatim per the
// migration faithfulness rule.

export const config = {
  modelName: "Kimi-K2.6",

  supportedHardware: ["h200", "b300", "gb300", "mi300x", "mi325x", "mi350x", "mi355x"],

  variants: [
    { id: "default", label: "Default" },
  ],
  quantizations: [
    { id: "int4",  label: "INT4",  subtitle: "Base checkpoint" },
    { id: "nvfp4", label: "NVFP4", subtitle: "Blackwell FP4" },
  ],
  // dpattention subtitles "Low Latency" / "High Throughput" -> these two tiers.
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|int4":  "moonshotai/Kimi-K2.6",
    "default|nvfp4": "nvidia/Kimi-K2.6-NVFP4",
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

  // ⚡ Reproduce — the legacy page's measured commands.
  // Speed: sglang.bench_serving on the `random` dataset (the legacy §5.2 / §5.3
  // shape). Accuracy: each suite is its own external harness (K2-Vendor-Verifier,
  // NeMo-Skills, Inspect AI, Kimi-Vendor-Verifier) — transcribed verbatim from
  // the legacy §5.1 evaluation commands.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --request-rate inf`,
    accuracy: {
      // Tool-calling validity — K2-Vendor-Verifier tool_calls_eval.py (2,000 reqs).
      toolcall_valid_pct:
`# K2-Vendor-Verifier: https://github.com/MoonshotAI/K2-Vendor-Verifier
cd K2-Vendor-Verifier
python tool_calls_eval.py tool-calls/samples.jsonl \\
  --model "{{MODEL_NAME}}" \\
  --base-url "http://{{CURL_HOST}}:{{CURL_PORT}}/v1" \\
  --api-key "placeholder" \\
  --concurrency 256 \\
  --temperature 1.0 \\
  --max-tokens 64000 \\
  --output kimi-k26-results.jsonl`,
      // AIME 2025 — NVIDIA NeMo-Skills, MathArena prompt, 32 seeds.
      aime25_pct:
`# NVIDIA NeMo-Skills: https://github.com/NVIDIA/NeMo-Skills
python3 nemo_skills/dataset/aime25/prepare.py
for RS in $(seq 0 31); do
  python3 nemo_skills/inference/generate.py \\
    input_file=nemo_skills/dataset/aime25/test.jsonl \\
    output_file=results/kimi-k26/aime25/output-rs\${RS}.jsonl \\
    prompt_config=eval/matharena/aime \\
    prompt_format=openai \\
    +server.server_type=openai \\
    +server.model={{MODEL_NAME}} \\
    +server.base_url=http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
    ++inference.temperature=1.0 \\
    ++inference.top_p=0.95 \\
    ++inference.tokens_to_generate=131072 \\
    ++inference.random_seed=\${RS} \\
    max_concurrent_requests=512 &
done`,
      // GPQA Diamond — Inspect AI, 4 epochs, cot=True.
      gpqa_pct:
`# Inspect AI: https://github.com/UKGovernmentBEIS/inspect_ai
OPENAI_BASE_URL=http://{{CURL_HOST}}:{{CURL_PORT}}/v1 OPENAI_API_KEY=placeholder \\
inspect eval inspect_evals/gpqa_diamond \\
  --model openai/{{MODEL_NAME}} \\
  --max-tokens 131072 \\
  --temperature 1.0 \\
  --top-p 0.95 \\
  --max-connections 128 \\
  -T cot=True`,
      // OCRBench — Kimi-Vendor-Verifier (inspect-ai based), thinking mode.
      ocrbench_pct:
`# Kimi-Vendor-Verifier: https://github.com/MoonshotAI/Kimi-Vendor-Verifier
cd Kimi-Vendor-Verifier
OPENAI_BASE_URL=http://{{CURL_HOST}}:{{CURL_PORT}}/v1 OPENAI_API_KEY=placeholder \\
python3 eval.py ocrbench \\
  --model openai/{{MODEL_NAME}} \\
  --max-tokens 4096 \\
  --think-mode opensource \\
  --thinking \\
  --max-connections 256`,
      // MMMU Pro Vision — Kimi-Vendor-Verifier, max-tokens 32768 (reasoning budget).
      mmmu_pro_pct:
`# Kimi-Vendor-Verifier: https://github.com/MoonshotAI/Kimi-Vendor-Verifier
cd Kimi-Vendor-Verifier
OPENAI_BASE_URL=http://{{CURL_HOST}}:{{CURL_PORT}}/v1 OPENAI_API_KEY=placeholder \\
python3 eval.py mmmu \\
  --model openai/{{MODEL_NAME}} \\
  --max-tokens 32768 \\
  --think-mode none \\
  --max-connections 256`,
    },
    numPromptsByConc: { 1: 10, 16: 80, 64: 320, 100: 500 },
  },

  // The eval set rendered in the benchmark card + "⚡ Reproduce". Transcribed
  // from the legacy §5.1 accuracy blocks (measured on 8xH200, INT4, sglang
  // 0.5.9, with the kimi_k2 reasoning + tool-call parsers on).
  accuracyLabels: [
    ["toolcall_valid_pct", "Tool Call Valid (K2-Vendor-Verifier)", "%"],
    ["aime25_pct",         "AIME 2025 (majority@32)",              "%"],
    ["gpqa_pct",           "GPQA Diamond (pass@1)",                "%"],
    ["ocrbench_pct",       "OCRBench (pass@1)",                    "%"],
    ["mmmu_pro_pct",       "MMMU Pro Vision (pass@1)",             "%"],
  ],

  // Only the docker tags the legacy page pinned (§3.2 AMD Docker Image tip):
  //   MI350X/MI355X -> v0.5.9-rocm700-mi35x ; MI300X/MI325X -> v0.5.9-rocm700-mi30x.
  // NVIDIA (h200/b300/gb300) had no pinned image on the legacy page -> :dev fallback.
  dockerImages: {
    mi300x: "lmsysorg/sglang:v0.5.9-rocm700-mi30x",
    mi325x: "lmsysorg/sglang:v0.5.9-rocm700-mi30x",
    mi350x: "lmsysorg/sglang:v0.5.9-rocm700-mi35x",
    mi355x: "lmsysorg/sglang:v0.5.9-rocm700-mi35x",
  },

  github: {
    cookbookModel: "moonshotai/Kimi-K2.6",
  },

  playgroundFeatures: {

    // ----- Attention Parallelism -----
    attention: {
      knobs: [
        { id: "tp",     label: "TP", values: [null, 1, 2, 4, 8] },
        { id: "cp",     label: "CP", values: [null, 1, 2, 4] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- MoE Parallelism -----  Kimi-K2.6 is a MoE model; the legacy recipe
    // doesn't pick a --moe-*-backend (engine auto-selects), so only the generic
    // DeepEP backend + the EP knob are offered for experimentation.
    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    // ----- Parsers -----  kimi_k2 reasoning + tool-call parsers. Add-only:
    // never part of a Deployment cell (legacy default was ON; the cells mirror
    // the parsers-OFF output, and this axis re-adds them on top).
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser kimi_k2" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser kimi_k2" },
      ],
    },

    // ----- Speculative Decoding -----  EAGLE3 MTP draft, NVIDIA-only (the
    // legacy generator gates the toggle off on AMD). Legacy default OFF -> not
    // baked into cells; this preset re-applies the exact legacy flag set.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "eagle3",  label: "EAGLE3 (MLA draft)",
          flags: [
            "--speculative-algorithm EAGLE3",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
            "--speculative-draft-model-path lightseekorg/kimi-k2.6-eagle3.1-mla",
          ],
          disable: { hw: ["mi300x", "mi325x", "mi350x", "mi355x"] },
          disableReason: "Speculative decoding for Kimi-K2.6 is only supported on NVIDIA GPUs (H200/B300/GB300)." },
      ],
    },

    // ----- PD Disaggregation -----
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl",     label: "NiXL" },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:30000 \\
  --decode http://<decode-host>:30001 \\
  --policy round_robin \\
  --host 0.0.0.0 --port 8000`,
      },
    },

    // ----- Hierarchical KV Cache -----
    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "hf3fs",    label: "HF3FS" },
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

  // Cells enumerated from the legacy generator (hw x quant x dpattention).
  // Parsers + speculative are Playground-only (not baked). NOTHING is verified
  // (no maintainer attestation). cells[0] = the legacy default selection
  // (H200 / INT4 / dpattention-off -> low-latency).
  cells: [
    {
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--attention-backend tokenspeed_mla",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--attention-backend tokenspeed_mla",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--attention-backend tokenspeed_mla",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--quantization modelopt_fp4",
        "--attention-backend tokenspeed_mla",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--attention-backend tokenspeed_mla",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--attention-backend tokenspeed_mla",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--quantization modelopt_fp4",
        "--attention-backend tokenspeed_mla",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--quantization modelopt_fp4",
        "--attention-backend tokenspeed_mla",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      env: ["SGLANG_USE_AITER=1", "SGLANG_ROCM_FUSED_DECODE_MLA=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.8",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_USE_AITER=1", "SGLANG_ROCM_FUSED_DECODE_MLA=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--mem-fraction-static 0.8",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      env: ["SGLANG_USE_AITER=1", "SGLANG_ROCM_FUSED_DECODE_MLA=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.8",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_USE_AITER=1", "SGLANG_ROCM_FUSED_DECODE_MLA=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--mem-fraction-static 0.8",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      env: ["SGLANG_USE_AITER=1", "SGLANG_ROCM_FUSED_DECODE_MLA=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.8",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_USE_AITER=1", "SGLANG_ROCM_FUSED_DECODE_MLA=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--mem-fraction-static 0.8",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
      env: ["SGLANG_USE_AITER=1", "SGLANG_ROCM_FUSED_DECODE_MLA=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.8",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
      env: ["SGLANG_USE_AITER=1", "SGLANG_ROCM_FUSED_DECODE_MLA=0"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--mem-fraction-static 0.8",
        "--kv-cache-dtype fp8_e4m3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
