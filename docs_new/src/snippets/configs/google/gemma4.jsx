// Gemma 4 cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see the _deployment.jsx header for the field contract.
//
// MIGRATED from the legacy Gemma 4 interactive command generator. Faithful port:
// every cell mirrors the legacy generator's output for that combination verbatim
// (modulo the {{HOST_IP}}/{{PORT}} tail + the EAGLE alias rewrite). No cell is
// `verified` — the legacy page carried no per-combo green-badge data and
// migration never flips a cell to verified.
//
// 5-dim mapping (legacy control -> new home):
//   modelSize  -> variant (e2b/e4b/12b/31b/26b-a4b)
//   checkpoint -> quant   (Standard BF16 -> bf16; QAT q4_0-unquantized -> qat;
//                          the QAT releases keep bf16 weights, so TP/mem match
//                          the standard checkpoints — only the model-path slug
//                          changes, routed via modelNames)
//   hardware   -> match.hw (h200 / b200; mi300x only for 31b + 26b-a4b, exactly
//                          as the legacy widget gated it)
//   speculative (MTP) -> strategies: the legacy toggle's own "Lower Latency"
//                          subtitle is the operating-point signal AND it couples
//                          with TP on the 26B-A4B MoE (tp1 -> tp2 when MTP is on),
//                          which the Playground cannot express -> MTP on =
//                          low-latency, MTP off = high-throughput. mi300x hides
//                          the toggle -> its single recipe -> balanced. The page
//                          ships the trio union; the engine greys unused chips.
//   reasoning / toolcall -> Playground `parsers` axis (DSv4 convention: parser
//                          flags are NEVER baked into Deploy cells; the axis adds
//                          them on top).
//
// Speculative is NOT exposed as a Playground axis: each MTP recipe needs a
// per-variant `--speculative-draft-model-path .../<variant>-it-assistant`, which
// a single shared axis preset cannot template (the LFM2.5 per-variant-parser
// precedent), and the engine's spec-strip list omits
// `--speculative-draft-model-path` (toggling off would leave it dangling). MTP is
// fully covered by the low-latency / high-throughput strategy dimension instead.

export const config = {
  modelName: "Gemma 4",

  supportedHardware: ["h200", "b200", "mi300x"],

  // 2nd dim — model sizes (the legacy "Model Variant" radio).
  variants: [
    { id: "e2b",     label: "E2B",     subtitle: "~2B dense" },
    { id: "e4b",     label: "E4B",     subtitle: "~4B dense" },
    { id: "12b",     label: "12B",     subtitle: "dense" },
    { id: "31b",     label: "31B",     subtitle: "dense" },
    { id: "26b-a4b", label: "26B-A4B", subtitle: "26B total / 4B active (MoE)" },
  ],
  // 3rd dim — the legacy "Checkpoint" radio (the checkpoint choice IS the quant axis).
  quantizations: [
    { id: "bf16", label: "Standard (BF16)" },
    { id: "qat",  label: "QAT (q4_0-unquantized)" },
  ],
  // 4th dim — driven by the MTP signal (see header). The page ships the union of
  // tiers used; the engine greys the chips a given (hw x variant x quant) lacks.
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "balanced",        label: "Balanced"        },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  // HF slug lookup. The QAT releases share the dense/MoE arch — only the slug
  // suffix differs — so keys are variant|quant (no per-hw repackaging).
  modelNames: {
    "e2b|bf16":     "google/gemma-4-E2B-it",
    "e2b|qat":      "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    "e4b|bf16":     "google/gemma-4-E4B-it",
    "e4b|qat":      "google/gemma-4-E4B-it-qat-q4_0-unquantized",
    "12b|bf16":     "google/gemma-4-12B-it",
    "12b|qat":      "google/gemma-4-12B-it-qat-q4_0-unquantized",
    "31b|bf16":     "google/gemma-4-31B-it",
    "31b|qat":      "google/gemma-4-31B-it-qat-q4_0-unquantized",
    "26b-a4b|bf16": "google/gemma-4-26B-A4B-it",
    "26b-a4b|qat":  "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
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

  // ⚡ Reproduce commands. The legacy benchmark numbers were measured on the
  // "gemma4 branch" — a moving development ref, not a reproducible anchor — so the
  // measured results (speed AND accuracy) are dropped (no -benchmarks.jsx). These
  // commands stay so users can re-measure against a pinned build (e.g. the Gemma 4
  // enabling PR #21952, or any release that ships it). GSM8K is the chat-template
  // run_eval harness (the few-shot completion harness under-elicits the
  // reasoning-oriented variants — see Configuration Tips).
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    accuracy: {
      gsm8k_pct:
`# Chat-template harness (robust answer extraction for the reasoning-oriented variants)
python3 -m sglang.test.run_eval \\
  --eval-name gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}}`,
    },
    numPromptsByConc: { 1: 10, 100: 1000 },
  },

  // Per-hw image for the `docker run` framing. The legacy page pinned dedicated
  // multi-arch (amd64 + arm64) Gemma-4 dev images; copy them verbatim. AMD ROCm
  // GPUs reuse the standard nightly ROCm tags (the page documented MI300X via
  // the same source install, no dedicated AMD image).
  dockerImages: {
    h200:   "lmsysorg/sglang:dev-gemma-4-12B",
    b200:   "lmsysorg/sglang:dev-gemma-4-12B",
    mi300x: "lmsysorg/sglang:dev-rocm720-mi30x",
  },

  // Prefills the issue template's free-form `model` field on "Submit verified cell".
  // Never prune — without it the engine falls back to deepseek-ai/deepseek-v4.
  github: {
    cookbookModel: "google/gemma4",
  },

  playgroundFeatures: {

    // ----- Attention Parallelism ----- the cells expose --tp; let users override.
    attention: {
      knobs: [
        { id: "tp",     label: "TP", values: [null, 1, 2, 4, 8] },
        { id: "cp",     label: "CP", values: [null, 1, 2, 4] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- MoE Parallelism ----- applies to the 26B-A4B MoE variant; greyed out
    // on the dense variants (e2b/e4b/12b/31b have no experts to shard).
    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"],
            disable: { variant: ["e2b", "e4b", "12b", "31b"] },
            disableReason: "MoE expert-parallel backends apply only to the 26B-A4B MoE variant. Switch the Deploy panel's Model Variant to 26B-A4B." },
        ],
      },
      ep: { label: "EP", values: [
        null,
        { value: 1, disable: { variant: ["e2b", "e4b", "12b", "31b"] }, disableReason: "Expert Parallelism applies only to the 26B-A4B MoE variant. Switch the Deploy panel's Model Variant to 26B-A4B." },
        { value: 2, disable: { variant: ["e2b", "e4b", "12b", "31b"] }, disableReason: "Expert Parallelism applies only to the 26B-A4B MoE variant. Switch the Deploy panel's Model Variant to 26B-A4B." },
        { value: 4, disable: { variant: ["e2b", "e4b", "12b", "31b"] }, disableReason: "Expert Parallelism applies only to the 26B-A4B MoE variant. Switch the Deploy panel's Model Variant to 26B-A4B." },
        { value: 8, disable: { variant: ["e2b", "e4b", "12b", "31b"] }, disableReason: "Expert Parallelism applies only to the 26B-A4B MoE variant. Switch the Deploy panel's Model Variant to 26B-A4B." },
      ] },
    },

    // ----- Parsers ----- Gemma 4 ships both a reasoning and a tool-call parser
    // (both keyed `gemma4`). Add-only: these are never in a Deploy cell.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser gemma4" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser gemma4" },
      ],
    },

    // ----- Hierarchical KV Cache ----- useful for the larger variants under
    // long-context / high-reuse serving.
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

  // ===== Cells — faithful 1:1 port of the legacy generator output (parsers
  // stripped to the Playground axis; spec flags baked into the low-latency tier;
  // EAGLE = the NEXTN alias the legacy generator emitted). All YELLOW. =====
  cells: [
    {
      match: { hw: "h200", variant: "e4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "e2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-E2B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "e2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "e2b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "e2b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "e4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-E4B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "e4b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "e4b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "12b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-12B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "12b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "12b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-12B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "12b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "31b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-31B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "31b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "31b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-31B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "31b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "26b-a4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-26B-A4B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "26b-a4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "26b-a4b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "26b-a4b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "e2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-E2B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "e2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "e2b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "e2b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "e4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-E4B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "e4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "e4b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "e4b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "12b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-12B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "12b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "12b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-12B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "12b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "31b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-31B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "31b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "31b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-31B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "31b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "26b-a4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-26B-A4B-it-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "26b-a4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "26b-a4b", quant: "qat", strategy: "low-latency", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant",
        "--speculative-num-steps 5",
        "--speculative-num-draft-tokens 6",
        "--speculative-eagle-topk 1",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "26b-a4b", quant: "qat", strategy: "high-throughput", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "31b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "31b", quant: "qat", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "26b-a4b", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "26b-a4b", quant: "qat", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
