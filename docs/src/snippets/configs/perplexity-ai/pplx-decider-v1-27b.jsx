// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// Single `export const config` literal with no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// PPLX-Decider-v1-27B: a decision checkpoint (Qwen3.8-27B backbone + a 255-row
// readout) served on /v1/systemone. Every request is prefill-only, one pass
// per question with no decode, so there are no parser, speculative-decoding or
// PD-disaggregation knobs. BF16 is the only published precision.
//
// H200: the original recipe, verified by the model-support PR (#42183) on one
// H200 with no extra flag. GB300: measured end to end on main @ 70f0b7351e for
// speed, Belebele/WinoGrande accuracy, and per-item parity against the
// checkpoint's own reference DecisionModel (see the benchmarks file).
export const config = {
  modelName: "PPLX-Decider-v1-27B",

  latencyPercentile: "P50",

  supportedHardware: ["h200", "gb300"],

  variants: [
    { id: "default", label: "Default", subtitle: "27B dense, 1 GPU" },
  ],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|bf16": "perplexity-ai/pplx-decider-v1-27b",
  },

  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/systemone \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"{{MODEL_NAME}}","state":"My Stripe integration keeps failing. Please help ASAP.","questions":{"urgency":{"type":"noul","instructions":"Does this message express urgency?"}}}'`,

  // Accuracy through /v1/systemone on public sets the model card also reports,
  // with this page's own conversion (see the page's section 3). Not comparable
  // number-for-number with the card, which used Perplexity's converters.
  accuracyLabels: [
    ["belebele_pct", "Belebele (eng_Latn, 900)", "%"],
    ["winogrande_pct", "WinoGrande (xl dev, 1267)", "%"],
  ],

  // `lmsysorg/sglang:dev` is multi-arch (amd64 + arm64), so GB300 hosts pull
  // the same tag as H200 hosts.
  dockerImages: {
    h200: "lmsysorg/sglang:dev",
    gb300: "lmsysorg/sglang:dev",
  },

  github: { cookbookModel: "perplexity-ai/pplx-decider-v1-27b" },

  // Every request is prefill-only (one pass per question, no decode), so the
  // general axes that act on generation are omitted: parsers (no text output),
  // speculative decoding and PD disaggregation (no decode phase). moe: dense
  // model. CP / DP-Attention: no model-side support for this architecture.
  // hicache: the prefix cache only resumes from where an earlier prompt ended
  // (the Gated DeltaNet state is saved there), so a host-memory tier adds
  // nothing to unique-state decision traffic.
  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4] },
      ],
    },
    flagSelects: [
      {
        // Replicas of the 1-GPU model behind one endpoint, the throughput lever.
        id: "dp", title: "Data Parallel Replicas",
        stripPrefixes: ["--dp-size"],
        options: [
          { id: "1", label: "1" },
          { id: "2", label: "2", flags: ["--dp-size 2"] },
          { id: "4", label: "4", flags: ["--dp-size 4"] },
        ],
      },
      {
        // Only an identical prompt can resume from the cache on this hybrid
        // model. Off also frees the per-request state slots (S=1).
        id: "prefixCache", title: "Prefix Cache",
        stripPrefixes: ["--disable-radix-cache"],
        options: [
          { id: "on",  label: "On (reuses identical prompts)" },
          { id: "off", label: "Off (unique states)", flags: ["--disable-radix-cache"] },
        ],
      },
      {
        // trtllm_mha needs SM100 and a 64-token page. With the prefix cache
        // off, Auto already resolves to it on GB300.
        id: "attnBackend", title: "Attention Backend",
        stripPrefixes: ["--attention-backend", "--page-size"],
        options: [
          { id: "auto", label: "Auto" },
          { id: "trtllm", label: "TRT-LLM MHA",
            flags: ["--attention-backend trtllm_mha", "--page-size 64"],
            disable: { hw: ["h200"] },
            disableReason: "trtllm_mha is a Blackwell (SM100) kernel" },
          { id: "triton", label: "Triton", flags: ["--attention-backend triton"] },
        ],
      },
      {
        id: "gdnPrefill", title: "Gated DeltaNet Prefill Kernel",
        stripPrefixes: ["--linear-attn-prefill-backend"],
        options: [
          { id: "auto", label: "Auto" },
          { id: "triton", label: "Triton", flags: ["--linear-attn-prefill-backend triton"] },
          { id: "flashinfer", label: "FlashInfer", flags: ["--linear-attn-prefill-backend flashinfer"] },
        ],
      },
      {
        // Load-time FP8 of the BF16 checkpoint: ~20% more throughput on GB300,
        // but it moves the calibrated probabilities (mean 0.010, max 0.20 vs
        // the reference implementation), so it is never in a cell.
        id: "weights", title: "Weight Precision",
        stripPrefixes: ["--quantization"],
        options: [
          { id: "bf16", label: "BF16 (checkpoint)" },
          { id: "fp8", label: "FP8 at load (shifts probabilities)", flags: ["--quantization fp8"] },
        ],
      },
    ],
  },

  cells: [
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 (SM103), one GPU. Prefix reuse on this hybrid model only happens
      // for identical prompts, so the cache is off: each request needs one
      // Gated DeltaNet state slot instead of five, and with the cache off the
      // Qwen3.5 model hook resolves attention to trtllm_mha with 64-token
      // pages (Triton with 1-token pages otherwise). Isolated A/B on main @
      // 70f0b7351e against the no-flag recipe: 8K-token state 377 -> 325 ms at
      // concurrency 1 and +15% saturated throughput, 380-token question 58 ->
      // 51 ms, same Belebele/WinoGrande accuracy and reference parity.
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--disable-radix-cache",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
}
