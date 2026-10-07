// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// Single `export const config` literal with no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// PPLX-Decider-v1.1-27B: the v1 decision checkpoint's successor, same
// Qwen3.8-27B backbone and 255-row readout, trained with noncausal full
// attention (decision_config.json `attention_mode: noncausal_full_attention`).
// Every request is prefill-only, one pass per question with no decode, so there
// are no parser, speculative-decoding or PD-disaggregation knobs. BF16 is the
// only published precision.
//
// SGLang reads the attention mode and turns off the radix cache and chunked
// prefill itself, so no cell needs a flag for it. GB300: measured end to end
// on PR #42645 for speed, Belebele/WinoGrande accuracy, and per-item parity
// against the checkpoint's own reference DecisionModel. H200: same command,
// not run yet.
export const config = {
  modelName: "PPLX-Decider-v1.1-27B",

  latencyPercentile: "P50",

  supportedHardware: ["h200", "gb300"],

  variants: [
    { id: "default", label: "Default", subtitle: "27B dense, 1 GPU" },
  ],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|bf16": "perplexity-ai/pplx-decider-v1.1-27b",
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

  // Same sets and conversion as the v1 page, so the two pages compare directly.
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

  github: { cookbookModel: "perplexity-ai/pplx-decider-v1.1-27b" },

  // Every request is prefill-only (one pass per question, no decode), so the
  // general axes that act on generation are omitted: parsers (no text output),
  // speculative decoding and PD disaggregation (no decode phase). moe: dense
  // model. CP / DP-Attention: no model-side support for this architecture.
  // Prefix cache and hicache: noncausal attention rules out any prefix reuse,
  // and the server turns the radix cache off itself.
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
        // All three run the full-attention layers noncausally. Auto resolves
        // to trtllm_mha with 64-token pages on GB300.
        id: "attnBackend", title: "Attention Backend",
        stripPrefixes: ["--attention-backend", "--page-size"],
        options: [
          { id: "auto", label: "Auto" },
          { id: "trtllm", label: "TRT-LLM MHA",
            flags: ["--attention-backend trtllm_mha", "--page-size 64"],
            disable: { hw: ["h200"] },
            disableReason: "trtllm_mha is a Blackwell (SM100) kernel" },
          { id: "flashinfer", label: "FlashInfer", flags: ["--attention-backend flashinfer"] },
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
    ],
  },

  cells: [
    {
      // Same command as GB300, not run on H200 yet.
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // GB300 (SM103), one GPU. The server resolves --disable-radix-cache,
      // --chunked-prefill-size -1 and trtllm_mha with 64-token pages on its
      // own. Full attention runs noncausally through trtllm_mha's
      // causal=False context kernel.
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
}
