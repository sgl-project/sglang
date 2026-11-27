// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// Single `export const config` literal with no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// PPLX-Decider-v1.3-27B: the protocol-v4 successor of v1.1, same Qwen3.8-27B
// backbone and 255-row readout. Full-attention layers are noncausal
// (decision_config.json `attention_mode: noncausal_full_attention`), and the
// readout is applied at the `<|fim_suffix|>` marker at the end of each question
// chunk. Every request is prefill-only, one pass per question with no decode,
// so there are no parser, speculative-decoding or PD-disaggregation knobs. BF16
// is the only published precision.
//
// SGLang reads the attention mode and turns off the radix cache and chunked
// prefill itself, so no cell needs a flag for it. H20 was measured end to end
// on one 96 GB GPU; the H200 and GB300 commands remain unverified.
export const config = {
  modelName: "PPLX-Decider-v1.3-27B",

  latencyPercentile: "P50",

  supportedHardware: ["h20", "h200", "gb300"],

  hardware: [
    { id: "h20", label: "H20", vram: "96GB", vendor: "hopper" },
  ],

  variants: [
    { id: "default", label: "Default", subtitle: "27B dense, 1 GPU" },
  ],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|bf16": "perplexity-ai/pplx-decider-v1.3",
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

  accuracyLabels: [
    ["belebele_pct", "Belebele (eng_Latn, 900)", "%"],
    ["winogrande_pct", "WinoGrande (xl dev, 1267)", "%"],
  ],

  // `lmsysorg/sglang:dev` is multi-arch (amd64 + arm64), so GB300 hosts pull
  // the same tag as H200 hosts.
  dockerImages: {
    h20: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
    gb300: "lmsysorg/sglang:dev",
  },

  github: { cookbookModel: "perplexity-ai/pplx-decider-v1.3" },

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
            disable: { hw: ["h20", "h200"] },
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
      // H20 (96 GB), one GPU. Auto resolves to FA3 for both text and vision.
      // The server resolves --disable-radix-cache and --chunked-prefill-size -1.
      match: { hw: "h20", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
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
      // causal=False context kernel. Not yet measured for v1.3.
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
}
