// Instantiated from cookbook-add-model/templates/config.jsx.tmpl.
// Single `export const config` literal with no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// StartLux-Decision-v1.1-27B-AutoThink: a decision model on the Qwen3.5-27B
// backbone that keeps its own LM head, whose rows of the answer letters A to Z
// are the readout. SGLang reads `decision_config.json` (format
// `lodestar-letter-readout-v1`): it renders the trained prompt, reads one letter
// per option at the last prompt position at the question type's own temperature,
// and reports the share of an extra "None of the above" candidate.
//
// Every request is prefill-only, one pass per question with no decode, so there
// are no parser, speculative-decoding or PD-disaggregation knobs. BF16 is the
// published precision. AutoThink's thought loop is a second read-out after a
// generated thought, which this route does not run: run the checkpoint's own
// protocol layer against the server for that.
export const config = {
  modelName: "StartLux-Decision-v1.1-27B-AutoThink",

  latencyPercentile: "P50",

  supportedHardware: ["h200", "a100"],

  variants: [
    { id: "default", label: "Default", subtitle: "27B dense, 2 GPUs on 40 GB cards" },
  ],
  quantizations: [{ id: "bf16", label: "BF16" }],
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|bf16": "startlux-models/StartLux-Decision-v1.1-27B-AutoThink",
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
  -d '{"model":"{{MODEL_NAME}}","state":{"ticket":"I was charged twice for order #4411 and the app still shows it as unpaid."},"questions":{"team":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":{"billing":"Payments and refunds","shipping":"Delivery","technical":"App and login"}},"urgent":{"type":"noul","instructions":"Should this ticket be answered today?"},"severity":{"type":"score","instructions":"How severe is the impact?","criteria":["cosmetic","annoying","blocks the customer"]}}}'`,

  tips: [
    "The checkpoint reads its prompt causally, so the radix cache and chunked prefill are safe; use `--disable-radix-cache` when comparing probabilities against the checkpoint's own server.",
    '`none_probability` is the share of the read-out that went to an extra "None of the above" option. Pass `none_check: false` to read the listed options only.',
    "`allow_none: true` lets `choice` be null when that extra option wins.",
    "A choice list over 25 options needs the checkpoint's own grouping rounds, which this route does not run; it refuses the request instead.",
    "AutoThink's thought loop (`think_mode`, `gate`, `mix`) is refused rather than ignored: this route reads one answer and generates nothing.",
  ],
}
