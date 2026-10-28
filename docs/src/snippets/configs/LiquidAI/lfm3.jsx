// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// LFM3 note: both variants run on ONE GPU (TP=1, TP=2 verified on the text
// model), so the matrix is hw × variant only (single quant / strategy /
// nodes). The attention backend resolves to flashinfer via the LFM3 model
// overrides, which also set SGLANG_FLASHINFER_USE_PAGED=True for
// sliding-window prefill correctness — neither needs a cell flag. Parsers are
// a Playground axis: `auto` resolves qwen3 (reasoning) and qwen3_coder (tool
// calls) from the chat template. `--dtype` is omitted: the checkpoints declare
// bfloat16 and auto resolves it.

export const config = {
  modelName: "LFM3",

  supportedHardware: [
    "h100", "h200", "b200", "b300", "gb200", "gb300", "dgx-spark",
    "mi300x", "mi325x", "mi350x", "mi355x",
  ],

  variants: [
    { id: "1.7b", label: "1.7B",      subtitle: "dense hybrid · text" },
    { id: "vl",   label: "VL 2B",   subtitle: "vision-language · image + video" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "1.7b|bf16": "LiquidAI/LFM3-1.7B",
    "vl|bf16":   "LiquidAI/LFM3-VL-2B",
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
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // The eval set rendered in the benchmark card (the engine ships no default —
  // every config declares its own).
  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  benchmarkCommands: {
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --temperature 1.0 --top-p 0.95`,
    },
  },

  // LFM3 support ships on SGLang `main`; not yet in a tagged release, so the
  // pinned Docker path is the PR-head dev image built from #46858. No pip
  // release carries LFM3 yet — NVIDIA cells are Docker-only for now.
  dockerImages: {
    h100: "lmsysorg/sglang:dev-lfm3",
    h200: "lmsysorg/sglang:dev-lfm3",
    b200: "lmsysorg/sglang:dev-lfm3",
    b300: "lmsysorg/sglang:dev-lfm3",
    gb200: "lmsysorg/sglang:dev-lfm3",
    gb300: "lmsysorg/sglang:dev-lfm3",
    // DGX Spark (GB10, sm121) needs a CUDA 13 build.
    "dgx-spark": "lmsysorg/sglang:dev-cu13-lfm3",
    // mi* intentionally unmapped: no published ROCm image carries LFM3 support
    // yet, so `runModes` below hides the Docker tab on AMD.
  },

  runModes: (s) => (["mi300x", "mi325x", "mi350x", "mi355x"].includes(s.hw)
    ? ["python"] : ["python", "docker"]),

  // Prefills the issue template's `model` field on "Submit verified cell".
  github: {
    cookbookModel: "LiquidAI/LFM3-1.7B",
  },

  // No speculative axis: the server rejects speculative decoding for LFM3.
  playgroundFeatures: {
    // Both variants are served at TP=1 in the cells; TP=2 is verified on the
    // text model.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, { value: 2, disable: { hw: ["dgx-spark"] } }] },
      ],
    },
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role", hide: { hw: ["dgx-spark"] } },
        { id: "decode",  label: "Decode role",  hide: { hw: ["dgx-spark"] } },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl",     label: "NiXL" },
      ],
      router: {
        port: 30080,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}}`,
      },
    },
    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "nixl",     label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },
  },

  cells: [
    // Verified: h200 (both variants), 2026-10-26, BF16, TP1, flashinfer.
    // H100: vendor-tested in FP16, not re-run by us. EVERYTHING ELSE below is
    // an inferred starting point (verified: false): same recipe on the other
    // NVIDIA platforms; aiter backend + SGLANG_USE_AITER=1 + the torch
    // gated-conv fallback on AMD.

    // ====================================================================
    // H200 (sm90) — verified 2026-10-26 (BF16, TP1, flashinfer). First in the
    // list because the Deploy panel seeds its initial selection from cells[0].
    // ====================================================================
    {
      match: { hw: "h200", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H100 (sm90) — same recipe as H200; vendor-tested in FP16, not re-run by us
    // ====================================================================
    {
      match: { hw: "h100", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 / B300 (sm100/sm103) — inferred from the verified H200 recipe
    // ====================================================================
    {
      match: { hw: "b200", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB200 / GB300 (sm100) — inferred. Single-node hosts carry 4 GPUs;
    // TP=1 uses one of them.
    // ====================================================================
    {
      match: { hw: "gb200", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // DGX Spark (GB10, sm121) — inferred. 1 GPU, unified memory; CUDA 13
    // build (dev-cu13 image). The CuTe conv kernel enables on sm120-class.
    // ====================================================================
    {
      match: { hw: "dgx-spark", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // AMD MI300X / MI325X (gfx942) / MI350X / MI355X (gfx950) — inferred.
    // The LFM3 overrides default the attention backend to aiter on HIP; the
    // flag is pinned explicitly here. The gated-conv blocks take the PyTorch
    // fallback path on ROCm (the CuTe DSL kernel is CUDA sm90/sm100/sm120).
    // ====================================================================
    {
      match: { hw: "mi300x", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--attention-backend aiter",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--attention-backend aiter",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--attention-backend aiter",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--attention-backend aiter",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--attention-backend aiter",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--attention-backend aiter",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--attention-backend aiter",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--attention-backend aiter",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
