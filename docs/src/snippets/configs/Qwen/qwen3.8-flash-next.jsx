// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Qwen3.8-Flash-Next: 176B total params (51B of that is the N-gram embedding
// table) with 6B active per token. Hybrid linear/full attention — 3 of every 4
// layers are Gated DeltaNet, the 4th is global attention running Qwen Sparse
// Attention (QSA) — over an ultra-sparse MoE, plus an in-checkpoint
// multi-step-trained MTP head. Multimodal (text + image in, text out).
//
// Every recipe on this page is single-node: BF16 and FP8 run TP4 on NVIDIA,
// NVFP4 runs on one Blackwell GPU, AMD FP8 runs on CDNA3 and CDNA4, and Quark
// MXFP4 is limited to CDNA4. The AMD paths use TP8+EP8. Plain TP8 is not a
// valid substitute for that expert-parallel topology on either checkpoint.
//
// A hardware x quantization x strategy combination with no launch recipe has no
// cell, and the engine greys it out.

export const config = {
  modelName: "Qwen3.8-Flash-Next",

  supportedHardware: [
    "h200", "b200", "b300", "gb300",
    "mi300x", "mi325x", "mi350x", "mi355x",
  ],

  variants: [
    { id: "default", label: "Default" },
  ],
  // Checkpoint precisions. NVFP4 is SGLang's own Blackwell-only quantization of
  // the BF16 weights (RadixArk). MXFP4 is AMD's Quark checkpoint for CDNA4.
  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
    { id: "mxfp4", label: "MXFP4" },
  ],
  // BF16, FP8, NVFP4, and MXFP4 expose low-latency and/or high-throughput
  // operating points. Low latency adds the in-checkpoint MTP head where tested.
  strategies: [
    { id: "low-latency",     label: "Low Latency"     },
    { id: "high-throughput", label: "High Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  // Orthogonal knobs — layered onto the matched cell, never part of the cell
  // key (see overlayDims in _deployment.jsx).
  overlayDims: [
    {
      id: "pleOffload",
      title: "PLE Offload",
      // Offloads the 51B N-gram embedding table to CPU pinned memory and
      // prefetches it on a side CUDA stream (--ple-offload-embedding). The
      // path is CUDA-only, so the row is hidden on the AMD cells. The server
      // already auto-enables it for BF16 on CUDA, hence the default chip is
      // Auto and adds no flag.
      showWhen: (sel) => !["mi300x", "mi325x", "mi350x", "mi355x"].includes(sel.hw),
      default: "auto",
      options: [
        { id: "auto", label: "Auto",
          hints: ["PLE Offload: auto-enabled for BF16 on CUDA, off otherwise"] },
        { id: "on",   label: "On",
          flags: ["--ple-offload-embedding"] },
        { id: "off",  label: "Off",
          flags: ["--no-ple-offload-embedding"] },
      ],
    },
  ],

  modelNames: {
    "default|bf16":  "Qwen/Qwen3.8-Flash-Next",
    // Separate repos, not revisions of the BF16 one.
    "default|fp8":   "Qwen/Qwen3.8-Flash-Next-FP8",
    "default|nvfp4": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "default|mxfp4": "amd/Qwen3.8-Flash-Next-Quark-MXFP4",
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

  // The "⚡ Reproduce" modal's benchmark command matches the AMD measurements
  // below. random-ids plus --tokenize-prompt fixes the submitted token count;
  // --random-range-ratio 1 fixes both requested lengths. The client ignores EOS
  // by default, and the explicit seed makes the generated token IDs repeatable.
  benchmarkCommands: {
    speed:
`HF_TOKEN="{{HF_TOKEN}}" python3 -m sglang.benchmark.serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --tokenizer {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --tokenize-prompt \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --request-rate inf --seed 42 \\
  --flush-cache --output-details`,
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  accuracyLabels: [
    ["gsm8k_pct",    "GSM8K",    "%"],
    ["aime26_pct",   "AIME26",   "%"],
    ["mmmu_pro_pct", "MMMU-Pro", "%"],
  ],

  // Launch images. AMD uses architecture-specific public images built from the
  // exact #36601 source revision used by the recipes below. The gfx942 image is
  // FP8-only in this cookbook; Quark MXFP4 remains a gfx950 recipe.
  dockerImages: {
    h200:   "lmsysorg/sglang:qwen38flashnext",
    b200:   "lmsysorg/sglang:qwen38flashnext",
    b300:   "lmsysorg/sglang:qwen38flashnext",
    gb300:  "lmsysorg/sglang:qwen38flashnext",
    mi300x: "aigmkt/qwen3.8-flash-next-gfx942-260827@sha256:89f79a33f48cc0f99c95902507643a86a181a5fffdcd9d8c61b66d9aacc44673",
    mi325x: "aigmkt/qwen3.8-flash-next-gfx942-260827@sha256:89f79a33f48cc0f99c95902507643a86a181a5fffdcd9d8c61b66d9aacc44673",
    mi350x: "aigmkt/qwen3.8-flash-next-gfx950-260827@sha256:51e4be1fde02780a5c39b37c464ebbceeffed5f6d307586f871611209a905828",
    mi355x: "aigmkt/qwen3.8-flash-next-gfx950-260827@sha256:51e4be1fde02780a5c39b37c464ebbceeffed5f6d307586f871611209a905828",
  },

  dockerGpuVendor: (sel) => ["mi300x", "mi325x", "mi350x", "mi355x"].includes(sel.hw)
    ? "amd" : "nvidia",
  dockerRunCommand: (sel) => ["mi300x", "mi325x", "mi350x", "mi355x"].includes(sel.hw)
    ? "python3 -m sglang.launch_server"
    : "sglang serve",
  runModes: (sel) => ["mi300x", "mi325x", "mi350x", "mi355x"].includes(sel.hw)
    ? ["docker"] : ["python", "docker"],

  github: {
    cookbookModel: "Qwen/Qwen3.8-Flash-Next",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // TP only. Every cell on the page is single-node TP (4 for BF16/FP8, 1 for
    // NVFP4, 8 for AMD FP8/MXFP4) with no DP-attention anywhere, and the values
    // stop at 8 because that is the widest single host here. CP and
    // DP-Attention are left out until there's a validated shape for them on
    // this checkpoint.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },

    // ----- Card: "MoE Parallelism" -----
    // EP degree only: the ultra-sparse MoE spreads its expert pool across ranks.
    // NVIDIA BF16/FP8 high-throughput uses EP4. The recommended eight-GPU AMD
    // FP8/MXFP4 topology uses EP8 so each rank receives complete expert
    // quantization groups. The AMD cells also pin AITER explicitly instead of
    // relying on backend auto-selection.
    moe: {
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    // ----- Card: "Parsers" -----
    // Both chips emit `auto`, so SGLang resolves the concrete parser from the
    // checkpoint's own chat template at startup (resolve_auto_parsers) instead
    // of the page pinning a name that a later chat-template revision could
    // outdate. Reasoning is baked into every NVIDIA cell — this model always
    // thinks, so a server without the parser returns the thinking inline — which
    // makes that chip an opt-OUT: the handler derives it as already-on and
    // strips the flag when toggled off. The tool-call chip is an opt-IN.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----
    // The in-checkpoint MTP head, trained with multiple steps. The tested AMD
    // implementation selects EAGLE; the NVIDIA recipes use NEXTN.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "MTP 3/1/4",
          flags: (sel) => [
            `--speculative-algorithm ${["mi300x", "mi325x", "mi350x", "mi355x"].includes(sel.hw) ? "EAGLE" : "NEXTN"}`,
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
          ] },
      ],
    },
  },

  // Ordering: the first cell seeds the Deploy panel's default selection.
  // Each cell carries its own verification state; inferred hardware variants
  // remain explicitly unverified.
  //
  // Within a quantization the NVIDIA cells are identical across
  // H200/B200/B300/GB300, and `--linear-attn-{prefill,decode}-backend flashinfer` is
  // pinned explicitly rather than left to the GDN default, which differs by GPU
  // generation (Triton on SM90, and the flashinfer decode default is gated on
  // `--mamba-ssm-dtype bfloat16`). Pinning both makes one recipe portable.
  cells: [
    // ==== BF16 ====
    // Low latency: MTP on, concurrency capped at 96. Without an explicit
    // --max-running-requests a speculative run takes the speculative hook's
    // default rather than a memory-derived ceiling.
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--linear-attn-verify-backend triton",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 96",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // High throughput: speculation off, EP4 across the same four ranks.
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 96",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 96",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // GB300 hosts are 4 GPUs, so TP4 is exactly one node.
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 96",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== FP8 ====
    // Two operating points per platform off one shape (TP4 + EP4): high
    // throughput as-is, low latency with the in-checkpoint MTP head added.
    // Unlike the BF16/NVFP4 low-latency cells these keep EP4 and pin no
    // --max-running-requests, so the panel's speculative hint applies.
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--linear-attn-verify-backend triton",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVFP4 (Blackwell only) ====
    // Single-GPU: the FP4 weights fit one card, so these run TP1 rather than the
    // TP4 the BF16/FP8 cells use. That leaves one rank, so there is no EP to
    // spend and the two tiers differ only by the MTP head — `ep_size *
    // moe_dp_size <= tp_size` would reject an --ep here.
    //
    // These are the measured commands verbatim, which is why --mem-fraction-static,
    // --chunked-prefill-size and --max-running-requests are absent where the
    // BF16/FP8 cells pin them: the runs left all three to their defaults (48
    // concurrent, 16384-token prefill chunks). That chunk size sits outside the
    // window the flashinfer GDN prefill default covers, so the explicit
    // --linear-attn-prefill-backend pin is what keeps prefill off Triton here.
    // No H200 cell: SM90 has no FP4 tensor cores.
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== AMD CDNA3 (MI300X / MI325X) ====
    // FP8 uses the same TP8+EP8 topology and exact #36601 source as CDNA4, in a
    // separate digest-pinned gfx942 image. These commands remain unverified
    // until the exact image/checkpoint pair is rerun on gfx942. MXFP4 is not
    // exposed on CDNA3.
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "This gfx942 FP8 command uses the exact #36601 image but has not been rerun end to end on MI300X.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--revision bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend aiter",
        "--moe-runner-backend aiter",
        "--page-size 64",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--max-running-requests 4",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "This command shares the MI300X gfx942 path but has not been independently rerun on MI325X.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--revision bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend aiter",
        "--moe-runner-backend aiter",
        "--page-size 64",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--max-running-requests 4",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== AMD CDNA4 (MI350X / MI355X) ====
    // FP8 and Quark MXFP4 were validated on eight MI350X GPUs with TP8+EP8,
    // explicit AITER attention/MoE, page size 64, radix cache disabled, and
    // full decode CUDA graphs. The FP8 evidence covers the EAGLE/MTP operating
    // point; MXFP4 also has a controlled non-speculative throughput result.
    {
      match: { hw: "mi350x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--revision bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend aiter",
        "--moe-runner-backend aiter",
        "--page-size 64",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--max-running-requests 4",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--revision 1ad7d941b239f6dc83cba6e49234c0efe1ca5477",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend aiter",
        "--moe-runner-backend aiter",
        "--page-size 64",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--max-running-requests 4",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 4",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", variant: "default", quant: "mxfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--revision 1ad7d941b239f6dc83cba6e49234c0efe1ca5477",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend aiter",
        "--moe-runner-backend aiter",
        "--page-size 64",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--max-running-requests 4",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "This command matches the validated MI350X gfx950 path but has not been rerun on MI355X.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--revision bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend aiter",
        "--moe-runner-backend aiter",
        "--page-size 64",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--max-running-requests 4",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      warn: "This command matches the validated MI350X gfx950 path but has not been rerun on MI355X.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--revision 1ad7d941b239f6dc83cba6e49234c0efe1ca5477",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend aiter",
        "--moe-runner-backend aiter",
        "--page-size 64",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--max-running-requests 4",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 4",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "mxfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "This command matches the validated MI350X gfx950 path but has not been rerun on MI355X.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--revision 1ad7d941b239f6dc83cba6e49234c0efe1ca5477",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend aiter",
        "--moe-runner-backend aiter",
        "--page-size 64",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--max-running-requests 4",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
