// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Qwen3.8-Flash-Next: 176B total params (51B of that is the N-gram embedding
// table) with 6B active per token. Hybrid linear/full attention — 3 of every 4
// layers are Gated DeltaNet, the 4th is global attention running Qwen Sparse
// Attention (QSA) — over an ultra-sparse MoE, plus an in-checkpoint
// multi-step-trained MTP head. Multimodal (text + image in, text out).
//
// Every recipe on this page is single-node: BF16 and FP8 run TP4 (so four GPUs
// of an 8-GPU H200/B200/B300 host, or a whole 4-GPU GB300 node), NVFP4 runs on
// a single GPU, and the AMD cells run TP8. That fits because 6B active params
// keeps compute small and the N-gram table is the only large weight block.
//
// A hardware x quantization x strategy combination with no launch recipe has no
// cell, and the engine greys it out.

export const config = {
  modelName: "Qwen3.8-Flash-Next",

  supportedHardware: ["h200", "b200", "b300", "gb300", "mi350x", "mi355x"],

  variants: [
    { id: "default", label: "Default" },
  ],
  // Checkpoint precisions. NVFP4 is SGLang's own Blackwell-only quantization of
  // the BF16 weights (RadixArk), so it has no H200 or AMD cell — SM90 and CDNA4
  // have no NVFP4 path. AMD serves the upstream BF16 and FP8 repos.
  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
  ],
  // BF16, FP8 and NVFP4 each ship two operating points, low latency adding the
  // in-checkpoint MTP head (NEXTN 3/1/4) on top of the high-throughput shape.
  // The two AMD platforms ship one recipe each, which parks under `balanced`.
  strategies: [
    { id: "low-latency",     label: "Low Latency"     },
    { id: "balanced",        label: "Balanced"        },
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
      showWhen: (sel) => !["mi350x", "mi355x"].includes(sel.hw),
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

  // The "⚡ Reproduce" modal's benchmark command. --random-range-ratio 1 pins ISL
  // exactly rather than drawing a range, so runs stay comparable.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang-oai \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --request-rate inf \\
  --flush-cache`,
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  accuracyLabels: [
    ["gsm8k_pct",    "GSM8K",    "%"],
    ["aime26_pct",   "AIME26",   "%"],
    ["mmmu_pro_pct", "MMMU-Pro", "%"],
  ],

  // Launch images — this is a day-0 model with no release cut, so both tags are
  // purpose-built rather than a version. The ROCm build targets CDNA4 (gfx950)
  // and is not interchangeable with the CUDA one.
  dockerImages: {
    h200:   "lmsysorg/sglang:qwen38flashnext",
    b200:   "lmsysorg/sglang:qwen38flashnext",
    b300:   "lmsysorg/sglang:qwen38flashnext",
    gb300:  "lmsysorg/sglang:qwen38flashnext",
    mi350x: "lmsysorg/sglang-rocm:qwen38flashnext",
    mi355x: "lmsysorg/sglang-rocm:qwen38flashnext",
  },

  github: {
    cookbookModel: "Qwen/Qwen3.8-Flash-Next",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // TP only. Every cell on the page is single-node TP (4 for BF16/FP8, 1 for
    // NVFP4, 8 on AMD) with no DP-attention anywhere, and the values stop at 8
    // because that is the widest single host here. CP and DP-Attention are left
    // out until there's a validated shape for them on this checkpoint.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },

    // ----- Card: "MoE Parallelism" -----
    // EP degree only: the ultra-sparse MoE spreads its expert pool across ranks,
    // and the BF16/FP8 high-throughput cells already pair TP4 with EP4 (the TP1
    // NVFP4 cells have only one rank, so EP stays 1 there). No a2a/runner
    // backend row — the cells leave `--moe-a2a-backend` and
    // `--moe-runner-backend` unset so the runner resolves from the checkpoint's
    // own quant_method, and no alternative backend is validated here yet.
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
    // The in-checkpoint MTP head, trained with multiple steps. Spelled NEXTN
    // with the same 3/1/4 numbers the low-latency cells carry, so the chip
    // derives as already-on there instead of showing a phantom diff.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "NEXTN / MTP",
          flags: ["--speculative-algorithm NEXTN", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
      ],
    },
  },

  // Every cell below is a verified recipe. Ordering: the first cell seeds the
  // Deploy panel's default selection.
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

    // ==== AMD CDNA4 (MI350X / MI355X) ====
    // One recipe, identical for BF16 and FP8 and for both cards (same gfx950,
    // same 288GB, same ROCm image) — hence `balanced` on all four cells. This is
    // its own shape rather than a port of the NVIDIA one: TP8, the aiter
    // attention backend with `--page-size 32`, and a 16384-token prefill chunk.
    // `--kv-cache-dtype auto` is stated rather than left off so the checkpoint's
    // own declaration is visibly what decides KV precision.
    {
      match: { hw: "mi350x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--attention-backend aiter",
        "--page-size 32",
        "--kv-cache-dtype auto",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--attention-backend aiter",
        "--page-size 32",
        "--kv-cache-dtype auto",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--attention-backend aiter",
        "--page-size 32",
        "--kv-cache-dtype auto",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--attention-backend aiter",
        "--page-size 32",
        "--kv-cache-dtype auto",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
