// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Qwen3.8-Flash-Next: 176B total params (51B of that is the N-gram embedding
// table) with 6B active per token. Hybrid linear/full attention — 3 of every 4
// layers are Gated DeltaNet, the 4th is global attention running Qwen Sparse
// Attention (QSA) — over an ultra-sparse MoE, plus an in-checkpoint
// multi-step-trained MTP head. Multimodal (text + image in, text out).
//
// Every datacenter recipe on this page is single-node: BF16 and FP8 run TP4 (so
// four GPUs of an 8-GPU H200/B200/B300 host, or a whole 4-GPU GB300 node), NVFP4
// runs on a single GPU, and the AMD cells run TP8. That fits because 6B active
// params keeps compute small and the N-gram table is the only large weight block.
// The one multi-node shape is NVFP4 on a pair of DGX Sparks (GB10): the 126 GiB
// checkpoint does not fit one 128 GB unified-memory box with the N-gram table
// resident, so it runs TP=2 across two of them over the 200GbE ConnectX-7 link;
// a single Spark serves it with the table file-backed on the local NVMe (the
// "On (NVMe file)" PLE Offload chip). NVFP4 also runs on one 96 GB
// RTX PRO 6000 Blackwell (SM120) once the 47.7 GiB FP8 N-gram table is offloaded
// to pinned host memory (--ple-offload-embedding), leaving the other 78 GiB of the checkpoint on
// the card.
//
// A hardware x quantization x strategy combination with no launch recipe has no
// cell, and the engine greys it out.

export const config = {
  modelName: "Qwen3.8-Flash-Next",

  supportedHardware: ["h200", "b200", "b300", "gb300", "rtx6000", "dgx-spark", "mi350x", "mi355x"],

  // RTX PRO 6000 (SM120, Blackwell workstation) is not in the shared
  // HARDWARE_CATALOG, so it carries a local vendor override here (same id and
  // label as the Qwen3.8-27B page).
  hardware: [
    { id: "rtx6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
  ],

  variants: [
    { id: "default", label: "Default" },
  ],
  // Checkpoint precisions. NVFP4 is SGLang's own Blackwell-only quantization of
  // the BF16 weights (RadixArk), so it has no H200 or AMD cell — SM90 and CDNA4
  // have no NVFP4 path. AMD serves the upstream BF16 and FP8 repos.
  // Two NVFP4 exports exist: RadixArk's (routed experts NVFP4, everything else
  // BF16 with an FP8 N-gram table) and NVIDIA's ModelOpt MIXED_PRECISION export
  // (NVFP4 experts, FP8 N-gram table, FP8 block-scaled MTP experts). The NVIDIA
  // one has recipes for the DGX Spark pair and the single RTX PRO 6000.
  quantizations: [
    { id: "bf16",       label: "BF16"         },
    { id: "fp8",        label: "FP8"          },
    { id: "nvfp4",      label: "NVFP4 (RDXA)" },
    { id: "nvfp4-nvda", label: "NVFP4 (NVDA)" },
  ],
  // BF16, FP8 and NVFP4 each ship two operating points, low latency adding the
  // in-checkpoint MTP head (NEXTN 3/1/4) on top of the high-throughput shape.
  // The two AMD platforms ship one recipe each, which parks under `balanced`.
  strategies: [
    { id: "low-latency",     label: "Low Latency"     },
    { id: "balanced",        label: "Balanced"        },
    { id: "high-throughput", label: "High Throughput" },
  ],
  // `multi-N` id carries the node count for `--nnodes N`; only the DGX Spark
  // NVFP4 cells use it.
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Node"  },
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
      //
      // DGX Spark (GB10) is unified memory: the pinned-host copy comes out of
      // the same 128 GB pool as the GPU weights, so offloading to RAM frees
      // nothing there and Auto (which resolves to on for this checkpoint) and
      // On are greyed out. The 2-node cells force Off (the TP-sharded table
      // fits in GPU memory); the single-Spark cells force "On (NVMe file)": the
      // file backend from sgl-project/sglang#37068 maps the table from a sparse
      // file on the local NVMe and the gather kernel reads it through the host
      // page tables, so it never has to be resident — the only way the 126 GiB
      // checkpoint boots on one 128 GB Spark. The engine snaps to the single
      // usable option in each case.
      //
      // RTX PRO 6000 is the opposite case: on a 96 GB discrete card the FP8
      // N-gram table (47.7 GiB) has to leave the GPU for the remaining 78 GiB
      // of weights plus the pools to fit, so Auto and Off are greyed out and On
      // is the only pick — the forced chip appends --ple-offload-embedding, so
      // the cells do not list it themselves.
      showWhen: (sel) => !["mi350x", "mi355x"].includes(sel.hw),
      default: "auto",
      options: [
        { id: "auto", label: "Auto",
          disabled: (sel) => sel.hw === "dgx-spark" || sel.hw === "rtx6000",
          disableReason: (sel) => sel.hw === "rtx6000"
            ? "RTX PRO 6000 (96 GB) only fits this checkpoint with the 47.7 GiB FP8 N-gram table in pinned host RAM; the verified cells pass --ple-offload-embedding explicitly, so On is the only pick."
            : "DGX Spark is unified memory: PLE offload to RAM frees nothing (the pinned table shares the 128 GB pool with the weights). The verified settings are Off for the two-node cells and On (NVMe file) for a single Spark.",
          hints: ["PLE Offload: auto-enabled for BF16 on CUDA, off otherwise"] },
        { id: "on",   label: "On",
          disabled: (sel) => sel.hw === "dgx-spark",
          disableReason: "DGX Spark is unified memory: PLE offload to RAM frees nothing (the pinned table shares the 128 GB pool with the weights). The verified settings are Off for the two-node cells and On (NVMe file) for a single Spark.",
          flags: ["--ple-offload-embedding"] },
        { id: "off",  label: "Off",
          disabled: (sel) => sel.hw === "rtx6000" || (sel.hw === "dgx-spark" && sel.nodes === "single"),
          disableReason: (sel) => sel.hw === "dgx-spark"
            ? "A single DGX Spark cannot hold the 126 GiB checkpoint in its 128 GB of unified memory; the verified single-Spark cells keep the 47.7 GiB FP8 N-gram table in a file on the local NVMe (On (NVMe file))."
            : "RTX PRO 6000 (96 GB) cannot hold the 47.7 GiB FP8 N-gram table alongside the other 78 GiB of the checkpoint; the table must be offloaded to pinned host RAM (On).",
          flags: ["--no-ple-offload-embedding"] },
        // File-backed table (sgl-project/sglang#37068, merged into qwen4-main-squashed):
        // a sparse 47.7 GiB file under $SGLANG_CACHE_DIR/ple/<model> (override
        // with --ple-offload-dir), created and filled by the server on boot and
        // read by the gather kernel through the host page tables. Requires the
        // device attribute cudaDevAttrPageableMemoryAccessUsesHostPageTables,
        // which GB10 has; hidden on other hardware. Verified only single-node —
        // the 2-node cells keep the table GPU-resident instead.
        { id: "file", label: "On (NVMe file)",
          showWhen: (sel) => sel.hw === "dgx-spark",
          disabled: (sel) => sel.nodes !== "single",
          disableReason: "The file-backed table is verified for the single-Spark cells; the 2-node cells shard the table across both GPUs instead (Off).",
          flags: ["--ple-offload-embedding", "--ple-offload-backend file"],
          hints: [
            "PLE table -> sparse 47.7 GiB file under $SGLANG_CACHE_DIR/ple/<model> (put it on local NVMe; --ple-offload-dir relocates it).",
            "Delete the previous table file before each boot until the rewrite is fixed upstream: rewriting a populated file runs at ~17 MB/s (~55 min), a fresh sparse file at GB/s (~8 min).",
          ] },
      ],
    },
  ],

  modelNames: {
    "default|bf16":  "Qwen/Qwen3.8-Flash-Next",
    // Separate repos, not revisions of the BF16 one.
    "default|fp8":   "Qwen/Qwen3.8-Flash-Next-FP8",
    "default|nvfp4": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "default|nvfp4-nvda": "nvidia/Qwen3.8-Flash-Next-NVFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"         },
    PORT:      { target: "command", label: "Bind port",         default: "30000"           },
    NODE0_IP:  { target: "command", label: "Head node IP",      default: "<node0-ip>"      },
    NODE_RANK: { target: "command", label: "This node rank",    default: "<node-rank>"     },
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
  // Prepended as `# ...` comments above multi-node commands.
  multiNodeHints: {
    "dgx-spark": [
      "Run the same command on both Sparks: rank 1 first, then rank 0 (node 0 = --dist-init-addr host).",
      "Point the rendezvous and NCCL at the ConnectX-7 link, not the management NIC:",
      "  NCCL_SOCKET_IFNAME=<200GbE-nic>  GLOO_SOCKET_IFNAME=<200GbE-nic>",
      "Cross-node decode CUDA graphs verified with the NCCL these images load (2.29.7 in dev-qwen38-next-local, 2.30.7 in qwen38flashnext);",
      "confirm with the startup log line 'sglang is using nccl=='.",
    ],
  },

  dockerImages: {
    h200:   "lmsysorg/sglang:qwen38flashnext",
    // DGX Spark and RTX PRO 6000 recipes need the qwen4-main-squashed build
    // (9b2aee2283: #38121 mixed-precision loader, file-backed PLE table); none
    // of them run on the qwen38flashnext image.
    "dgx-spark": "lmsysorg/sglang:dev-qwen38-next-local",
    rtx6000: "lmsysorg/sglang:dev-qwen38-next-local",
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

    // ==== NVFP4 on 2x DGX Spark (GB10, sm_121) — the only multi-node shape ====
    // One GB10 has 128 GB of unified memory and the NVFP4 checkpoint is 126 GiB
    // (78 GiB experts+dense, 47.7 GiB FP8 N-gram table), so a single Spark cannot
    // hold it; TP=2 across two Sparks over ConnectX-7 200GbE gives ~65 GB of
    // weights per node. Both cells are the model card's TP=2 recipe (modelopt_fp4,
    // flashinfer_cutlass FP4 GEMM, page 64, 4096-token prefill chunks, 262k
    // context) minus its `--mamba-track-interval 64` (the default 256 satisfies
    // the page-size/draft-token constraints and leaves a ~40% larger KV pool;
    // re-verified) and `--trust-remote-code` (not needed, the architecture is
    // native), with the concurrency pinned explicitly:
    // the hybrid model reserves mamba state slots per running request (5 with
    // the default extra_buffer strategy, 4 with extra_buffer_lazy), and the
    // scheduler silently caps --max-running-requests to what the mamba pool
    // admits unless --max-mamba-cache-size = requests x slots is set.
    // The PLE Offload row is forced to Off on this hardware (see overlayDims),
    // which appends --no-ple-offload-embedding: the FP8 table stays GPU-resident
    // and TP-sharded, since on unified memory the "offloaded" pinned-host copy
    // would come out of the same pool anyway. Verified 2026-09-04 on the qwen38flashnext image
    // (SGLang 593134d17a): 100k-token prefill 2,400-2,840 tok/s. GSM8K on the
    // full set pending.
    //
    // Low latency: in-checkpoint MTP head (NEXTN 3/1/4), 24 concurrent
    // requests (120 mamba slots), 1.48M-token KV pool, MTP accept length
    // 3.5-3.7 on non-thinking output. No env is required: the image loads its
    // pip NCCL by default (2.29.7 in dev-qwen38-next-local, 2.30.7 in
    // qwen38flashnext; verified via /proc/<pid>/maps), and the cell
    // passed the same checks without PYTORCH_CUDA_ALLOC_CONF; see the notes
    // for when expandable_segments is still worth setting.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      warn: "2x DGX Spark only (GB10 pair, TP=2 over ConnectX-7); in Docker mode use the lmsysorg/sglang:dev-qwen38-next-local image, the qwen4-main-squashed build the Spark rows are generated for. Memory headroom at --mem-fraction-static 0.85 is ~8-12 GiB per node; keep a host memory watchdog for long-context runs. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 24",
        "--max-mamba-cache-size 120",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // High throughput: speculation off, 96 concurrent requests. extra_buffer_lazy
    // allocates the mamba track buffer lazily (4 slots per request instead of
    // 5), so 384 slots admit 96 requests while leaving a 1.07M-token KV pool;
    // 332 tok/s aggregate on a 96-way chat workload.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      warn: "2x DGX Spark only (GB10 pair, TP=2 over ConnectX-7); in Docker mode use the lmsysorg/sglang:dev-qwen38-next-local image, the qwen4-main-squashed build the Spark rows are generated for. At 96 concurrent requests the KV pool is ~1.07M tokens (~11k per request when full); lower --max-running-requests for long-context workloads. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 96",
        "--max-mamba-cache-size 384",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVFP4 on 1x RTX PRO 6000 Blackwell (SM120, 96 GB) ====
    // A single 96 GB workstation card holds the NVFP4 checkpoint only with the
    // 47.7 GiB FP8 N-gram table offloaded to pinned host memory: the PLE Offload
    // row is forced to On on this hardware (see overlayDims), which appends
    // --ple-offload-embedding. The load log reports an 81 GiB delta (loader
    // temporaries included); once those are collected, 74.7 GiB stays resident
    // without speculative decoding and 81.8 GiB with it (the draft head is
    // 0.5 GiB, the rest is memory the loader still holds), leaving 19.4 / 12.3
    // GiB of the 94.2 GiB the process can use. --mem-fraction-static keeps
    // (1 - fraction) x 94.2 GiB of that as runtime slack and the pools take the
    // rest: 6.6 GiB slack + 12.8 GiB of pools at 0.93, 3.8 + 8.3 GiB at 0.96.
    // The host needs >= 64 GB of free RAM for the locked table and Docker
    // needs --ulimit memlock=-1.
    //
    // Both cells are the model card's recipe (modelopt_fp4, flashinfer_cutlass
    // FP4 GEMM and MoE runner, page 64, track interval 64, 4096-token prefill
    // chunks, 262k context) with prefix caching on and the concurrency pinned
    // explicitly. With the stock radix strategy (5 fp32 state slots per
    // request, 0.109 GiB each) the scheduler caps the card at 3 requests with
    // MTP and 12 without, so the cells use extra_buffer_lazy plus
    // SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 (3 slots per request: the running
    // request's prefix state is no longer pinned in the radix tree during
    // decode, a cache-retention trade, not a numerics change),
    // --mamba-ssm-dtype bfloat16 (halves the slot to 0.055 GiB), and pin
    // --max-mamba-cache-size = requests x 3. The linear-attention kernels are
    // left on auto: on SM120 that resolves to triton for decode, prefill and
    // verify (the bf16-state FlashInfer GDN auto-default is SM100-only), the
    // same kernel the fp32 runs and the DGX Spark cells use. An explicit
    // --linear-attn-decode-backend flashinfer also runs on SM120 and measured
    // the same TPOT and accuracy.
    //
    // With the state pool pinned, the KV pool takes whatever is left of the
    // static budget, so --mem-fraction-static is what sets the activation
    // headroom: 4096-token prefill chunks of real (ShareGPT-length) prompts
    // peak 1.5-2.6 GB above the post-graph-capture level, and cells left with
    // 2.4 GB OOMed in the GDN short-conv during prefill. The values below keep
    // >= 4 GB free after graph capture (>= 2.3 GB at the measured peak) and
    // were driven through a 1024-in/256-out random benchmark and a ShareGPT
    // chat sweep at every concurrency up to the pin.
    // Verified 2026-09-05 on the qwen38flashnext image (SGLang 593134d17a):
    // GSM8K (chat API, thinking off, n=200) 97.0% / 97.5% on two runs of the
    // low-latency cell, 98.0% / 97.0% on the high-throughput cell, inside the
    // 95-98% band of the datacenter runs. Re-verified 2026-09-06 on the
    // dev-qwen38-next-local image (qwen4-main-squashed 9b2aee2283), the image
    // the Docker tab now uses for this card: GSM8K 97.0% / 97.0%, 6.2 ms TPOT
    // at 1 request and 613 tok/s at 16 with MTP, 885 tok/s at 64 without,
    // same pools and peak headroom (2.8 GB / 4.0 GB left).
    //
    // Low latency: in-checkpoint MTP head (NEXTN 3/1/4), 16 concurrent
    // requests (48 state slots + 17 x 4 intermediate draft states, 6.4 GiB).
    // The draft states are what limit MTP concurrency on this card; at 0.96 the
    // KV pool is ~78k tokens (~4.9k per request when full) with 4.2 GB free
    // after graph capture. 1024-in/256-out: 5.9 ms TPOT at 1 request, 14.3 ms
    // at 8, 19.3 ms at 16 (vs 11.4 / - / 25.6 without MTP); MTP accept length
    // 3.3 of 4 on GSM8K / random prompts, 2.9 on long-form ShareGPT answers.
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "Single RTX PRO 6000 (96 GB). Use the lmsysorg/sglang:dev-qwen38-next-local image, the build this cell is verified on. The FP8 N-gram table lives in pinned host RAM: keep >= 64 GB of host memory free and run Docker with --ulimit memlock=-1. The KV pool is ~78k tokens (~4.9k per request at 16 concurrent); lower --max-running-requests for long-context work. See [RTX PRO 6000 notes](#rtx6000-note).",
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--moe-runner-backend flashinfer_cutlass",
        "--page-size 64",
        "--mamba-track-interval 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 16",
        "--max-mamba-cache-size 48",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.96",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // High throughput: speculation off, 64 concurrent requests (192 bf16 state
    // slots, 10.7 GiB). At 0.93 the KV pool is ~98k tokens (~1.5k per request
    // when full) with 5.3 GB free after graph capture (3.2 GB at the measured
    // peak). 1024-in/256-out at 64-way: 861 output tok/s, 3.4 req/s, 55 ms
    // TPOT; ShareGPT chat at 64-way: 1,258 output tok/s. (0.94 also passed
    // every check with a 138k-token pool and 2.3 GB at peak, if more KV per
    // request matters than headroom.)
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "Single RTX PRO 6000 (96 GB). Use the lmsysorg/sglang:dev-qwen38-next-local image, the build this cell is verified on. The FP8 N-gram table lives in pinned host RAM: keep >= 64 GB of host memory free and run Docker with --ulimit memlock=-1. At 64 concurrent requests the KV pool is ~98k tokens (~1.5k per request when full); lower --max-running-requests for long-context workloads. See [RTX PRO 6000 notes](#rtx6000-note).",
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--moe-runner-backend flashinfer_cutlass",
        "--page-size 64",
        "--mamba-track-interval 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 64",
        "--max-mamba-cache-size 192",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.93",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVFP4 (RDXA) on 1x DGX Spark — PLE table file-backed on NVMe ====
    // TP=1 on one GB10: the 78.3 GiB of experts and dense weights resident (~80 GB
    // in the log after load, including CUDA context and allocator overhead); the
    // 47.7 GiB FP8 N-gram table lives in a sparse file on the NVMe (the forced "On (NVMe file)" chip
    // appends --ple-offload-embedding --ple-offload-backend file). At
    // --mem-fraction-static 0.85 the two pools share ~12-18 GB, and at TP=1 a
    // mamba state slot is ~113 MB (fp32), so concurrency is pinned low:
    // 8 requests with MTP (5 slots each), 24 without on the lazy strategy (4
    // slots each). Verified 2026-09-06 on qwen4-main-squashed @ 9b2aee2283 (the
    // Python install path); boot ~10-11 min once the table file is fresh (see
    // the chip hint). GSM8K on the full set pending.
    //
    // Low latency: MTP head, 8 concurrent requests (40 slots), 93k-token KV
    // pool (~11.6k per request), 27.5 tok/s single stream (TPOT 33.6 ms),
    // 71.7 tok/s output at 8; MTP accept length 2.9-3.5 on non-thinking output.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "Single DGX Spark (GB10, 128 GB unified). The N-gram table is a 47.7 GiB sparse file on the local NVMe (PLE Offload = On (NVMe file)); keep ~50 GB free there and mount that directory into the container. Boot writes the whole table each time: delete the previous file first (a populated file rewrites at ~17 MB/s). Concurrency is memory-bound at 8 with MTP. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 8",
        "--max-mamba-cache-size 40",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // High throughput: speculation off, 24 concurrent requests (96 lazy slots),
    // 286k-token KV pool (~11.9k per request); 83 tok/s output at 24
    // (105 tok/s aggregate on a chat workload, vs 94 for the MTP cell at 8), 15.9 tok/s
    // single stream. MRR 16 / 64 slots was only +10% over the MTP cell, so 24.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "Single DGX Spark (GB10, 128 GB unified). The N-gram table is a 47.7 GiB sparse file on the local NVMe (PLE Offload = On (NVMe file)); keep ~50 GB free there and mount that directory into the container. Boot writes the whole table each time: delete the previous file first (a populated file rewrites at ~17 MB/s). At 24 concurrent requests the KV pool is ~286k tokens; lower --max-running-requests for long-context workloads. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 24",
        "--max-mamba-cache-size 96",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVFP4 (NVDA) on 2x DGX Spark — nvidia/Qwen3.8-Flash-Next-NVFP4 ====
    // ModelOpt MIXED_PRECISION export: NVFP4 routed experts, FP8 N-gram table,
    // FP8_BLOCK_SCALES (128-wide) MTP experts. Loading it needs the mixed-
    // precision loader from sgl-project/sglang#38121 (merged into
    // qwen4-main-squashed as 9b2aee2283, the branch the Python install path
    // builds); the qwen38flashnext image predates it. Same TP=2 shape and flags
    // as the RDXA cells, with two differences forced by this export:
    //   - `--quantization` is NOT passed (the checkpoint resolves to
    //     modelopt_mixed), and `--moe-runner-backend flashinfer_cutlass` is
    //     explicit: the modelopt_mixed auto-default picks flashinfer_trtllm on
    //     GB10, which the NVFP4 MoE method rejects at flashinfer autotune.
    //   - Low latency takes the MTP draft from the RadixArk export
    //     (`--speculative-draft-model-path` + `modelopt_fp4`): it is the same
    //     trained head kept in BF16 there, whereas this export's fp8
    //     block-scaled MTP experts cannot be TP-sharded (640/2 = 320 is not a
    //     multiple of the 128 block) and fault on the triton fp8 path under EP.
    // Measured 2026-09-05 on the qwen4-main-squashed tip 9b2aee2283 (#38121
    // merged), TP=2: bench (ISL 1024/OSL 256) ~48 tok/s single stream with MTP,
    // 253 tok/s output at 96 concurrent without. GSM8K on the full set pending.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      warn: "2x DGX Spark only (GB10 pair, TP=2 over ConnectX-7). Verified on the qwen4-main-squashed branch (the Python install path above). In Docker mode use the lmsysorg/sglang:dev-qwen38-next-local image (the qwen4-main-squashed build); the qwen38flashnext image predates the MIXED_PRECISION loader ([sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121)) and cannot load this export. The MTP draft is read from the RadixArk export (same head, BF16) because this export's fp8 block-scaled MTP experts cannot be split across two ranks. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_cutlass",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "--speculative-draft-model-quantization modelopt_fp4",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 24",
        "--max-mamba-cache-size 120",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      warn: "2x DGX Spark only (GB10 pair, TP=2 over ConnectX-7). Verified on the qwen4-main-squashed branch (the Python install path above). In Docker mode use the lmsysorg/sglang:dev-qwen38-next-local image (the qwen4-main-squashed build); the qwen38flashnext image predates the MIXED_PRECISION loader ([sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121)) and cannot load this export. At 96 concurrent requests the KV pool is ~1.1M tokens; lower --max-running-requests for long-context workloads. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_cutlass",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 96",
        "--max-mamba-cache-size 384",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVFP4 (NVDA) on 1x DGX Spark — nvidia/Qwen3.8-Flash-Next-NVFP4 ====
    // Same file-backed N-gram table and pins as the RDXA single-Spark cells
    // (memfrac 0.85; MTP: 8 requests / 40 slots, no MTP: 24 / 96 lazy slots),
    // with the export's own differences: no `--quantization` (resolves to
    // modelopt_mixed) and `--moe-runner-backend flashinfer_cutlass` explicit.
    // At TP=1 the in-checkpoint MTP head loads directly: its fp8 block-scaled
    // experts need no sharding, so the RadixArk draft used by the 2-node cell
    // is not needed. Verified 2026-09-06 on the dev-qwen38-next-local image
    // (9b2aee2283): MTP accept 3.0-3.6; boot ~11-12 min with a fresh table
    // file. GSM8K on the full set pending. The smaller fp8 draft leaves a
    // 174k-token KV pool with MTP (vs 93k for the RDXA cell) and 300k without.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "Single DGX Spark (GB10, 128 GB unified). Use the lmsysorg/sglang:dev-qwen38-next-local image: this ModelOpt MIXED_PRECISION export needs the loader from [sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121), which the qwen38flashnext image does not have. The N-gram table is a 47.7 GiB sparse file on the local NVMe (PLE Offload = On (NVMe file)); keep ~50 GB free there and mount that directory into the container. Boot writes the whole table each time: delete the previous file first (a populated file rewrites at ~17 MB/s). Concurrency is memory-bound at 8 with MTP. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--moe-runner-backend flashinfer_cutlass",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 8",
        "--max-mamba-cache-size 40",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "Single DGX Spark (GB10, 128 GB unified). Use the lmsysorg/sglang:dev-qwen38-next-local image: this ModelOpt MIXED_PRECISION export needs the loader from [sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121), which the qwen38flashnext image does not have. The N-gram table is a 47.7 GiB sparse file on the local NVMe (PLE Offload = On (NVMe file)); keep ~50 GB free there and mount that directory into the container. Boot writes the whole table each time: delete the previous file first (a populated file rewrites at ~17 MB/s). At 24 concurrent requests the KV pool is ~300k tokens; lower --max-running-requests for long-context workloads. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--moe-runner-backend flashinfer_cutlass",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 24",
        "--max-mamba-cache-size 96",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVFP4 (NVDA) on 1x RTX PRO 6000 — nvidia/Qwen3.8-Flash-Next-NVFP4 ====
    // Same ModelOpt MIXED_PRECISION export as the Spark NVDA cells; the loader
    // (sgl-project/sglang#38121) is in the dev-qwen38-next-local image the
    // Docker tab uses for this card. Same shape, pools, flags and headroom as
    // the RDXA cells above, with two differences:
    //   - no `--quantization` (the checkpoint resolves to modelopt_mixed);
    //   - low latency keeps the in-checkpoint MTP head. At TP=1 its fp8
    //     block-scaled experts need no sharding, and #38121 runs them on triton
    //     under the flashinfer_cutlass pin. The RadixArk BF16 draft
    //     (--speculative-draft-model-path) measured the same on this card
    //     (accept 3.33 vs 3.31, TPOT 18.5 vs 19.1 ms at 16), so the
    //     single-checkpoint command stays.
    // Verified 2026-09-06 on the dev-qwen38-next-local image (9b2aee2283),
    // TP=1, 1024-in/256-out random prompts. With MTP: 5.9 ms TPOT at 1
    // request, 18.7 ms / 634 tok/s at 16, accept length 3.4 of 4, GSM8K (chat
    // API, thinking off, n=200) 97.5% / 97.5% on two servers, 2.7 GB left at
    // peak. Without: 11.5 ms at 1, 25.1 ms at 16, 55.7 ms / 871 tok/s at 64,
    // GSM8K 96.5% / 97.0%, 4.0 GB left at peak. The smaller fp8 draft leaves a
    // ~170k-token KV pool with MTP (vs ~78k for the RDXA cell).
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "Single RTX PRO 6000 (96 GB). Use the lmsysorg/sglang:dev-qwen38-next-local image: this ModelOpt MIXED_PRECISION export needs the loader from [sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121), which the qwen38flashnext image does not have. The FP8 N-gram table lives in pinned host RAM: keep >= 64 GB of host memory free and run Docker with --ulimit memlock=-1. The KV pool is ~170k tokens (~10k per request at 16 concurrent). See [RTX PRO 6000 notes](#rtx6000-note).",
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--moe-runner-backend flashinfer_cutlass",
        "--page-size 64",
        "--mamba-track-interval 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 16",
        "--max-mamba-cache-size 48",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.96",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "Single RTX PRO 6000 (96 GB). Use the lmsysorg/sglang:dev-qwen38-next-local image: this ModelOpt MIXED_PRECISION export needs the loader from [sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121), which the qwen38flashnext image does not have. The FP8 N-gram table lives in pinned host RAM: keep >= 64 GB of host memory free and run Docker with --ulimit memlock=-1. At 64 concurrent requests the KV pool is ~98k tokens (~1.5k per request when full); lower --max-running-requests for long-context workloads. See [RTX PRO 6000 notes](#rtx6000-note).",
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--moe-runner-backend flashinfer_cutlass",
        "--page-size 64",
        "--mamba-track-interval 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 64",
        "--max-mamba-cache-size 192",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.93",
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
