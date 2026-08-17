// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Qwen3.8-27B: DENSE hybrid Gated Delta Networks VISION-LANGUAGE model — a 27B
// causal LM plus a vision encoder, served through SGLang's Qwen3-VL path
// (Qwen3_5ForConditionalGeneration extends Qwen3VLForConditionalGeneration and
// is registered in the multimodal arch lists). 64 layers as 16 repeats of
// 3 x (Gated DeltaNet -> FFN) then 1 x (Gated Attention -> FFN): 48
// linear-attention layers to 16 full-attention. GDN runs 48 value heads and 16
// QK heads at head_dim 128; attention is GQA 24/4 at head_dim 256. An MTP head
// trained with multiple steps ships in-checkpoint. Context 262,144 native,
// extensible to 1,000,000. Dense, so there is no MoE axis.
//
// Single-GPU on every supported card — H200 (SM90 datacenter), the SM120
// workstation pair (RTX PRO 6000 Blackwell / RTX 5090), and DGX Spark (GB10,
// SM121, 128GB unified memory) — hence one node and no parallelism flags in
// any cell.
//
// PROVENANCE — every flag and value below is transcribed from the
// pre-migration prose page (one unconditional launch command + its Configuration
// Tips). Model ids are the exception: BF16/FP8 point at the official Qwen
// checkpoints, NVFP4 at the RadixArk W4A4 build. That page pinned
// no sglang version for its measurements, so under the
// migration skill's reproducible-anchor rule NO measured numbers were carried
// over: there is no sibling `-benchmarks.jsx`. Cells are nevertheless marked
// `verified: true` at the maintainers' direction — the badge there reflects
// their own unpublished validation, not measured data carried by this page. The
// DGX Spark cells are the exception and stay unverified: that recipe is
// unvalidated on SM121 / aarch64, as both §2 and the cell comment below say.
// `benchmarkCommands` below records the page's measurement protocol so the
// numbers can be re-measured against a pinned build and then land as a
// benchmarks file.
//
// A hardware x quantization combination with no launch recipe has no cell, and
// the engine greys it out.

export const config = {
  modelName: "Qwen3.8-27B",

  supportedHardware: ["h200", "rtx6000", "rtx5090", "dgx-spark", "gb300"],

  // RTX PRO 6000 and RTX 5090 (SM120 / Blackwell Desktop) are workstation and
  // consumer cards, not datacenter GPUs, so they are not in the shared catalog.
  // Ids/labels match the DeepSeek-V4 config's entries for the same two cards.
  // DGX Spark needs no entry here: it is already in the shared catalog
  // (_deployment.jsx HARDWARE_CATALOG), with its multi-node Docker flags.
  hardware: [
    { id: "rtx6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "blackwell" },
  ],

  variants: [
    { id: "default", label: "Default" },
  ],
  // BF16/FP8 are the official Qwen checkpoints; NVFP4 is the RadixArk
  // W4A4 build. NVFP4 is W4A4 with FP8 projections and declares
  // `kv_cache_quant_algo: FP8`, so under the default `--kv-cache-dtype auto` its
  // KV pool runs fp8_e4m3 off the checkpoint's own calibration scales — no
  // `--kv-cache-dtype` flag in the recipe, and nothing accuracy-degrading added
  // by the cell.
  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
  ],
  // The source page documents ONE operating point: a single general-purpose
  // launch command with no latency/throughput toggle. MTP is described as an
  // opt-in in the tips, not as a second named recipe, so it rides the
  // Playground's speculative axis instead of splitting the strategy dimension.
  strategies: [
    { id: "balanced", label: "Balanced" },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16":  "Qwen/Qwen3.8-27B",
    "default|fp8":   "Qwen/Qwen3.8-27B-FP8",
    "default|nvfp4": "RadixArk/Qwen3.8-27B-NVFP4",
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

  // The measurement protocol the source page described, kept so its numbers can
  // be reproduced against a pinned build. --random-range-ratio 1 pins ISL
  // exactly rather than drawing a range; --flush-cache measures cache-cold
  // (bench_serving's `random` prompts are deterministic, so a warm rerun would
  // hit the radix cache and inflate throughput) — the page's own "prefix caching
  // disabled" protocol.
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
    accuracy: {
      gsm8k_pct:
`python3 -m sglang.test.run_eval \\
  --host http://{{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --eval-name gsm8k \\
  --num-examples 1319`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    h200:    "lmsysorg/sglang:qwen38-27b",
    rtx6000: "lmsysorg/sglang:qwen38-27b",
    rtx5090: "lmsysorg/sglang:qwen38-27b",
    // TODO: verify an arm64 build of this tag for DGX Spark (GB10 is aarch64);
    // the x86-only tag will not pull there.
    "dgx-spark": "lmsysorg/sglang:qwen38-27b",
    gb300:   "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "Qwen/Qwen3.8-27B",
  },

  playgroundFeatures: {

    // No "Attention Parallelism" card. The source page is single-GPU
    // throughout and no cell carries a parallelism flag, so there is nothing to
    // override: DP-Attention targets MLA models, prefill-CP has no model-side
    // integration for this architecture, and a TP knob would desync the ratio
    // calculator below (its geometry is TP1-only, so it would stop emitting and
    // the command would silently fall back to the 0.9 default this page warns
    // about). Re-add it together with TP-aware geometry in the calculator.

    // ----- Card: "Parsers" -----
    // Same parser pair the Qwen3.8 flagship page ships, and baked into every
    // cell: this model is used through agent harnesses, and a deploy command
    // without them returns tool calls as raw text instead of structured
    // `tool_calls`. So this card is an opt-OUT — the handler derives both chips
    // as already-on from the cell and strips the flag when one is toggled off.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----
    // The in-checkpoint 1-layer MTP head. The source page wrote the preset as
    // `--speculative-algorithm NEXTN`; NEXTN is an alias of EAGLE, so it is
    // normalized here (the Playground strips/derives by the first token, and an
    // alias would survive toggles and double up). DSpark is the trained draft
    // model, a separate checkpoint.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
        { id: "dspark",  label: "DSpark",
          flags: ["--speculative-algorithm DSPARK",
                  "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark"] },
      ],
    },

    // ----- Card: single-selects over one flag family each -----
    flagSelects: [
      {
        // trtllm_mha is SM100-only, so every cell bakes flashinfer (SM90 and
        // SM120 alike). FA3 is the SM90 alternative — measured slightly faster
        // at bs=1 on H200. Triton is the documented fallback when MTP runs on a
        // FlashInfer build whose prefill `plan` predates `uniform_q_len`
        // (<= 0.6.15.post1).
        id: "attnBackend", title: "Attention Backend",
        stripPrefixes: ["--attention-backend"],
        options: [
          { id: "flashinfer", label: "FlashInfer (default)",
            flags: ["--attention-backend flashinfer"] },
          { id: "fa3", label: "FlashAttention-3 (SM90 only)",
            flags: ["--attention-backend fa3"] },
          { id: "triton", label: "Triton — MTP fallback on older FlashInfer",
            flags: ["--attention-backend triton"] },
        ],
      },
      {
        // Halving kv_bytes_per_token (65.5 KB bf16 -> 32.8 KB fp8) doubles the
        // KV pool at a fixed --mamba-full-memory-ratio. Accuracy-degrading over
        // a bf16-KV checkpoint, so it stays an opt-in and is never in a cell —
        // the NVFP4 checkpoint gets fp8 KV on its own via kv_cache_quant_algo.
        id: "kvCacheDtype", title: "KV Cache Precision",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "auto", label: "Auto (checkpoint-declared)" },
          { id: "fp8",  label: "FP8 (E4M3) — halves KV memory", flags: ["--kv-cache-dtype fp8_e4m3"] },
          { id: "bf16", label: "BFloat16", flags: ["--kv-cache-dtype bfloat16"] },
        ],
      },
      {
        // One state slot is 154.7 MB at fp32, 79.2 MB at bf16 — the single
        // biggest lever on the GDN state pool, which is what bounds concurrency
        // on small-VRAM cards.
        id: "mambaSsmDtype", title: "GDN State Precision",
        stripPrefixes: ["--mamba-ssm-dtype"],
        options: [
          { id: "auto", label: "Auto (FP32)" },
          { id: "bf16", label: "BFloat16 — halves state memory", flags: ["--mamba-ssm-dtype bfloat16"] },
        ],
      },
      {
        // Whole-model prefix cache. Off drops the per-request state cost to
        // S=1 slot, which is the cheapest way to buy concurrency on a 32GB card
        // when the traffic has no shared prefixes (offline batch, evals).
        id: "prefixCache", title: "Prefix Cache",
        stripPrefixes: ["--disable-radix-cache"],
        options: [
          { id: "on",  label: "On" },
          { id: "off", label: "Off (S=1)", flags: ["--disable-radix-cache"] },
        ],
      },
      {
        // How GDN state buffers for radix reuse — slot cost per request S:
        // extra_buffer 5, extra_buffer_lazy 4, no_buffer 3. No strategy exists
        // with the prefix cache off, so the row hides (and stops emitting) there.
        // Changing S changes the balanced ratio — recompute it in the page's
        // calculator after picking a strategy here.
        id: "mambaRadix", title: "GDN Radix Cache Strategy",
        showWhen: (b, v, d) => (((v && v.prefixCache) ?? (d && d.prefixCache)) !== "off"),
        stripPrefixes: ["--mamba-radix-cache-strategy"],
        options: [
          { id: "auto",  label: "Auto (extra_buffer, S=5)" },
          { id: "lazy",  label: "extra_buffer_lazy (S=4)", flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
          { id: "nobuf", label: "no_buffer (S=3)",         flags: ["--mamba-radix-cache-strategy no_buffer"] },
        ],
      },
    ],
  },

  // Every cell is the source page's single launch command with only the model id
  // varying. The H200 and SM120 cells are `verified: true` at the maintainers'
  // direction; the DGX Spark cells are not, matching the unvalidated-on-SM121
  // note on those cells. The page carries no measured data of its own (no
  // `-benchmarks.jsx`), so a badge rests on validation held outside this page —
  // re-check it against the per-platform notes in §2 before trusting a cell.
  //
  // Cells carry NO --mamba-full-memory-ratio. The source page's worked 4.6 held
  // only for the NVFP4 recipe at 4096-in/1024-out; the ratio is a function of
  // the workload, of S (radix-cache strategy / prefix cache), of D (spec) and of
  // kv_bytes_per_token, all of which the Playground can change. So the page's
  // ratio calculator computes it live from the effective config and broadcasts
  // it, and the engines pin it into the rendered command — which they only do
  // while the cell itself stays ratio-free (_deployment.jsx `cellWithRatio`).
  // Adding the flag back here would silently freeze the value again.
  cells: [
    {
      // H200 141GB, FP8 blockwise (~28.5GB of weights). The 32768-token chunk
      // is the H200-validated setting: SM90 prefill is fast enough that a big
      // chunk stalls decode far less than on SM120, and the SM90 FlashInfer GDN
      // prefill default engages under it (fp32 state pool, chunk <= 32768).
      // No NVFP4 cell on this card: SM90 has no FP4 tensor cores, so the W4A4
      // checkpoint's MLP would fall back to the Marlin W4A16 weight-only path —
      // runnable, but not a recipe this page ships.
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // H200, BF16 reference checkpoint (~54GB of weights).
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // The page's headline recipe: NVFP4 W4A4 on the 96GB workstation card,
      // ~16.5GB of weights, fp8 KV auto-enabled by the checkpoint.
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // FP8 blockwise, ~28.5GB of weights — comfortable on 96GB.
      match: { hw: "rtx6000", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // BF16, the reference checkpoint.
      match: { hw: "rtx6000", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // RTX 5090 32GB. NVFP4 is the only checkpoint that fits with room to
      // serve (~16.5GB); FP8 at ~28.5GB is not serviceable past bs<=2 and BF16
      // does not fit, so neither has a cell. On this card the GDN state pool —
      // not KV — bounds concurrency: lower S with the Playground's radix-cache
      // strategy (or turn the prefix cache off for S=1) and recompute the ratio.
      match: { hw: "rtx5090", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DGX Spark (GB10, SM121): single node, 128GB coherent unified memory
    // shared with the CPU — every checkpoint fits, so all three quants get a
    // cell. FlashInfer attention comes from the SM120 pair; the platform gets
    // its own operating point at 8192-token prefill chunks, 0.95 static
    // fraction, and prefill CUDA graphs disabled. Unvalidated on SM121 /
    // aarch64.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.95",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 8192",
        "--disable-prefill-cuda-graph",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.95",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 8192",
        "--disable-prefill-cuda-graph",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.95",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 8192",
        "--disable-prefill-cuda-graph",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // GB300 (SM103), single 288GB GPU. All six cells measured on a 4xGB300
    // devbox on 2026-08-14 against lmsysorg/sglang:dev @ c4271c3fe1262fc2adbd162c33b25de5255251c5.
    // With no --attention-backend pin, :dev on GB300 resolves attention to
    // triton (the newer c7c03ec resolves trtllm_mha); the cells keep engine-default
    // resolution so the benchmark card and the cell see the same kernel. The
    // `high-throughput` strategy adds the in-checkpoint MTP head
    // (EAGLE / NEXTN semantics, num-steps 3, topk 1, draft-tokens 4).
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
