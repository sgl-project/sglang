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
// over: there is no sibling `-benchmarks.jsx`. The RTX 5090 / RTX PRO 6000
// cells are `verified: true` outright — their whole overlay envelope (tier,
// state dtype, both spec options) was measured. The h200/gb300 cells carry
// `verified` as a FUNCTION of the selection instead: their validation covers
// only the untouched overlay defaults (plus plain MTP on gb300), so any other
// overlay pick flips the badge to Not Verified rather than borrowing a green
// badge from a different configuration. The DGX Spark cells stay unverified
// everywhere: that recipe is unvalidated on SM121 / aarch64, as both §2 and
// the cell comment below say.
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

  // BF16/FP8 are the official Qwen checkpoints; NVFP4 is the RadixArk
  // W4A4 build. Every cell pins `--kv-cache-dtype fp8_e4m3` explicitly at
  // the maintainers' direction (recorded in the PR description). For NVFP4
  // this is a no-op made visible: that checkpoint declares
  // `kv_cache_quant_algo: FP8`, so `auto` already resolved to fp8_e4m3 off its
  // own calibration scales. For BF16/FP8 it is a real change — it halves
  // kv_bytes_per_token (65.5 KB -> 32.8 KB) and doubles the KV pool at a fixed
  // --mamba-full-memory-ratio, but those checkpoints carry no fp8 KV
  // calibration, so it is a quality/capacity trade, not free.
  //
  // Speculative decoding and GDN state precision are ORTHOGONAL knobs, so they
  // are overlay rows, not match dims: they layer flags onto the matched cell
  // instead of multiplying the cell count (3 x 2 would turn 12 cells into 72).
  // The former `strategies` row (Balanced / High-Throughput) is gone — its only
  // real content was "does the MTP head run", which the Speculative Decoding row
  // now states directly.
  // Options are inline, not `optionsKey`: the engine accepts either, but
  // docs/scripts/check_cookbook_configs.mjs only reads `dim.options`, so an
  // optionsKey form fails CI with "not an option of that dim" on every cell.
  matchDims: [
    { id: "variant", title: "Model Variant", options: [
      { id: "default", label: "Default" },
    ] },
    { id: "quant", title: "Quantization", options: [
      { id: "bf16",  label: "BF16"  },
      { id: "fp8",   label: "FP8"   },
      { id: "nvfp4", label: "NVFP4" },
    ] },
    { id: "nodes", title: "Nodes", options: [
      { id: "single", label: "Single Node" },
    ] },
  ],

  overlayDims: [
    {
      // EAGLE is the in-checkpoint MTP head (the page's old `high-throughput`
      // tier); DSpark is a separately-published trained draft model. Both run
      // on every quantization AND every card except one: the 32GB RTX 5090,
      // where only NVFP4 leaves room for a draft head on top of the weights
      // (FP8 is ~28.5GB and BF16 does not fit at all, so neither has a cell
      // there in the first place). Same predicate as the GDN-state row below.
      id: "spec",
      title: "Speculative Decoding",
      default: "none",
      options: [
        { id: "none", label: "None" },
        {
          id: "eagle", label: "EAGLE",
          // EAGLE / MTP is the in-checkpoint head and is offered on every
          // platform. The only availability constraint is the 32GB RTX 5090,
          // where the head needs the NVFP4 weights to leave room. The
          // ReplaySSM flag is scoped separately in `flags` below -- that is a
          // per-platform validation question, not an availability one.
          disabled: (sel) => sel.hw === "rtx5090" && sel.quant !== "nvfp4",
          disableReason:
            "On the 32GB RTX 5090 the MTP head only fits on top of the NVFP4 weights",
          // ReplaySSM spec-verify rides with EAGLE on the platforms where it
          // has been exercised — SM120/SM121, scoped in `flags` below; h200
          // and gb300 keep the plain MTP recipe. It replaces the per-draft
          // full-state snapshots with a fold-every-commit ring, which takes the D
          // intermediate SSM states off the per-request slot budget (so the
          // mamba ratio is computed with D=0, not the draft-token count --
          // see the compute-mamba-ratio skill's ReplaySSM caveat). On the
          // 32GB 5090 that is the difference between booting and a negative
          // total_rest_memory. Requires a linear draft chain (topk 1, set
          // just above) and a triton/flashinfer linear-attn decode backend,
          // which is the default everywhere here.
          // MEASURED on the 5090 (8192/1024, conc 1, radix on): the cell's
          // 0.9 does NOT boot EAGLE -- the state pool lands at K=1 against a
          // required ratio of 4 -- and 0.92 does. The two spec options fail in
          // OPPOSITE directions on this card (EAGLE starves the state pool at
          // boot and wants mem-fraction UP; DSpark starves runtime activations
          // and wants it DOWN), so one cell value cannot serve both. Strip and
          // re-pin per option rather than emit the flag twice.
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090" ? ["--mem-fraction-static"] : [],
          flags: (sel) => [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
            // ReplaySSM spec-verify ONLY on SM120 (RTX PRO 6000, RTX 5090)
            // and SM121 (DGX Spark), where it has been exercised. It moves the
            // D intermediate SSM states onto a fixed ring and is what makes MTP
            // fit a 32GB card at all. h200 (SM90) and gb300 (SM103) keep the
            // plain MTP recipe rather than inheriting an untested flag.
            ...(["rtx5090", "rtx6000", "dgx-spark"].includes(sel.hw)
              ? ["--enable-linear-replayssm-spec"]
              : []),
            // MEASURED: bf16 state serves at 0.92; fp32 state does NOT (K=3
            // against a required 4) and needs 0.94. fp32 slots are 146.81 MiB
            // vs bf16's 74.81, so the state pool needs a bigger slice.
            ...(sel.hw === "rtx5090"
              ? [sel.ssmDtype === "float32"
                  ? "--mem-fraction-static 0.94"
                  : "--mem-fraction-static 0.92"]
              : []),
          ],
        },
        {
          id: "dspark", label: "DSPARK",
          // One 5090 constraint, same predicate as EAGLE above: the separate
          // draft checkpoint only fits on top of the NVFP4 weights. Both GDN
          // state precisions are open with DSpark on this card — fp32 needs
          // the higher 0.92 mem-fraction re-pinned below, bf16 serves at 0.90.
          disabled: (sel) => sel.hw === "rtx5090" && sel.quant !== "nvfp4",
          disableReason:
            "On the 32GB RTX 5090 the DSpark draft model only fits on top of the NVFP4 weights",
          // MEASURED on the 5090: DSpark boots at the cell's 0.9 but then OOMs
          // at RUNTIME on the first prefill (75 MB free of 31.36 GB), because
          // the static pools leave nothing for an 8192-token single-chunk
          // prefill. 0.88 serves cleanly. Note this is the OPPOSITE direction
          // to EAGLE above -- see that comment.
          // NO --min-free-slots-delay here. Its semantic is *disable* the
          // MinFreeSlotsDelayer, and at the cell's --max-running-requests 1 it
          // is a strict no-op: resolve_min_free_slots (managers/
          // min_free_slots_delayer.py:4-22) returns None both ways, since the
          // DFlash auto-enable needs mrr >= 8 and an explicit 1 fails its own
          // `threshold > 1` test. Shipping it would only bite later — anyone
          // raising --max-running-requests to 8+ would silently lose the
          // delayer SGLang auto-enables for this (DFlash-family) algorithm.
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090" ? ["--mem-fraction-static"] : [],
          flags: (sel) => [
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark",
            "--speculative-draft-attention-backend flashinfer",
            // MEASURED at the engine-default 2048 prefill chunk: bf16 state
            // serves at 0.90 (0.88 was only needed to dodge the old 8192-chunk
            // runtime OOM) and fp32 state needs 0.92 -- at 0.90/0.88 its state
            // cache lands K=3/K=2 against a required 4.
            ...(sel.hw === "rtx5090"
              ? [sel.ssmDtype === "float32"
                  ? "--mem-fraction-static 0.92"
                  : "--mem-fraction-static 0.90"]
              : []),
          ],
        },
      ],
    },
    {
      // Serving tier, expressed through the GDN radix-cache strategy -- the
      // knob that sets S, state slots reserved per running request
      // (kv_cache_configurator._calculate_mamba_ratio):
      //   extra_buffer       S = 5   track buffer reserved up front
      //   extra_buffer_lazy  S = 4   allocated lazily at the track boundary
      // Fewer slots per request means more requests fit the same state pool,
      // so lazy is the throughput tier and the eager buffer the latency tier.
      // S also feeds the balanced ratio, and the page's calculator reads this
      // row directly (strategy -> slots), so the two stay in step.
      // The default is the ENGINE default (extra_buffer), so an untouched
      // selection emits a command semantically identical to each platform's
      // original recipe. Both options were characterized on RTX 5090 and
      // RTX PRO 6000 (the full 12-cell grid); on other platforms picking the
      // non-default option is a valid but unmeasured opt-in, and the verified
      // badge reports it as such (see the cells' `verified` functions).
      id: "tier",
      title: "Serving Strategy",
      default: "low-latency",
      // Owns the flag outright: strip whatever a cell pinned, then re-emit.
      stripPrefixes: ["--mamba-radix-cache-strategy"],
      options: [
        { id: "low-latency", label: "Low-Latency",
          flags: ["--mamba-radix-cache-strategy extra_buffer"] },
        { id: "high-throughput", label: "High-Throughput",
          flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
      ],
    },
    {
      // One GDN state slot is 146.81 MiB at fp32 and 74.81 MiB at bf16 — the
      // single biggest lever on the state pool, which is what bounds
      // concurrency on small-VRAM cards. On SM120 both precisions run on the
      // Triton linear-attn prefill path (the FlashInfer GDN prefill fast path
      // gates on SM100, so Triton is the SM120 default regardless of dtype).
      // The default is the ENGINE default (float32, the checkpoint's declared
      // precision), so an untouched selection matches each platform's original
      // recipe. Both precisions were characterized on RTX 5090 and RTX PRO
      // 6000; elsewhere bfloat16 is a valid but unmeasured opt-in and the
      // verified badge reports it as such.
      id: "ssmDtype",
      title: "Mamba SSM Dtype",
      default: "float32",
      options: [
        // float32 is open on every platform including the 5090. An fp32 slot
        // is 146.81 MB against bf16's 74.81, so it roughly doubles both the
        // state pool and the balanced ratio; on a 32GB card that is a real
        // squeeze but it serves. bf16 remains the better default there (bigger
        // KV pool at identical latency) -- this is a choice, not a trap.
        // float32 is open everywhere, including with DSPARK on the 5090.
        // It was blocked on evidence gathered at --chunked-prefill-size 8192;
        // once the cell dropped to the engine default (2048) the runtime
        // activation pressure fell and DSpark+fp32 serves at mf 0.92.
        { id: "float32", label: "float32", flags: ["--mamba-ssm-dtype float32"] },
        {
          id: "bfloat16", label: "bfloat16",
          disabled: (sel) => sel.hw === "rtx5090" && sel.quant !== "nvfp4",
          disableReason:
            "On the 32GB RTX 5090 the bf16 GDN state pool is only a live choice for NVFP4; " +
            "the BF16 and FP8 checkpoints have no serviceable cell on this card",
          flags: ["--mamba-ssm-dtype bfloat16"],
        },
      ],
    },
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
    // Multi-arch: this tag ships both linux/amd64 and linux/arm64, so it pulls
    // natively on DGX Spark (GB10 is aarch64).
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
          // Same three flags as the Deploy panel's DSPARK option, so the two
          // paths compose identical commands.
          flags: ["--speculative-algorithm DSPARK",
                  "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark",
                  "--speculative-draft-attention-backend flashinfer"] },
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
        // KV pool at a fixed --mamba-full-memory-ratio. Every deployment cell
        // now pins fp8_e4m3, so this row is the opt-OUT: pick BFloat16 to undo
        // it on the BF16/FP8 checkpoints, which carry no fp8 KV calibration.
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
      match: { hw: "h200", variant: "default", quant: "fp8", nodes: "single" },
      // Verified only at the untouched overlay defaults — the source page's
      // recipe has no speculative decoding and engine-default strategy/state
      // dtype. Any other overlay pick is a valid but unmeasured opt-in, and
      // the badge reports it as Not Verified.
      verified: (sel) =>
        sel.spec === "none" && sel.tier === "low-latency" &&
        sel.ssmDtype === "float32",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
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
      match: { hw: "h200", variant: "default", quant: "bf16", nodes: "single" },
      // Same verified envelope as the FP8 cell above.
      verified: (sel) =>
        sel.spec === "none" && sel.tier === "low-latency" &&
        sel.ssmDtype === "float32",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
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
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
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
      match: { hw: "rtx6000", variant: "default", quant: "fp8", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
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
      match: { hw: "rtx6000", variant: "default", quant: "bf16", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
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
      // serve; FP8 does not boot (total_rest_memory goes negative at every
      // mem-fraction, measured) and BF16 does not fit, so neither has a cell.
      // The published operating point is ONE request in flight:
      // --max-running-requests 1 caps admission at the validated envelope, and
      // --cuda-graph-max-bs 1 keeps graph capture from taxing the token pool
      // (the default capture set costs 39,247 -> 37,347 pool tokens and
      // K 8 -> 7, for batch shapes this card cannot admit anyway). To serve
      // more concurrency, raise BOTH pins together and re-derive the ratio and
      // mem-fraction with the calculator — on this card the GDN state pool,
      // not KV, is what bounds concurrency (lower S via the radix-cache
      // strategy, or prefix cache off for S=1).
      match: { hw: "rtx5090", variant: "default", quant: "nvfp4", nodes: "single" },
      verified: true,
      // Rendered with the cell so nobody ships the bs=1 pins into a
      // multi-user deployment unaware.
      warn:
        "This recipe serves ONE request at a time: --max-running-requests 1 " +
        "and --cuda-graph-max-bs 1 pin it to the validated single-stream " +
        "envelope. To handle more concurrent requests, raise both flags " +
        "together and re-derive --mamba-full-memory-ratio (and mem-fraction) " +
        "with the [Mamba ratio calculator](#mamba-ratio-calculator) — on this " +
        "32GB card the GDN state pool, not KV, is what runs out first.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.9",
        "--attention-backend flashinfer",
        "--max-running-requests 1",
        "--cuda-graph-max-bs 1",
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
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
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
      match: { hw: "dgx-spark", variant: "default", quant: "fp8", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
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
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", nodes: "single" },
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
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
    // GB300 (SM103), single 288GB GPU. The base and MTP arms were measured on
    // a 4xGB300 devbox on 2026-08-14 against
    // lmsysorg/sglang:dev @ c4271c3fe1262fc2adbd162c33b25de5255251c5.
    // With no --attention-backend pin, :dev on GB300 resolves attention to
    // triton (the newer c7c03ec resolves trtllm_mha); the cells keep
    // engine-default resolution so cell and measurement see the same kernel —
    // which also means the tier/ssmDtype overlays must stay at their
    // engine-default picks for the badge to read Verified. MTP here is the
    // Speculative Decoding row's EAGLE option, plain (no ReplaySSM on SM103).
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", nodes: "single" },
      // Verified for the measured arms: no-spec and plain MTP, both at
      // engine-default strategy/state dtype. DSPARK and the non-default
      // tier/ssmDtype picks are unmeasured on this card.
      verified: (sel) =>
        (sel.spec === "none" || sel.spec === "eagle") &&
        sel.tier === "low-latency" && sel.ssmDtype === "float32",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", nodes: "single" },
      // Same verified envelope as the NVFP4 cell above.
      verified: (sel) =>
        (sel.spec === "none" || sel.spec === "eagle") &&
        sel.tier === "low-latency" && sel.ssmDtype === "float32",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", nodes: "single" },
      // Same verified envelope as the NVFP4 cell above.
      verified: (sel) =>
        (sel.spec === "none" || sel.spec === "eagle") &&
        sel.tier === "low-latency" && sel.ssmDtype === "float32",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
