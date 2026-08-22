// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Qwen3.8-27B: dense hybrid Gated Delta Networks vision-language model (48
// linear-attention + 16 full-attention layers, in-checkpoint MTP head, no MoE
// axis). Single-GPU on every supported card, hence one node and no
// parallelism flags in any cell.
//
// The RTX 5090 / RTX PRO 6000 cells and both overlay rows carry measured
// values (8192/1024, concurrency 1); the other platforms' cells are the
// source page's recipes. Verification per cell is documented above cells[].
// A hardware x quantization combination with no launch recipe has no cell,
// and the engine greys it out.

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

  // Every cell pins `--kv-cache-dtype fp8_e4m3` at the maintainers' direction
  // (sign-off recorded in the PR description). NVFP4: a no-op made visible
  // (the checkpoint's `kv_cache_quant_algo: FP8` already resolved `auto` to
  // fp8_e4m3). BF16/FP8: a real quality/capacity trade — halves
  // kv_bytes_per_token but those checkpoints carry no fp8 KV calibration.
  //
  // Speculative decoding and GDN state precision are orthogonal knobs, so
  // they are overlay rows, not match dims (3 x 2 would turn 12 cells into
  // 72). Options are inline, not `optionsKey` —
  // docs/scripts/check_cookbook_configs.mjs only reads `dim.options`.
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
      id: "spec",
      title: "Speculative Decoding",
      default: "none",
      options: [
        { id: "none", label: "None" },
        {
          id: "eagle", label: "EAGLE",
          // In-checkpoint MTP head; the only availability constraint is the
          // 32GB RTX 5090, where it needs the NVFP4 weights to leave room.
          disabled: (sel) => sel.hw === "rtx5090" && sel.quant !== "nvfp4",
          disableReason:
            "On the 32GB RTX 5090 the MTP head only fits on top of the NVFP4 weights",
          // EAGLE and DSPARK need opposite mem-fraction corrections on the
          // 5090 (EAGLE starves the state pool at boot and wants it UP;
          // DSpark starves runtime activations and wants it DOWN), so each
          // option strips the cell's value and re-pins its own.
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090" ? ["--mem-fraction-static"] : [],
          flags: (sel) => [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
            // ReplaySSM spec-verify only on SM120/SM121, where it was
            // exercised; it moves the D intermediate SSM states onto a fixed
            // ring (ratio computed with D=0) and is what makes MTP fit 32GB.
            // h200 (SM90) and gb300 (SM103) keep the plain MTP recipe.
            ...(["rtx5090", "rtx6000", "dgx-spark"].includes(sel.hw)
              ? ["--enable-linear-replayssm-spec"]
              : []),
            // Measured on the 5090: bf16 state serves at 0.92, fp32 needs
            // 0.94 (an fp32 slot is 146.81 MiB vs bf16's 74.81).
            ...(sel.hw === "rtx5090"
              ? [sel.ssmDtype === "float32"
                  ? "--mem-fraction-static 0.94"
                  : "--mem-fraction-static 0.92"]
              : []),
          ],
        },
        {
          id: "dspark", label: "DSPARK",
          // Separately-published trained draft model; same 5090 constraint as
          // EAGLE. No --min-free-slots-delay: at --max-running-requests 1 it
          // is a strict no-op, and its real semantic (disable the delayer)
          // would silently bite anyone raising concurrency to 8+.
          disabled: (sel) => sel.hw === "rtx5090" && sel.quant !== "nvfp4",
          disableReason:
            "On the 32GB RTX 5090 the DSpark draft model only fits on top of the NVFP4 weights",
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090" ? ["--mem-fraction-static"] : [],
          flags: (sel) => [
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark",
            "--speculative-draft-attention-backend flashinfer",
            // Measured on the 5090: bf16 state serves at 0.90, fp32 needs
            // 0.92 — the opposite correction to EAGLE's (see above).
            ...(sel.hw === "rtx5090"
              ? [sel.ssmDtype === "float32"
                  ? "--mem-fraction-static 0.92"
                  : "--mem-fraction-static 0.90"]
              : []),
          ],
        },
        {
          id: "dflash", label: "DFLASH2",
          // Trained block-diffusion draft, a separate checkpoint. The
          // selector projects through the target lm_head — including
          // quantized heads — so it runs on the NVFP4 checkpoint too.
          // Validated on the SM120 pair (NVFP4 measured end to end; the
          // RTX PRO 6000 BF16/FP8 cells boot-and-serve). The platforms where
          // it has not been exercised carry verificationStatus "in-progress"
          // on their cells.
          disabled: (sel) => sel.hw === "rtx5090" && sel.quant !== "nvfp4",
          disableReason:
            "On the 32GB RTX 5090 the DFlash2 draft model only fits on top of the NVFP4 weights",
          // fp32 is the one case that needs the balanced ratio overridden, so
          // that family is stripped too and re-emitted below.
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090"
              ? sel.ssmDtype === "float32"
                ? ["--mem-fraction-static", "--mamba-full-memory-ratio"]
                : ["--mem-fraction-static"]
              : [],
          flags: (sel) => [
            "--speculative-algorithm DFLASH",
            "--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2",
            "--speculative-num-draft-tokens 8",
            // Measured on the 5090 at commit 1cf2b8c, the build the Install
            // accordion pins for this pick. bf16 serves at 0.88 on the balanced
            // ratio (0.90, DSPARK's pin, OOMs on the first request). fp32 needs
            // 0.895 AND the balanced ratio overridden to 10: these cells pin
            // --max-running-requests 1, so the balanced value provisions KV for
            // concurrency this recipe never uses, starving the state pool of the
            // slots fp32 needs. Only High-Throughput reaches fp32 (S=4); the SSM
            // dtype row greys fp32 out for Low-Latency (S=5).
            ...(sel.hw === "rtx5090"
              ? sel.ssmDtype === "float32"
                ? ["--mem-fraction-static 0.895",
                   "--mamba-full-memory-ratio 10"]
                : ["--mem-fraction-static 0.88"]
              : []),
          ],
        },
      ],
    },
    {
      // Serving tier via the GDN radix-cache strategy, the knob that sets S
      // (state slots per running request): extra_buffer S=5 (latency tier),
      // extra_buffer_lazy S=4 (throughput tier — fewer slots, more requests
      // per pool). The calculator reads this row for the balanced ratio.
      // The default equals the ENGINE default (extra_buffer), so an untouched
      // selection matches each platform's original recipe; both options were
      // characterized on RTX 5090 / RTX PRO 6000 only.
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
      // biggest lever on the state pool that bounds concurrency on small-VRAM
      // cards. The default equals the ENGINE default (float32, the
      // checkpoint's declared precision); both precisions were characterized
      // on RTX 5090 / RTX PRO 6000 only.
      id: "ssmDtype",
      title: "Mamba SSM Dtype",
      default: "float32",
      options: [
        // Open on every platform except the 32GB RTX 5090 under DFLASH2:
        // there the fp32 state pool and the prefill CUDA-graph capture cannot
        // both fit, at any mem-fraction. Measured on main (2026-08-21, ratio
        // pinned at 10 so slots are not the binding term): 0.945 and 0.92 OOM
        // inside `Capture target prefill CUDA graph`, 0.90 OOMs on the first
        // request, and 0.88 / 0.86 / 0.84 size the state pool below the 5 (LL)
        // / 4 (HT) slots one request needs. bf16 state halves the pool and
        // serves, and is the faster cell there anyway.
        {
          id: "float32", label: "float32",
          // Only the Low-Latency tier is out of reach: it needs S=5 fp32 slots
          // (735MB) plus >=9216 KV tokens for one request, and no mem-fraction
          // holds both -- at 0.8975/r14 the pool buys the 5th slot but KV falls
          // to 7752 tokens and generation stops after one token, while every
          // mem-fraction with a big enough pool (>=0.90) dies in graph capture.
          // High-Throughput needs one slot fewer and does fit; see the DFLASH2
          // option's pins.
          disabled: (sel) =>
            sel.hw === "rtx5090" &&
            sel.spec === "dflash" &&
            sel.tier === "low-latency",
          disableReason:
            "On the 32GB RTX 5090 the Low-Latency tier cannot hold five fp32 state " +
            "slots and a full request's KV at once — use bfloat16, or the " +
            "High-Throughput tier which fits fp32",
          flags: ["--mamba-ssm-dtype float32"],
        },
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

  // The source page's measurement protocol, kept reproducible:
  // --random-range-ratio 1 pins ISL exactly; --flush-cache measures
  // cache-cold (bench_serving's `random` prompts are deterministic).
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

    // No "Attention Parallelism" card: the page is single-GPU throughout,
    // and a TP knob would desync the ratio calculator (TP1-only geometry).
    // Re-add it together with TP-aware geometry in the calculator.

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
        { id: "dflash",  label: "DFlash2",
          // Same trio as the Deploy panel's DFLASH2 option.
          flags: ["--speculative-algorithm DFLASH",
                  "--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2",
                  "--speculative-num-draft-tokens 8"] },
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

  // Verification: RTX 5090 / RTX PRO 6000 cells were measured across their
  // whole overlay envelope; the h200/gb300 badges carry the source page's
  // validation, which covers the overlay defaults (plus plain MTP on gb300) —
  // non-default overlay picks there are valid but unmeasured. DGX Spark was
  // measured across its whole overlay envelope too, but to a weaker standard
  // (boot-and-serve only — see the cell block comment below).
  //
  // Cells carry NO --mamba-full-memory-ratio: the ratio depends on workload,
  // S, D and kv_bytes_per_token, so the page's calculator computes it live
  // and the engine pins it into the rendered command — which it only does
  // while the cell stays ratio-free (_deployment.jsx `cellWithRatio`).
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
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
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
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
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
      // RTX 5090 32GB. NVFP4 is the only checkpoint that fits (FP8 does not
      // boot — total_rest_memory negative at every mem-fraction, measured —
      // and BF16 does not fit). Published operating point is ONE request in
      // flight; --cuda-graph-max-bs 1 also protects the token pool (default
      // capture set costs 39,247 -> 37,347 and K 8 -> 7). The `warn` below
      // carries the user-facing guidance for raising concurrency.
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
    // cell. These cells reuse the RTX PRO 6000 recipe verbatim rather than a
    // separate SM121 operating point: both cards are SM12x Blackwell, and
    // GB10's 128GB unified pool is larger than the 6000's 96GB, so a recipe
    // that fits the smaller card has headroom here.
    //
    // Validated on GB10 (SM121 / aarch64): all 36 configurations booted and
    // served at ISL 8192 / OSL 1024, concurrency 1. Boot-and-serve only -- no
    // throughput or acceptance-length numbers were taken, so this is a weaker
    // standard than the SM120 pair's validation, and the Deploy-panel Note says
    // so.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", nodes: "single" },
      // All 12 overlay combinations served on GB10. DSPARK here also
      // exercises the 4-bit `lm_head` this checkpoint quantizes, with no shape
      // error.
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
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
      match: { hw: "dgx-spark", variant: "default", quant: "fp8", nodes: "single" },
      // All 12 overlay combinations served on GB10.
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
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
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", nodes: "single" },
      // All 12 overlay combinations served on GB10. Heaviest checkpoint, so
      // it holds the sweep's tightest cell: DSPARK + float32 + extra_buffer.
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
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
    // GB300 (SM103), single 288GB GPU. Base and plain-MTP arms measured
    // 2026-08-14 on lmsysorg/sglang:dev @ c4271c3fe (attention resolves to
    // engine default — no pin, so cell and measurement see the same kernel).
    // Verified envelope: spec none|eagle at engine-default tier/state dtype.
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", nodes: "single" },
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
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
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
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
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
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
