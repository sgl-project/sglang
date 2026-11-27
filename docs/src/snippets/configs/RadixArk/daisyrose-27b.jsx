// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// DaisyRose-27B is Qwen3.8-27B post-trained with reinforcement learning: same
// architecture, tensor shapes, tokenizer and chat template, so every recipe is
// the Qwen3.8-27B one for the same card and precision. The NVFP4 export packs
// the lm_head to FP4 and declares no KV scheme, like nvidia/Qwen3.8-27B-NVFP4,
// so its cells and RTX 5090 pins are that export's.
//
// Measured with DaisyRose weights: GB300, all three precisions, DSPARK. Every
// other cell carries the Qwen3.8-27B recipe and is marked unverified here.

export const config = {
  modelName: "DaisyRose-27B",

  supportedHardware: ["gb300", "h200", "rtxpro6000", "rtx5090", "dgx-spark"],

  hardware: [
    { id: "rtxpro6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "blackwell" },
  ],

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
      default: "dspark",
      options: [
        { id: "none", label: "None" },
        {
          id: "eagle", label: "EAGLE",
          // EAGLE and DSPARK need opposite mem-fraction corrections on the
          // 32GB RTX 5090, so each option strips the cell's value and re-pins
          // its own.
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090"
              ? ["--mem-fraction-static", "--mamba-full-memory-ratio",
                 "--max-total-tokens"]
              : [],
          flags: (sel) => [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
            ...(["rtx5090", "rtxpro6000", "dgx-spark"].includes(sel.hw)
              ? ["--enable-linear-replayssm-spec"]
              : []),
            ...(sel.hw === "rtx5090"
              ? [sel.ssmDtype === "float32"
                  ? "--mem-fraction-static 0.94"
                  : "--mem-fraction-static 0.93"]
              : []),
          ],
        },
        {
          id: "dspark", label: "DSPARK",
          // The draft model is trained on Qwen3.8-27B; on DaisyRose-27B it is
          // accepted for 3.4-3.5 tokens per step.
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090"
              ? ["--mem-fraction-static", "--mamba-full-memory-ratio",
                 "--chunked-prefill-size", "--max-total-tokens"]
              : [],
          flags: (sel) => [
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark",
            "--speculative-draft-attention-backend flashinfer",
            // On the 32GB RTX 5090 this row owns the pools: the draft model's
            // weights count against --mem-fraction-static, and the default
            // split leaves the GDN state pool short of its slots.
            ...(sel.hw === "rtx5090"
              ? [
                  "--max-total-tokens 16384",
                  ...(sel.ssmDtype === "float32"
                    ? ["--mem-fraction-static 0.91",
                       "--chunked-prefill-size 1024",
                       sel.tier === "low-latency"
                         ? "--mamba-full-memory-ratio 6.94"
                         : "--mamba-full-memory-ratio 5.56"]
                    : ["--mem-fraction-static 0.88",
                       sel.tier === "low-latency"
                         ? "--mamba-full-memory-ratio 3.38"
                         : "--mamba-full-memory-ratio 3.12"]),
                ]
              : []),
          ],
        },
      ],
    },
    {
      // The GDN radix-cache strategy sets S, the state slots per running
      // request: extra_buffer S=5, extra_buffer_lazy S=4.
      id: "tier",
      title: "Serving Strategy",
      default: "high-throughput",
      stripPrefixes: ["--mamba-radix-cache-strategy"],
      options: [
        { id: "low-latency", label: "Low-Latency",
          flags: ["--mamba-radix-cache-strategy extra_buffer"] },
        { id: "high-throughput", label: "High-Throughput",
          flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
      ],
    },
    {
      id: "ssmDtype",
      title: "Mamba SSM Dtype",
      default: "float32",
      options: [
        { id: "float32",  label: "float32",  flags: ["--mamba-ssm-dtype float32"] },
        { id: "bfloat16", label: "bfloat16", flags: ["--mamba-ssm-dtype bfloat16"] },
      ],
    },
  ],
  modelNames: {
    "default|bf16":  "RadixArk/DaisyRose-27B",
    "default|fp8":   "RadixArk/DaisyRose-27B-FP8",
    "default|nvfp4": "RadixArk/DaisyRose-27B-NVFP4",
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

  // --random-range-ratio 1 pins ISL exactly; --flush-cache measures cache-cold.
  benchmarkCommands: {
    speed:
`python3 -m sglang.benchmark.serving \\
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
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --eval-name gsm8k \\
  --num-examples 1319`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    h200:    "lmsysorg/sglang:latest",
    rtxpro6000: "lmsysorg/sglang:latest",
    rtx5090: "lmsysorg/sglang:latest",
    "dgx-spark": "lmsysorg/sglang:latest",
    gb300:   "lmsysorg/sglang:latest",
  },

  github: {
    cookbookModel: "RadixArk/DaisyRose-27B",
  },

  playgroundFeatures: {

    // ----- Card: "Parsers" -----
    // Baked into every cell, so this card is an opt-out: without the parsers
    // a harness receives tool calls as raw text.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
        { id: "dspark",  label: "DSpark",
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

  // Cells carry no --mamba-full-memory-ratio: the page's calculator computes
  // it live and the engine pins it into the rendered command.
  cells: [
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", nodes: "single" },
      verified: true,
      verificationStatus: (sel) =>
        sel.spec === "dspark" ? "verified" : "unverified",
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
      verificationStatus: (sel) =>
        sel.spec === "dspark" ? "verified" : "unverified",
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
      match: { hw: "gb300", variant: "default", quant: "nvfp4", nodes: "single" },
      verified: true,
      verificationStatus: (sel) =>
        sel.spec === "dspark" ? "verified" : "unverified",
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
      match: { hw: "h200", variant: "default", quant: "fp8", nodes: "single" },
      verified: false,
      verificationStatus: "unverified",
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
      match: { hw: "h200", variant: "default", quant: "bf16", nodes: "single" },
      verified: false,
      verificationStatus: "unverified",
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
      match: { hw: "rtxpro6000", variant: "default", quant: "nvfp4", nodes: "single" },
      verified: false,
      verificationStatus: "unverified",
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
      match: { hw: "rtxpro6000", variant: "default", quant: "fp8", nodes: "single" },
      verified: false,
      verificationStatus: "unverified",
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
      match: { hw: "rtxpro6000", variant: "default", quant: "bf16", nodes: "single" },
      verified: false,
      verificationStatus: "unverified",
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
      match: { hw: "rtx5090", variant: "default", quant: "nvfp4", nodes: "single" },
      verified: false,
      verificationStatus: "unverified",
      warn:
        "This recipe serves ONE request at a time: --max-running-requests 1 " +
        "and --cuda-graph-max-bs-decode 1 pin it to the validated single-stream " +
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
        "--cuda-graph-max-bs-decode 1",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", nodes: "single" },
      verified: false,
      verificationStatus: "unverified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.80",
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
      verified: false,
      verificationStatus: "unverified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.80",
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
      verified: false,
      verificationStatus: "unverified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.80",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
