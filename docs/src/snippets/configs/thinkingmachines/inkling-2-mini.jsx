// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr` literals — engine injects them.

export const config = {
  modelName: "Inkling-2-Mini",

  supportedHardware: ["h200", "gb300", "rtx6000", "rtx5090", "dgx-spark", "mi350x", "mi355x", "tpu7x"],

  hardware: [
    { id: "rtx6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "blackwell" },
    { id: "tpu7x", label: "TPU7x", vram: "192GB", vendor: "tpu" },
  ],

  dockerHostNetworkWhen: (s) => s.hw === "tpu7x",

  // Variant and Nodes are left out: one checkpoint, one node per recipe.
  matchDims: [
    {
      id: "quant",
      title: "Quantization",
      options: [
        // NVFP4 routed experts in layers 0-19, MXFP8 in layers 20-23.
        { id: "nvfp4", label: "NVFP4 + MXFP8" },
      ],
    },
    {
      // Plain decode, or DSpark speculative decoding on the bundled MTP head.
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "balanced", label: "Balanced" },
        { id: "dspark",   label: "DSpark"   },
      ],
    },
  ],

  modelNames: {
    default: "thinkingmachines/Inkling-2-Mini",
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

  dockerImages: {
    h200:  "lmsysorg/sglang:inkling-2-mini",
    gb300: "lmsysorg/sglang:inkling-2-mini",
    rtx6000: "lmsysorg/sglang:inkling-2-mini",
    rtx5090: "lmsysorg/sglang:inkling-2-mini",
    "dgx-spark": "lmsysorg/sglang:inkling-2-mini",
    mi350x: "lmsysorg/sglang:inkling-2-mini-mi35x",
    mi355x: "lmsysorg/sglang:inkling-2-mini-mi35x",
    tpu7x: "lmsysorg/sglang:inkling-2-mini-tpu",
  },

  github: {
    cookbookModel: "thinkingmachines/inkling-2-mini",
  },

  // Benchmark card: gsm8k only (sgl-eval 0.1.3, all 1319 questions), each run on its recipe command as shown.
  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  benchmarkCommands: {
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-examples 1319 --temperature 0 --max-tokens 8192 \\
  --num-threads 64`,
    },
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // TP only. TP2 was validated for the PD and HiCache roles on GB300.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 2, 4] },
      ],
    },

    // ----- Card: "MoE Parallelism" -----
    moe: {
      ep: { label: "EP", values: [null, 2, 4] },
    },

    // ----- Card: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser inkling" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser inkling2" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----  DSpark drives the bundled MTP draft.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (plain decode)" },
        { id: "dspark",  label: "DSpark MTP",
          flags: ["--speculative-algorithm DSPARK"] },
      ],
    },

    // ----- Card: "Loop" -----
    // Optional knobs over the loop core (layers 6-17, 3 passes by default):
    //   --inkling2-loop-num-passes N   (only on variable-loop checkpoints)
    //   --inkling2-loop-share-cache    (one K/V cache per core layer across passes)
    // Both flags need a variable-loop checkpoint (loop_config.is_varloop=true);
    // a fixed-loop checkpoint refuses them at startup. DSpark refuses a
    // non-default pass count, because the draft reads fixed layer visits.
    // Every option below was checked for output parity on GB300 and H200.
    // TPU7x: no server has run with either flag yet, so they are disabled there.
    flagSelects: [
      {
        id: "loopPasses", title: "Loop Passes",
        stripPrefixes: ["--inkling2-loop-num-passes"],
        options: [
          { id: "default", label: "3 (checkpoint default)", flags: [] },
          { id: "p1", label: "1", flags: ["--inkling2-loop-num-passes 1"],
            disable: [
              { when: { strategy: ["dspark"] }, reason: "DSpark drafts read fixed layer visits; serve them at the checkpoint's 3 passes." },
              { when: { hw: ["tpu7x"] }, reason: "Not yet run on TPU7x; the TPU7x recipes use the checkpoint's 3 passes." },
              { when: { hw: ["mi350x", "mi355x"] }, reason: "Not yet run on AMD; 2, 3 and 4 passes were." },
            ] },
          { id: "p2", label: "2", flags: ["--inkling2-loop-num-passes 2"],
            disable: [
              { when: { strategy: ["dspark"] }, reason: "DSpark drafts read fixed layer visits; serve them at the checkpoint's 3 passes." },
              { when: { hw: ["tpu7x"] }, reason: "Not yet run on TPU7x; the TPU7x recipes use the checkpoint's 3 passes." },
            ] },
          { id: "p4", label: "4", flags: ["--inkling2-loop-num-passes 4"],
            disable: [
              { when: { strategy: ["dspark"] }, reason: "DSpark drafts read fixed layer visits; serve them at the checkpoint's 3 passes." },
              { when: { hw: ["tpu7x"] }, reason: "Not yet run on TPU7x; the TPU7x recipes use the checkpoint's 3 passes." },
            ] },
        ],
      },
      {
        // One K/V cache per core layer instead of one per pass: about twice the
        // KV pool at 3 passes. Validated with DSpark, PD and HiCache.
        id: "loopShareCache", title: "Loop KV Cache",
        stripPrefixes: ["--inkling2-loop-share-cache"],
        options: [
          { id: "perPass", label: "Per pass (default)", flags: [] },
          { id: "shared",  label: "Shared across passes", flags: ["--inkling2-loop-share-cache"],
            disable: { hw: ["tpu7x"] },
            disableReason: "Not yet run on TPU7x; the TPU7x recipes keep one KV cache per pass." },
        ],
      },
    ],

    // ----- Card: "PD Disaggregation" -----
    // Validated as 1P1D at TP2 / EP2 per role on one 4-GPU GB300 tray (Mooncake
    // over NVLink) and on one MI350X node (NiXL). H200 and TPU7x have not been
    // run with PD roles.
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role", hide: { hw: ["h200", "tpu7x"] } },
        { id: "decode",  label: "Decode role",  hide: { hw: ["h200", "tpu7x"] } },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake",
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb300"] },
          // Over TCP on ROCm, Mooncake changed decode log-probabilities against a
          // colocated server; NiXL matched it token for token.
          hide: { hw: ["mi350x", "mi355x"] } },
        { id: "nixl", label: "NiXL",
          hide: { hw: ["h200", "gb300", "rtx6000", "rtx5090", "dgx-spark", "tpu7x"] } },
      ],
      router: {
        port: 8000,
        command:
`sgl-router \\
  --model-id {{MODEL_NAME}} \\
  --no-tokenizer \\
  --worker-urls http://<prefill-host>:{{PREFILL_PORT}} http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}}`,
      },
    },

    // ----- Card: "Hierarchical KV Cache" -----
    // The host tier was validated on GB300 (TP2) and MI350X (TP4); storage
    // backends were not run. Hidden on TPU7x: host offload has not been run there.
    hicache: {
      excludesHw: ["tpu7x"],
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
    {
      match: { hw: "gb300", quant: "nvfp4", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--attention-backend fa4",
        "--bf16-gemm-backend torch",
        "--dtype bfloat16",
        "--page-size 128",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", quant: "nvfp4", strategy: "dspark" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--attention-backend fa4",
        "--bf16-gemm-backend torch",
        "--dtype bfloat16",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--context-length 131072",
        "--max-running-requests 64",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", quant: "nvfp4", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--attention-backend fa4",
        "--bf16-gemm-backend torch",
        "--dtype bfloat16",
        "--page-size 128",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", quant: "nvfp4", strategy: "dspark" },
      nnodes: 1,
      verificationStatus: "in-progress",
      warn: "The GB300 DSpark recipe at the H200 Balanced parallelism; not yet run on H200.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--attention-backend fa4",
        "--bf16-gemm-backend torch",
        "--dtype bfloat16",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--context-length 131072",
        "--max-running-requests 64",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", quant: "nvfp4", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      env: ["ROCM_QUICK_REDUCE_QUANTIZATION=NONE"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--attention-backend triton",
        "--dtype bfloat16",
        "--page-size 128",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", quant: "nvfp4", strategy: "dspark" },
      nnodes: 1,
      verified: true,
      env: ["ROCM_QUICK_REDUCE_QUANTIZATION=NONE"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--attention-backend triton",
        "--dtype bfloat16",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "nvfp4", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      env: ["ROCM_QUICK_REDUCE_QUANTIZATION=NONE"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--attention-backend triton",
        "--dtype bfloat16",
        "--page-size 128",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "nvfp4", strategy: "dspark" },
      nnodes: 1,
      verified: true,
      env: ["ROCM_QUICK_REDUCE_QUANTIZATION=NONE"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--attention-backend triton",
        "--dtype bfloat16",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", quant: "nvfp4", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--ep 1",
        "--attention-backend fa4",
        "--dtype bfloat16",
        "--page-size 128",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", quant: "nvfp4", strategy: "dspark" },
      nnodes: 1,
      verified: true,
      env: [
        "SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1",
        "SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1",
        "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1",
        "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--ep 1",
        "--attention-backend fa4",
        "--dtype bfloat16",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--speculative-num-draft-tokens 8",
        "--max-running-requests 4",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx5090", quant: "nvfp4", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--ep 1",
        "--attention-backend fa4",
        "--dtype bfloat16",
        "--page-size 128",
        "--context-length 65536",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx5090", quant: "nvfp4", strategy: "dspark" },
      nnodes: 1,
      verified: true,
      env: [
        "SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1",
        "SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1",
        "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1",
        "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--ep 1",
        "--attention-backend fa4",
        "--dtype bfloat16",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--speculative-num-draft-tokens 8",
        "--context-length 32768",
        "--max-running-requests 4",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", quant: "nvfp4", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      env: [
        "MAX_JOBS=4",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--ep 1",
        "--attention-backend fa4",
        "--dtype bfloat16",
        "--page-size 128",
        "--mem-fraction-static 0.45",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", quant: "nvfp4", strategy: "dspark" },
      nnodes: 1,
      verified: true,
      env: [
        "MAX_JOBS=4",
        "SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1",
        "SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1",
        "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1",
        "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--ep 1",
        "--attention-backend fa4",
        "--dtype bfloat16",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--speculative-num-draft-tokens 8",
        "--mem-fraction-static 0.45",
        "--grammar-backend xgrammar",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling2",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // TPU7x (sglang-torchtpu): minimal command; everything else is left at its default.
    {
      match: { hw: "tpu7x", quant: "nvfp4", strategy: "dspark" },
      nnodes: 1,
      verified: true,
      warn: "Minimal SGLang-TorchTPU command; see [TPU7x deployment](#tpu).",
      env: [
        "SGLANG_TPU_INKLING2_NVFP4_EXPERTS=1",
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=sglang_tpu.multimodal",
        "SGLANG_EXTERNAL_MM_MODEL_ARCH=Inkling2LoopForConditionalGeneration",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--attention-backend pallas_tpu_rpa_v3",
        "--moe-runner-backend pallas_tpu_fused_rs",
        "--context-length 131072",
        "--mem-fraction-static 0.85",
        "--enable-torch-compile",
        "--speculative-algorithm DSPARK",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "tpu7x", quant: "nvfp4", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      warn: "The TPU7x DSpark command without speculative decoding, not yet run end to end; see [TPU7x deployment](#tpu).",
      env: [
        "SGLANG_TPU_INKLING2_NVFP4_EXPERTS=1",
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=sglang_tpu.multimodal",
        "SGLANG_EXTERNAL_MM_MODEL_ARCH=Inkling2LoopForConditionalGeneration",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep 8",
        "--attention-backend pallas_tpu_rpa_v3",
        "--moe-runner-backend pallas_tpu_fused_rs",
        "--context-length 131072",
        "--mem-fraction-static 0.85",
        "--enable-torch-compile",
        "--enable-multimodal",
        "--reasoning-parser inkling",
        "--tool-call-parser inkling",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
