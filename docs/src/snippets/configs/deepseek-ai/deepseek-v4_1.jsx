// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// Cells marked `verified` are transcribed from recorded runs on that hardware with
// real weights.
//
// Every DSpark cell caps --cuda-graph-max-bs-decode: the derived batch list does
// not fit while capturing the DSpark decode graphs, on any platform. H200 is the
// tightest board at 140 GiB and needs the cap on both cells plus a lower memory
// fraction; the B200 / B300 / GB300 High-Throughput cells start without either.
//
// DP-Attention, DeepEP and MegaMoE are absent by design: they have never been
// enabled on this model. EP is set equal to TP on every shape except MI355X.

export const config = {
  modelName: "DeepSeek-V4.1",

  latencyPercentile: "P50",

  supportedHardware: ["h200", "b200", "b300", "gb300", "mi355x"],

  // Hardware is the implicit first match dim; Strategy is the only other one.
  // Declaring matchDims replaces variants / quantizations / nodesOptions
  // wholesale, which drops three rows that each had exactly one option.
  matchDims: [
    {
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency",     label: "Low-Latency"     },
        { id: "high-throughput", label: "High-Throughput" },
      ],
    },
  ],

  modelNames: {
    default: "deepseek-ai/DeepSeek-V4.1-Flash",
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
    // DeepSeek-V4.1 support has not shipped in a CUDA release yet; the ROCm
    // images carry it from v0.5.21.
    h200:  "lmsysorg/sglang:dev-dsv41",
    b200:  "lmsysorg/sglang:dev-dsv41",
    b300:  "lmsysorg/sglang:dev-dsv41",
    gb300: "lmsysorg/sglang:dev-dsv41",
    mi355x: "lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261006",
  },

  github: {
    cookbookModel: "deepseek-ai/deepseek-v4.1",
  },

  playgroundFeatures: {

    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 2, 4, 8] },
      ],
    },

    // No backend chooser: `flashinfer_mxfp4` is selected automatically and is the
    // only MoE runner this model has run on. EP tracks TP.
    moe: {
      ep: { label: "EP", values: [null, 1, 4, 8] },
    },

    // Both parsers default to None; without them the DSML tool-call block and the
    // thinking block arrive as raw text inside `content`.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    // DSpark is the model's bundled 3-stage draft. There is no EAGLE/MTP path and
    // no `--speculative-num-steps` knob.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "dspark",  label: "DSpark",
          flags: ["--speculative-algorithm DSPARK", "--speculative-dspark-block-size 5"] },
      ],
    },

    pdDisagg: {
      incompatibleSpeculativeAlgorithms: ["DSPARK"],
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        // Fallback for hosts where the RDMA fabric is not visible in the container:
        // Mooncake then picks its NVLink transport, which only serves buffers from
        // its own allocator and fails to find the peer address.
        { id: "mooncake", label: "Mooncake (TCP)",
          env: ["MOONCAKE_PROTOCOL=tcp", "MC_FORCE_TCP=1"] },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }],
      router: {
        port: 8000,
        command:
`sglang-router launch \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}}`,
      },
    },

    // GPU → CPU KV offload (L2 only; no storage tier). Hidden on MI355X: not
    // validated on ROCm for this model yet.
    hicache: {
      excludesHw: ["mi355x"],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },

    flagSelects: [
      {
        id: "dsparkBlockSize",
        title: "DSpark Proposed Draft Tokens",
        showWhen: (base) => base.specAlgorithm === "DSPARK",
        control: "slider",
        stripPrefixes: ["--speculative-dspark-block-size"],
        options: [
          { id: "auto", label: "Checkpoint default" },
          { id: "1", label: "1", flags: ["--speculative-dspark-block-size 1"] },
          { id: "2", label: "2", flags: ["--speculative-dspark-block-size 2"] },
          { id: "3", label: "3", flags: ["--speculative-dspark-block-size 3"] },
          { id: "4", label: "4", flags: ["--speculative-dspark-block-size 4"] },
          { id: "5", label: "5", flags: ["--speculative-dspark-block-size 5"] },
        ],
      },
      {
        id: "engramHostTable",
        title: "Engram Host-Resident Tables",
        control: "select",
        options: [
          { id: "off", label: "Off (default)" },
          { id: "on", label: "On (larger KV pool)",
            env: ["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1"] },
        ],
      },
      {
        id: "decoderSwaBoundedReplay",
        title: "Decoder SWA Bounded Replay",
        control: "select",
        options: [
          { id: "off", label: "Off (default)" },
          { id: "on", label: "On (faster prefill)",
            flags: ["--enable-decoder-swa-bounded-replay"] },
        ],
      },
    ],
  },

  cells: [

    // ---------- GB300: 4x GB300 (SM103), TP4 + EP4. Reference platform. ----------
    {
      match: { hw: "gb300", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        // The derived value OOMs at 128K shapes while capturing the verify graphs.
        "--mem-fraction-static 0.8",
        "--speculative-algorithm DSPARK",
        "--speculative-dspark-block-size 5",
        // Decode CUDA graphs: the derived batch list does not fit here.
        "--cuda-graph-max-bs-decode 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        // No speculation: the DSpark step has a fixed cost over a plain decode
        // step, so it stops paying for itself once the batch is large.
        "--max-running-requests 256",
        // Leave the backends alone — they resolve to dsv4 / flashinfer_mxfp4 /
        // flashinfer_cutedsl. Overriding them is the usual cause of slow decode.
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- H200: 8x H200, TP8 + EP8. No MXFP8 dense path on Hopper. The
    // tightest board here at 140 GiB, so both cells also need the memory
    // fraction pulled back; the cap alone still leaves the graphs short. ------
    {
      match: { hw: "h200", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--mem-fraction-static 0.8",
        "--attention-backend dsv4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--enable-decoder-swa-bounded-replay",
        // Decode CUDA graphs: the derived batch list does not fit here.
        "--cuda-graph-max-bs-decode 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        "--mem-fraction-static 0.8",
        "--attention-backend dsv4",
        "--moe-runner-backend flashinfer_mxfp4",
        "--max-running-requests 256",
        // Decode CUDA graphs: the derived batch list does not fit here.
        "--cuda-graph-max-bs-decode 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- B200: 4x B200, TP4 + EP4. Mirrors the GB300 recipe — the
    // kernels dispatch by architecture family. ----------
    {
      match: { hw: "b200", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--mem-fraction-static 0.8",
        "--speculative-algorithm DSPARK",
        "--speculative-dspark-block-size 5",
        // Decode CUDA graphs: the derived batch list does not fit here.
        "--cuda-graph-max-bs-decode 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--max-running-requests 256",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- B300: 4x B300, TP4 + EP4. Same recipe as GB300 — the kernels
    // dispatch by architecture family. ----------
    {
      match: { hw: "b300", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--mem-fraction-static 0.8",
        "--speculative-algorithm DSPARK",
        "--speculative-dspark-block-size 5",
        // Decode CUDA graphs: the derived batch list does not fit here.
        "--cuda-graph-max-bs-decode 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--max-running-requests 256",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- MI355X: 2x MI355X (gfx950), TP2 + EP1.
    {
      match: { hw: "mi355x", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      env: [
        // Required until ROCm/aiter#5802 is in the aiter pin: below 256 tokens aiter
        // routes these fp4 experts to a bf16-activation MoE path the pin cannot run.
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank",
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--ep-size 1",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--speculative-dspark-block-size 5",
        // Fuses the shared expert into the routed MoE launch, which is worth more
        // at TP2 than TP4: the same fused GEMM covers twice the experts per rank.
        "--enforce-shared-experts-fusion",
        // The one backend worth naming on gfx950: aiter's fp8 GEMM beats what the
        // automatic selection resolves to for this model's 32-wide ue8m0 blocks.
        "--fp8-gemm-backend aiter",
        // Decode CUDA graphs: the derived batch list does not fit here; a cap of 64
        // fits and keeps 64-request decode batches on the graph.
        "--cuda-graph-max-bs-decode 64",
        "--enable-decoder-swa-bounded-replay",
        "--cuda-graph-backend-prefill disabled",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      env: [
        "AITER_BF16_FP8_MOE_BOUND=0",
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank",
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--ep-size 1",
        "--mem-fraction-static 0.85",
        // DSpark with graphs up to the running-request cap: 47% more throughput
        // than plain decode at 256 concurrent requests.
        "--speculative-algorithm DSPARK",
        "--speculative-dspark-block-size 5",
        "--enforce-shared-experts-fusion",
        "--fp8-gemm-backend aiter",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 128",
        "--enable-decoder-swa-bounded-replay",
        "--cuda-graph-backend-prefill disabled",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
