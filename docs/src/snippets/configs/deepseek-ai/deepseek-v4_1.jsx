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
// DP-Attention, DeepEP and MegaMoE are absent by design. DeepEP and MegaMoE have
// never been enabled on this model; DP-Attention runs in the MI355X agentic recipe
// (cookbook §3.6). EP is set equal to TP on every shape except MI355X.

export const config = {
  modelName: "DeepSeek-V4.1",

  latencyPercentile: "P50",

  supportedHardware: ["h200", "b200", "b300", "gb300", "mi355x", "rtx6000"],

  // RTX PRO 6000 (SM120, Blackwell workstation) is not in the shared catalog.
  hardware: [
    { id: "rtx6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
  ],

  variants: [
    { id: "flash", label: "DeepSeek-V4.1-Flash" },
    { id: "flash-1017", label: "DeepSeek-V4.1-Flash-1017" },
  ],
  quantizations: [
    { id: "fp4", label: "FP4" },
  ],
  strategies: [
    { id: "low-latency", label: "Low-Latency" },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "flash|fp4": "deepseek-ai/DeepSeek-V4.1-Flash",
    "flash-1017|fp4": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
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
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
    mi355x: "lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111",
    // First nightly with #45261, which SM120 needs to start this model.
    rtx6000: "lmsysorg/sglang:nightly-dev-20261119-6b374a55",
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
`sgl-router \\
  --model-id {{MODEL_NAME}} \\
  --no-tokenizer \\
  --worker-urls http://<prefill-host>:{{PREFILL_PORT}} http://<decode-host>:{{DECODE_PORT}} \\
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
            stripPrefixes: ["--engram-config"],
            flags: ["--engram-config '{\"placement\": \"cpu\"}'"] },
        ],
      },
      {
        id: "decoderSwaBoundedReplay",
        title: "Decoder SWA Bounded Replay",
        control: "select",
        options: [
          { id: "on", label: "Automatic (default)" },
          { id: "off", label: "Off (full prefill)",
            flags: ["--no-enable-decoder-swa-bounded-replay"] },
        ],
      },
    ],
  },

  cells: [

    // ---------- GB300: 4x GB300 (SM103), TP4 + EP4. Reference platform. ----------
    {
      match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
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
      match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
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
        // Decode CUDA graphs: the derived batch list does not fit here.
        "--cuda-graph-max-bs-decode 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
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
      match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
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
      match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
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
      match: { hw: "mi355x", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
      nnodes: 1,
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true",
        "SGLANG_HIP_FP4_INDEXER_ROWGROUP=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--ep-size 1",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--speculative-dspark-block-size 5",
        // The monokernel refuses SGLANG_HIP_DSV41_INDEX_K_PAGE8.
        "--enable-amd-monokernel",
        "--enforce-shared-experts-fusion",
        // The one backend worth naming on gfx950: aiter's fp8 GEMM beats what the
        // automatic selection resolves to for this model's 32-wide ue8m0 blocks.
        "--fp8-gemm-backend aiter",
        "--enable-decoder-swa-bounded-replay",
        // Decode CUDA graphs: the derived batch list does not fit here; a cap of 64
        // fits and keeps 64-request decode batches on the graph.
        "--cuda-graph-max-bs-decode 64",
        "--cuda-graph-backend-prefill disabled",
        "--engram-config '{\"placement\": \"cpu\"}'",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      nnodes: 1,
      verified: true,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true",
        "SGLANG_HIP_FP4_INDEXER_ROWGROUP=1",
        // Non-monokernel point. The monokernel refuses this layout.
        "SGLANG_HIP_DSV41_INDEX_K_PAGE8=1",
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
        "--enable-decoder-swa-bounded-replay",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 128",
        "--cuda-graph-backend-prefill disabled",
        "--engram-config '{\"placement\": \"cpu\"}'",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // 1017 reuses the original recipes; GPU validation is pending.
    {
      match: { hw: "gb300", variant: "flash-1017", quant: "fp4", strategy: "low-latency", nodes: "single" },
      nnodes: 1,
      verified: false,
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
      match: { hw: "gb300", variant: "flash-1017", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      nnodes: 1,
      verified: false,
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
    {
      match: { hw: "h200", variant: "flash-1017", quant: "fp4", strategy: "low-latency", nodes: "single" },
      nnodes: 1,
      verified: false,
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
      match: { hw: "h200", variant: "flash-1017", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      nnodes: 1,
      verified: false,
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
    {
      match: { hw: "b200", variant: "flash-1017", quant: "fp4", strategy: "low-latency", nodes: "single" },
      nnodes: 1,
      verified: false,
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
      match: { hw: "b200", variant: "flash-1017", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      nnodes: 1,
      verified: false,
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
    {
      match: { hw: "b300", variant: "flash-1017", quant: "fp4", strategy: "low-latency", nodes: "single" },
      nnodes: 1,
      verified: false,
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
      match: { hw: "b300", variant: "flash-1017", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      nnodes: 1,
      verified: false,
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
    {
      match: { hw: "mi355x", variant: "flash-1017", quant: "fp4", strategy: "low-latency", nodes: "single" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true",
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
        "--engram-config '{\"placement\": \"cpu\"}'",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash-1017", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_USE_ROCM700A=0",
        "TORCH_BLAS_PREFER_HIPBLASLT=1",
        "SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true",
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
        "--engram-config '{\"placement\": \"cpu\"}'",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---------- RTX PRO 6000: 8x RTX PRO 6000 Blackwell Server Edition (SM120,
    // 96 GB, PCIe only), TP8 + EP8. Verified on the 1017 checkpoint only.
    {
      match: { hw: "rtx6000", variant: "flash-1017", quant: "fp4", strategy: "low-latency", nodes: "single" },
      nnodes: 1,
      verified: true,
      env: [
        // No NVLink: let NCCL use PCIe peer-to-peer across the CPU sockets.
        "NCCL_P2P_LEVEL=SYS",
        "NCCL_MIN_NCHANNELS=16",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 8",
        // One host copy of the engram tables per node; the freed HBM goes to KV.
        "--engram-config '{\"placement\": \"cpu\", \"replication\": \"per_node\"}'",
        "--enable-deepseek-v4-fp4-indexer",
        "--context-length 262144",
        "--mem-fraction-static 0.70",
        // Faster here than the automatic choice: 21-22% more output tokens/s at
        // 32 and 128 concurrent requests, 1K in / 1K out.
        "--fp8-gemm-backend flashinfer_cutlass",
        "--speculative-algorithm DSPARK",
        "--speculative-dspark-block-size 5",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 128",
        "--chunked-prefill-size 8192",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
