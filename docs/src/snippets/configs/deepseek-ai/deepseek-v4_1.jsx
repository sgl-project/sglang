// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
//
// Cells marked `verified` are transcribed from recorded runs on that hardware with
// real weights.
//
// GPU DSpark cells cap --cuda-graph-max-bs-decode: the derived batch list does
// not fit while capturing the DSpark decode graphs. H200 is the
// tightest board at 140 GiB and needs the cap on both cells plus a lower memory
// fraction; the B200 / B300 / GB300 High-Throughput cells start without either.
//
// GPU deployment cells do not use DP-Attention, DeepEP or MegaMoE. DP-Attention
// is used in the MI355X agentic recipe (cookbook §3.6); the Ascend A5 recipe uses
// DP-Attention and DeepEP as configured by its deployment script. GPU EP tracks
// TP on every shape except MI355X.

export const config = {
  modelName: "DeepSeek-V4.1",

  latencyPercentile: "P50",

  supportedHardware: ["h200", "b200", "b300", "gb300", "mi355x", "a5"],

  variants: [
    { id: "flash", label: "DeepSeek-V4.1-Flash", showWhen: (s) => s.hw !== "a5" },
    { id: "flash-1017", label: "DeepSeek-V4.1-Flash-1017" },
  ],
  quantizations: [
    { id: "fp4", label: "FP4", showWhen: (s) => s.hw !== "a5" },
    { id: "auto", label: "Checkpoint default", showWhen: (s) => s.hw === "a5" },
  ],
  strategies: [
    { id: "low-latency", label: "Low-Latency", showWhen: (s) => s.hw !== "a5" },
    { id: "balanced", label: "Balanced", showWhen: (s) => s.hw === "a5" },
    { id: "high-throughput", label: "High-Throughput", showWhen: (s) => s.hw !== "a5" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "flash|fp4": "deepseek-ai/DeepSeek-V4.1-Flash",
    "flash-1017|fp4": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "flash-1017|auto": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"         },
    PORT:      { target: "command", label: "Bind port",         default: "30000"           },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    MODEL_PATH: { target: "command", label: "A5 model path", default: "/path/to/dsv4.1-1017", showWhen: (s) => s.hw === "a5" },
    NODE0_IP: { target: "command", label: "A5 node IP", default: "<node-ip>", showWhen: (s) => s.hw === "a5" },
    NODE_RANK: { target: "command", label: "A5 node rank", default: "0", showWhen: (s) => s.hw === "a5" },
    HCCL_INTERFACE: { target: "command", label: "A5 HCCL interface", default: "lo", showWhen: (s) => s.hw === "a5" },
    GLOO_INTERFACE: { target: "command", label: "A5 GLOO interface", default: "data0.3001", showWhen: (s) => s.hw === "a5" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost"       },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"           },
  },

  curl: (sel) => `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "${sel.hw === "a5" ? "dsv4" : "{{MODEL_NAME}}"}", "messages": [{"role":"user","content":"Hello"}] }'`,

  dockerImages: {
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
    mi355x: "lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111",
    a5: "swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.2.0-950-dsv41",
  },

  // Initialize Ascend first, then apply the recipe environment before launch.
  pythonSetupCommand: (sel) => sel.hw === "a5"
    ? `source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash &&
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash &&
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh &&
source /usr/local/Ascend/nnal/atb/set_env.sh &&
if [[ -f /usr/local/memfabric_hybrid/set_env.sh ]]; then source /usr/local/memfabric_hybrid/set_env.sh; fi`
    : "",
  pythonCommand: (sel) => sel.hw === "a5"
    ? "python3 -m sglang.launch_server"
    : "sglang serve",
  dockerRunCommand: (sel, env = []) => sel.hw === "a5"
    ? `bash -lc 'source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash &&
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash &&
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh &&
source /usr/local/Ascend/nnal/atb/set_env.sh &&
if [[ -f /usr/local/memfabric_hybrid/set_env.sh ]]; then source /usr/local/memfabric_hybrid/set_env.sh; fi &&
${env.map((entry) => `export ${entry}`).join(" &&\n")} &&
exec python3 -m sglang.launch_server "$@"' --`
    : "sglang serve",
  dockerMounts: (sel) => sel.hw === "a5" ? ["{{MODEL_PATH}}:{{MODEL_PATH}}"] : [],

  github: {
    cookbookModel: "deepseek-ai/deepseek-v4.1",
  },

  playgroundFeatures: {

    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 2, 4, 8], showWhen: (s) => s.hw !== "a5" },
      ],
    },

    // No backend chooser: `flashinfer_mxfp4` is selected automatically and is the
    // only MoE runner this model has run on. EP tracks TP.
    moe: {
      ep: { label: "EP", values: [null, 1, 4, 8], showWhen: (s) => s.hw !== "a5" },
    },

    // Both parsers default to None; without them the DSML tool-call block and the
    // thinking block arrive as raw text inside `content`.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto", showWhen: (s) => s.hw !== "a5" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto", showWhen: (s) => s.hw !== "a5" },
      ],
    },

    // DSpark is the model's bundled 3-stage draft. There is no EAGLE/MTP path and
    // no `--speculative-num-steps` knob.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "dspark", label: "DSpark", showWhen: (s) => s.hw !== "a5",
          flags: ["--speculative-algorithm DSPARK", "--speculative-dspark-block-size 5"] },
        { id: "dspark-a5", label: "DSpark (Ascend)", showWhen: (s) => s.hw === "a5",
          flags: ["--speculative-algorithm DSPARK", "--speculative-draft-attention-backend ascend",
            "--speculative-num-draft-tokens 6", "--speculative-dspark-block-size 5"],
          env: ["SGLANG_ENABLE_SPEC_V2=1", "SGLANG_RAGGED_VERIFY_MODE=static", "SGLANG_DSPARK_FAST_KERNEL=0"] },
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
      excludesHw: ["mi355x", "a5"],
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
    // ---------- Ascend A5: single node, 8 NPUs, TP8 + DP8. ----------
    {
      match: { hw: "a5", variant: "flash-1017", quant: "auto", strategy: "balanced", nodes: "single" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "ASCEND_LAUNCH_BLOCKING=1",
        "HCCL_CONNECT_TIMEOUT=300",
        "HCCL_EXEC_TIMEOUT=68",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "ACL_DEVICE_SYNC_TIMEOUT=60",
        "HCCL_HOST_SOCKET_PORT_RANGE=auto",
        "HCCL_SOCKET_IFNAME={{HCCL_INTERFACE}}",
        "GLOO_SOCKET_IFNAME={{GLOO_INTERFACE}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "STREAMS_PER_DEVICE=32",
        "TRANSFORMERS_VERBOSITY=error",
        "SGLANG_SET_CPU_AFFINITY=1",
        "TASK_QUEUE_ENABLE=1",
        "INF_NAN_MODE_FORCE_DISABLE=1",
        "SGLANG_DEFAULT_THINKING=1",
        "SGLANG_DSV4_REASONING_EFFORT=max",
        "HCCL_BUFFSIZE=400",
        "DEEPEP_HCCL_BUFFSIZE=1800",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=16",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=150",
        "SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False",
        "FORCE_DRAFT_MODEL_NON_QUANT=1",
        "SGLANG_DSV4_FP4_EXPERTS=True",
        "SGLANG_OPT_FUSE_WQA_WKV=0",
        "SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch",
        "SGLANG_OPT_USE_FUSED_HASH_TOPK=False",
        "SGLANG_OPT_USE_TILELANG_MHC_PRE=False",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM=False",
        "SGLANG_OPT_USE_TILELANG_MHC_POST=False",
        "SGLANG_OPT_FP8_WO_A_GEMM=False",
        "SGLANG_DSV4_NATIVE_ATTN=0",
        "SGLANG_DSV4_DBG_ATTN=1",
        "SGLANG_SHALLOW_FUSED_ROPE_FQ4=0",
        "SGLANG_COMPRESSOR_PREFILL_USE_TRITON=1",
        "SGLANG_COMPRESSOR_DECODE_USE_TRITON=1",
        "SGLANG_OPT_USE_HC_TRITON=0",
        "SGLANG_OPT_USE_HC_TILELANG=0",
        "SGLANG_DSV4_GRAPH_MAX_SEQ=131072",
      ],
      flags: [
        "--nnodes 1",
        "--node-rank {{NODE_RANK}}",
        "--dist-init-addr {{NODE0_IP}}:5567",
        "--model-path {{MODEL_PATH}}",
        "--served-model-name dsv4",
        "--host 0.0.0.0",
        "--port 30000",
        "--tp-size 8",
        "--page-size 128",
        "--trust-remote-code",
        "--attention-backend dsv4",
        "--device npu",
        "--enable-dp-attention",
        "--dp-size 8",
        "--enable-dp-lm-head",
        "--watchdog-timeout 9000",
        "--max-running-requests 200",
        "--mem-fraction-static 0.83",
        "--max-prefill-tokens 131072",
        "--chunked-prefill-size 131072",
        "--kv-cache-dtype auto",
        `--json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}'`,
        "--reasoning-parser deepseek-v4",
        "--language-model-only",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--enable-metrics",
        "--cuda-graph-bs-decode 1 2 4 8 16 25",
      ],
    },
  ],
};
