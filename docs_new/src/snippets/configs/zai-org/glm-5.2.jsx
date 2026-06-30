// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.

export const config = {
  modelName: "GLM-5.2",

  supportedHardware: [
    "h200", "b200", "gb300", "b300",
    "mi355x", "mi325x", "mi300x",
  ],

  // Single released checkpoint — no size/mode split.
  variants: [
    { id: "default", label: "GLM-5.2", subtitle: "MoE · DSA" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
    { id: "bf16", label: "BF16" },
    { id: "nvfp4", label: "NVFP4" },
  ],
  strategies: [
    { id: "low-latency",    label: "Low-Latency"    },
    { id: "balanced",       label: "Balanced"       },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  modelNames: {
    "default|fp8": "zai-org/GLM-5.2-FP8",
    "default|bf16": "zai-org/GLM-5.2",
    "default|nvfp4": "nvidia/GLM-5.2-NVFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    NODE0_IP:  { target: "command", label: "Head node IP",      default: "<node0-ip>"   },
    NODE_RANK: { target: "command", label: "This node rank",    default: "<node-rank>"  },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // Reproduce commands for the Benchmark card's "⚡ Reproduce" modal.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
      aime25_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 64000 \\
  --temperature 1.0 --top-p 0.95 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
    },
    numPromptsByConc: { 1: 8, 16: 64, 64: 128, 256: 512, 1024: 2048, 4096: 8192 },
  },

  // Per-variant accuracy applied to every cell; per-cell `accuracy` overrides.
  // Both measured via sgl-eval (thinking mode) on H200. aime25 = pass@1 avg-of-16
  // (n-repeats 16, max-tokens 64000, temp 1.0, top-p 0.95); pass@16 100%, majority@16 93.3%.
  defaultAccuracy: {
    default: { gsm8k_pct: 98.2, aime25_pct: 87.7 },
  },

  accuracyLabels: [
    ["aime25_pct", "AIME25",         "%"],
    ["gsm8k_pct",  "GSM8K (1-shot)", "%"],
  ],

  dockerImages: {
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    mi355x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618",
    mi325x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616",
    mi300x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616",
    // NVFP4 needs the dev image with modelopt_fp4 support (per-quant override).
    "b300|nvfp4":  "lmsysorg/sglang:dev-glm52-nvfp4",
    "gb300|nvfp4": "lmsysorg/sglang:dev-glm52-nvfp4",
  },

  github: {
    cookbookModel: "zai-org/glm-5.2",
  },

  playgroundFeatures: {

    // ----- Card 1: "Attention Parallelism" -----
    // DSA prefill Context Parallelism (CP) splits the long-prefill attention across
    // `cp` ranks — verified on Hopper (H200). On Blackwell the DSA-CP FP8 rope kernel
    // is not yet adapted, so keep CP off there for now.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 4, 8] },
        { id: "cp", label: "CP (DSA prefill)", values: [null, 1, 2, 4, 8],
          disable: { hw: ["b200", "gb300", "b300", "mi355x", "mi325x", "mi300x"] },
          disableReason: "DSA prefill Context Parallel is verified on Hopper (H200); the Blackwell sm100 DSA-CP FP8 rope kernel is not yet adapted, and the ROCm DSA-CP path is not yet validated on AMD (MI300X/MI325X/MI355X)." },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card 2: "MoE Parallelism" -----
    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 4, 8] },
    },

    // ----- Card 3: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser glm45" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser glm47" },
      ],
    },

    // ----- Card 4: "Speculative Decoding" -----
    // GLM-5.2 ships a single MTP (nextn) layer; index_share_for_mtp_iteration reuses the
    // DSA indexer topk across draft steps (topk==1 only).
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp-516", label: "EAGLE / MTP 5-1-6 (low-latency)",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 5",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 6"],
          disable: { hw: ["mi355x", "mi325x", "mi300x"] },
          disableReason: "MTP/EAGLE speculative decoding is not yet validated on AMD ROCm (MI300X/MI325X/MI355X): the gfx950 spec-decode draft kernel is not yet validated and at --speculative-num-steps > 3 hits a separate build issue; the DSA nextn draft path is CUDA-only." },
        { id: "mtp-112", label: "EAGLE / MTP 1-1-2 (balanced)",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 1",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 2"],
          disable: { hw: ["mi355x", "mi325x", "mi300x"] },
          disableReason: "MTP/EAGLE speculative decoding is not yet validated on AMD ROCm (MI300X/MI325X/MI355X): the gfx950 spec-decode draft kernel is not yet validated and at --speculative-num-steps > 3 hits a separate build issue; the DSA nextn draft path is CUDA-only." },
      ],
    },

    // ----- Card 5: "Hierarchical KV Cache" -----
    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
      ],
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },
  },

  cells: [
    // ====================================================================
    // H200 + FP8 (Hopper) — TP8. CP (DSA prefill) verified here.
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.8",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        // Large chunked-prefill is the dominant balanced lever (prefill-bound at this
        // concurrency); max-running tracks KV capacity (~60-80 for 8K+1K reqs on 8xH200).
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 + FP8 (Blackwell) — TP8.  low-latency verified on b200-verda-k8s
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.8",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        // Large chunked-prefill is the dominant balanced lever (prefill-bound at this
        // concurrency); max-running tracks KV capacity (~89 for 8K+1K reqs on 8xB200).
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
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
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 + FP8 (Grace-Blackwell, 4-GPU single node) — TP4.
    // Flags mirror the B200 (sm100) configs; all three strategies verified end-to-end on
    // a single 4xGB300 node (v0.5.13.post1). GB300 leads B200 per-GPU in every regime.
    // Stage the weights on node-local NVMe first — shared cluster-storage reads are slow.
    // ====================================================================
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        // Same prefill lever as H200/B200 balanced; max-running tracks the TP4 KV capacity.
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
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
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 + FP8 (Blackwell Ultra, 8-GPU single node) — TP8. Verified on 8xB300 (v0.5.13.post1).
    // Recipe mirrors the verified B200 (sm100) FP8 path. B300 (sm103) currently trails B200 per-GPU
    // because deep_gemm/DSA are tuned for sm100; expected to improve as sm103 gets first-class kernels.
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.8",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
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
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 + BF16 (Blackwell Ultra, 8-GPU single node) — TP8. Verified on 8xB300 (v0.5.13.post1).
    // The unquantized GLM-5.2 (~700B, ~1.51 TB) only fits single-node on 8xB300
    // (~2.1 TB HBM); smaller GPUs need multi-node (e.g. 2x 8xH200). balanced/HT run plain TP8
    // (no DP-Attention/DeepEP), so they trail the FP8 recipe at high concurrency.
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs 128",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
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
        "--tp 8",
        "--mem-fraction-static 0.9",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // BF16 multi-node (inferred) — the 1.51 TB checkpoint spread over 2 nodes.
    // 2x 8xH200 / 2x 8xB200 at TP16, 2x 4xGB300 at TP8. The engine injects
    // --nnodes / --node-rank / --dist-init-addr from the Multi-Nodes selector.
    // Recipes inferred from the single-node B300 path; not benchmarked → verified:false.
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--chunked-prefill-size 32768",
        "--max-running-requests 80",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // NVFP4 (Blackwell Ultra) — nvidia/GLM-5.2-NVFP4 (Model Optimizer).
    // TP4 on B300 / GB300, low-latency + balanced. GB300 mirrors the B300
    // recipe (same TP4 / flags; the 4-GPU GB300 node fits the ~381 GB build).
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--quantization modelopt_fp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--quantization modelopt_fp4",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--quantization modelopt_fp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--quantization modelopt_fp4",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // AMD MI300X / MI325X / MI355X (ROCm) — TP8, DSA tilelang backend.
    // No MTP: disabled in the Speculative card for AMD (the gfx950 spec-decode
    // draft kernel is not yet validated, and num-steps>3 hits a separate build
    // issue). Strategies differ only by batch-shaping levers
    // (cuda-graph-max-bs / max-running-requests / chunked-prefill):
    //   low-latency      — large chunked-prefill, default bs.
    //   balanced         — chunked-prefill 32768 + bs128, max-running 80.
    //   high-throughput  — bs256, max-running 256.
    // ACCURACY: the earlier gfx950 block-FP8 bpreshuffle miscompile (GSM8K ~0) is
    // fixed as of the pinned mi355x image (...-20260618); MI355X FP8 was re-validated
    // (GSM8K ~0.96, NIAH 15/15 to ~118K) and all three FP8 strategies are benchmarked
    // + marked verified:true (see glm-5.2-benchmarks.jsx). All BF16 and all gfx942
    // (MI325X/MI300X) cells stay verified:false (not yet benchmarked, but correct).
    // BF16 (~1.51 TB) only fits single-node on MI325X (2 TB) / MI355X (2.3 TB);
    // MI300X (1.5 TB) needs multi-node, so its BF16 cells are omitted.
    // ====================================================================
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
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
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
