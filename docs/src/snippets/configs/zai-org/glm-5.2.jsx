// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.

export const config = {
  modelName: "GLM-5.2",

  supportedHardware: [
    "h200", "b200", "gb300", "b300",
    "mi355x", "mi325x", "mi300x",
    // Ascend NPU. One Atlas 800I A3 node is 8 cards x 2 dies = 16 devices,
    // so every A3 recipe runs --tp-size 16 per node.
    "a3",
  ],

  // Single released checkpoint — no size/mode split.
  variants: [
    { id: "default", label: "GLM-5.2", subtitle: "MoE · DSA" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
    { id: "bf16", label: "BF16" },
    { id: "nvfp4", label: "NVFP4" },
    { id: "mxfp4", label: "MXFP4" },
    // Ascend ships its own INT8 checkpoint (Eco-Tech/GLM-5.2-w8a8, quantized
    // with msModelSlim and served with --quantization modelslim). The GPU
    // quantizations have no NPU kernels, so this chip is NPU-only.
    { id: "w8a8", label: "W8A8", showWhen: (s) => s.hw === "a3" },
  ],
  strategies: [
    { id: "low-latency",    label: "Low-Latency"    },
    { id: "balanced",       label: "Balanced"       },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    // The A3 reasons below only fire on that hardware, so every GPU page keeps
    // the plain single / multi-node pair.
    { id: "single",  label: "Single Node",
      disabled: (s) => s.hw === "a3" && s.strategy !== "balanced",
      disableReason: "On Atlas 800I A3, the single-node PD mixed recipe is the Balanced one — switch Strategy to Balanced." },
    { id: "multi-2", label: "Multi-Nodes",
      disabled: (s) => s.hw === "a3",
      disableReason: "Two A3 nodes serve GLM-5.2 with PD disaggregation (1P1D) — pick PD Prefill for the first node and PD Decode for the second." },
    // Ascend PD disaggregation puts one role on each A3 node (1P1D = 2 nodes),
    // and each role is its own server started with --nnodes 1 — so the role is
    // part of the topology choice rather than a `multi-2` cell. Front both with
    // sglang_router (see Configuration Tips).
    { id: "pd-prefill", label: "PD Prefill (2 nodes)",
      showWhen: (s) => s.hw === "a3",
      disabled: (s) => s.strategy !== "high-throughput",
      disableReason: "On Atlas 800I A3, PD disaggregation is the High-Throughput recipe — switch Strategy to High-Throughput first." },
    { id: "pd-decode",  label: "PD Decode (2 nodes)",
      showWhen: (s) => s.hw === "a3",
      disabled: (s) => s.strategy !== "high-throughput",
      disableReason: "On Atlas 800I A3, PD disaggregation is the High-Throughput recipe — switch Strategy to High-Throughput first." },
  ],

  modelNames: {
    "default|fp8": "zai-org/GLM-5.2-FP8",
    "default|bf16": "zai-org/GLM-5.2",
    "default|nvfp4": "nvidia/GLM-5.2-NVFP4",
    "default|mxfp4": "amd/GLM-5.2-MXFP4",
    // ModelScope repo — pair with SGLANG_USE_MODELSCOPE=1 (set in the A3 cells).
    "default|w8a8": "Eco-Tech/GLM-5.2-w8a8",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    NODE0_IP:  { target: "command", label: "Head node IP",      default: "<node0-ip>"   },
    NODE_RANK: { target: "command", label: "This node rank",    default: "<node-rank>"  },
    NETWORK_IFACE: { target: "command", label: "Cross-node NIC",  default: "<your-nic>"       },
    PREFILL_IP:    { target: "command", label: "Prefill node IP", default: "<prefill-node-ip>" },
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
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
      aime25_pct:
`# To install sgl-eval: pip install sgl-eval
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
    "mi355x|mxfp4": "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260910",
    mi325x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616",
    mi300x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616",
    // CANN 9.0.0 release image for Atlas 800I A3 (daily builds are tagged
    // main-cann9.0.0-a3). See the Ascend NPU quickstart for the A2 image.
    a3: "quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16",
  },

  // Each PD role launches with --nnodes 1, so the multi-node branch does not
  // fire, but prefill and decode still exchange bootstrap + KV traffic across
  // nodes over HCCL/RDMA — that needs the host network, not a published port.
  dockerHostNetworkWhen: (s) => s.nodes === "pd-prefill" || s.nodes === "pd-decode",

  github: {
    cookbookModel: "zai-org/glm-5.2",
  },

  playgroundFeatures: {

    // ----- Card 1: "Attention Parallelism" -----
    // DSA prefill Context Parallelism (CP) splits the long-prefill attention across
    // `cp` ranks — runs on Hopper (H200) and Blackwell (B200/GB300/B300).
    // CP sizes auto-gate in the engine to the runtime derivation
    // attn_cp_size = tp/dp (a user-passed --attn-cp-size is overridden).
    // CP is single-machine only (tp_size <= 8). Interleave CP + DP-Attention
    // currently fails the runtime's dp_size == 1 assert but is allowed here
    // with a warning (combined support is planned upstream).
    // Strategy knob: interleave (ex round-robin-split) is the layout verified
    // here and the default; zigzag (ex in-seq-split) is exposed as an
    // experiment — the runtime auto-configures deepep + ep=tp for it and
    // restricts it to batch_size=1 (long-context single-request runs).
    attention: {
      // Hidden on Atlas 800I A3: the NPU recipes are fixed at --tp-size 16
      // (one node = 16 dies), NPU DP-Attention runs at the recipe's own degree
      // (4 on prefill, 16 on decode), and NPU prefill context parallelism uses
      // a different flag family (--enable-nsa-prefill-context-parallel /
      // --nsa-prefill-cp-mode), so none of these knobs emits a valid NPU flag.
      showWhen: (b) => b.hw !== "a3",
      knobs: [
        { id: "tp", label: "TP", values: [null, 4, 8] },
        { id: "cp", label: "CP (DSA prefill)",
          values: [null, { value: 1, label: "Off" }, 4, 8],
          disable: [
            { when: { hw: ["mi355x", "mi325x", "mi300x"] },
              reason: "The ROCm DSA-CP path is not yet validated on AMD (MI300X/MI325X/MI355X) — keep CP off there for now." },
            { when: { nodes: ["multi-2"] },
              reason: "Prefill Context Parallel is single-machine only (SGLang asserts tp_size <= 8; cross-machine CP has precision issues)." },
          ] },
        { id: "cpStrategy", label: "CP Strategy",
          values: [
            null,
            "interleave",
            { value: "zigzag", label: "zigzag (experimental)" },
          ],
          disable: [
            { when: { hw: ["mi355x", "mi325x", "mi300x"] },
              reason: "The ROCm DSA-CP path is not yet validated on AMD (MI300X/MI325X/MI355X) — keep CP off there for now." },
            { when: { nodes: ["multi-2"] },
              reason: "Prefill Context Parallel is single-machine only (SGLang asserts tp_size <= 8; cross-machine CP has precision issues)." },
          ] },
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
      // DeepEP is the NPU a2a backend too, so the backend select stays. The EP
      // degrees do not: the A3 decode recipe runs EP16 across the node's dies.
      ep: { label: "EP", values: [null, 4, 8], showWhen: (b) => b.hw !== "a3" },
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
          disable: [
            { when: { hw: ["mi300x", "mi325x"] },
              reason: "MTP/EAGLE speculative decoding is not yet validated for GLM-5.2 on MI300X or MI325X." },
            { when: { hw: ["mi355x"], quant: ["fp8", "bf16", "nvfp4"] },
              reason: "The five-step MI355X recipe is validated only with amd/GLM-5.2-MXFP4." },
            { when: { hw: ["a3"] }, reason: "Ascend NPU drives the same MTP head through --speculative-algorithm NEXTN (plus --speculative-draft-model-quantization unquant), which every A3 recipe already carries — the EAGLE presets here are CUDA-only." },
          ] },
        { id: "mtp-112", label: "EAGLE / MTP 1-1-2 (balanced)",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 1",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 2"],
          disable: [
            { when: { hw: ["mi355x", "mi325x", "mi300x"] },
              reason: "MTP/EAGLE speculative decoding is not yet validated on AMD ROCm (MI300X/MI325X/MI355X): the gfx950 spec-decode draft kernel is not yet validated and at --speculative-num-steps > 3 hits a separate build issue; the DSA nextn draft path is CUDA-only." },
            { when: { hw: ["a3"] }, reason: "Ascend NPU drives the same MTP head through --speculative-algorithm NEXTN (plus --speculative-draft-model-quantization unquant), which every A3 recipe already carries — the EAGLE presets here are CUDA-only." },
          ] },
        { id: "mtp-314", label: "EAGLE / MTP 3-1-4 (agentic · MI355X MXFP4)",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"],
          enable: { hw: ["mi355x"], quant: ["mxfp4"] },
          enableReason: "Validated on MI355X gfx950 with amd/GLM-5.2-MXFP4 (InferenceX AgentX sweep, GSM8K em_strict 0.971). num-steps=3 stays within the validated gfx950 spec-decode build envelope (≤3). Pair with SGLANG_SIMULATE_ACC_LEN=2.99 for benchmarking (golden AL from golden_al_distribution/glm5.2_mtp.yaml, thinking_on, num_speculative_tokens=3)." },
      ],
    },

    // ----- Card 5: "PD Disaggregation" -----
    // GLM-5.2 is a DSA model (same family as DeepSeek-V3.2/V4) and supports
    // prefill/decode disaggregation. Owns the `--disaggregation-*` flags; the
    // engine also pins role-specific serving ports (spaced apart) so prefill +
    // decode don't collide on one host.
    pdDisagg: {
      // Hidden on Atlas 800I A3: the NPU prefill and decode roles are whole
      // recipes of their own (different parallelism, DeepEP mode, graph capture
      // and MTP depth), so they ship as cells under the Deploy panel's Nodes row
      // — PD Prefill / PD Decode — instead of being layered on the PD mixed recipe.
      showWhen: (b) => b.hw !== "a3",
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        // Mooncake (recommended). The NCCL/MNNVL env is only needed on the
        // NVLink-multinode Grace-Blackwell platform (GB300 here).
        { id: "mooncake", label: "Mooncake",
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb300"] } },
        { id: "nixl",     label: "NiXL" },
      ],
      // No IB-device knob: mooncake auto-detects the HCA. Pass
      // --disaggregation-ib-device only if discovery picks the wrong NIC
      // (see Configuration Tips).
      // Router fronting the prefill + decode roles; substitute <prefill-host>/<decode-host>.
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`,
      },
    },

    // ----- Card 6: "Hierarchical KV Cache" -----
    hicache: {
      // Hierarchical KV cache is not part of the verified GLM-5.2 NPU matrix.
      showWhen: (b) => b.hw !== "a3",
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
        // Large chunked-prefill is the dominant balanced lever (prefill-bound at this
        // concurrency); max-running tracks KV capacity (~60-80 for 8K+1K reqs on 8xH200).
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
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
        // Large chunked-prefill is the dominant balanced lever (prefill-bound at this
        // concurrency); max-running tracks KV capacity (~89 for 8K+1K reqs on 8xB200).
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
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
        // Same prefill lever as H200/B200 balanced; max-running tracks the TP4 KV capacity.
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--dp 4",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--mem-fraction-static 0.85",
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
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
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
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // NVFP4 — nvidia/GLM-5.2-NVFP4 (Model Optimizer). TP8 on B200/B300, TP4 on GB300.
    // B200/B300: 8-GPU single node, TP8 (low-latency / balanced / high-throughput); balanced &
    // high-throughput add DP-Attention (dp8). low-latency uses MTP 5-1-6, balanced MTP 2-1-3.
    // GB300: 4-GPU single node, TP4 (the node fits the ~381 GB build); GB300 adds dp4 on
    // balanced & high-throughput; low-latency uses MTP 5-1-6.
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--dp 8",
        "--enable-dp-attention",
        // Shorter draft (MTP 2-1-3) than low-latency's 5-1-6: at this concurrency the
        // verify overhead of a long draft outweighs the accept-length gain.
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        // Larger chunked-prefill (32768 → ~4096/rank under dp8) is the dominant balanced lever.
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.92",
        "--max-running-requests 256",
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
        "--tp 8",
        "--quantization modelopt_fp4",
        "--dp 8",
        "--enable-dp-attention",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.92",
        "--max-running-requests 512",
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
        "--tp 8",
        "--quantization modelopt_fp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--max-prefill-tokens 8192",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--quantization modelopt_fp4",
        // Shorter draft (MTP 2-1-3) than low-latency's 5-1-6: at this concurrency the
        // verify overhead of a long draft outweighs the accept-length gain.
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        // Two required flags for DP-Attention + MTP here: `decode`-mode spec attention
        // avoids a CUDA-graph capture deadlock, and max-running 256 lifts the default
        // ~48-request throttle so DP-Attention can fill all 8 ranks.
        "--speculative-attention-mode decode",
        "--max-running-requests 256",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
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
        "--tp 8",
        "--dp 8",
        "--enable-dp-attention",
        "--quantization modelopt_fp4",
        "--max-running-requests 1024",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
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
        "--tp 4",
        "--quantization modelopt_fp4",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.85",
        "--max-running-requests 16",
        "--cuda-graph-max-bs-decode 16",
        "--max-prefill-tokens 8192",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--quantization modelopt_fp4",
        "--dp 4",
        "--enable-dp-attention",
        // Shorter draft (MTP 2-1-3) than low-latency's 5-1-6: at this concurrency the
        // verify overhead of a long draft outweighs the accept-length gain.
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.92",
        "--max-running-requests 256",
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
        "--tp 4",
        "--quantization modelopt_fp4",
        "--dp 4",
        "--enable-dp-attention",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.92",
        "--max-running-requests 512",
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
        "--cuda-graph-max-bs-decode 128",
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
        "--cuda-graph-max-bs-decode 256",
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
        "--cuda-graph-max-bs-decode 128",
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
        "--cuda-graph-max-bs-decode 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ====================================================================
    // AMD MI355X + MXFP4 — amd/GLM-5.2-MXFP4 (Quark). TP4: the 4-bit MoE
    // weights fit a 4-GPU slice, mirroring the amd/GLM-5.1-MXFP4 MI355X recipe (same DSA
    // architecture family) — --trust-remote-code (Quark custom quant config)
    // and --kv-cache-dtype fp8_e4m3 both come from that precedent. Pinned to a
    // newer image (v0.5.19, see dockerImages["mi355x|mxfp4"]) than the FP8/BF16
    // mi355x cells. Low-Latency uses validated TP8/EP1; High-Throughput uses
    // validated TP4/EP4. Both use five-step MTP from InferenceX PR #2900.
    // DSA backend: triton (SGLang's ROCm default).
    // ====================================================================
    {
      match: { hw: "mi355x", variant: "default", quant: "mxfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--ep-size 1",
        "--kv-cache-dtype fp8_e4m3",
        "--dsa-prefill-backend triton",
        "--dsa-decode-backend triton",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "mxfp4", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--kv-cache-dtype fp8_e4m3",
        "--dsa-prefill-backend triton",
        "--dsa-decode-backend triton",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 128",
        "--max-running-requests 80",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep-size 4",
        "--kv-cache-dtype fp8_e4m3",
        "--dsa-prefill-backend triton",
        "--dsa-decode-backend triton",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MI355X + MXFP4 + MTP (mtp-314): validated AgentX recipe.
    // steps=3 stays within the gfx950 spec-decode build envelope (≤3).
    // mem-fraction-static 0.80: headroom for MTP draft buffer on top of
    // 4-bit MoE weights + KV cache (matches InferenceX AgentX harness conc≤16).
    // For benchmarking: set SGLANG_SIMULATE_ACC_LEN=2.99,
    // SGLANG_SIMULATE_ACC_METHOD=match-expected,
    // SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
    // (golden AL: golden_al_distribution/glm5.2_mtp.yaml, thinking_on, num_speculative_tokens=3).
    {
      match: { hw: "mi355x", variant: "default", quant: "mxfp4", strategy: "mtp-314", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--kv-cache-dtype fp8_e4m3",
        "--dsa-prefill-backend triton",
        "--dsa-decode-backend triton",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--chunked-prefill-size 131072",
        "--mem-fraction-static 0.80",
        "--cuda-graph-max-bs-decode 160",
        "--max-running-requests 160",
        "--watchdog-timeout 1800",
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
        "--cuda-graph-max-bs-decode 128",
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
        "--cuda-graph-max-bs-decode 256",
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
        "--cuda-graph-max-bs-decode 128",
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
        "--cuda-graph-max-bs-decode 256",
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
        "--cuda-graph-max-bs-decode 128",
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
        "--cuda-graph-max-bs-decode 256",
        "--max-running-requests 256",
        "--watchdog-timeout 1200",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // Atlas 800I A3 (Ascend NPU) + W8A8 (msModelSlim INT8).
    // One A3 node is 8 cards x 2 dies = 16 devices, so every recipe here runs
    // --tp-size 16 per node. Transcribed from the Ascend NPU GLM-5.2 tutorial
    // (/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2):
    //   Single Node  -> PD Mixed on one A3 node (16 dies).
    //   PD Prefill   -> the prefill half of a 1P1D pair (2 A3 nodes, 32 dies).
    //   PD Decode    -> the decode half of the same pair.
    // The NPU verification round is open, so all three render "In progress".
    // ====================================================================
    {
      match: { hw: "a3", variant: "default", quant: "w8a8", strategy: "balanced", nodes: "single" },
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_USE_MODELSCOPE=1",
        "SGLANG_SET_CPU_AFFINITY=1",
        "STREAMS_PER_DEVICE=32",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "SGLANG_NPU_USE_MULTI_STREAM=1",
        "HCCL_BUFFSIZE=1000",
        "HCCL_OP_EXPANSION_MODE=AIV",
        // Single node: the rendezvous never leaves the host.
        "HCCL_SOCKET_IFNAME=lo",
        "GLOO_SOCKET_IFNAME=lo",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=72",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024",
        "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--device npu",
        "--attention-backend ascend",
        "--quantization modelslim",
        "--tp-size 16",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--mem-fraction-static 0.7",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 280000",
        "--cuda-graph-bs-decode 16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--speculative-draft-model-quantization unquant",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "a3", variant: "default", quant: "w8a8", strategy: "high-throughput", nodes: "pd-prefill" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "This is the **prefill half** of a 1P1D pair: run it on the first A3 node, run **PD Decode** on the second, then front both with the router. Both nodes need the same `ASCEND_MF_STORE_URL` — see [Ascend NPU (Atlas 800I A3)](#ascend-npu-atlas-800i-a3).",
      env: [
        "SGLANG_USE_MODELSCOPE=1",
        "SGLANG_SET_CPU_AFFINITY=1",
        "STREAMS_PER_DEVICE=32",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        // KV-transfer rendezvous. Same value on the prefill and decode node.
        "ASCEND_MF_STORE_URL=tcp://{{PREFILL_IP}}:24707",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "HCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "TASK_QUEUE_ENABLE=2",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=72",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024",
        "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
        "TRANSFORMERS_VERBOSITY=error",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--device npu",
        "--attention-backend ascend",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--tp-size 16",
        "--dp-size 4",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--moe-dense-tp-size 1",
        "--moe-a2a-backend deepep",
        // Prefill is compute-bound and batches whole sequences: normal dispatch,
        // no NPU graph capture.
        "--deepep-mode normal",
        "--disable-shared-experts-fusion",
        "--cuda-graph-backend-decode disabled",
        "--cuda-graph-backend-prefill disabled",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend ascend",
        "--disaggregation-bootstrap-port 8998",
        "--mem-fraction-static 0.8",
        "--chunked-prefill-size 524288",
        "--max-prefill-tokens 180000",
        "--max-running-requests 64",
        "--load-balance-method round_robin",
        // One draft step on the prefill side; the decode role runs the deeper
        // 3-1-4 MTP schedule.
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--speculative-draft-model-quantization unquant",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "a3", variant: "default", quant: "w8a8", strategy: "high-throughput", nodes: "pd-decode" },
      verified: false,
      verificationStatus: "in-progress",
      warn: "This is the **decode half** of a 1P1D pair: run it on the second A3 node, after **PD Prefill** is up on the first, then front both with the router. Both nodes need the same `ASCEND_MF_STORE_URL` — see [Ascend NPU (Atlas 800I A3)](#ascend-npu-atlas-800i-a3).",
      env: [
        "SGLANG_USE_MODELSCOPE=1",
        "SGLANG_SET_CPU_AFFINITY=1",
        "STREAMS_PER_DEVICE=32",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "ASCEND_MF_STORE_URL=tcp://{{PREFILL_IP}}:24707",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "HCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "HCCL_BUFFSIZE=650",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "SGLANG_NPU_USE_MULTI_STREAM=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32",
        "TASK_QUEUE_ENABLE=0",
        "TRANSFORMERS_VERBOSITY=error",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--device npu",
        "--attention-backend ascend",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--tp-size 16",
        "--dp-size 16",
        "--ep-size 16",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        // Decode dispatches a few tokens per step: low-latency EP kernels plus
        // a small captured graph range.
        "--deepep-mode low_latency",
        "--disable-shared-experts-fusion",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend ascend",
        "--mem-fraction-static 0.8",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 4",
        "--context-length 180000",
        "--tokenizer-worker-num 4",
        "--load-balance-method round_robin",
        // Weight load over 16 dies is slow; keep the watchdog out of the way.
        "--watchdog-timeout 9000",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--speculative-draft-model-quantization unquant",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
