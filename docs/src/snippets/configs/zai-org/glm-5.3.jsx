// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.

export const config = {
  modelName: "GLM-5.3",

  supportedHardware: [
    "h200", "b200", "gb300", "b300",
    "mi355x", "mi325x", "mi300x",
  ],

  // Single released checkpoint — no size/mode split.
  variants: [
    { id: "default", label: "GLM-5.3", subtitle: "MoE · DSA" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
    { id: "bf16", label: "BF16" },
    { id: "nvfp4", label: "NVFP4 (Experimental)" },
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
    "default|fp8": "zai-org/GLM-5.3",
    "default|bf16": "zai-org/GLM-5.3-BF16",
    "default|nvfp4": "RadixArk/GLM-5.3-NVFP4",
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
  --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
      aime26_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run aime26 \\
  --model {{MODEL_NAME}} --api-key <api-key> \\
  --n-repeats 16 --max-tokens 64000 \\
  --temperature 1.0 --top-p 0.95 --thinking \\
  --out-dir /sgl-workspace/logs \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1`,
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

  accuracyLabels: [
    ["aime26_pct", "AIME26",         "%"],
    ["aime25_pct", "AIME25",         "%"],
    ["gsm8k_pct", "GSM8K (1-shot)", "%"],
  ],

  dockerImages: {
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    mi355x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618",
    mi325x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616",
    mi300x: "lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi30x-20260616",
  },

  github: {
    cookbookModel: "zai-org/glm-5.3",
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
    // Strategy knob: interleave (ex round-robin-split) is the default;
    // zigzag (ex in-seq-split) is exposed as an
    // experiment — the runtime auto-configures deepep + ep=tp for it and
    // restricts it to batch_size=1 (long-context single-request runs).
    attention: {
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
    // GLM-5.3 ships a single MTP (nextn) layer; index_share_for_mtp_iteration reuses the
    // DSA indexer topk across draft steps (topk==1 only). DFlash2 is the one
    // algorithm no Deploy cell ships: its draft is a separate checkpoint.
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
        { id: "dflash", label: "DFlash2 (block diffusion)",
          // Block-wise draft from a separate checkpoint, not the in-checkpoint
          // MTP layer. The block size (8) comes from the draft's own
          // dflash_config, so no --speculative-num-draft-tokens here. The draft
          // is a small dense model and does not run on the target's DSA
          // backends, hence the explicit draft attention backend.
          flags: ["--speculative-algorithm DFLASH",
                  "--speculative-draft-model-path incoai/GLM-5.3-DFlash2",
                  "--speculative-draft-attention-backend fa4"],
          // The DFlash2 drafter (PR #35371) merged after v0.5.18, so neither the
          // release wheel nor the lmsysorg/sglang:latest image this page pins
          // carries it. Drop this note once a release ships it.
          note: "⚠️ Needs a nightly image: the DFlash2 drafter (PR #35371) is not in the release wheel nor the lmsysorg/sglang:latest image this page pins — install SGLang from main or use a lmsysorg/sglang:dev image. The draft is a separate checkpoint, so fetch incoai/GLM-5.3-DFlash2 alongside the target; it is public but licensed CC BY-NC-ND 4.0 for research and evaluation.",
          disable: [
            { when: { dpAttnOn: [true] },
              reason: "DFLASH speculative decoding does not support DP-Attention — the server rejects the combination at startup. Turn DP-Attention off in the Attention card above (the high-throughput recipes enable it)." },
            { when: { hw: ["mi355x", "mi325x", "mi300x"] },
              reason: "DFLASH speculative decoding only supports CUDA and NPU devices; the server rejects it on ROCm at startup." },
          ] },
      ],
    },

    // ----- Card 5: "PD Disaggregation" -----
    // GLM-5.3 is a DSA model (same family as DeepSeek-V3.2/V4) and supports
    // prefill/decode disaggregation. Owns the `--disaggregation-*` flags; the
    // engine also pins role-specific serving ports (spaced apart) so prefill +
    // decode don't collide on one host.
    pdDisagg: {
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
    // H200 + FP8 (Hopper) — TP8.
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
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
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
        // Large chunked-prefill is the main high-throughput tuning lever;
        // max-running should track the available KV capacity.
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 + FP8 (Blackwell) — TP8.
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
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
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
        // Large chunked-prefill is the main high-throughput tuning lever;
        // max-running should track the available KV capacity.
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 + FP8 (Grace-Blackwell, 4-GPU single node) — TP4.
    // The flags follow the B200 recipe with TP4/DP4 for a 4-GPU node.
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
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
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
        // Same prefill lever as H200/B200; max-running tracks the TP4 KV capacity.
        "--chunked-prefill-size 32768",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 + FP8 (Blackwell Ultra, 8-GPU single node) — TP8.
    // The recipe follows the B200 FP8 path.
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
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
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

    // ====================================================================
    // B300 + BF16 (Blackwell Ultra, 8-GPU single node) — TP8.
    // The unquantized GLM-5.3 (~700B, ~1.51 TB) only fits single-node on 8xB300
    // (~2.1 TB HBM); smaller GPUs need multi-node (e.g. 2x 8xH200). Balanced/HT run plain TP8
    // without DP-Attention or DeepEP.
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
    // Recipes reuse the single-node B300 flags and remain unverified.
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
    // NVFP4 — RadixArk/GLM-5.3-NVFP4 (Model Optimizer, experts-only W4A4).
    // Same runtime contract as nvidia/GLM-5.2-NVFP4 (experts-only NVFP4, KV FP8,
    // no per-tensor k/v scale tensors), so the GLM-5.2 NVFP4 recipes carry over:
    // TP8 on B200/B300, TP4 on GB300; low-latency MTP 5-1-6, balanced MTP 2-1-3.
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
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
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--quantization modelopt_fp4",
        "--dp 8",
        "--enable-dp-attention",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.92",
        "--max-running-requests 256",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
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
        "--cuda-graph-max-bs 16",
        "--max-prefill-tokens 8192",
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
        "--quantization modelopt_fp4",
        "--dp 8",
        "--enable-dp-attention",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 2",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 3",
        "--chunked-prefill-size 32768",
        "--mem-fraction-static 0.92",
        "--max-running-requests 256",
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
        "--cuda-graph-max-bs 16",
        "--max-prefill-tokens 8192",
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

    // ====================================================================
    // AMD MI300X / MI325X / MI355X (ROCm) — TP8, DSA tilelang backend.
    // No MTP: disabled in the Speculative card for AMD (the gfx950 spec-decode
    // draft kernel is not yet validated, and num-steps>3 hits a separate build
    // issue). Strategies differ only by batch-shaping levers
    // (cuda-graph-max-bs / max-running-requests / chunked-prefill):
    //   low-latency      — large chunked-prefill, default bs.
    //   balanced         — chunked-prefill 32768 + bs128, max-running 80.
    //   high-throughput  — bs256, max-running 256.
    // BF16 (~1.51 TB) only fits single-node on MI325X (2 TB) / MI355X (2.3 TB);
    // MI300X (1.5 TB) needs multi-node, so its BF16 cells are omitted.
    // ====================================================================
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
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
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
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
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
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
