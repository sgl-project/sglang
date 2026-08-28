export const config = {
  modelName: "GLM-5.3-Flash",

  supportedHardware: [
    "gb300", "h100", "h200", "b200", "b300", "gb200",
    "mi300x", "mi325x", "mi350x", "mi355x",
  ],

  matchDims: [
    {
      id: "quant",
      title: "Checkpoint",
      options: [
        { id: "fp8", label: "FP8", subtitle: "zai-org" },
        {
          id: "mxfp4",
          label: "Quark MXFP4",
          subtitle: "AMD · gfx950",
          showWhen: (s) => ["mi350x", "mi355x"].includes(s.hw),
        },
      ],
    },
    {
      id: "strategy",
      title: "Recipe",
      options: [
        {
          id: "low-latency",
          label: "Low Latency",
          subtitle: "Adaptive MTP 5/1/6",
          showWhen: (s) => s.quant === "fp8",
          disabled: (s) => ["mi300x", "mi325x", "mi350x", "mi355x"].includes(s.hw),
          disableReason: "MTP speculative decoding has not been validated for GLM-5.3-Flash on AMD ROCm; use the non-speculative High Throughput recipe.",
        },
        {
          id: "high-throughput",
          label: "High Throughput",
          subtitle: "Spec decode off",
          showWhen: (s) => s.quant === "fp8",
        },
        {
          id: "mxfp4-tp1",
          label: "TP1",
          subtitle: "Full decode graph",
          showWhen: (s) => s.quant === "mxfp4",
        },
        {
          id: "mxfp4-tp2",
          label: "TP2",
          subtitle: "Full decode graph",
          showWhen: (s) => s.quant === "mxfp4",
        },
        {
          id: "mxfp4-tp4",
          label: "TP4",
          subtitle: "Full decode graph",
          showWhen: (s) => s.quant === "mxfp4",
        },
        {
          id: "mxfp4-tp8-ep8",
          label: "TP8 + EP8",
          subtitle: "TEP8 · full decode graph",
          showWhen: (s) => s.quant === "mxfp4",
        },
      ],
    },
  ],

  isRecommendedSelection(s) {
    const pairing = ["h100", "h200", "mi300x", "mi325x", "mi350x", "mi355x"].includes(s.hw)
      ? "bf16-tilelang"
      : "fp8-trtllm";
    return (
      s.kvDsaPair === pairing &&
      s.mmTransport === "auto" &&
      s.hicache === "off" &&
      s.dcp === "off"
    );
  },

  overlayDims: [
    {
      id: "kvDsaPair",
      title: "KV Cache + DSA Backend",
      default: "fp8-trtllm",
      options: [
        {
          id: "fp8-trtllm",
          label: "FP8 + TRT-LLM",
          disabled: (s) => ["h100", "h200", "mi300x", "mi325x", "mi350x", "mi355x"].includes(s.hw),
          disableReason: "This recipe uses BF16 KV cache with TileLang DSA on Hopper and AMD ROCm GPUs.",
          stripPrefixes: ["--kv-cache-dtype", "--dsa-prefill-backend", "--dsa-decode-backend"],
          flags: [
            "--kv-cache-dtype fp8_e4m3",
            "--dsa-prefill-backend trtllm",
            "--dsa-decode-backend trtllm",
          ],
          hints: ["Measured on GB300: faster than BF16 + TileLang with about 1.8x the KV token capacity."],
        },
        {
          id: "bf16-tilelang",
          label: "BF16 + TileLang",
          stripPrefixes: ["--kv-cache-dtype", "--dsa-prefill-backend", "--dsa-decode-backend"],
          flags: [
            "--kv-cache-dtype bfloat16",
            "--dsa-prefill-backend tilelang",
            "--dsa-decode-backend tilelang",
          ],
        },
      ],
    },
    {
      id: "mmTransport",
      title: "VLM Transport",
      default: "auto",
      options: [
        { id: "auto", label: "Auto", subtitle: "Topology-aware" },
        {
          id: "cpu",
          label: "CPU",
          subtitle: "Save GPU memory",
          flags: ["--mm-feature-transport cpu"],
        },
      ],
    },
    {
      id: "hicache",
      title: "HiCache",
      default: "off",
      options: [
        { id: "off", label: "Off" },
        {
          id: "l2",
          label: "L1 + L2",
          subtitle: "Host memory",
          flags: ["--enable-hierarchical-cache", "--hicache-size 32"],
          disabled: (s) => s.quant === "mxfp4",
          disableReason: "The validated MXFP4 recipes disable radix caching, so HiCache is unavailable.",
          hints: ["32 GB host tier; the default ratio can demand more host RAM than the node has free."],
        },
        {
          id: "l3",
          label: "+ L3",
          subtitle: "Mooncake",
          flags: ["--enable-hierarchical-cache", "--hicache-size 32", "--hicache-storage-backend mooncake"],
          env: ["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH={{MOONCAKE_CONFIG}}"],
          disabled: (s) => s.quant === "mxfp4",
          disableReason: "The validated MXFP4 recipes disable radix caching, so HiCache is unavailable.",
          hints: ["Start Mooncake and place the configuration file on every serving node."],
        },
      ],
    },
    {
      id: "dcp",
      title: "Context Parallelism",
      default: "off",
      options: [
        { id: "off", label: "Off" },
        {
          id: "4",
          label: "DCP 4",
          disabled: (s) => s.hw !== "gb300",
          disableReason: "DCP is validated only on 4x GB300 TP4/EP4 for now.",
          flags: ["--dcp-size 4", "--dcp-comm-backend a2a", "--dcp-replicate-q-proj"],
          hints: ["Measured on 4x GB300 with both KV/DSA pairings, adaptive MTP 5/1/6, full decode graph."],
        },
      ],
    },
  ],

  modelNames: {
    fp8: "zai-org/GLM-5.3-Flash",
    mxfp4: "amd/GLM-5.3-Flash-Quark-MXFP4",
  },

  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    MOONCAKE_CONFIG: { target: "command", label: "Mooncake config", default: "<mooncake.json>" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  benchmarkCommands: {
    speed:
`# Low Latency speed runs serve with SGLANG_SIMULATE_ACC_LEN=3 to pin the accept
# length; that number is throughput evidence only. Never run accuracy against it.
python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --request-rate inf --temperature 0 --seed 42 \\
  --flush-cache`,
    // num_prompts = 5 × concurrency (measured floor 16).
    numPromptsByConc: { 1: 16, 16: 80, 64: 320, 256: 1280, 1024: 5120 },
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --num-threads 64 \\
  --max-tokens 32768 \\
  --temperature 1.0 \\
  --top-p 0.95 \\
  --thinking`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  // Support is not in a public sglang release yet. NVIDIA uses the
  // purpose-built CUDA 13 image. AMD validation used these ROCm 7.2 images
  // with the GLM-5.3 ROCm engine branch mounted over the image source tree.
  dockerImages: {
    gb300: "lmsysorg/sglang:glm-5.3-flash",
    h100: "lmsysorg/sglang:glm-5.3-flash",
    h200: "lmsysorg/sglang:glm-5.3-flash",
    b200: "lmsysorg/sglang:glm-5.3-flash",
    b300: "lmsysorg/sglang:glm-5.3-flash",
    gb200: "lmsysorg/sglang:glm-5.3-flash",
    mi300x: "lmsysorg/sglang:v0.5.18-rocm720-mi30x",
    mi325x: "lmsysorg/sglang:v0.5.18-rocm720-mi30x",
    mi350x: "lmsysorg/sglang:v0.5.18-rocm720-mi35x",
    mi355x: "lmsysorg/sglang:v0.5.18-rocm720-mi35x",
  },

  // GLM-5.3-Flash AMD support currently lives on the open #36507 support
  // branch, including the merged #36607 ROCm work. Do not emit a standalone
  // Docker command until a published ROCm image contains that source stack.
  runModes: (s) =>
    ["mi300x", "mi325x", "mi350x", "mi355x"].includes(s.hw)
      ? ["python"]
      : ["python", "docker"],

  github: {
    cookbookModel: "zai-org/glm-5.3-flash",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null, 1, 2, 4,
          {
            value: 8,
            disable: [
              {
                when: { hw: ["gb300", "gb200"] },
                reason: "TP=8 needs 8 GPUs; the GB300 and GB200 recipes run on 4.",
              },
            ],
          },
        ]},
        { id: "cp", label: "CP", values: [null, 1, 2, 4] },
        {
          id: "dpAttn",
          label: "DP-Attention",
          values: [
            null, false, 1, 2, 4,
            {
              value: 8,
              disable: [
                {
                  when: { hw: ["gb300", "gb200"] },
                  reason: "DP-Attention=8 needs 8 ranks; the GB300 and GB200 recipes run on 4.",
                },
              ],
            },
          ],
          labels: { auto: "Auto", false: "Off" },
          disable: [
            {
              when: { strategy: ["low-latency"] },
              reason: "Low Latency uses adaptive MTP, which does not support DP-Attention.",
            },
          ],
          disableReason: "Low Latency uses adaptive MTP, which does not support DP-Attention.",
        },
      ],
    },

    moe: {
      backend: {
        options: [
          { id: null, label: "Inherited" },
          {
            id: "deep_gemm",
            label: "DeepGemm",
            flags: ["--moe-runner-backend deep_gemm"],
            disabled: (s) => ["mi300x", "mi325x", "mi350x", "mi355x"].includes(s.hw),
            disableReason: "DeepGemm is not part of the AMD ROCm recipes; gfx942 FP8 uses Triton, while gfx950 FP8 and MXFP4 use AITER.",
          },
        ],
      },
      ep: { label: "EP", values: [
        null, 2, 4,
        {
          value: 8,
          disable: [
            {
              when: { hw: ["gb300", "gb200"] },
              reason: "EP=8 needs 8 GPUs; the GB300 and GB200 recipes run on 4.",
            },
          ],
        },
      ]},
    },

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser glm45" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser glm47" },
      ],
    },

    // ----- Card: "Speculative" -----
    // The Deploy panel only picks speculation through the Strategy dim (Low
    // Latency = the checkpoint's adaptive MTP head, High Throughput = off).
    // This card is the finer control, and it adds the one algorithm no cell
    // ships: DFlash2, whose draft is a separate checkpoint.
    //
    // The EAGLE preset is byte-identical to what the Low Latency cells carry,
    // so a Low Latency base derives onto that chip instead of showing
    // "Inherited from base", and re-picking it is a no-op.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off", label: "Off (greedy)" },
        {
          id: "eagle",
          label: "EAGLE / Adaptive MTP 5-1-6",
          flags: [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 5",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 6",
            "--speculative-adaptive",
          ],
          disable: [
            {
              when: { dpAttnOn: [true] },
              reason: "Adaptive MTP does not support DP-Attention — the server falls back to a static draft depth and warns. Turn DP-Attention off in the Attention card above.",
            },
            {
              when: { hw: ["mi300x", "mi325x", "mi350x", "mi355x"] },
              reason: "MTP speculative decoding has not been validated for GLM-5.3-Flash on AMD ROCm; the Strategy row disables Low Latency there for the same reason.",
            },
          ],
        },
        {
          id: "dflash",
          label: "DFlash2",
          // Block-wise draft: the block size comes from the draft checkpoint,
          // so no --speculative-num-draft-tokens here. The draft is a dense
          // model and does not run on the target's DSA backends, hence the
          // explicit draft attention backend.
          flags: [
            "--speculative-algorithm DFLASH",
            "--speculative-draft-model-path incoai/GLM-5.3-Flash-DFlash2",
            "--speculative-draft-attention-backend fa4",
          ],
          // DFLASH needs this model's hidden-state capture, which landed on the
          // GLM-5.3-Flash support branch (PR #36708 into #36507's
          // xinyuan/glm-5.3-flash-support), not on main — so it postdates the
          // image the Install accordion pins. Drop this note once #36507 merges
          // and a published image carries it.
          note: "⚠️ Needs the GLM-5.3-Flash hidden-state capture from PR #36708. It is merged into the PR #36507 support branch (xinyuan/glm-5.3-flash-support), not into main, so pull that branch at its current head — or add #36708's commit on top of an older checkout — before serving. The lmsysorg/sglang:glm-5.3-flash image alone is not enough.",
          disable: [
            {
              when: { dpAttnOn: [true] },
              reason: "DFLASH speculative decoding does not support DP-Attention — the server rejects the combination at startup. Turn DP-Attention off in the Attention card above.",
            },
            {
              when: { hw: ["mi300x", "mi325x", "mi350x", "mi355x"] },
              reason: "DFLASH speculative decoding only supports CUDA and NPU devices; the server rejects it on ROCm at startup.",
            },
          ],
        },
      ],
    },

  },

  cells: [
    {
      match: { hw: "gb300", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        ["bf16-tilelang", "fp8-trtllm"].includes(s.kvDsaPair) &&
        s.mmTransport === "auto" &&
        s.hicache === "off" &&
        ["off", "4"].includes(s.dcp)
          ? "verified"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend deep_gemm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        ["bf16-tilelang", "fp8-trtllm"].includes(s.kvDsaPair) &&
        s.mmTransport === "auto" &&
        s.hicache === "off" &&
        s.dcp === "off"
          ? "verified"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend deep_gemm",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.mmTransport === "auto" && s.hicache === "off"
          ? "verified"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--mem-fraction-static 0.70",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend deep_gemm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        ["off", "l2"].includes(s.hicache) ? "verified" : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--mem-fraction-static 0.75",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend deep_gemm",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.mmTransport === "auto" && s.hicache === "off"
          ? "verified"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--mem-fraction-static 0.75",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend deep_gemm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        ["off", "l2"].includes(s.hicache) ? "verified" : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend deep_gemm",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.hicache === "off" ? "verified" : "unverified"),
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend deep_gemm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        ["off", "l2"].includes(s.hicache) ? "verified" : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend deep_gemm",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) => (s.hicache === "off" ? "verified" : "unverified"),
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend deep_gemm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        ["off", "l2"].includes(s.hicache) ? "verified" : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend deep_gemm",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", quant: "fp8", strategy: "low-latency" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend deep_gemm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--ep-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend deep_gemm",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // AMD FP8 — use expert parallelism on all eight GPUs instead of replicating
    // every routed expert on every TP rank. The gfx942 cells use Triton MoE;
    // the gfx950 cells use AITER MoE. Both paths use full decode graphs for
    // batch sizes 1 and 32. The MI300X cell completed full GSM8K with this exact
    // TP8+EP8 command. MI325X carries that verification because it uses the same
    // gfx942 path with more HBM; the gfx950 FP8 cells still require full runs.
    {
      match: { hw: "mi300x", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.kvDsaPair === "bf16-tilelang" &&
        s.mmTransport === "auto" &&
        s.hicache === "off" &&
        s.dcp === "off"
          ? "verified"
          : "unverified",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend dsa",
        "--trust-remote-code",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend triton",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-backend-prefill disabled",
        "--cuda-graph-bs-decode 1 32",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi325x", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.kvDsaPair === "bf16-tilelang" &&
        s.mmTransport === "auto" &&
        s.hicache === "off" &&
        s.dcp === "off"
          ? "verified"
          : "unverified",
      warn: "Architecture-equivalent verification: the exact TP8+EP8 graph command completed full GSM8K on MI300X. MI325X uses the same gfx942 path with greater HBM capacity; it was not measured in a separate run.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend dsa",
        "--trust-remote-code",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend triton",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-backend-prefill disabled",
        "--cuda-graph-bs-decode 1 32",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      warn: "The gfx950 TP8+EP8 graph path passed an MI355X runtime pilot, but this exact MI350X command still needs a full accuracy run.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend dsa",
        "--trust-remote-code",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-backend-prefill disabled",
        "--cuda-graph-bs-decode 1 32",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "fp8", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      warn: "The TP8+EP8 topology, AITER MoE path, and decode graphs passed an MI355X runtime pilot. Full GSM8K on this exact command is still pending.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend dsa",
        "--trust-remote-code",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-backend-prefill disabled",
        "--cuda-graph-bs-decode 1 32",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // AMD Quark MXFP4 — MI350X and MI355X share the CDNA4 gfx950 target,
    // 288 GB per GPU, and mi35x ROCm image path, so they use identical cells.
    // Direct validation ran on MI350X. Mixed Quark MXFP4 + block-FP8 uses the
    // AITER MoE runner and full decode graphs. PR #36607 commit 654df43cbee1
    // supplies the required loader compatibility and is merged into #36507.
    {
      match: { hw: "mi350x", quant: "mxfp4", strategy: "mxfp4-tp1" },
      nnodes: 1,
      verified: true,
      warn: "Use the open PR #36507 support branch; it contains the merged AMD changes from PR #36607, including the MXFP4 loader fix.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend dsa",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--context-length 65536",
        "--max-running-requests 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 64",
        "--mem-fraction-static 0.85",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}'",
        "--watchdog-timeout 1200",
        "--trust-remote-code",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", quant: "mxfp4", strategy: "mxfp4-tp2" },
      nnodes: 1,
      verified: true,
      warn: "Use the open PR #36507 support branch; it contains the merged AMD changes from PR #36607, including the MXFP4 loader fix.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 2",
        "--attention-backend dsa",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--context-length 65536",
        "--max-running-requests 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 64",
        "--mem-fraction-static 0.85",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}'",
        "--watchdog-timeout 1200",
        "--trust-remote-code",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", quant: "mxfp4", strategy: "mxfp4-tp4" },
      nnodes: 1,
      verified: true,
      warn: "Use the open PR #36507 support branch; it contains the merged AMD changes from PR #36607, including the MXFP4 loader fix.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--attention-backend dsa",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--context-length 65536",
        "--max-running-requests 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 64",
        "--mem-fraction-static 0.85",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}'",
        "--watchdog-timeout 1200",
        "--trust-remote-code",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", quant: "mxfp4", strategy: "mxfp4-tp8-ep8" },
      nnodes: 1,
      verified: true,
      warn: "Use the open PR #36507 support branch; it contains the merged AMD changes from PR #36607, including the MXFP4 loader fix.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend dsa",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--context-length 65536",
        "--max-running-requests 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 64",
        "--mem-fraction-static 0.85",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}'",
        "--watchdog-timeout 1200",
        "--trust-remote-code",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "mxfp4", strategy: "mxfp4-tp1" },
      nnodes: 1,
      verified: true,
      warn: "Use the open PR #36507 support branch; it contains the merged AMD changes from PR #36607, including the MXFP4 loader fix.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--attention-backend dsa",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--context-length 65536",
        "--max-running-requests 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 64",
        "--mem-fraction-static 0.85",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}'",
        "--watchdog-timeout 1200",
        "--trust-remote-code",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "mxfp4", strategy: "mxfp4-tp2" },
      nnodes: 1,
      verified: true,
      warn: "Use the open PR #36507 support branch; it contains the merged AMD changes from PR #36607, including the MXFP4 loader fix.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 2",
        "--attention-backend dsa",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--context-length 65536",
        "--max-running-requests 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 64",
        "--mem-fraction-static 0.85",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}'",
        "--watchdog-timeout 1200",
        "--trust-remote-code",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "mxfp4", strategy: "mxfp4-tp4" },
      nnodes: 1,
      verified: true,
      warn: "Use the open PR #36507 support branch; it contains the merged AMD changes from PR #36607, including the MXFP4 loader fix.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--attention-backend dsa",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--context-length 65536",
        "--max-running-requests 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 64",
        "--mem-fraction-static 0.85",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}'",
        "--watchdog-timeout 1200",
        "--trust-remote-code",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", quant: "mxfp4", strategy: "mxfp4-tp8-ep8" },
      nnodes: 1,
      verified: true,
      warn: "Use the open PR #36507 support branch; it contains the merged AMD changes from PR #36607, including the MXFP4 loader fix.",
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--attention-backend dsa",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--linear-attn-backend triton",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend aiter",
        "--disable-shared-experts-fusion",
        "--disable-radix-cache",
        "--context-length 65536",
        "--max-running-requests 64",
        "--cuda-graph-backend-decode full",
        "--cuda-graph-max-bs-decode 64",
        "--mem-fraction-static 0.85",
        "--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":8}'",
        "--watchdog-timeout 1200",
        "--trust-remote-code",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
