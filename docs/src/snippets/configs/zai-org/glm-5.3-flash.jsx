export const config = {
  modelName: "GLM-5.3-Flash",

  supportedHardware: ["gb300", "h100", "h200", "b200", "b300", "gb200"],

  matchDims: [
    {
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency", label: "Low Latency", subtitle: "Adaptive MTP 5/1/6" },
        { id: "high-throughput", label: "High Throughput", subtitle: "Spec decode off" },
      ],
    },
    {
      id: "quant",
      title: "Quantization",
      options: [
        { id: "fp8", label: "FP8" },
        {
          id: "nvfp4",
          label: "NVFP4",
          disabled: (s) => !["gb300", "gb200", "b200", "b300"].includes(s.hw),
          disableReason: "The NVFP4 W4A4 kernels are Blackwell-only; Hopper cannot serve this checkpoint.",
        },
      ],
    },
  ],

  isRecommendedSelection(s) {
    const pairing = ["h100", "h200"].includes(s.hw) ? "bf16-tilelang" : "fp8-trtllm";
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
          disabled: (s) => ["h100", "h200"].includes(s.hw),
          disableReason: "FP8 KV cache with TRT-LLM DSA is not supported on Hopper GPUs.",
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
          hints: ["32 GB host tier; the default ratio can demand more host RAM than the node has free."],
        },
        {
          id: "l3",
          label: "+ L3",
          subtitle: "Mooncake",
          flags: ["--enable-hierarchical-cache", "--hicache-size 32", "--hicache-storage-backend mooncake"],
          env: ["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH={{MOONCAKE_CONFIG}}"],
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
    default: "zai-org/GLM-5.3-Flash",
    nvfp4: "RadixArk/GLM-5.3-Flash-NVFP4",
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
    ["aime2026_pct", "AIME 2026", "%"],
  ],

  // Support is not in a public sglang release yet, so the nightly images do
  // not work; every NVIDIA lane uses the purpose-built CUDA 13 image.
  dockerImages: {
    gb300: "lmsysorg/sglang:glm-5.3-flash",
    h100: "lmsysorg/sglang:glm-5.3-flash",
    h200: "lmsysorg/sglang:glm-5.3-flash",
    b200: "lmsysorg/sglang:glm-5.3-flash",
    b300: "lmsysorg/sglang:glm-5.3-flash",
    gb200: "lmsysorg/sglang:glm-5.3-flash",
  },

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
          ],
        },
      ],
    },

  },

  cells: [
    {
      match: { hw: "gb300", strategy: "low-latency", quant: "fp8" },
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
      match: { hw: "gb300", strategy: "high-throughput", quant: "fp8" },
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
    // RadixArk NVFP4 W4A4 checkpoint (ModelOpt 0.46.0, abs-max, group size
    // 16): routed and shared experts plus the dense MLPs are FP4; attention,
    // router, MTP, embeddings, and the vision tower stay BF16. Validated on
    // 4x GB300 on the stock image with both KV/DSA pairings: the speed rows
    // were measured with BF16 KV + TileLang DSA, while FP8 KV + TRT-LLM DSA
    // passed smoke, a 200-example GSM8K check, a 600-request soak, and the
    // TB2.1 run without separate speed measurements.
    {
      match: { hw: "gb300", strategy: "low-latency", quant: "nvfp4" },
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
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend flashinfer_cutlass",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", strategy: "high-throughput", quant: "nvfp4" },
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
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend flashinfer_cutlass",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // Same NVFP4 recipe on the remaining Blackwell platforms, at each fp8
    // cell's parallelism (gb200 TP4/EP4, b200/b300 TP8/EP8). Not measured on
    // this hardware, so every cell here reports unverified.
    {
      match: { hw: "gb200", strategy: "low-latency", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--ep-size 4",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend flashinfer_cutlass",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", strategy: "high-throughput", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--ep-size 4",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend flashinfer_cutlass",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", strategy: "low-latency", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend flashinfer_cutlass",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", strategy: "high-throughput", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend flashinfer_cutlass",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", strategy: "low-latency", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend flashinfer_cutlass",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--speculative-adaptive",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 32",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", strategy: "high-throughput", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend flashinfer_cutlass",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", strategy: "low-latency", quant: "fp8" },
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
      match: { hw: "h100", strategy: "high-throughput", quant: "fp8" },
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
      match: { hw: "h200", strategy: "low-latency", quant: "fp8" },
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
      match: { hw: "h200", strategy: "high-throughput", quant: "fp8" },
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
      match: { hw: "b200", strategy: "low-latency", quant: "fp8" },
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
      match: { hw: "b200", strategy: "high-throughput", quant: "fp8" },
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
      match: { hw: "b300", strategy: "low-latency", quant: "fp8" },
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
      match: { hw: "b300", strategy: "high-throughput", quant: "fp8" },
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
      match: { hw: "gb200", strategy: "low-latency", quant: "fp8" },
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
      match: { hw: "gb200", strategy: "high-throughput", quant: "fp8" },
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
  ],
};
