export const config = {
  modelName: "GLM-5.3-Flash",

  supportedHardware: ["b300", "gb300", "h100", "h200", "b200", "gb200", "mi355x", "tpu7x", "a3", "a5"],

  // One TPU7x (Ironwood) host: 4 chips, 8 TensorCores, served by sglang-torchtpu.
  hardware: [{ id: "tpu7x", label: "TPU7x", vram: "192GB", vendor: "tpu" },
    { id: "a3", label: "A3 Series", vram: "64GB/die", vendor: "npu", npuDevices: 16 },
    { id: "a5", label: "950PR/DT series", vram: "128GB", vendor: "npu", npuDevices: 8 },
  ],

  matchDims: [
    {
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency", label: "Low Latency", subtitle: "MTP 5/1/6" },
        {
          id: "high-throughput",
          label: "High Throughput",
          subtitle: "Spec decode off",
          disabled: (s) => s.hw === "tpu7x",
          disableReason: "The TPU7x recipe is qualified for Low Latency only.",
        },
      ],
    },
    {
      id: "quant",
      title: "Quantization",
      options: [
        { showWhen: (s) => s.hw === "a3", id: "w8a8", label: "W8A8 (ModelSlim)", disabled: (s) => s.hw !== "a3", disableReason: "The W8A8 checkpoint is the A3 recipe." },
        { showWhen: (s) => s.hw === "a5", id: "w8a8-mxfp8", label: "W8A8_MXFP8 (ModelSlim)", disabled: (s) => s.hw !== "a5", disableReason: "The W8A8_MXFP8 checkpoint is the 950PR/DT series recipe." },
        { showWhen: (s) => !["a3", "a5"].includes(s.hw), id: "fp8", label: "FP8" },
        { showWhen: (s) => !["a3", "a5"].includes(s.hw),
          id: "mxfp4",
          label: "MXFP4",
          disabled: (s) => s.hw !== "mi355x",
          disableReason: "The MXFP4 recipe is qualified on MI355X only.",
        },
        { showWhen: (s) => !["a3", "a5"].includes(s.hw),
          id: "nvfp4",
          label: "NVFP4 (RDXA)",
          disabled: (s) => !["gb300", "gb200", "b200", "b300"].includes(s.hw),
          disableReason: "The NVFP4 W4A4 kernels are Blackwell-only; Hopper cannot serve this checkpoint.",
        },
        { showWhen: (s) => !["a3", "a5"].includes(s.hw),
          id: "nvfp4-nvda",
          label: "NVFP4 (NVDA)",
          disabled: (s) => !["gb300", "gb200", "b200", "b300"].includes(s.hw),
          disableReason: "The NVFP4 W4A4 kernels are Blackwell-only; Hopper cannot serve this checkpoint.",
        },
      ],
    },
  ],

  isRecommendedSelection(s) {
    if (["a3", "a5"].includes(s.hw)) return false;
    const pairing = s.hw === "mi355x" ? "fp8-tilelang"
      : s.hw === "tpu7x" ? "fp8-pallas"
      : ["h100", "h200"].includes(s.hw) ? "bf16-tilelang" : "fp8-trtllm";
    return (
      s.kvDsaPair === pairing &&
      s.mmTransport === "auto" &&
      s.hicache === "off" &&
      s.bcg === "off"
    );
  },

  overlayDims: [
    {
      id: "pdMode", title: "NPU PD mode", default: "unified",
      showWhen: (s) => ["a3", "a5"].includes(s.hw),
      options: [
        { id: "unified", label: "Unified" },
        { id: "prefill", label: "Prefill" },
        { id: "decode", label: "Decode" },
        { id: "router", label: "Router" },
      ],
    },

    {
      id: "kvDsaPair",
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      title: "KV Cache + DSA Backend",
      default: "fp8-trtllm",
      options: [
        {
          id: "fp8-trtllm",
          label: "FP8 + TRT-LLM",
          disabled: (s) => ["h100", "h200", "mi355x", "tpu7x"].includes(s.hw),
          disableReason: "FP8 KV cache with TRT-LLM DSA is not supported on Hopper, MI355X or TPU7x.",
          stripPrefixes: ["--kv-cache-dtype", "--dsa-prefill-backend", "--dsa-decode-backend"],
          flags: [
            "--kv-cache-dtype fp8_e4m3",
            "--dsa-prefill-backend trtllm",
            "--dsa-decode-backend trtllm",
          ],
          hints: ["Measured on GB300: faster than BF16 + TileLang with about 1.8x the KV token capacity."],
        },
        {
          id: "fp8-tilelang",
          label: "FP8 + TileLang (ROCm)",
          disabled: (s) => s.hw !== "mi355x",
          disableReason: "This FP8 KV + TileLang DSA pairing is qualified on ROCm, not CUDA.",
          stripPrefixes: ["--kv-cache-dtype", "--dsa-prefill-backend", "--dsa-decode-backend"],
          flags: [
            "--kv-cache-dtype fp8_e4m3",
            "--dsa-prefill-backend tilelang",
            "--dsa-decode-backend tilelang",
          ],
        },
        {
          id: "bf16-tilelang",
          label: "BF16 + TileLang",
          disabled: (s) => s.hw === "tpu7x",
          disableReason: "TileLang is not available on TPU; TPU7x uses the Pallas sparse MLA kernels.",
          stripPrefixes: ["--kv-cache-dtype", "--dsa-prefill-backend", "--dsa-decode-backend"],
          flags: [
            "--kv-cache-dtype bfloat16",
            "--dsa-prefill-backend tilelang",
            "--dsa-decode-backend tilelang",
          ],
        },
        // Order matters: a disabled pick is re-seated onto the first enabled
        // option, so this entry must stay after BF16 + TileLang to keep the
        // Hopper default unchanged.
        {
          id: "bf16-q8-prefill",
          label: "BF16 + Q8 sparse prefill",
          disabled: (s) => !["h100", "h200"].includes(s.hw),
          disableReason: "The Q8KV8 sparse prefill kernel is SM90-only (Hopper).",
          stripPrefixes: ["--kv-cache-dtype", "--dsa-prefill-backend", "--dsa-decode-backend"],
          flags: [
            "--kv-cache-dtype bfloat16",
            "--dsa-prefill-backend flashmla_sparse_q8",
            "--dsa-decode-backend tilelang",
          ],
          hints: ["Measured on H200: higher prefill throughput than BF16 + TileLang; requires a build with PR #41790."],
        },
        {
          id: "fp8-pallas",
          label: "FP8 + Pallas (TPU)",
          disabled: (s) => s.hw !== "tpu7x",
          disableReason: "The Pallas sparse MLA kernels are TPU-only.",
          stripPrefixes: ["--kv-cache-dtype", "--attention-backend"],
          flags: [
            "--kv-cache-dtype fp8_e4m3",
            "--attention-backend pallas_tpu_mla_sparse",
          ],
        },
      ],
    },
    {
      id: "bcg",
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      title: "Breakable Cuda Graph",
      default: "off",
      options: [
        { id: "off", label: "Off" },
        {
          id: "on",
          label: "On",
          disabled: (s) => ["mi355x", "tpu7x"].includes(s.hw),
          disableReason: "The breakable prefill CUDA graph path is not qualified on ROCm or TPU.",
          flags: ["--cuda-graph-backend-prefill breakable"],
          hints: ["Enables breakable prefill CUDA graphs; requires a build with PR #38522."],
        },
      ],
    },
    {
      id: "mmTransport",
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      title: "VLM Transport",
      default: "auto",
      options: [
        { id: "auto", label: "Auto", subtitle: "Topology-aware",
          flags: (s) => s.hw === "mi355x" ? ["--mm-feature-transport cpu"] : [] },
        {
          id: "cpu",
          label: "CPU",
          subtitle: "Save GPU memory",
          disabled: (s) => s.hw === "tpu7x",
          disableReason: "The TPU7x recipe serves the language model only (--language-model-only).",
          stripPrefixes: ["--mm-feature-transport"],
          flags: ["--mm-feature-transport cpu"],
        },
      ],
    },
    {
      id: "hicache",
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      title: "HiCache",
      default: "off",
      options: [
        { id: "off", label: "Off" },
        {
          id: "l2",
          label: "L1 + L2",
          subtitle: "Host memory",
          disabled: (s) => s.hw === "tpu7x",
          disableReason: "HiCache is not qualified for GLM-5.3-Flash on TPU7x.",
          flags: (s) => s.hw === "mi355x"
            ? ["--enable-hierarchical-cache", "--hicache-size 180",
              "--hicache-write-policy write_through"]
            : ["--enable-hierarchical-cache", "--hicache-size 32"],
          hints: (s) => s.hw === "mi355x"
            ? ["About 750 GB of host memory per node; the measured MI355X agentic recipe."]
            : ["32 GB host tier; the default ratio can demand more host RAM than the node has free."],
        },
        {
          id: "l3",
          label: "+ L3",
          subtitle: "Mooncake",
          disabled: (s) => s.hw === "tpu7x",
          disableReason: "HiCache is not qualified for GLM-5.3-Flash on TPU7x.",
          flags: ["--enable-hierarchical-cache", "--hicache-size 32", "--hicache-storage-backend mooncake"],
          env: ["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH={{MOONCAKE_CONFIG}}"],
          hints: ["Start Mooncake and place the configuration file on every serving node."],
        },
      ],
    },
    // Parser flags live in one overlay dim so every generated command gets
    // them without per-cell duplication; the Parsers card toggles derive
    // on/off from the composed flags. `auto` needs the GLM-5.3 template
    // detection (v0.5.20+).
    {
      id: "parsers",
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      title: "Parsers",
      default: "auto",
      options: [
        {
          id: "auto",
          label: "Auto (glm45 + glm47)",
          stripPrefixes: ["--reasoning-parser", "--tool-call-parser"],
          flags: [
            "--reasoning-parser auto",
            "--tool-call-parser auto",
          ],
        },
        {
          id: "off",
          label: "Off",
          stripPrefixes: ["--reasoning-parser", "--tool-call-parser"],
          flags: [],
        },
      ],
    },
  ],

  modelNames: {
    "a3|w8a8": "GLM-5.3-Flash",
    "a5|w8a8-mxfp8": "GLM-5.3-Flash",
    default: "zai-org/GLM-5.3-Flash",
    nvfp4: "RadixArk/GLM-5.3-Flash-NVFP4",
    "mi355x|mxfp4": "OneNexus/GLM-5.3-Flash-MXFP4",
    "nvfp4-nvda": "nvidia/GLM-5.3-Flash-NVFP4",
  },

  // MI355X uses a ROCm environment and TPU7x an sglang-torchtpu install; the
  // Docker images below are NVIDIA-only.
  runModes: (s) => ["mi355x", "tpu7x", "a3", "a5"].includes(s.hw) ? ["python"] : ["python", "docker"],

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
`# Low Latency speed rows whose notes mention SGLANG_SIMULATE_ACC_LEN=3 were served
# with it to pin the accept length; that number is throughput evidence only. Never
# run accuracy against it.
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
`# To install sgl-eval: pip install sgl-eval
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

  // v0.5.20 (= latest) carries GLM-5.3-Flash support (#36507) and the GLM-5.3
  // template parser detection (#38297) that `--*-parser auto` needs; the old
  // glm-5.3-flash dev image (2026-09-03) predates #38297 and misdetects.
  dockerImages: {
    a3: "swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-glm5.3-flash",
    gb300: "lmsysorg/sglang:latest",
    h100: "lmsysorg/sglang:latest",
    h200: "lmsysorg/sglang:latest",
    b200: "lmsysorg/sglang:latest",
    b300: "lmsysorg/sglang:latest",
    gb200: "lmsysorg/sglang:latest",
  },

  github: {
    cookbookModel: "zai-org/glm-5.3-flash",
  },

  playgroundFeatures: {
    attention: {
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
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
        {
          id: "cp", label: "CP",
          values: [null, 1, 2, 4, {
            value: 8,
            disable: { hw: ["gb300", "gb200", "h100", "h200", "mi355x", "tpu7x"] },
            disableReason: "CP=8 is available on the eight-GPU B200 and B300 recipes. Set TP=8 first.",
          }],
          disable: { pdMode: ["prefill", "decode"] },
          disableReason: "GLM-5.3-Flash CP/TP group sharing does not support PD disaggregation. Set PD mode to Off to enable CP.",
        },
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
              reason: "MTP speculative decoding with DP-Attention is not validated on this model.",
            },
          ],
          disableReason: "MTP speculative decoding with DP-Attention is not validated on this model.",
        },
      ],
    },

    moe: {
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      backend: {
        options: [
          { id: null, label: "Inherited" },
          {
            id: "flashinfer_trtllm",
            label: "flashinfer_trtllm",
            flags: ["--moe-runner-backend flashinfer_trtllm"],
            disable: { hw: ["h100", "h200", "mi355x", "tpu7x"] },
            disableReason: "Use flashinfer_trtllm on B200, B300, GB200 or GB300; Hopper uses DeepGemm, MI355X uses AITER and TPU7x its own Pallas MoE kernel.",
          },
          {
            id: "deep_gemm",
            label: "DeepGemm",
            flags: ["--moe-runner-backend deep_gemm"],
            disable: [
              { when: { hw: ["mi355x"] }, reason: "DeepGemm is a CUDA backend; MI355X uses AITER." },
              { when: { hw: ["tpu7x"] }, reason: "DeepGemm is a CUDA backend; TPU7x uses its own Pallas MoE kernel." },
            ],
          },
          {
            id: "aiter",
            label: "AITER (ROCm)",
            flags: ["--moe-runner-backend aiter"],
            disable: [{ when: { hw: ["gb300", "h100", "h200", "b200", "b300", "gb200", "tpu7x"] }, reason: "AITER is a ROCm backend." }],
          },
        ],
      },
      ep: { label: "EP", matchCp: true, denseTpMatchesCp: true, values: [
        null, 1, 2, 4,
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

    pdDisagg: {
      modes: [
        { id: "off", label: "Off" },
        { id: "prefill", label: "Prefill",
          stripFlags: ["--attn-cp-size", "--enable-prefill-cp", "--cp-strategy"] },
        { id: "decode", label: "Decode",
          stripFlags: ["--attn-cp-size", "--enable-prefill-cp", "--cp-strategy"] },
      ],
      transferBackends: [{ id: "mooncake", label: "Mooncake" }],
      notes: [{ mode: ["prefill", "decode"], when: { hw: ["tpu7x"] },
        text: "PD disaggregation is not qualified for GLM-5.3-Flash on TPU7x." },
        { mode: ["prefill", "decode"], when: { pdMode: ["prefill", "decode"] },
        text: "PD serving is a preview. Prefill CP is disabled: CP/TP group sharing cannot currently transfer the KDA state between PD workers." }],
    },

    parsers: {
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    // ----- Card: "Speculative" -----
    // The Deploy panel only picks speculation through the Strategy dim (Low
    // Latency = the checkpoint's MTP head, High Throughput = off).
    // This card is the finer control, and it adds the one algorithm no cell
    // ships: DFlash2, whose draft is a separate checkpoint.
    //
    // The EAGLE preset is byte-identical to what the Low Latency cells carry,
    // so a Low Latency base derives onto that chip instead of showing
    // "Inherited from base", and re-picking it is a no-op.
    speculative: {
      showWhen: (s) => !["a3", "a5"].includes(s.hw),
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off", label: "Off (greedy)" },
        {
          id: "eagle",
          label: "EAGLE / MTP 5-1-6",
          flags: [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 5",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 6",
          ],
          disable: [
            {
              when: { dpAttnOn: [true] },
              reason: "MTP speculative decoding with DP-Attention is not validated on this model. Turn DP-Attention off in the Attention card above.",
            },
            {
              when: { hw: ["tpu7x"] },
              reason: "TPU7x runs MTP through the NEXTN flags its Low Latency recipe already carries.",
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
          note: "⚠️ The draft checkpoint incoai/GLM-5.3-Flash-DFlash2 is access-gated: request access on its Hugging Face page, then download it alongside the target before serving.",
          disable: [
            { when: { hw: ["mi355x", "tpu7x"] }, reason: "The FA4 draft-attention path is CUDA-only." },
            {
              when: { dpAttnOn: [true] },
              reason: "DFLASH speculative decoding does not support DP-Attention — the server rejects the combination at startup. Turn DP-Attention off in the Attention card above.",
            },
          ],
        },
      ],
    },

  },

  // Ascend recipes are transcribed from the linked tutorial. PD roles own their
  // worker topology; changing a GPU overlay must never rewrite these commands.
  resolveRecipe: (cell, s) => {
    if (!(["a3", "a5"].includes(s.hw)) || !["prefill", "decode", "router"].includes(s.pdMode)) return cell;
    const router = { port: 30000, command: config.npuRouterCommand };
    if (s.pdMode === "router") return { ...cell, pd: true, pdMode: "router",
      nnodes: 1, verified: false, env: [], flags: [], warn: undefined,
      commands: { python: config.npuRouterCommand, docker: `docker run --network host --rm swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-glm5.3-flash ${config.npuRouterCommand}` }, router };
    const recipe = config.npuPdRecipes.find((r) => r.hw === s.hw && r.pdMode === s.pdMode);
    if (!recipe) return cell;
    const flags = recipe.flags.map((flag) => flag.startsWith("--port ")
      ? "--port " + (recipe.pdMode === "decode" ? 30100 : 30000) : flag);
    if (recipe.nnodes > 1) flags.push("--nnodes " + recipe.nnodes,
      "--node-rank <node-rank>", "--dist-init-addr <node0-ip>:20000");
    const command = [...recipe.env, "sglang serve", ...flags.map((flag) => "  " + flag)].join(" \\\n");
    return { ...cell, ...recipe, pd: true, verified: false, warn: undefined, router,
      commands: { python: command } };
  },

  npuRouterCommand: "python3 -m sglang_router.launch_router \\\n  --pd-disaggregation \\\n  --prefill http://<prefill-head-ip>:30000 8998 \\\n  --decode http://<decode-head-ip>:30100 \\\n  --host {{HOST_IP}} --port {{PORT}} \\\n  --policy round_robin \\\n  --worker-startup-timeout-secs 3600 \\\n  --request-timeout-secs 7200 \\\n  --disable-retries",

  npuPdRecipes: [
    {
      hw: "a3",
      pdMode: "prefill",
      nnodes: 1,
      env: [
        "SGLANG_SET_CPU_AFFINITY=1",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "STREAMS_PER_DEVICE=32",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "HCCL_SOCKET_IFNAME=\"<network-interface>\"",
        "GLOO_SOCKET_IFNAME=\"<network-interface>\"",
        "ASCEND_MF_STORE_URL=\"tcp://<prefill-head-ip>:24707\"",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200",
        "SGLANG_HOST_IP=\"<this-node-ip>\"",
        "HCCL_BUFFSIZE=1200",
        "DEEPEP_HCCL_BUFFSIZE=1200",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=128",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
        "TASK_QUEUE_ENABLE=2",
        "HCCL_IF_BASE_PORT=61000"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3-Flash-w8a8\"",
        "--trust-remote-code",
        "--tp-size 4",
        "--pp-size 4",
        "--dp-size 4",
        "--quantization modelslim",
        "--device npu",
        "--attention-backend ascend",
        "--enable-multimodal",
        "--served-model-name GLM-5.3-Flash",
        "--disaggregation-mode prefill",
        "--disaggregation-bootstrap-port 8998",
        "--disaggregation-transfer-backend ascend",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--moe-dense-tp 1",
        "--moe-a2a-backend deepep",
        "--deepep-mode normal",
        "--load-balance-method round_robin",
        "--mem-fraction-static 0.78",
        "--page-size 128",
        "--max-running-requests 16",
        "--chunked-prefill-size 4096",
        "--max-prefill-tokens 8192",
        "--watchdog-timeout 9000",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      hw: "a3",
      pdMode: "decode",
      nnodes: 1,
      env: [
        "SGLANG_SET_CPU_AFFINITY=1",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "STREAMS_PER_DEVICE=32",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "HCCL_SOCKET_IFNAME=\"<network-interface>\"",
        "GLOO_SOCKET_IFNAME=\"<network-interface>\"",
        "ASCEND_MF_STORE_URL=\"tcp://<prefill-head-ip>:24707\"",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200",
        "SGLANG_HOST_IP=\"<this-node-ip>\"",
        "HCCL_BUFFSIZE=512",
        "DEEPEP_HCCL_BUFFSIZE=512",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16",
        "SGLANG_ENABLE_SPEC_V2=1",
        "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW=1",
        "SGLANG_NPU_USE_MULTI_STREAM=1",
        "TASK_QUEUE_ENABLE=0",
        "HCCL_IF_BASE_PORT=61010"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3-Flash-w8a8\"",
        "--trust-remote-code",
        "--tp-size 16",
        "--pp-size 1",
        "--dp-size 16",
        "--quantization modelslim",
        "--device npu",
        "--attention-backend ascend",
        "--enable-multimodal",
        "--served-model-name GLM-5.3-Flash",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend ascend",
        "--enable-dp-attention",
        "--moe-a2a-backend deepep",
        "--deepep-mode low_latency",
        "--load-balance-method round_robin",
        "--mem-fraction-static 0.75",
        "--page-size 128",
        "--max-running-requests 32",
        "--cuda-graph-bs-decode 1 2",
        "--speculative-draft-model-path \"/models/GLM-5.3-Flash-w8a8\"",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--watchdog-timeout 9000",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      hw: "a5",
      pdMode: "prefill",
      nnodes: 1,
      env: [
        "SGLANG_NPU_PROFILING=0",
        "HCCL_BUFFSIZE=256",
        "ASCEND_MF_STORE_URL=\"tcp://<prefill-head-ip>:24670\"",
        "ASCEND_MF_TRANSFER_PROTOCOL=device_urma",
        "SGLANG_HOST_IP=\"<this-node-ip>\""
      ],
      flags: [
        "--model-path \"/models/GLM-5.3-Flash-w8a8-mxfp8\"",
        "--trust-remote-code",
        "--tp-size 8",
        "--pp-size 1",
        "--dp-size 2",
        "--quantization modelslim",
        "--load-format auto",
        "--device npu",
        "--attention-backend ascend",
        "--served-model-name GLM-5.3-Flash",
        "--disaggregation-mode prefill",
        "--disaggregation-bootstrap-port 8998",
        "--disaggregation-transfer-backend ascend",
        "--enable-dp-attention",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--mem-fraction-static 0.83",
        "--page-size 64",
        "--max-running-requests 16",
        "--speculative-draft-model-path \"/models/GLM-5.3-Flash-w8a8-mxfp8\"",
        "--speculative-draft-kv-cache-dtype bf16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 4",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 5",
        "--pre-warm-nccl",
        "--watchdog-timeout 1200",
        "--cuda-graph-bs-decode 1 8 64",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      hw: "a5",
      pdMode: "decode",
      nnodes: 1,
      env: [
        "SGLANG_NPU_PROFILING=0",
        "HCCL_BUFFSIZE=256",
        "ASCEND_MF_STORE_URL=\"tcp://<prefill-head-ip>:24670\"",
        "ASCEND_MF_TRANSFER_PROTOCOL=device_urma",
        "SGLANG_HOST_IP=\"<this-node-ip>\""
      ],
      flags: [
        "--model-path \"/models/GLM-5.3-Flash-w8a8-mxfp8\"",
        "--trust-remote-code",
        "--tp-size 8",
        "--pp-size 1",
        "--dp-size 2",
        "--quantization modelslim",
        "--load-format auto",
        "--device npu",
        "--attention-backend ascend",
        "--served-model-name GLM-5.3-Flash",
        "--disaggregation-mode decode",
        "--disaggregation-bootstrap-port 8998",
        "--disaggregation-transfer-backend ascend",
        "--enable-dp-attention",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--mem-fraction-static 0.83",
        "--page-size 64",
        "--max-running-requests 16",
        "--speculative-draft-model-path \"/models/GLM-5.3-Flash-w8a8-mxfp8\"",
        "--speculative-draft-kv-cache-dtype bf16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 4",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 5",
        "--pre-warm-nccl",
        "--watchdog-timeout 1200",
        "--cuda-graph-bs-decode 1 8 64",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    }
  ],

  cells: [
    // The first cell seeds both widgets on a fresh page load.
    {
      match: { hw: "b300", strategy: "low-latency", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" : (s.hicache === "off" ? "in-progress" : "unverified"),
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // Four MI355X VFs, TP4/EP4. The MXFP4 Low Latency chunk and memory fraction
    // come from AgentX with the 180 GB/rank HiCache L2 tier; the other cells
    // are supported starting points, not measurements of every overlay.
    {
      match: { hw: "mi355x", strategy: "low-latency", quant: "fp8" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_USE_AITER=1", "AITER_ONLINE_TUNE=0",
        "SGLANG_OPT_FUSED_KDA_VERIFY=1", "SGLANG_DSA_FUSE_TOPK=0",
        "SGLANG_ENABLE_WAR_BARRIER=1", "SGLANG_FORCE_COARSE_WAR_BARRIER=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--quantization fp8",
        "--tp-size 4", "--ep-size 4",
        "--page-size 64",
        "--mem-fraction-static 0.82", "--max-running-requests 96",
        "--cuda-graph-max-bs-decode 96", "--min-free-slots-delay 1",
        "--chunked-prefill-size 8192", "--max-prefill-tokens 8192",
        "--prefill-decode-interval 0",
        "--dsa-prefill-backend tilelang", "--dsa-decode-backend tilelang",
        "--linear-attn-verify-backend triton", "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend aiter",
        "--speculative-algorithm EAGLE", "--speculative-num-steps 5",
        "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 6",
        "--speculative-attention-mode prefill",
        "--speculative-accept-threshold-single 1.0",
        "--speculative-accept-threshold-acc 1.0",
        "--speculative-moe-a2a-backend none",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47", "--watchdog-timeout 1800",
        "--dist-timeout 600", "--host {{HOST_IP}}", "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", strategy: "low-latency", quant: "mxfp4" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_USE_AITER=1", "AITER_ONLINE_TUNE=0",
        "SGLANG_OPT_FUSED_KDA_VERIFY=1", "SGLANG_DSA_FUSE_TOPK=0",
        "SGLANG_ENABLE_WAR_BARRIER=1", "SGLANG_FORCE_COARSE_WAR_BARRIER=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--quantization quark",
        "--tp-size 4", "--ep-size 4",
        "--page-size 64",
        "--mem-fraction-static 0.85", "--max-running-requests 96",
        "--cuda-graph-max-bs-decode 96", "--min-free-slots-delay 1",
        "--chunked-prefill-size 65536", "--max-prefill-tokens 65536",
        "--prefill-decode-interval 0",
        "--dsa-prefill-backend tilelang", "--dsa-decode-backend tilelang",
        "--linear-attn-verify-backend triton", "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend aiter",
        "--speculative-algorithm EAGLE", "--speculative-num-steps 5",
        "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 6",
        "--speculative-attention-mode prefill",
        "--speculative-accept-threshold-single 1.0",
        "--speculative-accept-threshold-acc 1.0",
        "--speculative-moe-a2a-backend none",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47", "--watchdog-timeout 1800",
        "--dist-timeout 600", "--host {{HOST_IP}}", "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", strategy: "high-throughput", quant: "fp8" },
      nnodes: 1,
      verified: false,
      env: ["SGLANG_USE_AITER=1", "AITER_ONLINE_TUNE=0"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code", "--quantization fp8",
        "--tp-size 4", "--ep-size 4",
        "--page-size 64", "--mem-fraction-static 0.82",
        "--max-running-requests 96", "--cuda-graph-max-bs-decode 96",
        "--min-free-slots-delay 1", "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192", "--prefill-decode-interval 0",
        "--dsa-prefill-backend tilelang", "--dsa-decode-backend tilelang",
        "--linear-attn-verify-backend triton", "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend aiter",
        "--reasoning-parser glm45", "--tool-call-parser glm47",
        "--watchdog-timeout 1800", "--dist-timeout 600",
        "--host {{HOST_IP}}", "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", strategy: "high-throughput", quant: "mxfp4" },
      nnodes: 1,
      verified: false,
      env: ["SGLANG_USE_AITER=1", "AITER_ONLINE_TUNE=0"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code", "--quantization quark",
        "--tp-size 4", "--ep-size 4",
        "--page-size 64", "--mem-fraction-static 0.82",
        "--max-running-requests 96", "--cuda-graph-max-bs-decode 96",
        "--min-free-slots-delay 1", "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192", "--prefill-decode-interval 0",
        "--dsa-prefill-backend tilelang", "--dsa-decode-backend tilelang",
        "--linear-attn-verify-backend triton", "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend aiter",
        "--reasoning-parser glm45", "--tool-call-parser glm47",
        "--watchdog-timeout 1800", "--dist-timeout 600",
        "--host {{HOST_IP}}", "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", strategy: "low-latency", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" || s.mmTransport !== "auto" || s.hicache !== "off" ? "unverified" :
        s.kvDsaPair === "fp8-trtllm" ? "verified" :
        s.kvDsaPair === "bf16-tilelang"
          ? "in-progress"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", strategy: "high-throughput", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" :
        ["bf16-tilelang", "fp8-trtllm"].includes(s.kvDsaPair) &&
        s.mmTransport === "auto" &&
        s.hicache === "off"
          ? "in-progress"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // RadixArk NVFP4 W4A4 checkpoint (ModelOpt 0.46.0, abs-max, group size
    // 16): routed and shared experts plus the dense MLPs are FP4; attention,
    // router, MTP, embeddings, and the vision tower stay BF16. Commands match
    // the NVFP4 (NVDA) cells. Only the 8x B300 cells are measured.
    {
      match: { hw: "gb300", strategy: "low-latency", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", strategy: "high-throughput", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", strategy: "low-latency", quant: "nvfp4" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
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
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
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
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
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
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", strategy: "low-latency", quant: "nvfp4" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg === "off" &&
        s.kvDsaPair === "fp8-trtllm" &&
        s.mmTransport === "auto" &&
        s.hicache === "off"
          ? "verified"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", strategy: "high-throughput", quant: "nvfp4" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg === "off" &&
        s.kvDsaPair === "fp8-trtllm" &&
        s.mmTransport === "auto" &&
        s.hicache === "off"
          ? "verified"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // NVIDIA NVFP4 checkpoint (nvidia/GLM-5.3-Flash-NVFP4, ModelOpt 0.47.0,
    // group size 16). Same FP8 KV + TRT-LLM DSA and flashinfer_trtllm MoE
    // recipe as the FP8 Blackwell cells, at each platform's TP size (gb300 /
    // gb200 TP4, b200 / b300 TP8). Only the 8x B300 cells are measured.
    {
      match: { hw: "gb300", strategy: "low-latency", quant: "nvfp4-nvda" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", strategy: "high-throughput", quant: "nvfp4-nvda" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", strategy: "low-latency", quant: "nvfp4-nvda" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", strategy: "high-throughput", quant: "nvfp4-nvda" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 4",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", strategy: "low-latency", quant: "nvfp4-nvda" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", strategy: "high-throughput", quant: "nvfp4-nvda" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", strategy: "low-latency", quant: "nvfp4-nvda" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg === "off" &&
        s.kvDsaPair === "fp8-trtllm" &&
        s.mmTransport === "auto" &&
        s.hicache === "off"
          ? "verified"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", strategy: "high-throughput", quant: "nvfp4-nvda" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg === "off" &&
        s.kvDsaPair === "fp8-trtllm" &&
        s.mmTransport === "auto" &&
        s.hicache === "off"
          ? "verified"
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--quantization modelopt_fp4",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // Hopper keeps EP8 + deep_gemm: FlashInfer's trtllm-gen MoE kernels are SM100-only.
    {
      match: { hw: "h100", strategy: "low-latency", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" :
        s.mmTransport === "auto" && s.hicache === "off"
          ? "in-progress"
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
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", strategy: "high-throughput", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" :
        ["off", "l2"].includes(s.hicache)
          ? (s.kvDsaPair === "bf16-q8-prefill" ? "in-progress" : "verified")
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
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", strategy: "low-latency", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" :
        s.mmTransport === "auto" && s.hicache === "off"
          ? "in-progress"
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
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", strategy: "high-throughput", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" :
        ["off", "l2"].includes(s.hicache)
          ? (s.kvDsaPair === "bf16-q8-prefill" ? "in-progress" : "verified")
          : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--ep-size 8",
        "--dsa-prefill-backend tilelang",
        "--dsa-decode-backend tilelang",
        "--kv-cache-dtype bfloat16",
        "--moe-runner-backend deep_gemm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", strategy: "low-latency", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" : (s.hicache === "off" ? "in-progress" : "unverified"),
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", strategy: "high-throughput", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" :
        ["off", "l2"].includes(s.hicache) ? "in-progress" : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", strategy: "high-throughput", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.bcg !== "off" ? "unverified" :
        ["off", "l2"].includes(s.hicache) ? "in-progress" : "unverified",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
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
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--speculative-algorithm EAGLE",
        "--speculative-num-steps 5",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 6",
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
        "--dsa-prefill-backend trtllm",
        "--dsa-decode-backend trtllm",
        "--kv-cache-dtype fp8_e4m3",
        "--moe-runner-backend flashinfer_trtllm",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // One TPU7x host (4 chips, 8 TensorCores), sglang-torchtpu main: DSA through the
    // Pallas sparse MLA kernels, MTP 3/1/4 on the chain spec worker. Sized for
    // 8k-in / 1k-out at concurrency 1.
    {
      match: { hw: "tpu7x", strategy: "low-latency", quant: "fp8" },
      nnodes: 1,
      verified: true,
      verificationStatus: (s) =>
        s.kvDsaPair === "fp8-pallas" && s.parsers === "auto" ? "verified" : "unverified",
      env: [
        "SGLANG_PLATFORM=tpu",
        'LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_latency_hiding_scheduler=true --xla_latency_hiding_scheduler_rerun=10 --xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative --xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_buffer_assignment_enable_fast_merge_fallback=false"',
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false",
        "SGLANG_MAMBA_SSM_DTYPE=float32",
        "SGLANG_OPT_USE_TILELANG_MHC_PRE=0",
        "SGLANG_OPT_USE_TILELANG_MHC_POST=0",
        "SGLANG_OPT_FUSE_MHC_POST_PRE=0",
        "SGLANG_TPU_ENABLE_GLM5_NEXT_FUSED_MHC=1",
        "SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL=1",
        "SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL=1",
        "SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL=1",
        "SGLANG_TPU_FP8_MOE_FP32_SCALES=1",
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD=512",
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1",
        "SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER=1",
        "SGLANG_SPEC_TP_SYNC=rng",
        "SGLANG_SET_CPU_AFFINITY=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--device tpu",
        "--tp-size 8",
        "--ep-size 1",
        "--language-model-only",
        "--quantization fp8",
        "--kv-cache-dtype fp8_e4m3",
        "--attention-backend pallas_tpu_mla_sparse",
        "--page-size 128",
        "--context-length 10240",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--max-running-requests 1",
        "--max-total-tokens 20480",
        "--max-mamba-cache-size 1",
        "--mem-fraction-static 0.9",
        "--disable-radix-cache",
        "--disable-shared-experts-fusion",
        "--enable-attn-tp-input-scattered",
        "--enable-torch-compile",
        "--torch-compile-max-bs 1",
        "--tpu-compile-bs 1",
        "--tpu-compile-tokens 1024 2048 4096 8192",
        "--grammar-backend none",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        `--json-model-override-args '{"index_share_for_mtp_iteration": false, "text_config": {"index_share_for_mtp_iteration": false}}'`,
        "--gc-threshold 50000 100 100",
        "--watchdog-timeout 1800",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // Ascend NPU: final verification of the tutorial recipes is in progress.
    {
      match: {
        hw: "a3",
        strategy: "low-latency",
        quant: "w8a8"
      },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "HCCL_BUFFSIZE=1536"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3-Flash-w8a8\"",
        "--trust-remote-code",
        "--tp-size 16",
        "--quantization modelslim",
        "--attention-backend ascend",
        "--device npu",
        "--chunked-prefill-size 4096",
        "--max-prefill-tokens 8192",
        "--mem-fraction-static 0.78",
        "--page-size 128",
        "--served-model-name GLM-5.3-Flash",
        "--cuda-graph-bs-decode 16",
        "--max-running-requests 16",
        "--load-balance-method round_robin",
        "--enable-multimodal",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--speculative-draft-model-path \"/models/GLM-5.3-Flash-w8a8\"",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--watchdog-timeout 9000",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },
    {
      match: {
        hw: "a5",
        strategy: "low-latency",
        quant: "w8a8-mxfp8"
      },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_NPU_PROFILING=0",
        "ASCEND_LAUNCH_BLOCKING=1",
        "HCCL_BUFFSIZE=256"
      ],
      flags: [
        "--model-path \"/models/GLM-5.3-Flash-w8a8-mxfp8\"",
        "--trust-remote-code",
        "--tp-size 8",
        "--quantization modelslim",
        "--attention-backend ascend",
        "--device npu",
        "--chunked-prefill-size 8192",
        "--max-prefill-tokens 8192",
        "--mem-fraction-static 0.83",
        "--page-size 64",
        "--served-model-name GLM-5.3-Flash",
        "--load-format auto",
        "--max-running-requests 16",
        "--pre-warm-nccl",
        "--speculative-draft-model-path \"/models/GLM-5.3-Flash-w8a8-mxfp8\"",
        "--speculative-draft-kv-cache-dtype bf16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 4",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 5",
        "--watchdog-timeout 1200",
        "--cuda-graph-bs-decode 1 8 64",
        "--host {{HOST_IP}}",
        "--port {{PORT}}"
      ]
    },

  ],
};
