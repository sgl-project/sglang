export const config = {
  modelName: "Mistral Large 4",

  supportedHardware: ["b300", "gb300", "vera-rubin", "b200", "h200"],

  hardware: [
    { id: "vera-rubin", label: "Vera Rubin", vram: "288GB", vendor: "rubin" },
  ],

  variants: [
    { id: "default", label: "Default", subtitle: "1T / 52B active" },
  ],
  quantizations: [
    { id: "fp8", label: "FP8" },
    { id: "nvfp4", label: "NVFP4" },
  ],
  strategies: [
    { id: "low-latency", label: "Low Latency" },
    { id: "high-throughput", label: "High Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
    { id: "multi-2", label: "2 Nodes" },
  ],

  modelNames: {
    "default|fp8": "mistralai/Mistral-Large-4-1T-A52B",
    "default|nvfp4": "mistralai/Mistral-Large-4-1T-A52B-NVFP4",
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
  --seed 42 --warmup-requests 0 \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct:
`pip install "sgl-eval @ git+https://github.com/sgl-project/sgl-eval.git@93bc2d851a59e57d54fe755202756589a04d3d86"
# The Vera Rubin cards average seeds 1, 2 and 3; the B300 card reports seed 1.
for SEED in 1 2 3; do
  sgl-eval run gsm8k \\
    --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
    --model {{MODEL_NAME}} \\
    --temperature 1.0 --top-p 0.95 --max-tokens 8192 \\
    --reasoning-effort low \\
    --seed $SEED --num-threads 64 \\
    --out-dir gsm8k/seed$SEED
done`,
      gsm8k_100_pct:
`pip install "sgl-eval @ git+https://github.com/sgl-project/sgl-eval.git@93bc2d851a59e57d54fe755202756589a04d3d86"
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --temperature 1.0 --top-p 0.95 --max-tokens 8192 \\
  --reasoning-effort low \\
  --num-examples 100 --seed 1 --num-threads 64 \\
  --out-dir gsm8k-100`,
      mmlu_pct:
`pip install "sgl-eval @ git+https://github.com/sgl-project/sgl-eval.git@93bc2d851a59e57d54fe755202756589a04d3d86"
# The Vera Rubin cards average seeds 1, 2 and 3.
for SEED in 1 2 3; do
  sgl-eval run mmlu \\
    --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
    --model {{MODEL_NAME}} \\
    --temperature 1.0 --top-p 0.95 --max-tokens 8192 \\
    --reasoning-effort low \\
    --seed $SEED --num-threads 128 \\
    --out-dir mmlu/seed$SEED
done`,
      mmmu_pro_pct:
`pip install "sgl-eval @ git+https://github.com/sgl-project/sgl-eval.git@93bc2d851a59e57d54fe755202756589a04d3d86"
# Pin the MMMU-Pro standard (10 options) split, then evaluate offline against it.
export HF_HOME=$PWD/hf-mmmu-pro
hf download MMMU/MMMU_Pro --repo-type dataset \\
  --revision 0d7426df4ccb3d8704a992fe6850abee7c262127 \\
  --include "standard (10 options)/*" --include README.md
python3 -c "from datasets import load_dataset; load_dataset('MMMU/MMMU_Pro', 'standard (10 options)', split='test', revision='0d7426df4ccb3d8704a992fe6850abee7c262127')"
# The Vera Rubin cards average seeds 1, 2 and 3; the GB300 card reports seed 1.
for SEED in 1 2 3; do
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 sgl-eval run mmmu_pro \\
    --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
    --model {{MODEL_NAME}} \\
    --temperature 1.0 --top-p 0.95 --max-tokens 8192 \\
    --reasoning-effort low \\
    --seed $SEED --num-threads 64 \\
    --out-dir mmmu_pro/seed$SEED
done`,
    },
    numPromptsByConc: { 1: 64, 4: 64, 16: 64, 64: 64 },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
    ["gsm8k_100_pct", "GSM8K, first 100 questions", "%"],
    ["mmlu_pct", "MMLU", "%"],
    ["mmmu_pro_pct", "MMMU-Pro (standard, 10-option)", "%"],
    ["dspark_accept_len", "DSpark accepted tokens per 4-token verify step", ""],
  ],

  latencyPercentile: "P50",

  dockerImages: {
    b300: "lmsysorg/sglang:mistral-large-4",
    gb300: "lmsysorg/sglang:mistral-large-4",
    b200: "lmsysorg/sglang:mistral-large-4",
    h200: "lmsysorg/sglang:mistral-large-4",
    "vera-rubin": "lmsysorg/sglang:mistral-large-4",
  },

  github: {
    cookbookModel: "mistralai/Mistral-Large-4-1T-A52B",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "cp", label: "Prefill CP (interleave)",
          freeSize: true,
          values: [null, { value: 1, label: "Off" },
            { value: 2, disable: { hw: ["h200"], nodes: ["multi-2"] }, disableReason: "SGLang rejects prefill CP with pipeline parallelism, which the two-node H200 recipe uses." },
            { value: 4, disable: { hw: ["h200"], nodes: ["multi-2"] }, disableReason: "SGLang rejects prefill CP with pipeline parallelism, which the two-node H200 recipe uses." }] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 2, 4,
            { value: 8, disable: { hw: ["gb300"] },
              disableReason: "DP-Attention 8 needs 8 ranks; the GB300 recipe runs TP4." }],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 1,
        { value: 4, hide: { hw: ["b300", "b200", "h200"] } },
        { value: 8, hide: { hw: ["gb300"] } }] },
    },

    speculative: {
      options: [
        { id: "off",    label: "Off",    flags: [] },
        { id: "dspark", label: "DSpark", flags: ["--speculative-algorithm DSPARK"],
          disable: [{ when: { hw: ["h200"], nodes: ["multi-2"] },
            reason: "DSpark accepts --pp-size above 1 only on a PD prefill server, and the two-node H200 recipe uses pipeline parallelism." }] },
      ],
    },

    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [{ id: "mooncake", label: "Mooncake" }],
      notes: [{ mode: ["prefill", "decode"], when: { pdMode: ["prefill", "decode"] },
        text: "PD disaggregation runs with a peer-config fingerprint guard; a 1P1D setup (one prefill and one decode server) is validated end to end." }],
    },

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    hicache: {
      backends: [
        { id: null,   label: "Auto" },
        { id: "file", label: "File" },
      ],
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },

    flagSelects: [
      {
        id: "kvCacheDtype", title: "KV Cache Precision",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "fp8",  label: "Default — FP8 (E4M3) on B200, B300, GB300 and Vera Rubin; BF16 on H200", flags: [] },
          { id: "bf16", label: "BF16 — the KV precision of the Vera Rubin benchmark cards, about half the KV capacity",
            flags: ["--kv-cache-dtype bfloat16"], hide: { hw: ["h200"] } },
          { id: "fp8-hopper", label: "FP8 (E4M3) on H200 — selects the flashmla_kv DSA backends; not yet verified",
            flags: ["--kv-cache-dtype fp8_e4m3"], hide: { hw: ["b300", "gb300", "vera-rubin", "b200"] } },
        ],
      },
      {
        id: "hisparse", title: "HiSparse (host-offloaded sparse KV)",
        stripPrefixes: ["--mem-fraction-static", "--enable-hisparse", "--hisparse-config", "--disable-radix-cache"],
        showWhen: (base, rows) => !rows || !rows.dcp || rows.dcp === "off",
        options: [
          { id: "off", label: "Off", flags: ["--mem-fraction-static 0.85"] },
          { id: "ratio2", label: "On, host ratio 2 (DSpark requires static ragged verification)",
            flags: ["--mem-fraction-static 0.80", "--enable-hisparse", "--disable-radix-cache"],
            disable: [
              { when: { cpOn: [true] }, reason: "SGLang rejects HiSparse together with prefill CP — pick one." },
              { when: { hicacheOn: [true] }, reason: "HiSparse needs --disable-radix-cache, which HiCache builds on — turn HiCache off." },
            ] },
          { id: "ratio3", label: "On, host ratio 3 (about 148 GB/GPU; DSpark requires static ragged verification)",
            flags: ["--mem-fraction-static 0.80", "--enable-hisparse", "--disable-radix-cache",
              `--hisparse-config '{"host_to_device_ratio": 3}'`],
            disable: [
              { when: { cpOn: [true] }, reason: "SGLang rejects HiSparse together with prefill CP — pick one." },
              { when: { hicacheOn: [true] }, reason: "HiSparse needs --disable-radix-cache, which HiCache builds on — turn HiCache off." },
            ] },
        ],
      },
      {
        id: "dcp", title: "Decode Context Parallel (DCP)",
        stripPrefixes: ["--dcp-size"],
        showWhen: (base, rows) => !rows || !rows.hisparse || rows.hisparse === "off",
        options: [
          { id: "off", label: "Off", flags: [] },
          { id: "dcp2", label: "DCP 2", flags: ["--dcp-size 2"],
            disable: [
              { when: { hw: ["h200"], nodes: ["multi-2"] }, reason: "SGLang rejects --dcp-size with pipeline parallelism, which the two-node H200 recipe uses." },
              { when: { cpOn: [true] }, reason: "SGLang rejects --dcp-size together with prefill CP — pick one." },
              { when: { hicacheOn: [true] }, reason: "SGLang rejects --dcp-size together with HiCache for this model — turn HiCache off." },
              { when: { pdMode: ["prefill", "decode"] }, reason: "SGLang rejects --dcp-size together with PD disaggregation." },
            ] },
          { id: "dcp4", label: "DCP 4", flags: ["--dcp-size 4"],
            disable: [
              { when: { hw: ["h200"], nodes: ["multi-2"] }, reason: "SGLang rejects --dcp-size with pipeline parallelism, which the two-node H200 recipe uses." },
              { when: { cpOn: [true] }, reason: "SGLang rejects --dcp-size together with prefill CP — pick one." },
              { when: { hicacheOn: [true] }, reason: "SGLang rejects --dcp-size together with HiCache for this model — turn HiCache off." },
              { when: { pdMode: ["prefill", "decode"] }, reason: "SGLang rejects --dcp-size together with PD disaggregation." },
            ] },
        ],
      },
    ],
  },

  cells: [
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
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
        "--tp-size 4",
        "--mem-fraction-static 0.85",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
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
        "--tp-size 4",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--max-running-requests 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "vera-rubin", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      env: [
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1",
        "CUTE_DSL_ARCH=sm_107a",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "vera-rubin", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "multi-2" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. It is the two-tray Vera Rubin FP8 recipe with DSpark.",
      env: [
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1",
        "CUTE_DSL_ARCH=sm_107a",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "vera-rubin", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      redirect: true,
      warn: "Not offered on one Vera Rubin tray: at TP4 the FP8 weights need about 257 GiB per GPU, more than --mem-fraction-static 0.85 allows. Pick 2 Nodes for TP8 across two trays, or the NVFP4 checkpoint on one tray.",
      env: [],
      flags: [],
    },
    {
      match: { hw: "vera-rubin", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1",
        "CUTE_DSL_ARCH=sm_107a",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--mem-fraction-static 0.85",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "vera-rubin", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. It is the one-tray Vera Rubin NVFP4 recipe with the 4× GB300 DSpark settings.",
      env: [
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1",
        "CUTE_DSL_ARCH=sm_107a",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 4",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--max-running-requests 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. It is the 4× GB300 NVFP4 recipe at TP8. The weights take about 73 GB of each GPU at TP8, half the 145.8 GB measured at TP4, which leaves a large KV pool. The TP4 command on 4× B300 has not been validated here.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. It is the 4× GB300 NVFP4 DSpark recipe at TP8. The weights take about 73 GB of each GPU at TP8, half the 145.8 GB measured at TP4, which leaves a large KV pool.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--max-running-requests 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      redirect: true,
      warn: "Not offered on GB300: at TP4 the FP8 weights need about 257 GiB per GPU (the routed experts alone about 240 GiB), more than the 236 GiB that --mem-fraction-static 0.85 allows on a 278 GiB GB300. NVFP4 at TP4 is the GB300 recipe for this model.",
      env: [],
      flags: [],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. It is the 8× B300 FP8 recipe. The weights take about 129 GiB of each GPU, so the KV pool is much smaller than on B300.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. It is the 8× B300 FP8 DSpark recipe. The weights take about 129 GiB of each GPU, so the KV pool is much smaller than on B300.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. It is the 4× GB300 NVFP4 recipe at TP8. The weights take about 73 GB of each GPU at TP8, half the 145.8 GB measured at TP4, which leaves a large KV pool.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. It is the 4× GB300 NVFP4 DSpark recipe at TP8. The weights take about 73 GB of each GPU at TP8, half the 145.8 GB measured at TP4, which leaves a large KV pool.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--speculative-algorithm DSPARK",
        "--max-running-requests 64",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      redirect: true,
      warn: "Not offered on one H200 node: the FP8 weights take about 129 GiB of each 140 GiB GPU at TP8, which leaves almost nothing for the KV cache. Pick 2 Nodes.",
      env: [],
      flags: [],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "multi-2" },
      verified: false,
      warn: "Not yet verified: this command is derived from the verified recipes and has not been run on this hardware. Two 8× H200 nodes run TP8 within each node and two pipeline stages, which leaves most of each GPU for the KV cache. SGLANG_PP_LAYER_PARTITION=35,37 keeps layer 35 and the layers that read its KV cache on one stage, the split validated with PP on B300. On Hopper the KV cache defaults to BF16.",
      env: ["SGLANG_PP_LAYER_PARTITION=35,37"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--mem-fraction-static 0.85",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      redirect: true,
      warn: "No DSpark command for H200: one node leaves no memory for the draft model and its KV cache, and the two-node layout uses pipeline parallelism, which DSpark accepts only on a PD prefill server.",
      env: [],
      flags: [],
    },
    {
      match: { hw: "h200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      redirect: true,
      warn: "Not offered on H200: the NVFP4 checkpoint targets Blackwell GPUs. Use the FP8 checkpoint on 2 Nodes.",
      env: [],
      flags: [],
    },
  ],
};
