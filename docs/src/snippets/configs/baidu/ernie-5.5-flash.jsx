// ERNIE-5.5-Flash cookbook config, consumed by _deployment.jsx and _playground.jsx.
// The ERNIE-5.5-Flash-Preview chat checkpoints: FP4 (MXFP4 routed experts), FP8 (UE8M0
// block scales) and BF16. Each cell uses the fewest GPUs that leave room for the KV cache,
// with SGLang's defaults otherwise.

export const config = {
  modelName: "ERNIE-5.5-Flash",

  supportedHardware: ["b300", "gb300", "b200", "gb200", "h100"],

  // No variant row: the page serves the Preview checkpoints, so the HF slug is keyed on quant.
  matchDims: [
    { id: "quant", title: "Quantization", options: [
      { id: "fp4",  label: "FP4"  },
      { id: "fp8",  label: "FP8"  },
      { id: "bf16", label: "BF16" },
    ] },
    // Low-latency drafts with the bundled DSpark draft; high-throughput runs attention DP
    // without a draft.
    { id: "strategy", title: "Strategy", options: [
      { id: "low-latency",     label: "Low-Latency"     },
      { id: "high-throughput", label: "High-Throughput" },
    ] },
    { id: "nodes", title: "Nodes", options: [
      { id: "single",  label: "Single Node" },
      { id: "multi-2", label: "Multi-Nodes" },
    ] },
  ],

  modelNames: {
    // FP4 is the default Preview checkpoint (MXFP4 routed experts, SM100 only).
    fp4:  "baidu/ERNIE-5.5-Flash-Preview",
    fp8:  "baidu/ERNIE-5.5-Flash-Preview-FP8",
    bf16: "baidu/ERNIE-5.5-Flash-Preview-BF16",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"         },
    PORT:      { target: "command", label: "Bind port",         default: "30000"           },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost"       },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"           },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role": "user", "content": "Introduce yourself briefly."}], "max_tokens": 512 }'`,

  dockerImages: {
    b300:  "lmsysorg/sglang:dev-ernie-55-flash-pr51236-fe204897",
    gb300: "lmsysorg/sglang:dev-ernie-55-flash-pr51236-fe204897",
    b200:  "lmsysorg/sglang:dev-ernie-55-flash-pr51236-fe204897",
    gb200: "lmsysorg/sglang:dev-ernie-55-flash-pr51236-fe204897",
    h100: "lmsysorg/sglang:dev-ernie-55-flash-pr51236-fe204897",
  },

  github: {
    cookbookModel: "baidu/ERNIE-5.5-Flash-Preview",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 2, 4, 8] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          // DeepEP needs --ep 4 or more: 512 experts over 2 ranks is past its per-rank limit.
          { id: "deepep", label: "DeepEP", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
      ep: { label: "EP", values: [null, 4, 8] },
    },

    // For ERNIE-5.5-Flash; the Base checkpoints are completion-only.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser ernie55" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser ernie55" },
      ],
    },

    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off" },
        // ECHO: the backbone's final layer is the draft, reused once per step.
        // Mean accept length on GSM8K at batch size 1: 1.93 / 2.28 / 2.36 for 1 / 2 / 3 steps.
        { id: "echo-1", label: "ECHO, 1 step",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 1",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 2"] },
        { id: "echo-2", label: "ECHO, 2 steps",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 2",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 3"] },
        { id: "echo-3", label: "ECHO, 3 steps",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
        // The draft ships inside the ERNIE-5.5-Flash checkpoint. DSpark under attention DP
        // requires the DP LM head; without attention DP the flag is reset to off.
        { id: "dspark", label: "DSpark",
          flags: ["--speculative-algorithm DSPARK", "--enable-dp-lm-head"],
          note: "The DSpark draft ships with ERNIE-5.5-Flash: set --model-path baidu/ERNIE-5.5-Flash; the Base checkpoints carry no draft." },
      ],
    },

    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [{ id: "mooncake", label: "Mooncake" }],
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

    hicache: {
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
      ],
    },
  },

  // GB300 / GB200 run the B300 / B200 recipes; a 4-GPU tray fits all
  // but BF16 on GB200, which spans two trays. Inside this IIFE because Mintlify's snippet
  // compiler drops top-level module code.
  cells: (() => {
    const cells = [
    // FP8: 2 x B300, 4 x B200, 8 x H100.
    {
      match: { hw: "b300", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 2",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--speculative-algorithm DSPARK",
        "--enable-dp-lm-head",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 2",
        "--attn-dp-size 2",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verificationStatus: "unverified",
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 4",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--speculative-algorithm DSPARK",
        "--enable-dp-lm-head",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 4",
        "--attn-dp-size 4",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verificationStatus: "unverified",
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 8",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--speculative-algorithm DSPARK",
        "--enable-dp-lm-head",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 8",
        "--attn-dp-size 8",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // FP4 (Preview chat checkpoint): MXFP4 routed experts + FP8 activations (w4a8),
    // SM100 only, ~155 GB. The experts run w4a8 through DeepGEMM MegaMoE, which needs
    // expert parallelism, so FP4 uses the FP8 GPU counts. Without attention DP each rank
    // prefills the whole chunk, so low-latency caps it at MegaMoE's default 8192 tokens per rank.
    {
      match: { hw: "b300", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 2",
        "--ep 2",
        "--moe-runner-backend deep_gemm",
        "--moe-a2a-backend megamoe",
        "--enable-fp4-indexer",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.9",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--speculative-algorithm DSPARK",
        "--enable-dp-lm-head",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 2",
        "--attn-dp-size 2",
        "--ep 2",
        "--moe-runner-backend deep_gemm",
        "--moe-a2a-backend megamoe",
        "--enable-fp4-indexer",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "fp4", strategy: "low-latency", nodes: "single" },
      verificationStatus: "unverified",
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 4",
        "--ep 4",
        "--moe-runner-backend deep_gemm",
        "--moe-a2a-backend megamoe",
        "--enable-fp4-indexer",
        "--chunked-prefill-size 8192",
        "--mem-fraction-static 0.9",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--speculative-algorithm DSPARK",
        "--enable-dp-lm-head",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "fp4", strategy: "high-throughput", nodes: "single" },
      verificationStatus: "unverified",
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 4",
        "--attn-dp-size 4",
        "--ep 4",
        "--moe-runner-backend deep_gemm",
        "--moe-a2a-backend megamoe",
        "--enable-fp4-indexer",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // BF16: 4 x B300, 8 x B200; H100 needs more than one node.
    {
      match: { hw: "b300", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 4",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--speculative-algorithm DSPARK",
        "--enable-dp-lm-head",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 4",
        "--attn-dp-size 4",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verificationStatus: "unverified",
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 8",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--speculative-algorithm DSPARK",
        "--enable-dp-lm-head",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verificationStatus: "unverified",
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 8",
        "--attn-dp-size 8",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // BF16 on H100: 2 nodes x 8 GPUs (~35 GB of weights per GPU). The engine adds
    // --nnodes / --node-rank / --dist-init-addr. Attention TP must divide the 8 output
    // groups, so even low-latency keeps attention DP 2 (attention TP 8).
    {
      match: { hw: "h100", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verificationStatus: "unverified",
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 16",
        "--attn-dp-size 2",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--speculative-algorithm DSPARK",
        "--enable-dp-lm-head",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verificationStatus: "unverified",
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp 16",
        "--attn-dp-size 16",
        "--context-length 131072",
        "--max-running-requests 256",
        "--reasoning-parser ernie55",
        "--tool-call-parser ernie55",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    ];
    const grace = cells
      .filter((c) => c.match.hw === "b300" || c.match.hw === "b200")
      .map(({ verified, verificationStatus, ...c }) => {
        const hw = c.match.hw === "b300" ? "gb300" : "gb200";
        const nodes = hw === "gb200" && c.match.quant === "bf16" ? "multi-2" : c.match.nodes;
        // Only the FP8 high-throughput recipe carries over unchanged from the verified Base runs.
        const status = c.match.quant === "fp8" && c.match.strategy === "high-throughput"
          ? { verified: true } : { verificationStatus: "unverified" };
        return { ...c, match: { ...c.match, hw, nodes }, ...status };
      });
    return [...cells, ...grace];
  })(),
};
