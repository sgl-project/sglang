// MiniMax-M3.1 cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.

export const config = (() => {
// Shown under every Blackwell command; details in the Configuration Tips section.
const BLACKWELL_NOTE =
  "First launch JIT-compiles the MSA kernels (needs `FMHA_SM100_ALLOW_JIT=1`) and expects CUDA toolkit >= 13.4; "
  + "see [Blackwell prerequisites](#blackwell-prerequisites).";

// Shown under the AMD High-Throughput command; details in the AMD tips.
const AMD_HICACHE_NOTE =
  "HiCache host pools take about 350 GB per rank (1.4 TB for this server) at the default ratio; "
  + "see [AMD MI350X / MI355X](#amd-tips).";

// DFlash2: RadixArk's trained draft, a separate checkpoint (DSpark's draft ships in
// the model repo). DFLASH needs an explicit draft path. Block verification accepts
// sampled draft proposals by their joint probability; for a greedy draft it is the
// same as token-wise verification and the server logs that it has no effect.
const DFLASH2_FLAGS = [
  "--speculative-algorithm DFLASH",
  "--speculative-draft-model-path RadixArk/Minimax-M3.1-Flash-Dflash2",
  "--speculative-use-block-verification",
];
// M3.1 NVFP4 DFLASH requires Blackwell and colocated serving, and DFLASH rejects
// DP attention off NPU, so the B300 / B200 High-Throughput recipes
// (--attn-dp-size 2) cannot switch to it.
const BLACKWELL_HW = ["b300", "gb300", "b200"];
const DFLASH2_NOTE =
  "DFlash2 loads its draft from RadixArk/Minimax-M3.1-Flash-Dflash2 and runs in colocated serving only "
  + "(no PD disaggregation).";

// B300 Low-Latency: the MSA-V1 MonoKernel serves single-request DSpark steps
// The plugin declares sm_100 and sm_103 at TP4 only;
// only B300 has been measured, so GB300 / B200 get no cell yet.
// SGLANG_FORCE_COARSE_WAR_BARRIER=1 enables canonical
// MegaAttention placement (~3-4% faster), which traps if another process holds SMs.
// Measured on 4x B300 at batch size 1, acceptance fixed at 5: 1028 tok/s at 1K, 919 at
// 256K, 723 at 1M; GSM8K 0.97 with real acceptance.
const MONOKERNEL_NOTE =
  "The MonoKernel serves batch-size-1 steps only (larger batches take the standard path), and canonical "
  + "MegaAttention placement needs the GPUs to itself; see [MonoKernel](#monokernel).";

const cfg = {
  modelName: "MiniMax-M3.1-Flash",

  supportedHardware: ["b300", "gb300", "b200", "h200", "mi355x", "mi350x"],

  variants: [
    { id: "flash", label: "Flash" },
  ],
  quantizations: [
    { id: "nvfp4", label: "NVFP4" },
    // AMD's Quark MXFP4 export of the same weights, validated on MI355X; MI350X
    // (same gfx950) runs the same command.
    { id: "mxfp4", label: "MXFP4",
      disabled: (s) => !["mi355x", "mi350x"].includes(s.hw),
      disableReason: "The Quark MXFP4 checkpoint is for AMD MI350X / MI355X only." },
  ],
  // Low-Latency: AMD, plus the B300 / GB300 MonoKernel recipe. Balanced + High-Throughput elsewhere.
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "balanced",        label: "Balanced"        },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  // Orthogonal to the cell grid: the picked option layers onto whichever cell is
  // showing. Blackwell and AMD cells already carry --speculative-algorithm DSPARK
  // with block verification, so DSpark adds nothing; DFlash2 swaps both flags in
  // place and adds its draft flag. H200 runs without speculation and hides the row.
  overlayDims: [
    {
      id: "spec",
      title: "Speculative Decoding",
      default: "dspark",
      showWhen: (s) => s.hw !== "h200",
      options: [
        { id: "dspark", label: "DSpark" },
        { id: "dflash2", label: "DFlash2",
          // The AMD cells also carry an fp8 DSpark draft KV dtype; DFLASH sizes its
          // own draft KV, so that flag goes too.
          stripPrefixes: ["--speculative-algorithm", "--speculative-use-block-verification", "--speculative-draft-kv-cache-dtype"],
          flags: DFLASH2_FLAGS,
          hints: [DFLASH2_NOTE],
          disabled: (s) => !BLACKWELL_HW.includes(s.hw) || cellAttnDp(s) > 1 || cellMonokernel(s),
          disableReason: (s) => !BLACKWELL_HW.includes(s.hw)
            ? "M3.1 NVFP4 DFlash2 requires NVIDIA Blackwell (B300 / GB300 / B200)."
            : cellMonokernel(s)
              ? "The MonoKernel runs the DSpark draft and verify steps; DFlash2 is not supported with --monokernel-backend msa."
              : "DFlash2 does not support DP attention, and this recipe runs --attn-dp-size "
                + cellAttnDp(s) + ". Pick Balanced, or GB300 High-Throughput." },
      ],
    },
  ],

  modelNames: {
    "flash|nvfp4": "MiniMaxAI/MiniMax-M3.1-Flash",
    "mi355x|flash|mxfp4": "amd/MiniMax-M3.1-Flash-Quark-MXFP4",
    "mi350x|flash|mxfp4": "amd/MiniMax-M3.1-Flash-Quark-MXFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost"      },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"          },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  dockerImages: {
    b300:  "lmsysorg/sglang:dev-minimax-m31-flash",
    gb300: "lmsysorg/sglang:dev-minimax-m31-flash",
    b200:  "lmsysorg/sglang:dev-minimax-m31-flash",
    h200:  "lmsysorg/sglang:dev-minimax-m31-flash",
    mi355x: "lmsysorg/sglang:dev-minimax-m31-flash-mi35x ",
    mi350x: "lmsysorg/sglang:dev-minimax-m31-flash-mi35x ",
  },

  github: {
    cookbookModel: "MiniMaxAI/MiniMax-M3.1-Flash",
  },

  playgroundFeatures: {

    // ----- Attention Parallelism -----
    // No CP knob: prefill Context Parallel needs model-side integration that
    // the MiniMax-M3 family does not have (same as the MiniMax-M3 page).
    attention: {
      knobs: [
        { id: "tp",     label: "TP", values: [null, 1, 2, 4, 8] },
        { id: "dpAttn", label: "DP-Attention",
          values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- MoE Parallelism -----
    // On Blackwell the two measured expert paths are mutually exclusive:
    // TRT-LLM routed keeps the MoE in TP form (no --ep); MegaMoE is
    // expert-parallel only (EP == TP) and needs the per-rank token budget to
    // cover the prefill chunk. Hopper auto-resolves to Marlin W4A16 and
    // accepts neither.
    moe: {
      backend: {
        options: [
          { id: null, label: "Inherited" },
          { id: "flashinfer_trtllm_routed", label: "TRT-LLM routed (MoE TP)",
            flags: ["--moe-runner-backend flashinfer_trtllm_routed"],
            disable: { hw: ["h200", "mi355x", "mi350x"] },
            disableReason: "The TRT-LLM routed MoE needs the Blackwell NVFP4 kernels; Hopper serves the experts through Marlin W4A16, and AMD gfx950 requantizes them to MXFP4 for aiter." },
          { id: "megamoe", label: "MegaMoE (EP = TP)",
            flags: ["--moe-a2a-backend megamoe"],
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768"],
            requiresHw: ["b200", "b300", "gb200", "gb300"] },
        ],
      },
      ep: { label: "EP", values: [null, 4, 8] },
    },

    // ----- Parsers -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser minimax-m3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser minimax-m3" },
      ],
    },

    // ----- Speculative Decoding -----
    // The checkpoint ships the DSpark draft in its own dspark/ subfolder; the server
    // finds it there and defaults --speculative-draft-model-path to that folder
    // (local path or Hub snapshot), so no draft flag is needed.
    // DSpark: Blackwell or AMD gfx950; Hopper is rejected. DFlash2: Blackwell only,
    // no DP attention, colocated serving only (the note says so; this card is
    // applied before the PD card, so it cannot read the PD role).
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "dspark",  label: "DSpark",
          flags: ["--speculative-algorithm DSPARK", "--speculative-use-block-verification"],
          disable: { hw: ["h200"] },
          disableReason: "M3.1 NVFP4 DSpark requires Blackwell; the server rejects it on Hopper at argument resolution." },
        { id: "dflash2", label: "DFlash2",
          flags: DFLASH2_FLAGS,
          note: DFLASH2_NOTE,
          disable: [
            { when: { hw: ["h200", "mi355x", "mi350x"] },
              reason: "M3.1 NVFP4 DFlash2 requires NVIDIA Blackwell (B300 / GB300 / B200)." },
            { when: { dpAttnOn: [true] },
              reason: "DFlash2 does not support DP attention. Turn DP-Attention off in the Attention card above to use DFlash2." },
          ] },
      ],
    },

    // ----- PD Disaggregation -----
    // Correctness validated as 1P1D with TP4 on each side on 8x B300 over
    // Mooncake RDMA, with DSpark and HiCache on. On B300 both roles switch to
    // the recommended DP8 / TP1 / EP8 layout (section 3.5, not benchmarked yet):
    // EP needs MegaMoE, so the roles drop the TRT-LLM routed runner. Other
    // hardware keeps the base cell's parallelism. Prefill and decode must run
    // the same attention TP, and PP = 1.
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role",
          when: { hw: ["b300"] },
          flags: [
            "--tp 8",
            "--attn-dp-size 8",
            "--enable-dp-lm-head",
            "--ep 8",
            "--moe-a2a-backend megamoe",
          ],
          stripFlags: ["--moe-runner-backend"],
          env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768"] },
        { id: "decode",  label: "Decode role",
          when: { hw: ["b300"] },
          flags: [
            "--tp 8",
            "--attn-dp-size 8",
            "--enable-dp-lm-head",
            "--ep 8",
            "--moe-a2a-backend megamoe",
          ],
          stripFlags: ["--moe-runner-backend"],
          env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768"] },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake",
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb200", "gb300"] } },
        { id: "nixl",     label: "NiXL" },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --policy round_robin \\
  --host 0.0.0.0 --port {{ROUTER_PORT}}`,
      },
    },

    // ----- Hierarchical KV Cache -----
    // GPU -> CPU KV offload (L2), with an optional Mooncake storage tier (L3).
    // The NVFP4 host pools carry the packed KV, its block scales, the index K
    // and the draft KV in page rows. L3 is validated with Mooncake on GB300 and B300,
    // with and without DSpark / DFlash, so it is the only storage backend
    // exposed here, and it stays hidden on Hopper and AMD, where it has not
    // been run. The long-prefix capacity rounds ran the TRT-LLM + DSpark
    // recipe with --enable-hierarchical-cache --hicache-size 192 behind a
    // cache-aware router; multi-turn traffic needs prefix-affinity routing in
    // front of several replicas for the host cache to pay off.
    hicache: {
      backends: [
        { id: null,       label: "None (L2 only)" },
        { id: "mooncake", label: "Mooncake",
          hide: { hw: ["h200", "mi350x", "mi355x"] } },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },
  },

  // ==== Cells ====
  cells: [
    // ---- B300 (sm_103): every recipe measured on 8x B300 ----
    {
      match: { hw: "b300", variant: "flash", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      warn: BLACKWELL_NOTE,
      env: ["FMHA_SM100_ALLOW_JIT=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: MONOKERNEL_NOTE,
      env: ["FMHA_SM100_ALLOW_JIT=1", "ALLOW_JIT=1", "SGLANG_FORCE_COARSE_WAR_BARRIER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 8",
        "--speculative-dspark-block-size 7",
        "--monokernel-backend msa",
        "--disable-attn-tp-gather",
        "--max-running-requests 8",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "flash", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: BLACKWELL_NOTE,
      env: [
        "FMHA_SM100_ALLOW_JIT=1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 8",
        "--attn-dp-size 2",
        "--enable-dp-lm-head",
        "--ep 8",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--chunked-prefill-size 65536",
        "--max-running-requests 256",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---- GB300 (sm_103, 4 GPUs per node): same 4-GPU recipes; accuracy
    // validated on 4x GB300, throughput not yet measured there ----
    {
      match: { hw: "gb300", variant: "flash", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      warn: BLACKWELL_NOTE,
      env: ["FMHA_SM100_ALLOW_JIT=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: MONOKERNEL_NOTE,
      env: ["FMHA_SM100_ALLOW_JIT=1", "ALLOW_JIT=1", "SGLANG_FORCE_COARSE_WAR_BARRIER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--page-size 128",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 8",
        "--speculative-dspark-block-size 7",
        "--monokernel-backend msa",
        "--disable-attn-tp-gather",
        "--max-running-requests 8",
        "--mem-fraction-static 0.8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "flash", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: BLACKWELL_NOTE,
      env: [
        "FMHA_SM100_ALLOW_JIT=1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--ep 4",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---- B200 (sm_100): inferred-supported from B300 (same SM100 family and
    // code gates). 192 GB per GPU against B300's 288 GB. Both recipes were run
    // on B200. Balanced runs as written (AIME26 avg@16 91.88%, MMMU-Pro
    // 78.09%). High-throughput (AIME26 91.67%, MMMU-Pro 76.88%) runs out of
    // GPU memory with B300's settings, first capturing the 32k-token prefill
    // CUDA graphs, then in multimodal image preprocessing on GPU 0, hence
    // --cuda-graph-max-bs-prefill 16384 and --mem-fraction-static 0.75. ----
    {
      match: { hw: "b200", variant: "flash", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verificationStatus: "in-progress",
      warn: BLACKWELL_NOTE,
      env: ["FMHA_SM100_ALLOW_JIT=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--moe-runner-backend flashinfer_trtllm_routed",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "flash", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verificationStatus: "in-progress",
      warn: BLACKWELL_NOTE,
      env: [
        "FMHA_SM100_ALLOW_JIT=1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 8",
        "--attn-dp-size 2",
        "--enable-dp-lm-head",
        "--ep 8",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--chunked-prefill-size 65536",
        "--max-running-requests 256",
        "--cuda-graph-max-bs-decode 128",
        "--cuda-graph-max-bs-prefill 16384",
        "--mem-fraction-static 0.75",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---- H200 (sm_90): portable stack, no speculative decoding. Both recipes
    // measured on 8x H200 at 128k input / 1k output: tp8 leads up to ~8
    // concurrent requests (174.6 tok/s at batch 1), tp8 + attention-DP 2 leads
    // from 16 up (1,283 vs 1,203 tok/s at 16; 2,543 at 58) and holds twice the
    // KV (about 58 concurrent 128k requests). ----
    {
      match: { hw: "h200", variant: "flash", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 8",
        "--context-length 140000",
        "--cuda-graph-backend-prefill breakable",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 64",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "flash", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 8",
        "--attn-dp-size 2",
        "--context-length 140000",
        "--cuda-graph-backend-prefill breakable",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 64",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ---- AMD MI350X / MI355X (gfx950), ROCm 7.2 / Triton 3.7: one TP4 server per four
    // GPUs, since M3.1 NVFP4 on ROCm refuses TP above 4. SGLANG_USE_AITER=1 is required
    // (code default off; the ROCm image sets it). The NVFP4 KV pool, the INT4/INT8 quick
    // all-reduce and aiter's tuned M3.1 tables are code defaults on gfx950
    // (sglang-minimax-collab amd-support-m31 6d0abea6ba). Low-Latency on MI350X with
    // MiniMax-M3.1-Flash: GSM8K 96.44, AIME26 avg@16 92.08%; MI355X verified on Flash,
    // NVFP4 and Quark MXFP4. ----
    {
      match: { hw: "mi350x", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--fp8-gemm-backend aiter",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--fp8-gemm-backend aiter",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "mxfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--fp8-gemm-backend aiter",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // MI350X (gfx950): same Quark MXFP4 command as MI355X.
    {
      match: { hw: "mi350x", variant: "flash", quant: "mxfp4", strategy: "low-latency", nodes: "single" },
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill breakable",
        "--fp8-gemm-backend aiter",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // High-Throughput: AgentX at 48-128 concurrent sessions on MI350X (3600 s per point,
    // 0.90-0.93x the B300 chart's throughput per GPU); this command on MiniMax-M3.1-Flash:
    // GSM8K 96.36, HiCache auto-sized to ratio 2 (325 GiB of host pools per rank). The fp8
    // draft KV fits 9% more tokens in the same pool; the Quark export keeps an fp8 main KV
    // and has not been measured with this recipe.
    {
      match: { hw: "mi350x", variant: "flash", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: AMD_HICACHE_NOTE,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--speculative-draft-kv-cache-dtype fp8_e4m3",
        "--cuda-graph-backend-prefill breakable",
        "--fp8-gemm-backend aiter",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--enable-hierarchical-cache",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "flash", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      warn: AMD_HICACHE_NOTE,
      env: ["SGLANG_USE_AITER=1"],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser minimax-m3",
        "--tool-call-parser minimax-m3",
        "--tp 4",
        "--speculative-algorithm DSPARK",
        "--speculative-use-block-verification",
        "--speculative-draft-kv-cache-dtype fp8_e4m3",
        "--cuda-graph-backend-prefill breakable",
        "--fp8-gemm-backend aiter",
        "--chunked-prefill-size 32768",
        "--max-running-requests 128",
        "--cuda-graph-max-bs-decode 128",
        "--mem-fraction-static 0.8",
        "--tokenizer-worker-num 8",
        "--enable-hierarchical-cache",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};

// Attention DP size of the cell the Deploy panel is showing (1 when none is set).
function cellAttnDp(s) {
  const cell = cfg.cells.find((c) => Object.entries(c.match).every(([k, v]) => s[k] === v));
  const flag = cell && cell.flags.find((f) => f.startsWith("--attn-dp-size "));
  return flag ? parseInt(flag.split(/\s+/)[1], 10) : 1;
}

// Whether the showing cell runs the MSA-V1 MonoKernel (--monokernel-backend msa).
function cellMonokernel(s) {
  const cell = cfg.cells.find((c) => Object.entries(c.match).every(([k, v]) => s[k] === v));
  return !!cell && cell.flags.some((f) => f.startsWith("--monokernel-backend "));
}

return cfg;
})();
