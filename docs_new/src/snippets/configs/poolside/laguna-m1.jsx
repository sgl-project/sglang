// Laguna-M.1 (poolside) — config-driven cookbook page.
// Consumed by the shared _deployment.jsx + _playground.jsx engines (no model code there).
//
// Build: M.1 needs SGLang PR #28400 (softplus per-element output gating, MERGED) AND PR #28604
// (global-attention SWA fix — M.1 is sliding_window=0 / all-global; without it M.1 crashes ~1s
// into any concurrent batch with AssertionError: ... swa_lock_ref=0). Both are merged on main
// (verified on a 3f668733 build). The shipped recipe carries NO SWA workaround flag, but the pinned
// build MUST contain BOTH; the #28400-merge wheel 0.5.14.dev20260618+g343aeeef39 is #28400-ONLY
// and crashes under load. Pin dockerImages + benchmarks.sglang_version to a build at a commit
// ≥ #28604. See /sgl-workspace/laguna-m1-day0-checklist.md (step 2) + laguna-m1-results.md.
//
// --trust-remote-code is required: M.1 ships custom config code on the Hub (the transformers-native
// `laguna` config is incompatible). Carried on every cell.
//
// Hardware: H200 (Hopper) + B200/B300/GB200/GB300 (Blackwell).
//   - BF16 runs everywhere.
//   - FP8 runs everywhere. On Blackwell (sm_100) the compressed-tensors block-FP8 weight scales
//     aren't UE8M0-packed, so the default DeepGEMM path produces garbage → the Blackwell FP8 cells
//     add `--fp8-gemm-backend triton` (correct, ~19% slower than DeepGEMM). Temporary until the
//     ue8m0-requant fix (PR #28662) lands; H200 FP8 (Hopper) is unaffected and needs no flag.
//   - NVFP4 is Blackwell-only.
// Cells: H200×{BF16,FP8}; each Blackwell×{BF16,FP8,NVFP4}.
// TP: 8-GPU HGX nodes (H200/B200/B300) → --tp 8 (the maintainer's baseline); GB200/GB300
// (Grace-Blackwell, typically 4-GPU single node) → --tp 4. Adjust --tp to your node size.
//
// Strategy: a SINGLE "Balanced" operating point (maintainer decision — no LL/HT split; an
// earlier TP=8+DP-Attention "high-throughput" idea was dropped: DP-Attention is ~15% SLOWER
// on this GQA model, see laguna-m1-results.md).

export const config = {
  modelName: "Laguna-M.1",

  supportedHardware: ["h200", "b200", "b300", "gb200", "gb300"],

  variants: [
    { id: "default", label: "Default" },
  ],

  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
  ],

  // Single balanced operating point (maintainer decision — no LL/HT split).
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],

  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16":  "poolside/Laguna-M.1",
    "default|fp8":   "poolside/Laguna-M.1-FP8",
    "default|nvfp4": "poolside/Laguna-M.1-NVFP4",
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
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    // GSM8K is the required accuracy sanity on every verified cell (cookbook_guide §3), via sgl-eval.
    // (AIME 25 to be added back once truncation-free numbers are measured.)
    accuracy: {
      gsm8k_pct:
`# pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 128`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 128: 256, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // Hardware-independent accuracy default (null = no variant-wide default; real numbers are per-cell
  // in laguna-m1-benchmarks.jsx).
  defaultAccuracy: {
    default: { gsm8k_pct: null },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  // lmsysorg/sglang:latest (cu13) covers H200 + all Blackwell and carries the
  // Laguna-M.1 build (PR #28400 + #28604 + #28649, incl. the FP8 g_proj fix).
  dockerImages: {
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    gb200: "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
  },

  github: {
    cookbookModel: "poolside/Laguna-M.1",
  },

  playgroundFeatures: {

    // M.1 is global-attention (no SWA); expose TP + DP-Attention here. No CP: the default
    // trtllm_mha backend has no CP-aware KV-store (crashes), and the engine's built-in attention CP
    // knob emits prefill-CP flags (--enable-prefill-cp / --cp-strategy / --attn-cp-size) that apply
    // to DeepSeek-family models, not M.1. (CP works only via the fa3 backend, which is Hopper
    // SM90 — left out here.)
    // DP-Attention: VERIFIED functionally correct on 8×B200 BF16 (GSM8K 0.94, identical to the TP
    // baseline) but ~15–28% slower on this GQA model (8 KV heads). Playground experiment only —
    // deliberately NOT in the shipped Balanced recipe.
    attention: {
      knobs: [
        { id: "tp",     label: "TP",           values: [null, 1, 2, 4, 8] },
        { id: "dpAttn", label: "DP-Attention", values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // 256-expert top-16 MoE — EP degree only.
    // EP: VERIFIED on 8×B200 BF16 (--ep-size 8, GSM8K 0.94, identical to the TP baseline).
    // DeepEP intentionally NOT exposed — it does not work on M.1: top-16 routing exceeds DeepEP's
    // low-latency internode kernel cap of 11 (internode_ll.cu kNumMaxTopK=11) → assert at decode
    // CUDA-graph capture, and `--deepep-mode normal` is NotImplemented for unquantized weights. Use EP.
    moe: {
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    // Reasoning + tool-call parsers (poolside_v1, same family as Laguna-XS.2). ALSO baked into
    // every Deploy cell below (the maintainer's baseline carries them).
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser poolside_v1" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser poolside_v1" },
      ],
    },

    // HiCache (hierarchical KV cache). VERIFIED on 8×B200 BF16: enabling the host L2 tier on a
    // zipfian shared-prefix workload cut mean TTFT ~36% (median ~43%) and lifted throughput ~19% vs
    // GPU-only, with ~1.14M tokens served from the host tier (TPOT unchanged — the win is on prefill
    // / prefix reuse). Biggest gains on reuse-heavy traffic: shared system prompts, multi-turn
    // agentic coding, repeated long contexts. "Enable" emits --enable-hierarchical-cache (+host L2);
    // Write policy is optional. (L3 storage backends exist but were not validated, so none exposed.)
    hicache: {
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },

    // Prefill-Decode disaggregation (§3.3). M.1 is standard-KV (global attention, no sparse
    // index buffer), so it disaggregates with just the --disaggregation-* flags — no model-specific
    // backend pinning. Verified on 2×8×H200 (TP8+TP8, BF16) over InfiniBand.
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        // mooncake (recommended): honors --disaggregation-ib-device, no transfer cold-start.
        // The NCCL/MNNVL env is only needed on NVLink-multinode Grace-Blackwell (GB200/GB300).
        { id: "mooncake", label: "Mooncake",
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb200", "gb300"] } },
        // NiXL ignores --disaggregation-ib-device; its UCX backend needs the NIC pinned via
        // UCX_NET_DEVICES or every KV transfer hangs to the 300s timeout (§3.3). Baked in here
        // for the IB-based HGX platforms; also expect a ~38s one-time UCX cold-start.
        { id: "nixl",     label: "NiXL",
          env: ["UCX_NET_DEVICES=mlx5_0:1"],
          envWhen: { hw: ["h200", "b200", "b300"] } },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --policy round_robin \\
  --host 0.0.0.0 --port {{ROUTER_PORT}}`,
      },
    },
  },

  // One Balanced cell per valid (hw × quant): H200×{BF16,FP8}; each Blackwell×{BF16,FP8,NVFP4}.
  // Blackwell FP8 cells add `--fp8-gemm-backend triton` (DeepGEMM UE8M0 workaround, pending #28662);
  // H200 FP8 needs no such flag. Baseline recipe (parsers poolside_v1 + --trust-remote-code) on every cell.
  // TP: H200/B200/B300 = --tp 8; GB200/GB300 = --tp 4 (4-GPU single node).
  // verified:true = ran that exact command on that hardware and it served correctly + passed a
  // GSM8K-class eval. Absent verified = yellow/unverified badge.
  cells: [
    // ===== NVIDIA Hopper (H200) — BF16 / FP8 (NVFP4 is Blackwell-only) =====
    {
      // VERIFIED on 8xH200 (BF16, tp8): GSM8K 93.02% + perf (laguna-m1 H200 results).
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED on 8xH200 (FP8, tp8): GSM8K 93.25%. FP8 needs the g_proj fix (PR #28649, MERGED) on
      // top of #28400+#28604 — lmsysorg/sglang:latest has it. Hopper does NOT hit the
      // Blackwell DeepGEMM UE8M0 issue, so no --fp8-gemm-backend flag here.
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Blackwell B200 (8-GPU HGX) — BF16 / FP8 / NVFP4 =====
    {
      // VERIFIED on 8xB200 (BF16, tp8): served clean under batched shared-prefix load, GSM8K 91.88%.
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED on 8xB200 (FP8, tp8): GSM8K 93.78% with --fp8-gemm-backend triton (laguna-m1-results.md).
      // The triton backend sidesteps the DeepGEMM UE8M0 weight-scale bug on Blackwell (~19% slower than
      // the DeepGEMM fast path). Drop the flag once PR #28662 (ue8m0 requant) merges.
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--fp8-gemm-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // VERIFIED on 8xB200 (NVFP4, tp8): GSM8K 89.38% (laguna-m1-results.md).
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Blackwell B300 (8-GPU HGX) — BF16 / FP8 / NVFP4 (UNVERIFIED) =====
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // FP8 on Blackwell → --fp8-gemm-backend triton (DeepGEMM UE8M0 workaround, pending #28662).
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--fp8-gemm-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Grace-Blackwell GB200 (4-GPU single node) — BF16 / FP8 / NVFP4 (UNVERIFIED) =====
    {
      match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // FP8 on Blackwell → --fp8-gemm-backend triton (DeepGEMM UE8M0 workaround, pending #28662).
      match: { hw: "gb200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--fp8-gemm-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Grace-Blackwell GB300 (4-GPU single node) — BF16 / FP8 / NVFP4 (UNVERIFIED) =====
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // FP8 on Blackwell → --fp8-gemm-backend triton (DeepGEMM UE8M0 workaround, pending #28662).
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--fp8-gemm-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
