// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Recipes transcribed from the K3 serving benchmark scripts
// (benchmark/H200/script/v1/launch-k3.sh, benchmark/B300/script/v1/launch-k3.sh)
// and the B200 2×8 / GB200 4×4 / H100 4×8 / MI35x 1×8 reference launches.
// Kimi-K3 is a hybrid MoE VLM: 93 layers = 69 KDA (linear) + 24 MLA, 896 routed
// experts + 1 shared. Served today from the DarkSharpness/sglang-kimi fork.

export const config = {
  modelName: "Kimi-K3",

  // B300 (1×8 TP8), GB300 (2×4 TP8 MNNVL), B200 (2×8 TP16, or TP8/PP2 for
  // Long-Context), GB200 (4×4 TP16 MNNVL), H200 (2×8 TP16/EP16), H100
  // (4×8 TP32/EP32), and MI350X/MI355X (1×8 TP8) have serving recipes.
  supportedHardware: ["b300", "gb300", "b200", "gb200", "h200", "h100", "mi350x", "mi355x"],

  // Single checkpoint and a single shipped quantization (MXFP4), so neither is a
  // reader-facing axis. Node count is fixed by the hardware recipe (B200 2x8,
  // H100 4x8, B300 1x8, H200 2x8, GB200 4x4, GB300 2x4,
  // MI350X/MI355X 1x8), so it rides on the cell rather than on a selector.
  matchDims: [
    {
      id: "pdMode",
      title: "PD Mode",
      options: [
        { id: "unified", label: "Unified"  },
        { id: "prefill", label: "Prefill"  },
        { id: "decode",  label: "Decode"   },
      ],
    },
    {
      // Prefill nodes are sized by context length; unified and decode nodes are
      // sized by the latency/throughput operating point. One row, two option sets.
      id: "strategy",
      title: "Strategy",
      options: [
        { id: "low-latency",     label: "Low-Latency",     showWhen: (s) => s.pdMode !== "prefill" },
        { id: "balanced",        label: "Balanced",        showWhen: (s) => s.pdMode !== "prefill" },
        { id: "high-throughput", label: "High-Throughput", showWhen: (s) => s.pdMode !== "prefill" },
        { id: "default",         label: "Default",         showWhen: (s) => s.pdMode === "prefill" },
        {
          id: "long-context",
          label: "Long-Context",
          showWhen: (s) =>
            s.pdMode === "prefill" ||
            (s.pdMode === "unified" && s.hw === "b200"),
        },
      ],
    },
  ],

  // Orthogonal to the cell grid: the picked option layers flags onto whichever
  // cell is showing, so turning speculation on does not triple the cell count.
  overlayDims: [
    {
      id: "spec",
      title: "Spec Decode",
      default: "dspark",
      options: [
        { id: "none", label: "Non-Spec" },
        {
          id: "dspark",
          label: "DSPARK",
          disabled: (s) => s.strategy === "long-context",
          disableReason:
            "The Long-Context recipes use pipeline parallelism (--pp-size 2 on B200 Unified, --pp-size 8 on Prefill), while DSPARK currently requires pp_size == 1.",
          // Every DSPARK recipe layers ReplaySSM on: it moves the per-draft
          // intermediate SSM states onto a fixed ring, lifting the concurrency
          // the state pool admits (needs the Triton decode kernel, the K3
          // default). Only the PD prefill role opts out — it never runs verify
          // and rejects the flag at startup.
          flags: (s) => [
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path RadixArk/Kimi-K3-DSpark",
            "--speculative-dspark-block-size 7",
            ...(s.pdMode === "prefill" ? [] : ["--enable-linear-replayssm-spec"]),
          ],
        },
        {
          id: "dflash",
          label: "DFLASH",
          // Listed so the axis is complete, but not selectable: no K3 DFLASH draft
          // checkpoint has been published, so there is nothing to point
          // --speculative-draft-model-path at. DFLASH is also CUDA-only, rejects DP
          // attention, and requires pp_size == 1.
          disabled: true,
          disableReason:
            "No K3 DFLASH draft checkpoint published yet — DSPARK is the available speculative path.",
          flags: [
            "--speculative-algorithm DFLASH",
            "--speculative-draft-model-path <dflash-draft>",
          ],
        },
      ],
    },
    {
      // Recipes transcribed from the measured HiCache rounds. The `direct` io
      // backend is deliberately not offered here: measured functionally
      // equivalent to `kernel` (bit-identical per-tier hit rate), so it belongs
      // in the Playground, not as a deployment choice.
      id: "hicache",
      title: "HiCache",
      default: "off",
      options: [
        { id: "off", label: "Off" },
        {
          id: "l2",
          label: "L1+L2 (host)",
          flags: [
            "--enable-hierarchical-cache",
          ],
          // L1/L2 host tiering IS supported under DCP, so DCP stays — except with
          // speculative decoding, which HiCache under DCP rejects at startup (the
          // draft host pool has no DCP index translation). There the DCP operating
          // point is dropped instead of blocking the option, same as L3 does. The
          // ratio calculator reads --dcp-size off this command, so dropping it also
          // re-solves --mamba-full-memory-ratio for the plain-TP shape.
          stripPrefixes: (s) =>
            ["b300", "gb300", "b200", "gb200"].includes(s.hw) &&
            ["balanced", "high-throughput"].includes(s.strategy) &&
            s.spec !== "none"
              ? ["--dcp-size", "--dcp-comm-backend"]
              : [],
          hints: (s) =>
            ["b300", "gb300", "b200", "gb200"].includes(s.hw) &&
            ["balanced", "high-throughput"].includes(s.strategy) &&
            s.spec !== "none"
              ? [
                  "HiCache under DCP rejects speculative decoding, so this recipe drops DCP",
                  "and runs plain TP. Per-request context is far shorter than the DCP",
                  "version — DCP is what buys KV capacity. Run the cell NOSPEC to keep DCP.",
                ]
              : [],
        },
        {
          id: "l3",
          label: "+ L3 (Mooncake)",
          flags: [
            "--enable-hierarchical-cache",
            "--hicache-storage-backend mooncake",
          ],
          env: ["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH={{MOONCAKE_CONFIG}}"],
          // L3 under DCP is rejected at startup (the rank-0 replicated-MLA backup
          // and the storage keys are not dcp_rank-aware), so on the DCP recipes L3
          // drops the DCP operating point and runs plain TP instead. The ratio
          // calculator reads --dcp-size off this command, so dropping it also
          // re-solves --mamba-full-memory-ratio for the plain-TP shape.
          stripPrefixes: (s) =>
            ["b300", "gb300", "b200", "gb200"].includes(s.hw) &&
            ["balanced", "high-throughput"].includes(s.strategy)
              ? ["--dcp-size", "--dcp-comm-backend"]
              : [],
          hints: (s) =>
            [
              "L3 also needs a mooncake_master process on rank 0 and the config file",
              "above present on every rank — the launch command alone is not enough.",
              ...(["b300", "gb300", "b200", "gb200"].includes(s.hw) &&
              ["balanced", "high-throughput"].includes(s.strategy)
                ? [
                    "L3 storage keys are not dcp_rank-aware yet, so this recipe drops DCP",
                    "and runs plain TP. Concurrency lands on a similar target, but",
                    "per-request context is far shorter than the DCP version — DCP is",
                    "what buys KV capacity.",
                  ]
                : []),
            ],
        },
      ],
    },
  ],

  modelNames: {
    default: "moonshotai/Kimi-K3",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    NODE0_IP:  { target: "command", label: "Head node IP",     default: "<node0-ip>"     },
    NODE_RANK: { target: "command", label: "This node rank",   default: "<node-rank>"    },
    LOCAL_IP:  { target: "command", label: "This node IP",     default: "<this-node-ip>" },
    NETWORK_IFACE: { target: "command", label: "Cross-node NIC", default: "<your-nic>"   },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    MOONCAKE_CONFIG: { target: "command", label: "Mooncake config path", default: "<mooncake.json>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost"      },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"          },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // Reproduce command for the benchmark card's "⚡ Reproduce" modal. The
  // sweep is 8k-in / 1k-out random, num_prompts = 5 × concurrency, cache-cold.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    // num_prompts = 5 × concurrency (measured floor 16).
    numPromptsByConc: { 1: 16, 16: 80, 64: 320, 256: 1280, 1024: 5120 },
  },

  // Recommend the published CUDA 13 image for every NVIDIA recipe. MI35x keeps
  // the ROCm 7.2 image.
  dockerImages: {
    h100:   "lmsysorg/sglang:kimi-k3",
    h200:   "lmsysorg/sglang:kimi-k3",
    b300:   "lmsysorg/sglang:kimi-k3",
    gb300:  "lmsysorg/sglang:kimi-k3",
    b200:   "lmsysorg/sglang:kimi-k3",
    gb200:  "lmsysorg/sglang:kimi-k3",
    mi350x: "lmsysorg/sglang:dev-rocm720-mi35x",
    mi355x: "lmsysorg/sglang:dev-rocm720-mi35x",
  },
  // Pre-selects the issue template's `model` field on "Submit verified cell".
  github: {
    cookbookModel: "moonshotai/kimi-k3",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" -----
    // DP-Attention is a combined knob: value = DP degree AND toggles `--enable-dp-attention`.
    // K3's MLA latent KV is TP-replicated, so DP-attention RAISES per-GPU KV pressure —
    // dp=8/attn_tp=1 OOMs on a single node; use dp=2/attn_tp=4. No CP knob: K3 uses
    // decode context parallel (`--dcp-size`), a different lever from prefill `--attn-cp-size`.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null, 8,
          {
            value: 16,
            disable: [
              {
                when: { hw: ["b300", "gb300"] },
                reason: "TP=16 needs 16 ranks; the B300 and GB300 recipes have 8 ranks.",
              },
              {
                when: { hw: ["b200"], strategy: ["long-context"] },
                reason: "The B200 Long-Context recipe already uses all 16 GPUs as TP8 × PP2; changing TP to 16 would require 32 ranks.",
              },
            ],
          },
        ]},
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null, false, 2, 4,
            { value: 8, disable: { hw: ["b300", "gb300"] },
              disableReason: "On an 8-rank deployment (B300 1×8, GB300 2×4) dp=8 leaves attn_tp=1, so each rank holds the full unsharded MLA KV and OOMs — prefer dp=2/attn_tp=4." },
            {
              value: 16,
              disable: [
                {
                  when: { hw: ["b300", "gb300"] },
                  reason: "DP-Attention=16 needs 16 TP ranks; the B300 and GB300 recipes have 8.",
                },
                {
                  when: { hw: ["b200"], strategy: ["long-context"] },
                  reason: "The B200 Long-Context recipe uses TP8 within each PP stage, so DP-Attention cannot exceed 8.",
                },
              ],
            },
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // ----- Card: "MoE Parallelism" -----
    // K3 = 896 routed experts + 1 shared. Marlin (W4A16) is the accuracy runner; the
    // MXFP4 / a2a runners (FlashInfer MXFP4, DeepEP, MegaMoE) are throughput levers.
    moe: {
      backend: {
        options: [
          { id: null,               label: "Inherited" },
          { id: "deepep",           label: "DeepEP",            flags: ["--moe-a2a-backend deepep"] },
          // Blackwell-only kernel-fusion path; selecting it reveals the Quantization sub-select.
          { id: "megamoe",          label: "MegaMoE",           flags: ["--moe-a2a-backend megamoe"],
            requiresHw: ["b200", "b300", "gb200", "gb300"] },
          // Blackwell-only: runs the prebuilt trtllm-gen SiTU cubins; needs the
          // downloadable SiTU cubin pool unpacked and pointed to by the env var.
          { id: "flashinfer_mxfp4", label: "FlashInfer (MXFP4)", flags: ["--moe-runner-backend flashinfer_mxfp4"],
            env: ["SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=/path/to/trtllm_gen_moe_cubin_pool"],
            requiresHw: ["b200", "b300", "gb200", "gb300"] },
          { id: "marlin",           label: "Marlin (W4A16)",    flags: ["--moe-runner-backend marlin"] },
        ],
      },
      // MegaMoE quantization sub-select — shown only when backend === "megamoe".
      megamoeQuant: {
        stripEnv: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
        options: [
          { id: "w4a8", label: "W4A8",
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
          { id: "w4a4", label: "W4A4",
            env: [
              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320",
              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1",
              "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1",
            ] },
        ],
      },
      ep: { label: "EP", values: [
        null, 1, 2, 4, 8,
        {
          value: 16,
          disable: [
            {
              when: { hw: ["b300", "gb300"] },
              reason: "EP=16 needs 16 TP ranks; the B300 and GB300 recipes have 8.",
            },
            {
              when: { hw: ["b200"], strategy: ["long-context"] },
              reason: "The B200 Long-Context recipe uses TP8 within each PP stage, so EP cannot exceed 8.",
            },
          ],
        },
      ]},
    },

    // ----- Card: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser kimi_k3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser kimi_k3" },
      ],
    },

    // ----- Card: "PD Disaggregation" ----- (validated functionally on B300×2 mooncake)
    // No `modes` list: the role is a Deploy-panel dimension (PD Mode), so this card
    // only tunes the transport and stays hidden until a role is selected there.
    pdDisagg: {
      showWhen: (b) => b.pdMode === "prefill" || b.pdMode === "decode",
      transferBackends: [
        { id: "nixl", label: "NiXL" },
        { id: "mooncake", label: "Mooncake" },
      ],
      // `auto` is a sentinel (emits no --disaggregation-ib-device flag).
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0"],
      router: {
        port: 8000,
        // Ports come from the engine's PD_PORTS, the same source the role
        // commands rewrite `--port` from, so the router always targets the
        // ports the two roles actually bind. The positional after --prefill is
        // the bootstrap port; it must match the prefill server's
        // --disaggregation-bootstrap-port (default 8998) or only the decode
        // worker registers.
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`,
      },
    },

    // ----- Card: "HiCache" ----- (K3 hybrid L1/L2/L3, incl. KDA state)
    hicache: {
      showWhen: (b) => b.hicache !== undefined && b.hicache !== "off",
      // Picking a storage backend here IS L3, and L3 under DCP is rejected at
      // startup. The Deploy panel's own L3 option drops DCP first; reaching L3
      // through this card would not, so gate it on the DCP recipes unless Deploy
      // already switched to L3 (hicache "off"/"l2" means DCP is still standing).
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File",
          disable: [{ when: { hw: ["b300", "gb300", "b200", "gb200"], strategy: ["balanced", "high-throughput"], hicache: ["l2"], spec: ["none"] },
                      reason: "This recipe runs DCP, and a storage backend (L3) under DCP is rejected at startup. Switch HiCache to L3 in the Deploy panel (that drops DCP), or stay on L1+L2." }] },
        { id: "mooncake", label: "Mooncake",
          disable: [{ when: { hw: ["b300", "gb300", "b200", "gb200"], strategy: ["balanced", "high-throughput"], hicache: ["l2"], spec: ["none"] },
                      reason: "This recipe runs DCP, and a storage backend (L3) under DCP is rejected at startup. Switch HiCache to L3 in the Deploy panel (that drops DCP), or stay on L1+L2." }] },
        { id: "hf3fs",    label: "HF3FS",
          disable: [{ when: { hw: ["b300", "gb300", "b200", "gb200"], strategy: ["balanced", "high-throughput"], hicache: ["l2"], spec: ["none"] },
                      reason: "This recipe runs DCP, and a storage backend (L3) under DCP is rejected at startup. Switch HiCache to L3 in the Deploy panel (that drops DCP), or stay on L1+L2." }] },
        { id: "nixl",     label: "NiXL",
          disable: [{ when: { hw: ["b300", "gb300", "b200", "gb200"], strategy: ["balanced", "high-throughput"], hicache: ["l2"], spec: ["none"] },
                      reason: "This recipe runs DCP, and a storage backend (L3) under DCP is rejected at startup. Switch HiCache to L3 in the Deploy panel (that drops DCP), or stay on L1+L2." }] },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },

    // ----- Axis: Flag Selects (K3 hybrid dual-pool knobs) -----
    // KDA state pool vs full-KV pool levers (measured dual-pool analysis). The
    // flagless option is the accuracy-safe default; the others are capacity/long-ctx
    // levers whose accuracy A/B is workload-gated — Playground opt-ins, not cells.
    flagSelects: [
      {
        // Deployment picks this from the strategy; the row is the override, and the
        // engine derives the current pick from the base command so it shows whatever
        // Deploy resolved until you change it.
        //
        // The unit is tokens PROPOSED per step, which each algorithm spells its own
        // way — strip all three families so switching never leaves a stale flag:
        //   DSPARK  --speculative-dspark-block-size N   (gamma, == proposed)
        //   DFLASH  --speculative-dflash-block-size N+1 (verify window)
        //   EAGLE   --speculative-num-steps N           (chain; topk>1 is a tree)
        // Only DSPARK is selectable today, so only its form is emitted.
        id: "proposedDraftTokens", title: "Proposed Draft Tokens",
        showWhen: (b) => b.spec === "dspark",
        control: "slider",
        stripPrefixes: [
          "--speculative-dspark-block-size",
          "--speculative-dflash-block-size",
          "--speculative-num-steps",
        ],
        options: [
          { id: "1", label: "1", flags: ["--speculative-dspark-block-size 1"] },
          { id: "2", label: "2", flags: ["--speculative-dspark-block-size 2"] },
          { id: "3", label: "3", flags: ["--speculative-dspark-block-size 3"] },
          { id: "4", label: "4", flags: ["--speculative-dspark-block-size 4"] },
          { id: "5", label: "5", flags: ["--speculative-dspark-block-size 5"] },
          { id: "6", label: "6", flags: ["--speculative-dspark-block-size 6"] },
          { id: "7", label: "7", flags: ["--speculative-dspark-block-size 7"] },
        ],
      },
      {
        // ReplaySSM moves the per-draft intermediate SSM states onto a fixed ring,
        // freeing state slots for more concurrency at a lower --mamba-full-memory-ratio.
        // Spec-only, so gate the row on DSPARK; every DSPARK recipe (except the PD
        // prefill role) turns it on in the base, so this row derives to On and
        // exists mainly as the opt-out.
        // Needs the Triton linear-attn decode backend (the K3 default).
        id: "replaySsm", title: "ReplaySSM (spec)",
        showWhen: (b) => b.spec === "dspark",
        stripPrefixes: ["--enable-linear-replayssm-spec"],
        options: [
          { id: "off", label: "Off" },
          {
            id: "on", label: "On",
            // The ring is spec-verify-only scratch and a prefill server never
            // runs verify, so the engine rejects the flag outright there.
            disable: { pdMode: ["prefill"] },
            disableReason:
              "A PD prefill server never runs speculative verify, so --enable-linear-replayssm-spec is rejected at startup.",
            flags: ["--enable-linear-replayssm-spec"],
          },
        ],
      },
      {
        // Compact prunes the verify layout to the SPS budget. SILENT-INERT
        // without --speculative-dspark-sps-table-path (every step still
        // verifies full width); fails fast with ReplaySSM or DCP > 1.
        id: "raggedVerify", title: "Ragged Verify Mode (spec)",
        showWhen: (b) => b.spec === "dspark",
        stripEnv: ["SGLANG_RAGGED_VERIFY_MODE"],
        options: [
          { id: "static",  label: "Auto (static)" },
          { id: "compact", label: "Compact (requires SPS table)", env: ["SGLANG_RAGGED_VERIFY_MODE=compact"] },
        ],
      },
      {
        id: "kvCacheDtype", title: "KV Cache Precision",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "auto", label: "Auto (BF16)" },
          { id: "fp8",  label: "FP8 (E4M3) — halves KV memory", flags: ["--kv-cache-dtype fp8_e4m3"] },
        ],
      },
      {
        id: "mambaSsmDtype", title: "KDA State Precision",
        stripPrefixes: ["--mamba-ssm-dtype"],
        options: [
          { id: "auto", label: "Auto (FP32)" },
          { id: "bf16", label: "BFloat16 — halves state memory", flags: ["--mamba-ssm-dtype bfloat16"] },
          // The dtype --enable-mamba-cache-stochastic-rounding requires; no
          // serving round has tried it, unlike bf16.
          { id: "fp16", label: "Float16 — stochastic-rounding capable", flags: ["--mamba-ssm-dtype float16"] },
        ],
      },
      {
        // Whole-model prefix cache (the radix tree spans MLA KV + KDA state).
        // Off suits prefix-free traffic (offline batch, evals): 1 state slot
        // per request instead of 4-5.
        id: "prefixCache", title: "Prefix Cache",
        stripPrefixes: ["--disable-radix-cache"],
        options: [
          { id: "on",  label: "On" },
          { id: "off", label: "Off", flags: ["--disable-radix-cache"] },
        ],
      },
      {
        // How KDA state buffers for radix reuse; no strategy exists with the
        // prefix cache off, so the row hides (and stops emitting) there.
        // Slot cost per request: extra_buffer 5, extra_buffer_lazy 4.
        id: "mambaRadix", title: "KDA Radix Cache Strategy",
        showWhen: (b, v, d) => (((v && v.prefixCache) ?? (d && d.prefixCache)) !== "off"),
        stripPrefixes: ["--mamba-radix-cache-strategy"],
        options: [
          { id: "auto",  label: "Auto (extra_buffer)" },
          { id: "lazy",  label: "extra_buffer_lazy", flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
          { id: "nobuf", label: "no_buffer",         flags: ["--mamba-radix-cache-strategy no_buffer"] },
        ],
      },
      {
        // Experimental env toggle: SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK skips the
        // decode-time mamba lock, freeing one resident state slot per request
        // (extra_buffer 5→4, extra_buffer_lazy 4→3; no_buffer stays 3). Off by
        // default. Env var, not a flag, so it emits via env/stripEnv.
        id: "mambaSlotSaving", title: "KDA Slot Saving (experimental)",
        stripEnv: ["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"],
        options: [
          { id: "off", label: "Off" },
          { id: "on",  label: "On", env: ["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"] },
        ],
      },
      {
        // Only meaningful with EP a2a on (MoE card or a large-scale preset).
        id: "eplb", title: "Expert Rebalancing (EPLB)",
        stripPrefixes: ["--enable-eplb"],
        options: [
          { id: "off", label: "Off" },
          { id: "on",  label: "On (requires EP a2a)", flags: ["--enable-eplb"] },
        ],
      },
      {
        // Prefill has no graph by default; BCG captures it as a breakable graph.
        // Validated on the no-a2a MXFP4 runner; untested against SBO (EP a2a)
        // and DP attention.
        id: "prefillGraph", title: "Prefill CUDA Graph",
        stripPrefixes: ["--cuda-graph-backend-prefill"],
        options: [
          { id: "auto", label: "Auto (off)" },
          { id: "bcg",  label: "Breakable (BCG)", flags: ["--cuda-graph-backend-prefill breakable"] },
        ],
      },
      {
        // A decode server runs a chunk cache by default (1 state slot/req);
        // radix restores prefix reuse at the unified per-request slot cost.
        id: "pdDecodeRadix", title: "PD Decode Radix Cache",
        showWhen: (b) => b.pdMode === "decode",
        stripPrefixes: ["--disaggregation-decode-enable-radix-cache"],
        options: [
          { id: "off", label: "Off (chunk cache)" },
          { id: "on",  label: "On", flags: ["--disaggregation-decode-enable-radix-cache"] },
        ],
      },
      {
        // The two rows below compose: Cluster Size picks N, Large-Scale Preset
        // resolves it into the full parallelism shape (tp/ep/dp/dcp; attn-tp =
        // tp/dp); pool sizing rides the calculator-driven ratio.
        id: "lsGpus", title: "Cluster Size (large-scale)",
        showWhen: (b) => b.pdMode === undefined || b.pdMode === "unified",
        // Default follows the base cell's own GPU count (tp8 lanes -> 8,
        // tp16 lanes -> 16), so a preset starts from "same hardware, new shape".
        default: (b) =>
          ({ b300: "8", gb300: "8", b200: "16", gb200: "16",
             h200: "16", h100: "32", mi350x: "8", mi355x: "8" })[(b || {}).hw] || "32",
        stripPrefixes: [],
        options: [
          { id: "8",  label: "8 GPUs" },
          { id: "16", label: "16 GPUs" },
          { id: "32", label: "32 GPUs" },
          { id: "64", label: "64 GPUs" },
        ],
      },
      {
        id: "lsPreset", title: "Large-Scale Preset",
        showWhen: (b) => b.pdMode === undefined || b.pdMode === "unified",
        stripPrefixes: [
          "--tp-size", "--tp", "--tensor-parallel-size",
          "--ep-size", "--ep", "--expert-parallel-size",
          "--enable-dp-attention", "--dp-size", "--enable-dp-lm-head",
          "--dcp-size", "--dcp-comm-backend",
          // The B200 Long-Context cell carries --pp-size 2; left standing it
          // multiplies against the preset's --tp-size for a world size the
          // preset's own --nnodes cannot satisfy.
          "--pp-size", "--pipeline-parallel-size",
          "--moe-a2a-backend", "--moe-runner-backend",
          "--kv-cache-dtype", "--mamba-ssm-dtype", "--mamba-radix-cache-strategy",
          "--mem-fraction-static", "--disable-radix-cache", "--enable-symm-mem",
        ],
        stripEnv: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK"],
        options: [
          { id: "off", label: "Off", flags: () => null },
          {
            id: "serving", label: "Peak Throughput",
            disable: { hw: ["h100", "h200", "mi350x", "mi355x"] },
            disableReason: "The large-scale presets ride the MegaMoE a2a lane (SM100/SM103) — Blackwell only.",
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480"],
            flags: (v, b) => {
              const n = Number(v.lsGpus) || 32;
              const dp = n / 8;
              const gpusPerNode = ["gb200", "gb300"].includes(b.hw) ? 4 : 8;
              const nnodes = n / gpusPerNode;
              return [
                `--tp-size ${n}`, `--ep-size ${n}`,
                ...(dp > 1 ? ["--enable-dp-attention", `--dp-size ${dp}`, "--enable-dp-lm-head"] : []),
                ...(nnodes > 1 ? [`--nnodes ${nnodes}`, "--node-rank {{NODE_RANK}}", "--dist-init-addr {{NODE0_IP}}:20000"] : []),
                "--moe-a2a-backend megamoe", "--moe-runner-backend deep_gemm",
                "--kv-cache-dtype fp8_e4m3", "--mamba-ssm-dtype bfloat16",
                "--mamba-radix-cache-strategy extra_buffer_lazy",
                "--mem-fraction-static 0.92",
              ];
            },
          },
          {
            id: "capacity", label: "Peak Capacity (+DCP8)",
            // The only playground option that re-adds DCP, so it is also the only
            // one that can resurrect the DCP + L3 combination the Deploy panel
            // just stripped out.
            disable: [
              {
                when: { hw: ["h100", "h200", "mi350x", "mi355x"] },
                reason: "The large-scale presets ride the MegaMoE a2a lane (SM100/SM103) — Blackwell only.",
              },
              {
                when: { hicache: ["l3"] },
                reason: "L3 storage keys are not dcp_rank-aware, so DCP and L3 cannot run together. Use Peak Throughput, or switch HiCache to L1+L2.",
              },
              {
                when: { hicache: ["l2"], spec: ["dspark"] },
                reason: "HiCache under DCP rejects speculative decoding, so this preset cannot add DCP here. Use Peak Throughput, or run the cell NOSPEC.",
              },
            ],
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480"],
            flags: (v, b) => {
              const n = Number(v.lsGpus) || 32;
              const dp = n / 8;
              const gpusPerNode = ["gb200", "gb300"].includes(b.hw) ? 4 : 8;
              const nnodes = n / gpusPerNode;
              return [
                `--tp-size ${n}`, `--ep-size ${n}`,
                ...(dp > 1 ? ["--enable-dp-attention", `--dp-size ${dp}`, "--enable-dp-lm-head"] : []),
                "--dcp-size 8",
                ...(nnodes > 1 ? [`--nnodes ${nnodes}`, "--node-rank {{NODE_RANK}}", "--dist-init-addr {{NODE0_IP}}:20000"] : []),
                "--moe-a2a-backend megamoe", "--moe-runner-backend deep_gemm",
                "--kv-cache-dtype fp8_e4m3", "--mamba-ssm-dtype bfloat16",
                "--mamba-radix-cache-strategy extra_buffer_lazy",
                "--mem-fraction-static 0.92",
              ];
            },
          },
        ],
      },
    ],
  },

  cells: [
    {
      match: { hw: "b300", pdMode: "unified", strategy: "low-latency" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--disable-custom-all-reduce",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "unified", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Plain TP16 avoids pipeline bubbles at the shallow 16-request point.
      match: { hw: "b200", pdMode: "unified", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--mem-fraction-static 0.85",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // TP16 + DCP16 on two B200 nodes.
      match: { hw: "b200", pdMode: "unified", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Balanced baseline; High-Throughput routes to the large-scale presets.
      match: { hw: "b200", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Reference long-context launch: PP2 halves the layer-local KV/state
      // footprint per GPU while TP8 spans each 8-GPU pipeline stage.
      match: { hw: "b200", pdMode: "unified", strategy: "long-context" },
      nnodes: 2,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--mem-fraction-static 0.85",
        "--context-length 131072",
        "--chunked-prefill-size 8192",
        "--mamba-radix-cache-strategy extra_buffer",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI350X and MI355X use the same single-node TP8 ROCm/AITER profile.
      match: { hw: "mi350x", pdMode: "unified", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 8",
        "--attention-backend triton",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same ROCm/AITER profile as MI350X.
      match: { hw: "mi355x", pdMode: "unified", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 8",
        "--attention-backend triton",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Latency operating point. Keep the fixed H100 state budget but use the
      // default extra_buffer strategy; no explicit request-concurrency cap.
      match: { hw: "h100", pdMode: "unified", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Accuracy-preserving default. The fixed state budget is safer than
      // guessing a KDA/KV ratio on 80 GB GPUs.
      match: { hw: "h100", pdMode: "unified", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Throughput operating point from the 4×8 H100 reference launch.
      // extra_buffer_lazy lowers the per-request state cost from 5 to 4 slots.
      match: { hw: "h100", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "unified", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Accuracy-preserving default (eval-command: mem-frac 0.85, graph-bs 64).
      match: { hw: "h200", pdMode: "unified", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Throughput-tuned (perf-command: mem-frac 0.90, graph-bs 256, extra_buffer_lazy → max_running 98).
      match: { hw: "h200", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.90",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "unified", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Balanced: the DCP8 lane at stock mem-frac 0.85; MoE runner and
      // attention backends resolve automatically on SM100/SM103.
      match: { hw: "gb300", pdMode: "unified", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same low-latency operating point as GB300, expanded from 2×4 to 4×4.
      match: { hw: "gb200", pdMode: "unified", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same balanced operating point as GB300; TP/DCP span all 16 ranks.
      match: { hw: "gb200", pdMode: "unified", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same high-throughput operating point as GB300; TP/DCP span all 16 ranks.
      match: { hw: "gb200", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ----- Prefill role: chunked, on the TP8 platforms. The prefill role keeps
    // radix caching, so the Unified 5-slots-per-request state cost still holds
    // (pool split rides the calculator-driven ratio). Default is TP8;
    // Long-Context is one pipeline stage per GPU, which turns the parallelism
    // comm from something you wait for into something the next microbatch hides.
    // Both roles must agree on --page-size and --kv-cache-dtype (the transfer
    // sanity-checks them at connect), so neither is pinned here. -----
    {
      match: { hw: "b300", pdMode: "prefill", strategy: "default" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--disable-custom-all-reduce",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        // Must match the positional bootstrap port the router passes after
        // --prefill, or only the decode worker registers.
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // PP8xTP1: no TP collective left to accelerate, so --enable-symm-mem is
      // dropped rather than carried over from the TP8 cell.
      match: { hw: "b300", pdMode: "prefill", strategy: "long-context" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "prefill", strategy: "default" },
      nnodes: 2,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "prefill", strategy: "long-context" },
      nnodes: 2,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 8",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ----- Decode role: the unified cell for the same hw and strategy, plus
    // the PD role and transport flags, and re-sized KDA state.
    //
    // Decode runs the KV cache as a chunk cache, so the unified 5-slots-per-
    // request reservation (1 state + ping-pong copies for radix reuse) drops to
    // a single slot, and --mamba-radix-cache-strategy stops having any effect.
    // In-transfer requests holding a slot before decode starts are the only
    // extra, so the pool is sized max-running-requests + extra slots. Pinning
    // both keeps the identity explicit instead of leaning on the auto-default,
    // which reserves nothing at all once the batch exceeds 32. -----
    {
      match: { hw: "b300", pdMode: "decode", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "decode", strategy: "low-latency" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--disable-custom-all-reduce",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.92",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", pdMode: "decode", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", pdMode: "decode", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.92",
        "--disaggregation-decode-extra-slots 16",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", pdMode: "decode", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 8",
        "--attention-backend triton",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", pdMode: "decode", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 8",
        "--attention-backend triton",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", pdMode: "decode", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", pdMode: "decode", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "decode", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "decode", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.90",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "decode", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "decode", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.92",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", pdMode: "decode", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", pdMode: "decode", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.92",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],

  // Cross-node fabric env (substitute the NIC used by every rank).
  multiNodeHints: {
    b200: [
      "Low-Latency, Balanced, and High-Throughput use TP16 across both nodes; Long-Context uses TP8 within each PP2 stage.",
      "Multi-node K3 needs the cross-node NIC pinned on BOTH ranks:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # bootstrap interface",
      "  NCCL_SOCKET_IFNAME=<your-nic>   # force NCCL off kube-ipvs0",
      "  SGLANG_HOST_IP=<this-node-ip>",
      "  NCCL_IB_HCA=<hca0,hca1,...>     # RDMA fabrics only",
    ],
    h100: [
      "Set This node IP separately on each node; use the same cross-node NIC name on all four nodes.",
    ],
    h200: [
      "Multi-node K3 needs the cross-node NIC pinned on BOTH ranks:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # e.g. bond0",
      "  NCCL_SOCKET_IFNAME=<your-nic>   # force NCCL off kube-ipvs0",
      "  SGLANG_HOST_IP=<this-node-ip>",
    ],
  },
};
