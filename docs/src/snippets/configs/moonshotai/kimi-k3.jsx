// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Recipes transcribed from the K3 serving benchmark scripts
// (benchmark/H200/script/v1/launch-k3.sh, benchmark/B300/script/v1/launch-k3.sh)
// and the B200 2×8 / GB200 4×4 / H100 4×8 / MI35x 1×8 reference launches.
// Kimi-K3 is a hybrid MoE VLM: 93 layers = 69 KDA (linear) + 24 MLA, 896 routed
// experts + 1 shared. Served today from the public sgl-project/sglang kimi-k3 branch.

export const config = {
  modelName: "Kimi-K3",

  // B300 (1×8 TP8), GB300 (2×4 TP8 MNNVL), B200 (2×8: NOSPEC is PP2 × TP8 on
  // Low-Latency, PP2 × DCPEP8 on Balanced and High-Throughput, while DSPARK
  // re-lays the same 16 as flat TP16 / DCPEP16), GB200 (4×4 TP16 MNNVL),
  // H200 (2×8 TP16/EP16, or 4×8 TP32/EP32 for High-Throughput), H100
  // (4×8 TP32/EP32), and MI350X/MI355X (1×8 TP8) have serving recipes.
  supportedHardware: ["b300", "gb300", "b200", "gb200", "h200", "h100", "mi350x", "mi355x"],

  // ---- Cell introspection (config-internal; the engines ignore these keys) ----
  //
  // The overlay callbacks below are handed only the SELECTION, never the resolved
  // cell, so anything they need to know about the recipe has to be looked up.
  // Doing that lookup here — instead of restating it as per-hardware tables —
  // keeps one source of truth: the cell's own flags. A recipe that gains DCP, or
  // drops a pipeline stage, or a whole new platform, then needs no edit to the
  // overlay logic. The tables it replaces had to be revised by hand for every
  // such change, and a missed revision is invisible (the panel just quietly
  // emits a shape the server rejects).
  //
  // Mintlify strips module-level statements from snippets, so these live as
  // properties on the exported object and are called as `config.cellFor(s)`.
  cellFor(s) {
    return (config.cells || []).find(
      (c) => c.match.hw === s.hw &&
             c.match.pdMode === s.pdMode &&
             c.match.strategy === s.strategy);
  },
  // Flags are authored as "--name value" / "--name=value" strings, matching how
  // _deployment.jsx splits them for stripPrefixes.
  flagOf(cell, name) {
    return ((cell || {}).flags || []).find((f) => f.split(/[\s=]/)[0] === name);
  },
  // Absent parallelism flags mean 1 (SGLang's own default), so callers can do
  // arithmetic without null checks.
  sizeOf(cell, name) {
    const f = config.flagOf(cell, name);
    return f ? Number(f.split(/[\s=]/)[1]) || 1 : 1;
  },
  // Does this recipe shard the TP-replicated MLA KV? Replaces the per-platform
  // "which cells carry DCP" tables the HiCache tiers used to hardcode.
  hasDcp(s) {
    return !!config.flagOf(config.cellFor(s), "--dcp-size");
  },
  // A pp_size == 1 speculative algorithm cannot run a pipelined recipe. Cells
  // that would rather be re-laid flat than blocked opt in with
  // `specCollapsePp: true`; the rest are simply unavailable under speculation.
  isPipelined(s) {
    return config.sizeOf(config.cellFor(s), "--pp-size") > 1;
  },
  specCollapses(s) {
    return config.isPipelined(s) && !!(config.cellFor(s) || {}).specCollapsePp;
  },
  // The playground's chip `disable` is declarative only (no predicates), so the
  // DCP-carrying recipes are enumerated — but generated from the cells here
  // rather than typed out per platform, so it is the same single source of truth
  // as hasDcp() above. One rule per (hardware, PD role): which strategies carry
  // DCP varies along both axes — B200 reaches DCP only at High-Throughput when
  // unified, but carries it on Balanced too in the decode role — and keying on
  // hw+strategy alone (as the hand-written rules did) cannot express that without
  // over-matching one role or under-matching the other.
  get dcpStorageDisableRules() {
    const groups = new Map();
    for (const c of (config.cells || [])) {
      if (!config.flagOf(c, "--dcp-size")) continue;
      const k = `${c.match.hw}|${c.match.pdMode}`;
      if (!groups.has(k)) groups.set(k, new Set());
      groups.get(k).add(c.match.strategy);
    }
    return [...groups].map(([k, strategies]) => {
      const [hw, pdMode] = k.split("|");
      return {
        when: { hw: [hw], pdMode: [pdMode], strategy: [...strategies],
                hicache: ["l2"], spec: ["none"] },
        reason: "This recipe runs DCP, and a storage backend (L3) under DCP is rejected at startup. Switch HiCache to L3 in the Deploy panel (that drops DCP), or stay on L1+L2.",
      };
    });
  },
  // Companion to the hand-written TP/DP-Attention rules below, which enumerate
  // the platforms whose recipes are too small for a 16-rank knob. Those cover the
  // flat recipes; this covers the other reason a knob cannot widen — the recipe
  // spends its ranks on pipeline stages instead. Only cells that never collapse
  // the pipeline are listed (a specCollapsePp cell becomes flat under DSPARK and
  // is handled by its own spec-keyed rule), so no `spec` key is needed here.
  get pipelinedKnobDisableRules() {
    const groups = new Map();
    for (const c of (config.cells || [])) {
      if (config.sizeOf(c, "--pp-size") <= 1 || c.specCollapsePp) continue;
      const k = `${c.match.hw}|${c.match.pdMode}`;
      if (!groups.has(k)) groups.set(k, new Set());
      groups.get(k).add(c.match.strategy);
    }
    return [...groups].map(([k, strategies]) => {
      const [hw, pdMode] = k.split("|");
      return {
        when: { hw: [hw], pdMode: [pdMode], strategy: [...strategies] },
        reason: "This recipe spends its ranks on pipeline stages (--pp-size > 1 with a small --tp-size), so widening the attention parallelism would need more GPUs than the deployment has. Pick a flat recipe to change TP or DP-Attention.",
      };
    });
  },
  // Fold the pipeline back into the other axes at constant world size:
  // TP8 × PP2 -> TP16, and DCP8/EP8 scale with it -> DCP16/EP16. Derived from
  // the cell rather than written out, so it stays right if a cell is re-shaped.
  specCollapsedFlags(s) {
    const cell = config.cellFor(s);
    const pp = config.sizeOf(cell, "--pp-size");
    const out = [`--tp-size ${config.sizeOf(cell, "--tp-size") * pp}`];
    // HiCache under DCP rejects speculative decoding, so when a host tier is on
    // the DCP half is dropped instead of scaled (the HiCache options' own
    // stripPrefixes reach cell flags only, never these overlay flags).
    if (config.flagOf(cell, "--dcp-size") && s.hicache !== "l2" && s.hicache !== "l3") {
      out.push(`--dcp-size ${config.sizeOf(cell, "--dcp-size") * pp}`);
    }
    if (config.flagOf(cell, "--ep-size")) {
      out.push(`--ep-size ${config.sizeOf(cell, "--ep-size") * pp}`);
    }
    return out;
  },

  // Single checkpoint and a single shipped quantization (MXFP4), so neither is a
  // reader-facing axis. Node count is fixed by the hardware recipe (B200 2x8,
  // H100 4x8, B300 1x8, H200 2x8 — 4x8 on Unified High-Throughput, GB200 4x4,
  // GB300 2x4, MI350X/MI355X 1x8), so it rides on the cell rather than on a
  // selector.
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
        { id: "long-context",    label: "Long-Context",    showWhen: (s) => s.pdMode === "prefill" },
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
          // DSPARK requires pp_size == 1, so a pipelined recipe is either re-laid
          // flat (cells opting in with `specCollapsePp`) or unavailable. Both
          // branches read --pp-size off the cell, so no platform is named here:
          // whichever recipes happen to be pipelined are the ones affected.
          // Combinations with no published cell are left alone here — they render
          // an empty command panel either way, and gating on cell existence would
          // change this button on every such combination across every platform,
          // which is a separate decision from how speculation reads a recipe.
          disabled: (s) => config.isPipelined(s) && !config.specCollapses(s),
          disableReason:
            "This recipe is pipelined (--pp-size > 1) and DSPARK requires pp_size == 1. Pick a recipe that runs a single pipeline stage, or run this one NOSPEC.",
          // Where a cell does opt in, DSPARK rewrites its parallelism instead of
          // layering on top: the pipeline is stripped and folded into the other
          // axes at constant world size (see specCollapsedFlags). Cells that are
          // already flat are untouched.
          stripPrefixes: (s) =>
            config.specCollapses(s)
              ? ["--tp-size", "--pp-size", "--dcp-size", "--ep-size"]
              : [],
          // Every DSPARK recipe layers ReplaySSM on: it moves the per-draft
          // intermediate SSM states onto a fixed ring, lifting the concurrency
          // the state pool admits (needs the Triton decode kernel, the K3
          // default). Only the PD prefill role opts out — it never runs verify
          // and rejects the flag at startup.
          flags: (s) => [
            ...(config.specCollapses(s) ? config.specCollapsedFlags(s) : []),
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
          //
          // Which cells carry DCP is read off the cells themselves, so adding or
          // reshaping a DCP recipe needs no edit here.
          stripPrefixes: (s) =>
            s.spec !== "none" && config.hasDcp(s)
              ? ["--dcp-size", "--dcp-comm-backend"]
              : [],
          hints: (s) =>
            s.spec !== "none" && config.hasDcp(s)
              ? [
                  "HiCache under DCP rejects speculative decoding, so this recipe drops DCP",
                  "and serves the MLA KV TP-replicated (any PP/EP in the cell stays).",
                  "Per-request context is far shorter than the DCP",
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
          // drops the DCP operating point instead (only DCP — any PP/EP in the
          // cell stays, so on B200 the result is not plain TP). The ratio
          // calculator reads --dcp-size off this command, so dropping it also
          // re-solves --mamba-full-memory-ratio for the plain-TP shape.
          //
          // Same cell-derived DCP test as the L1+L2 option above, except this one
          // fires NOSPEC too rather than only under speculation.
          stripPrefixes: (s) =>
            config.hasDcp(s) ? ["--dcp-size", "--dcp-comm-backend"] : [],
          hints: (s) =>
            [
              "L3 also needs a mooncake_master process on rank 0 and the config file",
              "above present on every rank — the launch command alone is not enough.",
              ...(config.hasDcp(s)
                ? [
                    "L3 storage keys are not dcp_rank-aware yet, so this recipe drops DCP",
                    "and serves the MLA KV TP-replicated (any PP/EP in the cell stays).",
                    "Concurrency lands on a similar target, but",
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
    mi350x: "lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727",
    mi355x: "lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727",
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
            get disable() { return [
              {
                when: { hw: ["b300", "gb300"] },
                reason: "TP=16 needs 16 ranks; the B300 and GB300 recipes have 8 ranks.",
              },
              {
                when: { hw: ["b200"], pdMode: ["unified"], spec: ["none"] },
                reason: "With Spec Decode off, the B200 Unified recipes already use all 16 GPUs as TP8 × PP2, so changing TP to 16 would require 32 ranks. Switch Spec Decode to DSPARK — it drops the pipeline and re-lays the same GPUs as flat TP16.",
              },
              ...config.pipelinedKnobDisableRules,
            ]; },
          },
        ]},
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null, false, 2, 4,
            {
              value: 8,
              get disable() { return [
                {
                  when: { hw: ["b300", "gb300"] },
                  reason: "On an 8-rank deployment (B300 1×8, GB300 2×4) dp=8 leaves attn_tp=1, so each rank holds the full unsharded MLA KV and OOMs — prefer dp=2/attn_tp=4.",
                },
                {
                  when: { hw: ["b200"], pdMode: ["unified"], spec: ["none"] },
                  reason: "With Spec Decode off, the B200 Unified recipes run TP8 within each PP2 stage, so dp=8 leaves attn_tp=1 and each rank holds the full unsharded MLA KV — prefer dp=2/attn_tp=4, or switch Spec Decode to DSPARK for the flat TP16 shape.",
                },
                ...config.pipelinedKnobDisableRules,
              ]; },
            },
            {
              value: 16,
              get disable() { return [
                {
                  when: { hw: ["b300", "gb300"] },
                  reason: "DP-Attention=16 needs 16 TP ranks; the B300 and GB300 recipes have 8.",
                },
                {
                  when: { hw: ["b200"], pdMode: ["unified"], spec: ["none"] },
                  reason: "With Spec Decode off, the B200 Unified recipes use TP8 within each PP2 stage, so DP-Attention cannot exceed 8. Switch Spec Decode to DSPARK for the flat TP16 shape.",
                },
                ...config.pipelinedKnobDisableRules,
              ]; },
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
          get disable() { return [
            {
              when: { hw: ["b300", "gb300"] },
              reason: "EP=16 needs 16 TP ranks; the B300 and GB300 recipes have 8.",
            },
            {
              when: { hw: ["b200"], pdMode: ["unified"], spec: ["none"] },
              reason: "With Spec Decode off, the B200 Unified recipes use TP8 within each PP2 stage, so EP cannot exceed 8. Switch Spec Decode to DSPARK for the flat TP16 shape.",
            },
            ...config.pipelinedKnobDisableRules,
          ]; },
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
          get disable() { return config.dcpStorageDisableRules; } },
        { id: "mooncake", label: "Mooncake",
          get disable() { return config.dcpStorageDisableRules; } },
        { id: "hf3fs",    label: "HF3FS",
          get disable() { return config.dcpStorageDisableRules; } },
        { id: "nixl",     label: "NiXL",
          get disable() { return config.dcpStorageDisableRules; } },
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
          // Every B200 Unified cell carries --pp-size 2; left standing it
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

  // Verification marks: every cell carries "Final Verification In Progress" —
  // the recipe runs, but its serving round on the final weights and current
  // code is still open. Flip a cell to `verified: true` (and drop its
  // `verificationStatus`) once that round lands.
  cells: [
    {
      match: { hw: "b300", pdMode: "unified", strategy: "low-latency" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      // No --enable-symm-mem: it makes the fused all-reduce auto-probe skip.
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
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
      verificationStatus: "in-progress",
      env: [],
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
      match: { hw: "b300", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
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
      // NOSPEC shape. PP2 × TP8 keeps every TP collective inside one node and
      // halves the layer-local KV/state bill per GPU; the cross-node hop is a
      // pipeline P2P the next microbatch hides, not an all-reduce to wait on.
      // With DSPARK the Spec Decode overlay rewrites this to flat TP16 —
      // speculation requires pp_size == 1.
      match: { hw: "b200", pdMode: "unified", strategy: "low-latency" },
      nnodes: 2,
      // Under a pp_size == 1 speculative algorithm, re-lay this recipe flat at
      // constant world size instead of making speculation unavailable.
      specCollapsePp: true,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
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
      // PP2 × DCPEP8: DCP8 deduplicates the TP-replicated MLA KV within each
      // pipeline stage, EP8 shards the 896 experts across the same 8 ranks.
      // EP here is plain expert sharding (a2a backend stays `none`), so unlike
      // DeepEP/MegaMoE it allocates no dispatch buffers to reclaim the KV DCP
      // just bought. DSPARK rewrites the pair to TP16 + DCP16 + EP16.
      match: { hw: "b200", pdMode: "unified", strategy: "balanced" },
      nnodes: 2,
      // Under a pp_size == 1 speculative algorithm, re-lay this recipe flat at
      // constant world size instead of making speculation unavailable.
      specCollapsePp: true,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--dcp-size 8",
        "--ep-size 8",
        // Both pinned to the brought-up shape rather than left to the auto
        // resolution the rest of Blackwell uses. The MXFP4 runner needs the SiTU
        // cubin pool the published image ships; drop it to get the Marlin
        // fallback on an install without one.
        "--moe-runner-backend flashinfer_mxfp4",
        "--decode-attention-backend cutedsl_mla",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
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
      // Still redirects: the 16-GPU cell is the floor of the large-scale lane.
      match: { hw: "b200", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 2,
      // Under a pp_size == 1 speculative algorithm, re-lay this recipe flat at
      // constant world size instead of making speculation unavailable.
      specCollapsePp: true,
      verified: false,
      verificationStatus: "in-progress",
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--dcp-size 8",
        "--ep-size 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--decode-attention-backend cutedsl_mla",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      // The one H200 cell that widens past a single pair of nodes. As run — with
      // DSPARK and HiCache L1+L2 layered on, at ratio 0.058 — the static
      // allocation leaves 12.54 GB free per GPU and 1940352 KV tokens.
      match: { hw: "h200", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_MNNVL_ENABLE=1",
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
        "--enable-symm-mem",
        "--mem-fraction-static 0.90",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--dist-timeout 3600",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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

    // ----- Prefill role: chunked. The prefill role keeps radix caching, so the
    // Unified 5-slots-per-request state cost still holds (pool split rides the
    // calculator-driven ratio). Two shapes, split by cell width: the 8-GPU
    // platforms (B300 1×8, GB300 2×4) run Default as TP8 and reserve deep PP for
    // Long-Context, while the 16-GPU platforms (B200 2×8, GB200 4×4) run PP16 ×
    // TP1 on both and differ only in --mem-fraction-static. Deep PP is one
    // pipeline stage per GPU, which turns the parallelism comm from something you
    // wait for into something the next microbatch hides.
    // Both roles must agree on --page-size and --kv-cache-dtype (the transfer
    // sanity-checks them at connect), so neither is pinned here. -----
    {
      match: { hw: "b300", pdMode: "prefill", strategy: "default" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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

    {
      // PP16 x TP1 spans all 16 ranks. Prefill tok/s/GPU at ISL 8192,
      // concurrency 32: 4550 here vs 3596 (PP8 x TP2), 2407 (TEP16), 1652 (TP16);
      // flat past 32. Below concurrency ~8 the pipeline cannot fill and TEP16
      // leads instead (1947 vs 1227) — use `--tp-size 16 --ep-size 16` there.
      // That four-way comparison was measured aggregated at OSL 1; the shape
      // itself is as-run in this PD role. Pairs with the Balanced and
      // High-Throughput decode cells; the Low-Latency decode cell runs pp=2 and
      // needs a PP2 x TP8 prefill instead.
      match: { hw: "gb200", pdMode: "prefill", strategy: "default" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 16",
        "--mem-fraction-static 0.85",
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
      // Same chunked-PP shape as the Default cell, with mem-fraction raised to
      // 0.90 for KV headroom — the B300/GB300 Long-Context recipes make the same
      // trade. Keeping TP at 1 is what buys the context length here: with TP > 1
      // the MLA KV is replicated across the TP ranks, so TP2 x PP8 would hold
      // roughly half the tokens of TP1 x PP16 for the same memory.
      // Not yet benchmarked on long-context workloads.
      match: { hw: "gb200", pdMode: "prefill", strategy: "long-context" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 16",
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
      // B200 2×8 has the same 16 ranks as GB200 4×4, so the prefill role takes
      // the same PP16 × TP1 shape — one pipeline stage per GPU. The pipeline is
      // what makes this portable off MNNVL: the only cross-node traffic is a
      // pipeline P2P handoff at the node boundary (one stage boundary out of 15,
      // activations not weights), which the next microbatch hides, where a
      // TP16 or TEP16 prefill would put an all-reduce on the same link and wait
      // on it. Same caveat as GB200: below concurrency ~8 the pipeline cannot
      // fill and `--tp-size 16 --ep-size 16` leads instead. No --enable-symm-mem
      // and no DCP — at TP1 there is no TP collective to accelerate and no
      // TP-replicated KV to shard. Pairs with all three B200 decode cells: SGLang
      // requires `decode pp_size == prefill pp_size or 1`, and every B200 decode
      // cell is flat TP16 (pp=1), so none of them carry the GB200 Low-Latency
      // cell's PP2 constraint that forces a PP2 × TP8 prefill instead.
      match: { hw: "b200", pdMode: "prefill", strategy: "default" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 16",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        // Explicit multithread_load is also what keeps it on: prefetch otherwise
        // forces the single-threaded loader to avoid I/O oversubscription.
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same PP16 × TP1 shape as the Default cell with mem-fraction raised to
      // 0.90 for KV headroom, exactly the GB200 pair's trade. TP1 is what buys
      // the context length: with TP > 1 the MLA KV is replicated across the TP
      // ranks, so TP2 × PP8 would hold roughly half the tokens for the same
      // memory. Not yet benchmarked on long-context workloads.
      match: { hw: "b200", pdMode: "prefill", strategy: "long-context" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 16",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
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
    // GB200 decode figures below: ISL 8192 / OSL 1024, 16 GPU decode, behind a
    // shared PP2 x TP8 prefill. Comparisons hold; absolutes would be higher
    // behind the PP16 x TP1 prefill cell above.
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      verificationStatus: "in-progress",
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
      // PP2 x TP8 is the fastest decode shape below concurrency ~100:
      // 13.3 out tok/s/GPU @ 57.7 out tok/s/user at concurrency 8, against
      // 9.5 @ 40.7 (TP16), 9.3 @ 39.9 (DCP16) and 9.3 @ 39.7 (DCP16+EP16) —
      // +40% throughput and +45% interactivity over the best pp=1 shape. It
      // stays ahead through concurrency 64 and is overtaken by DCP16+EP16 at 128.
      // Measured without --enable-symm-mem.
      match: { hw: "gb200", pdMode: "decode", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      warn: "SGLang requires `decode pp_size == prefill pp_size or 1`, so this cell must be paired with a PP2 x TP8 prefill (`--tp-size 8 --pp-size 2`) rather than the PP16 x TP1 Prefill recipe. That prefill delivers 2919 prefill tok/s/GPU against PP16's 4550, which is the trade for the decode-side latency.",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
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
      // EP16 on top of DCP16 is +6% to +10% out tok/s/GPU at every concurrency
      // measured, at the same batch size: 56.2 vs 52.4 (c64), 92.7 vs 84.5
      // (c128), 136.1 vs 126.1 (c256), 153.5 vs 144.2 (c512).
      match: { hw: "gb200", pdMode: "decode", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--ep-size 16",
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
      // Grouping DCP as 8 x dp2 rather than a flat DCP16 is what makes MegaMoE
      // pay off here. out tok/s/GPU @ out tok/s/user, concurrency 256 / 512:
      //   DCP8 x dp2 + EP16   145.5 @ 25.4   157.5 @ 25.2   <- this cell
      //   DCP16 + EP16        136.1 @ 20.6   153.5 @ 17.5
      //   DCP16               126.1 @ 18.7   144.2 @ 16.1
      //   TP16                 87.4 @ 23.9    90.2 @ 23.9   (batch caps at ~125)
      // Pair this cell with MegaMoE in the MoE Parallelism card: +8.3% / +10.0% /
      // +11.0% over FlashInfer MXFP4 at concurrency 64 / 128 / 256, and the
      // combination validated to concurrency 512. The gain needs dp > 1 — on the
      // flat DCP16 cell MegaMoE is the slower of the two below concurrency 256.
      match: { hw: "gb200", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 8",
        "--dp-size 2",
        "--enable-dp-attention",
        "--ep-size 16",
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
  ],

  // Cross-node fabric env (substitute the NIC used by every rank).
  multiNodeHints: {
    b200: [
      // One hint list per hw, shared by every cell, so it has to name the shape
      // per role rather than assume the Unified one.
      "Unified with Spec Decode off runs TP8 within a node and PP2 across the two (+DCP8/EP8 on Balanced and High-Throughput); with DSPARK (pp_size == 1) it runs TP16 across both nodes instead.",
      "Prefill is TP1 × PP16 — one pipeline stage per GPU, non-speculative only. Decode is flat TP16 (+DCP16 on Balanced and High-Throughput).",
      "Multi-node K3 needs the cross-node NIC pinned on BOTH ranks:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # bootstrap interface",
      "  NCCL_SOCKET_IFNAME=<your-nic>   # force NCCL off kube-ipvs0",
      "  SGLANG_HOST_IP=<this-node-ip>",
      "  NCCL_IB_HCA=<hca0,hca1,...>     # RDMA fabrics only",
    ],
    gb200: [
      "Allocate all four nodes within a single NVL72 domain.",
      "MNNVL is off by default — set on every rank:",
      "  NCCL_MNNVL_ENABLE=1",
      "  NCCL_CUMEM_ENABLE=1",
      "Point the JIT caches at GB200-only paths; GB200 is SM100 and GB300 is SM103,",
      "so the two architectures need separate caches:",
      "  TORCH_EXTENSIONS_DIR / TRITON_CACHE_DIR / TVM_FFI_CACHE_DIR",
    ],
    h100: [
      "Set This node IP separately on each node; use the same cross-node NIC name on all four nodes.",
    ],
    h200: [
      "Low-Latency and Balanced run TP16/EP16 across 2 nodes; Unified High-Throughput widens to TP32/EP32 across 4.",
      "Multi-node K3 needs the cross-node NIC pinned on EVERY node:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # e.g. bond0",
      "  NCCL_SOCKET_IFNAME=<your-nic>   # force NCCL off kube-ipvs0",
      "  SGLANG_HOST_IP=<this-node-ip>",
    ],
  },
};
