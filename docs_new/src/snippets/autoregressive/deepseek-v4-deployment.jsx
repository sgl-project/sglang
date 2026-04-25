export const DeepSeekV4Deployment = () => {
  // DeepSeek-V4 deployment matrix (small / real checkpoint):
  //   Hardware × Recipe → concrete launch command.
  //
  //   Hardware (quantization determined by GPU generation):
  //     B200  → FP4 weights, Flash TP=4 / Pro TP=8 single-node
  //     GB300 → FP4 weights, Flash TP=4 / Pro TP=4 single-node
  //     H200  → FP8 weights, Flash TP=4 / Pro TP=16 2-node
  //   Model variant → HF slug:
  //     Flash (285B) → deepseek-ai/DeepSeek-V4-Flash
  //     Pro   (1.6T) → deepseek-ai/DeepSeek-V4-Pro
  //
  //   Recipe:
  //     low-latency    → TP(+DP on H200 no, Blackwell no), MTP 3/4
  //     balanced       → DP-attn + DeepEP + MTP 1/2
  //     max-throughput → DP-attn + DeepEP, no MTP
  //     cp             → TP + DeepEP + context-parallel flags, no MTP
  //     pd-disagg      → 1P1D (prefill + decode + router), separate commands shown together
  //
  // HF slugs, parser names, and `sglang serve` flag parity are all confirmed —
  // see cookbook_v2/DISCUSSION.md ("人类提供的事实" and 设计决定 §3).

  const options = {
    hardware: {
      name: "hardware",
      title: "Hardware Platform",
      items: [
        { id: "b200",  label: "B200 (FP4)",  default: true  },
        { id: "b300",  label: "B300 (FP4)",  default: false  },
        { id: "gb300", label: "GB300 (FP4)", default: false },
        { id: "h200",  label: "H200 (FP8)",  default: false },
      ],
    },
    modelSize: {
      name: "modelSize",
      title: "Model Variant",
      items: [
        { id: "small", label: "Flash", default: true,  subtitle: "285B" },
        { id: "big",   label: "Pro",   default: false, subtitle: "1.6T" },
      ],
    },
    recipe: {
      name: "recipe",
      title: "Recipe",
      items: [
        { id: "low-latency",    label: "Low-Latency",      default: true  },
        { id: "balanced",       label: "Balanced",         default: false },
        { id: "max-throughput", label: "Max-Throughput",   default: false },
        { id: "cp",             label: "Context-Parallel", default: false },
        { id: "pd-disagg",      label: "PD-Disagg",        default: false },
      ],
    },
    reasoningParser: {
      name: "reasoningParser",
      title: "Reasoning Parser",
      items: [
        { id: "disabled", label: "Disabled", default: true  },
        { id: "enabled",  label: "Enabled",  default: false, subtitle: "deepseek-v4" },
      ],
    },
    toolcall: {
      name: "toolcall",
      title: "Tool Call Parser",
      items: [
        { id: "disabled", label: "Disabled", default: true  },
        { id: "enabled",  label: "Enabled",  default: false, subtitle: "deepseekv4" },
      ],
    },
  };

  const resolveItems = (option) => option.items;

  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      const items = resolveItems(option);
      const def = items.find((i) => i.default && !i.disabled) || items.find((i) => !i.disabled) || items[0];
      initialState[key] = def.id;
    }
    return initialState;
  };

  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDarkMode = () => {
      const html = document.documentElement;
      const isDarkMode =
        html.classList.contains("dark") ||
        html.getAttribute("data-theme") === "dark" ||
        html.style.colorScheme === "dark";
      setIsDark(isDarkMode);
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"],
    });
    return () => observer.disconnect();
  }, []);

  const handleRadioChange = (optionName, value) => {
    setValues((prev) => ({ ...prev, [optionName]: value }));
  };

  // ============================================================================
  // generateCommand — strict mirror of sunrise_allinone.py LAUNCH_COMMANDS
  // for BOTH small and big (1.6T) real-checkpoint rows.
  //
  // SOURCE OF TRUTH: sunrise_final/sunrise_allinone.py LAUNCH_COMMANDS dict.
  // Allowed deviations are documented in cookbook_v2/DISCUSSION.md
  // → "Human-approved diffs from allinone":
  //   1. NVSHMEM env (B200) removed — personal hardware NIC mapping
  //   2. Model path uses HF slug instead of allinone's local paths
  //   3. `sglang serve` instead of `python3 -m sglang.launch_server`
  //   4. (retired — big is now a real ckpt and exposed)
  //   5. GB300 PD MNNVL topology envs (MC_FORCE_MNNVL / NCCL_*) removed;
  //      SGLANG_MOONCAKE_CUSTOM_MEM_POOL kept.
  //
  // Any other diff vs allinone is a bug — fix the JSX, not the whitelist.
  // ============================================================================

  // === SHARED BEGIN ===
  // Constants reachable by both generateCommand and buildPDDisaggCommand.
  // verify_commands.mjs also scrapes this block between the SHARED markers and
  // prepends it to the extracted function bodies (since `new Function(body)`
  // loses closure scope). Don't rename the markers.

  // Per (hardware, modelSize) spec derived from allinone _MODEL_SPEC.
  // "small" (JSX id) = DeepSeek-V4-Flash (285B); "big" = DeepSeek-V4-Pro (1.6T).
  // The internal ids match allinone's model="small" / model="big" keys so the
  // verify_commands.py diff is mechanical. One HF repo per variant holds both
  // FP8 and FP4 weights (quantization picked by hardware, not by repo suffix).
  const HW_SIZE_SPEC = {
    "b200|small":  { slug: "deepseek-ai/DeepSeek-V4-Flash", tp: 4,  multinode: false },
    "b200|big":    { slug: "deepseek-ai/DeepSeek-V4-Pro",   tp: 8,  multinode: false },
    "gb300|small": { slug: "deepseek-ai/DeepSeek-V4-Flash", tp: 4,  multinode: false },
    "gb300|big":   { slug: "deepseek-ai/DeepSeek-V4-Pro",   tp: 4,  multinode: false },
    // H200 needs an FP8-only Instruct ckpt (deepseek-ai's Flash/Pro repos ship
    // FP4-mixed weights that Hopper can't run). sgl-project publishes FP8
    // repackagings for both variants.
    "h200|small":  { slug: "sgl-project/DeepSeek-V4-Flash-FP8",        tp: 4,  multinode: false },
    "h200|big":    { slug: "sgl-project/DeepSeek-V4-Pro-FP8",          tp: 16, multinode: true, nnodes: 2 },
  };
  // Per (hardware, modelSize) PD role TP (from allinone _PD_SPEC).
  const PD_TP_SPEC = {
    "b200|small":  { tp: 2,  multinode: false },
    "b200|big":    { tp: 8,  multinode: false },
    "gb300|small": { tp: 4,  multinode: false },
    "gb300|big":   { tp: 4,  multinode: false },
    "h200|small":  { tp: 4,  multinode: false },
    "h200|big":    { tp: 16, multinode: true, nnodes: 2 },
  };
  // Recipes that have been end-to-end verified on the latest (Flash/Pro) HF
  // checkpoints. Every cell NOT listed here is emitted with its entire body
  // commented out (every line prefixed with `# `) plus a "being verified"
  // banner on top — so copy-pasting an unverified command is a no-op in shell.
  // To mark a cell verified, add its "hardware|modelSize|recipe" string here
  // and the cell renders as a normal, runnable command.
  // pd-disagg is verified as a single unit (both prefill and decode together).
  const VERIFIED_RECIPES = new Set([
    "b200|small|low-latency",
    "b200|small|balanced",
    "b200|small|max-throughput",
    "b200|small|cp",
    "b200|small|pd-disagg",
    "b200|big|low-latency",
    "b200|big|balanced",
    "b200|big|max-throughput",
    "b200|big|cp",
    "h200|small|low-latency",
    "h200|small|balanced",
    "h200|small|max-throughput",
    "gb300|small|low-latency",
    "gb300|small|balanced",
    "gb300|small|max-throughput",
    "h200|small|cp",
    "h200|small|pd-disagg",
    // h200|big|pd-disagg: pending verification (needs 4-node H200 cluster with
    //   shared IB fabric: 2-node prefill + 2-node decode).
    "gb300|small|cp",
    "gb300|big|cp",
    "gb300|small|pd-disagg",
    "gb300|big|pd-disagg",
  ]);
  // Recipes whose command is intentionally not yet provided (e.g. blocked by an
  // upstream limitation). Showing a minimal placeholder is friendlier to users
  // than emitting a commented-out invalid command.
  const TBD_RECIPES = new Set([
    "h200|big|cp",
  ]);
  const TBD_PLACEHOLDER = "# to be provided";
  const BEING_VERIFIED_NOTE =
    "# NOTE: this recipe is being verified on the latest checkpoint";

  // Prefix every line with "# " so the whole command becomes a shell no-op.
  const commentOutCommand = (cmd) =>
    cmd
      .split("\n")
      .map((line) => (line.length ? `# ${line}` : "#"))
      .join("\n");

  // DeepEP large SMS flag (allinone _DEEPEP_LARGE_SMS_FLAG).
  const DEEPEP_LARGE_SMS_FLAG =
    `  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'`;

  // Multi-node flags (renders with <node0-ip> / <node-rank> placeholders;
  // allinone template uses {node0_ip} / {node_rank} that verify_commands.py formats
  // with the same placeholder strings so dynamic-diff stays exact).
  const multiNodeFlags = (nnodes) => [
    `  --nnodes ${nnodes}`,
    `  --node-rank <node-rank>`,
    `  --dist-init-addr <node0-ip>:20000`,
  ];

  const prependMultiNodeNote = (cmd, nnodes) =>
    `# Multi-node (${nnodes} nodes). Run the same command on every node with:\n` +
    `#   <node-rank> = 0 on the head node, 1..${nnodes - 1} on the others\n` +
    `#   <node0-ip>  = IP of the head node (reachable from all others)\n` +
    `${cmd}`;
  // === SHARED END ===

  const generateCommand = () => {
    const { hardware: rawHardware, modelSize, recipe, reasoningParser, toolcall } = values;
    // B300 usage is identical to B200 — alias so we don't duplicate every spec entry.
    const hardware = rawHardware === "b300" ? "b200" : rawHardware;
    const specKey = `${hardware}|${modelSize}`;
    const spec = HW_SIZE_SPEC[specKey];
    const { slug, tp, multinode, nnodes } = spec;
    const isBig = modelSize === "big";

    if (recipe === "pd-disagg") {
      return buildPDDisaggCommand(hardware, modelSize);
    }

    // ---- env ----
    // _LAUNCH_HEAD always prepends these:
    const COMMON_ENV = ["SGLANG_JIT_DEEPGEMM_PRECOMPILE=0"];
    // Per-hardware env (whitelist #1: NVSHMEM removed for B200).
    const HW_ENV = {
      h200:  ["SGLANG_DSV4_FP4_EXPERTS=0"],   // allinone _ENV_H200
      b200:  [],                              // _ENV_B200 minus NVSHMEM
      gb300: [],                              // _ENV_GB300
    }[hardware];

    // Recipe-specific env (matches allinone exactly, taking size into account).
    const recipeEnv = [];
    if (recipe === "low-latency") {
      // H200 big low-latency has extra dispatch-token cap (allinone line 233).
      if (hardware === "h200" && isBig) {
        recipeEnv.push("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128");
      }
    } else if (recipe === "balanced") {
      if (hardware === "h200") {
        recipeEnv.push("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256");
      } else {
        // Blackwell: small=1024, big=256 (allinone ternary).
        recipeEnv.push(isBig
          ? "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"
          : "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024");
      }
    } else if (recipe === "max-throughput") {
      if (hardware === "h200") {
        recipeEnv.push("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256");
      } else {
        recipeEnv.push(isBig
          ? "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"
          : "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024");
      }
    } else if (recipe === "cp") {
      recipeEnv.push("SGLANG_OPT_USE_JIT_INDEXER_METADATA=1");
      if (hardware === "h200") {
        recipeEnv.push("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024");
      } else {
        // Blackwell cp: small=1024, big=256 (allinone ternary).
        recipeEnv.push(isBig
          ? "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"
          : "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024");
      }
    }
    // SGLANG_ENABLE_SPEC_V2=1 was in allinone's _ENV_MTP for low-latency / balanced
    // recipes, but V4 auto-enables spec-v2 when MTP is detected — human confirmed
    // the env is redundant on the public cookbook path. Kept as a no-op reference
    // in allinone for legacy runs.

    // ---- flags ----
    const flags = [];
    flags.push("  --trust-remote-code");                              // _LAUNCH_HEAD
    flags.push(`  --model-path ${slug}`);

    if (recipe === "low-latency") {
      // allinone:
      //   H200 small: pure TP + MTP_314
      //   H200 big:   DP-attn + DeepEP + MTP_314 + cg=32 max-run=64 + multi-node + mem-frac 0.82
      //   Blackwell:  TP + flashinfer_mxfp4 + MTP_314 + chunked-prefill-size 4096 + autotune-fix
      //               Big Blackwell additionally: mem-frac 0.82
      flags.push(`  --tp ${tp}`);
      if (hardware === "h200" && isBig) {
        flags.push(`  --dp ${tp}`);
        flags.push("  --enable-dp-attention");
      }
      if (multinode) flags.push(...multiNodeFlags(nnodes));
      if (hardware === "h200" && isBig) {
        flags.push("  --moe-a2a-backend deepep");
      }
      if (hardware !== "h200") {
        flags.push("  --moe-runner-backend flashinfer_mxfp4");
      }
      if (hardware === "h200" && isBig) {
        flags.push("  --cuda-graph-max-bs 32");
        flags.push("  --max-running-requests 64");
      }
      // MTP 3/4
      flags.push("  --speculative-algo EAGLE");
      flags.push("  --speculative-num-steps 3");
      flags.push("  --speculative-eagle-topk 1");
      flags.push("  --speculative-num-draft-tokens 4");
      if (hardware !== "h200") {
        flags.push("  --chunked-prefill-size 4096");
        flags.push("  --disable-flashinfer-autotune");
      }
      if (isBig) flags.push("  --mem-fraction-static 0.82");
    } else if (recipe === "balanced") {
      // allinone balanced: TP + DP + DP-attn + DeepEP + MTP_112.
      //   H200 small: cg=128 max-run=128  |  H200 big: cg=128 max-run=128 (same)
      //   B200 small: no cg/max-run       |  B200 big: cg=64  max-run=128
      //   GB300 small: no cg/max-run      |  GB300 big: cg=128 max-run=256
      flags.push(`  --tp ${tp}`);
      flags.push(`  --dp ${tp}`);
      flags.push("  --enable-dp-attention");
      if (multinode) flags.push(...multiNodeFlags(nnodes));
      flags.push("  --moe-a2a-backend deepep");
      flags.push("  --speculative-algo EAGLE");
      flags.push("  --speculative-num-steps 1");
      flags.push("  --speculative-eagle-topk 1");
      flags.push("  --speculative-num-draft-tokens 2");
      if (isBig) flags.push("  --mem-fraction-static 0.82");
      if (hardware === "h200") {
        flags.push("  --cuda-graph-max-bs 128");
        flags.push("  --max-running-requests 128");
      } else if (isBig && hardware === "b200") {
        flags.push("  --cuda-graph-max-bs 64");
        flags.push("  --max-running-requests 128");
      } else if (isBig && hardware === "gb300") {
        flags.push("  --cuda-graph-max-bs 128");
        flags.push("  --max-running-requests 256");
      }
      // allinone H200 gates DEEPEP_LARGE_SMS_FLAG on !multinode — only H200 big
      // is multi-node; all Blackwell cells get the flag unconditionally.
      if (!multinode) flags.push(DEEPEP_LARGE_SMS_FLAG);
    } else if (recipe === "max-throughput") {
      // allinone max-throughput: TP + DP + DP-attn + DeepEP (NO MTP).
      //   H200 small: cg=128 max-run=256  |  H200 big: cg=128 max-run=256 (same)
      //   B200 small: no cg/max-run       |  B200 big: cg=64  max-run=256
      //   GB300 small: no cg/max-run      |  GB300 big: cg=128 max-run=256
      flags.push(`  --tp ${tp}`);
      flags.push(`  --dp ${tp}`);
      flags.push("  --enable-dp-attention");
      if (multinode) flags.push(...multiNodeFlags(nnodes));
      flags.push("  --moe-a2a-backend deepep");
      if (isBig) flags.push("  --mem-fraction-static 0.82");
      if (hardware === "h200") {
        flags.push("  --cuda-graph-max-bs 128");
        flags.push("  --max-running-requests 256");
      } else if (isBig && hardware === "b200") {
        flags.push("  --cuda-graph-max-bs 64");
        flags.push("  --max-running-requests 256");
      } else if (isBig && hardware === "gb300") {
        flags.push("  --cuda-graph-max-bs 128");
        flags.push("  --max-running-requests 256");
      }
      if (!multinode) flags.push(DEEPEP_LARGE_SMS_FLAG);
    } else if (recipe === "cp") {
      // allinone cp: TP (NO --dp) + DeepEP + _CP_FLAGS (mem-frac 0.78, max-run 1024).
      //   Blackwell big additionally: mem-frac 0.70 (overrides), cg=256, max-run=256.
      //   No flashinfer_mxfp4 even on Blackwell (allinone omits).
      flags.push(`  --tp ${tp}`);
      if (multinode) flags.push(...multiNodeFlags(nnodes));
      flags.push("  --moe-a2a-backend deepep");
      flags.push("  --enable-nsa-prefill-context-parallel");
      flags.push("  --nsa-prefill-cp-mode round-robin-split");
      flags.push("  --chunked-prefill-size 16384");
      // GB300 big CP needs higher mem-fraction-static: Pro 1.6T weights at
      // tp=4 are ~224 GB/card on a 273 GB GB300, so 0.78 leaves a negative
      // KV pool (init_memory_pool fails: "Not enough memory ... weights
      // 224 GB > static target 213 GB"). 0.88 gives weights 224 + KV 16 +
      // runtime 33. Other Blackwell tp=8 paths fit fine at 0.78.
      // Verified on 2026-04-25 (journal 2026-04-25-001 Cell B, Δ4).
      if (hardware === "gb300" && isBig) {
        flags.push("  --mem-fraction-static 0.88");
      } else {
        flags.push("  --mem-fraction-static 0.78");
      }
      // allinone _CP_FLAGS has --max-running-requests 1024; Blackwell big cp overrides
      // to 256. Human directed (2026-04-24) to emit only one value — keep 256 override
      // for big Blackwell, else the default 1024.
      if (isBig && hardware !== "h200") {
        flags.push("  --cuda-graph-max-bs 256");
        flags.push("  --max-running-requests 256");
      } else {
        flags.push("  --max-running-requests 1024");
      }
      // H200 CP gates DEEPEP_LARGE_SMS_FLAG on !multinode; Blackwell always gets it.
      if (!multinode) flags.push(DEEPEP_LARGE_SMS_FLAG);
    }

    // Optional parsers (cookbook UI extension; not in allinone — opt-in toggles only).
    if (toolcall === "enabled") flags.push("  --tool-call-parser deepseekv4");
    if (reasoningParser === "enabled") flags.push("  --reasoning-parser deepseek-v4");

    flags.push("  --host 0.0.0.0");
    flags.push("  --port 30000");

    // Assemble: [HW env] [recipe env] [common env] \ sglang serve \ flags...
    const envAll = [...HW_ENV, ...recipeEnv, ...COMMON_ENV];
    const envBlock = envAll.length ? envAll.join(" \\\n") + " \\\n" : "";
    const base = `${envBlock}sglang serve \\\n${flags.join(" \\\n")}`;
    const withMultinode = multinode ? prependMultiNodeNote(base, nnodes) : base;
    const verifyKey = `${hardware}|${modelSize}|${recipe}`;
    if (TBD_RECIPES.has(verifyKey)) return TBD_PLACEHOLDER;
    return VERIFIED_RECIPES.has(verifyKey)
      ? withMultinode
      : `${BEING_VERIFIED_NOTE}\n${commentOutCommand(withMultinode)}`;
  };

  // ============================================================================
  // buildPDDisaggCommand — mirror of allinone pd-p / pd-d for small AND big.
  //
  //   _PD_SPEC[(hw, size)] → tp (and whether multinode).
  //     H200-fp8 small: tp=4  single-node,  ib=mlx5_0
  //     H200-fp8 big:   tp=16 2-node,       ib=mlx5_0
  //     B200 small:     tp=2  single-node,  ib=mlx5_7
  //     B200 big:       tp=8  single-node,  ib=mlx5_7
  //     GB300 small/big: tp=4 single-node,  ib=""    (uses MNNVL, no IB device)
  //
  //   deepep flag only on Blackwell PD; H200 PD does NOT use deepep.
  //   cap_env (SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024) only on B200 decode.
  //   SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True only on GB300.
  //   --dist-init-addr for disagg wiring only on non-GB300.
  //   --max-running-requests 256 only on decode (PD decode can't retract).
  //   No flashinfer_mxfp4 / autotune-fix / MTP / mem-fraction-static on PD (allinone omits).
  // ============================================================================
  const buildPDDisaggCommand = (rawHardware, modelSize) => {
    // B300 usage is identical to B200 — alias so we don't duplicate every spec entry.
    const hardware = rawHardware === "b300" ? "b200" : rawHardware;
    const specKey = `${hardware}|${modelSize}`;
    const { tp: pdTp, multinode, nnodes } = PD_TP_SPEC[specKey];
    const slug = HW_SIZE_SPEC[specKey].slug;
    const ibDevice = { h200: "mlx5_0", b200: "mlx5_7", gb300: "" }[hardware];
    const isGB300 = hardware === "gb300";
    const isBlackwell = hardware === "b200" || isGB300;

    const HW_ENV = {
      h200:  ["SGLANG_DSV4_FP4_EXPERTS=0"],
      b200:  [],
      gb300: [],
    }[hardware];
    // Whitelist #5: only SGLANG_MOONCAKE_CUSTOM_MEM_POOL kept; MC_FORCE_MNNVL /
    // NCCL_MNNVL_ENABLE / NCCL_CUMEM_ENABLE may also be needed depending on the
    // GB300 cluster's NVLink/IB topology — see §3.2 "Configuration Tips" note.
    const MNNVL_ENV = isGB300 ? ["SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True"] : [];
    const COMMON_ENV = ["SGLANG_JIT_DEEPGEMM_PRECOMPILE=0"];

    const buildRole = (mode, port, distPort) => {
      const roleEnv = [];
      if (hardware === "b200" && mode === "decode") {
        roleEnv.push("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024");
      }
      // GB300 PD needs DeepEP dispatch buffer cap on BOTH prefill + decode;
      // without it, the first forward fails `deep_ep.cpp:1233` assertion
      // `x.size(0) <= num_max_dispatch_tokens_per_rank`. The cap also
      // co-moves with --max-running-requests below: 256 for big (which
      // uses --max-running-requests 128, per-rank=32 ≤ 256), 1024 for
      // small (--max-running-requests 256, per-rank=64 ≤ 1024).
      // Verified on 2026-04-25 (journal 2026-04-25-001 §C/§D).
      if (isGB300) {
        roleEnv.push(modelSize === "big"
          ? "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"
          : "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024");
      }
      const envAll = [...HW_ENV, ...roleEnv, ...MNNVL_ENV, ...COMMON_ENV];
      const envBlock = envAll.length ? envAll.join(" \\\n") + " \\\n" : "";

      const flags = [];
      flags.push("  --trust-remote-code");
      flags.push(`  --model-path ${slug}`);
      flags.push(`  --tp ${pdTp}`);
      flags.push(`  --dp ${pdTp}`);
      flags.push("  --enable-dp-attention");
      if (multinode) flags.push(...multiNodeFlags(nnodes));
      if (isBlackwell) flags.push("  --moe-a2a-backend deepep");
      flags.push(`  --disaggregation-mode ${mode}`);
      flags.push("  --disaggregation-transfer-backend mooncake");
      if (ibDevice) flags.push(`  --disaggregation-ib-device ${ibDevice}`);
      if (!isGB300) flags.push(`  --dist-init-addr 127.0.0.1:${distPort}`);
      if (mode === "decode") {
        // GB300 big PD decode is the most memory-pressured PD role: Pro 1.6T
        // weights at tp=4 take ~224 GB/card on a 273 GB GB300; runtime needs
        // headroom for DeepEP buffer + mooncake KV recv + CG private pool.
        // Cookbook defaults (mem-frac 0.874, cg_max_bs 512, max-running 256)
        // OOM during CG capture. Verified working on 2026-04-25 (journal
        // 2026-04-25-001 Cell D, Δ10).
        if (isGB300 && modelSize === "big") {
          flags.push("  --max-running-requests 128");
          flags.push("  --mem-fraction-static 0.83");
          flags.push("  --cuda-graph-max-bs 128");
        } else {
          flags.push("  --max-running-requests 256");
        }
      }
      flags.push("  --host 0.0.0.0");
      flags.push(`  --port ${port}`);

      return `${envBlock}sglang serve \\\n${flags.join(" \\\n")}`;
    };

    const prefillHeader = multinode
      ? `# --- Prefill role (port 30000) — multi-node, run on each of ${nnodes} nodes ---`
      : "# --- Prefill role (port 30000) ---";
    const decodeHeader = multinode
      ? `# --- Decode role (port 30001) — multi-node, run on each of ${nnodes} nodes ---`
      : "# --- Decode role (port 30001) ---";

    const prefill = `${prefillHeader}\n${buildRole("prefill", 30000, 30335)}`;
    const decode  = `${decodeHeader}\n${buildRole("decode",  30001, 30435)}`;
    // Router addresses prefill / decode by their reachable hostnames / IPs.
    // Substitute <prefill-host> / <decode-host> with the actual hosts before
    // running. On a same-host deployment, both can be 127.0.0.1.
    const router  = `# --- Router (port 8000) ---
python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:30000 \\
  --decode http://<decode-host>:30001 \\
  --host 0.0.0.0 --port 8000 \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`;

    const full = `${prefill}\n\n${decode}\n\n${router}`;
    const verifyKey = `${hardware}|${modelSize}|pd-disagg`;
    if (TBD_RECIPES.has(verifyKey)) return TBD_PLACEHOLDER;
    return VERIFIED_RECIPES.has(verifyKey)
      ? full
      : `${BEING_VERIFIED_NOTE}\n${commentOutCommand(full)}`;
  };

  // ---- styles ----
  const containerStyle = { maxWidth: "900px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "4px" };
  const cardStyle = {
    padding: "8px 12px",
    border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
    borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
    borderRadius: "4px",
    display: "flex",
    alignItems: "center",
    gap: "12px",
    background: isDark ? "#1f2937" : "#fff",
  };
  const titleStyle = { fontSize: "13px", fontWeight: "600", minWidth: "140px", flexShrink: 0, color: isDark ? "#e5e7eb" : "inherit" };
  const itemsStyle = { display: "flex", rowGap: "2px", columnGap: "6px", flexWrap: "wrap", alignItems: "center", flex: 1 };
  const labelBaseStyle = {
    padding: "4px 10px",
    border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
    borderRadius: "3px",
    cursor: "pointer",
    display: "inline-flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "500",
    fontSize: "13px",
    transition: "all 0.2s",
    userSelect: "none",
    minWidth: "45px",
    textAlign: "center",
    flex: 1,
    background: isDark ? "#374151" : "#fff",
    color: isDark ? "#e5e7eb" : "inherit",
  };
  const checkedStyle = { background: "#D45D44", color: "white", borderColor: "#D45D44" };
  const disabledStyle = { cursor: "not-allowed", opacity: 0.4 };
  const subtitleStyle = { display: "block", fontSize: "9px", marginTop: "1px", lineHeight: "1.1", opacity: 0.7 };
  const commandDisplayStyle = {
    flex: 1,
    padding: "12px 16px",
    background: isDark ? "#111827" : "#f5f5f5",
    borderRadius: "6px",
    fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    fontSize: "12px",
    lineHeight: "1.5",
    color: isDark ? "#e5e7eb" : "#374151",
    whiteSpace: "pre-wrap",
    overflowX: "auto",
    margin: 0,
    border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
  };

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        const items = resolveItems(option);
        return (
          <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {items.map((item) => {
                const isChecked = values[option.name] === item.id;
                const isDisabled = !!item.disabled;
                return (
                  <label
                    key={item.id}
                    style={{ ...labelBaseStyle, ...(isChecked ? checkedStyle : {}), ...(isDisabled ? disabledStyle : {}) }}
                    title={item.disabledReason || ""}
                  >
                    <input
                      type="radio"
                      name={option.name}
                      value={item.id}
                      checked={isChecked}
                      disabled={isDisabled}
                      onChange={() => !isDisabled && handleRadioChange(option.name, item.id)}
                      style={{ display: "none" }}
                    />
                    {item.label}
                    {item.subtitle && (
                      <small style={{ ...subtitleStyle, color: isChecked ? "rgba(255,255,255,0.85)" : "inherit" }}>
                        {item.subtitle}
                      </small>
                    )}
                  </label>
                );
              })}
            </div>
          </div>
        );
      })}
      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{generateCommand()}</pre>
      </div>
    </div>
  );
};
