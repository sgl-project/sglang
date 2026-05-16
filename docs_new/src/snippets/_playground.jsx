// Shared playground skeleton — the ENGINE half of the SGLang cookbook
// playground widget. Pair this with a per-model config file under
// `/src/snippets/configs/<vendor>/<model>.jsx` and an MDX page that imports
// both:
//
//     import { Playground } from "/src/snippets/_playground.jsx";
//     import { config }    from "/src/snippets/configs/deepseek-ai/deepseek-v4.jsx";
//     <Playground config={config} />
//
// Like the deployment skeleton, this engine reads from `config`:
//   - config.cells / modelNames / placeholders / curl / multiNodeHints /
//     dockerImages — same fields the deployment engine uses (the playground
//     reuses §3.1's base cell from the URL hash to seed its diff baseline).
//   - config.playgroundFeatures — opt-in feature axes. Each known axis is
//     rendered ONLY if its key is present in config. Six axes are recognised:
//       attention   : knobs[] with TP/DP/CP/DP-Attention sub-controls
//       moe         : backend.options[] + ep.values[]
//       parsers     : items[] (each emits one toggle flag)
//       speculative : options[] (single-select chip group)
//       pdDisagg    : modes[] + ibDevices[]
//       hicache     : backends[] + writePolicies[]
//     Each axis's widget type and strip/insert behaviour is fixed here in the
//     engine; config supplies labels, option values, and the actual flag
//     strings to emit per option (so model-specific things like the parser
//     slug or the MTP_314 numbers live in config, not here).
//
// Mintlify caveats this file routes around are the same as `_deployment.jsx`:
// module-level statements are stripped, custom JSX component imports are
// silently rebound — so everything stays inside this wrapper function and
// uses only lowercase HTML JSX tags.
export const Playground = ({ config }) => {
  if (!config) {
    return <div style={{padding: 12, color: "#b91c1c"}}>Playground: missing <code>config</code> prop</div>;
  }

  // ==========================================================================
  // Playground feature axes — read from config (opt-in by presence).
  // ==========================================================================
  // Each axis defaults to `undefined` if the per-model config doesn't supply
  // it. The render section below uses these as render guards: missing axis →
  // its card is omitted entirely (no empty placeholder).
  const pgFeatures = config.playgroundFeatures || {};
  const pgAttention   = pgFeatures.attention;
  const pgMoe         = pgFeatures.moe;
  const pgParsers     = pgFeatures.parsers;
  const pgSpec        = pgFeatures.speculative;
  const pgPdDisagg    = pgFeatures.pdDisagg;
  const pgHicache     = pgFeatures.hicache;

  // Convenience: a per-knob value list for the parallelism card. Look up by
  // knob id; fall back to `[null]` (just "auto") if the knob isn't declared.
  const attnValuesFor = (knobId) => {
    const k = (pgAttention?.knobs || []).find((kk) => kk.id === knobId);
    return k?.values || [null];
  };

  // ==========================================================================
  // Pure helpers (shape-identical to §3 where they overlap)
  // ==========================================================================
  const DIMENSIONS = ["hw", "variant", "quant", "strategy", "nodes"];
  const findCell = (cells, sel) =>
    cells.find((c) => DIMENSIONS.every((d) => c.match[d] === sel[d]));

  const resolveModelName = (sel) => {
    const triple = `${sel.hw}|${sel.variant}|${sel.quant}`;
    const pair = `${sel.variant}|${sel.quant}`;
    return config.modelNames[triple] ?? config.modelNames[pair] ?? "";
  };

  const interpolate = (text, env, modelName) =>
    text.replace(/{{(\w+)}}/g, (_, key) =>
      key === "MODEL_NAME" ? modelName : (env[key] ?? `{{${key}}}`));

  const parseNnodes = (id) => {
    if (id === "single") return 1;
    const m = /^multi-(\d+)$/.exec(id);
    return m ? parseInt(m[1], 10) : 1;
  };

  // Strip any flag whose first whitespace-delimited token equals one of
  // `prefixes`. Used to remove the base's values for an axis before re-emitting
  // the playground's choice. We must match exactly the first token because
  // values may contain hyphens / equals (e.g. `--moe-runner-backend marlin`).
  const stripFlagsByFirstToken = (flags, prefixes) => {
    const set = new Set(prefixes);
    return flags.filter((f) => {
      const head = f.split(/[\s=]/)[0];
      return !set.has(head);
    });
  };

  // Insert a list of new flags just before the trailing --host/--port pair so
  // the diff stays visually grouped with the existing structure.
  const insertBeforeTail = (flags, additions) => {
    const idx = flags.findIndex((f) => f.startsWith("--host"));
    const at = idx === -1 ? flags.length : idx;
    const out = flags.slice();
    out.splice(at, 0, ...additions);
    return out;
  };

  // Apply playground deltas on top of the base cell's flags.
  // Insert one or more new flags right after the FIRST flag whose first token
  // is in `afterAnyOf`. Falls back to right-after --model-path if none match.
  // Used so the playground's overrides land near their conceptual siblings
  // (e.g. --dp lands next to --tp).
  const insertAfter = (flags, afterAnyOf, additions) => {
    const set = new Set(afterAnyOf);
    let idx = flags.findIndex((f) => set.has(f.split(/[\s=]/)[0]));
    if (idx === -1) idx = flags.findIndex((f) => f.startsWith("--model-path"));
    const out = flags.slice();
    out.splice(idx + 1, 0, ...additions);
    return out;
  };

  const applyDeltas = (baseFlags, d, sel) => {
    let flags = [...baseFlags];

    // --- Attention parallelism overrides ---
    // Per-knob: only strip + re-emit knobs the user actually set. A null
    // override means "inherit from base" — so the base's flag for that knob
    // stays untouched.
    if (d.attn.tp !== null) {
      flags = stripFlagsByFirstToken(flags, ["--tp"]);
      flags = insertAfter(flags, ["--model-path"], [`--tp ${d.attn.tp}`]);
    }
    if (d.attn.dp !== null) {
      flags = stripFlagsByFirstToken(flags, ["--dp"]);
      if (d.attn.dp > 1) {
        flags = insertAfter(flags, ["--tp", "--model-path"], [`--dp ${d.attn.dp}`]);
      }
    }
    if (d.attn.dpAttn !== null) {
      flags = stripFlagsByFirstToken(flags, ["--enable-dp-attention"]);
      if (d.attn.dpAttn === true) {
        flags = insertAfter(flags, ["--dp", "--tp", "--model-path"],
          ["--enable-dp-attention"]);
      }
    }
    if (d.attn.cp !== null) {
      flags = stripFlagsByFirstToken(flags, [
        "--enable-nsa-prefill-context-parallel", "--nsa-prefill-cp-mode",
      ]);
      if (d.attn.cp > 1) {
        flags = insertAfter(flags,
          ["--enable-dp-attention", "--dp", "--tp", "--model-path"],
          ["--enable-nsa-prefill-context-parallel",
           "--nsa-prefill-cp-mode round-robin-split"]);
      }
    }

    // --- MoE backend (single-select) — config-driven option emit ---
    // The option's `flags` array IS the source of truth for which `--moe-*`
    // flag (and value) to insert. Engine just splices it after stripping the
    // base's MoE flags.
    if (d.moe.backend !== null) {
      flags = stripFlagsByFirstToken(flags, [
        "--moe-a2a-backend", "--moe-runner-backend",
      ]);
      const opt = (pgMoe?.backend?.options || []).find((o) => o.id === d.moe.backend);
      if (opt && opt.flags && opt.flags.length) {
        flags = insertAfter(flags,
          ["--enable-dp-attention", "--dp", "--tp", "--model-path"],
          opt.flags);
      }
    }
    // --- MoE EP knob ---
    if (d.moe.ep !== null) {
      flags = stripFlagsByFirstToken(flags, ["--ep"]);
      if (d.moe.ep > 1) {
        flags = insertAfter(flags,
          ["--moe-a2a-backend", "--moe-runner-backend", "--enable-dp-attention",
           "--dp", "--tp", "--model-path"],
          [`--ep ${d.moe.ep}`]);
      }
    }

    // --- Speculative decoding — config-driven option emit ---
    // Engine strips the 4 known --speculative-* prefixes and (if a non-current
    // option is picked) splices the option's `flags` array. "current" leaves
    // the base's spec flags untouched; "off" strips without re-adding.
    if (d.spec !== "current") {
      flags = stripFlagsByFirstToken(flags, [
        "--speculative-algo", "--speculative-num-steps",
        "--speculative-eagle-topk", "--speculative-num-draft-tokens",
      ]);
      const preset = (pgSpec?.options || []).find((p) => p.id === d.spec);
      if (preset && preset.flags && preset.flags.length) {
        flags = insertBeforeTail(flags, preset.flags);
      }
    }

    // --- Parsers (multi-toggle) — config-driven per-item flags ---
    // Engine strips both --reasoning-parser and --tool-call-parser
    // unconditionally (covers the case where the base cell already has
    // either parser configured). Then iterates the config's parser items;
    // for each item whose toggle is ON in `d.parsers`, splices `item.flag`.
    flags = stripFlagsByFirstToken(flags, ["--reasoning-parser", "--tool-call-parser"]);
    const parserAdds = [];
    for (const item of (pgParsers?.items || [])) {
      if (d.parsers && d.parsers[item.id]) parserAdds.push(item.flag);
    }
    if (parserAdds.length) flags = insertBeforeTail(flags, parserAdds);

    // --- PD Disaggregation (role-specific flags for one of prefill/decode) ---
    flags = stripFlagsByFirstToken(flags, [
      "--disaggregation-mode", "--disaggregation-transfer-backend",
      "--disaggregation-ib-device", "--disaggregation-bootstrap-port",
    ]);
    if (d.pd.mode === "prefill" || d.pd.mode === "decode") {
      const pdAdds = [
        `--disaggregation-mode ${d.pd.mode}`,
        "--disaggregation-transfer-backend mooncake",
      ];
      if (d.pd.ibDevice && d.pd.ibDevice !== "auto") {
        pdAdds.push(`--disaggregation-ib-device ${d.pd.ibDevice}`);
      }
      // Single-host bootstrap port (only when base isn't already multi-node —
      // the renderer adds NODE0_IP-based --dist-init-addr in that case and
      // adding a second one here would collide).
      if (sel.nodes === "single" && !flags.some((f) => f.startsWith("--dist-init-addr"))) {
        const bootstrapPort = d.pd.mode === "prefill" ? 30335 : 30435;
        pdAdds.push(`--dist-init-addr 127.0.0.1:${bootstrapPort}`);
      }
      flags = insertBeforeTail(flags, pdAdds);
    }

    // --- Hierarchical KV Cache ---
    // Emission follows the canonical upstream form documented in
    // docs/advanced_features/hicache_best_practices.mdx.
    flags = stripFlagsByFirstToken(flags, [
      "--enable-hierarchical-cache", "--hicache-ratio", "--hicache-size",
      "--hicache-write-policy", "--hicache-mem-layout", "--hicache-io-backend",
      "--hicache-storage-backend", "--hicache-storage-prefetch-policy",
    ]);
    if (d.hicache.enable) {
      const hcAdds = [
        "--enable-hierarchical-cache",
        "--hicache-ratio 2",
        "--hicache-size 0",
      ];
      if (d.hicache.backend) {
        hcAdds.push(
          "--hicache-mem-layout page_first_direct",
          "--hicache-io-backend direct",
        );
      }
      const writePolicy = d.hicache.writePolicy && d.hicache.writePolicy !== "auto"
        ? d.hicache.writePolicy
        : "write_through";
      hcAdds.push(`--hicache-write-policy ${writePolicy}`);
      if (d.hicache.backend) {
        hcAdds.push(
          `--hicache-storage-backend ${d.hicache.backend}`,
          "--hicache-storage-prefetch-policy wait_complete",
        );
      }
      flags = insertBeforeTail(flags, hcAdds);
    }

    return flags;
  };

  // Renderer (same shape as §3 — multi-node prepending, env block, hints).
  // `pdMode` is one of: null (skip banner), "prefill", "decode" — when present
  // a banner is prepended explaining that the emitted command is only ONE of
  // the three PD-Disagg roles (prefill + decode + router). The renderer
  // intentionally only emits the role itself; the operator pairs it with the
  // sibling role and a router invocation separately.
  // `mode` is "python" (bare `sglang serve`) or "docker" (wrapped in `docker run`
  // against the per-hardware image from config.dockerImages).
  const renderCommandLines = (cell, flags, sel, envValues, pdMode = null, mode = "python") => {
    const modelName = resolveModelName(sel);
    const nnodes = parseNnodes(sel.nodes);
    const multinode = nnodes > 1;
    let f = [...flags];
    if (multinode && !f.some((x) => x.startsWith("--nnodes"))) {
      const at = f.findIndex((x) => x.startsWith("--model-path")) + 1;
      f.splice(at, 0,
        `--nnodes ${nnodes}`,
        `--node-rank {{NODE_RANK}}`,
        `--dist-init-addr {{NODE0_IP}}:20000`);
    }
    let cmd;
    if (mode === "docker") {
      const image = (config.dockerImages && config.dockerImages[sel.hw]) || "lmsysorg/sglang:dev";
      const dockerLines = [
        "docker run --gpus all",
        "  --shm-size 32g",
        "  -p {{PORT}}:{{PORT}}",
        "  -v ~/.cache/huggingface:/root/.cache/huggingface",
        `  --env "HF_TOKEN={{HF_TOKEN}}"`,
        ...cell.env.map((e) => `  --env ${e}`),
        "  --ipc=host",
        `  ${image}`,
        "  sglang serve",
        ...f.map((x) => "    " + x),
      ];
      cmd = dockerLines.join(" \\\n");
    } else {
      const flagBlock = f.map((x) => "  " + x).join(" \\\n");
      const envBlock = cell.env.length ? cell.env.join(" \\\n") + " \\\n" : "";
      cmd = `${envBlock}sglang serve \\\n${flagBlock}`;
    }
    if (multinode && config.multiNodeHints && config.multiNodeHints[sel.hw]) {
      const hint = config.multiNodeHints[sel.hw]
        .map((line) => (line.length ? "# " + line : "#")).join("\n");
      cmd = `${hint}\n${cmd}`;
    }
    cmd = interpolate(cmd, envValues, modelName);
    if (multinode) {
      const header =
        `# Multi-node (${nnodes} nodes). Run the same command on every node with:\n` +
        `#   <node-rank> = 0 on the head node, 1..${nnodes - 1} on the others\n` +
        `#   <node0-ip>  = IP of the head node (reachable from all others)`;
      cmd = `${header}\n${cmd}`;
    }
    if (pdMode === "prefill" || pdMode === "decode") {
      const sibling = pdMode === "prefill" ? "decode" : "prefill";
      const banner =
        `# === PD Disaggregation: ${pdMode.toUpperCase()} role only ===\n` +
        `# This command runs the ${pdMode} server. To complete the topology,\n` +
        `# also run the ${sibling} role on its peer GPU host, then start the\n` +
        `# sglang_router with --pd-disaggregation pointing at both. See the\n` +
        `# §3.2 PD-Disagg notes above for IB device, ulimit, and MNNVL caveats.`;
      cmd = `${banner}\n${cmd}`;
    }
    return cmd;
  };

  // Line-level diff. Returns array of {line, kind: 'unchanged'|'added'|'removed'}.
  // Greedy LCS-like: walk both sides, emit `unchanged` when they agree,
  // `added` for playground-only, `removed` for base-only. This isn't optimal
  // but produces readable output when the two share most lines (the common case).
  const computeDiff = (baseStr, pgStr) => {
    const a = baseStr.split("\n");
    const b = pgStr.split("\n");
    const aSet = new Set(a);
    const bSet = new Set(b);
    let i = 0, j = 0;
    const out = [];
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j]) {
        out.push({ line: b[j], kind: "unchanged" });
        i++; j++;
      } else if (j < b.length && !aSet.has(b[j])) {
        out.push({ line: b[j], kind: "added" });
        j++;
      } else if (i < a.length && !bSet.has(a[i])) {
        out.push({ line: a[i], kind: "removed" });
        i++;
      } else if (i < a.length) {
        i++;
      } else {
        j++;
      }
    }
    return out;
  };

  const placeholderDefaults = (schema) => {
    const out = {};
    for (const [k, v] of Object.entries(schema || {})) out[k] = v.default ?? "";
    return out;
  };

  // Initial parser-toggle state derived from config: each item's id starts at
  // `false`. If the parsers axis is absent, the parsers state is just `{}`.
  const initialParsers = () => {
    const out = {};
    for (const item of (pgParsers?.items || [])) out[item.id] = false;
    return out;
  };

  // ==========================================================================
  // Style helper (mostly shared with §3 — adds diff-line colors)
  // ==========================================================================
  const makeStyles = (isDark) => ({
    container: { maxWidth: "900px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "8px" },
    card: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#A78BFA" : "#8B5CF6"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
    },
    cardRow: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#A78BFA" : "#8B5CF6"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      display: "flex", alignItems: "center", gap: "12px",
    },
    baseStrip: {
      padding: "8px 12px",
      borderRadius: "4px",
      background: isDark ? "#064e3b" : "#d1fae5",
      color: isDark ? "#a7f3d0" : "#065f46",
      fontSize: "12px",
      display: "flex", alignItems: "center", gap: "10px",
    },
    title: { fontSize: "13px", fontWeight: "600", color: isDark ? "#e5e7eb" : "inherit", marginBottom: "8px" },
    titleInline: { fontSize: "13px", fontWeight: "600", minWidth: "180px", flexShrink: 0, color: isDark ? "#e5e7eb" : "inherit" },
    rowFlex: { display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center", flex: 1 },
    cardStack: { display: "flex", flexDirection: "column", gap: "6px" },
    subRow: { display: "flex", alignItems: "center", gap: "10px" },
    subLabel: {
      fontSize: "11px",
      fontWeight: 600,
      color: isDark ? "#9ca3af" : "#6b7280",
      minWidth: "96px",
      flexShrink: 0,
      letterSpacing: "0.02em",
    },
    chipRow: { display: "flex", flexWrap: "wrap", gap: "6px", flex: 1 },
    chip: {
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px",
      cursor: "pointer",
      fontSize: "12px",
      userSelect: "none",
      background: isDark ? "#374151" : "#fff",
      color: isDark ? "#e5e7eb" : "inherit",
      minWidth: "44px",
      textAlign: "center",
    },
    chipChecked: { background: "#8B5CF6", color: "white", borderColor: "#8B5CF6" },
    chipDisabled: { cursor: "not-allowed", opacity: 0.4 },
    axisLabel: { fontSize: "11px", color: isDark ? "#9ca3af" : "#6b7280", marginRight: "6px" },
    commandWrap: {
      position: "relative",
      background: isDark ? "#111827" : "#f5f5f5",
      borderRadius: "6px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      overflow: "hidden",
    },
    commandHeader: {
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "6px 10px",
      borderBottom: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      background: isDark ? "#1f2937" : "#fafafa",
    },
    commandPre: {
      padding: "12px 16px",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontSize: "12px", lineHeight: "1.5",
      color: isDark ? "#e5e7eb" : "#374151",
      whiteSpace: "pre-wrap", overflowX: "auto", margin: 0,
    },
    diffLineUnchanged: { display: "block" },
    diffLineAdded: {
      display: "block",
      background: isDark ? "rgba(16,185,129,0.15)" : "rgba(16,185,129,0.18)",
      color: isDark ? "#a7f3d0" : "#065f46",
      borderLeft: `3px solid #10b981`,
      paddingLeft: "8px", marginLeft: "-8px",
    },
    diffLineRemoved: {
      display: "block",
      background: isDark ? "rgba(239,68,68,0.10)" : "rgba(239,68,68,0.10)",
      color: isDark ? "#fca5a5" : "#991b1b",
      textDecoration: "line-through",
      opacity: 0.7,
      borderLeft: `3px solid #ef4444`,
      paddingLeft: "8px", marginLeft: "-8px",
    },
    badge: {
      display: "inline-flex", alignItems: "center", gap: "6px",
      padding: "2px 8px", borderRadius: "10px",
      background: isDark ? "#78350f" : "#fef3c7",
      color: isDark ? "#fde68a" : "#92400e",
      fontSize: "11px", fontWeight: 600,
    },
    badgeDot: {
      width: "8px", height: "8px", borderRadius: "50%", background: "#f59e0b",
    },
    iconButton: {
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#374151",
      fontSize: "11px", fontWeight: 500, cursor: "pointer",
      display: "inline-flex", alignItems: "center", gap: "4px",
    },
    iconRow: { display: "inline-flex", gap: "6px" },
    runModeWrap: {
      display: "inline-flex",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "10px",
      overflow: "hidden",
      fontSize: "11px",
      fontWeight: 600,
      userSelect: "none",
    },
    runModeChip: (active) => ({
      padding: "2px 10px",
      cursor: "pointer",
      background: active ? (isDark ? "#1f2937" : "#fff") : "transparent",
      color: active ? (isDark ? "#e5e7eb" : "#111827") : (isDark ? "#9ca3af" : "#6b7280"),
      borderRight: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
    }),
    runModeChipLast: (active) => ({
      padding: "2px 10px",
      cursor: "pointer",
      background: active ? (isDark ? "#1f2937" : "#fff") : "transparent",
      color: active ? (isDark ? "#e5e7eb" : "#111827") : (isDark ? "#9ca3af" : "#6b7280"),
    }),
    headerLeft: { display: "inline-flex", alignItems: "center", gap: "8px" },
    modalBackdrop: {
      position: "fixed", inset: 0,
      background: "rgba(0,0,0,0.5)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 9999,
    },
    modalBox: {
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      borderRadius: "8px", padding: "20px",
      maxWidth: "720px", width: "92%", maxHeight: "85vh", overflowY: "auto",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
    },
    modalHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" },
    modalTitle: { fontSize: "15px", fontWeight: 600 },
    modalCloseBtn: {
      background: "transparent", border: "none", color: "inherit",
      fontSize: "20px", cursor: "pointer", padding: "0 6px", lineHeight: 1,
    },
    formField: { display: "flex", flexDirection: "column", gap: "4px", marginBottom: "10px" },
    formLabel: { fontSize: "12px", fontWeight: 500, color: isDark ? "#9ca3af" : "#4b5563" },
    formInput: {
      padding: "6px 10px", fontSize: "13px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#111827" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    },
    sectionHeading: {
      fontSize: "12px", fontWeight: 600, textTransform: "uppercase",
      letterSpacing: "0.04em",
      color: isDark ? "#9ca3af" : "#6b7280",
      margin: "12px 0 6px 0",
    },
    primaryBtn: {
      padding: "6px 14px", background: "#8B5CF6", color: "white",
      border: "none", borderRadius: "4px", cursor: "pointer",
      fontSize: "13px", fontWeight: 500,
    },
    resetBtn: {
      marginLeft: "auto",
      padding: "2px 8px",
      fontSize: "11px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "3px",
      background: "transparent",
      color: isDark ? "#9ca3af" : "#6b7280",
      cursor: "pointer",
    },
  });

  // ==========================================================================
  // State + effects
  // ==========================================================================
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const check = () => {
      const html = document.documentElement;
      setIsDark(
        html.classList.contains("dark") ||
          html.getAttribute("data-theme") === "dark" ||
          html.style.colorScheme === "dark"
      );
    };
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"],
    });
    return () => observer.disconnect();
  }, []);

  // Shared env store with §3 (same localStorage key) so HOST/PORT/etc. are
  // unified across the page.
  const STORAGE_KEY = "sglang-deploy-env";
  const [env, setEnv] = useState(() => placeholderDefaults(config.placeholders));
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        setEnv({ ...placeholderDefaults(config.placeholders), ...parsed });
      }
    } catch {}
  }, []);
  const saveEnv = (next) => {
    setEnv(next);
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch {}
  };

  // Base selection: live-link to §3 via custom event + URL hash fallback.
  const initialBaseFromHash = () => {
    const fallback = config.cells[0].match;
    if (typeof window === "undefined") return { ...fallback };
    const raw = window.location.hash.replace(/^#/, "");
    if (!raw) return { ...fallback };
    const params = new URLSearchParams(raw);
    const out = { ...fallback };
    params.forEach((value, key) => { if (key in out) out[key] = value; });
    return out;
  };
  const [base, setBase] = useState(() => initialBaseFromHash());
  useEffect(() => {
    const onHash = () => setBase(initialBaseFromHash());
    const onSelEvent = (e) => {
      const fallback = config.cells[0].match;
      const incoming = (e && e.detail) || {};
      const next = { ...fallback };
      for (const k of Object.keys(next)) {
        if (incoming[k] !== undefined) next[k] = incoming[k];
      }
      setBase(next);
    };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("sglang-deploy-sel", onSelEvent);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("sglang-deploy-sel", onSelEvent);
    };
  }, []);

  // Playground deltas. `null` = inherit from base for that knob.
  // Parser keys come from config.playgroundFeatures.parsers.items (each item's
  // id maps to a boolean toggle). Other axes have fixed shapes that mirror
  // the engine's strip/insert logic.
  const [deltas, setDeltas] = useState({
    attn: { tp: null, dp: null, cp: null, dpAttn: null },
    moe:  { backend: null, ep: null },
    spec: "current",
    parsers: initialParsers(),
    pd: { mode: "off", ibDevice: "auto" },
    hicache: { enable: false, backend: null, writePolicy: "auto" },
  });

  const [modal, setModal] = useState(null);
  useEffect(() => {
    if (modal === null) return;
    const onKey = (e) => { if (e.key === "Escape") setModal(null); };
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [modal]);

  const [copied, setCopied] = useState(false);
  const [curlCopied, setCurlCopied] = useState(false);
  const [envDraft, setEnvDraft] = useState(env);
  useEffect(() => { if (modal === "env") setEnvDraft(env); }, [modal, env]);
  const [runMode, setRunMode] = useState("python");

  // ==========================================================================
  // Derived
  // ==========================================================================
  const s = makeStyles(isDark);
  const baseCell = findCell(config.cells, base);
  const modelName = resolveModelName(base);

  let baseCommand = "";
  let playgroundCommand = "";
  let diffLines = [];
  if (baseCell) {
    baseCommand = renderCommandLines(baseCell, baseCell.flags, base, env, null, runMode);
    const pgFlags = applyDeltas(baseCell.flags, deltas, base);
    playgroundCommand = renderCommandLines(baseCell, pgFlags, base, env, deltas.pd.mode, runMode);
    diffLines = computeDiff(baseCommand, playgroundCommand);
  }

  const curlText = interpolate(config.curl || "", env, modelName);

  const resetAll = () => setDeltas({
    attn: { tp: null, dp: null, cp: null, dpAttn: null },
    moe:  { backend: null, ep: null },
    spec: "current",
    parsers: initialParsers(),
    pd: { mode: "off", ibDevice: "auto" },
    hicache: { enable: false, backend: null, writePolicy: "auto" },
  });

  const placeholderGroups = (() => {
    const out = { command: [], curl: [] };
    for (const [key, meta] of Object.entries(config.placeholders || {})) {
      (out[meta.target] || (out[meta.target] = [])).push({ key, ...meta });
    }
    return out;
  })();

  const handleCopy = () => {
    navigator.clipboard.writeText(playgroundCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  const copyCurl = () => {
    navigator.clipboard.writeText(curlText);
    setCurlCopied(true);
    setTimeout(() => setCurlCopied(false), 1200);
  };

  // ==========================================================================
  // JSX render
  // ==========================================================================
  // A row of chip selectors. `current` is the value bound to the chip group;
  // `onPick(v)` is called when the user clicks a chip. Disabled chips are
  // unclickable (used for placeholder spec algorithms).
  const renderChip = (label, current, value, onPick, opts = {}) => {
    const checked = current === value;
    const disabled = !!opts.disabled;
    return (
      <span
        key={`${label}-${value === null ? "auto" : value}`}
        style={{
          ...s.chip,
          ...(checked ? s.chipChecked : {}),
          ...(disabled ? s.chipDisabled : {}),
        }}
        title={disabled ? (opts.disabledReason || "Not available") : ""}
        onClick={() => { if (!disabled) onPick(value); }}
      >
        {label}
      </span>
    );
  };

  const setAttn   = (k, v) => setDeltas((d) => ({ ...d, attn: { ...d.attn, [k]: v } }));
  const setMoe    = (k, v) => setDeltas((d) => ({ ...d, moe:  { ...d.moe,  [k]: v } }));
  const setParser = (k, v) => setDeltas((d) => ({ ...d, parsers: { ...d.parsers, [k]: v } }));
  const setPd     = (k, v) => setDeltas((d) => ({ ...d, pd: { ...d.pd, [k]: v } }));
  const setHiCache = (k, v) => setDeltas((d) => ({ ...d, hicache: { ...d.hicache, [k]: v } }));

  // Format a hash-suffixed badge of the inherited base.
  const baseSummary = baseCell
    ? `${base.hw.toUpperCase()} · ${base.variant} · ${base.quant.toUpperCase()} · ${base.strategy} · ${base.nodes}`
    : "(no verified cell at current §3 selection — showing playground only)";

  // Look up a knob's value range. Used by Attention's TP/DP/CP/DP-Attention
  // chip rows. DP-Attention's chips are rendered manually below because the
  // values include booleans (need custom labels), not just numbers.
  const tpValues = attnValuesFor("tp");
  const dpValues = attnValuesFor("dp");
  const cpValues = attnValuesFor("cp");
  const dpAttnKnob = (pgAttention?.knobs || []).find((k) => k.id === "dpAttn");
  const epValues = pgMoe?.ep?.values || [null];

  return (
    <div style={s.container} className="not-prose">
      {/* Inherited base summary */}
      <div style={s.baseStrip}>
        <span style={{ fontWeight: 600 }}>Inherited base from §3:</span>
        <code style={{ fontFamily: "Menlo, monospace" }}>{baseSummary}</code>
        <button style={s.resetBtn} onClick={resetAll}>Reset all overrides</button>
      </div>

      {/* Axis 1: Attention Parallelism — only renders if config declares it */}
      {pgAttention && (
        <div style={{ ...s.card, ...s.cardStack }}>
          <div style={s.title}>Attention Parallelism</div>
          {(pgAttention.knobs || []).some((k) => k.id === "tp") && (
            <div style={s.subRow}>
              <span style={s.subLabel}>TP</span>
              <div style={s.chipRow}>
                {tpValues.map((v) =>
                  renderChip(v === null ? "auto" : String(v), deltas.attn.tp, v,
                    (nv) => setAttn("tp", nv)))}
              </div>
            </div>
          )}
          {(pgAttention.knobs || []).some((k) => k.id === "dp") && (
            <div style={s.subRow}>
              <span style={s.subLabel}>DP</span>
              <div style={s.chipRow}>
                {dpValues.map((v) =>
                  renderChip(v === null ? "auto" : String(v), deltas.attn.dp, v,
                    (nv) => setAttn("dp", nv)))}
              </div>
            </div>
          )}
          {(pgAttention.knobs || []).some((k) => k.id === "cp") && (
            <div style={s.subRow}>
              <span style={s.subLabel}>CP</span>
              <div style={s.chipRow}>
                {cpValues.map((v) =>
                  renderChip(v === null ? "auto" : String(v), deltas.attn.cp, v,
                    (nv) => setAttn("cp", nv)))}
              </div>
            </div>
          )}
          {dpAttnKnob && (
            <div style={s.subRow}>
              <span style={s.subLabel}>{dpAttnKnob.label || "DP-Attention"}</span>
              <div style={s.chipRow}>
                {(dpAttnKnob.values || [null, true, false]).map((v) => {
                  const labelMap = dpAttnKnob.labels || { "auto": "auto", "true": "on", "false": "off" };
                  const k = v === null ? "auto" : String(v);
                  return renderChip(labelMap[k] || k, deltas.attn.dpAttn, v,
                    (nv) => setAttn("dpAttn", nv));
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Axis 2: MoE Parallelism — Backend row + EP row */}
      {pgMoe && (
        <div style={{ ...s.card, ...s.cardStack }}>
          <div style={s.title}>MoE Parallelism</div>
          {pgMoe.backend && (
            <div style={s.subRow}>
              <span style={s.subLabel}>Backend</span>
              <div style={s.chipRow}>
                {(pgMoe.backend.options || []).map((o) =>
                  renderChip(o.label, deltas.moe.backend, o.id, (v) => setMoe("backend", v)))}
              </div>
            </div>
          )}
          {pgMoe.ep && (
            <div style={s.subRow}>
              <span style={s.subLabel}>{pgMoe.ep.label || "EP"}</span>
              <div style={s.chipRow}>
                {epValues.map((v) =>
                  renderChip(v === null ? "auto" : String(v), deltas.moe.ep, v,
                    (nv) => setMoe("ep", nv)))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Axis 3: Parsers — one toggle row per config-supplied item */}
      {pgParsers && (pgParsers.items || []).length > 0 && (
        <div style={{ ...s.card, ...s.cardStack }}>
          <div style={s.title}>Parsers</div>
          {(pgParsers.items || []).map((item) => (
            <div key={item.id} style={s.subRow}>
              <span style={s.subLabel}>{item.label}</span>
              <div style={s.chipRow}>
                {renderChip(item.label, deltas.parsers[item.id], true,
                  () => setParser(item.id, !deltas.parsers[item.id]))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Axis 4: Speculative Decoding — single-select chip group */}
      {pgSpec && (pgSpec.options || []).length > 0 && (
        <div style={s.card}>
          <div style={s.title}>Speculative Decoding</div>
          <div style={s.rowFlex}>
            {(pgSpec.options || []).map((p) =>
              renderChip(p.label, deltas.spec, p.id,
                (v) => setDeltas((d) => ({ ...d, spec: v })),
                { disabled: p.disabled, disabledReason: p.disabledReason }))}
          </div>
        </div>
      )}

      {/* Axis 5: PD Disaggregation — Mode row + IB Device row */}
      {pgPdDisagg && (
        <div style={{ ...s.card, ...s.cardStack }}>
          <div style={s.title}>PD Disaggregation</div>
          {(pgPdDisagg.modes || []).length > 0 && (
            <div style={s.subRow}>
              <span style={s.subLabel}>Mode</span>
              <div style={s.chipRow}>
                {pgPdDisagg.modes.map((m) =>
                  renderChip(m.label, deltas.pd.mode, m.id, (v) => setPd("mode", v)))}
              </div>
            </div>
          )}
          {(pgPdDisagg.ibDevices || []).length > 0 && (
            <div style={s.subRow}>
              <span style={s.subLabel}>IB Device</span>
              <div style={s.chipRow}>
                {pgPdDisagg.ibDevices.map((v) =>
                  renderChip(v, deltas.pd.ibDevice, v, (nv) => setPd("ibDevice", nv)))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Axis 6: Hierarchical KV Cache — Enable row + Storage row + Write Policy row */}
      {pgHicache && (
        <div style={{ ...s.card, ...s.cardStack }}>
          <div style={s.title}>Hierarchical KV Cache (HiCache)</div>
          <div style={s.subRow}>
            <span style={s.subLabel}>Enable</span>
            <div style={s.chipRow}>
              {renderChip("HiCache", deltas.hicache.enable, true,
                () => setHiCache("enable", !deltas.hicache.enable))}
            </div>
          </div>
          {(pgHicache.backends || []).length > 0 && (
            <div style={s.subRow}>
              <span style={s.subLabel}>Storage</span>
              <div style={s.chipRow}>
                {pgHicache.backends.map((o) =>
                  renderChip(o.label, deltas.hicache.backend, o.id,
                    (v) => setHiCache("backend", v)))}
              </div>
            </div>
          )}
          {(pgHicache.writePolicies || []).length > 0 && (
            <div style={s.subRow}>
              <span style={s.subLabel}>Write Policy</span>
              <div style={s.chipRow}>
                {pgHicache.writePolicies.map((p) =>
                  renderChip(p, deltas.hicache.writePolicy, p,
                    (v) => setHiCache("writePolicy", v)))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Command box */}
      <div style={s.card}>
        <div style={s.title}>Playground Command (diff vs verified base)</div>
        <div style={s.commandWrap}>
          <div style={s.commandHeader}>
            <div style={s.headerLeft}>
              <div style={s.badge}>
                <span style={s.badgeDot} />
                Auto-Estimated
              </div>
              <div style={s.runModeWrap} role="tablist" aria-label="Output format">
                <span
                  style={s.runModeChip(runMode === "python")}
                  onClick={() => setRunMode("python")}
                  role="tab"
                  aria-selected={runMode === "python"}
                >
                  Python
                </span>
                <span
                  style={s.runModeChipLast(runMode === "docker")}
                  onClick={() => setRunMode("docker")}
                  role="tab"
                  aria-selected={runMode === "docker"}
                >
                  Docker
                </span>
              </div>
            </div>
            <div style={s.iconRow}>
              <button style={s.iconButton} onClick={handleCopy}>
                {copied ? "✓ Copied" : "⧉ Copy"}
              </button>
              <button style={s.iconButton} onClick={() => setModal("curl")}>$ cURL</button>
              <button style={s.iconButton} onClick={() => setModal("env")}>⚙ Env</button>
            </div>
          </div>
          <pre style={s.commandPre}>
            {baseCell ? diffLines.map((d, i) => (
              <span
                key={i}
                style={
                  d.kind === "added" ? s.diffLineAdded :
                  d.kind === "removed" ? s.diffLineRemoved :
                  s.diffLineUnchanged
                }
              >
                {d.kind === "added" ? "+ " : d.kind === "removed" ? "- " : "  "}
                {d.line}{"\n"}
              </span>
            )) : "# No verified base cell at the current §3 selection.\n# Pick a supported hardware/variant in §3 to populate the playground base."}
          </pre>
        </div>
      </div>

      {/* cURL modal */}
      {modal === "curl" && (
        <div style={s.modalBackdrop} onClick={() => setModal(null)}>
          <div style={s.modalBox} onClick={(e) => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <div style={s.modalTitle}>cURL example</div>
              <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
            </div>
            <div style={s.commandWrap}>
              <div style={s.commandHeader}>
                <div style={{ fontSize: 11, opacity: 0.7 }}>
                  Model: <code>{modelName || "(unresolved)"}</code>
                </div>
                <button style={s.iconButton} onClick={copyCurl}>
                  {curlCopied ? "✓ Copied" : "⧉ Copy"}
                </button>
              </div>
              <pre style={s.commandPre}>{curlText}</pre>
            </div>
            <p style={{ fontSize: 11, opacity: 0.7, marginTop: 8 }}>
              Edit <code>CURL_HOST</code> / <code>CURL_PORT</code> in the Env panel.
            </p>
          </div>
        </div>
      )}

      {/* Env modal */}
      {modal === "env" && (
        <div style={s.modalBackdrop} onClick={() => setModal(null)}>
          <div style={s.modalBox} onClick={(e) => e.stopPropagation()}>
            <div style={s.modalHeader}>
              <div style={s.modalTitle}>Env / placeholder values</div>
              <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
            </div>
            {placeholderGroups.curl.length > 0 && (
              <div>
                <div style={s.sectionHeading}>cURL placeholders</div>
                {placeholderGroups.curl.map(({ key, label }) => (
                  <div key={key} style={s.formField}>
                    <label style={s.formLabel}>
                      {label} <code style={{ opacity: 0.6 }}>{`{{${key}}}`}</code>
                    </label>
                    <input
                      style={s.formInput}
                      value={envDraft[key] ?? ""}
                      onChange={(e) => setEnvDraft({ ...envDraft, [key]: e.target.value })}
                    />
                  </div>
                ))}
              </div>
            )}
            {placeholderGroups.command.length > 0 && (
              <div>
                <div style={s.sectionHeading}>Command placeholders</div>
                {placeholderGroups.command.map(({ key, label }) => (
                  <div key={key} style={s.formField}>
                    <label style={s.formLabel}>
                      {label} <code style={{ opacity: 0.6 }}>{`{{${key}}}`}</code>
                    </label>
                    <input
                      style={s.formInput}
                      value={envDraft[key] ?? ""}
                      onChange={(e) => setEnvDraft({ ...envDraft, [key]: e.target.value })}
                    />
                  </div>
                ))}
              </div>
            )}
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 16 }}>
              <button style={{ ...s.iconButton, padding: "6px 14px" }} onClick={() => setModal(null)}>Cancel</button>
              <button style={s.primaryBtn} onClick={() => { saveEnv(envDraft); setModal(null); }}>Save</button>
            </div>
            <p style={{ fontSize: 11, opacity: 0.7, marginTop: 10 }}>
              Values persist in localStorage and are shared with §3.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
