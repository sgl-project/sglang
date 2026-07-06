// ENGINE half of the SGLang cookbook deployment-command generator. Reads a
// per-model `config` prop (no model-specific code here). Full field semantics
// (resolution rules, key layering) live in the cookbook-add-model skill:
// .claude/skills/cookbook-add-model/references/authoring-reference.md.
//
// Config fields the engine reads:
//   modelName          display label
//   supportedHardware  hw ids shown in the catalog (subset of HARDWARE_CATALOG ∪ config.hardware)
//   hardware           optional — per-model GPUs the shared HARDWARE_CATALOG lacks:
//                      {id, label, vram, vendor}[] merged into the catalog at render
//                      (so a model-specific GPU never needs an engine-catalog edit)
//   variants/quantizations/strategies/nodesOptions  the 5-dim option lists
//                      (nodesOptions id is `single` or `multi-N` → --nnodes N)
//   cells              {match, verified?, env, flags}[] — one per
//                      (hw × variant × quant × strategy × nodes); env/flags are
//                      flat literals, only {{PLACEHOLDER}} subst applied
//   modelNames         HF slug lookup, `hw|variant|quant` then `variant|quant`
//   placeholders       {{KEY}} → {target: 'command'|'curl', label, default?}
//   curl               cURL template (uses {{MODEL_NAME}} + placeholders)
//   benchmarkCommands  optional — powers the "⚡ Reproduce" modal (speed +
//                      per-eval accuracy templates)
//   defaultAccuracy    optional — per-variant accuracy merged under cell.accuracy
//   accuracyLabels     [key, label, unit][] — the eval set shown in the
//                      benchmark card + "⚡ Reproduce". NO engine default:
//                      required whenever benchmarks carry accuracy data
//   latencyPercentile  optional, TEMPORARY — "Mean" | "P50" (default "P50"); the
//                      percentile the TTFT/TPOT values are, shown as "TTFT (<pct>)".
//                      Legacy "Mean" data is being re-measured to P50; drop once done
//   multiNodeHints     optional — {[hwId]: string[]} prepended as `# ...` lines
//   dockerImages       optional — `docker run` image, keyed by `hw|quant`
//                      then `hw`; falls back to `lmsysorg/sglang:dev`
//   github             optional — "Submit verified cell" issue-template overrides
//   playgroundFeatures optional — consumed by _playground.jsx (see its header)
//
// Mintlify caveats this file routes around:
//   - Module-level statements are stripped — everything lives inside the
//     wrapper function body.
//   - Capitalized JSX tags get rebound by _provideComponents() — lowercase
//     HTML tags only; factor into helper functions, not sub-components.
//   - Import plain-data config from the MDX file, pass through as a prop.

export const Deployment = ({ config, benchmarks }) => {
  if (!config) {
    return <div style={{padding: 12, color: "#b91c1c"}}>Deployment: missing <code>config</code> prop</div>;
  }

  // ==== 1. Hardware catalog (shared across cookbooks) ====
  // VRAM is per-GPU on-chip memory, not per-module.
  const HARDWARE_CATALOG = {
    nvidia: [
      { id: "h100",  label: "H100",  vram: "80GB"  },
      { id: "h200",  label: "H200",  vram: "141GB" },
      { id: "b200",  label: "B200",  vram: "192GB" },
      { id: "b300",  label: "B300",  vram: "288GB" },
      { id: "gb200", label: "GB200", vram: "192GB" },
      { id: "gb300", label: "GB300", vram: "288GB" },
    ],
    amd: [
      { id: "mi300x", label: "MI300X", vram: "192GB" },
      { id: "mi325x", label: "MI325X", vram: "256GB" },
      { id: "mi350x", label: "MI350X", vram: "288GB" },
      { id: "mi355x", label: "MI355X", vram: "288GB" },
    ],
  };

  // ==== 2. Style helper (dark-mode-aware) ====
  const makeStyles = (isDark) => ({
    container: { maxWidth: "900px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "3px" },
    card: {
      padding: "5px 10px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      display: "flex", alignItems: "center", gap: "10px",
      background: isDark ? "#1f2937" : "#fff",
    },
    cardColumn: {
      padding: "5px 10px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      display: "flex", flexDirection: "column", gap: "4px",
      background: isDark ? "#1f2937" : "#fff",
    },
    title: { fontSize: "12px", fontWeight: "600", minWidth: "108px", flexShrink: 0, color: isDark ? "#e5e7eb" : "inherit" },
    vendorRow: { display: "flex", alignItems: "center", gap: "6px" },
    vendorLabel: {
      fontSize: "10px", fontWeight: "600",
      color: isDark ? "#9ca3af" : "#6b7280",
      minWidth: "38px", textTransform: "uppercase", letterSpacing: "0.04em",
    },
    // auto-fit + a real min width: columns wrap on narrow screens instead of
    // shrinking below their label (the old minmax(0,1fr) let buttons overlap on
    // mobile). `cols` no longer needed — auto-fit never exceeds the item count.
    itemsGrid: () => ({
      display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(72px, 1fr))",
      gap: "4px", flex: 1,
    }),
    labelBase: {
      padding: "2px 8px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px", cursor: "pointer",
      display: "inline-flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      fontWeight: "500", fontSize: "12px",
      transition: "all 0.2s", userSelect: "none",
      minHeight: "26px", textAlign: "center",
      background: isDark ? "#374151" : "#fff",
      color: isDark ? "#e5e7eb" : "inherit",
    },
    checked: { background: "#D45D44", color: "white", borderColor: "#D45D44" },
    disabled: { cursor: "not-allowed", opacity: 0.4 },
    subtitle: { display: "block", fontSize: "9px", marginTop: "1px", lineHeight: "1.1", opacity: 0.7 },
    commandWrap: {
      position: "relative", flex: 1,
      background: isDark ? "#111827" : "#f5f5f5",
      borderRadius: "6px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      overflow: "hidden",
    },
    commandHeader: {
      display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center",
      gap: "6px 10px",
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
    // Amber callout under the command when speculative decoding (MTP) is on
    // but --max-running-requests isn't set (SGLang then caps it at 48).
    mtpWarn: {
      margin: "8px 0 0", padding: "8px 12px", borderRadius: "8px",
      fontSize: "12px", lineHeight: "1.45",
      background: isDark ? "#78350f" : "#fef3c7",
      color: isDark ? "#fde68a" : "#92400e",
      border: `1px solid ${isDark ? "#92400e" : "#fcd34d"}`,
    },
    badge: (verified) => ({
      display: "inline-flex", alignItems: "center", gap: "6px",
      padding: "2px 8px", borderRadius: "10px",
      background: verified ? (isDark ? "#064e3b" : "#d1fae5")
                           : (isDark ? "#78350f" : "#fef3c7"),
      color:      verified ? (isDark ? "#a7f3d0" : "#065f46")
                           : (isDark ? "#fde68a" : "#92400e"),
      fontSize: "11px", fontWeight: 600,
    }),
    badgeDot: (verified) => ({
      width: "8px", height: "8px", borderRadius: "50%",
      background: verified ? "#10b981" : "#f59e0b",
    }),
    iconButton: {
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#374151",
      fontSize: "11px", fontWeight: 500, cursor: "pointer",
      display: "inline-flex", alignItems: "center", gap: "4px",
    },
    iconRow: { display: "inline-flex", flexWrap: "wrap", gap: "6px" },
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
      background: active
        ? (isDark ? "#1f2937" : "#fff")
        : "transparent",
      color: active
        ? (isDark ? "#e5e7eb" : "#111827")
        : (isDark ? "#9ca3af" : "#6b7280"),
      borderRight: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
    }),
    runModeChipLast: (active) => ({
      padding: "2px 10px",
      cursor: "pointer",
      background: active
        ? (isDark ? "#1f2937" : "#fff")
        : "transparent",
      color: active
        ? (isDark ? "#e5e7eb" : "#111827")
        : (isDark ? "#9ca3af" : "#6b7280"),
    }),
    headerLeft: { display: "inline-flex", flexWrap: "wrap", alignItems: "center", gap: "8px" },
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
      padding: "6px 14px", background: "#D45D44", color: "white",
      border: "none", borderRadius: "4px", cursor: "pointer",
      fontSize: "13px", fontWeight: 500,
    },

    benchCard: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      display: "flex", flexDirection: "column", gap: "8px",
    },
    benchHeader: {
      display: "flex", flexWrap: "wrap", alignItems: "baseline", justifyContent: "space-between",
      gap: "6px 12px",
    },
    benchTitle: {
      fontSize: "13px", fontWeight: 600,
      color: isDark ? "#e5e7eb" : "inherit",
    },
    benchVersion: {
      fontSize: "11px",
      color: isDark ? "#9ca3af" : "#6b7280",
    },
    benchHeaderRight: {
      display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 10px", flexShrink: 0,
    },
    benchChipRow: {
      display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap",
      margin: "2px 0 8px",
    },
    benchChip: {
      padding: "2px 10px", fontSize: "12px", cursor: "pointer",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#374151",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    },
    benchChipActive: { background: "#D45D44", color: "white", borderColor: "#D45D44" },
    benchBlock: {
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderRadius: "4px",
      padding: "8px 10px",
      background: isDark ? "#111827" : "#fafafa",
    },
    benchBlockTitle: {
      fontSize: "11px", fontWeight: 600, textTransform: "uppercase",
      letterSpacing: "0.04em",
      color: isDark ? "#9ca3af" : "#6b7280",
      marginBottom: "4px",
    },
    benchWorkload: {
      fontSize: "11px", fontStyle: "italic",
      color: isDark ? "#9ca3af" : "#6b7280",
      marginBottom: "6px",
      lineHeight: "1.3",
    },
    benchRow: {
      display: "flex", justifyContent: "space-between",
      fontSize: "12px", padding: "2px 0",
    },
    benchKey: { color: isDark ? "#9ca3af" : "#6b7280" },
    benchVal: {
      color: isDark ? "#e5e7eb" : "#111827",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontWeight: 500,
    },
    benchNotes: {
      fontSize: "11px", fontStyle: "italic",
      color: isDark ? "#9ca3af" : "#6b7280",
    },
    benchLegend: {
      fontSize: "10px", fontStyle: "italic",
      color: isDark ? "#6b7280" : "#9ca3af",
      marginTop: "6px",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    },
    benchEmpty: {
      fontSize: "12px", fontStyle: "italic",
      color: isDark ? "#9ca3af" : "#6b7280",
    },

    // grid (not <table>) — Mintlify wraps <table> with scroll wrappers.
    // gridTemplateColumns set inline (depends on measurements.length).
    benchTable: {
      display: "grid",
      // columnGap 0 so cells' bottom borders form one continuous line.
      columnGap: 0,
      rowGap: "3px",
      marginTop: "4px",
      alignItems: "baseline",
    },
    benchTableHead: {
      textAlign: "right",
      fontWeight: 500, fontSize: "11px",
      color: isDark ? "#9ca3af" : "#6b7280",
      paddingLeft: "16px",
      paddingBottom: "4px",
      whiteSpace: "nowrap",
    },
    benchTableCornerHead: {
      paddingBottom: "4px",
    },
    // Header underline — one div spanning all columns (continuous line).
    benchTableSeparator: {
      gridColumn: "1 / -1",
      height: "1px",
      background: isDark ? "#374151" : "#e5e7eb",
      marginTop: "-3px", // negate the rowGap so it hugs the header row
    },
    benchTableLabel: {
      textAlign: "left", fontSize: "12px",
      color: isDark ? "#9ca3af" : "#6b7280",
      whiteSpace: "nowrap",
    },
    benchTableValue: {
      textAlign: "right", fontSize: "12px",
      color: isDark ? "#e5e7eb" : "#111827",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontWeight: 500,
      paddingLeft: "16px",
      whiteSpace: "nowrap",
    },
    benchTableValueMissing: {
      color: isDark ? "#6b7280" : "#9ca3af",
    },
  });

  // ==== 3. Pure helpers (no React state) ====
  // DIMENSIONS is ordered by priority — higher-index dims adapt to lower-index
  // picks, never the reverse. Drives the grey-out/snap logic below.
  const DIMENSIONS = ["hw", "variant", "quant", "strategy", "nodes"];
  const findCell = (cells, sel) =>
    cells.find((c) => DIMENSIONS.every((d) => c.match[d] === sel[d]));

  const findBenchmark = (list, sel) =>
    (list || []).find((b) => DIMENSIONS.every((d) => b.match[d] === sel[d])) || null;

  // Accepts a single measurement object or an array; always returns an array.
  const normalizeSpeed = (speed) => {
    if (!speed) return [];
    return Array.isArray(speed) ? speed : [speed];
  };

  // Variant default accuracy merged UNDER per-cell measured accuracy — but ONLY when
  // a benchmark entry exists for the cell. A cell with no entry was never measured, so
  // it shows the empty/"pending" state instead of borrowing the variant's accuracy.
  const effectiveAccuracy = (entry, sel) =>
    entry
      ? {
          ...((config.defaultAccuracy && config.defaultAccuracy[sel.variant]) || {}),
          ...(entry.accuracy || {}),
        }
      : {};

  // Empty = every speed measurement null-only AND accuracy null-only. `workload`
  // is metadata, not a measurement — skip it so a workload-only stub stays empty.
  const benchmarkIsEmpty = (entry, accuracy) => {
    for (const m of normalizeSpeed(entry && entry.speed)) {
      if (m && typeof m === "object") {
        for (const [key, v] of Object.entries(m)) {
          if (key === "workload") continue;
          if (v !== null && v !== undefined) return false;
        }
      }
    }
    if (accuracy && typeof accuracy === "object") {
      for (const v of Object.values(accuracy)) {
        if (v !== null && v !== undefined) return false;
      }
    }
    return true;
  };

  // Grey-out predicate: (dim, value) is enabled iff some cell matches every
  // HIGHER-priority dim in `sel` AND has dim === value. Lower dims may differ
  // (snapToValidCell adapts them on click).
  const isOptionAvailable = (cells, sel, dim, value) => {
    const idx = DIMENSIONS.indexOf(dim);
    const higher = DIMENSIONS.slice(0, idx);
    return cells.some(
      (c) => c.match[dim] === value && higher.every((d) => c.match[d] === sel[d]),
    );
  };

  // Snap to a real cell on click: higher dims stay locked, `dim` := value,
  // lower dims adopt the best-fit cell (the one preserving the most lower picks).
  const snapToValidCell = (cells, sel, dim, value) => {
    const idx = DIMENSIONS.indexOf(dim);
    const higher = DIMENSIONS.slice(0, idx);
    const lower = DIMENSIONS.slice(idx + 1);
    let best = null, bestLowerMatches = -1;
    for (const c of cells) {
      if (c.match[dim] !== value) continue;
      if (!higher.every((d) => c.match[d] === sel[d])) continue;
      let s = 0;
      for (const d of lower) if (c.match[d] === sel[d]) s++;
      if (s > bestLowerMatches) { bestLowerMatches = s; best = c; }
    }
    if (!best) return sel; // defensive — shouldn't be reachable
    const next = { ...sel, [dim]: value };
    for (const d of lower) next[d] = best.match[d];
    return next;
  };

  // Snap a parsed (possibly stale) URL-hash selection to a real cell, walking
  // dims in priority order and falling back per-dim to the first consistent cell.
  const validateSelection = (cells, parsed) => {
    const valid = {};
    for (const dim of DIMENSIONS) {
      const want = parsed[dim];
      const works = cells.some(
        (c) =>
          c.match[dim] === want &&
          DIMENSIONS.slice(0, DIMENSIONS.indexOf(dim)).every((d) => c.match[d] === valid[d]),
      );
      if (works) {
        valid[dim] = want;
      } else {
        const fallback = cells.find((c) =>
          DIMENSIONS.slice(0, DIMENSIONS.indexOf(dim)).every((d) => c.match[d] === valid[d]),
        );
        valid[dim] = fallback ? fallback.match[dim] : want;
      }
    }
    return valid;
  };

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

  // python mode → bare `sglang serve`; docker mode → wrapped in `docker run`.
  const renderCommand = (cell, sel, envValues, mode = "python") => {
    if (!cell) return "# No command available for the current selection.";
    const modelName = resolveModelName(sel);
    const nnodes = parseNnodes(sel.nodes);
    const multinode = nnodes > 1;
    const cellEnv = cell.env || [];
    const flags = [...(cell.flags || [])];
    if (multinode) {
      // Insert the multi-node trio after the last parallelism flag,
      // falling back to right after --model-path.
      const PARALLELISM_ANCHORS = ["--enable-dp-attention", "--dp", "--tp"];
      let i = -1;
      for (const anchor of PARALLELISM_ANCHORS) {
        i = flags.findIndex((f) => f.split(/[\s=]/)[0] === anchor);
        if (i !== -1) break;
      }
      if (i === -1) i = flags.findIndex((f) => f.startsWith("--model-path"));
      flags.splice(i + 1, 0,
        `--nnodes ${nnodes}`,
        `--node-rank {{NODE_RANK}}`,
        `--dist-init-addr {{NODE0_IP}}:20000`);
    }

    let cmd;
    if (mode === "docker") {
      // Image keyed by `hw|quant` (most specific) then `hw`; `:dev` if unmapped.
      const di = config.dockerImages || {};
      const image = di[`${sel.hw}|${sel.quant}`] || di[sel.hw] || "lmsysorg/sglang:dev";
      const portFlag = flags.find((x) => x.split(/[\s=]/)[0] === "--port");
      const servePort = portFlag ? portFlag.slice("--port".length).trim() : "{{PORT}}";
      const vendorOf = (hwId) => {
        for (const [vendor, list] of Object.entries(HARDWARE_CATALOG)) {
          if (list.some((h) => h.id === hwId)) return vendor;
        }
        const extra = (config.hardware || []).find((h) => h.id === hwId);
        return (extra && extra.vendor) || "nvidia";
      };
      const gpuAccessLines = vendorOf(sel.hw) === "amd"
        ? [
            "docker run",
            "  --device=/dev/kfd --device=/dev/dri",
            "  --group-add video",
            "  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined",
            "  --shm-size 32g",
          ]
        : [
            "docker run --gpus all",
            "  --shm-size 32g",
          ];
      const dockerLines = [
        ...gpuAccessLines,
        // Multi-node needs host networking so the cross-node rendezvous port
        // (--dist-init-addr) and NCCL/GLOO traffic are reachable; single-node
        // just maps the serve port.
        multinode ? "  --network host" : `  -p ${servePort}:${servePort}`,
        "  -v ~/.cache/huggingface:/root/.cache/huggingface",
        // HF token only for gated checkpoints — configs that declare an HF_TOKEN placeholder.
        ...(config.placeholders && config.placeholders.HF_TOKEN
          ? [`  --env "HF_TOKEN={{HF_TOKEN}}"`] : []),
        ...cellEnv.map((e) => `  --env ${e}`),
        "  --ipc=host",
        `  ${image}`,
        "  sglang serve",
        ...flags.map((f) => "    " + f),
      ];
      cmd = dockerLines.join(" \\\n");
    } else {
      const flagBlock = flags.map((f) => "  " + f).join(" \\\n");
      const envBlock = cellEnv.length ? cellEnv.join(" \\\n") + " \\\n" : "";
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
    return cmd;
  };

  // Accuracy labels: [field-key, display-label, unit]. Declared per model via
  // `config.accuracyLabels` — the engine ships NO default eval set. A config
  // without it renders no accuracy rows (and no Accuracy section in the
  // "⚡ Reproduce" modal). Keys must match the `accuracy` fields in the
  // benchmarks file + `benchmarkCommands.accuracy`.
  const ACCURACY_LABELS = config.accuracyLabels || [];

  const renderBenchmarkCard = (entry) => {
    // [key, label, unit, compute?]. Optional compute(measurement) supplies
    // derived metrics (preferred over measurement[key] when present).
    const pct = config.latencyPercentile || "P50";
    const SPEED_LABELS = [
      ["ttft_ms",                `TTFT (${pct})`,      "ms"],
      ["tpot_ms",                `TPOT (${pct})`,      "ms"],
      // throughput per gpu = total(input+output)/elapsed/GPU (SemiAnalysis convention);
      // stored directly in the benchmarks file (= output tok/s/GPU × (isl+osl)/osl).
      ["tokens_per_sec_per_gpu", "throughput per gpu", "tok/s"],
      ["interactivity",          "interactivity",   "tokens/s/user",
        (m) => (m.tpot_ms != null && m.tpot_ms !== 0)
          ? Math.round((1000 / m.tpot_ms) * 10) / 10
          : null],
    ];
    const WORKLOAD_KEYS = ["dataset", "isl", "osl", "max_concurrency"];

    const fmt = (val, unit) => {
      if (val === null || val === undefined) return null;
      return `${val}${unit ? " " + unit : ""}`;
    };

    // Format a subset of workload keys into a comma-separated phrase.
    const formatWorkloadParts = (workload, keys) => {
      if (!workload) return "";
      const parts = [];
      if (keys.has("dataset") && workload.dataset) parts.push(workload.dataset);
      // in/out rendered as one token even if only one of isl/osl is present.
      if (keys.has("isl") || keys.has("osl")) {
        if (workload.isl != null || workload.osl != null) {
          parts.push(`in/out=${workload.isl != null ? workload.isl : "?"}/${workload.osl != null ? workload.osl : "?"}`);
        }
      }
      if (keys.has("max_concurrency") && workload.max_concurrency != null) {
        parts.push(`max-concurrency=${workload.max_concurrency}`);
      }
      return parts.join(", ");
    };

    // Split workload fields into shared (uniform → context line) vs differing
    // (→ per-column header). max_concurrency is always per-column.
    const ALWAYS_PER_COLUMN = new Set(["max_concurrency"]);
    const partitionWorkload = (measurements) => {
      const shared = new Set();
      const differing = new Set();
      for (const k of WORKLOAD_KEYS) {
        const seen = new Set();
        let anyPresent = false;
        for (const m of measurements) {
          const v = m && m.workload ? m.workload[k] : undefined;
          if (v != null) anyPresent = true;
          seen.add(v);
        }
        if (!anyPresent) continue;
        if (ALWAYS_PER_COLUMN.has(k) || seen.size > 1) differing.add(k);
        else shared.add(k);
      }
      return { shared, differing };
    };

    const renderBenchTable = ({ title, sharedText, colHeaders, rows, colCount, legend }) => {
      if (rows.length === 0) return null;
      const showColHeaders = colHeaders.length > 0
        && colHeaders.some((h) => h !== "");
      return (
        <div style={s.benchBlock}>
          <div style={s.benchBlockTitle}>{title}</div>
          {sharedText && (
            <div style={s.benchWorkload}>{sharedText}</div>
          )}
          <div
            style={{
              ...s.benchTable,
              gridTemplateColumns:
                `max-content repeat(${colCount}, minmax(0, 1fr))`,
            }}
          >
            {showColHeaders && (
              <div key="corner" style={s.benchTableCornerHead}></div>
            )}
            {showColHeaders && colHeaders.map((h, i) => (
              <div key={`hdr-${i}`} style={s.benchTableHead}>{h}</div>
            ))}
            {showColHeaders && (
              <div key="sep" style={s.benchTableSeparator}></div>
            )}
            {rows.map((r) => [
              <div key={`lbl-${r.label}`} style={s.benchTableLabel}>{r.label}</div>,
              ...r.values.map((v, i) => (
                <div key={`val-${r.label}-${i}`} style={
                  v === null
                    ? { ...s.benchTableValue, ...s.benchTableValueMissing }
                    : s.benchTableValue
                }>
                  {v !== null ? v : "—"}
                </div>
              )),
            ])}
          </div>
          {legend && (
            <div style={s.benchLegend}>
              {(Array.isArray(legend) ? legend : [legend]).map((line, i) => (
                <div key={`legend-${i}`}>{line}</div>
              ))}
            </div>
          )}
        </div>
      );
    };

    // All four metric rows always render (unmeasured cells show "—") so the
    // table shape is identical across every cell.
    const buildSpeedTable = (measurements) => {
      if (measurements.length === 0) return null;
      const { shared, differing } = partitionWorkload(measurements);
      const sharedText = formatWorkloadParts(
        measurements[0] && measurements[0].workload, shared);
      const colHeaders = measurements.map((m) =>
        formatWorkloadParts(m && m.workload, differing));
      const rows = SPEED_LABELS.map((tup) => {
        const [key, label, unit, compute] = tup;
        const values = measurements.map((m) => {
          const raw = compute ? compute(m) : m[key];
          return fmt(raw, unit);
        });
        return { label, values };
      });
      return { title: "Speed", sharedText, colHeaders, rows,
               colCount: measurements.length,
               legend: [
                 `throughput per gpu = (input+output tokens)/elapsed/GPU`,
                 `interactivity = 1000/TPOT(ms) (tokens/s/user)`,
               ] };
    };

    // One row per ACCURACY_LABELS entry with a non-null value; single value column.
    const buildAccuracyTable = (accuracy) => {
      if (!accuracy) return null;
      const rows = ACCURACY_LABELS
        .map(([key, label, unit]) => {
          const v = fmt(accuracy[key], unit);
          if (v === null) return null;
          return { label, values: [v] };
        })
        .filter((r) => r !== null);
      if (rows.length === 0) return null;
      return { title: "Accuracy", sharedText: null, colHeaders: [],
               rows, colCount: 1 };
    };

    const accuracy = effectiveAccuracy(entry, sel);
    const isEmpty = benchmarkIsEmpty(entry, accuracy);
    const measurements = !isEmpty ? normalizeSpeed(entry && entry.speed) : [];
    const accuracyTable = !isEmpty ? buildAccuracyTable(accuracy) : null;
    const speedTable = !isEmpty ? buildSpeedTable(measurements) : null;
    const hasBenchCmds = !isEmpty && buildBenchCommands(entry, sel) !== null;

    return (
      <div style={s.benchCard}>
        <div style={s.benchHeader}>
          <div style={s.benchTitle}>Benchmark</div>
          <div style={s.benchHeaderRight}>
            {!isEmpty && entry && entry.sglang_version && (
              <div style={s.benchVersion}>measured on sglang <code>{entry.sglang_version}</code></div>
            )}
            {hasBenchCmds && (
              <button style={s.iconButton} onClick={() => setModal("bench")}>⚡ Reproduce</button>
            )}
          </div>
        </div>
        {isEmpty ? (
          <div style={s.benchEmpty}>
            Benchmark data pending for this combination — submit yours via the Playground's Submit ↗ button.
          </div>
        ) : (
          <>
            {accuracyTable && renderBenchTable(accuracyTable)}
            {speedTable && renderBenchTable(speedTable)}
            {entry && entry.notes && (
              <div style={s.benchNotes}>{entry.notes}</div>
            )}
          </>
        )}
      </div>
    );
  };

  // Build "⚡ Reproduce" modal data (raw templates + fill metadata). The {{...}}
  // interpolation happens in the modal where `env` is in scope. Returns null
  // when nothing is renderable (caller hides the button).
  const buildBenchCommands = (entry, sel) => {
    const bc = config.benchmarkCommands;
    if (!bc) return null;

    // One entry per eval with a value AND a template. A template is a string,
    // or a {[variant]: string} object resolved by sel.variant.
    const acc = effectiveAccuracy(entry, sel);
    const accuracy = [];
    if (bc.accuracy) {
      for (const [key, label] of ACCURACY_LABELS) {
        if (acc[key] == null) continue;
        const tmpl = bc.accuracy[key];
        const resolved = (typeof tmpl === "string")
          ? tmpl
          : (tmpl && tmpl[sel.variant]) || null;
        if (resolved) accuracy.push({ key, label, template: resolved });
      }
    }

    let speed = null;
    if (bc.speed && entry) {
      const ms = normalizeSpeed(entry.speed)
        .filter((m) => m && m.workload && m.workload.max_concurrency != null);
      const concurrencies = [...new Set(ms.map((m) => m.workload.max_concurrency))]
        .sort((a, b) => a - b);
      if (concurrencies.length) {
        speed = {
          template: bc.speed,
          concurrencies,
          workload: ms[0].workload,
          // {{NUM_PROMPTS}} priority: per-row override → numPromptsByConc[c]
          // → max(c*2, 200).
          numPromptsOf: (c) => {
            const m = ms.find((x) => x.workload.max_concurrency === c);
            if (m && m.workload.num_prompts != null) return m.workload.num_prompts;
            const tbl = bc.numPromptsByConc;
            if (tbl && tbl[c] != null) return tbl[c];
            return Math.max(c * 2, 200);
          },
        };
      }
    }

    if (accuracy.length === 0 && !speed) return null;
    return { accuracy, speed };
  };

  const buildHardwareGroups = () => {
    const supported = new Set(config.supportedHardware);
    // Effective catalog = shared common GPUs + the model's own `config.hardware`
    // (model-specific / desktop / future GPUs the shared catalog doesn't carry).
    // A model-specific GPU is therefore pure config data — no engine-catalog edit.
    const catalog = {};
    for (const [vendor, list] of Object.entries(HARDWARE_CATALOG)) catalog[vendor] = [...list];
    for (const hw of (config.hardware || [])) {
      const vendor = hw.vendor || "nvidia";
      const list = catalog[vendor] || (catalog[vendor] = []);
      const entry = { id: hw.id, label: hw.label, vram: hw.vram };
      const i = list.findIndex((x) => x.id === hw.id);
      if (i >= 0) list[i] = entry; else list.push(entry); // config overrides by id
    }
    const groups = [];
    for (const [vendor, list] of Object.entries(catalog)) {
      const items = list.filter((hw) => supported.has(hw.id))
        .map((hw) => ({ id: hw.id, label: hw.label, subtitle: hw.vram }));
      if (items.length) groups.push({ label: vendor.toUpperCase(), items });
    }
    return groups;
  };

  const initialSelectionFromCells = () => {
    const first = config.cells[0];
    if (!first) return Object.fromEntries(DIMENSIONS.map((d) => [d, ""]));
    return {
      hw: first.match.hw, variant: first.match.variant, quant: first.match.quant,
      strategy: first.match.strategy, nodes: first.match.nodes,
    };
  };

  const placeholderDefaults = (schema) => {
    const out = {};
    for (const [k, v] of Object.entries(schema || {})) out[k] = v.default ?? "";
    return out;
  };

  // ==== 4. React state + effects ====
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

  const [sel, setSel] = useState(() => initialSelectionFromCells());
  useEffect(() => {
    const hydrate = () => {
      const raw = window.location.hash.replace(/^#/, "");
      if (!raw) return;
      const params = new URLSearchParams(raw);
      const initial = initialSelectionFromCells();
      const parsed = { ...initial };
      let touched = false;
      params.forEach((value, key) => {
        if (key in parsed) { parsed[key] = value; touched = true; }
      });
      if (!touched) return;
      // Snap to a real cell if the hash named an impossible combo (stale link).
      setSel(validateSelection(config.cells, parsed));
      // Scroll the Deploy section into view. Heading slugs to "deployment" or
      // "deploy"; only fires on hash navigation (not replaceState chip clicks).
      const el = document.getElementById("deployment")
        || document.getElementById("deploy");
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    };
    hydrate();
    window.addEventListener("hashchange", hydrate);
    return () => window.removeEventListener("hashchange", hydrate);
  }, []);
  // history.replaceState does NOT fire hashchange — dispatch a custom event so
  // the Playground hears chip-click selection changes.
  useEffect(() => {
    const target = "#" + new URLSearchParams(sel).toString();
    if (window.location.hash !== target) {
      window.history.replaceState(null, "", target);
    }
    window.dispatchEvent(new CustomEvent("sglang-deploy-sel", { detail: sel }));
  }, [sel]);

  const [modal, setModal] = useState(null); // 'curl' | 'env' | 'bench' | null
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
  // "⚡ Reproduce" modal: chip-selected concurrency / eval / last-copied block.
  // null falls back to the first option in the render.
  const [benchConc, setBenchConc] = useState(null);
  const [benchAcc, setBenchAcc] = useState(null);
  const [benchCopied, setBenchCopied] = useState(null);
  const [runMode, setRunMode] = useState("python"); // "python" | "docker"
  useEffect(() => { if (modal === "env") setEnvDraft(env); }, [modal, env]);

  // ==== 5. Derived values ====
  const s = makeStyles(isDark);
  const cell = findCell(config.cells, sel);
  const command = renderCommand(cell, sel, env, runMode);
  // MTP hint: fire on the actual command (speculative decoding ON) — NOT on
  // strategy=low-latency, since a low-latency cell may not enable MTP. SGLang
  // resets --max-running-requests to 48 when spec is on and it's unset.
  const mtpHint =
    !!cell &&
    (cell.flags || []).some((f) => f.split(/[\s=]/)[0] === "--speculative-algorithm") &&
    !(cell.flags || []).some((f) => f.split(/[\s=]/)[0] === "--max-running-requests");
  const modelName = resolveModelName(sel);
  const curlText = interpolate(config.curl || "", env, modelName);
  const hwGroups = buildHardwareGroups();
  const benchEntry = benchmarks ? findBenchmark(benchmarks, sel) : null;

  const isEnabled = (dim, value) => isOptionAvailable(config.cells, sel, dim, value);

  const handleSelect = (dim, value) => {
    setSel((prev) => snapToValidCell(config.cells, prev, dim, value));
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  const copyCurl = () => {
    navigator.clipboard.writeText(curlText);
    setCurlCopied(true);
    setTimeout(() => setCurlCopied(false), 1200);
  };
  // `key` identifies which Reproduce-modal block flashed "✓ Copied".
  const copyBench = (key, text) => {
    navigator.clipboard.writeText(text);
    setBenchCopied(key);
    setTimeout(() => setBenchCopied(null), 1200);
  };

  // Group placeholders by `target` for the Env modal.
  const placeholderGroups = (() => {
    const out = { command: [], curl: [] };
    for (const [key, meta] of Object.entries(config.placeholders || {})) {
      (out[meta.target] || (out[meta.target] = [])).push({ key, ...meta });
    }
    return out;
  })();

  // ==== 6. JSX render ====
  const renderButton = (item, dim, selectedId) => {
    const checked = selectedId === item.id;
    const disabled = !isEnabled(dim, item.id);
    return (
      <label
        key={item.id}
        style={{
          ...s.labelBase,
          ...(checked ? s.checked : {}),
          ...(disabled ? s.disabled : {}),
        }}
        title={disabled ? "Not supported for current selection" : ""}
        onClick={(e) => {
          if (disabled) { e.preventDefault(); return; }
          handleSelect(dim, item.id);
        }}
      >
        <input type="radio" checked={checked} disabled={disabled} readOnly style={{ display: "none" }} />
        <span>{item.label}</span>
        {item.subtitle && (
          <small style={{ ...s.subtitle, color: checked ? "rgba(255,255,255,0.85)" : "inherit" }}>
            {item.subtitle}
          </small>
        )}
      </label>
    );
  };

  const renderFlatSection = (title, options, dim, selectedId) => (
    <div style={s.card}>
      <div style={s.title}>{title}</div>
      <div style={s.itemsGrid(options.length)}>
        {options.map((item) => renderButton(item, dim, selectedId))}
      </div>
    </div>
  );

  const maxHwCols = Math.max(...hwGroups.map((x) => x.items.length));

  return (
    <div style={s.container} className="not-prose">
      {/* Hardware section (2 vendor rows in one card, equal-width grid) */}
      <div style={s.cardColumn}>
        <div style={{ ...s.title, marginBottom: "2px" }}>Hardware Platform</div>
        {hwGroups.map((g) => (
          <div key={g.label} style={s.vendorRow}>
            <div style={s.vendorLabel}>{g.label}</div>
            <div style={s.itemsGrid(maxHwCols)}>
              {g.items.map((item) => renderButton(item, "hw", sel.hw))}
              {Array.from({ length: maxHwCols - g.items.length }).map((_, i) => (
                <div key={`pad-${i}`} />
              ))}
            </div>
          </div>
        ))}
      </div>

      {renderFlatSection("Model Variant", config.variants,        "variant",  sel.variant)}
      {renderFlatSection("Quantization",  config.quantizations,   "quant",    sel.quant)}
      {renderFlatSection("Strategy",      config.strategies,      "strategy", sel.strategy)}
      {renderFlatSection("Nodes",         config.nodesOptions,    "nodes",    sel.nodes)}

      {/* Command box */}
      <div style={s.card}>
        <div style={s.title}>Run this Command:</div>
        <div style={s.commandWrap}>
          <div style={s.commandHeader}>
            <div style={s.headerLeft}>
              <div style={s.badge(Boolean(cell && cell.verified))}>
                <span style={s.badgeDot(Boolean(cell && cell.verified))} />
                {cell && cell.verified ? "Verified" : "Not Verified"}
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
          <pre style={s.commandPre}>{command}</pre>
          {mtpHint && (
            <div style={s.mtpWarn}>
              ⚠️ Speculative decoding (MTP) is on — SGLang resets <code>--max-running-requests</code> to <strong>48</strong> when it isn't set. Add <code>--max-running-requests &lt;N&gt;</code> sized for your target concurrency.
            </div>
          )}
        </div>
      </div>

      {/* Benchmark card (only with a `benchmarks` prop + matched cell). */}
      {benchmarks && cell && renderBenchmarkCard(benchEntry)}

      {/* Playground link — scrollIntoView, not an href, so the hash (which
          carries the selection) isn't overwritten. */}
      <div
        style={{
          padding: "6px 12px",
          fontSize: "12px",
          color: isDark ? "#9ca3af" : "#6b7280",
          display: "flex",
          alignItems: "center",
          gap: "6px",
        }}
      >
        <span>Need to go beyond the verified matrix?</span>
        <button
          type="button"
          onClick={() => {
            const el = document.getElementById("playground");
            if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
          style={{
            background: "transparent",
            border: "none",
            padding: 0,
            color: isDark ? "#FDBA74" : "#C2410C",
            cursor: "pointer",
            fontSize: "12px",
            fontWeight: 600,
            textDecoration: "underline",
            textUnderlineOffset: "2px",
          }}
        >
          Open the Playground →
        </button>
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
              Values persist in localStorage and are reused the next time you visit any cookbook.
            </p>
          </div>
        </div>
      )}

      {/* "⚡ Reproduce" modal — benchmark commands for the current selection. */}
      {modal === "bench" && benchEntry && (() => {
        const bc = buildBenchCommands(benchEntry, sel);
        if (!bc) return null;
        const selSummary =
          `${sel.hw.toUpperCase()} · ${sel.variant} · ${sel.quant.toUpperCase()} · ${sel.strategy} · ${sel.nodes}`;
        let selConc = null;
        let speedCmd = null;
        if (bc.speed) {
          selConc = bc.speed.concurrencies.includes(benchConc)
            ? benchConc : bc.speed.concurrencies[0];
          const w = bc.speed.workload;
          speedCmd = interpolate(bc.speed.template, {
            ...env,
            DATASET: w.dataset,
            ISL: w.isl,
            OSL: w.osl,
            MAX_CONCURRENCY: selConc,
            NUM_PROMPTS: bc.speed.numPromptsOf(selConc),
          }, modelName);
        }
        // Accuracy chip-selected; stale benchAcc falls back to the first eval.
        let selAcc = null;
        let accCmd = null;
        if (bc.accuracy.length > 0) {
          selAcc = bc.accuracy.find((a) => a.key === benchAcc) || bc.accuracy[0];
          accCmd = interpolate(selAcc.template, env, modelName);
        }
        return (
          <div style={s.modalBackdrop} onClick={() => setModal(null)}>
            <div style={s.modalBox} onClick={(e) => e.stopPropagation()}>
              <div style={s.modalHeader}>
                <div style={s.modalTitle}>Benchmark commands</div>
                <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
              </div>
              <p style={{ fontSize: 11, opacity: 0.7, margin: "0 0 12px" }}>
                For <code>{selSummary}</code>. Start the server with the Deploy command above, then run these against it.
              </p>

              {selAcc && (
                <div>
                  <div style={s.sectionHeading}>Accuracy</div>
                  {bc.accuracy.length > 1 && (
                    <div style={s.benchChipRow}>
                      <span style={{ fontSize: 11, opacity: 0.7 }}>benchmark:</span>
                      {bc.accuracy.map((a) => (
                        <button
                          key={a.key}
                          style={{ ...s.benchChip, ...(a.key === selAcc.key ? s.benchChipActive : {}) }}
                          onClick={() => setBenchAcc(a.key)}
                        >
                          {a.label}
                        </button>
                      ))}
                    </div>
                  )}
                  <div style={{ ...s.commandWrap, marginBottom: 6 }}>
                    <div style={s.commandHeader}>
                      <div style={{ fontSize: 11, opacity: 0.7 }}>{selAcc.label}</div>
                      <button style={s.iconButton} onClick={() => copyBench("acc", accCmd)}>
                        {benchCopied === "acc" ? "✓ Copied" : "⧉ Copy"}
                      </button>
                    </div>
                    <pre style={s.commandPre}>{accCmd}</pre>
                  </div>
                  {bc.accuracy.length > 1 && (
                    <p style={{ fontSize: 11, opacity: 0.7, margin: "0 0 4px" }}>
                      Switch the benchmark chip to see each eval's command.
                    </p>
                  )}
                </div>
              )}

              {bc.speed && (
                <div>
                  <div style={s.sectionHeading}>Speed</div>
                  {bc.speed.concurrencies.length > 1 && (
                    <div style={s.benchChipRow}>
                      <span style={{ fontSize: 11, opacity: 0.7 }}>max-concurrency:</span>
                      {bc.speed.concurrencies.map((c) => (
                        <button
                          key={c}
                          style={{ ...s.benchChip, ...(c === selConc ? s.benchChipActive : {}) }}
                          onClick={() => setBenchConc(c)}
                        >
                          {c}
                        </button>
                      ))}
                    </div>
                  )}
                  <div style={{ ...s.commandWrap, marginBottom: 6 }}>
                    <div style={s.commandHeader}>
                      <div style={{ fontSize: 11, opacity: 0.7 }}>max-concurrency = {selConc}</div>
                      <button style={s.iconButton} onClick={() => copyBench("speed", speedCmd)}>
                        {benchCopied === "speed" ? "✓ Copied" : "⧉ Copy"}
                      </button>
                    </div>
                    <pre style={s.commandPre}>{speedCmd}</pre>
                  </div>
                  <p style={{ fontSize: 11, opacity: 0.7, margin: "0 0 4px" }}>
                    One command — switch the concurrency chip (or edit <code>--max-concurrency</code>) to reproduce each Speed column.
                  </p>
                </div>
              )}

              <p style={{ fontSize: 11, opacity: 0.7, marginTop: 12 }}>
                Edit <code>CURL_HOST</code> / <code>CURL_PORT</code> in the Env panel.
              </p>
            </div>
          </div>
        );
      })()}
    </div>
  );
};
