// Shared deployment skeleton — the ENGINE half of the SGLang cookbook
// deployment-command generator. Pair this with a per-model config file under
// `/src/snippets/configs/<vendor>/<model>.jsx` and an MDX page that imports
// both:
//
//     import { Deployment } from "/src/snippets/_deployment.jsx";
//     import { config }     from "/src/snippets/configs/deepseek-ai/deepseek-v4.jsx";
//     <Deployment config={config} />
//
// This file contains nothing model-specific. Add a new cookbook recipe by
// dropping a new config file and pointing an MDX at it — no engine edits
// required. To improve the engine (new button, new UI feature, VRAM update),
// edit this file once and every cookbook gets the upgrade.
//
// AUTHORING — read .claude/rules/cookbook-authoring.md for the step-by-step
// workflow on adding a new cookbook or extending the engine.
//
// CONFIG CONTRACT (what the engine reads from `config`)
// -----------------------------------------------------
//   supportedHardware  string[]                    — hw ids visible in the UI
//                                                    catalog (subset of
//                                                    HARDWARE_CATALOG below).
//   variants           {id, label, subtitle?}[]    — 2nd-dim option list.
//   quantizations      {id, label}[]               — 3rd-dim option list.
//   strategies         {id, label}[]               — 4th-dim option list.
//   nodesOptions       {id, label}[]               — 5th-dim option list. The
//                                                    id must be either
//                                                    `single` or `multi-N`
//                                                    (engine extracts N from
//                                                    the id for --nnodes).
//   cells              {match, verified?, env,
//                       flags}[]                   — one per supported
//                                                    (hw × variant × quant ×
//                                                    strategy × nodes) combo.
//                                                    `match` keys MUST be the
//                                                    five DIMENSIONS below.
//                                                    env/flags are flat
//                                                    literals consumed
//                                                    verbatim (no
//                                                    interpolation other than
//                                                    {{PLACEHOLDER}} subst).
//   modelNames         {`hw|variant|quant` |
//                       `variant|quant`: string}   — HF slug lookup. Layered:
//                                                    triple key wins, falls
//                                                    back to pair.
//   placeholders       {[key]: {target: 'command'
//                       | 'curl', label, default?}}
//                                                  — {{KEY}} interpolation
//                                                    map for command + curl.
//                                                    Editable via Env modal.
//   curl               string                      — cURL template (uses
//                                                    {{MODEL_NAME}} +
//                                                    placeholders).
//   multiNodeHints     {[hwId]: string[]}          — comment lines prepended
//                                                    to multi-node commands.
//                                                    Each string becomes one
//                                                    `# ...` line.
//   dockerImages       {[hwId]: string}            — per-hw image name for
//                                                    `docker run` mode.
//                                                    Falls back to
//                                                    `lmsysorg/sglang:dev` if
//                                                    a hw id is missing.
//   benchmarkCommands  {speed: string,             — optional. Powers the
//                       accuracy: {[accKey]:          benchmark card's
//                       string},                      "⚡ Reproduce" modal.
//                       numPromptsByConc?:           `speed` is ONE
//                       {[c]: number}}                bench_serving template;
//                                                    the engine fills
//                                                    {{DATASET}}/{{ISL}}/{{OSL}}
//                                                    from the cell's workload,
//                                                    the chip-picked
//                                                    {{MAX_CONCURRENCY}}, and
//                                                    {{NUM_PROMPTS}} resolved
//                                                    as workload.num_prompts
//                                                    ?? numPromptsByConc[c]
//                                                    ?? max(c*2, 200).
//                                                    `accuracy` maps an
//                                                    accuracy field (e.g.
//                                                    gsm8k_pct) to a per-eval
//                                                    template — a string, or a
//                                                    {[variant]: string} object
//                                                    when the command differs
//                                                    per variant. Both also use
//                                                    {{MODEL_NAME}} +
//                                                    {{CURL_HOST}}/{{CURL_PORT}}
//                                                    like `curl`.
//   defaultAccuracy    {[variant]:                 — optional. Model-level
//                       {[accKey]: number}}           accuracy applied to EVERY
//                                                    cell of a variant (e.g.
//                                                    GPQA/AIME — hardware-
//                                                    independent). Merged UNDER
//                                                    each cell's measured
//                                                    `accuracy` (per-cell wins).
//
// MINTLIFY CAVEATS THIS FILE ROUTES AROUND
// ----------------------------------------
//   - Module-level statements inside a `.jsx` snippet are stripped — so every
//     helper / hook / handler stays inside the wrapper function body below.
//   - Capitalized JSX tags inside a snippet's function body get rebound via
//     `_provideComponents()`, so we use only lowercase HTML tags here (and
//     factor sub-regions into `renderButton()` / `renderFlatSection()` helper
//     functions, not into sub-components).
//   - Imports of plain-data exports from other `.jsx` files DO work when the
//     import lives in an MDX file. So the per-model config is imported at
//     MDX level and passed through here as the `config` prop.

export const Deployment = ({ config, benchmarks }) => {
  if (!config) {
    return <div style={{padding: 12, color: "#b91c1c"}}>Deployment: missing <code>config</code> prop</div>;
  }

  // ==========================================================================
  // 1. Hardware catalog (shared across cookbooks — single source of truth)
  // ==========================================================================
  // VRAM is reported as per-GPU on-chip memory (HBM3/3e/E), not per-module.
  // Adding a new SKU here makes it eligible for any cookbook to list in
  // `config.supportedHardware`. Removing one hides it everywhere.
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

  // ==========================================================================
  // 2. Style helper (dark-mode-aware)
  // ==========================================================================
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
    itemsGrid: (cols) => ({
      display: "grid", gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
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
    iconRow: { display: "inline-flex", gap: "6px" },
    // Two-segment "Python | Docker" pill that sits next to the verified badge.
    // Reuses the badge's pill silhouette so it visually groups with it.
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
      padding: "6px 14px", background: "#D45D44", color: "white",
      border: "none", borderRadius: "4px", cursor: "pointer",
      fontSize: "13px", fontWeight: 500,
    },

    // ---------- Benchmark card (rendered when the `benchmarks` prop is
    // passed AND the current cell has a matching entry). Uses the same
    // orange left-border family as the deploy panel so the two visually
    // group together.
    benchCard: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
      display: "flex", flexDirection: "column", gap: "8px",
    },
    benchHeader: {
      display: "flex", alignItems: "baseline", justifyContent: "space-between",
      gap: "12px",
    },
    benchTitle: {
      fontSize: "13px", fontWeight: 600,
      color: isDark ? "#e5e7eb" : "inherit",
    },
    benchVersion: {
      fontSize: "11px",
      color: isDark ? "#9ca3af" : "#6b7280",
    },
    // Right side of the benchmark header: version annotation + "⚡ Reproduce".
    benchHeaderRight: {
      display: "flex", alignItems: "center", gap: "10px", flexShrink: 0,
    },
    // Concurrency chip row inside the Reproduce modal's Speed section.
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
    // Selected concurrency chip — same terracotta as the Deploy panel's
    // selected button (`checked`).
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
    // Workload description (e.g. "ShareGPT, in/out=1024/1024, bs=1") that
    // sits between the block title and the metric rows. Italic + slightly
    // muted so it reads as context, not data.
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
    // Small legend row below a bench block — used today to define
    // `interactivity = 1000 / TPOT(ms)`. Muted, italic, mono for the
    // formula so the equality reads as a definition not a metric.
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

    // Speed table — rows are metrics (TTFT, TPOT, …), columns are
    // per-workload measurements (typically varying max-concurrency).
    // Implemented as CSS grid on a <div>, NOT a real <table>: Mintlify's
    // page renderer wraps every <table> element in two extra divs that
    // apply `margin: 4px -20px 12px` (negative horizontal margins so the
    // table can bleed into the page padding for horizontal scrolling)
    // PLUS `[&_td]:min-w-[150px]` on the table itself — both fire even
    // inside .not-prose, and the negative-margin trick mis-aligns the
    // contents inside a nested card. Using a grid of plain divs
    // sidesteps the entire <table> auto-wrapping pipeline.
    //
    // The grid columns are set inline (`gridTemplateColumns`) since they
    // depend on `measurements.length`: 1 auto label col + N value cols.
    benchTable: {
      display: "grid",
      // columnGap is 0 (not 16) so adjacent cells' bottom borders touch
      // to form a single continuous line under the header row. Inter-
      // column visual spacing comes from per-cell paddingLeft instead.
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
    // Header underline. Single div that spans every column (gridColumn
    // "1 / -1"), so the line is guaranteed continuous regardless of cell
    // heights or column count. Previous approach (border-bottom per
    // cell) broke when the empty corner cell collapsed shorter than the
    // header cells under `alignItems: baseline`, putting their bottom
    // borders at different Y positions.
    benchTableSeparator: {
      gridColumn: "1 / -1",
      height: "1px",
      background: isDark ? "#374151" : "#e5e7eb",
      // Negate the rowGap above so the line sits right against the
      // header row instead of floating 3px below it.
      marginTop: "-3px",
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

  // ==========================================================================
  // 3. Pure helpers (no React state)
  // ==========================================================================
  // DIMENSIONS is ordered by priority — higher-index dims adapt to lower-index
  // picks, never the other way around. Order is Hardware → Variant →
  // Quantization → Strategy → Nodes, matching the on-screen sections from top
  // to bottom and the user's mental model ("first I pick my GPU, then the
  // model size, then quant, then how I want to serve, then how many nodes").
  const DIMENSIONS = ["hw", "variant", "quant", "strategy", "nodes"];
  const findCell = (cells, sel) =>
    cells.find((c) => DIMENSIONS.every((d) => c.match[d] === sel[d]));

  // Mirror of findCell, but over the `benchmarks` array (per-cell results
  // keyed by the same 5-dim match tuple). Returns the matching entry or
  // null. The engine treats null and "no measured fields" the same way —
  // both surface the empty-state card.
  const findBenchmark = (list, sel) =>
    (list || []).find((b) => DIMENSIONS.every((d) => b.match[d] === sel[d])) || null;

  // Normalize the `speed` slot. Cookbooks may supply either:
  //   - a single measurement object (legacy single-workload form), or
  //   - an array of measurements (one per workload, typically varying
  //     max-concurrency for a Pareto-style sweep).
  // The engine always works with an array internally.
  const normalizeSpeed = (speed) => {
    if (!speed) return [];
    return Array.isArray(speed) ? speed : [speed];
  };

  // Variant-level default accuracy merged UNDER the per-cell measured accuracy.
  // GPQA / AIME are model-quality numbers — identical across every cell of a
  // variant — so they live ONCE in `config.defaultAccuracy[variant]` instead of
  // being copied onto all ~40 benchmark entries. A per-cell `entry.accuracy`
  // field still overrides (e.g. a cell-specific gsm8k score). Returns the
  // merged accuracy object the card + modal both read from.
  const effectiveAccuracy = (entry, sel) => ({
    ...((config.defaultAccuracy && config.defaultAccuracy[sel.variant]) || {}),
    ...((entry && entry.accuracy) || {}),
  });

  // True iff the entry has no numeric content worth rendering. Used to
  // decide between rendering the metric blocks vs the empty-state line.
  // An entry is "empty" if every speed measurement is null-only AND the
  // (already-merged) `accuracy` is null-only. The `workload` slot is metadata
  // about the test conditions (dataset / isl / osl / max_concurrency), not a
  // measurement — skip it when deciding whether a measurement has data,
  // otherwise a workload-only stub would falsely suppress the "pending" card.
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

  // Grey-out predicate: a (dim, value) button is enabled iff there exists at
  // least one cell that matches every HIGHER-priority dimension in the current
  // selection AND has `dim === value`. Lower-priority dims are free to differ
  // — they'll be adapted by snapToValidCell on click.
  //
  // Example: with sel = { hw:b200, variant:flash, quant:fp4, strategy:low-latency, nodes:single },
  // the "Multi-Nodes" button on the Nodes section asks: is there a cell with
  // hw=b200 AND variant=flash AND quant=fp4 AND strategy=low-latency AND nodes=multi-2?
  // There isn't (B200 has no multi-node cells) → button is greyed out.
  // It will NOT silently switch hardware to find a multi-node cell.
  const isOptionAvailable = (cells, sel, dim, value) => {
    const idx = DIMENSIONS.indexOf(dim);
    const higher = DIMENSIONS.slice(0, idx);
    return cells.some(
      (c) => c.match[dim] === value && higher.every((d) => c.match[d] === sel[d]),
    );
  };

  // When the user clicks an ENABLED button at `dim`, snap to a real cell:
  //   - higher-priority dims (above `dim`) stay locked to `sel[d]`
  //   - the clicked `dim` becomes `value`
  //   - lower-priority dims (below `dim`) adopt the best-fit cell's values,
  //     preferring the cell that keeps the most of the user's current lower-
  //     priority choices (minimises perceived churn).
  // If the button was correctly greyed out by isOptionAvailable, this function
  // is never called for a value with no matching cell.
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

  // Validate a selection from URL hash: walk dimensions in priority order,
  // accept each parsed value if a matching cell exists with all already-
  // accepted dims; otherwise adopt the value from the first cell consistent
  // with the dims accepted so far. This guarantees the result is a real cell
  // even if the URL hash names an impossible combination (e.g. a stale link
  // after the catalog changed).
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

  // Render a cell as either a bare `sglang serve` invocation (python mode) or
  // wrapped in `docker run` against the per-hardware image (docker mode). Both
  // share the multi-node flag injection, hint comments, and {{placeholder}}
  // interpolation; only the outer framing differs.
  const renderCommand = (cell, sel, envValues, mode = "python") => {
    if (!cell) return "# No command available for the current selection.";
    const modelName = resolveModelName(sel);
    const nnodes = parseNnodes(sel.nodes);
    const multinode = nnodes > 1;
    // Per-model configs are pure flat literals — cell.env and cell.flags are
    // consumed verbatim. No fragment expansion, no aliasing, no preprocessing.
    const cellEnv = cell.env || [];
    const flags = [...(cell.flags || [])];
    if (multinode) {
      // Insert the multi-node trio after the LAST parallelism flag in the
      // cell (--enable-dp-attention > --dp > --tp), falling back to right
      // after --model-path. This matches the convention used by the
      // legacy hand-coded generator on origin/main (which is what the
      // live cookbook page renders today) — keeps cookbook commands
      // byte-identical across the refactor.
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
      // Wrap as `docker run`. The port mapping uses the same {{PORT}} that the
      // --port flag will resolve to, so they stay in sync when the user edits
      // PORT in the Env modal. Image picked by hardware from config.dockerImages
      // (mirrors the single image in the cookbook's "Install SGLang" accordion).
      // If the hw isn't mapped (shouldn't happen for supported cells), we fall
      // back to `:dev`.
      const image = (config.dockerImages && config.dockerImages[sel.hw]) || "lmsysorg/sglang:dev";
      const dockerLines = [
        "docker run --gpus all",
        "  --shm-size 32g",
        "  -p {{PORT}}:{{PORT}}",
        "  -v ~/.cache/huggingface:/root/.cache/huggingface",
        `  --env "HF_TOKEN={{HF_TOKEN}}"`,
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

  // Render the per-cell benchmark card. Inline function (not a sub-
  // component) — capitalized React component names get rebound by
  // Mintlify's _provideComponents, so we keep this as a lowercase-only
  // JSX tree. Engine consumer renders <renderBenchmarkCall(...)>{}</>
  // by inlining the return.
  //
  // Layout (top → bottom):
  //   header   : "Benchmark"   "measured on sglang vX.Y.Z" (right-aligned)
  //   accuracy : single row collapsing whatever accuracy fields are
  //              present (GSM8K today; add more via ACCURACY_LABELS).
  //              Comes BEFORE speed — model quality leads serving speed.
  //   speed    : metric × workload table. Rows are TTFT / TPOT /
  //              tokens/sec/GPU / interactivity (1000/TPOT_ms) — all
  //              four ALWAYS render so every cell shares the same
  //              schema; unmeasured entries show "—". Columns are
  //              per-workload measurements (typically varying
  //              max-concurrency). Shared workload context (dataset,
  //              in/out) is lifted to an italic line above the table;
  //              the differing parts become the column headers.
  //              A small `legend` strip below the table defines
  //              derived metrics (e.g. `interactivity = 1000/TPOT(ms)`).
  //   notes    : optional italic line
  //   empty    : when the entry has no measured numbers
  //
  // The version annotation lives at the header, NOT under accuracy,
  // since every block in the entry is part of the same measurement run
  // on the same sglang build.
  //
  // Schema (consumed from `benchmarks` prop entries):
  //   {
  //     match: { hw, variant, quant, strategy, nodes },   // REQUIRED
  //     sglang_version: string,                           // header annotation
  //     speed: [
  //       { workload: { dataset, isl, osl, max_concurrency },
  //         ttft_ms, tpot_ms, tokens_per_sec_per_gpu },
  //       ...                                             // one per workload
  //     ],                                                // also accepts a
  //                                                       //   single object
  //                                                       //   (wrapped to [obj])
  //     accuracy: { gsm8k_pct, ... },                     // extensible — add
  //                                                       //   more keys to
  //                                                       //   ACCURACY_LABELS
  //     notes:    string,                                 // italic caveat
  //   }
  // `interactivity` is derived from tpot_ms (1000 / tpot_ms in tok/s),
  // not stored. Missing metrics render as "—". A measurement that
  // carries only a workload (no measured numbers) doesn't count toward
  // "has data" — the card falls back to the pending empty state when
  // nothing across all measurements has a value.
  // Accuracy labels. Starts with GSM8K; add new accuracy benchmarks here as
  // they come online (e.g. MATH, HumanEval). Shared by the benchmark card
  // (collapses present fields into a row) AND the "⚡ Reproduce" modal (one
  // chip per present-and-templated field, picking which eval command shows).
  // Keys must match the `accuracy` field names in the benchmarks file +
  // `benchmarkCommands.accuracy`.
  const ACCURACY_LABELS = [
    ["gpqa_pct",   "GPQA Diamond", "%"],
    ["aime25_pct", "AIME25",       "%"],
    ["gsm8k_pct",  "GSM8K (1-shot)", "%"],
  ];

  const renderBenchmarkCard = (entry) => {
    // Speed labels. Four metrics measured per workload: TTFT, TPOT,
    // tokens/sec/GPU (system aggregate), interactivity (per-user
    // generation speed, derived as 1000 / TPOT_ms = tok/s). The 4th
    // tuple slot is an optional `compute(measurement)` function used
    // for derived metrics that aren't stored — engine prefers compute()
    // over measurement[key] when present.
    const SPEED_LABELS = [
      ["ttft_ms",                "TTFT",            "ms"],
      ["tpot_ms",                "TPOT",            "ms"],
      ["tokens_per_sec_per_gpu", "tokens/sec/GPU",  ""],
      ["interactivity",          "interactivity",   "tok/s",
        (m) => (m.tpot_ms != null && m.tpot_ms !== 0)
          ? Math.round((1000 / m.tpot_ms) * 10) / 10
          : null],
    ];
    // Workload field keys, in display order. Used both for the shared
    // workload line and to decide column headers via set diff.
    const WORKLOAD_KEYS = ["dataset", "isl", "osl", "max_concurrency"];

    const fmt = (val, unit) => {
      if (val === null || val === undefined) return null;
      return `${val}${unit ? " " + unit : ""}`;
    };

    // Format the given subset of workload keys into a comma-separated
    // phrase. Used twice: once for the shared workload line above the
    // table (with all shared `keys`), once for each column header
    // (with only the `differing` keys). Same vocabulary in both spots
    // so readers don't have to translate between abbreviations.
    const formatWorkloadParts = (workload, keys) => {
      if (!workload) return "";
      const parts = [];
      if (keys.has("dataset") && workload.dataset) parts.push(workload.dataset);
      // in/out is rendered as a single token even if only one of
      // isl/osl is in the keys set — they read better together.
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

    // Split workload fields into "same across all measurements" (lifted
    // out to the italic context line above the table) vs "differs"
    // (becomes the per-column header). `max_concurrency` is forced
    // into the per-column header regardless of whether it varies —
    // it's the primary run-axis dimension and the reader needs to
    // know at WHICH concurrency the numbers were measured, even when
    // a cell ships with only one measurement. Other workload fields
    // fall back to auto-decide (shared if uniform, differing if not).
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

    // Generic table renderer. Both speed and accuracy blocks feed it
    // a {title, sharedText, colHeaders, rows, colCount} structure —
    // the visual output (gray benchBlock with a CSS grid inside) is
    // identical, only the data shape differs.
    //
    // The grid uses divs (not <table>) because Mintlify's page
    // renderer auto-wraps every <table> element in two scroll
    // wrappers with negative horizontal margins, which mis-align
    // inside our nested card. `.not-prose` only disables prose
    // typography, NOT the table component wrapping.
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
            <div style={s.benchLegend}>{legend}</div>
          )}
        </div>
      );
    };

    // Build the speed-table data structure: rows are metrics (TTFT,
    // TPOT, …), columns are per-workload measurements (typically
    // varying max-concurrency). All four metric rows ALWAYS render —
    // unmeasured cells show "—" — so the table shape is identical
    // across every cell of the catalog (low-latency, balanced,
    // high-throughput). Without this, partial sweeps would look like
    // two different benchmark schemas (a "latency table" with no
    // tokens/sec row vs a "throughput table" with no TTFT/TPOT rows),
    // which is exactly the confusion this design unifies.
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
               legend: "interactivity = 1000 / TPOT(ms)" };
    };

    // Build the accuracy-table data structure. One row per benchmark
    // in ACCURACY_LABELS that has a non-null value. Single value
    // column today (no column header) — extend to multi-column if
    // cookbooks ever need to compare eval configs (e.g. shots=0 vs
    // shots=5) by changing the accuracy schema to an array.
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
    // Show "⚡ Reproduce" only when the cookbook supplied benchmarkCommands AND
    // there is at least one renderable command for this entry.
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

  // Build the data for the "⚡ Reproduce" modal from `config.benchmarkCommands`
  // + a matched benchmark `entry`. Returns the raw templates + the metadata
  // the modal needs to fill them (concurrency list, a representative workload,
  // a per-concurrency num-prompts resolver). The actual {{...}} interpolation
  // happens in the modal (where `env` is in scope), reusing `interpolate`.
  //
  //   accuracy : [{ key, label, template }]  — one per ACCURACY_LABELS field
  //              that has a value (merged: per-cell `entry.accuracy` over
  //              `config.defaultAccuracy[variant]`) AND a template. A template
  //              may be a string OR a `{[variant]: string}` object (resolved by
  //              `sel.variant`) — used when a command differs per variant.
  //   speed    : { template, concurrencies[], workload, numPromptsOf(c) }
  //              or null. `workload` is the first measured workload (dataset/
  //              isl/osl are uniform across a cell's speed rows; only
  //              concurrency varies, which is the chip-switched knob).
  // Returns null when nothing is renderable (caller hides the button).
  const buildBenchCommands = (entry, sel) => {
    const bc = config.benchmarkCommands;
    if (!bc) return null;

    // Accuracy commands run for every eval that has a value (merged: per-cell
    // override or variant default) AND a template. A template may be a plain
    // string (variant-agnostic, e.g. gsm8k) OR a `{flash, pro, …}` object
    // keyed by variant (e.g. GPQA/AIME differ only in --max-tokens per variant).
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
          // Resolve {{NUM_PROMPTS}} in priority order, mirroring the
          // harness's NUM_PROMPTS_BY_CONC lookup with fallback max(c*2, 200):
          //   1. workload.num_prompts on the matched measurement (per-row
          //      override — most faithful to what was actually measured)
          //   2. config.benchmarkCommands.numPromptsByConc[c] (cookbook-wide
          //      canonical table)
          //   3. max(c * 2, 200) (sane default when neither is supplied)
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
    const groups = [];
    for (const [vendor, list] of Object.entries(HARDWARE_CATALOG)) {
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

  // ==========================================================================
  // 4. React state + effects
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
      // Snap to a real cell if the URL named an impossible combination
      // (stale share link, manual edit, catalog changes). The hash mirror
      // useEffect below rewrites the URL to match the validated state.
      setSel(validateSelection(config.cells, parsed));
      // Deep-link UX: when the hash explicitly carries a selection (shared
      // link, or an in-page anchor like "command panel above" pointing at a
      // specific combo), scroll the Deploy section into view so the reader
      // lands on the prefilled panel. We target the auto-slug id of the
      // Deploy heading; cookbooks title it either "## Deployment" (→ id
      // "deployment") or "## Deploy" (→ id "deploy"), so try both. If a
      // cookbook MDX uses a different heading the lookup silently no-ops.
      // NOTE: this only fires for hash navigations (link click, manual URL,
      // back/forward) — chip clicks inside the panel mirror via
      // `history.replaceState`, which does NOT trigger `hashchange`, so the
      // page never self-scrolls.
      const el = document.getElementById("deployment")
        || document.getElementById("deploy");
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    };
    hydrate();
    window.addEventListener("hashchange", hydrate);
    return () => window.removeEventListener("hashchange", hydrate);
  }, []);
  // Mirror sel → hash AND notify in-page listeners (like the §3.3 Playground).
  // `history.replaceState` does NOT fire a `hashchange` event by spec — that
  // event only fires when the user navigates to a new fragment (anchor click,
  // address-bar edit, back/forward). So we dispatch a custom event ourselves
  // and let the Playground subscribe to it; Playground also still listens to
  // `hashchange` for the manual-URL case.
  useEffect(() => {
    const target = "#" + new URLSearchParams(sel).toString();
    if (window.location.hash !== target) {
      window.history.replaceState(null, "", target);
    }
    window.dispatchEvent(new CustomEvent("sglang-deploy-sel", { detail: sel }));
  }, [sel]);

  const [modal, setModal] = useState(null); // 'curl' | 'env' | null
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
  // "⚡ Reproduce" modal state. Both Speed and Accuracy are chip-selected:
  //   benchConc — which concurrency the Speed command shows (null → the
  //               cell's first measured concurrency).
  //   benchAcc  — which eval the Accuracy command shows (null → first eval).
  //   benchCopied — which command area last flashed "✓ Copied" ("speed" /
  //               "acc"). Both stale picks fall back gracefully in the render.
  const [benchConc, setBenchConc] = useState(null);
  const [benchAcc, setBenchAcc] = useState(null);
  const [benchCopied, setBenchCopied] = useState(null);
  // Output framing: "python" emits a bare `sglang serve ...`, "docker" wraps
  // the same args in `docker run` against the per-hardware image. The Copy /
  // cURL / Env buttons all behave identically — the toggle only changes the
  // text inside the <pre>.
  const [runMode, setRunMode] = useState("python");
  useEffect(() => { if (modal === "env") setEnvDraft(env); }, [modal, env]);

  // ==========================================================================
  // 5. Derived values
  // ==========================================================================
  const s = makeStyles(isDark);
  const cell = findCell(config.cells, sel);
  const command = renderCommand(cell, sel, env, runMode);
  const modelName = resolveModelName(sel);
  const curlText = interpolate(config.curl || "", env, modelName);
  const hwGroups = buildHardwareGroups();
  // Matched benchmark entry for the current selection (null when no
  // `benchmarks` prop or no entry). Computed once; reused by both the
  // benchmark card and the "⚡ Reproduce" modal.
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
  // Per-block copy for the "⚡ Reproduce" modal — `key` identifies which block
  // flashed "✓ Copied" (so only that block's button changes label).
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

  // ==========================================================================
  // 6. JSX render
  // ==========================================================================
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
        </div>
      </div>

      {/* Benchmark card. Renders only when the cookbook MDX passed a
          `benchmarks` prop AND there is currently a matched cell. Inside
          renderBenchmarkCard the empty-state path covers "match exists
          but no measured numbers yet" — so cookbooks can ship the file
          with placeholder entries and the card stays useful. */}
      {benchmarks && cell && renderBenchmarkCard(benchEntry)}

      {/* Playground link. scrollIntoView (instead of an href anchor) so the
          URL hash — which carries the Deploy panel's selection — isn't
          overwritten. The target id "playground" is auto-generated by
          Mintlify from the "## Playground" heading slug. */}
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

      {/* "⚡ Reproduce" modal — benchmark commands for the current selection.
          Accuracy: one command per present-and-templated eval. Speed: ONE
          command, with concurrency chips that rewrite --max-concurrency /
          --num-prompts. Fill reuses `interpolate` (same machinery as cURL);
          speed merges the picked workload values into the env map. */}
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
        // Accuracy is chip-selected too — chips swap WHICH eval template shows.
        // Same stale-pick fallback as Speed (a benchAcc carried over from a
        // cell that doesn't have that eval falls back to the first one here).
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
