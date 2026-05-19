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

export const Deployment = ({ config }) => {
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
    container: { maxWidth: "900px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "4px" },
    card: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      display: "flex", alignItems: "center", gap: "12px",
      background: isDark ? "#1f2937" : "#fff",
    },
    cardColumn: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px",
      display: "flex", flexDirection: "column", gap: "6px",
      background: isDark ? "#1f2937" : "#fff",
    },
    title: { fontSize: "13px", fontWeight: "600", minWidth: "140px", flexShrink: 0, color: isDark ? "#e5e7eb" : "inherit" },
    vendorRow: { display: "flex", alignItems: "center", gap: "8px" },
    vendorLabel: {
      fontSize: "11px", fontWeight: "600",
      color: isDark ? "#9ca3af" : "#6b7280",
      minWidth: "48px", textTransform: "uppercase", letterSpacing: "0.04em",
    },
    itemsGrid: (cols) => ({
      display: "grid", gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
      gap: "6px", flex: 1,
    }),
    labelBase: {
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px", cursor: "pointer",
      display: "inline-flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      fontWeight: "500", fontSize: "13px",
      transition: "all 0.2s", userSelect: "none",
      minHeight: "32px", textAlign: "center",
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
      const i = flags.findIndex((f) => f.startsWith("--model-path"));
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
      // (mirrors §2 "Docker Images by Hardware Platform" table). If the hw isn't
      // mapped (shouldn't happen for supported cells), we fall back to `:dev`.
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
                {cell && cell.verified ? "Verified" : "Auto-Estimated"}
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

      {/* Playground link. scrollIntoView (instead of an href anchor) so the
          URL hash — which carries §3's selection — isn't overwritten. The
          target id "3-3-playground" is auto-generated by Mintlify from the
          "### 3.3 Playground" heading slug. */}
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
            const el = document.getElementById("3-3-playground");
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
    </div>
  );
};
