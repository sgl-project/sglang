// Interactive SRT Slurm recipe picker for cookbook deployment sections.
// Keep model-specific data in a plain config module and pass it through MDX.
//
// Config fields:
//   model/hardware          fixed display chips
//   workloads/strategies   selectable dimensions
//   recipes                {workload, strategy, path, prefillNodes,
//                           decodeNodes, description}[]
//   repositoryUrl/branch   source link for the selected YAML
//   recipesPath            local path used in generated srtctl commands

export const SrtSlurmDeployment = ({ config }) => {
  if (!config || !Array.isArray(config.recipes) || config.recipes.length === 0) {
    return (
      <div style={{padding: 12, color: "#b91c1c"}}>
        SRT Slurm deployment: missing recipe configuration
      </div>
    );
  }

  const makeStyles = (isDark) => ({
    container: {
      maxWidth: "900px", margin: "0 auto", display: "flex",
      flexDirection: "column", gap: "3px",
    },
    card: {
      padding: "5px 10px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#E85D4D" : "#D45D44"}`,
      borderRadius: "4px", display: "flex", alignItems: "center", gap: "10px",
      background: isDark ? "#1f2937" : "#fff",
    },
    title: {
      fontSize: "12px", fontWeight: 600, minWidth: "108px", flexShrink: 0,
      color: isDark ? "#e5e7eb" : "inherit",
    },
    itemsGrid: {
      display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))",
      gap: "4px", flex: 1,
    },
    choice: {
      padding: "3px 8px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px", cursor: "pointer", minHeight: "30px",
      display: "inline-flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", fontWeight: 500, fontSize: "12px",
      transition: "all 0.2s", userSelect: "none", textAlign: "center",
      background: isDark ? "#374151" : "#fff",
      color: isDark ? "#e5e7eb" : "inherit",
    },
    selected: { background: "#D45D44", color: "white", borderColor: "#D45D44" },
    disabled: { cursor: "not-allowed", opacity: 0.4 },
    subtitle: { fontSize: "9px", marginTop: "1px", lineHeight: 1.1, opacity: 0.75 },
    topology: {
      flex: 1, display: "flex", flexWrap: "wrap", alignItems: "center",
      justifyContent: "space-between", gap: "4px 12px", fontSize: "12px",
      color: isDark ? "#e5e7eb" : "#374151",
    },
    topologyMeta: { fontSize: "10px", color: isDark ? "#9ca3af" : "#6b7280" },
    commandWrap: {
      position: "relative", flex: 1, overflow: "hidden", borderRadius: "6px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      background: isDark ? "#111827" : "#f5f5f5",
    },
    commandHeader: {
      display: "flex", flexWrap: "wrap", justifyContent: "space-between",
      alignItems: "center", gap: "6px 10px", padding: "6px 10px",
      borderBottom: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      background: isDark ? "#1f2937" : "#fafafa",
    },
    headerLeft: { display: "inline-flex", flexWrap: "wrap", alignItems: "center", gap: "8px" },
    badge: {
      display: "inline-flex", alignItems: "center", gap: "6px",
      padding: "2px 8px", borderRadius: "10px", fontSize: "11px", fontWeight: 600,
      background: isDark ? "#064e3b" : "#d1fae5",
      color: isDark ? "#a7f3d0" : "#065f46",
    },
    badgeDot: { width: "8px", height: "8px", borderRadius: "50%", background: "#10b981" },
    runModeWrap: {
      display: "inline-flex", overflow: "hidden", borderRadius: "10px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      fontSize: "11px", fontWeight: 600, userSelect: "none",
    },
    runModeChip: (active, last) => ({
      padding: "2px 10px", cursor: "pointer",
      borderRight: last ? "none" : `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      background: active ? (isDark ? "#374151" : "#fff") : "transparent",
      color: active
        ? (isDark ? "#e5e7eb" : "#111827")
        : (isDark ? "#9ca3af" : "#6b7280"),
    }),
    iconRow: { display: "inline-flex", flexWrap: "wrap", gap: "6px" },
    iconButton: {
      padding: "4px 10px", borderRadius: "4px", cursor: "pointer",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#374151",
      fontSize: "11px", fontWeight: 500, lineHeight: 1.4,
      display: "inline-flex", alignItems: "center", gap: "4px", textDecoration: "none",
    },
    commandPre: {
      padding: "12px 16px", margin: 0, overflowX: "auto",
      fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
      fontSize: "12px", lineHeight: 1.5, whiteSpace: "pre-wrap",
      color: isDark ? "#e5e7eb" : "#374151",
    },
  });

  const first = config.recipes[0];
  const [selection, setSelection] = useState({
    workload: first.workload,
    strategy: first.strategy,
  });
  const [runMode, setRunMode] = useState("dry-run");
  const [copied, setCopied] = useState(false);
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

  const recipe = config.recipes.find(
    (item) => item.workload === selection.workload && item.strategy === selection.strategy
  ) || first;
  const styles = makeStyles(isDark);
  const recipesPath = (config.recipesPath || "../srt-slurm-recipes").replace(/\/$/, "");
  const command = [
    `uv run srtctl ${runMode} -f \\`,
    `  ${recipesPath}/${recipe.path}`,
  ].join("\n");
  const sourceUrl = `${config.repositoryUrl}/blob/${config.branch || "main"}/${recipe.path}`;
  const totalNodes = recipe.prefillNodes + recipe.decodeNodes;
  const totalGpus = totalNodes * config.hardware.gpusPerNode;

  const optionAvailable = (dimension, id) => config.recipes.some((item) => {
    const other = dimension === "workload" ? "strategy" : "workload";
    return item[dimension] === id && item[other] === selection[other];
  });

  const selectOption = (dimension, id) => {
    const next = { ...selection, [dimension]: id };
    const exact = config.recipes.find(
      (item) => item.workload === next.workload && item.strategy === next.strategy
    );
    if (exact) {
      setSelection(next);
      return;
    }
    const fallback = config.recipes.find((item) => item[dimension] === id);
    if (fallback) {
      setSelection({ workload: fallback.workload, strategy: fallback.strategy });
    }
  };

  const copyCommand = () => {
    navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  const renderOptions = (title, options, dimension) => (
    <div style={styles.card}>
      <div style={styles.title}>{title}</div>
      <div style={styles.itemsGrid}>
        {options.map((option) => {
          const selected = selection[dimension] === option.id;
          const disabled = !optionAvailable(dimension, option.id);
          return (
            <button
              type="button"
              key={option.id}
              disabled={disabled}
              aria-pressed={selected}
              style={{
                ...styles.choice,
                ...(selected ? styles.selected : {}),
                ...(disabled ? styles.disabled : {}),
              }}
              onClick={() => selectOption(dimension, option.id)}
              title={disabled ? "No official recipe for the current selection" : ""}
            >
              <span>{option.label}</span>
              {option.subtitle && (
                <span style={{...styles.subtitle, color: selected ? "rgba(255,255,255,0.85)" : "inherit"}}>
                  {option.subtitle}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );

  const renderFixed = (title, item) => (
    <div style={styles.card}>
      <div style={styles.title}>{title}</div>
      <div style={styles.itemsGrid}>
        <div style={{...styles.choice, ...styles.selected, cursor: "default"}}>
          <span>{item.label}</span>
          {item.subtitle && (
            <span style={{...styles.subtitle, color: "rgba(255,255,255,0.85)"}}>{item.subtitle}</span>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div style={styles.container} className="not-prose">
      {renderFixed("Hardware Platform", config.hardware)}
      {renderFixed("Model Variant", config.model)}
      {renderOptions("Workload", config.workloads, "workload")}
      {renderOptions("Strategy", config.strategies, "strategy")}

      <div style={styles.card}>
        <div style={styles.title}>Topology</div>
        <div style={styles.topology}>
          <strong>
            {recipe.prefillNodes} prefill {recipe.prefillNodes === 1 ? "node" : "nodes"} + {" "}
            {recipe.decodeNodes} decode {recipe.decodeNodes === 1 ? "node" : "nodes"}
          </strong>
          <span style={styles.topologyMeta}>
            {config.hardware.gpusPerNode} GPUs/node · {totalGpus} GPUs total · {recipe.description}
          </span>
        </div>
      </div>

      <div style={styles.card}>
        <div style={styles.title}>Run this Command:</div>
        <div style={styles.commandWrap}>
          <div style={styles.commandHeader}>
            <div style={styles.headerLeft}>
              <span style={styles.badge}>
                <span style={styles.badgeDot} />
                Official Recipe
              </span>
              <span style={styles.runModeWrap} role="tablist" aria-label="SRT Slurm action">
                <span
                  style={styles.runModeChip(runMode === "dry-run", false)}
                  onClick={() => setRunMode("dry-run")}
                  role="tab"
                  aria-selected={runMode === "dry-run"}
                >
                  Dry Run
                </span>
                <span
                  style={styles.runModeChip(runMode === "apply", true)}
                  onClick={() => setRunMode("apply")}
                  role="tab"
                  aria-selected={runMode === "apply"}
                >
                  Submit
                </span>
              </span>
            </div>
            <span style={styles.iconRow}>
              <button type="button" style={styles.iconButton} onClick={copyCommand}>
                {copied ? "✓ Copied" : "⧉ Copy"}
              </button>
              <a href={sourceUrl} target="_blank" rel="noopener noreferrer" style={styles.iconButton}>
                View YAML ↗
              </a>
            </span>
          </div>
          <pre style={styles.commandPre}>{command}</pre>
        </div>
      </div>
    </div>
  );
};
