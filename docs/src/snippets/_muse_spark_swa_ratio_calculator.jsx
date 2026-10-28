export const MuseSparkSwaRatioCalculator = ({ calculator }) => {
  const [contextLength, setContextLength] = useState("100000");
  const [isDark, setIsDark] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");

  useEffect(() => {
    const root = document.documentElement;
    const updateTheme = () => {
      setIsDark(
        root.classList.contains("dark") ||
        root.getAttribute("data-theme") === "dark" ||
        root.style.colorScheme === "dark"
      );
    };
    updateTheme();
    const observer = new MutationObserver(updateTheme);
    observer.observe(root, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"],
    });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (copyStatus !== "Copied") return;
    const timer = window.setTimeout(() => setCopyStatus(""), 2000);
    return () => window.clearTimeout(timer);
  }, [copyStatus]);

  const ratio = calculator.calculate(contextLength);
  const valid = ratio !== null;
  const result = valid ? Number(ratio.toPrecision(8)).toString() : "—";
  const cliFlag = valid ? `--swa-full-tokens-ratio ${result}` : "";
  const error = valid
    ? ""
    : contextLength.trim() !== "" && Number(contextLength) === 0
      ? "At 0 tokens, the ratio is undefined. Enter a positive context length."
      : `Enter a whole number from 1 to ${calculator.maxContextLength.toLocaleString("en-US")}.`;

  const copyFlag = async () => {
    if (!cliFlag) return;
    try {
      await navigator.clipboard.writeText(cliFlag);
      setCopyStatus("Copied");
    } catch {
      setCopyStatus("Copy failed — select the flag to copy it manually.");
    }
  };

  const colors = {
    border: isDark ? "#374151" : "#e5e7eb",
    panel: isDark ? "#1f2937" : "#ffffff",
    input: isDark ? "#111827" : "#f8fafc",
    text: isDark ? "#e5e7eb" : "#1f2937",
    muted: isDark ? "#9ca3af" : "#64748b",
    accent: "#B94332",
    error: isDark ? "#fca5a5" : "#b91c1c",
  };

  return (
    <div
      className="not-prose"
      style={{
        display: "grid",
        gap: "12px",
        padding: "14px",
        border: `1px solid ${colors.border}`,
        borderRadius: "8px",
        background: colors.panel,
        color: colors.text,
      }}
    >
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "16px" }}>
        <label
          htmlFor="muse-swa-context-length"
          style={{ display: "flex", flexDirection: "column", gap: "5px", flex: "1 1 200px", fontSize: "12px", fontWeight: 600 }}
        >
          Context length (input + output tokens)
          <input
            id="muse-swa-context-length"
            type="number"
            min={0}
            max={calculator.maxContextLength}
            step={1}
            value={contextLength}
            aria-invalid={!valid}
            aria-describedby="muse-swa-range muse-swa-error"
            onChange={(event) => {
              setContextLength(event.target.value);
              setCopyStatus("");
            }}
            style={{
              width: "100%",
              boxSizing: "border-box",
              padding: "8px 10px",
              border: `1px solid ${colors.border}`,
              borderRadius: "5px",
              background: colors.input,
              color: colors.text,
              fontSize: "13px",
            }}
          />
        </label>
        <div style={{ flex: "1 1 200px", fontSize: "12px", color: colors.muted, lineHeight: 1.7 }}>
          <div>SWA window: {calculator.windowSize.toLocaleString("en-US")} tokens</div>
          <div id="muse-swa-range">Range: 0–1M (1M = {calculator.maxContextLength.toLocaleString("en-US")})</div>
        </div>
      </div>
      <div id="muse-swa-error" role="status" style={{ color: colors.error, fontSize: "12px", display: error ? "block" : "none" }}>
        {error}
      </div>
      <div
        aria-live="polite"
        aria-atomic="true"
        style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px", paddingTop: "12px", borderTop: `1px solid ${colors.border}` }}
      >
        <div>
          <div style={{ fontSize: "11px", color: colors.muted }}>Theoretical balanced ratio</div>
          <div style={{ fontSize: "26px", fontWeight: 700 }}>{result}</div>
        </div>
        {valid && (
          <code style={{ flex: "1 1 240px", minWidth: 0, padding: "8px 10px", borderRadius: "5px", background: colors.input, fontSize: "12px", overflowWrap: "anywhere" }}>
            {cliFlag}
          </code>
        )}
        <button
          type="button"
          disabled={!valid}
          onClick={copyFlag}
          style={{
            padding: "7px 11px",
            border: 0,
            borderRadius: "5px",
            background: colors.accent,
            color: "#ffffff",
            fontSize: "12px",
            fontWeight: 600,
            cursor: valid ? "pointer" : "not-allowed",
            opacity: valid ? 1 : 0.5,
          }}
        >
          {copyStatus === "Copied" ? "Copied" : "Copy flag"}
        </button>
      </div>
      <div role="status" style={{ color: colors.muted, fontSize: "12px", display: copyStatus && copyStatus !== "Copied" ? "block" : "none" }}>
        {copyStatus}
      </div>
    </div>
  );
};
