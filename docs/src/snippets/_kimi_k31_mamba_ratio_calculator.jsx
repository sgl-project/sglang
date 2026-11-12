// Kimi-K3.1 calculator (Kimi-K3 calculator with K3.1's DSA KV geometry, the DFLASH draft and
// its KV ring), live-coupled to the Deploy panel: every parameter
// except the average request length is derived from the effective config the
// Playground broadcasts (base cell + Deploy overlays + Playground overrides),
// and the computed --mamba-full-memory-ratio is broadcast back for the Deploy
// panel to pin into its command.

export const KimiK31MambaRatioCalculator = () => {
  const [isDark, setIsDark] = useState(false);
  const [requestLength, setRequestLength] = useState("32768");
  const [copied, setCopied] = useState(false);
  // Effective serving config; empty until the Playground's first broadcast
  // (the parse below then falls back to the stock defaults: tp8, fp32 state,
  // extra_buffer, NOSPEC, no DCP; K3.1 KV is always fp8).
  const [cfg, setCfg] = useState({ flags: [], env: [], baseFlags: [], baseEnv: [] });

  useEffect(() => {
    const checkTheme = () => {
      const html = document.documentElement;
      setIsDark(
        html.classList.contains("dark") ||
          html.getAttribute("data-theme") === "dark" ||
          html.style.colorScheme === "dark"
      );
    };
    checkTheme();
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"],
    });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onCfg = (e) =>
      setCfg({
        flags: (e.detail && e.detail.flags) || [],
        env: (e.detail && e.detail.env) || [],
        baseFlags: (e.detail && e.detail.baseFlags) || [],
        baseEnv: (e.detail && e.detail.baseEnv) || [],
      });
    window.addEventListener("sglang-k3-effective-config", onCfg);
    return () => window.removeEventListener("sglang-k3-effective-config", onCfg);
  }, []);

  const length = Number.parseFloat(requestLength);

  // Derive the serving parameters from a flag/env list and evaluate the balance
  // formula, written as a per-request cost ratio:
  //
  //   r = ((S + D) x state_bytes + ring_bytes) / (L x per_token_kv_bytes)
  //
  // The state side (S main slots plus D verify intermediates) is per-GPU and never
  // DCP-sharded. The KV side is per-GPU per logical token: K3.1 stores fp8 MLA KV plus
  // the DSA indexer K on every MLA layer, and DCP shards both by token owner. The
  // DFLASH SWA draft keeps a per-request ring (window + block rows) instead of a
  // per-token draft table, so with the ring on it is a per-request term.
  const derive = (flags, env) => {
    const flagArg = (name) => {
      for (const f of flags) {
        const parts = f.split(/\s+/);
        if (parts[0] === name) return parts[1];
      }
      return null;
    };
    const hasFlag = (name) => flags.some((f) => f.split(/[\s=]/)[0] === name);

    const tp = Number(flagArg("--tp-size")) || 8;
    const dp = Number(flagArg("--attn-dp-size")) || 1;
    // KDA heads shard over the attention-TP group; PP needs no term (it splits the
    // layers of both pools alike); PCP replicates the MLA KV like TP.
    const attnTp = Math.max(1, Math.round(tp / dp));
    const dcp = Number(flagArg("--dcp-size")) || 1;
    const dsaBackend = flagArg("--dsa-decode-backend") || flagArg("--dsa-prefill-backend") || "flashmla_kv";
    const specOn = hasFlag("--speculative-algorithm");
    const replaySpec = hasFlag("--enable-linear-replayssm-spec");
    const ssmFlag = flagArg("--mamba-ssm-dtype");
    const ssmDtype =
      ssmFlag === "bfloat16" || ssmFlag === "float16" ? ssmFlag : "float32";
    const radixOff = hasFlag("--disable-radix-cache");
    const strategyFlag = flagArg("--mamba-radix-cache-strategy");
    const strategy =
      strategyFlag === "no_buffer" || strategyFlag === "extra_buffer_lazy"
        ? strategyFlag
        : "extra_buffer";
    const skipLock = env.some((e) => e.startsWith("SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"));
    const pdRole = flagArg("--disaggregation-mode");
    const decodeRadix = hasFlag("--disaggregation-decode-enable-radix-cache");
    const overlapOff =
      hasFlag("--disable-overlap-schedule") || (Number(flagArg("--pp-size")) || 1) > 1;
    // Same slot rule as Kimi-K3 (kv_cache_configurator._calculate_mamba_ratio); a PD
    // decode server runs without the radix cache unless it is turned back on.
    const slots = (pdRole === "decode" && !decodeRadix) || radixOff
      ? 1
      : strategy === "no_buffer"
        ? 3
        : 3 -
          (skipLock ? 1 : 0) +
          (overlapOff || strategy === "extra_buffer_lazy" ? 1 : 2);
    const block =
      Number(flagArg("--speculative-dflash-block-size")) ||
      Number(flagArg("--speculative-num-draft-tokens")) || 8;
    const drafts = specOn && !replaySpec && pdRole !== "prefill" ? block : 0;
    const ringOn =
      specOn &&
      !pdRole &&
      ["cutedsl_mla", "trtllm_mla"].includes(flagArg("--speculative-draft-attention-backend")) &&
      !hasFlag("--enable-unified-memory") &&
      !env.some((e) => e.startsWith("SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=0"));

    const ssmBytes = ssmDtype === "float32" ? 4 : 2;
    // Fixed K3.1 geometry:
    // KDA: 69 layers, 96 heads, head_dim 128, conv kernel 4 (conv state always bf16).
    // DSA/MLA: 24 layers; fp8 latent KV 656 B (flashmla_kv) or 576 B (trtllm / prims_ts)
    // per layer, plus a 132 B indexer K.
    // DFLASH draft: 5 MLA layers at 656 B, sliding window 1024.
    const stateBytesPerSlot =
      69 * ((96 / attnTp) * 128 * 128 * ssmBytes + 3 * 3 * (96 / attnTp) * 128 * 2);
    const mlaBytes = dsaBackend === "flashmla_kv" ? 656 : 576;
    const kvBytesPerTokenPerRank = (24 * (mlaBytes + 132)) / dcp +
      (specOn && !ringOn ? (5 * 656) / dcp : 0);
    const ringRowsPerReq = Math.ceil((1024 + block) / dcp) + 1;
    const ringBytes = specOn && ringOn ? ringRowsPerReq * 5 * 656 : 0;
    const ratio =
      ((slots + drafts) * stateBytesPerSlot + ringBytes) / (kvBytesPerTokenPerRank * length);
    return { ratio, tp, dp, attnTp, dcp, dsaBackend, ssmDtype, radixOff, strategy, skipLock, slots, specOn, replaySpec, block, pdRole, ringOn };
  };

  // Two evaluations: `eff` matches the Playground's composed command, `bs`
  // matches the Deploy command (cell + overlays only).
  const eff = derive(cfg.flags, cfg.env);
  const bs = derive(cfg.baseFlags.length ? cfg.baseFlags : cfg.flags,
                    cfg.baseFlags.length ? cfg.baseEnv : cfg.env);
  // Recipes that size the dual pool without the ratio neither render one nor
  // broadcast one:
  //   - a --max-mamba-cache-size cell pins the KDA slot count explicitly;
  //   - the Ascend NPU and TPU recipes size both pools internally and never set
  //     --mamba-full-memory-ratio (every such cell carries --device npu|tpu).
  const baseFlagList = cfg.baseFlags.length ? cfg.baseFlags : cfg.flags;
  const explicitSizing = baseFlagList.some((f) => f.startsWith("--max-mamba-cache-size"));
  // A cell that pins the measured ratio keeps it; the Deploy panel never overrides it.
  const pinnedRatio = baseFlagList.some((f) => f.startsWith("--mamba-full-memory-ratio"));
  // A PD decode server sizes the KDA pool from --max-running-requests (plus extra slots),
  // which every decode cell pins; a ratio would override that sizing.
  const decodeRole = baseFlagList.some((f) => f.trim() === "--disaggregation-mode decode");
  const deviceRecipe = baseFlagList.some((f) => {
    const [head, ...rest] = f.trim().split(/[\s=]+/);
    return head === "--device" && ["npu", "tpu"].includes(rest[0]);
  });
  const ratioNotApplicable = explicitSizing || deviceRecipe || pinnedRatio || decodeRole;
  const { ratio, tp, dp, attnTp, dcp, dsaBackend, ssmDtype, radixOff, strategy, skipLock, slots, specOn, replaySpec, block, pdRole, ringOn } = eff;
  const valid = Number.isFinite(ratio) && ratio > 0 && length > 0 && 96 % attnTp === 0;
  const baseValid = Number.isFinite(bs.ratio) && bs.ratio > 0 && length > 0;

  const formatRatio = (value) => {
    if (!Number.isFinite(value)) return "—";
    if (value >= 10) return value.toFixed(1).replace(/\.0$/, "");
    if (value >= 1) return value.toFixed(2).replace(/\.?0+$/, "");
    return Number(value.toPrecision(2)).toString();
  };

  const result = valid ? formatRatio(ratio) : "—";
  const baseResult = baseValid ? formatRatio(bs.ratio) : "—";
  const cliFlag = valid ? `--mamba-full-memory-ratio ${result}` : "";

  // Broadcast both results: the Deploy command takes the base-config value,
  // the Playground's composed command takes the effective one. On a recipe
  // that does not use the ratio, broadcast nulls instead of skipping the
  // dispatch — the panels must drop a ratio pinned for a recipe the reader
  // has since left rather than keep injecting it.
  useEffect(() => {
    window.dispatchEvent(
      new CustomEvent("sglang-k3-mamba-ratio", {
        detail: ratioNotApplicable
          ? { ratio: null, baseRatio: null }
          : {
              ratio: valid ? result : null,
              baseRatio: baseValid ? baseResult : null,
            },
      })
    );
  }, [result, valid, baseResult, baseValid, ratioNotApplicable]);

  const copyFlag = () => {
    if (!cliFlag || typeof navigator === "undefined" || !navigator.clipboard) return;
    navigator.clipboard.writeText(cliFlag);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  const colors = {
    border: isDark ? "#374151" : "#e5e7eb",
    panel: isDark ? "#1f2937" : "#ffffff",
    input: isDark ? "#111827" : "#f8fafc",
    text: isDark ? "#e5e7eb" : "#1f2937",
    muted: isDark ? "#9ca3af" : "#64748b",
    accent: isDark ? "#E85D4D" : "#D45D44",
    error: isDark ? "#fca5a5" : "#b91c1c",
  };

  const inputStyle = {
    width: "100%",
    boxSizing: "border-box",
    padding: "8px 10px",
    border: `1px solid ${colors.border}`,
    borderRadius: "5px",
    background: colors.input,
    color: colors.text,
    fontSize: "13px",
  };
  const labelStyle = {
    display: "flex",
    flexDirection: "column",
    gap: "5px",
    fontSize: "12px",
    fontWeight: 600,
  };
  const chipStyle = {
    padding: "3px 9px",
    border: `1px solid ${colors.border}`,
    borderRadius: "999px",
    background: colors.input,
    color: colors.text,
    fontSize: "12px",
    whiteSpace: "nowrap",
  };

  const specLabel = !specOn
    ? "NOSPEC"
    : `DFLASH block ${block}${replaySpec ? " + ReplaySSM (D folded)" : pdRole === "prefill" ? "" : ` (D = ${block})`}${ringOn ? " · draft ring" : ""}`;
  const derivedChips = [
    // With DP attention on, show the whole topology so a large-scale preset is
    // visibly understood: total GPUs = attention DP size x attention-TP group width.
    dp > 1
      ? `${tp} GPUs = Attention DP ${dp} × Attention TP ${attnTp}`
      : `Attention TP ${attnTp}`,
    `DCP ${dcp}`,
    `KV FP8 · ${dsaBackend}`,
    `State ${ssmDtype === "float32" ? "FP32" : ssmDtype === "bfloat16" ? "BF16" : "FP16"}`,
    pdRole === "decode" && slots === 1
      ? "PD decode: radix off (S = 1)"
      : radixOff
        ? "Radix off (S = 1)"
        : `${strategy}${skipLock ? " + slot saving" : ""} (S = ${slots})`,
    pdRole === "prefill" ? "PD prefill (no verify states)" : null,
    specLabel,
  ].filter(Boolean);

  if (ratioNotApplicable) {
    return (
      <div className="not-prose" style={{ padding: "14px", border: `1px solid ${colors.border}`, borderRadius: "8px", background: colors.panel, color: colors.muted, fontSize: "13px" }}>
        {deviceRecipe
          ? <>This platform's recipe sizes the KDA state and KV pools itself, so the ratio calculator does not apply.</>
          : explicitSizing
          ? <>This recipe sizes the KDA state pool explicitly with <code style={{ color: colors.text }}>--max-mamba-cache-size</code>, so the ratio calculator does not apply.</>
          : pinnedRatio
            ? <>This recipe pins the <code style={{ color: colors.text }}>--mamba-full-memory-ratio</code> it was measured with, so the calculator does not override it.</>
            : <>A PD decode server sizes the KDA state pool from <code style={{ color: colors.text }}>--max-running-requests</code> and <code style={{ color: colors.text }}>--disaggregation-decode-extra-slots</code>, so the ratio calculator does not apply.</>}
      </div>
    );
  }

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
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(180px, 260px) 1fr",
          gap: "14px",
          alignItems: "start",
        }}
      >
        <label htmlFor="k31-ratio-length" style={labelStyle}>
          Average request length
          <input
            id="k31-ratio-length"
            type="number"
            min="1"
            step="1"
            value={requestLength}
            onChange={(event) => setRequestLength(event.target.value)}
            style={inputStyle}
          />
          <span style={{ color: colors.muted, fontSize: "11px", fontWeight: 400 }}>
            Input + output tokens — the only free parameter
          </span>
        </label>

        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <span style={{ fontSize: "12px", fontWeight: 600 }}>
            Serving configuration (follows the Deploy panel and Playground)
          </span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {derivedChips.map((c) => (
              <span key={c} style={chipStyle}>{c}</span>
            ))}
          </div>
        </div>
      </div>

      {!valid ? (
        <div style={{ color: colors.error, fontSize: "12px" }}>
          Enter a valid request length.
        </div>
      ) : (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            paddingTop: "12px",
            borderTop: `1px solid ${colors.border}`,
            flexWrap: "wrap",
          }}
        >
          <div>
            <div style={{ color: colors.muted, fontSize: "11px" }}>
              Balanced ratio — pinned into the commands above
            </div>
            <div style={{ fontSize: "26px", fontWeight: 700 }}>{result}</div>
            {baseResult !== result ? (
              <div style={{ color: colors.muted, fontSize: "11px" }}>
                Deploy command (without Playground overrides): {baseResult}
              </div>
            ) : null}
          </div>
          <code style={{ flex: 1, minWidth: "240px", color: colors.text }}>
            {cliFlag}
          </code>
          <button
            type="button"
            onClick={copyFlag}
            style={{
              padding: "7px 11px",
              border: 0,
              borderRadius: "5px",
              background: colors.accent,
              color: "#ffffff",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {copied ? "Copied" : "Copy flag"}
          </button>
        </div>
      )}
    </div>
  );
};
