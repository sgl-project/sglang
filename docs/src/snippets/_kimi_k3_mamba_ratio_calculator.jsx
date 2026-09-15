// Kimi-K3-only calculator, live-coupled to the Deploy panel: every parameter
// except the average request length is derived from the effective config the
// Playground broadcasts (base cell + Deploy overlays + Playground overrides),
// and the computed --mamba-full-memory-ratio is broadcast back for the Deploy
// panel to pin into its command.

export const KimiK3MambaRatioCalculator = () => {
  const [isDark, setIsDark] = useState(false);
  const [requestLength, setRequestLength] = useState("11264");
  const [copied, setCopied] = useState(false);
  // Effective serving config; empty until the Playground's first broadcast
  // (the parse below then falls back to the stock defaults: tp8, bf16 KV,
  // fp32 state, extra_buffer, NOSPEC, no DCP).
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
  // formula (measured dual-pool balance), written as a per-request cost ratio:
  //
  //   r = (S + D) x state_bytes / (L x per_token_kv_bytes)
  //
  // The state side (S main slots plus D verify intermediates) is per-GPU and
  // never DCP-sharded. The KV side is per-GPU per logical token: DCP shards the
  // MLA latent KV across its ranks, while the DSPARK draft model's own KV is
  // replicated on every rank, so it stays a flat term. Without DCP the draft
  // term is ~10% noise; once DCP shards MLA it is the same order as the MLA
  // share, which is why it cannot be folded into a plain x dcp factor.
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
    const dp = hasFlag("--enable-dp-attention") ? Number(flagArg("--dp-size")) || 1 : 1;
    // KDA state and the replicated MLA KV both live per attention-TP group.
    // PP needs no term: it splits the layers of both pools equally.
    const attnTp = Math.max(1, Math.round(tp / dp));
    const dcp = Number(flagArg("--dcp-size")) || 1;
    const kvDtype = flagArg("--kv-cache-dtype") === "fp8_e4m3" ? "fp8_e4m3" : "bfloat16";
    const specOn = hasFlag("--speculative-algorithm");
    const replaySpec = hasFlag("--enable-linear-replayssm-spec");
    // ReplaySSM removes the D intermediate states but does NOT pin the state
    // dtype: an unset --mamba-ssm-dtype defaults to fp32 either way, and an
    // explicit 16-bit state is accepted (warned for drift), so read the flag.
    const ssmFlag = flagArg("--mamba-ssm-dtype");
    const ssmDtype =
      ssmFlag === "bfloat16" || ssmFlag === "float16" ? ssmFlag : "float32";
    const radixOff = hasFlag("--disable-radix-cache");
    // "auto" (and anything unrecognized) resolves to extra_buffer.
    const strategyFlag = flagArg("--mamba-radix-cache-strategy");
    const strategy =
      strategyFlag === "no_buffer" || strategyFlag === "extra_buffer_lazy"
        ? strategyFlag
        : "extra_buffer";
    const skipLock = env.some((e) => e.startsWith("SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"));
    const pdRole = flagArg("--disaggregation-mode");
    // Pipeline parallelism is incompatible with the overlap scheduler, so pp > 1
    // turns it off for you — and the track buffer then costs one slot, not two.
    const overlapOff =
      hasFlag("--disable-overlap-schedule") || (Number(flagArg("--pp-size")) || 1) > 1;
    // Mirrors kv_cache_configurator._calculate_mamba_ratio: base 3, minus 1 under
    // the decode-lock skip, plus the ping-pong track buffer (2 under the overlap
    // scheduler, 1 for lazy or without overlap). no_buffer has no track buffer and
    // adds the skip's drop back, so it stays 3; a disabled radix cache is 1.
    // A PD decode server runs a chunk cache: one live slot per request, and the
    // radix-strategy knobs are inert there.
    const slots = pdRole === "decode"
      ? 1
      : radixOff
        ? 1
        : strategy === "no_buffer"
          ? 3
          : 3 -
            (skipLock ? 1 : 0) +
            (overlapOff || strategy === "extra_buffer_lazy" ? 1 : 2);
    const block = Number(flagArg("--speculative-dspark-block-size")) || 7;
    const drafts = specOn && !replaySpec && pdRole !== "prefill" ? block + 1 : 0;

    const ssmBytes = ssmDtype === "float32" ? 4 : 2;
    const kvBytes = kvDtype === "fp8_e4m3" ? 1 : 2;
    // Fixed K3 geometry:
    // KDA: 69 layers, 96 heads, head_dim 128, conv kernel 4 (conv state always bf16).
    // MLA: 24 layers, kv_lora_rank 512, qk_rope_head_dim 64.
    const stateBytesPerSlot =
      69 * ((96 / attnTp) * 128 * 128 * ssmBytes + 3 * 3 * (96 / attnTp) * 128 * 2);
    const kvBytesPerToken = 24 * (512 + 64) * kvBytes;
    // DCP shards the MLA latent KV across its ranks; the DSPARK draft model's KV
    // is replicated on every rank (~1.4 KB/token on trtllm_mha), so it does not
    // shard and is added flat.
    const draftKvBytesPerToken = specOn ? 1400 : 0;
    const kvBytesPerTokenPerRank = kvBytesPerToken / dcp + draftKvBytesPerToken;
    const ratio =
      ((slots + drafts) * stateBytesPerSlot) / (kvBytesPerTokenPerRank * length);
    return { ratio, tp, dp, attnTp, dcp, kvDtype, ssmDtype, radixOff, strategy, skipLock, slots, specOn, replaySpec, block, pdRole };
  };

  // Two evaluations: `eff` matches the Playground's composed command, `bs`
  // matches the Deploy command (cell + overlays only).
  const eff = derive(cfg.flags, cfg.env);
  const bs = derive(cfg.baseFlags.length ? cfg.baseFlags : cfg.flags,
                    cfg.baseFlags.length ? cfg.baseEnv : cfg.env);
  // A --max-mamba-cache-size cell sizes the pool explicitly — no ratio to
  // compute or broadcast.
  const explicitSizing = (cfg.baseFlags.length ? cfg.baseFlags : cfg.flags)
    .some((f) => f.startsWith("--max-mamba-cache-size"));
  const { ratio, tp, dp, attnTp, dcp, kvDtype, ssmDtype, radixOff, strategy, skipLock, slots, specOn, replaySpec, block, pdRole } = eff;
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
  // the Playground's composed command takes the effective one.
  useEffect(() => {
    if (explicitSizing) return;
    window.dispatchEvent(
      new CustomEvent("sglang-k3-mamba-ratio", {
        detail: {
          ratio: valid ? result : null,
          baseRatio: baseValid ? baseResult : null,
        },
      })
    );
  }, [result, valid, baseResult, baseValid, explicitSizing]);

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
    : replaySpec
      ? "DSPARK + ReplaySSM (D folded)"
      : `DSPARK (D = ${block + 1})`;
  const derivedChips = [
    // With DP attention on, show the whole topology so a large-scale preset is
    // visibly understood: total GPUs = DP replicas x attention-TP group width.
    dp > 1
      ? `${tp} GPUs = DP ${dp} × Attention TP ${attnTp}`
      : `Attention TP ${attnTp}`,
    `DCP ${dcp}`,
    `KV ${kvDtype === "fp8_e4m3" ? "FP8" : "BF16"}`,
    `State ${ssmDtype === "float32" ? "FP32" : ssmDtype === "bfloat16" ? "BF16" : "FP16"}`,
    pdRole === "decode"
      ? "PD decode: chunk cache (S = 1)"
      : radixOff
        ? "Radix off (S = 1)"
        : `${strategy}${skipLock ? " + slot saving" : ""} (S = ${slots})`,
    pdRole === "prefill" ? "PD prefill (no verify states)" : null,
    specLabel,
  ].filter(Boolean);

  if (explicitSizing) {
    return (
      <div className="not-prose" style={{ padding: "14px", border: `1px solid ${colors.border}`, borderRadius: "8px", background: colors.panel, color: colors.muted, fontSize: "13px" }}>
        This recipe sizes the KDA state pool explicitly with <code style={{ color: colors.text }}>--max-mamba-cache-size</code>, so the ratio calculator does not apply.
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
        <label htmlFor="k3-ratio-length" style={labelStyle}>
          Average request length
          <input
            id="k3-ratio-length"
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
