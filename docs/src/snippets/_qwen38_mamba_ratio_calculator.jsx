// Qwen3.8-27B-only calculator, live-coupled to the Deploy panel (same wiring as
// _kimi_k3_mamba_ratio_calculator.jsx): every serving parameter except the
// average request length and the target concurrency is derived from the
// effective config the Playground broadcasts (base cell + Deploy overlays +
// Playground overrides), and the computed --mamba-full-memory-ratio is
// broadcast back for the Deploy panel to pin into its command.
//
// Geometry constants are validated against boot logs on RTX PRO 6000
// (state 153.9 MB/slot at fp32, KV 32.8 KB/token at fp8). Byte-exact against
// the boot log once the pool's +1 padding slot is counted: the log's
// "ssm_state size: 27.00GB" at max_mamba_cache_size 191 is 192 slots x
// 150,994,944 B = 27.0000 GiB — divide by 191 and you get the wrong 154.7.

export const Qwen38MambaRatioCalculator = () => {
  const [isDark, setIsDark] = useState(false);
  const [requestLength, setRequestLength] = useState("5120");
  const [targetConcurrency, setTargetConcurrency] = useState("64");
  const [copied, setCopied] = useState(false);
  // Effective serving config; empty until the Playground's first broadcast
  // (the parse below then falls back to the stock defaults: fp32 state,
  // extra_buffer, no spec).
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

  // The Deploy panel's live selection. Needed because the effective-config
  // broadcast carries the RAW cell flags — `--model-path` is still the
  // unresolved `{{MODEL_NAME}}` there — so the checkpoint precision, which
  // decides what `--kv-cache-dtype auto` resolves to, is only knowable from the
  // selection's quant.
  const [quant, setQuant] = useState("nvfp4-bf16-head");
  useEffect(() => {
    const onSel = (e) => {
      if (e.detail && e.detail.quant) setQuant(e.detail.quant);
    };
    window.addEventListener("sglang-deploy-sel", onSel);
    return () => window.removeEventListener("sglang-deploy-sel", onSel);
  }, []);

  const L = Number.parseFloat(requestLength);
  const C = Number.parseFloat(targetConcurrency);

  // Derive the serving parameters from a flag list and evaluate the balance
  // formula, written as a per-request cost ratio:
  //
  //   r = (S + D) x state_bytes / (L x kv_bytes_per_token)
  //
  const derive = (flags, env) => {
    const flagArg = (name) => {
      for (const f of flags) {
        const parts = f.split(/\s+/);
        if (parts[0] === name) return parts[1];
      }
      return null;
    };
    const hasFlag = (name) => flags.some((f) => f.split(/[\s=]/)[0] === name);

    // Read both spellings: the cookbook convention is --tp, but some configs
    // still emit --tp-size. The geometry below is TP1-only (this is a
    // single-GPU page and no cell carries a TP flag), so anything else is
    // reported as out of range rather than silently mis-computed.
    const tp = Number(flagArg("--tp")) || Number(flagArg("--tp-size")) || 1;

    // The NVFP4 checkpoint declares kv_cache_quant_algo: FP8, so the default
    // --kv-cache-dtype auto lands on fp8_e4m3 there with no flag present; the
    // BF16 / FP8 checkpoints keep a bf16 KV pool under the same default. An
    // explicit flag always wins over the checkpoint's declaration.
    const kvFlag = flagArg("--kv-cache-dtype");
    const kvDtype =
      kvFlag === "fp8_e4m3"
        ? "fp8_e4m3"
        : kvFlag === "bfloat16" || kvFlag === "bf16"
          ? "bfloat16"
          : String(quant).startsWith("nvfp4")
            ? "fp8_e4m3"
            : "bfloat16";

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

    // S mirrors kv_cache_configurator._calculate_mamba_ratio: base 3, minus 1
    // when SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1, plus the ping-pong track
    // buffer (2 under the overlap scheduler, 1 for extra_buffer_lazy or with
    // overlap off). no_buffer has no track buffer and no decode lock, so it
    // stays 3; with the radix cache disabled S is 1. At the stock defaults:
    // extra_buffer=5, extra_buffer_lazy=4, no_buffer=3, radix off=1.
    const skipLock = env.some((e) =>
      e.startsWith("SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"));
    const overlapOff =
      hasFlag("--disable-overlap-schedule") ||
      (Number(flagArg("--pp-size")) || 1) > 1;
    const slots = radixOff
      ? 1
      : strategy === "no_buffer"
        ? 3
        : 3 -
          (skipLock ? 1 : 0) +
          (overlapOff || strategy === "extra_buffer_lazy" ? 1 : 2);

    // D: verify intermediate states under speculative decoding, 0 with spec
    // off.
    const specOn = hasFlag("--speculative-algorithm");
    const algo = (flagArg("--speculative-algorithm") || "").toUpperCase();
    // ReplaySSM spec-verify keeps the verify intermediates on a fixed ring
    // rather than per-request state slots, so D is 0 even with spec on.
    const replaySpec = hasFlag("--enable-linear-replayssm-spec");
    // DSPARK takes no --speculative-num-draft-tokens: its verify window is
    // --speculative-dspark-block-size (gamma) + 1, and gamma is read from the
    // draft checkpoint when the flag is omitted — block_size 7 for
    // RadixArk/Qwen3.8-27B-DSpark, so D = 8.
    const dsparkBlock = Number(flagArg("--speculative-dspark-block-size")) || 7;
    const drafts = !specOn || replaySpec
      ? 0
      : algo === "DSPARK"
        ? dsparkBlock + 1
        : Number(flagArg("--speculative-num-draft-tokens")) || 4;

    // Fixed Qwen3.8-27B geometry (TP1):
    // GDN: 48 layers, 48 value heads x 128 x 128 SSM state (--mamba-ssm-dtype),
    //      conv state 10240 x 3 always bf16.
    // Full attention: 16 layers, GQA 4 kv heads x head_dim 256, K+V.
    const ssmBytes = ssmDtype === "float32" ? 4 : 2;
    const kvBytes = kvDtype === "fp8_e4m3" ? 1 : 2;
    const stateBytesPerSlot = 48 * (48 * 128 * 128 * ssmBytes + 10240 * 3 * 2);
    const kvBytesPerToken = 16 * 4 * 256 * 2 * kvBytes;

    const ratio = ((slots + drafts) * stateBytesPerSlot) / (kvBytesPerToken * L);
    return { ratio, tp, kvDtype, ssmDtype, radixOff, strategy, slots, specOn,
             drafts, stateBytesPerSlot, kvBytesPerToken };
  };

  // Two evaluations: `eff` matches the Playground's composed command, `bs`
  // matches the Deploy command (cell + overlays only).
  const eff = derive(cfg.flags, cfg.env);
  const bs = derive(
    cfg.baseFlags.length ? cfg.baseFlags : cfg.flags,
    // The env rides with the command it belongs to: a base command with an
    // empty env must not inherit the Playground's overlay env.
    cfg.baseFlags.length ? cfg.baseEnv : cfg.env,
  );
  const { ratio, tp, kvDtype, ssmDtype, radixOff, strategy, slots, specOn,
          drafts, stateBytesPerSlot, kvBytesPerToken } = eff;

  const valid = Number.isFinite(ratio) && ratio > 0 && L > 0 && tp === 1;
  const baseValid = Number.isFinite(bs.ratio) && bs.ratio > 0 && L > 0 && bs.tp === 1;
  // The engine divides the state pool by S alone (kv_cache_configurator.py:
  // mamba_cap = max_mamba_cache_size // _calculate_mamba_ratio()) and sizes
  // the speculative verify buffer separately from D, so the pin is C x S,
  // not C x (S + D).
  const pin = Math.ceil(C * slots);
  const pinValid = valid && Number.isFinite(pin) && pin > 0 && C > 0;

  const formatRatio = (value) => (Math.round(value * 100) / 100).toString();
  const ratioStr = valid ? formatRatio(ratio) : "—";
  const baseRatioStr = baseValid ? formatRatio(bs.ratio) : "—";

  const flagText = valid
    ? pinValid
      ? `--mamba-full-memory-ratio ${ratioStr}   # or: --max-mamba-cache-size ${pin}`
      : `--mamba-full-memory-ratio ${ratioStr}`
    : "";

  // Broadcast both results: the Deploy command takes the base-config value,
  // the Playground's composed command takes the effective one.
  useEffect(() => {
    window.dispatchEvent(
      new CustomEvent("sglang-k3-mamba-ratio", {
        detail: {
          ratio: valid ? ratioStr : null,
          baseRatio: baseValid ? baseRatioStr : null,
        },
      })
    );
  }, [ratioStr, valid, baseRatioStr, baseValid]);

  const copy = () => {
    if (!valid) return;
    navigator.clipboard.writeText(flagText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
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

  // Everything the ratio depends on except L and the concurrency target, read
  // back from the panels so the reader can see what the number was derived from.
  const derivedChips = [
    `KV ${kvDtype === "fp8_e4m3" ? "FP8" : "BF16"}`,
    `State ${ssmDtype === "float32" ? "FP32" : ssmDtype === "bfloat16" ? "BF16" : "FP16"}`,
    radixOff ? "Radix off (S = 1)" : `${strategy} (S = ${slots})`,
    specOn ? `Spec on (D = ${drafts})` : "NOSPEC",
  ];

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
          gridTemplateColumns: "minmax(180px, 240px) minmax(150px, 200px) 1fr",
          gap: "14px",
          alignItems: "start",
        }}
      >
        <label htmlFor="qwen38-ratio-length" style={labelStyle}>
          Average request length
          <input
            id="qwen38-ratio-length"
            type="number"
            min="1"
            step="1"
            value={requestLength}
            onChange={(event) => setRequestLength(event.target.value)}
            style={inputStyle}
          />
          <span style={{ color: colors.muted, fontSize: "11px", fontWeight: 400 }}>
            Input + output tokens — the ratio's only free parameter
          </span>
        </label>

        <label htmlFor="qwen38-ratio-concurrency" style={labelStyle}>
          Target concurrency
          <input
            id="qwen38-ratio-concurrency"
            type="number"
            min="1"
            step="1"
            value={targetConcurrency}
            onChange={(event) => setTargetConcurrency(event.target.value)}
            style={inputStyle}
          />
          <span style={{ color: colors.muted, fontSize: "11px", fontWeight: 400 }}>
            Requests in flight — sizes the explicit pin, not the ratio
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
          {tp !== 1
            ? `This calculator models the single-GPU (TP1) geometry; the panels are at TP${tp}.`
            : "Enter a valid request length."}
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
            <div style={{ fontSize: "26px", fontWeight: 700 }}>{ratioStr}</div>
            {baseValid && baseRatioStr !== ratioStr ? (
              <div style={{ color: colors.muted, fontSize: "11px" }}>
                Deploy command (without Playground overrides): {baseRatioStr}
              </div>
            ) : null}
          </div>
          <code style={{ flex: 1, minWidth: "240px", color: colors.text }}>
            {flagText}
          </code>
          <button
            type="button"
            onClick={copy}
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
      <div style={{ color: colors.muted, fontSize: "11px" }}>
        state/slot {(stateBytesPerSlot / 1e6).toFixed(1)} MB · KV/token{" "}
        {(kvBytesPerToken / 1e3).toFixed(1)} KB · the ratio prices{" "}
        {slots + drafts} state slots per request; the pin counts S = {slots},
        so {targetConcurrency || "N"} concurrent requests pin{" "}
        {pinValid ? pin : "—"} slots.
      </div>
    </div>
  );
};
