export const MiMoV25Deployment = () => {
  // MiMo-V2.5 family deployment matrix:
  //   Variant × Hardware → slug, tp, multinode, blackwell
  //
  //   V2.5-Pro (1.02T / 42B active) — text-only:
  //     H200  → tp=16, 2 nodes,     FP8 (Hopper: fa3 + DeepEP)
  //     H100  → tp=16, 2 nodes,     FP8 (Hopper: fa3 + DeepEP)
  //     B200  → tp=8,  single-node, FP8 (Blackwell verified: fa4 + flashinfer_trtllm)
  //     GB300 → tp=8,  2 nodes,     FP8 (Blackwell verified: fa4 + flashinfer_trtllm + NCCL_MNNVL)
  //   V2.5 (310B / 15B active) — multimodal. Checkpoint is TP=4 interleaved,
  //   so attention-TP per DP group must be 4; effective parallelism = TP/DP = 4.
  //     H200  → tp=8, dp=2, single-node, FP8 (verified)
  //     H100  → tp=8, dp=2, single-node, FP8
  //     B200  → tp=4, dp=1, single-node, FP8
  //     GB300 → tp=4, dp=1, single-node, FP8
  //
  //   Optional toggles:
  //     EAGLE MTP — adds --speculative-* flags + SGLANG_ENABLE_SPEC_V2=1.
  //     DeepEP    — Hopper only (Blackwell uses flashinfer_trtllm). Adds
  //                 --moe-a2a-backend deepep + --moe-dense-tp-size 1
  //                 (and --ep on Pro) + SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256.
  //                 Requires `pip install deep_ep`.

  const options = {
    modelVariant: {
      name: "modelVariant",
      title: "Model Variant",
      items: [
        { id: "pro",  label: "V2.5-Pro", default: true,  subtitle: "1.02T / 42B" },
        { id: "base", label: "V2.5",     default: false, subtitle: "310B / 15B"  },
      ],
    },
    hardware: {
      name: "hardware",
      title: "Hardware Platform",
      items: [
        { id: "h200",     label: "H200",     default: true  },
        { id: "h100",     label: "H100",     default: false },
        { id: "b200",     label: "B200",     default: false },
        { id: "gb300",    label: "GB300",    default: false },
        { id: "tpu-v7x",  label: "TPU v7x",  default: false, subtitle: "sgl-jax, Pro only" },
        { id: "tpu-v6e",  label: "TPU v6e",  default: false, subtitle: "sgl-jax, Pro only" },
      ],
    },
    eagleMtp: {
      name: "eagleMtp",
      title: "EAGLE MTP",
      items: [
        { id: "enabled",  label: "Enabled",  default: true,  subtitle: "EAGLE" },
        { id: "disabled", label: "Disabled", default: false },
      ],
    },
    dpAttention: {
      name: "dpAttention",
      title: "DP Attention",
      items: [
        { id: "enabled",  label: "Enabled",  default: false, subtitle: "auto for V2.5" },
        { id: "disabled", label: "Disabled", default: true },
      ],
    },
    expertParallelism: {
      name: "expertParallelism",
      title: "Expert Parallelism",
      items: [
        { id: "enabled",  label: "Enabled",  default: false, subtitle: "Pro Hopper" },
        { id: "disabled", label: "Disabled", default: true },
      ],
    },
    deepep: {
      name: "deepep",
      title: "DeepEP",
      items: [
        { id: "enabled",  label: "Enabled",  default: false, subtitle: "needs deep_ep" },
        { id: "disabled", label: "Disabled", default: true,  subtitle: "default" },
      ],
    },
    reasoningParser: {
      name: "reasoningParser",
      title: "Reasoning Parser",
      items: [
        { id: "enabled",  label: "Enabled",  default: true,  subtitle: "mimo" },
        { id: "disabled", label: "Disabled", default: false },
      ],
    },
    toolcall: {
      name: "toolcall",
      title: "Tool Call Parser",
      items: [
        { id: "enabled",  label: "Enabled",  default: true,  subtitle: "mimo" },
        { id: "disabled", label: "Disabled", default: false },
      ],
    },
  };

  // Per (variant, hardware): HF slug, tp, multinode info, Blackwell flag.
  // V2.5 (base) checkpoint has TP=4-interleaved fused qkv_proj, so attention
  // TP per DP group MUST be 4. Effective TP/DP = 4. With tp=8 → dp=2; tp=4 → dp=1.
  // TPU rows go through the sgl-jax stack (`python -m sgl_jax.launch_server`),
  // not the CUDA `sglang serve` binary; tp == total JAX devices across nodes.
  const HW_VARIANT_SPEC = {
    "pro|h200":     { slug: "XiaomiMiMo/MiMo-V2.5-Pro", tp: 16, multinode: true,  nnodes: 2,  blackwell: false, jax: false },
    "pro|h100":     { slug: "XiaomiMiMo/MiMo-V2.5-Pro", tp: 16, multinode: true,  nnodes: 2,  blackwell: false, jax: false },
    "pro|b200":     { slug: "XiaomiMiMo/MiMo-V2.5-Pro", tp: 8,  multinode: false,             blackwell: true,  jax: false },
    "pro|gb300":    { slug: "XiaomiMiMo/MiMo-V2.5-Pro", tp: 8,  multinode: true,  nnodes: 2,  blackwell: true,  jax: false },
    "pro|tpu-v7x":  { slug: "XiaomiMiMo/MiMo-V2.5-Pro", tp: 32, multinode: true,  nnodes: 4,  blackwell: false, jax: true  },
    "pro|tpu-v6e":  { slug: "XiaomiMiMo/MiMo-V2.5-Pro", tp: 64, multinode: true,  nnodes: 16, blackwell: false, jax: true  },
    "base|h200":    { slug: "XiaomiMiMo/MiMo-V2.5",     tp: 8,  multinode: false,             blackwell: false, jax: false, dp: 2 },
    "base|h100":    { slug: "XiaomiMiMo/MiMo-V2.5",     tp: 8,  multinode: false,             blackwell: false, jax: false, dp: 2 },
    "base|b200":    { slug: "XiaomiMiMo/MiMo-V2.5",     tp: 4,  multinode: false,             blackwell: true,  jax: false, dp: 1 },
    "base|gb300":   { slug: "XiaomiMiMo/MiMo-V2.5",     tp: 4,  multinode: false,             blackwell: true,  jax: false, dp: 1 },
  };

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

  // Toggles whose value is forced by the current variant + hardware. Returns
  // { optionName -> { force: "enabled" | "disabled", reason } }. The render
  // layer grays out the OTHER radio, and a useEffect snaps the value to the
  // forced choice so the UI never disagrees with the generated command.
  const computeConstraints = (variant, hardware) => {
    const isPro = variant === "pro";
    const spec = HW_VARIANT_SPEC[`${variant}|${hardware}`];
    const blackwell = spec ? spec.blackwell : false;
    const jax = spec ? spec.jax : false;
    const c = {};
    if (!isPro) {
      // V2.5 checkpoint is TP=4-interleaved; tp/dp must equal 4. With dp>1 we
      // must use DP-attention; with dp=1 it must be off (single attention group).
      if (spec && spec.dp > 1) {
        c.dpAttention = { force: "enabled", reason: "V2.5 checkpoint is TP=4-interleaved; DP-attention is required (--dp = tp/4)." };
      } else {
        c.dpAttention = { force: "disabled", reason: "Single attention group on this hardware (tp=4, dp=1)." };
      }
    }
    if (blackwell) {
      // DeepEP upstream targets Ampere/Hopper PTX; only experimental paths exist
      // for sm_100 in sglang and the verified Blackwell stack uses flashinfer_trtllm.
      c.deepep = { force: "disabled", reason: "Blackwell uses flashinfer_trtllm; DeepEP is Hopper / Ampere only." };
    }
    if (jax) {
      // sgl-jax stack: only V2.5-Pro is supported on TPU today; speculative
      // decoding and the DeepEP CUDA backend do not apply to the JAX runtime.
      // EP is always on (both verified launch commands set --ep-size = --tp-size).
      c.modelVariant = { force: "pro", reason: "sgl-jax TPU runtime only supports MiMo-V2.5-Pro today." };
      c.eagleMtp = { force: "disabled", reason: "EAGLE MTP is not supported on the sgl-jax TPU runtime." };
      c.deepep = { force: "disabled", reason: "DeepEP is a CUDA-only backend; sgl-jax uses the fused Pallas MoE kernel." };
      c.expertParallelism = { force: "enabled", reason: "sgl-jax TPU recipes always use EP = TP." };
    }
    return c;
  };

  const resolveItems = (option, constraints) => {
    const c = constraints[option.name];
    if (!c) return option.items;
    // Gray out every item that doesn't match the forced choice. Works for both
    // binary (enabled/disabled) toggles and N-way options like modelVariant.
    return option.items.map((item) =>
      item.id !== c.force ? { ...item, disabled: true, disabledReason: c.reason } : item,
    );
  };

  const getInitialState = () => {
    const initialState = {};
    const constraints = computeConstraints("pro", "h200");
    for (const [key, option] of Object.entries(options)) {
      const items = resolveItems(option, constraints);
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

  // Snap forced toggles to their required value whenever variant/hardware
  // changes — keeps the visible radio in sync with the generated command.
  useEffect(() => {
    const constraints = computeConstraints(values.modelVariant, values.hardware);
    let patch = null;
    for (const [key, c] of Object.entries(constraints)) {
      if (values[key] !== c.force) {
        patch = patch || {};
        patch[key] = c.force;
      }
    }
    if (patch) setValues((prev) => ({ ...prev, ...patch }));
  }, [values.modelVariant, values.hardware]);

  const handleRadioChange = (optionName, value) => {
    setValues((prev) => ({ ...prev, [optionName]: value }));
  };

  const generateCommand = () => {
    const { modelVariant, hardware, eagleMtp, dpAttention, expertParallelism, deepep, reasoningParser, toolcall } = values;
    const specKey = `${modelVariant}|${hardware}`;
    const spec = HW_VARIANT_SPEC[specKey];
    const { slug, tp, multinode, nnodes, blackwell, jax } = spec;
    const isPro = modelVariant === "pro";

    // ---------------- sgl-jax (TPU) branch ----------------
    if (jax) {
      // Recipe sources:
      //   v7x: tp=ep=32, dp=4, omits --attention-backend, mem-frac 0.95, swa 0.25
      //   v6e: tp=ep=64, dp=8, --attention-backend fa,    mem-frac 0.92, swa 0.15
      const isV7x = hardware === "tpu-v7x";
      const useEp = expertParallelism === "enabled";
      const useDpAttn = dpAttention === "enabled";
      const dpSize = isV7x ? 4 : 8;
      const flags = [];
      flags.push(`  --model-path ${slug}`);
      flags.push("  --trust-remote-code");
      flags.push(`  --tp-size ${tp}`);
      if (useEp) flags.push(`  --ep-size ${tp}`);
      if (useDpAttn) flags.push(`  --dp-size ${dpSize}`, "  --enable-dp-attention");
      flags.push("  --moe-backend fused");
      if (!isV7x) flags.push("  --attention-backend fa");
      flags.push("  --host 0.0.0.0");
      flags.push("  --port 30000");
      flags.push("  --page-size 256");
      flags.push("  --context-length 262144");
      flags.push("  --chunked-prefill-size 4096");
      flags.push("  --max-running-requests 512");
      if (isV7x) {
        flags.push("  --dtype bfloat16");
        flags.push("  --mem-fraction-static 0.95");
        flags.push("  --swa-full-tokens-ratio 0.25");
        flags.push("  --log-level info");
      } else {
        flags.push("  --max-seq-len 4096");
        flags.push("  --max-prefill-tokens 16384");
        flags.push("  --mem-fraction-static 0.92");
        flags.push("  --swa-full-tokens-ratio 0.15");
      }
      if (reasoningParser === "enabled") flags.push("  --reasoning-parser mimo");
      if (toolcall === "enabled") flags.push("  --tool-call-parser mimo");
      flags.push(`  --nnodes ${nnodes}`);
      flags.push("  --node-rank <node-rank>");
      flags.push("  --dist-init-addr <node0-ip>:20000");
      const cmd = `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server \\\n${flags.join(" \\\n")}`;
      return prependMultiNodeNote(cmd, nnodes);
    }

    // ---------------- CUDA (sglang serve) branch ----------------
    // Toggles. EAGLE MTP / EP / DeepEP / DP-attn are gated by hardware + variant
    // through computeConstraints; here we just read the (already-snapped) value.
    const useMtp = eagleMtp === "enabled";
    const useDeepep = !blackwell && deepep === "enabled";
    const useEp = isPro && !blackwell && expertParallelism === "enabled";
    const useDpAttn = dpAttention === "enabled";
    // dp size: V2.5 picks tp/4 from spec; Pro picks tp.
    const dpSize = !isPro ? spec.dp : tp;

    // ---- env (kept inline before `sglang serve`, matching the verified launch style) ----
    const envVars = [];
    if (isPro && blackwell && multinode) {
      envVars.push("NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1");
    }
    if (useMtp) envVars.push("SGLANG_ENABLE_SPEC_V2=1");
    if (useDeepep) envVars.push("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256");

    // ---- flags ----
    const flags = [];
    flags.push("  --trust-remote-code");
    flags.push(`  --model-path ${slug}`);
    flags.push(`  --tp ${tp}`);

    if (useDpAttn) {
      flags.push(`  --dp ${dpSize}`);
      flags.push("  --enable-dp-attention");
      if (!isPro) {
        flags.push("  --enable-dp-lm-head");
        flags.push("  --mm-enable-dp-encoder");
      }
    }

    if (useEp) flags.push(`  --ep ${tp}`);

    if (multinode) flags.push(...multiNodeFlags(nnodes));

    // MoE backend: Blackwell uses flashinfer_trtllm (hardware-driven); Hopper
    // optionally uses DeepEP (toggle).
    if (isPro && blackwell) {
      flags.push("  --moe-runner-backend flashinfer_trtllm");
    } else if (useDeepep) {
      flags.push("  --moe-a2a-backend deepep");
      if (!isPro) flags.push("  --deepep-mode auto");
      flags.push("  --moe-dense-tp-size 1");
    }

    if (isPro) {
      if (blackwell) {
        flags.push("  --attention-backend fa4");
        flags.push("  --mem-fraction-static 0.8");
        flags.push("  --max-running-requests 128");
        flags.push("  --chunked-prefill-size 16384");
        if (hardware === "b200") flags.push("  --swa-full-tokens-ratio 0.1");
        flags.push(`  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}'`);
      } else {
        flags.push("  --mem-fraction-static 0.7");
        flags.push("  --max-running-requests 128");
        flags.push("  --chunked-prefill-size 32768");
        flags.push("  --cuda-graph-max-bs 64");
        flags.push("  --page-size 64");
        flags.push("  --swa-full-tokens-ratio 0.3");
        flags.push(`  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}'`);
      }
    } else {
      flags.push("  --mem-fraction-static 0.65");
      flags.push("  --chunked-prefill-size 16384");
    }

    if (useMtp) {
      flags.push("  --speculative-algorithm EAGLE");
      flags.push("  --speculative-num-steps 3");
      flags.push("  --speculative-eagle-topk 1");
      flags.push("  --speculative-num-draft-tokens 4");
      if (!blackwell) flags.push("  --enable-multi-layer-eagle");
    }

    if (reasoningParser === "enabled") flags.push("  --reasoning-parser mimo");
    if (toolcall === "enabled") flags.push("  --tool-call-parser mimo");

    flags.push("  --host 0.0.0.0");
    flags.push("  --port 30000");

    const envInline = envVars.length ? envVars.join(" ") + " " : "";
    const base = `${envInline}sglang serve \\\n${flags.join(" \\\n")}`;
    return multinode ? prependMultiNodeNote(base, nnodes) : base;
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

  const constraints = computeConstraints(values.modelVariant, values.hardware);

  return (
    <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
        const items = resolveItems(option, constraints);
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
