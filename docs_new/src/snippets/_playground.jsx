// Engine half of the SGLang cookbook playground widget. Pairs with a
// per-model config under `/src/snippets/configs/<vendor>/<model>.jsx`.
// See the cookbook-add-model skill (.claude/skills/cookbook-add-model/) for the
// authoring workflow; references/engine-axis.md covers adding a new axis.
//
// Consumes the same config shape as `_deployment.jsx`, plus
// `config.playgroundFeatures` — a keyed map where each present key opts that
// axis in. Recognised axes:
//   attention   — TP/CP/DP-Attention knobs
//   moe         — backend (+ MegaMoE quantization sub-select) + EP
//   parsers     — per-item toggle flags
//   speculative — single-select preset
//   pdDisagg     — role + transfer backend + IB device + optional router
//   hicache     — enable + backend + write policy
//   hisparse    — enable + host ratio (decode-only)
//   flagSelects — generic: a config-declared LIST of single-selects, each with
//                 its own title + strip-prefixes + options (no per-feature code)
//
// Adding an axis = one entry in AXIS_HANDLERS below; nothing else switches on
// an axis id. Each handler implements initState / revertHidden / apply /
// render, plus optional deriveFromBase (recover state from base cell flags)
// and getRenderHints.
//
// Mintlify caveats (same as _deployment.jsx):
//   - Module-level statements are stripped → everything lives in the wrapper.
//   - Capitalized JSX tags get rebound → lowercase HTML tags only.
//   - `!(x in y)` crashes the AST walker → use `obj.key === undefined`.

export const Playground = ({ config }) => {
  if (!config) {
    return <div style={{padding: 12, color: "#b91c1c"}}>Playground: missing <code>config</code> prop</div>;
  }

  // ==========================================================================
  // 1. Constants
  // ==========================================================================
  const DIMENSIONS = ["hw", "variant", "quant", "strategy", "nodes"];
  // Shared with `_deployment.jsx` (HOST/PORT/etc. unified across the page).
  const STORAGE_KEY = "sglang-deploy-env";

  const pgFeatures = config.playgroundFeatures || {};
  // Single-host PD runs prefill + decode as two engines on one box. Each derives
  // 5 consecutive ZMQ/dist ports from its --port (port+233, see server_args.py
  // ZMQ_TCP_PORT_DELTA), so the serve ports are spaced 100 apart to keep those
  // derived ranges from overlapping — no --dist-init-addr needed single-host.
  // `dist` is only used by the multi-node renderer (cross-node rendezvous).
  const PD_PORTS = {
    prefill: { serve: 30000, dist: 30335 },
    decode:  { serve: 30100, dist: 30435 },
  };

  // ==========================================================================
  // 2. Pure data helpers
  // ==========================================================================
  const findCell = (cells, sel) =>
    cells.find((c) => DIMENSIONS.every((d) => c.match[d] === sel[d]));

  // After applying overrides, the resulting (env, flags) may equal another
  // cell sharing the same (hw, variant, quant, nodes) but a different
  // strategy. flags compared ordered; env compared as a set.
  const findMatchingCell = (cells, sel, pgEnv, pgFlags) => {
    const flagsEq = (a, b) =>
      a.length === b.length && a.every((x, i) => x === b[i]);
    const envEq = (a, b) => {
      if (a.length !== b.length) return false;
      const set = new Set(a);
      for (const x of b) if (!set.has(x)) return false;
      return true;
    };
    for (const c of cells) {
      if (c.match.hw !== sel.hw) continue;
      if (c.match.variant !== sel.variant) continue;
      if (c.match.quant !== sel.quant) continue;
      if (c.match.nodes !== sel.nodes) continue;
      if (flagsEq(c.flags || [], pgFlags || []) && envEq(c.env || [], pgEnv || [])) {
        return c;
      }
    }
    return null;
  };

  // hw|variant|quant → variant|quant → "".
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

  const placeholderDefaults = (schema) => {
    const out = {};
    for (const [k, v] of Object.entries(schema || {})) out[k] = v.default ?? "";
    return out;
  };

  // ==========================================================================
  // 3. Per-chip constraint evaluation
  // ==========================================================================
  // A chip entry is a bare value, a `{value, hide?, disable?, ...}` wrapper,
  // or a rich `{id, ...}` option. `hide`/`disable` are constraint objects
  // mapping base-cell dims to allowed-value arrays; matches when every key
  // matches (AND across keys, OR within a key). Empty/malformed never match.
  // `disabled: true` / `disable: true` are static always-disabled forms.
  const matchConstraint = (base, constraint) => {
    if (!constraint || typeof constraint !== "object") return false;
    const entries = Object.entries(constraint);
    if (entries.length === 0) return false;
    return entries.every(([k, vs]) =>
      Array.isArray(vs) && vs.includes(base[k]));
  };

  // Normalize a chip entry into `{value, label?, hidden, disabled,
  // disableReason, ...rest}`. `value` resolves to `entry.id` (rich form) or
  // `entry.value` (wrapper form), or the entry itself for bare values.
  const evaluateChip = (entry, base) => {
    if (entry === null || typeof entry !== "object") {
      return {
        value: entry, label: undefined,
        hidden: false, disabled: false, disableReason: "",
      };
    }
    const hidden = entry.hide ? matchConstraint(base, entry.hide) : false;
    let disabled = entry.disabled === true || entry.disable === true;
    if (!disabled && entry.disable && typeof entry.disable === "object") {
      disabled = matchConstraint(base, entry.disable);
    }
    return {
      ...entry,
      value: entry.id !== undefined ? entry.id : entry.value,
      label: entry.label,
      hidden,
      disabled,
      disableReason: entry.disableReason || "",
    };
  };

  const findEntry = (entries, picked) => {
    for (const e of (entries || [])) {
      const v = (e === null || typeof e !== "object")
        ? e : (e.id !== undefined ? e.id : e.value);
      if (v === picked) return e;
    }
    return null;
  };
  const isHidden = (entries, picked, base) => {
    const e = findEntry(entries, picked);
    if (e === null || e === undefined) return false;
    return evaluateChip(e, base).hidden;
  };

  // ==========================================================================
  // 4. Flag/env mutation primitives
  // ==========================================================================
  // Strip flags whose first whitespace/equals-delimited token is in `prefixes`.
  const stripFlagsByFirstToken = (flags, prefixes) => {
    const set = new Set(prefixes);
    return flags.filter((f) => !set.has(f.split(/[\s=]/)[0]));
  };

  // Strip env entries whose name (before `=`) is in `prefixes`.
  const stripEnvByPrefix = (envList, prefixes) => {
    if (!prefixes || !prefixes.length) return envList;
    const set = new Set(prefixes);
    return envList.filter((e) => !set.has(e.split("=")[0]));
  };

  // Insert new flags just before the trailing --host/--port pair.
  const insertBeforeTail = (flags, additions) => {
    const idx = flags.findIndex((f) => f.startsWith("--host"));
    const at = idx === -1 ? flags.length : idx;
    const out = flags.slice();
    out.splice(at, 0, ...additions);
    return out;
  };

  // Insert after the first present anchor in priority-ordered `afterAnyOf`,
  // falling back to right-after --model-path. Priority order keeps insertion
  // position-stable so the diff doesn't drop shared lines around the swap.
  const insertAfter = (flags, afterAnyOf, additions) => {
    let idx = -1;
    for (const anchor of afterAnyOf) {
      idx = flags.findIndex((f) => f.split(/[\s=]/)[0] === anchor);
      if (idx !== -1) break;
    }
    if (idx === -1) idx = flags.findIndex((f) => f.startsWith("--model-path"));
    const out = flags.slice();
    out.splice(idx + 1, 0, ...additions);
    return out;
  };

  // -------- Flag-reading helpers (used by `deriveFromBase` methods) ------
  // First integer arg of `--prefix N` (space/`=`-delimited), or null.
  const parseIntFlag = (flags, prefix) => {
    for (const f of (flags || [])) {
      if (f.split(/[\s=]/)[0] !== prefix) continue;
      const rest = f.slice(prefix.length).replace(/^[\s=]+/, "");
      const n = parseInt(rest, 10);
      if (!isNaN(n)) return n;
    }
    return null;
  };
  // True if any flag's first token equals `name` (boolean flags).
  const hasFlag = (flags, name) =>
    (flags || []).some((f) => f.split(/[\s=]/)[0] === name);
  // String arg of `--prefix arg` (space/`=`-delimited), or null.
  const findFlagArg = (flags, prefix) => {
    for (const f of (flags || [])) {
      if (f.split(/[\s=]/)[0] !== prefix) continue;
      const rest = f.slice(prefix.length).replace(/^[\s=]+/, "");
      return rest.length ? rest : null;
    }
    return null;
  };

  // Insertion-anchor sets (priority-ordered; each includes siblings so
  // insertion still works in partial cells).
  const ANCHOR_NEAR_MODEL_PATH = ["--model-path"];
  const ANCHOR_NEAR_TP         = ["--tp", "--model-path"];
  const ANCHOR_NEAR_DP         = ["--dp", "--tp", "--model-path"];
  const ANCHOR_NEAR_DPATTN     = ["--enable-dp-attention", "--dp", "--tp", "--model-path"];
  const ANCHOR_NEAR_MOE        = ["--moe-a2a-backend", "--moe-runner-backend",
                                  "--enable-dp-attention", "--dp", "--tp", "--model-path"];

  // Helper bundle passed to every axis handler.
  const helpers = {
    matchConstraint, evaluateChip, findEntry, isHidden,
    stripFlagsByFirstToken, stripEnvByPrefix, insertBeforeTail, insertAfter,
    parseIntFlag, hasFlag, findFlagArg,
    ANCHOR_NEAR_MODEL_PATH, ANCHOR_NEAR_TP, ANCHOR_NEAR_DP,
    ANCHOR_NEAR_DPATTN, ANCHOR_NEAR_MOE,
  };

  // ==========================================================================
  // 5. AXIS_HANDLERS — the built-in playground axis registry
  // ==========================================================================
  // Each entry implements initState / revertHidden / apply / render (plus
  // optional deriveFromBase / getRenderHints). Iterated in insertion order;
  // axes absent from config.playgroundFeatures are skipped.
  //   - inherit-from-base sentinels: null / "current" / "auto" / "off" /
  //     "disabled" / false — apply no-ops on these (except always-strip axes).
  //   - apply must not mutate its inputs.
  //   - strip policy: most axes strip ONLY when overridden; pdDisagg and
  //     hicache strip unconditionally.
  const AXIS_HANDLERS = {

    // ---- Axis: Attention Parallelism ----------------------------------------
    // TP / CP / DP-Attention sub-knobs; `null` = inherit. DP-Attention is
    // combined: a numeric value emits `--dp N --enable-dp-attention`, `false`
    // strips both.
    attention: {
      initState: () => ({ tp: null, cp: null, dpAttn: null }),

      // DP-Attention: `--dp N --enable-dp-attention` → N; neither → false;
      // bare `--enable-dp-attention` → 1. CP only resolves on/off (→ 2).
      deriveFromBase: (cell, fc, h) => {
        const flags = (cell && cell.flags) || [];
        const dpVal = h.parseIntFlag(flags, "--dp");
        const hasDpAttn = h.hasFlag(flags, "--enable-dp-attention");
        let dpAttn;
        if (dpVal !== null) dpAttn = dpVal;
        else if (hasDpAttn) dpAttn = 1;
        else dpAttn = false;
        return {
          tp: h.parseIntFlag(flags, "--tp"),
          cp: h.hasFlag(flags, "--enable-nsa-prefill-context-parallel") ? 2 : null,
          dpAttn,
        };
      },

      revertHidden: (value, fc, base, h) => {
        let changed = false;
        const next = { ...value };
        for (const knob of (fc.knobs || [])) {
          const cur = next[knob.id];
          if (cur !== null && cur !== undefined
              && h.isHidden(knob.values, cur, base)) {
            next[knob.id] = null; changed = true;
          }
        }
        return changed ? next : value;
      },

      apply: ({ flags, env, value, h }) => {
        if (value.tp !== null) {
          flags = h.stripFlagsByFirstToken(flags, ["--tp"]);
          flags = h.insertAfter(flags, h.ANCHOR_NEAR_MODEL_PATH, [`--tp ${value.tp}`]);
        }
        if (value.dpAttn !== null && value.dpAttn !== undefined) {
          flags = h.stripFlagsByFirstToken(flags, ["--dp", "--enable-dp-attention"]);
          if (typeof value.dpAttn === "number" && value.dpAttn > 0) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_TP, [
              `--dp ${value.dpAttn}`,
              "--enable-dp-attention",
            ]);
          }
        }
        if (value.cp !== null) {
          flags = h.stripFlagsByFirstToken(flags, [
            "--enable-nsa-prefill-context-parallel", "--nsa-prefill-cp-mode",
          ]);
          if (value.cp > 1) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_DPATTN, [
              "--enable-nsa-prefill-context-parallel",
              "--nsa-prefill-cp-mode round-robin-split",
            ]);
          }
        }
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, renderSelect, derived }) => {
        const knobs = fc.knobs || [];
        if (!knobs.length) return null;
        const setKnob = (k, v) => setValue({ ...value, [k]: v });
        const labelFor = (knob) => (c) => {
          if (c.label !== undefined) return c.label;
          if (knob.id === "dpAttn") {
            const labelMap = knob.labels || { "auto": "Auto", "false": "Off" };
            const k = c.value === null ? "auto" : String(c.value);
            return labelMap[k] || k;
          }
          return c.value === null ? "Auto" : String(c.value);
        };
        // Display: explicit pick > derived > null sentinel.
        const knobDisplay = (knob) => {
          const v = value[knob.id];
          if (v !== null && v !== undefined) return v;
          if (derived && derived[knob.id] !== undefined) return derived[knob.id];
          return null;
        };
        const hideNullFor = (knob) => {
          const d = derived ? derived[knob.id] : null;
          return (d !== null && d !== undefined) ? [null] : [];
        };
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>Attention</span>
              {knobs.map((knob) => (
                <span key={knob.id} style={s.field}>
                  <span style={s.fieldLabel}>{knob.label || knob.id.toUpperCase()}</span>
                  {renderSelect(knobDisplay(knob), knob.values || [null],
                    (nv) => setKnob(knob.id, nv), base, labelFor(knob),
                    { hideValues: hideNullFor(knob) })}
                </span>
              ))}
            </div>
          </div>
        );
      },
    },

    // ---- Axis: MoE Parallelism ----------------------------------------------
    // Backend single-select + EP numeric knob; either is optional. Picking the
    // "megamoe" backend reveals a Quantization sub-select (W4A8 / W4A4) in the same
    // row — W4A4 adds the FP4-activations env vars.
    moe: {
      initState: () => ({ backend: null, ep: null, mmQuant: null }),

      // Prefer --moe-a2a-backend over --moe-runner-backend when both present.
      // mmQuant is derived from the base env (FP4 activations present → W4A4).
      deriveFromBase: (cell, fc, h) => {
        const flags = (cell && cell.flags) || [];
        const baseEnv = (cell && cell.env) || [];
        const a2a    = h.findFlagArg(flags, "--moe-a2a-backend");
        const runner = h.findFlagArg(flags, "--moe-runner-backend");
        const fp4Acts = baseEnv.some(
          (e) => e.startsWith("SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS"));
        return {
          backend: a2a || runner || null,
          ep: h.parseIntFlag(flags, "--ep"),
          mmQuant: fp4Acts ? "w4a4" : "w4a8",
        };
      },

      revertHidden: (value, fc, base, h) => {
        let changed = false;
        const next = { ...value };
        if (next.backend !== null && fc.backend?.options
            && h.isHidden(fc.backend.options, next.backend, base)) {
          next.backend = null; changed = true;
        }
        // MegaMoE backend availability — gated by its option's requiresHw /
        // excludesStrategy (this model gates by hw only; the check is generic).
        const mmOpt = (fc.backend?.options || []).find((o) => o.id === "megamoe");
        const mmAvail = !!mmOpt
          && (!mmOpt.requiresHw || mmOpt.requiresHw.includes(base.hw))
          && (!mmOpt.excludesStrategy || !mmOpt.excludesStrategy.includes(base.strategy));
        if (next.backend === "megamoe" && !mmAvail) {
          next.backend = null; changed = true;
        }
        if (next.ep !== null && fc.ep?.values
            && h.isHidden(fc.ep.values, next.ep, base)) {
          next.ep = null; changed = true;
        }
        return changed ? next : value;
      },

      apply: ({ flags, env, value, fc, h, derived }) => {
        if (value.backend !== null) {
          flags = h.stripFlagsByFirstToken(flags, [
            "--moe-a2a-backend", "--moe-runner-backend",
          ]);
          const opt = (fc.backend?.options || []).find((o) => o.id === value.backend);
          if (opt?.flags?.length) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_DPATTN, opt.flags);
          }
        }
        // MegaMoE owns the MoE path: when the effective backend is megamoe, strip the
        // DeepEP dispatch + any prior megamoe env, then re-add the selected quant's
        // env. When the backend is explicitly switched away from megamoe, only drop
        // the megamoe quant env (leave DeepEP dispatch intact).
        const mq = fc.megamoeQuant;
        if (mq) {
          const quantKeys = [];
          for (const o of (mq.options || [])) {
            for (const e of (o.env || [])) quantKeys.push(e.split("=")[0]);
          }
          const effBackend = value.backend !== null
            ? value.backend : (derived && derived.backend);
          if (effBackend === "megamoe") {
            env = h.stripEnvByPrefix(env, [...(mq.stripEnv || []), ...quantKeys]);
            const quant = value.mmQuant != null
              ? value.mmQuant : ((derived && derived.mmQuant) || "w4a8");
            const opt = (mq.options || []).find((o) => o.id === quant);
            if (opt?.env?.length) env = [...env, ...opt.env];
          } else if (value.backend !== null) {
            env = h.stripEnvByPrefix(env, quantKeys);
          }
        }
        if (value.ep !== null) {
          flags = h.stripFlagsByFirstToken(flags, ["--ep"]);
          if (value.ep > 1) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_MOE, [`--ep ${value.ep}`]);
          }
        }
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, renderSelect, derived }) => {
        if (!fc.backend && !fc.ep) return null;
        const setSlot = (k, v) => setValue({ ...value, [k]: v });
        // Display: explicit > derived > null.
        const slotDisplay = (k) => {
          const v = value[k];
          if (v !== null && v !== undefined) return v;
          if (derived && derived[k] !== undefined) return derived[k];
          return null;
        };
        const hideNull = (k) => {
          const d = derived ? derived[k] : null;
          return (d !== null && d !== undefined) ? [null] : [];
        };
        // Hide the MegaMoE backend option where its requiresHw / excludesStrategy exclude this base.
        const mmOpt = (fc.backend?.options || []).find((o) => o.id === "megamoe");
        const mmAvail = !!mmOpt
          && (!mmOpt.requiresHw || mmOpt.requiresHw.includes(base.hw))
          && (!mmOpt.excludesStrategy || !mmOpt.excludesStrategy.includes(base.strategy));
        const backendIsMega = slotDisplay("backend") === "megamoe";
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>MoE</span>
              {fc.backend && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>Backend</span>
                  {renderSelect(slotDisplay("backend"), fc.backend.options || [],
                    (v) => setSlot("backend", v), base, undefined,
                    { hideValues: [...hideNull("backend"), ...(mmAvail ? [] : ["megamoe"])] })}
                </span>
              )}
              {fc.megamoeQuant && backendIsMega && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>Quantization</span>
                  {renderSelect(
                    value.mmQuant != null ? value.mmQuant : ((derived && derived.mmQuant) || "w4a8"),
                    fc.megamoeQuant.options || [],
                    (v) => setSlot("mmQuant", v), base)}
                </span>
              )}
              {fc.ep && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>{fc.ep.label || "EP"}</span>
                  {renderSelect(slotDisplay("ep"), fc.ep.values || [null],
                    (v) => setSlot("ep", v), base, undefined,
                    { hideValues: hideNull("ep") })}
                </span>
              )}
            </div>
          </div>
        );
      },
    },

    // ---- Axis: Parsers ------------------------------------------------------
    // Per-item toggles (tri-state: null = inherit). Strips only when overridden,
    // then re-emits one item.flag per toggled-on item.
    parsers: {
      initState: (fc) => {
        const out = {};
        for (const item of (fc.items || [])) out[item.id] = null;
        return out;
      },

      deriveFromBase: (cell, fc, h) => {
        const flags = (cell && cell.flags) || [];
        const out = {};
        for (const item of (fc.items || [])) {
          const prefix = item.flag.split(/[\s=]/)[0];
          out[item.id] = h.hasFlag(flags, prefix);
        }
        return out;
      },

      revertHidden: (value, fc, base, h) => {
        let changed = false;
        const next = { ...value };
        for (const item of (fc.items || [])) {
          if (next[item.id] !== null && next[item.id] !== undefined
              && h.evaluateChip(item, base).hidden) {
            next[item.id] = null; changed = true;
          }
        }
        return changed ? next : value;
      },

      apply: ({ flags, env, value, fc, h, derived }) => {
        const items = fc.items || [];
        // Effective state per item: explicit > derived > false. Skip
        // strip+emit when nothing differs from base.
        const eff = {};
        const baseOf = {};
        for (const item of items) {
          baseOf[item.id] = derived ? !!derived[item.id] : false;
          const v = value[item.id];
          eff[item.id] = (v === null || v === undefined) ? baseOf[item.id] : v;
        }
        const anyOverride = items.some((it) => eff[it.id] !== baseOf[it.id]);
        if (!anyOverride) return { flags, env };
        flags = h.stripFlagsByFirstToken(flags, ["--reasoning-parser", "--tool-call-parser"]);
        const adds = [];
        for (const item of items) {
          if (eff[item.id]) adds.push(item.flag);
        }
        if (adds.length) flags = h.insertBeforeTail(flags, adds);
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip, derived }) => {
        const visible = (fc.items || [])
          .map((item) => ({ item, c: h.evaluateChip(item, base) }))
          .filter(({ c }) => !c.hidden);
        if (visible.length === 0) return null;
        // Effective on/off: explicit pick > derived > off.
        const effOn = (id) => {
          const v = value[id];
          if (v !== null && v !== undefined) return v;
          if (derived && derived[id] !== undefined) return derived[id];
          return false;
        };
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>Parsers</span>
              {visible.map(({ item, c }) => (
                <span key={item.id} style={s.field}>
                  {renderChip(item.label, effOn(item.id), true,
                    () => setValue({ ...value, [item.id]: !effOn(item.id) }),
                    { disabled: c.disabled, disabledReason: c.disableReason })}
                </span>
              ))}
            </div>
          </div>
        );
      },
    },

    // ---- Axis: Speculative Decoding -----------------------------------------
    // Single-select. "current" = leave base untouched, "off" = strip (greedy),
    // <other> = strip + splice option.flags.
    speculative: {
      initState: () => "current",

      // Match base's `--speculative-*` set against each option. No spec flags
      // → "off"; some but no match → "current".
      deriveFromBase: (cell, fc) => {
        const flags = (cell && cell.flags) || [];
        const baseSpec = flags.filter((f) => {
          const head = f.split(/[\s=]/)[0];
          return head === "--speculative-algorithm"
              || head === "--speculative-num-steps"
              || head === "--speculative-eagle-topk"
              || head === "--speculative-num-draft-tokens"
              || head === "--speculative-ngram-max-bfs-breadth";
        });
        if (baseSpec.length === 0) return "off";
        for (const opt of (fc.options || [])) {
          if (!opt.flags || opt.flags.length !== baseSpec.length) continue;
          const ok = opt.flags.every((pf) => baseSpec.includes(pf));
          if (ok) return opt.id;
        }
        return "current";
      },

      revertHidden: (value, fc, base, h) => {
        if (value !== "current" && h.isHidden(fc.options || [], value, base)) {
          return "current";
        }
        return value;
      },

      apply: ({ flags, env, value, fc, h, derived }) => {
        if (value === "current") return { flags, env };
        // No-op when the pick already matches base (preserves flag position).
        if (derived && value === derived) return { flags, env };
        const picked = (fc.options || []).find((p) => p.id === value);
        if (picked && h.evaluateChip(picked,
            { dpAttnOn: h.hasFlag(flags, "--enable-dp-attention") }).disabled) {
          return { flags, env };
        }
        flags = h.stripFlagsByFirstToken(flags, [
          "--speculative-algorithm", "--speculative-num-steps",
          "--speculative-eagle-topk", "--speculative-num-draft-tokens",
          "--speculative-ngram-max-bfs-breadth",
        ]);
        const preset = (fc.options || []).find((p) => p.id === value);
        if (preset?.flags?.length) flags = h.insertBeforeTail(flags, preset.flags);
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip, derived }) => {
        const opts = fc.options || [];
        if (!opts.length) return null;
        // Display: explicit pick (!= "current") > derived.
        const display = (value !== "current") ? value
                       : (derived ? derived : "current");
        // Hide "current" when derive resolved to a real preset.
        const hideCurrent = !!(derived && derived !== "current");
        const visible = opts
          .map((opt) => h.evaluateChip(opt, base))
          .filter((c) => !c.hidden && !(hideCurrent && c.value === "current"));
        if (visible.length === 0) return null;
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>Speculative</span>
              {visible.map((c) => (
                <span key={c.value} style={s.field}>
                  {renderChip(c.label, display, c.value,
                    () => setValue(c.value),
                    { disabled: c.disabled, disabledReason: c.disableReason })}
                </span>
              ))}
            </div>
          </div>
        );
      },
    },

    // ---- Axis: PD Disaggregation --------------------------------------------
    // Role (off/prefill/decode) + transfer backend + optional IB device.
    // Owns the `--disaggregation-*` flags (unconditional strip). A backend may
    // carry hw-gated env (transferBackends[].env + .envWhen).
    pdDisagg: {
      initState: () => ({ mode: "off", transferBackend: "mooncake", ibDevice: "auto" }),

      revertHidden: (value, fc, base, h) => {
        let changed = false;
        const next = { ...value };
        if (next.mode !== "off" && fc.modes
            && h.isHidden(fc.modes, next.mode, base)) {
          next.mode = "off"; changed = true;
        }
        if (next.ibDevice !== "auto" && fc.ibDevices
            && h.isHidden(fc.ibDevices, next.ibDevice, base)) {
          next.ibDevice = "auto"; changed = true;
        }
        if (next.transferBackend !== "mooncake" && fc.transferBackends
            && h.isHidden(fc.transferBackends, next.transferBackend, base)) {
          next.transferBackend = "mooncake"; changed = true;
        }
        return changed ? next : value;
      },

      apply: ({ flags, env, value, sel, fc, h }) => {
        flags = h.stripFlagsByFirstToken(flags, [
          "--disaggregation-mode", "--disaggregation-transfer-backend",
          "--disaggregation-ib-device", "--disaggregation-bootstrap-port",
        ]);
        const backends = fc.transferBackends || [];

        if (value.mode === "prefill" || value.mode === "decode") {
          const backend = value.transferBackend || "mooncake";
          const adds = [
            `--disaggregation-mode ${value.mode}`,
            `--disaggregation-transfer-backend ${backend}`,
          ];
          if (value.ibDevice && value.ibDevice !== "auto") {
            adds.push(`--disaggregation-ib-device ${value.ibDevice}`);
          }
          // Single-host needs no --dist-init-addr: prefill/decode derive their
          // ZMQ/dist ports from the role-specific --port (spaced 100 apart, see
          // PD_PORTS), so the ranges don't overlap. Multi-node still gets a
          // cross-node --dist-init-addr from the renderer.
          flags = h.insertBeforeTail(flags, adds);

          // Role-specific serving port so the router's prefill / decode targets
          // line up (and prefill+decode don't collide on a single host).
          const servePort = PD_PORTS[value.mode].serve;
          flags = flags.map((f) =>
            f.split(/[\s=]/)[0] === "--port" ? `--port ${servePort}` : f);

          // Add the selected backend's env (gated by hw via `envWhen`), keeping
          // any the base cell already carries in place. We don't strip base env
          // (e.g. gb200 NCCL_*): a blanket strip would drop it when PD is off and
          // show a spurious remove+add in the diff. apply is pure from baseEnv, so
          // no stale backend env accumulates across renders.
          const meta = backends.find((b) => b.id === backend);
          if (meta && meta.env && meta.env.length) {
            const gate = meta.envWhen;
            const ok = !gate || Object.keys(gate).every(
              (k) => (gate[k] || []).includes(sel[k]));
            if (ok) env = [...env, ...meta.env.filter((e) => !env.includes(e))];
          }
        }
        return { flags, env };
      },

      getRenderHints: (value) => {
        if (value.mode === "prefill" || value.mode === "decode") {
          return { pdMode: value.mode };
        }
        return null;
      },

      render: ({ axisId, value, setValue, fc, base, s, renderSelect }) => {
        const setSlot = (k, v) => setValue({ ...value, [k]: v });
        const showModes    = (fc.modes            || []).length > 0;
        const showBackends = (fc.transferBackends || []).length > 0;
        const showIb       = (fc.ibDevices        || []).length > 0;
        if (!showModes && !showBackends && !showIb) return null;
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>PD Disagg</span>
              {showModes && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>Mode</span>
                  {renderSelect(value.mode, fc.modes,
                    (v) => setSlot("mode", v), base)}
                </span>
              )}
              {showBackends && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>Transfer Backend</span>
                  {renderSelect(value.transferBackend, fc.transferBackends,
                    (v) => setSlot("transferBackend", v), base)}
                </span>
              )}
              {showIb && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>IB Device</span>
                  {renderSelect(value.ibDevice, fc.ibDevices,
                    (v) => setSlot("ibDevice", v), base)}
                </span>
              )}
            </div>
          </div>
        );
      },
    },

    // ---- Axis: HiSparse (hierarchical sparse attention) ---------------------
    // Enable + host_to_device_ratio. Owns `--enable-hisparse`/`--hisparse-config`
    // plus fc.requiredFlags (unconditional strip, re-added when enabled).
    // Decode-only: gated on PD-Disagg mode == "decode" in both render and apply.
    hisparse: {
      initState: (fc) => ({ enable: false, hostRatio: (fc && fc.defaultHostRatio) || null }),

      revertHidden: (value, fc, base, h) => {
        if (value.hostRatio !== null && fc.hostRatios
            && h.isHidden(fc.hostRatios, value.hostRatio, base)) {
          return { ...value, hostRatio: (fc && fc.defaultHostRatio) || null };
        }
        return value;
      },

      apply: ({ flags, env, value, fc, h }) => {
        const ownedHeads = [
          "--enable-hisparse", "--hisparse-config",
          ...((fc.requiredFlags || []).map((f) => f.split(/\s/)[0])),
        ];
        flags = h.stripFlagsByFirstToken(flags, ownedHeads);
        // Decode gate: pdDisagg runs first and inserts this flag.
        const isDecode = flags.includes("--disaggregation-mode decode");
        if (value.enable && isDecode) {
          const ratio = (value.hostRatio !== null && value.hostRatio !== undefined)
            ? value.hostRatio
            : (fc.defaultHostRatio || 10);
          const cfg = { ...(fc.config || {}), host_to_device_ratio: ratio };
          const adds = [
            ...(fc.requiredFlags || []),
            "--enable-hisparse",
            `--hisparse-config '${JSON.stringify(cfg)}'`,
          ];
          flags = h.insertBeforeTail(flags, adds);
        }
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, renderChip, renderSelect }) => {
        if (base.pdMode !== "decode") return null;
        const setSlot = (k, v) => setValue({ ...value, [k]: v });
        const hasRatios = (fc.hostRatios || []).length > 0;
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>HiSparse</span>
              <span style={s.field}>
                {renderChip("Enable", value.enable, true,
                  () => setSlot("enable", !value.enable))}
              </span>
              {hasRatios && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>Host ratio</span>
                  {renderSelect(value.hostRatio, fc.hostRatios,
                    (v) => setSlot("hostRatio", v), base)}
                </span>
              )}
            </div>
          </div>
        );
      },
    },

    // ---- Axis: Hierarchical KV Cache ----------------------------------------
    // Enable + optional backend + write policy. Owns the `--hicache-*` family
    // (unconditional strip).
    hicache: {
      initState: () => ({ enable: false, backend: null, writePolicy: "auto" }),

      revertHidden: (value, fc, base, h) => {
        let changed = false;
        const next = { ...value };
        if (next.backend !== null && fc.backends
            && h.isHidden(fc.backends, next.backend, base)) {
          next.backend = null; changed = true;
        }
        if (next.writePolicy !== "auto" && fc.writePolicies
            && h.isHidden(fc.writePolicies, next.writePolicy, base)) {
          next.writePolicy = "auto"; changed = true;
        }
        return changed ? next : value;
      },

      apply: ({ flags, env, value, fc, sel, h }) => {
        if (fc.excludesHw && sel && fc.excludesHw.includes(sel.hw)) return { flags, env };
        flags = h.stripFlagsByFirstToken(flags, [
          "--enable-hierarchical-cache", "--hicache-ratio", "--hicache-size",
          "--hicache-write-policy", "--hicache-mem-layout", "--hicache-io-backend",
          "--hicache-storage-backend", "--hicache-storage-prefetch-policy",
        ]);
        if (value.enable) {
          const adds = [
            "--enable-hierarchical-cache",
            "--hicache-ratio 2",
            "--hicache-size 0",
          ];
          if (value.backend) {
            adds.push("--hicache-mem-layout page_first_direct",
                      "--hicache-io-backend direct");
          }
          const writePolicy = (value.writePolicy && value.writePolicy !== "auto")
            ? value.writePolicy : "write_through";
          adds.push(`--hicache-write-policy ${writePolicy}`);
          if (value.backend) {
            adds.push(`--hicache-storage-backend ${value.backend}`,
                      "--hicache-storage-prefetch-policy wait_complete");
          }
          flags = h.insertBeforeTail(flags, adds);
        }
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, renderChip, renderSelect }) => {
        if (fc.excludesHw && fc.excludesHw.includes(base.hw)) return null;
        const setSlot = (k, v) => setValue({ ...value, [k]: v });
        const hasBackends = (fc.backends || []).length > 0;
        const hasPolicies = (fc.writePolicies || []).length > 0;
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>HiCache</span>
              <span style={s.field}>
                {renderChip("Enable", value.enable, true,
                  () => setSlot("enable", !value.enable))}
              </span>
              {hasBackends && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>Storage</span>
                  {renderSelect(value.backend, fc.backends,
                    (v) => setSlot("backend", v), base)}
                </span>
              )}
              {hasPolicies && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>Write</span>
                  {renderSelect(value.writePolicy, fc.writePolicies,
                    (v) => setSlot("writePolicy", v), base)}
                </span>
              )}
            </div>
          </div>
        );
      },
    },

    // ---- Axis: Flag Selects (generic, config-declared) ----------------------
    // A LIST of single-selects, each declared entirely in config:
    //   { id, title, stripPrefixes: [...], options: [{ id, label, flags? }] }
    // Same shape as `speculative` minus its hardcoded title + strip list: pick
    // an option → strip the family, splice the option's flags. A flagless
    // option is the "none" / accuracy-safe choice (matches a base carrying none
    // of the family). Model-specific controls (KV-cache dtype, mamba scheduler
    // strategy, …) live here as DATA — no per-feature engine code. Supports
    // multiple selects per page. State: { [selectId]: optionId | null }
    // (null = inherit base).
    flagSelects: {
      initState: (fc) => {
        const out = {};
        for (const spec of (fc || [])) out[spec.id] = null;
        return out;
      },

      // Per select: match base's family flags (first token ∈ stripPrefixes)
      // against each option's flags. A flagless option matches an empty family.
      deriveFromBase: (cell, fc) => {
        const flags = (cell && cell.flags) || [];
        const out = {};
        for (const spec of (fc || [])) {
          const prefixes = spec.stripPrefixes || [];
          const fam = flags.filter((f) => prefixes.includes(f.split(/[\s=]/)[0]));
          let hit = null;
          for (const opt of (spec.options || [])) {
            const of = opt.flags || [];
            if (of.length === fam.length && of.every((x) => fam.includes(x))) {
              hit = opt.id; break;
            }
          }
          out[spec.id] = hit;
        }
        return out;
      },

      revertHidden: (value, fc, base, h) => {
        let changed = false;
        const next = { ...value };
        for (const spec of (fc || [])) {
          const cur = next[spec.id];
          if (cur !== null && cur !== undefined
              && h.isHidden(spec.options, cur, base)) {
            next[spec.id] = null; changed = true;
          }
        }
        return changed ? next : value;
      },

      apply: ({ flags, env, value, fc, sel, h, derived }) => {
        const evalBase = {
          ...(sel || {}),
          dpAttnOn: h.hasFlag(flags, "--enable-dp-attention"),
          pdMode: h.findFlagArg(flags, "--disaggregation-mode") || "off",
        };
        for (const spec of (fc || [])) {
          const v = value ? value[spec.id] : null;
          if (v === null || v === undefined) continue;          // inherit base
          const d = derived ? derived[spec.id] : null;
          if (v === d) continue;                                // already == base
          const opt = (spec.options || []).find((o) => o.id === v);
          if (!opt) continue;
          if (h.evaluateChip(opt, evalBase).disabled) continue;
          flags = h.stripFlagsByFirstToken(flags, spec.stripPrefixes || []);
          if (opt.flags && opt.flags.length) {
            flags = h.insertBeforeTail(flags, opt.flags);
          }
        }
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip, derived }) => {
        const cards = [];
        for (const spec of (fc || [])) {
          const opts = (spec.options || [])
            .map((o) => h.evaluateChip(o, base))
            .filter((c) => !c.hidden);
          if (!opts.length) continue;
          const explicit = value ? value[spec.id] : null;
          const display = (explicit !== null && explicit !== undefined)
            ? explicit : (derived ? derived[spec.id] : null);
          cards.push(
            <div key={`${axisId}-${spec.id}`} style={s.card}>
              <div style={s.compactRow}>
                <span style={s.axisTitle}>{spec.title}</span>
                {opts.map((c) => (
                  <span key={c.value} style={s.field}>
                    {renderChip(c.label, display, c.value,
                      () => setValue({ ...value, [spec.id]: c.value }),
                      { disabled: c.disabled, disabledReason: c.disableReason })}
                  </span>
                ))}
              </div>
            </div>
          );
        }
        return cards.length ? cards : null;
      },
    },

  };

  // ==========================================================================
  // 6. Apply pipeline + render command
  // ==========================================================================
  // Thread the base cell's (flags, env) through every declared axis's apply,
  // in declaration order; collect render hints (pdDisagg's role banner).
  const applyAllDeltas = (baseFlags, baseEnv, allDeltas, sel, derivedMap) => {
    let flags = [...baseFlags];
    let env = [...(baseEnv || [])];
    let pdMode = null;
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (!fc) continue;
      const value = allDeltas[axisId];
      if (value === undefined) continue;
      const derived = derivedMap ? derivedMap[axisId] : null;
      const out = handler.apply({ flags, env, value, fc, sel, h: helpers, derived });
      flags = out.flags;
      env = out.env;
      if (handler.getRenderHints) {
        const hints = handler.getRenderHints(value, fc) || {};
        if (hints.pdMode) pdMode = hints.pdMode;
      }
    }
    return { flags, env, pdMode };
  };

  // Renderer (same shape as _deployment.jsx). pdMode prepends a PD-role
  // banner; mode is "python" | "docker"; cellEnv is decoupled from cell so
  // callers can pass a modified env (e.g. MegaMoE's stripEnv + append).
  const renderCommandLines = (cell, flags, cellEnv, sel, envValues, pdMode = null, mode = "python") => {
    const modelName = resolveModelName(sel);
    const nnodes = parseNnodes(sel.nodes);
    const multinode = nnodes > 1;
    let f = [...flags];
    if (multinode && !f.some((x) => x.startsWith("--nnodes"))) {
      // Insert the multi-node trio after the last parallelism flag (matches
      // _deployment.jsx so untouched-base output is byte-identical).
      const PARALLELISM_ANCHORS = ["--enable-dp-attention", "--dp", "--tp"];
      let at = -1;
      for (const anchor of PARALLELISM_ANCHORS) {
        at = f.findIndex((x) => x.split(/[\s=]/)[0] === anchor);
        if (at !== -1) break;
      }
      if (at === -1) at = f.findIndex((x) => x.startsWith("--model-path"));
      // PD roles need distinct rendezvous ports so prefill+decode don't collide on a
      // shared head host; non-PD multi-node keeps :20000 (matches _deployment.jsx).
      const distPort = (pdMode && PD_PORTS[pdMode]) ? PD_PORTS[pdMode].dist : 20000;
      f.splice(at + 1, 0,
        `--nnodes ${nnodes}`,
        `--node-rank {{NODE_RANK}}`,
        `--dist-init-addr {{NODE0_IP}}:${distPort}`);
    }
    let cmd;
    if (mode === "docker") {
      // Image keyed by `hw|quant` (most specific) then `hw`; `:dev` if unmapped (matches _deployment.jsx).
      const di = config.dockerImages || {};
      const image = di[`${sel.hw}|${sel.quant}`] || di[sel.hw] || "lmsysorg/sglang:dev";
      const portFlag = f.find((x) => x.split(/[\s=]/)[0] === "--port");
      const servePort = portFlag ? portFlag.slice("--port".length).trim() : "{{PORT}}";
      const dockerLines = [
        "docker run --gpus all",
        "  --shm-size 32g",
        (multinode || pdMode) ? "  --network host" : `  -p ${servePort}:${servePort}`,
        "  -v ~/.cache/huggingface:/root/.cache/huggingface",
        `  --env "HF_TOKEN={{HF_TOKEN}}"`,
        ...cellEnv.map((e) => `  --env ${e}`),
        "  --ipc=host",
        `  ${image}`,
        "  sglang serve",
        ...f.map((x) => "    " + x),
      ];
      cmd = dockerLines.join(" \\\n");
    } else {
      const flagBlock = f.map((x) => "  " + x).join(" \\\n");
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
    if (pdMode === "prefill" || pdMode === "decode") {
      const sibling = pdMode === "prefill" ? "decode" : "prefill";
      const routerCfg = config.playgroundFeatures
        && config.playgroundFeatures.pdDisagg
        && config.playgroundFeatures.pdDisagg.router;
      const routerPort = (routerCfg && routerCfg.port) || 8000;
      const routerLine = routerCfg
        ? `# then front BOTH with the Router (SGLang Model Gateway) shown below.\n`
          + `# Client traffic (cURL) targets the router (:${routerPort}), not this role server.`
        : `# then front BOTH with a router; client traffic targets the router, not this role server.`;
      const banner =
        `# === PD Disaggregation: ${pdMode.toUpperCase()} role ===\n` +
        `# Runs the ${pdMode} server. Also run the ${sibling} role on its peer host,\n` +
        routerLine;
      cmd = `${banner}\n${cmd}`;
    }
    return cmd;
  };

  // ==========================================================================
  // 7. Diff (line-level, true LCS)
  // ==========================================================================
  // DP LCS + backtrace → unchanged / added / removed lines. Full LCS (not
  // greedy) because apply may reorder flags; greedy would drop shared lines
  // around a swap. Commands are ~15 lines so O(m·n) is trivial.
  const computeDiff = (baseStr, pgStr) => {
    const a = baseStr.split("\n");
    const b = pgStr.split("\n");
    const m = a.length, n = b.length;
    const dp = Array(m + 1).fill(null).map(() => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
        else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    const out = [];
    let i = m, j = n;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
        out.unshift({ line: a[i - 1], kind: "unchanged" });
        i--; j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        out.unshift({ line: b[j - 1], kind: "added" });
        j--;
      } else {
        out.unshift({ line: a[i - 1], kind: "removed" });
        i--;
      }
    }
    return out;
  };

  // ==========================================================================
  // 7b. Verified-cell submission helpers
  // ==========================================================================
  // Format a cell object as the cookbook config does, for paste into
  // `cells: [...]`. Keeps {{MODEL_NAME}} etc. as raw placeholders.
  const serializeCell = (sel, env, flags) => {
    const matchEntries = [
      `hw: ${JSON.stringify(sel.hw)}`,
      `variant: ${JSON.stringify(sel.variant)}`,
      `quant: ${JSON.stringify(sel.quant)}`,
      `strategy: ${JSON.stringify(sel.strategy)}`,
      `nodes: ${JSON.stringify(sel.nodes)}`,
    ].join(", ");
    const fmtList = (items) => {
      if (!items || items.length === 0) return "[]";
      const lines = items.map((s) => `        ${JSON.stringify(s)},`).join("\n");
      return `[\n${lines}\n      ]`;
    };
    return [
      "    {",
      `      match: { ${matchEntries} },`,
      "      verified: true,",
      `      env: ${fmtList(env)},`,
      `      flags: ${fmtList(flags)},`,
      "    },",
    ].join("\n");
  };

  // GitHub Issue prefill URL. Query keys must match the `id:` values in
  // .github/ISSUE_TEMPLATE/3-playground-verified-cell.yml. `config.github`
  // overrides the repo/template; defaults match the SGLang repo.
  const buildSubmitUrl = (sel, fields) => {
    const gh = (config.github) || {};
    const owner = gh.owner || "sgl-project";
    const repo  = gh.repo  || "sglang";
    const tmpl  = gh.issueTemplate || "3-playground-verified-cell.yml";
    const cookbookModel = gh.cookbookModel || "deepseek-ai/deepseek-v4";
    const combo = `${sel.hw} / ${sel.variant} / ${sel.quant} / ${sel.strategy} / ${sel.nodes}`;
    const params = new URLSearchParams({
      template: tmpl,
      title: `[Playground] Verified cell: ${combo}`,
      model: cookbookModel,
      combination: combo,
      "cell-snippet": fields.cellSnippet || "",
      "existing-cell": fields.existingCell || "",
      "sglang-version": fields.sglangVersion || "",
      "bench-result": fields.benchResult || "",
      notes: fields.notes || "",
    });
    return `https://github.com/${owner}/${repo}/issues/new?${params.toString()}`;
  };

  // ==========================================================================
  // 8. Style helper (dark-mode-aware)
  // ==========================================================================
  const makeStyles = (isDark) => ({
    container: { maxWidth: "900px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "6px" },
    card: {
      padding: "6px 10px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#FDBA74" : "#FB923C"}`,
      borderRadius: "4px",
      background: isDark ? "#1f2937" : "#fff",
    },
    cardStack: { display: "flex", flexDirection: "column", gap: "6px" },
    baseStrip: {
      padding: "8px 12px",
      borderRadius: "4px",
      background: isDark ? "#064e3b" : "#d1fae5",
      color: isDark ? "#a7f3d0" : "#065f46",
      fontSize: "12px",
      display: "flex", alignItems: "center", gap: "10px",
    },
    title: { fontSize: "13px", fontWeight: "600", color: isDark ? "#e5e7eb" : "inherit", marginBottom: "8px" },
    compactRow: {
      display: "flex", flexWrap: "wrap", alignItems: "center",
      gap: "10px", rowGap: "4px",
    },
    axisTitle: {
      fontSize: "12px", fontWeight: 700,
      color: isDark ? "#FDBA74" : "#C2410C",
      letterSpacing: "0.02em",
      minWidth: "100px", flexShrink: 0,
    },
    field: { display: "inline-flex", alignItems: "center", gap: "4px" },
    fieldLabel: {
      fontSize: "11px", fontWeight: 500,
      color: isDark ? "#9ca3af" : "#6b7280",
    },
    select: {
      padding: "2px 6px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "3px",
      fontSize: "12px",
      background: isDark ? "#111827" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      cursor: "pointer",
      lineHeight: "1.4",
    },
    rowFlex: { display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center", flex: 1 },
    subRow: { display: "flex", alignItems: "center", gap: "10px" },
    subLabel: {
      fontSize: "11px", fontWeight: 600,
      color: isDark ? "#9ca3af" : "#6b7280",
      minWidth: "96px", flexShrink: 0,
      letterSpacing: "0.02em",
    },
    chipRow: { display: "flex", flexWrap: "wrap", gap: "6px", flex: 1 },
    chip: {
      padding: "3px 9px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px",
      cursor: "pointer",
      fontSize: "12px",
      userSelect: "none",
      background: isDark ? "#374151" : "#fff",
      color: isDark ? "#e5e7eb" : "inherit",
      textAlign: "center",
    },
    // Terracotta matches _deployment.jsx's selected button.
    chipChecked: {
      background: "#D45D44", color: "white", borderColor: "#D45D44",
    },
    chipDisabled: { cursor: "not-allowed", opacity: 0.4 },
    commandWrap: {
      position: "relative",
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
    // Amber callout under the playground command when the effective (post-
    // override) command turns speculative decoding on without setting
    // --max-running-requests (SGLang then caps it at 48).
    mtpWarn: {
      margin: "8px 0 0", padding: "8px 12px", borderRadius: "8px",
      fontSize: "12px", lineHeight: "1.45",
      background: isDark ? "#78350f" : "#fef3c7",
      color: isDark ? "#fde68a" : "#92400e",
      border: `1px solid ${isDark ? "#92400e" : "#fcd34d"}`,
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
    // Two-state badge — matches _deployment.jsx's verified/unverified badge.
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
      fontSize: "11px", fontWeight: 600,
      userSelect: "none",
    },
    runModeChip: (active) => ({
      padding: "2px 10px", cursor: "pointer",
      background: active ? (isDark ? "#1f2937" : "#fff") : "transparent",
      color: active ? (isDark ? "#e5e7eb" : "#111827") : (isDark ? "#9ca3af" : "#6b7280"),
      borderRight: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
    }),
    runModeChipLast: (active) => ({
      padding: "2px 10px", cursor: "pointer",
      background: active ? (isDark ? "#1f2937" : "#fff") : "transparent",
      color: active ? (isDark ? "#e5e7eb" : "#111827") : (isDark ? "#9ca3af" : "#6b7280"),
    }),
    headerLeft: { display: "inline-flex", flexWrap: "wrap", alignItems: "center", gap: "8px" },
    // Native <dialog> in top-layer mode (.showModal()) escapes Mintlify's
    // `container-type: inline-size` trap that catches plain fixed-position
    // modals. ::backdrop can't be styled inline (injected via useEffect).
    dialog: {
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      borderRadius: "8px", padding: "20px",
      maxWidth: "720px", width: "92%",
      maxHeight: "calc(100vh - 80px)", overflowY: "auto",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
      margin: "auto",
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
      padding: "6px 14px",
      background: isDark ? "#FDBA74" : "#FB923C",
      color: isDark ? "#7C2D12" : "white",
      border: "none", borderRadius: "4px", cursor: "pointer",
      fontSize: "13px", fontWeight: 500,
    },
    resetBtn: {
      marginLeft: "auto", padding: "2px 8px", fontSize: "11px",
      border: `1px solid ${isDark ? "#4b5563" : "#d1d5db"}`,
      borderRadius: "3px",
      background: "transparent",
      color: isDark ? "#9ca3af" : "#6b7280",
      cursor: "pointer",
    },
    switchBaseBtn: {
      padding: "2px 8px", fontSize: "11px", fontWeight: 600,
      border: `1px solid ${isDark ? "#FDBA74" : "#FB923C"}`,
      borderRadius: "3px",
      background: "transparent",
      color: isDark ? "#FDBA74" : "#C2410C",
      cursor: "pointer",
    },
    // Annotation next to the verified pill when the command matches a
    // sibling cell (different strategy).
    matchedHint: {
      fontSize: "11px",
      color: isDark ? "#9ca3af" : "#6b7280",
      marginLeft: "8px",
      display: "inline-flex", alignItems: "center", gap: "4px",
    },
    matchedSwitchBtn: {
      marginLeft: "4px",
      background: "transparent",
      border: "none",
      padding: 0,
      color: isDark ? "#FDBA74" : "#C2410C",
      cursor: "pointer",
      fontSize: "11px", fontWeight: 600,
      textDecoration: "underline",
      textUnderlineOffset: "2px",
    },
  });

  // ==========================================================================
  // 9. React state + effects
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

  // Env / placeholder values, shared with _deployment.jsx via STORAGE_KEY.
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

  // Base selection — live-linked to the Deployment panel via URL hash + custom event
  // (history.replaceState doesn't fire hashchange, hence the event too).
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

  // Deltas: one slot per declared axis.
  const initialDeltas = () => {
    const out = {};
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (fc) out[axisId] = handler.initState(fc);
    }
    return out;
  };
  const [deltas, setDeltas] = useState(initialDeltas);

  // On base change, revert any now-hidden picks to their inherit default.
  // Disabled picks are NOT reverted (soft warning).
  useEffect(() => {
    setDeltas((d) => {
      let next = d;
      let mutated = false;
      for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
        const fc = pgFeatures[axisId];
        if (!fc || d[axisId] === undefined) continue;
        const nv = handler.revertHidden(d[axisId], fc, base, helpers);
        if (nv !== d[axisId]) {
          if (!mutated) { next = { ...d }; mutated = true; }
          next[axisId] = nv;
        }
      }
      return mutated ? next : d;
    });
  }, [base.hw, base.variant, base.quant, base.strategy, base.nodes]);

  const [modal, setModal] = useState(null); // 'curl' | 'env' | 'submit' | null

  // Callback-ref: show on mount. .showModal() (not .show()) gives the top
  // layer plus native ESC / focus-trap / scroll-lock.
  const openDialog = (el) => {
    if (el && !el.open) {
      try { el.showModal(); } catch { /* already open or unsupported */ }
    }
  };

  // Click-outside-to-close: backdrop clicks hit the dialog element with
  // coordinates outside its rect.
  const onDialogClick = (e) => {
    if (e.target !== e.currentTarget) return;
    const r = e.currentTarget.getBoundingClientRect();
    const { clientX: x, clientY: y } = e;
    if (x < r.left || x > r.right || y < r.top || y > r.bottom) setModal(null);
  };

  // ::backdrop can't be styled inline — inject one dim-overlay rule.
  useEffect(() => {
    const ID = "__playground_dialog_backdrop";
    if (document.getElementById(ID)) return undefined;
    const style = document.createElement("style");
    style.id = ID;
    style.textContent = `dialog::backdrop { background: rgba(0, 0, 0, 0.5); }`;
    document.head.appendChild(style);
    return () => { const el = document.getElementById(ID); if (el) el.remove(); };
  }, []);

  const [copied, setCopied] = useState(false);
  const [curlCopied, setCurlCopied] = useState(false);
  const [routerCopied, setRouterCopied] = useState(false);
  const [envDraft, setEnvDraft] = useState(env);
  useEffect(() => { if (modal === "env") setEnvDraft(env); }, [modal, env]);
  const [runMode, setRunMode] = useState("python");

  // Submit-verified-cell modal state, reset each time the modal opens.
  const [submitDraft, setSubmitDraft] = useState({
    sglangVersion: "", benchResult: "", notes: "",
  });
  const [submitAttest, setSubmitAttest] = useState({
    ranCommand: false, reachedReady: false, outputCorrect: false,
  });
  useEffect(() => {
    if (modal === "submit") {
      setSubmitDraft({ sglangVersion: "", benchResult: "", notes: "" });
      setSubmitAttest({ ranCommand: false, reachedReady: false, outputCorrect: false });
    }
  }, [modal]);

  // ==========================================================================
  // 10. Derived values
  // ==========================================================================
  const s = makeStyles(isDark);
  // PD-disaggregated cells (`roles`) have no single flat flag list to overlay
  // playground deltas on — treat them as no-base (playground-only view).
  const cellAtSel = findCell(config.cells, base);
  const baseCell = cellAtSel && cellAtSel.roles ? null : cellAtSel;
  const modelName = resolveModelName(base);

  // Per-axis state recovered from the base cell's flags (deriveFromBase).
  const derivedMap = {};
  if (baseCell) {
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (!fc || !handler.deriveFromBase) continue;
      derivedMap[axisId] = handler.deriveFromBase(baseCell, fc, helpers);
    }
  }

  // Cross-axis facts folded into the `base` handed to chip-constraint
  // matching, so `hide`/`disable` can react to another axis's live state
  // (render path only; revertHidden keeps the clean 5-dim base).
  //   dpAttnOn — effective DP-Attention resolves to "on" (positive degree
  //              or true), explicit override else derived-from-base.
  //   pdMode   — live PD-Disagg role; gates the decode-only HiSparse card.
  const attnDelta   = deltas.attention || {};
  const attnDerived = derivedMap.attention || {};
  const effDpAttn = (attnDelta.dpAttn !== null && attnDelta.dpAttn !== undefined)
    ? attnDelta.dpAttn
    : (attnDerived.dpAttn !== undefined ? attnDerived.dpAttn : null);
  const dpAttnOn = (effDpAttn === true)
    || (typeof effDpAttn === "number" && effDpAttn > 0);
  const pdMode = (deltas.pdDisagg && deltas.pdDisagg.mode) || "off";
  const constraintBase = { ...base, dpAttnOn, pdMode };

  let baseCommand = "";
  let playgroundCommand = "";
  let diffLines = [];
  let pgFlagsLatest = [];
  let pgEnvLatest = [];
  if (baseCell) {
    baseCommand = renderCommandLines(baseCell, baseCell.flags, baseCell.env, base, env, null, runMode);
    const { flags: pgFlags, env: pgEnv, pdMode } = applyAllDeltas(baseCell.flags, baseCell.env, deltas, base, derivedMap);
    pgFlagsLatest = pgFlags;
    pgEnvLatest = pgEnv;
    playgroundCommand = renderCommandLines(baseCell, pgFlags, pgEnv, base, env, pdMode, runMode);
    diffLines = computeDiff(baseCommand, playgroundCommand);
  }
  // Cross-cell verified detection: the emitted (env, flags) may match the
  // base cell itself or a sibling (different strategy). A sibling match
  // still shows Verified, plus a "switch base" link.
  const matchedCell = baseCell
    ? findMatchingCell(config.cells, base, pgEnvLatest, pgFlagsLatest)
    : null;
  const playgroundVerified = !!(matchedCell && matchedCell.verified);
  const matchedSiblingCell = (matchedCell && matchedCell !== baseCell)
    ? matchedCell : null;
  // MTP hint on the EFFECTIVE (post-override) command — fires when the user
  // toggles speculative decoding on without setting --max-running-requests
  // (NOT keyed on strategy). Mirrors the Deploy panel's hint.
  const pgMtpHint =
    pgFlagsLatest.some((f) => f.split(/[\s=]/)[0] === "--speculative-algorithm") &&
    !pgFlagsLatest.some((f) => f.split(/[\s=]/)[0] === "--max-running-requests");

  // Submission snippets: proposed cell + existing cell at the same match.
  const proposedCellSnippet = baseCell
    ? serializeCell(base, pgEnvLatest, pgFlagsLatest) : "";
  const existingCellSnippet = baseCell
    ? serializeCell(base, baseCell.env || [], baseCell.flags) : "";
  const submitUrl = baseCell ? buildSubmitUrl(base, {
    cellSnippet: proposedCellSnippet,
    existingCell: existingCellSnippet,
    sglangVersion: submitDraft.sglangVersion,
    benchResult: submitDraft.benchResult,
    notes: submitDraft.notes,
  }) : "";
  const submitReady =
    submitAttest.ranCommand && submitAttest.reachedReady && submitAttest.outputCorrect
    && submitDraft.sglangVersion.trim().length > 0;

  // PD-Disagg router, if configured. When a PD role is active, cURL retargets
  // to the router port and a companion router block renders below the command.
  const pdRouter = (pdMode !== "off"
    && config.playgroundFeatures
    && config.playgroundFeatures.pdDisagg
    && config.playgroundFeatures.pdDisagg.router) || null;
  const curlEnv = (pdRouter && pdRouter.port != null)
    ? { ...env, CURL_PORT: String(pdRouter.port) }
    : env;
  const curlText = interpolate(config.curl || "", curlEnv, modelName);
  const routerText = pdRouter && pdRouter.command
    ? interpolate(pdRouter.command, {
        ...env,
        PREFILL_PORT: PD_PORTS.prefill.serve,
        DECODE_PORT: PD_PORTS.decode.serve,
        ROUTER_PORT: pdRouter.port,
      }, modelName)
    : "";

  const resetAll = () => setDeltas(initialDeltas());

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

  const baseSummary = baseCell
    ? `${base.hw.toUpperCase()} · ${base.variant} · ${base.quant.toUpperCase()} · ${base.strategy} · ${base.nodes}`
    : "(no verified cell at the current Deploy selection — showing playground only)";

  // ==========================================================================
  // 11. Render helpers
  // ==========================================================================
  // Chip: checked when `current === value`; disabled chips are unclickable.
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

  // Dropdown over a chip-schema `entries` array. Hidden entries excluded;
  // disabled get a "(n/a)" suffix. Uses the option index as the <select>
  // value to dodge form-value serialization; `onPick` gets the original
  // value. `labelFor` is an optional label resolver; `opts.hideValues`
  // suppresses values (e.g. the inherit sentinel when a base default exists).
  const renderSelect = (current, entries, onPick, base, labelFor, opts = {}) => {
    const hideSet = new Set(opts.hideValues || []);
    const items = [];
    for (const entry of (entries || [])) {
      const c = helpers.evaluateChip(entry, base);
      if (c.hidden) continue;
      if (hideSet.has(c.value)) continue;
      const lbl = labelFor
        ? labelFor(c)
        : (c.label !== undefined ? c.label
          : c.value === null ? "Auto" : String(c.value));
      items.push({ ...c, label: lbl });
    }
    let idx = items.findIndex((c) => c.value === current);
    if (idx === -1) idx = 0;
    return (
      <select
        style={s.select}
        value={idx}
        onChange={(e) => {
          const next = items[parseInt(e.target.value, 10)];
          if (next && !next.disabled) onPick(next.value);
        }}
      >
        {items.map((c, i) => (
          <option
            key={i}
            value={i}
            disabled={c.disabled}
          >
            {c.label}{c.disabled ? " (n/a)" : ""}
          </option>
        ))}
      </select>
    );
  };

  // ==========================================================================
  // 12. JSX render
  // ==========================================================================
  return (
    <div style={s.container} className="not-prose">
      {/* Inherited base summary */}
      <div style={s.baseStrip}>
        <span style={{ fontWeight: 600 }}>Inherited base from Deployment:</span>
        <code style={{ fontFamily: "Menlo, monospace" }}>{baseSummary}</code>
        {/* scrollIntoView (not hash nav) so the base-cell hash survives.
            Deploy heading slugs to "deployment" or "deploy". */}
        <button
          type="button"
          style={s.switchBaseBtn}
          onClick={() => {
            const el = document.getElementById("deployment")
              || document.getElementById("deploy");
            if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
        >
          ↑ Switch base
        </button>
        <button style={s.resetBtn} onClick={resetAll}>Reset all overrides</button>
      </div>

      {/* One card per declared axis (render() may return null to gate). */}
      {Object.entries(AXIS_HANDLERS).map(([axisId, handler]) => {
        const fc = pgFeatures[axisId];
        if (!fc) return null;
        const setValue = (next) => setDeltas((d) => ({ ...d, [axisId]: next }));
        return handler.render({
          axisId, value: deltas[axisId], setValue,
          // constraintBase = 5 cell dims + cross-axis facts (dpAttnOn, pdMode).
          fc, base: constraintBase, s, h: helpers, renderChip, renderSelect,
          derived: derivedMap[axisId] || null,
        });
      })}

      {/* Command box (diff vs verified base) */}
      <div style={s.card}>
        <div style={s.title}>Playground Command (compare with base)</div>
        <div style={s.commandWrap}>
          <div style={s.commandHeader}>
            <div style={s.headerLeft}>
              <div style={s.badge(playgroundVerified)}>
                <span style={s.badgeDot(playgroundVerified)} />
                {playgroundVerified ? "Verified" : "Not Verified"}
              </div>
              {/* Sibling-cell match: offer to switch the Deployment panel's base to it. */}
              {matchedSiblingCell && (
                <span style={s.matchedHint}>
                  matches <code style={{ fontFamily: "Menlo, monospace" }}>
                    {matchedSiblingCell.match.strategy}
                  </code>
                  <button
                    type="button"
                    style={s.matchedSwitchBtn}
                    onClick={() => {
                      const m = matchedSiblingCell.match;
                      setDeltas(initialDeltas());
                      const hash = new URLSearchParams(m).toString();
                      window.location.hash = hash;
                      window.dispatchEvent(new CustomEvent("sglang-deploy-sel",
                        { detail: m }));
                    }}
                  >
                    switch base →
                  </button>
                </span>
              )}
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
              {/* Submit only when the playground differs from the verified base. */}
              {!playgroundVerified && baseCell && (
                <button
                  style={{ ...s.iconButton, borderColor: isDark ? "#FDBA74" : "#FB923C",
                           color: isDark ? "#FDBA74" : "#C2410C", fontWeight: 600 }}
                  onClick={() => setModal("submit")}
                  title="I verified this command on my hardware — open a pre-filled GitHub issue to land it as a cookbook cell."
                >
                  Submit ↗
                </button>
              )}
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
            )) : "# No verified base cell at the current Deployment selection.\n# Pick a supported hardware/variant in the Deployment panel to populate the playground base."}
          </pre>
          {pgMtpHint && (
            <div style={s.mtpWarn}>
              ⚠️ Speculative decoding (MTP) is on — SGLang resets <code>--max-running-requests</code> to <strong>48</strong> when it isn't set. Add <code>--max-running-requests &lt;N&gt;</code> sized for your target concurrency.
            </div>
          )}
        </div>
      </div>

      {/* PD-Disagg router companion (separate block so the role diff stays pure). */}
      {pdRouter && routerText && (
        <div style={s.card}>
          <div style={s.title}>Router (SGLang Model Gateway)</div>
          <div style={{ fontSize: 11, opacity: 0.7, margin: "0 0 6px" }}>
            Run after both roles are up. Substitute <code>{"<prefill-host>"}</code> /{" "}
            <code>{"<decode-host>"}</code> with reachable hosts (both <code>127.0.0.1</code>{" "}
            on a same-host deployment). Client traffic (cURL) targets this router.
          </div>
          <div style={s.commandWrap}>
            <div style={s.commandHeader}>
              <div style={{ fontSize: 11, opacity: 0.7 }}>port {pdRouter.port}</div>
              <button
                style={s.iconButton}
                onClick={() => {
                  navigator.clipboard.writeText(routerText);
                  setRouterCopied(true);
                  setTimeout(() => setRouterCopied(false), 1200);
                }}
              >
                {routerCopied ? "✓ Copied" : "⧉ Copy"}
              </button>
            </div>
            <pre style={s.commandPre}>{routerText}</pre>
          </div>
        </div>
      )}

      {/* cURL modal — native <dialog> in top layer (escapes @container) */}
      {modal === "curl" && (
        <dialog ref={openDialog} style={s.dialog}
                onClose={() => setModal(null)} onClick={onDialogClick}>
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
          {pdRouter && (
            <p style={{ fontSize: 11, opacity: 0.85, marginTop: 8 }}>
              <strong>PD-Disaggregation active</strong> — this targets the router on
              {" "}<code>:{pdRouter.port}</code>; client traffic must not hit the role
              servers directly.
            </p>
          )}
          <p style={{ fontSize: 11, opacity: 0.7, marginTop: 8 }}>
            Edit <code>CURL_HOST</code> / <code>CURL_PORT</code> in the Env panel.
          </p>
        </dialog>
      )}

      {/* Env modal — native <dialog> in top layer */}
      {modal === "env" && (
        <dialog ref={openDialog} style={s.dialog}
                onClose={() => setModal(null)} onClick={onDialogClick}>
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
            Values persist in localStorage and are shared with the Deployment panel.
          </p>
        </dialog>
      )}

      {/* Submit-verified-cell modal — native <dialog> in top layer */}
      {modal === "submit" && (
        <dialog ref={openDialog} style={s.dialog}
                onClose={() => setModal(null)} onClick={onDialogClick}>
            <div style={s.modalHeader}>
              <div style={s.modalTitle}>Submit verified cell</div>
              <button style={s.modalCloseBtn} onClick={() => setModal(null)} aria-label="Close">×</button>
            </div>
            <p style={{ fontSize: 12, opacity: 0.85, marginTop: 0, marginBottom: 12 }}>
              You've put together a combination that isn't in the verified
              catalog yet. After you've run the command end-to-end on the
              target hardware, this submits a pre-filled GitHub Issue that a
              maintainer can convert into a PR.
            </p>

            <div style={s.sectionHeading}>Combination</div>
            <code style={{ fontFamily: "Menlo, monospace", fontSize: 12 }}>
              {base.hw} / {base.variant} / {base.quant} / {base.strategy} / {base.nodes}
            </code>
            {/* Overrides summary — the diff vs base in flag form. */}
            {(() => {
              const adds = diffLines.filter((d) => d.kind === "added");
              const rems = diffLines.filter((d) => d.kind === "removed");
              if (adds.length === 0 && rems.length === 0) return null;
              return (
                <>
                  <div style={{ ...s.sectionHeading, marginTop: 10 }}>
                    Overrides vs base ({adds.length} added · {rems.length} removed)
                  </div>
                  <pre style={{
                    margin: 0, padding: "8px 10px",
                    background: isDark ? "#111827" : "#f5f5f5",
                    border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
                    borderRadius: 4,
                    fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
                    fontSize: 12, lineHeight: 1.4,
                    maxHeight: 160, overflowY: "auto",
                    whiteSpace: "pre-wrap",
                  }}>
                    {[...rems, ...adds].map((d, i) => (
                      <div
                        key={i}
                        style={d.kind === "added" ? s.diffLineAdded : s.diffLineRemoved}
                      >
                        {d.kind === "added" ? "+ " : "- "}
                        {d.line.replace(/^\s*/, "")}
                      </div>
                    ))}
                  </pre>
                </>
              );
            })()}

            <div style={{ ...s.sectionHeading, marginTop: 14 }}>Attestation (all required)</div>
            <div style={s.formField}>
              <label style={{ fontSize: 12, display: "flex", alignItems: "flex-start", gap: 6 }}>
                <input type="checkbox" checked={submitAttest.ranCommand}
                  onChange={(e) => setSubmitAttest({ ...submitAttest, ranCommand: e.target.checked })} />
                I ran this exact command on the listed hardware.
              </label>
              <label style={{ fontSize: 12, display: "flex", alignItems: "flex-start", gap: 6 }}>
                <input type="checkbox" checked={submitAttest.reachedReady}
                  onChange={(e) => setSubmitAttest({ ...submitAttest, reachedReady: e.target.checked })} />
                The server reached READY and answered a cURL request successfully.
              </label>
              <label style={{ fontSize: 12, display: "flex", alignItems: "flex-start", gap: 6 }}>
                <input type="checkbox" checked={submitAttest.outputCorrect}
                  onChange={(e) => setSubmitAttest({ ...submitAttest, outputCorrect: e.target.checked })} />
                Output looked correct on at least one prompt.
              </label>
            </div>

            <div style={{ ...s.sectionHeading, marginTop: 14 }}>SGLang version (required)</div>
            <input
              style={{ ...s.formInput, width: "100%", boxSizing: "border-box" }}
              placeholder="sglang==0.5.4  (or git SHA abc1234)"
              value={submitDraft.sglangVersion}
              onChange={(e) => setSubmitDraft({ ...submitDraft, sglangVersion: e.target.value })}
            />

            <div style={{ ...s.sectionHeading, marginTop: 14 }}>Benchmark result (optional)</div>
            <input
              style={{ ...s.formInput, width: "100%", boxSizing: "border-box" }}
              placeholder="TTFT 95 ms / TPOT 18 ms / 1820 tok/s @ bs=64"
              value={submitDraft.benchResult}
              onChange={(e) => setSubmitDraft({ ...submitDraft, benchResult: e.target.value })}
            />

            <div style={{ ...s.sectionHeading, marginTop: 14 }}>Notes / caveats (optional)</div>
            <textarea
              style={{ ...s.formInput, width: "100%", boxSizing: "border-box",
                       minHeight: 110, resize: "vertical", fontFamily: "inherit" }}
              placeholder="Cluster config, env-var quirks, NIC mappings, multi-node bootstrap details, …"
              value={submitDraft.notes}
              onChange={(e) => setSubmitDraft({ ...submitDraft, notes: e.target.value })}
            />

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 16, alignItems: "center" }}>
              {!submitReady && (
                <span style={{ fontSize: 11, opacity: 0.7, marginRight: "auto" }}>
                  Tick all attestations and fill SGLang version to enable submit.
                </span>
              )}
              <button style={{ ...s.iconButton, padding: "6px 14px" }} onClick={() => setModal(null)}>Cancel</button>
              <a
                href={submitReady ? submitUrl : undefined}
                target="_blank" rel="noopener noreferrer"
                onClick={(e) => { if (!submitReady) e.preventDefault(); else setModal(null); }}
                style={{
                  ...s.primaryBtn,
                  textDecoration: "none",
                  display: "inline-flex", alignItems: "center",
                  opacity: submitReady ? 1 : 0.4,
                  cursor: submitReady ? "pointer" : "not-allowed",
                }}
              >
                Open submission on GitHub →
              </a>
            </div>
          <p style={{ fontSize: 11, opacity: 0.7, marginTop: 10 }}>
            The CTA opens a pre-filled GitHub Issue using the
            <code> 3-playground-verified-cell.yml</code> template. A
            maintainer with the listed hardware will review and convert it
            into a cookbook PR.
          </p>
        </dialog>
      )}
    </div>
  );
};
