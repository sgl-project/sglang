// Shared playground skeleton — the ENGINE half of the SGLang cookbook
// playground widget. Pair this with a per-model config file under
// `/src/snippets/configs/<vendor>/<model>.jsx` and an MDX page that imports
// both:
//
//     import { Playground } from "/src/snippets/_playground.jsx";
//     import { config }    from "/src/snippets/configs/deepseek-ai/deepseek-v4.jsx";
//     <Playground config={config} />
//
// AUTHORING — read _AUTHORING.md in this directory for the step-by-step
// workflow on adding a new cookbook or extending the engine with a new
// playground axis.
//
// CONTRACT WITH PER-MODEL CONFIG
// ------------------------------
// Same shape as `_deployment.jsx` consumes (cells / modelNames / placeholders
// / curl / multiNodeHints / dockerImages). The playground reuses §3.1's base
// cell from the URL hash to seed its diff baseline, then layers overrides on
// top.
//
// Plus one playground-specific field:
//   config.playgroundFeatures   — keyed map; presence of a key opts that axis
//                                 in for this cookbook. Recognised keys
//                                 (current set):
//     attention   — knobs[] with TP/DP/CP/DP-Attention sub-controls
//     moe         — backend.options[] + ep.values[]
//     parsers     — items[] (each emits one toggle flag)
//     speculative — options[] (single-select chip group)
//     pdDisagg    — modes[] + ibDevices[] (engine handles role banner +
//                   single-host bootstrap port internally)
//     hicache     — backends[] + writePolicies[]
//     megamoe     — requiresHw + excludesStrategy + stripEnv + options[]
//                   (axis-level gating + env mutation)
//
// EXTENDING WITH A NEW AXIS
// -------------------------
// Adding a new sglang feature axis is a one-place change: add an entry to
// `AXIS_HANDLERS` below. The state shape, hidden-revert effect, apply
// pipeline, reset, and JSX render loop all derive from this table. No other
// part of the engine should ever switch on an axis id.
//
// Each handler exposes a uniform 4-method interface:
//   initState(featureConfig)              → initial delta value
//   revertHidden(value, fc, base, helpers) → next value (same ref if unchanged)
//   apply({flags, env, value, fc, sel, helpers}) → next {flags, env}
//   render({axisId, value, setValue, fc, base, s, helpers, renderChip}) → JSX | null
// Plus one optional method:
//   getRenderHints(value, fc)             → {pdMode?, ...} hints for renderer
//
// MINTLIFY CAVEATS THIS FILE ROUTES AROUND
// ----------------------------------------
// Same as `_deployment.jsx`:
//   - Module-level statements in `.jsx` snippets are stripped → every
//     constant / helper / handler lives inside the wrapper function.
//   - Capitalized JSX tags inside a snippet body get rebound, so we use
//     only lowercase HTML JSX tags here.
//   - Plain-data imports from other `.jsx` files DO work when the import
//     lives in an MDX file — that's how `config` gets in.

export const Playground = ({ config }) => {
  if (!config) {
    return <div style={{padding: 12, color: "#b91c1c"}}>Playground: missing <code>config</code> prop</div>;
  }

  // ==========================================================================
  // 1. Constants
  // ==========================================================================
  // Mirror §3's dimension priority. Used for cell lookup, hash hydration,
  // and the hidden-revert effect's dependency array.
  const DIMENSIONS = ["hw", "variant", "quant", "strategy", "nodes"];
  // Shared with `_deployment.jsx` (HOST/PORT/etc. unified across the page).
  const STORAGE_KEY = "sglang-deploy-env";

  const pgFeatures = config.playgroundFeatures || {};

  // ==========================================================================
  // 2. Pure data helpers
  // ==========================================================================
  const findCell = (cells, sel) =>
    cells.find((c) => DIMENSIONS.every((d) => c.match[d] === sel[d]));

  // Layered lookup: hw|variant|quant → variant|quant → "".
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
  // Each chip entry in a playground axis can be either:
  //   * a bare value           — `null`, `1`, `true`, `"auto"`, ...
  //   * a wrapper object       — `{value, hide?, disable?, disableReason?, label?}`
  //   * a rich option object   — `{id, label, flags?, env?, hide?, disable?, ...}`
  //
  // The optional `hide` / `disable` fields are constraint objects — each key
  // is a base-cell field name (hw / variant / quant / strategy / nodes)
  // mapped to an array of allowed values. The chip is hidden / disabled when
  // EVERY key in the constraint matches the current base cell (AND across
  // keys; OR within each key's value list). Empty constraints (or constraints
  // with non-array values) never match — keeps malformed schemas from
  // accidentally hiding everything.
  //
  // `disabled: true` / `disable: true` are static always-disabled forms
  // (used by speculative's "Coming soon" chips). `disabled: false` /
  // `disable: false` are no-ops.
  const matchConstraint = (base, constraint) => {
    if (!constraint || typeof constraint !== "object") return false;
    const entries = Object.entries(constraint);
    if (entries.length === 0) return false;
    return entries.every(([k, vs]) =>
      Array.isArray(vs) && vs.includes(base[k]));
  };

  // Normalize a chip entry into `{value, label?, hidden, disabled,
  // disableReason, ...rest}`. For bare values, `value` is the entry itself
  // and nothing is hidden/disabled. For object forms, `value` resolves to
  // `entry.value` (bare-value wrapper form) or `entry.id` (rich option form
  // — `id` doubles as the chip's identity in the delta state).
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
      value: "value" in entry ? entry.value : entry.id,
      label: entry.label,
      hidden,
      disabled,
      disableReason: entry.disableReason || "",
    };
  };

  // Lookup helpers used by both render code and revertHidden handlers.
  const findEntry = (entries, picked) => {
    for (const e of (entries || [])) {
      const v = (e === null || typeof e !== "object")
        ? e : ("value" in e ? e.value : e.id);
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
  // Strip any flag whose first whitespace/equals-delimited token equals one
  // of `prefixes`. Used to remove the base's values for an axis before
  // re-emitting the playground's choice. Must match exactly the first token
  // because values may contain hyphens / equals (e.g.
  // `--moe-runner-backend marlin`).
  const stripFlagsByFirstToken = (flags, prefixes) => {
    const set = new Set(prefixes);
    return flags.filter((f) => !set.has(f.split(/[\s=]/)[0]));
  };

  // Strip env entries whose name (the part before `=`) matches one of the
  // given prefixes. Used by axes that need to remove the base cell's
  // incompatible env vars before adding their own (currently only MegaMoE).
  const stripEnvByPrefix = (envList, prefixes) => {
    if (!prefixes || !prefixes.length) return envList;
    const set = new Set(prefixes);
    return envList.filter((e) => !set.has(e.split("=")[0]));
  };

  // Insert a list of new flags just before the trailing --host/--port pair so
  // the diff stays visually grouped with the existing structure.
  const insertBeforeTail = (flags, additions) => {
    const idx = flags.findIndex((f) => f.startsWith("--host"));
    const at = idx === -1 ? flags.length : idx;
    const out = flags.slice();
    out.splice(at, 0, ...additions);
    return out;
  };

  // Insert one or more new flags right after the FIRST flag whose first token
  // is in `afterAnyOf`. Falls back to right-after --model-path if none match.
  // The order of `afterAnyOf` is irrelevant (set semantics); pass any subset
  // of conceptual siblings so the override lands near them.
  const insertAfter = (flags, afterAnyOf, additions) => {
    const set = new Set(afterAnyOf);
    let idx = flags.findIndex((f) => set.has(f.split(/[\s=]/)[0]));
    if (idx === -1) idx = flags.findIndex((f) => f.startsWith("--model-path"));
    const out = flags.slice();
    out.splice(idx + 1, 0, ...additions);
    return out;
  };

  // Insertion-anchor sets — declared once so axis handlers stay terse and
  // the "what lives near what" intent is visible at the top of the file.
  // Each anchor includes its sibling flags too, so insertion still works in
  // partial cells where some sibling flags are absent.
  const ANCHOR_NEAR_MODEL_PATH = ["--model-path"];
  const ANCHOR_NEAR_TP         = ["--tp", "--model-path"];
  const ANCHOR_NEAR_DP         = ["--dp", "--tp", "--model-path"];
  const ANCHOR_NEAR_DPATTN     = ["--enable-dp-attention", "--dp", "--tp", "--model-path"];
  const ANCHOR_NEAR_MOE        = ["--moe-a2a-backend", "--moe-runner-backend",
                                  "--enable-dp-attention", "--dp", "--tp", "--model-path"];

  // Helper bundle passed to every axis handler (render + apply + revertHidden).
  // Adding a new primitive available to handlers = add a field here.
  const helpers = {
    matchConstraint, evaluateChip, findEntry, isHidden,
    stripFlagsByFirstToken, stripEnvByPrefix, insertBeforeTail, insertAfter,
    ANCHOR_NEAR_MODEL_PATH, ANCHOR_NEAR_TP, ANCHOR_NEAR_DP,
    ANCHOR_NEAR_DPATTN, ANCHOR_NEAR_MOE,
  };

  // ==========================================================================
  // 5. AXIS_HANDLERS — the built-in playground axis registry
  // ==========================================================================
  // Each entry implements `initState / revertHidden / apply / render` (plus
  // optional `getRenderHints`). The engine iterates this map in insertion
  // order for render and apply, and skips any axis whose key is absent from
  // `config.playgroundFeatures`. Adding a new axis = adding one entry here.
  //
  // Conventions:
  //   - `value` is whatever shape `initState` returns for that axis.
  //   - `null` / "current" / "auto" / "off" / "disabled" / `false` are the
  //     per-axis "inherit-from-base" sentinels — `apply` should be a no-op
  //     for those (except for axes that always strip; see notes).
  //   - `apply` is a pure function: it must not mutate its inputs.
  //
  // Strip-policy summary (for the reviewer):
  //   - attention.tp/dp/dpAttn/cp : strip ONLY when value !== null
  //                                  (so base's flag survives "inherit")
  //   - moe.backend / moe.ep       : strip ONLY when value !== null
  //   - speculative                : strip ONLY when value !== "current"
  //                                  ("current" = inherit, "off" = strip
  //                                  without re-emit)
  //   - parsers                    : UNCONDITIONAL strip (engine owns parser
  //                                  flags whenever the parsers axis is
  //                                  declared)
  //   - pdDisagg                   : UNCONDITIONAL strip
  //   - hicache                    : UNCONDITIONAL strip
  //   - megamoe                    : strip ONLY when value !== "disabled"
  //                                  (also mutates env: stripEnvByPrefix +
  //                                  appends option.env)
  const AXIS_HANDLERS = {

    // ---- Axis: Attention Parallelism ----------------------------------------
    // Four sub-knobs (TP / DP / CP / DP-Attention). Per-knob state slot;
    // `null` means inherit base. Each knob has its own strip prefix +
    // insertion anchor so overrides land near their conceptual siblings.
    attention: {
      initState: () => ({ tp: null, dp: null, cp: null, dpAttn: null }),

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
        if (value.dp !== null) {
          flags = h.stripFlagsByFirstToken(flags, ["--dp"]);
          if (value.dp > 1) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_TP, [`--dp ${value.dp}`]);
          }
        }
        if (value.dpAttn !== null) {
          flags = h.stripFlagsByFirstToken(flags, ["--enable-dp-attention"]);
          if (value.dpAttn === true) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_DP, ["--enable-dp-attention"]);
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

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip }) => {
        const knobs = fc.knobs || [];
        if (!knobs.length) return null;
        const setKnob = (k, v) => setValue({ ...value, [k]: v });
        // DP-Attention chip labels are boolean-aware; others use plain numeric.
        const labelFor = (knob, c) => {
          if (c.label !== undefined) return c.label;
          if (knob.id === "dpAttn") {
            const labelMap = knob.labels || { "auto": "auto", "true": "on", "false": "off" };
            const k = c.value === null ? "auto" : String(c.value);
            return labelMap[k] || k;
          }
          return c.value === null ? "auto" : String(c.value);
        };
        return (
          <div key={axisId} style={{ ...s.card, ...s.cardStack }}>
            <div style={s.title}>Attention Parallelism</div>
            {knobs.map((knob) => (
              <div key={knob.id} style={s.subRow}>
                <span style={s.subLabel}>{knob.label || knob.id.toUpperCase()}</span>
                <div style={s.chipRow}>
                  {(knob.values || [null]).map((entry) => {
                    const c = h.evaluateChip(entry, base);
                    if (c.hidden) return null;
                    return renderChip(labelFor(knob, c), value[knob.id], c.value,
                      (nv) => setKnob(knob.id, nv),
                      { disabled: c.disabled, disabledReason: c.disableReason });
                  })}
                </div>
              </div>
            ))}
          </div>
        );
      },
    },

    // ---- Axis: MoE Parallelism ----------------------------------------------
    // Backend (single-select; each option's `flags` is the source of truth)
    // and EP (numeric knob). Either sub-axis is independently optional.
    moe: {
      initState: () => ({ backend: null, ep: null }),

      revertHidden: (value, fc, base, h) => {
        let changed = false;
        const next = { ...value };
        if (next.backend !== null && fc.backend?.options
            && h.isHidden(fc.backend.options, next.backend, base)) {
          next.backend = null; changed = true;
        }
        if (next.ep !== null && fc.ep?.values
            && h.isHidden(fc.ep.values, next.ep, base)) {
          next.ep = null; changed = true;
        }
        return changed ? next : value;
      },

      apply: ({ flags, env, value, fc, h }) => {
        if (value.backend !== null) {
          flags = h.stripFlagsByFirstToken(flags, [
            "--moe-a2a-backend", "--moe-runner-backend",
          ]);
          const opt = (fc.backend?.options || []).find((o) => o.id === value.backend);
          if (opt?.flags?.length) {
            flags = h.insertAfter(flags, h.ANCHOR_NEAR_DPATTN, opt.flags);
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

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip }) => {
        if (!fc.backend && !fc.ep) return null;
        const setSlot = (k, v) => setValue({ ...value, [k]: v });
        return (
          <div key={axisId} style={{ ...s.card, ...s.cardStack }}>
            <div style={s.title}>MoE Parallelism</div>
            {fc.backend && (
              <div style={s.subRow}>
                <span style={s.subLabel}>Backend</span>
                <div style={s.chipRow}>
                  {(fc.backend.options || []).map((o) => {
                    const c = h.evaluateChip(o, base);
                    if (c.hidden) return null;
                    return renderChip(c.label, value.backend, c.value,
                      (v) => setSlot("backend", v),
                      { disabled: c.disabled, disabledReason: c.disableReason });
                  })}
                </div>
              </div>
            )}
            {fc.ep && (
              <div style={s.subRow}>
                <span style={s.subLabel}>{fc.ep.label || "EP"}</span>
                <div style={s.chipRow}>
                  {(fc.ep.values || [null]).map((entry) => {
                    const c = h.evaluateChip(entry, base);
                    if (c.hidden) return null;
                    const lbl = c.label ?? (c.value === null ? "auto" : String(c.value));
                    return renderChip(lbl, value.ep, c.value,
                      (nv) => setSlot("ep", nv),
                      { disabled: c.disabled, disabledReason: c.disableReason });
                  })}
                </div>
              </div>
            )}
          </div>
        );
      },
    },

    // ---- Axis: Parsers ------------------------------------------------------
    // Multi-toggle: one boolean per item. Engine UNCONDITIONALLY strips
    // `--reasoning-parser` and `--tool-call-parser` whenever the parsers axis
    // is declared (so the playground takes ownership of these flags), then
    // re-emits one item.flag per toggled-on item.
    parsers: {
      initState: (fc) => {
        const out = {};
        for (const item of (fc.items || [])) out[item.id] = false;
        return out;
      },

      revertHidden: (value, fc, base, h) => {
        let changed = false;
        const next = { ...value };
        for (const item of (fc.items || [])) {
          if (next[item.id] && h.evaluateChip(item, base).hidden) {
            next[item.id] = false; changed = true;
          }
        }
        return changed ? next : value;
      },

      apply: ({ flags, env, value, fc, h }) => {
        flags = h.stripFlagsByFirstToken(flags, ["--reasoning-parser", "--tool-call-parser"]);
        const adds = [];
        for (const item of (fc.items || [])) {
          if (value[item.id]) adds.push(item.flag);
        }
        if (adds.length) flags = h.insertBeforeTail(flags, adds);
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip }) => {
        const visible = (fc.items || [])
          .map((item) => ({ item, c: h.evaluateChip(item, base) }))
          .filter(({ c }) => !c.hidden);
        if (visible.length === 0) return null;
        return (
          <div key={axisId} style={{ ...s.card, ...s.cardStack }}>
            <div style={s.title}>Parsers</div>
            {visible.map(({ item, c }) => (
              <div key={item.id} style={s.subRow}>
                <span style={s.subLabel}>{item.label}</span>
                <div style={s.chipRow}>
                  {renderChip(item.label, value[item.id], true,
                    () => setValue({ ...value, [item.id]: !value[item.id] }),
                    { disabled: c.disabled, disabledReason: c.disableReason })}
                </div>
              </div>
            ))}
          </div>
        );
      },
    },

    // ---- Axis: Speculative Decoding -----------------------------------------
    // Single-select chip group. Sentinel values:
    //   "current" — leave base cell's `--speculative-*` flags untouched
    //   "off"     — strip them with no replacement (force greedy)
    //   <other>   — strip + splice option.flags
    speculative: {
      initState: () => "current",

      revertHidden: (value, fc, base, h) => {
        if (value !== "current" && h.isHidden(fc.options || [], value, base)) {
          return "current";
        }
        return value;
      },

      apply: ({ flags, env, value, fc, h }) => {
        if (value === "current") return { flags, env };
        flags = h.stripFlagsByFirstToken(flags, [
          "--speculative-algo", "--speculative-num-steps",
          "--speculative-eagle-topk", "--speculative-num-draft-tokens",
        ]);
        const preset = (fc.options || []).find((p) => p.id === value);
        if (preset?.flags?.length) flags = h.insertBeforeTail(flags, preset.flags);
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip }) => {
        const opts = fc.options || [];
        if (!opts.length) return null;
        return (
          <div key={axisId} style={s.card}>
            <div style={s.title}>Speculative Decoding</div>
            <div style={s.rowFlex}>
              {opts.map((p) => {
                const c = h.evaluateChip(p, base);
                if (c.hidden) return null;
                return renderChip(c.label, value, c.value, setValue,
                  { disabled: c.disabled, disabledReason: c.disableReason });
              })}
            </div>
          </div>
        );
      },
    },

    // ---- Axis: PD Disaggregation --------------------------------------------
    // Role select (off / prefill / decode) + optional IB device pick. Engine
    // OWNS the `--disaggregation-*` flags (unconditional strip). When a role
    // is picked, emits the role flag + transfer-backend + optional IB device
    // + (single-host only) bootstrap port. `getRenderHints` reports the
    // chosen role back to the renderer so it can prepend the role banner.
    pdDisagg: {
      initState: () => ({ mode: "off", ibDevice: "auto" }),

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
        return changed ? next : value;
      },

      apply: ({ flags, env, value, sel, h }) => {
        flags = h.stripFlagsByFirstToken(flags, [
          "--disaggregation-mode", "--disaggregation-transfer-backend",
          "--disaggregation-ib-device", "--disaggregation-bootstrap-port",
        ]);
        if (value.mode === "prefill" || value.mode === "decode") {
          const adds = [
            `--disaggregation-mode ${value.mode}`,
            "--disaggregation-transfer-backend mooncake",
          ];
          if (value.ibDevice && value.ibDevice !== "auto") {
            adds.push(`--disaggregation-ib-device ${value.ibDevice}`);
          }
          // Single-host bootstrap port only — multi-node cells already have
          // a NODE0_IP-based --dist-init-addr from the renderer.
          if (sel.nodes === "single"
              && !flags.some((f) => f.startsWith("--dist-init-addr"))) {
            const bootstrapPort = value.mode === "prefill" ? 30335 : 30435;
            adds.push(`--dist-init-addr 127.0.0.1:${bootstrapPort}`);
          }
          flags = h.insertBeforeTail(flags, adds);
        }
        return { flags, env };
      },

      getRenderHints: (value) => {
        if (value.mode === "prefill" || value.mode === "decode") {
          return { pdMode: value.mode };
        }
        return null;
      },

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip }) => {
        const setSlot = (k, v) => setValue({ ...value, [k]: v });
        const showModes  = (fc.modes      || []).length > 0;
        const showIb     = (fc.ibDevices  || []).length > 0;
        if (!showModes && !showIb) return null;
        return (
          <div key={axisId} style={{ ...s.card, ...s.cardStack }}>
            <div style={s.title}>PD Disaggregation</div>
            {showModes && (
              <div style={s.subRow}>
                <span style={s.subLabel}>Mode</span>
                <div style={s.chipRow}>
                  {fc.modes.map((m) => {
                    const c = h.evaluateChip(m, base);
                    if (c.hidden) return null;
                    return renderChip(c.label, value.mode, c.value,
                      (v) => setSlot("mode", v),
                      { disabled: c.disabled, disabledReason: c.disableReason });
                  })}
                </div>
              </div>
            )}
            {showIb && (
              <div style={s.subRow}>
                <span style={s.subLabel}>IB Device</span>
                <div style={s.chipRow}>
                  {fc.ibDevices.map((entry) => {
                    const c = h.evaluateChip(entry, base);
                    if (c.hidden) return null;
                    const lbl = c.label ?? String(c.value);
                    return renderChip(lbl, value.ibDevice, c.value,
                      (nv) => setSlot("ibDevice", nv),
                      { disabled: c.disabled, disabledReason: c.disableReason });
                  })}
                </div>
              </div>
            )}
          </div>
        );
      },
    },

    // ---- Axis: Hierarchical KV Cache ----------------------------------------
    // Enable toggle + optional backend + optional write policy. Engine OWNS
    // the `--hicache-*` family of flags (unconditional strip). Emission
    // follows the canonical upstream form documented in
    // docs/advanced_features/hicache_best_practices.mdx.
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

      apply: ({ flags, env, value, h }) => {
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

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip }) => {
        const setSlot = (k, v) => setValue({ ...value, [k]: v });
        return (
          <div key={axisId} style={{ ...s.card, ...s.cardStack }}>
            <div style={s.title}>Hierarchical KV Cache (HiCache)</div>
            <div style={s.subRow}>
              <span style={s.subLabel}>Enable</span>
              <div style={s.chipRow}>
                {renderChip("HiCache", value.enable, true,
                  () => setSlot("enable", !value.enable))}
              </div>
            </div>
            {(fc.backends || []).length > 0 && (
              <div style={s.subRow}>
                <span style={s.subLabel}>Storage</span>
                <div style={s.chipRow}>
                  {fc.backends.map((o) => {
                    const c = h.evaluateChip(o, base);
                    if (c.hidden) return null;
                    return renderChip(c.label, value.backend, c.value,
                      (v) => setSlot("backend", v),
                      { disabled: c.disabled, disabledReason: c.disableReason });
                  })}
                </div>
              </div>
            )}
            {(fc.writePolicies || []).length > 0 && (
              <div style={s.subRow}>
                <span style={s.subLabel}>Write Policy</span>
                <div style={s.chipRow}>
                  {fc.writePolicies.map((entry) => {
                    const c = h.evaluateChip(entry, base);
                    if (c.hidden) return null;
                    const lbl = c.label ?? String(c.value);
                    return renderChip(lbl, value.writePolicy, c.value,
                      (v) => setSlot("writePolicy", v),
                      { disabled: c.disabled, disabledReason: c.disableReason });
                  })}
                </div>
              </div>
            )}
          </div>
        );
      },
    },

    // ---- Axis: MegaMoE ------------------------------------------------------
    // Single-select. Has BOTH axis-level gating (requiresHw / excludesStrategy
    // — entire card hidden) AND per-option `hide` constraints, plus env
    // mutation (stripEnv + option.env). The axis-level gates are enforced in
    // `render` (returns null) and in `revertHidden` (auto-reset to "disabled"
    // when base moves into a gated cell).
    megamoe: {
      initState: () => "disabled",

      // Centralised gate check — used by both render and revertHidden.
      _gateOpen: (fc, base) => {
        const hwGate    = !fc.requiresHw       || fc.requiresHw.includes(base.hw);
        const stratGate = !fc.excludesStrategy || !fc.excludesStrategy.includes(base.strategy);
        return hwGate && stratGate;
      },

      revertHidden: (value, fc, base, h) => {
        if (!AXIS_HANDLERS.megamoe._gateOpen(fc, base)) {
          return value === "disabled" ? value : "disabled";
        }
        if (value !== "disabled" && h.isHidden(fc.options || [], value, base)) {
          return "disabled";
        }
        return value;
      },

      apply: ({ flags, env, value, fc, h }) => {
        if (!value || value === "disabled") return { flags, env };
        const opt = (fc.options || []).find((o) => o.id === value);
        if (!opt) return { flags, env };
        // The axis-level gate isn't re-checked here — revertHidden keeps the
        // state in sync with the base, so by the time apply runs, a non-
        // "disabled" value implies the card is visible and the user opted in.
        flags = h.stripFlagsByFirstToken(flags, [
          "--moe-a2a-backend", "--moe-runner-backend",
        ]);
        if (opt.flags?.length) {
          flags = h.insertAfter(flags, h.ANCHOR_NEAR_DPATTN, opt.flags);
        }
        env = h.stripEnvByPrefix(env, fc.stripEnv || []);
        if (opt.env?.length) env = [...env, ...opt.env];
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, h, renderChip }) => {
        if (!AXIS_HANDLERS.megamoe._gateOpen(fc, base)) return null;
        return (
          <div key={axisId} style={s.card}>
            <div style={s.title}>MegaMoE</div>
            <div style={s.rowFlex}>
              {(fc.options || []).map((o) => {
                const c = h.evaluateChip(o, base);
                if (c.hidden) return null;
                return renderChip(c.label, value, c.value, setValue,
                  { disabled: c.disabled, disabledReason: c.disableReason });
              })}
            </div>
          </div>
        );
      },
    },

  };

  // ==========================================================================
  // 6. Apply pipeline + render command
  // ==========================================================================
  // Thread the base cell's (flags, env) through every declared axis's apply
  // method, in AXIS_HANDLERS declaration order. Also collects render hints
  // from any axis that emits them (currently only pdDisagg's role banner).
  const applyAllDeltas = (baseFlags, baseEnv, allDeltas, sel) => {
    let flags = [...baseFlags];
    let env = [...(baseEnv || [])];
    let pdMode = null;
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (!fc) continue;
      const value = allDeltas[axisId];
      if (value === undefined) continue;
      ({ flags, env } = handler.apply({ flags, env, value, fc, sel, h: helpers }));
      if (handler.getRenderHints) {
        const hints = handler.getRenderHints(value, fc) || {};
        if (hints.pdMode) pdMode = hints.pdMode;
      }
    }
    return { flags, env, pdMode };
  };

  // Renderer (same shape as _deployment.jsx — multi-node prepending, env
  // block, hints, docker framing).
  //   pdMode — null | "prefill" | "decode": when present, prepends a banner
  //     explaining that the emitted command is only one of the PD-Disagg
  //     roles. The renderer doesn't generate prefill+decode+router pairs;
  //     the operator pairs the sibling role and router separately.
  //   mode   — "python" | "docker": shell-style invocation vs `docker run`
  //     wrapping. Image comes from `config.dockerImages[sel.hw]`.
  //   cellEnv is decoupled from `cell` so callers can pass a modified env
  //     (e.g. after MegaMoE's stripEnv + env append).
  const renderCommandLines = (cell, flags, cellEnv, sel, envValues, pdMode = null, mode = "python") => {
    const modelName = resolveModelName(sel);
    const nnodes = parseNnodes(sel.nodes);
    const multinode = nnodes > 1;
    let f = [...flags];
    if (multinode && !f.some((x) => x.startsWith("--nnodes"))) {
      const at = f.findIndex((x) => x.startsWith("--model-path")) + 1;
      f.splice(at, 0,
        `--nnodes ${nnodes}`,
        `--node-rank {{NODE_RANK}}`,
        `--dist-init-addr {{NODE0_IP}}:20000`);
    }
    let cmd;
    if (mode === "docker") {
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
      const banner =
        `# === PD Disaggregation: ${pdMode.toUpperCase()} role only ===\n` +
        `# This command runs the ${pdMode} server. To complete the topology,\n` +
        `# also run the ${sibling} role on its peer GPU host, then start the\n` +
        `# sglang_router with --pd-disaggregation pointing at both. See the\n` +
        `# §3.2 PD-Disagg notes above for IB device, ulimit, and MNNVL caveats.`;
      cmd = `${banner}\n${cmd}`;
    }
    return cmd;
  };

  // ==========================================================================
  // 7. Diff (line-level, greedy LCS-like)
  // ==========================================================================
  // Walk both sides, emit `unchanged` when they agree, `added` for
  // playground-only, `removed` for base-only. Not optimal but produces
  // readable output when the two share most lines (the common case).
  const computeDiff = (baseStr, pgStr) => {
    const a = baseStr.split("\n");
    const b = pgStr.split("\n");
    const aSet = new Set(a);
    const bSet = new Set(b);
    let i = 0, j = 0;
    const out = [];
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j]) {
        out.push({ line: b[j], kind: "unchanged" });
        i++; j++;
      } else if (j < b.length && !aSet.has(b[j])) {
        out.push({ line: b[j], kind: "added" });
        j++;
      } else if (i < a.length && !bSet.has(a[i])) {
        out.push({ line: a[i], kind: "removed" });
        i++;
      } else if (i < a.length) {
        i++;
      } else {
        j++;
      }
    }
    return out;
  };

  // ==========================================================================
  // 8. Style helper (dark-mode-aware)
  // ==========================================================================
  const makeStyles = (isDark) => ({
    container: { maxWidth: "900px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "8px" },
    card: {
      padding: "8px 12px",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      borderLeft: `3px solid ${isDark ? "#A78BFA" : "#8B5CF6"}`,
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
      padding: "4px 10px",
      border: `1px solid ${isDark ? "#9ca3af" : "#d1d5db"}`,
      borderRadius: "3px",
      cursor: "pointer",
      fontSize: "12px",
      userSelect: "none",
      background: isDark ? "#374151" : "#fff",
      color: isDark ? "#e5e7eb" : "inherit",
      minWidth: "44px",
      textAlign: "center",
    },
    chipChecked: { background: "#8B5CF6", color: "white", borderColor: "#8B5CF6" },
    chipDisabled: { cursor: "not-allowed", opacity: 0.4 },
    commandWrap: {
      position: "relative",
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
    badge: {
      display: "inline-flex", alignItems: "center", gap: "6px",
      padding: "2px 8px", borderRadius: "10px",
      background: isDark ? "#78350f" : "#fef3c7",
      color: isDark ? "#fde68a" : "#92400e",
      fontSize: "11px", fontWeight: 600,
    },
    badgeDot: { width: "8px", height: "8px", borderRadius: "50%", background: "#f59e0b" },
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
      padding: "6px 14px", background: "#8B5CF6", color: "white",
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

  // Env / placeholder values, shared with _deployment.jsx (same localStorage
  // key) so HOST/PORT/etc. are unified across the page.
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

  // Base selection — live-link to §3 via URL hash + custom event.
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

  // Deltas: one slot per declared axis. Shape is derived from AXIS_HANDLERS
  // crossed with `pgFeatures` — adding a new axis adds one slot here for
  // free.
  const initialDeltas = () => {
    const out = {};
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (fc) out[axisId] = handler.initState(fc);
    }
    return out;
  };
  const [deltas, setDeltas] = useState(initialDeltas);

  // When base cell changes, any picked override whose chip is now hidden
  // would silently linger as a ghost override. Walk every axis and revert
  // hidden picks back to that axis's inherit default. Disabled picks are
  // intentionally NOT reverted — `disable` is a soft warning, the user can
  // keep that override across base changes if they explicitly opted in.
  useEffect(() => {
    setDeltas((d) => {
      let next = d;
      let mutated = false;
      for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
        const fc = pgFeatures[axisId];
        if (!fc || !(axisId in d)) continue;
        const nv = handler.revertHidden(d[axisId], fc, base, helpers);
        if (nv !== d[axisId]) {
          if (!mutated) { next = { ...d }; mutated = true; }
          next[axisId] = nv;
        }
      }
      return mutated ? next : d;
    });
  }, [base.hw, base.variant, base.quant, base.strategy, base.nodes]);

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
  useEffect(() => { if (modal === "env") setEnvDraft(env); }, [modal, env]);
  const [runMode, setRunMode] = useState("python");

  // ==========================================================================
  // 10. Derived values
  // ==========================================================================
  const s = makeStyles(isDark);
  const baseCell = findCell(config.cells, base);
  const modelName = resolveModelName(base);

  let baseCommand = "";
  let playgroundCommand = "";
  let diffLines = [];
  if (baseCell) {
    baseCommand = renderCommandLines(baseCell, baseCell.flags, baseCell.env, base, env, null, runMode);
    const { flags: pgFlags, env: pgEnv, pdMode } = applyAllDeltas(baseCell.flags, baseCell.env, deltas, base);
    playgroundCommand = renderCommandLines(baseCell, pgFlags, pgEnv, base, env, pdMode, runMode);
    diffLines = computeDiff(baseCommand, playgroundCommand);
  }

  const curlText = interpolate(config.curl || "", env, modelName);

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

  // Inherited-base summary line (top strip).
  const baseSummary = baseCell
    ? `${base.hw.toUpperCase()} · ${base.variant} · ${base.quant.toUpperCase()} · ${base.strategy} · ${base.nodes}`
    : "(no verified cell at current §3 selection — showing playground only)";

  // ==========================================================================
  // 11. Render helpers
  // ==========================================================================
  // Chip selector. `current` is the value bound to the chip group; `onPick(v)`
  // is called when the user clicks a chip. Disabled chips are unclickable
  // (used for static "Coming soon" entries and for chips whose `disable`
  // constraint matches the current base).
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

  // ==========================================================================
  // 12. JSX render
  // ==========================================================================
  return (
    <div style={s.container} className="not-prose">
      {/* Inherited base summary */}
      <div style={s.baseStrip}>
        <span style={{ fontWeight: 600 }}>Inherited base from §3:</span>
        <code style={{ fontFamily: "Menlo, monospace" }}>{baseSummary}</code>
        <button style={s.resetBtn} onClick={resetAll}>Reset all overrides</button>
      </div>

      {/* Axes — one card per declared axis, in AXIS_HANDLERS declaration order.
          Axes whose key is missing from pgFeatures are skipped entirely. Each
          handler's render() may also return null for axis-level gating (e.g.
          MegaMoE hidden on Hopper / low-latency). */}
      {Object.entries(AXIS_HANDLERS).map(([axisId, handler]) => {
        const fc = pgFeatures[axisId];
        if (!fc) return null;
        const setValue = (next) => setDeltas((d) => ({ ...d, [axisId]: next }));
        return handler.render({
          axisId, value: deltas[axisId], setValue,
          fc, base, s, h: helpers, renderChip,
        });
      })}

      {/* Command box (diff vs verified base) */}
      <div style={s.card}>
        <div style={s.title}>Playground Command (diff vs verified base)</div>
        <div style={s.commandWrap}>
          <div style={s.commandHeader}>
            <div style={s.headerLeft}>
              <div style={s.badge}>
                <span style={s.badgeDot} />
                Auto-Estimated
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
            )) : "# No verified base cell at the current §3 selection.\n# Pick a supported hardware/variant in §3 to populate the playground base."}
          </pre>
        </div>
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
              Values persist in localStorage and are shared with §3.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
