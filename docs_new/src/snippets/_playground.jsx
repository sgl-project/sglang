// Shared playground skeleton — the ENGINE half of the SGLang cookbook
// playground widget. Pair this with a per-model config file under
// `/src/snippets/configs/<vendor>/<model>.jsx` and an MDX page that imports
// both:
//
//     import { Playground } from "/src/snippets/_playground.jsx";
//     import { config }    from "/src/snippets/configs/deepseek-ai/deepseek-v4.jsx";
//     <Playground config={config} />
//
// AUTHORING — read .claude/rules/cookbook-authoring.md for the step-by-step
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
//     attention   — knobs[] with TP/CP/DP-Attention sub-controls (DP-Attention
//                   is a combined knob: numeric value sets --dp N AND
//                   --enable-dp-attention; `false` strips both)
//     moe         — backend.options[] + ep.values[]
//     parsers     — items[] (each emits one toggle flag)
//     speculative — options[] (single-select chip group)
//     pdDisagg    — modes[] + transferBackends[] (each may carry env +
//                   envWhen hw-gate) + ibDevices[] (engine handles role
//                   banner + single-host bootstrap port internally)
//     hicache     — backends[] + writePolicies[]
//     hisparse    — requiredFlags[] + config{} + hostRatios[] +
//                   defaultHostRatio (decode-only: card gated on the live
//                   PD-Disagg mode == "decode" via cross-axis fact pdMode)
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
//   initState(featureConfig)               → initial delta value (sentinels)
//   revertHidden(value, fc, base, helpers) → next value (same ref if unchanged)
//   apply({flags, env, value, fc, sel, h, derived}) → next {flags, env}
//   render({axisId, value, setValue, fc, base, s, h, renderChip,
//           renderSelect, derived}) → JSX | null
// Plus two optional methods:
//   deriveFromBase(cell, fc, h)            → value-shaped object recovered
//                                            from the base cell's flags.
//                                            Render uses it for default
//                                            dropdown display when state
//                                            slot is the inherit sentinel;
//                                            apply uses it (where needed)
//                                            to no-op when the user's pick
//                                            matches base.
//   getRenderHints(value, fc)              → {pdMode?, ...} hints for renderer
//
// DROPDOWN DEFAULT-FROM-BASE
// --------------------------
// Sentinel state (`null` / `"current"` / `"disabled"`) still represents
// "follow base." `deriveFromBase` reads the base cell's flags back into
// the same shape and the render layer DISPLAYS that derived value as the
// dropdown's selected entry — so users see their cell's actual --tp / MoE
// backend / spec preset, not an opaque "auto." When derive returns a real
// value, the inherit-sentinel option is filtered out of the dropdown via
// `renderSelect`'s `opts.hideValues`. State only flips off the sentinel
// when the user explicitly picks a different value; clicking "Reset all
// overrides" puts every axis back at its sentinel (and thus back at
// derived-from-base).
//
// Flag-parsing helpers available on the helpers bundle: `parseIntFlag`,
// `hasFlag`, `findFlagArg` — used inside `deriveFromBase` methods.
//
// LAYOUT CONVENTION
// -----------------
// Each axis card renders as a single compact horizontal row:
//
//     [Axis Title]  [field-label] [field-input]  [field-label] [field-input] ...
//
// Use `s.compactRow` for the outer flex, `s.axisTitle` for the leading label,
// `s.field` to wrap each `[label, input]` pair, `s.fieldLabel` for the inline
// sub-label. Multi-option fields render via `renderSelect(current, entries,
// onPick, base, labelFor?)` — emits a <select> with hidden-filtered options
// and disabled options greyed out. Pure enable/disable toggles still use
// `renderChip(label, current, true, () => onPick(!current))`.
//
// MINTLIFY GOTCHAS LEARNED THE HARD WAY
// -------------------------------------
// Mintlify's MDX/JSX AST walker has dispatch handlers for common node types
// but NOT all. We hit one: `!(x in y)` — i.e. `in` operator inside `!()` —
// crashes the walker with `TypeError: this[e] is not a function` because
// the UnaryExpression visitor recurses into a BinaryExpression child whose
// sub-handler is missing. Workaround patterns:
//   - Avoid `!(... in ...)`. Use `obj.key === undefined` or
//     `obj.id !== undefined` when checking key existence.
//   - Bare `if (key in obj)` (not wrapped in unary) works fine.
// If you hit a similar `this[e] is not a function` in dev server output,
// look for unusual JS expressions you added; rewrite to plainer JS.
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

  // Cross-cell verified-match lookup: after applying the playground's
  // overrides, the resulting (env, flags) tuple may equal SOME OTHER cell
  // in the catalog (e.g. low-latency + MTP-1-1-2 == balanced on H200/FP4).
  // Scope the search to cells sharing the same (hw, variant, quant, nodes)
  // — the playground can't override those four axes (they're set in §3.1),
  // so any match must agree on them. The remaining dimension (`strategy`)
  // is the natural axis across which two cells can render the same
  // command. Returns the matched cell or null.
  //
  // Matching policy:
  //   - flags : ordered-array equality (order matters — both for the
  //             rendered command's readability and for the diff display).
  //   - env   : set equality (env vars are essentially key=value bindings
  //             at runtime; their order in the cell array doesn't change
  //             behavior).
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
      // Rich option form has `id`; wrapper form has `value` (and no `id`).
      // Avoid the `in` operator — Mintlify's AST walker crashes on it.
      value: entry.id !== undefined ? entry.id : entry.value,
      label: entry.label,
      hidden,
      disabled,
      disableReason: entry.disableReason || "",
    };
  };

  // Lookup helpers used by both render code and revertHidden handlers.
  const findEntry = (entries, picked) => {
    for (const e of (entries || [])) {
      // Same rich-vs-wrapper resolution as evaluateChip.
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

  // Insert one or more new flags right after the most-specific anchor flag
  // present in `flags`. `afterAnyOf` is PRIORITY-ORDERED — the engine tries
  // each anchor in turn and uses the first one actually present, falling
  // through to less-specific siblings only when the more-specific anchors
  // are absent. Final fallback is right-after --model-path (always present).
  //
  // Priority order matters because cells often contain multiple anchor
  // candidates (e.g. both --tp and --model-path). Treating `afterAnyOf` as
  // a SET — i.e. returning the first flag in the FLAG list whose token is
  // anywhere in the set — would land on --model-path in many cells, even
  // though --tp is the closer conceptual sibling. That tiny placement
  // error then ripples into the diff: when a re-emitted flag lands at a
  // different position from base, the LCS-like diff silently drops shared
  // lines around the swap. Iterating anchors in priority order keeps
  // position-stable when the canonical sibling is present.
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
  // These exist so each axis can recover its initial display state from the
  // base cell's flag array — e.g. attention reads back "--tp 8" → 8, moe
  // reads back "--moe-a2a-backend deepep" → "deepep". All three are pure
  // and tolerate missing/empty inputs (return null when nothing matches).

  // Return the first integer arg of `--prefix N` (space- or `=`-delimited)
  // anywhere in `flags`, or null if the flag is absent / its arg isn't an int.
  const parseIntFlag = (flags, prefix) => {
    for (const f of (flags || [])) {
      if (f.split(/[\s=]/)[0] !== prefix) continue;
      const rest = f.slice(prefix.length).replace(/^[\s=]+/, "");
      const n = parseInt(rest, 10);
      if (!isNaN(n)) return n;
    }
    return null;
  };
  // True if any flag's first token equals `name` (used for boolean flags
  // like `--enable-dp-attention` that take no argument).
  const hasFlag = (flags, name) =>
    (flags || []).some((f) => f.split(/[\s=]/)[0] === name);
  // Return the string arg of `--prefix arg` (space- or `=`-delimited),
  // or null. Used for enum flags like --moe-a2a-backend.
  const findFlagArg = (flags, prefix) => {
    for (const f of (flags || [])) {
      if (f.split(/[\s=]/)[0] !== prefix) continue;
      const rest = f.slice(prefix.length).replace(/^[\s=]+/, "");
      return rest.length ? rest : null;
    }
    return null;
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
    parseIntFlag, hasFlag, findFlagArg,
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
  //                                  AND value !== derived (latter guard
  //                                  keeps base's spec flags in their
  //                                  original middle-of-list position when
  //                                  the user picks the preset that
  //                                  already matches base)
  //   - parsers                    : strip ONLY when ANY item's effective
  //                                  on/off differs from derived-from-base
  //                                  (used to be unconditional; switching
  //                                  to "only when overridden" preserves
  //                                  base's parser flag position when none
  //                                  of the toggles are user-touched)
  //   - pdDisagg                   : UNCONDITIONAL strip
  //   - hicache                    : UNCONDITIONAL strip
  //   - megamoe                    : strip ONLY when value !== "disabled"
  //                                  (also mutates env: stripEnvByPrefix +
  //                                  appends option.env)
  const AXIS_HANDLERS = {

    // ---- Axis: Attention Parallelism ----------------------------------------
    // Three sub-knobs (TP / CP / DP-Attention). Per-knob state slot; `null`
    // means inherit base. Each knob has its own strip prefix + insertion
    // anchor so overrides land near their conceptual siblings.
    //
    // DP-Attention is a COMBINED knob: its numeric value is the DP degree
    // AND simultaneously toggles `--enable-dp-attention`. Picking 4 emits
    // `--dp 4 --enable-dp-attention`; picking `false` ("off") strips both;
    // `null` ("auto") inherits the base cell verbatim. This matches the
    // DeepSeek-V4 deployment convention where plain `--dp` without
    // `--enable-dp-attention` is never used.
    attention: {
      initState: () => ({ tp: null, cp: null, dpAttn: null }),

      // Read the base cell's parallelism flags back into the same shape that
      // initState/apply use. The dropdowns DISPLAY this when the user hasn't
      // explicitly overridden a knob (state slot is null). Apply is unchanged
      // — it still no-ops on null, so untouched slots leave the base cell's
      // flag in place at its original position and the diff stays clean.
      //
      // For DP-Attention: if base has `--dp N --enable-dp-attention`, derive
      // returns N. If base has neither, return false ("off"). The edge case
      // of `--enable-dp-attention` without `--dp` collapses to 1.
      //
      // CP currently emits no numeric arg in apply (`--nsa-prefill-cp-mode
      // round-robin-split`), so derivation can only tell you on/off — we
      // collapse a present `--enable-nsa-prefill-context-parallel` to 2.
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
        // DP-Attention: a combined knob. value can be null (inherit), false
        // (off — strip both flags), or a positive number (emit the --dp +
        // --enable-dp-attention pair).
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
        // Label resolution. DP-Attention is the combined-DP knob — its
        // value can be `null` (auto/inherit), `false` (off, both flags
        // stripped), or a positive integer (the DP degree, which also
        // turns on --enable-dp-attention). The `labels` map covers the
        // sentinels (auto / off); numeric values fall through to their
        // bare digits.
        const labelFor = (knob) => (c) => {
          if (c.label !== undefined) return c.label;
          if (knob.id === "dpAttn") {
            const labelMap = knob.labels || { "auto": "auto", "false": "off" };
            const k = c.value === null ? "auto" : String(c.value);
            return labelMap[k] || k;
          }
          return c.value === null ? "auto" : String(c.value);
        };
        // Display rule: explicit user pick (non-null) > derived-from-base >
        // null sentinel. When derive produced a real value, hide the
        // sentinel option from the dropdown.
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
    // Backend (single-select; each option's `flags` is the source of truth)
    // and EP (numeric knob). Either sub-axis is independently optional.
    moe: {
      initState: () => ({ backend: null, ep: null }),

      // Recover MoE backend choice from the base cell's flags. When BOTH
      // `--moe-a2a-backend` and `--moe-runner-backend` are present (e.g.
      // h200 fp8 high-throughput layers both), we prefer `--moe-a2a-backend`
      // — it's the "louder" architectural pick and matches the
      // single-select dropdown's semantics.
      deriveFromBase: (cell, fc, h) => {
        const flags = (cell && cell.flags) || [];
        const a2a    = h.findFlagArg(flags, "--moe-a2a-backend");
        const runner = h.findFlagArg(flags, "--moe-runner-backend");
        return {
          backend: a2a || runner || null,
          ep: h.parseIntFlag(flags, "--ep"),
        };
      },

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

      render: ({ axisId, value, setValue, fc, base, s, renderSelect, derived }) => {
        if (!fc.backend && !fc.ep) return null;
        const setSlot = (k, v) => setValue({ ...value, [k]: v });
        // Same display rule as attention: explicit > derived > null.
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
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>MoE</span>
              {fc.backend && (
                <span style={s.field}>
                  <span style={s.fieldLabel}>Backend</span>
                  {renderSelect(slotDisplay("backend"), fc.backend.options || [],
                    (v) => setSlot("backend", v), base, undefined,
                    { hideValues: hideNull("backend") })}
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
    // Multi-toggle: one boolean per item. Engine UNCONDITIONALLY strips
    // `--reasoning-parser` and `--tool-call-parser` whenever the parsers axis
    // is declared (so the playground takes ownership of these flags), then
    // re-emits one item.flag per toggled-on item.
    parsers: {
      // Tri-state per item: null = inherit-from-base, true/false = explicit
      // user pick. Stored as null so the toggle button can show base's
      // current parser state by default and apply can preserve base's flag
      // position when nothing's been overridden.
      initState: (fc) => {
        const out = {};
        for (const item of (fc.items || [])) out[item.id] = null;
        return out;
      },

      // Look up each item's `--reasoning-parser` / `--tool-call-parser`
      // (etc.) flag in the base cell and report on/off accordingly.
      deriveFromBase: (cell, fc, h) => {
        const flags = (cell && cell.flags) || [];
        const out = {};
        for (const item of (fc.items || [])) {
          // Each `item.flag` is like "--reasoning-parser deepseek-v4". Take
          // the first token as the prefix to look up.
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
        // Per-item effective state: explicit value if non-null, else derived
        // from base, else false. The `effective !== base-derived` check below
        // determines whether ANY user override is in play — if none, we skip
        // the strip+emit so base's parser flag (if any) stays in place at
        // its original position and the diff stays clean.
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
        // Effective on/off per item: explicit pick > derived-from-base > off.
        // Click flips the effective value and stores it as an explicit pick.
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
    // Single-select chip group. Sentinel values:
    //   "current" — leave base cell's `--speculative-*` flags untouched
    //   "off"     — strip them with no replacement (force greedy)
    //   <other>   — strip + splice option.flags
    speculative: {
      initState: () => "current",

      // Try to recover which preset the base cell already encodes by
      // comparing the base's `--speculative-*` flag set against each
      // option's `flags` array as a multiset. If nothing matches:
      //   - no spec flags at all in base → "off"
      //   - some spec flags but no preset matches → "current" (preserve
      //     base as-is; the dropdown will fall back to "Inherited" since
      //     "current" is still a valid sentinel here)
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
        // Position-preservation shortcut: if the user picked the same preset
        // the base cell already encodes, no-op. Without this, strip +
        // insertBeforeTail would re-place the spec block at the tail, but
        // base cells keep their spec flags in the middle — so the diff
        // would show base's lines as removed and identical lines as added.
        if (derived && value === derived) return { flags, env };
        flags = h.stripFlagsByFirstToken(flags, [
          "--speculative-algorithm", "--speculative-num-steps",
          "--speculative-eagle-topk", "--speculative-num-draft-tokens",
          "--speculative-ngram-max-bfs-breadth",
        ]);
        const preset = (fc.options || []).find((p) => p.id === value);
        if (preset?.flags?.length) flags = h.insertBeforeTail(flags, preset.flags);
        return { flags, env };
      },

      render: ({ axisId, value, setValue, fc, base, s, renderSelect, derived }) => {
        const opts = fc.options || [];
        if (!opts.length) return null;
        // Display rule: explicit pick (!= "current") wins; otherwise show
        // whatever deriveFromBase matched. Hide the "current" sentinel from
        // the dropdown when derive resolved to a real preset (or "off") —
        // keep it only when nothing matched, so "Inherited from base" is
        // still pickable as the no-match fallback.
        const display = (value !== "current") ? value
                       : (derived ? derived : "current");
        const hideValues = (derived && derived !== "current") ? ["current"] : [];
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>Speculative</span>
              <span style={s.field}>
                {renderSelect(display, opts, setValue, base, undefined,
                  { hideValues })}
              </span>
            </div>
          </div>
        );
      },
    },

    // ---- Axis: PD Disaggregation --------------------------------------------
    // Role select (off / prefill / decode) + transfer-backend select + optional
    // IB device pick. Engine OWNS the `--disaggregation-*` flags (unconditional
    // strip). When a role is picked, emits the role flag + the selected
    // transfer backend + optional IB device + (single-host only) bootstrap
    // port. A transfer backend may also carry per-backend env vars gated by
    // the base cell's hw (config `transferBackends[].env` + `.envWhen`) — e.g.
    // mooncake's MNNVL vars on GB200/GB300. `getRenderHints` reports the chosen
    // role back to the renderer so it can prepend the role banner.
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
        return changed ? next : value;
      },

      apply: ({ flags, env, value, sel, fc, h }) => {
        flags = h.stripFlagsByFirstToken(flags, [
          "--disaggregation-mode", "--disaggregation-transfer-backend",
          "--disaggregation-ib-device", "--disaggregation-bootstrap-port",
        ]);
        // This axis also owns whatever per-backend env its transfer backends
        // declare (e.g. mooncake's MNNVL vars). Strip them all up front so
        // apply stays idempotent; re-add below only when a role is active AND
        // the chosen backend's hw gate matches the base cell.
        const backends = fc.transferBackends || [];
        const ownedEnvKeys = [];
        for (const b of backends) {
          for (const e of (b.env || [])) ownedEnvKeys.push(e.split("=")[0]);
        }
        if (ownedEnvKeys.length) env = h.stripEnvByPrefix(env, ownedEnvKeys);

        if (value.mode === "prefill" || value.mode === "decode") {
          const backend = value.transferBackend || "mooncake";
          const adds = [
            `--disaggregation-mode ${value.mode}`,
            `--disaggregation-transfer-backend ${backend}`,
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

          // Per-backend env, gated by the base cell's hw via config `envWhen`
          // (constraint object — every key must match; no gate = always on).
          const meta = backends.find((b) => b.id === backend);
          if (meta && meta.env && meta.env.length) {
            const gate = meta.envWhen;
            const ok = !gate || Object.keys(gate).every(
              (k) => (gate[k] || []).includes(sel[k]));
            if (ok) env = [...env, ...meta.env];
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
    // Enable toggle + host_to_device_ratio select. Engine OWNS the
    // `--enable-hisparse` / `--hisparse-config` flags PLUS the required
    // decode-side companions declared in `fc.requiredFlags`
    // (`--kv-cache-dtype bfloat16`, `--nsa-decode-backend flashmla_sparse`).
    // All are stripped unconditionally and re-added only when enabled.
    //
    // HiSparse is a DECODE-INSTANCE-ONLY optimization that requires PD
    // disaggregation (per docs/advanced_features/hisparse_guide.mdx), so it
    // is gated TWO ways:
    //   - render hides the whole card unless the live PD-Disagg mode is
    //     `decode` (cross-axis fact `base.pdMode`); and
    //   - apply only emits when `--disaggregation-mode decode` is already in
    //     the flag list. pdDisagg's apply runs BEFORE this axis (declaration
    //     order), so that flag is present by the time we check — this keeps
    //     emission correct even if the enable toggle is left on while the
    //     user flips PD-Disagg away from decode.
    // top_k / device_buffer_size come from `fc.config`; only the memory-
    // dependent host_to_device_ratio is exposed as a knob.
    hisparse: {
      initState: (fc) => ({ enable: false, hostRatio: (fc && fc.defaultHostRatio) || null }),

      revertHidden: (value, fc, base, h) => {
        // hostRatio chips carry no hide constraints today; nothing to revert.
        // Card visibility (PD decode) is enforced in render + the apply decode
        // gate, NOT here — revertHidden only sees the clean 5-dim base.
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
        // Decode-instance gate: pdDisagg (which runs first) inserts
        // `--disaggregation-mode decode` when that role is picked.
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
        // Decode-only: hide the card unless PD-Disagg mode is decode.
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

      render: ({ axisId, value, setValue, fc, base, s, renderChip, renderSelect }) => {
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

    // ---- Axis: MegaMoE ------------------------------------------------------
    // Single-select. Has BOTH axis-level gating (requiresHw / excludesStrategy
    // — entire card hidden) AND per-option `hide` constraints, plus env
    // mutation (stripEnv + option.env). The axis-level gates are enforced in
    // `render` (returns null) and in `revertHidden` (auto-reset to "disabled"
    // when base moves into a gated cell).
    megamoe: {
      initState: () => "disabled",

      revertHidden: (value, fc, base, h) => {
        // Axis-level gates: hw must be in requiresHw, strategy must NOT be in
        // excludesStrategy. Same check is duplicated in the JSX render below
        // for hiding the card; keep them in sync.
        const hwGate    = !fc.requiresHw       || fc.requiresHw.includes(base.hw);
        const stratGate = !fc.excludesStrategy || !fc.excludesStrategy.includes(base.strategy);
        if (!hwGate || !stratGate) {
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

      render: ({ axisId, value, setValue, fc, base, s, renderSelect }) => {
        // Inline axis-level gate check (matches the one in revertHidden above).
        // Kept in sync rather than extracted to a shared method to avoid the
        // cross-reference complexity that hurt readability earlier.
        const hwGate    = !fc.requiresHw       || fc.requiresHw.includes(base.hw);
        const stratGate = !fc.excludesStrategy || !fc.excludesStrategy.includes(base.strategy);
        if (!hwGate || !stratGate) return null;
        return (
          <div key={axisId} style={s.card}>
            <div style={s.compactRow}>
              <span style={s.axisTitle}>MegaMoE</span>
              <span style={s.field}>
                {renderSelect(value, fc.options || [], setValue, base)}
              </span>
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
      // Match _deployment.jsx: insert the multi-node trio after the last
      // parallelism flag rather than after --model-path. Keeps the
      // playground's untouched-base command byte-identical to the Deploy
      // panel's command (and to the live website's IC generator).
      const PARALLELISM_ANCHORS = ["--enable-dp-attention", "--dp", "--tp"];
      let at = -1;
      for (const anchor of PARALLELISM_ANCHORS) {
        at = f.findIndex((x) => x.split(/[\s=]/)[0] === anchor);
        if (at !== -1) break;
      }
      if (at === -1) at = f.findIndex((x) => x.startsWith("--model-path"));
      f.splice(at + 1, 0,
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
  // 7. Diff (line-level, true LCS)
  // ==========================================================================
  // Dynamic-programming LCS + backtrace. Emits `unchanged` for matching
  // line pairs, `added` for playground-only lines, `removed` for base-only.
  // Never silently drops lines — every input line ends up in the output
  // exactly once.
  //
  // Why bother with full LCS over a greedy one-pass: the playground's apply
  // pipeline sometimes reorders flags (e.g. picking a different speculative
  // preset moves the `--speculative-*` block from the middle of the cell
  // toward `--host` via insertBeforeTail). A greedy diff that walks both
  // sides in lockstep silently drops the shared lines (e.g. `--speculative-
  // algo EAGLE`) when they appear at different positions on the two sides,
  // producing visibly truncated commands. True LCS handles reordering by
  // computing the maximal matching set first, then aligning.
  //
  // Our commands are ~15 lines so the O(m·n) cost is trivial.
  const computeDiff = (baseStr, pgStr) => {
    const a = baseStr.split("\n");
    const b = pgStr.split("\n");
    const m = a.length, n = b.length;
    // dp[i][j] = LCS length between a[0..i] and b[0..j].
    const dp = Array(m + 1).fill(null).map(() => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
        else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    // Backtrace. Walk from (m, n) → (0, 0); when both sides match, emit
    // `unchanged`; otherwise prefer the side whose DP value is higher
    // (standard LCS tie-breaking).
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
  // Format a cell object the same way the cookbook config does, so a
  // maintainer can paste the snippet directly into `cells: [...]` without
  // reformatting. Each line is indented to match the surrounding array
  // (8 spaces for cell fields, 8 spaces continuation for env/flags items).
  //
  // The output intentionally keeps {{MODEL_NAME}}, {{HOST_IP}}, {{PORT}}
  // etc. as raw placeholders (NOT interpolated) — that's how the rest of
  // the catalog stores them so the deployment widget can render the right
  // model name per (hw|variant|quant) lookup.
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

  // Build the GitHub Issue prefill URL. Field IDs must match the
  // `id:` values in .github/ISSUE_TEMPLATE/3-playground-verified-cell.yml
  // — those are the URL query keys GitHub recognizes.
  //
  // `config.github` lets per-cookbook configs override the repo target
  // and the issue-template filename; defaults match the SGLang repo.
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
    // Compact one-line axis row: title + fields all on the same line.
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
    chipChecked: {
      background: isDark ? "#FDBA74" : "#FB923C",
      color: isDark ? "#7C2D12" : "white",
      borderColor: isDark ? "#FDBA74" : "#FB923C",
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
    // Two-state badge — same shape and colors as the §3.1 Deployment
    // widget's verified/unverified badge so the playground can inherit the
    // base cell's verified status when no overrides are in play.
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
    // Native <dialog> in top-layer mode (via .showModal()) bypasses all
    // stacking-context rules — necessary because Mintlify's mdx-content
    // container uses `container-type: inline-size` (CSS Container Queries),
    // which makes it a containing block for `position: fixed` descendants
    // and traps any plain fixed-position modal inside that container. The
    // browser-native top layer is rendered ABOVE all page content,
    // including the sticky Mintlify navbar.
    //
    // We style the dialog element directly. The ::backdrop pseudo-element
    // can't be styled inline, so a small global stylesheet is injected
    // once via useEffect.
    dialog: {
      background: isDark ? "#1f2937" : "#fff",
      color: isDark ? "#e5e7eb" : "#111827",
      borderRadius: "8px", padding: "20px",
      maxWidth: "720px", width: "92%",
      maxHeight: "calc(100vh - 80px)", overflowY: "auto",
      border: `1px solid ${isDark ? "#374151" : "#e5e7eb"}`,
      boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
      // Browser-default dialog margin = "auto" → centers in viewport,
      // which is exactly what we want.
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
    // Inline annotation that appears next to the verified pill when the
    // playground's command matches a SIBLING verified cell (different
    // strategy). The hint stays subdued — it's informational, not the
    // primary status indicator. The "switch base" link sits inside the
    // hint and uses the orange brand color to advertise as an action.
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
        // Avoid `!(axisId in d)` — Mintlify's AST walker crashes on the
        // `in` operator wrapped in a unary. Use undefined check instead;
        // deltas never store explicit-undefined values.
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

  // Open the native <dialog> as soon as React mounts it. The callback-ref
  // form fires once per mount so this is effectively "show on render". The
  // top-layer behavior comes from .showModal() (NOT plain .show()) —
  // showModal also handles ESC, focus trap, and body-scroll lock natively,
  // which is why we no longer need a manual keydown / overflow effect.
  const openDialog = (el) => {
    if (el && !el.open) {
      try { el.showModal(); } catch { /* already open or unsupported */ }
    }
  };

  // Click-outside-to-close. Clicks on the ::backdrop dispatch to the dialog
  // element itself, but with coordinates outside the dialog's bounding rect
  // — checking the click point against the rect distinguishes "clicked the
  // backdrop" from "clicked inside the box".
  const onDialogClick = (e) => {
    if (e.target !== e.currentTarget) return;
    const r = e.currentTarget.getBoundingClientRect();
    const { clientX: x, clientY: y } = e;
    if (x < r.left || x > r.right || y < r.top || y > r.bottom) setModal(null);
  };

  // ::backdrop is a pseudo-element so it can't be styled inline. Inject
  // a single dim-overlay rule into <head> once for the lifetime of the
  // component instance.
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
  const [envDraft, setEnvDraft] = useState(env);
  useEffect(() => { if (modal === "env") setEnvDraft(env); }, [modal, env]);
  const [runMode, setRunMode] = useState("python");

  // Submit-verified-cell modal state: free-text fields + the 3 required
  // attestation booleans. Reset each time the modal opens so a re-open
  // doesn't carry stale text into a different combination.
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
  const baseCell = findCell(config.cells, base);
  const modelName = resolveModelName(base);

  // Per-axis "derived from base" map — each handler's `deriveFromBase`
  // (when defined) reads its own slots out of the base cell's flag array.
  // Used by render (for default dropdown display when state slot is the
  // inherit sentinel) AND by apply (parsers needs it to decide whether
  // anything is actually overridden vs matching base).
  const derivedMap = {};
  if (baseCell) {
    for (const [axisId, handler] of Object.entries(AXIS_HANDLERS)) {
      const fc = pgFeatures[axisId];
      if (!fc || !handler.deriveFromBase) continue;
      derivedMap[axisId] = handler.deriveFromBase(baseCell, fc, helpers);
    }
  }

  // Cross-axis constraint facts. Some chip constraints must react to the
  // LIVE state of a DIFFERENT axis, not just the base cell's 5 match dims —
  // e.g. NGRAM speculative is incompatible with DP-Attention no matter which
  // strategy the base cell uses, and the user can flip DP-Attention live in
  // the Attention card. We derive those facts here and fold them into the
  // `base` object handed to chip-constraint matching (`matchConstraint`
  // reads them exactly like a real dim). They are NOT cell dimensions —
  // only `hide`/`disable` constraint objects reference them, and only the
  // RENDER path sees them (revertHidden keeps the clean 5-dim base, which
  // is fine because cross-axis constraints use `disable`, never auto-
  // reverted `hide`).
  //
  //   dpAttnOn — true when the effective DP-Attention resolves to "on":
  //              the explicit Attention-card override if set, else the
  //              value derived from the base cell. A positive DP degree
  //              (or boolean true) is "on"; false / 0 / null is "off".
  //   pdMode   — the live PD-Disaggregation role ("off" / "prefill" /
  //              "decode"). PD is a playground-only axis (no base derive),
  //              so this reads straight from the delta. Used to gate the
  //              decode-only HiSparse card.
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
  // Cross-cell verified detection: search the catalog for any cell whose
  // (env, flags) tuple equals what the playground would emit. The match can
  // be the base cell itself (untouched playground — current behavior) OR a
  // sibling with a different `strategy` (e.g. low-latency + MTP-1-1-2 on
  // H200/FP4 produces the same command as the balanced cell).
  //
  // When the match is a different cell, we still show the green Verified
  // badge — the recipe IS in the verified catalog — but annotate which
  // cell it matches and offer a "switch base" link so the user can align
  // §3.1's selection with the cell they've actually built.
  const matchedCell = baseCell
    ? findMatchingCell(config.cells, base, pgEnvLatest, pgFlagsLatest)
    : null;
  const playgroundVerified = !!(matchedCell && matchedCell.verified);
  const matchedSiblingCell = (matchedCell && matchedCell !== baseCell)
    ? matchedCell : null;

  // Submission snippets — the proposed cell to add to the cookbook config,
  // plus the existing one at the same match (for the maintainer's diff).
  // Computed only when the playground actually differs from base; otherwise
  // there's nothing to submit (the badge is already Verified).
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

  // Dropdown selector. Returns a <select> whose options come from a chip-schema
  // `entries` array (same shape as renderChip consumes — bare value, wrapper
  // object, or rich option object). Hidden entries are excluded; disabled
  // entries appear with a "(n/a)" suffix and cannot be selected.
  //
  // To sidestep all the string/number/null/boolean serialization headaches
  // that come with HTML form values, we use the option's array index as the
  // <select> value and resolve back to the original chip value on change.
  // `current` is the picked value (whatever the axis stores in state) and
  // `onPick(v)` receives the original value (not the index).
  //
  // `labelFor(c)` is an optional custom label resolver — receives the
  // evaluated chip and returns a string. Falls back to c.label, then to
  // "auto" for null, then to String(c.value).
  //
  // `opts.hideValues` (optional) — array of chip values to suppress from
  // the dropdown entirely. Used when an axis derives a real default from
  // the base cell, so the inherit-sentinel ("auto" / "Inherited" /
  // "current") doesn't clutter the dropdown.
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
          : c.value === null ? "auto" : String(c.value));
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
        <span style={{ fontWeight: 600 }}>Inherited base from §3.1:</span>
        <code style={{ fontFamily: "Menlo, monospace" }}>{baseSummary}</code>
        {/* Scroll back to §3.1 (the Interactive Command Generator). Use
            scrollIntoView so the URL hash — which carries the base cell
            selection — isn't overwritten. The target id is the auto-
            generated Mintlify slug for "## Deploy". */}
        <button
          type="button"
          style={s.switchBaseBtn}
          onClick={() => {
            const el = document.getElementById("deploy");
            if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
        >
          ↑ Switch base
        </button>
        <button style={s.resetBtn} onClick={resetAll}>Reset all overrides</button>
      </div>

      {/* Axes — one card per declared axis, in AXIS_HANDLERS declaration
          order. Axes whose key is missing from pgFeatures are skipped
          entirely. Each handler's render() may also return null for
          axis-level gating (e.g. MegaMoE hidden on Hopper / low-latency).
          Engine never switches on axis id here. */}
      {Object.entries(AXIS_HANDLERS).map(([axisId, handler]) => {
        const fc = pgFeatures[axisId];
        if (!fc) return null;
        const setValue = (next) => setDeltas((d) => ({ ...d, [axisId]: next }));
        return handler.render({
          axisId, value: deltas[axisId], setValue,
          // constraintBase = the 5 cell dims + cross-axis derived facts
          // (e.g. dpAttnOn) so chip `disable`/`hide` can react to other
          // axes' live state. Real dims are preserved, so base.hw etc.
          // direct reads inside render still work.
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
              {/* When the playground matches a sibling cell (different
                  `strategy`) — e.g. low-latency + MTP-1-1-2 on H200/FP4
                  matches the balanced cell — surface that fact and offer
                  to switch §3.1's base over to the matched cell. Updating
                  the URL hash propagates via the existing hashchange
                  listener; clearing deltas restores the clean
                  inherit-from-base state at the new selection. */}
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
              {/* "Submit verified cell" only makes sense when the playground
                  differs from the verified base — otherwise there's nothing
                  to submit. Hide the button when the badge is already green. */}
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
            )) : "# No verified base cell at the current §3 selection.\n# Pick a supported hardware/variant in §3 to populate the playground base."}
          </pre>
        </div>
      </div>

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
            Values persist in localStorage and are shared with §3.
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
            {/* Overrides summary — the actual diff vs base in flag form.
                Lists every added / removed line so the maintainer (and the
                user double-checking) can see exactly what's new in this
                submission without scrolling back to the command box. */}
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
