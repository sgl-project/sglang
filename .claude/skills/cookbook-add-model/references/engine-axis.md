# Engine extension: add a new playground feature axis

Loaded on demand by the `cookbook-add-model` skill. **Rare** — adding a model
cookbook is data-only and never needs this. The current 8 built-in axes
(`attention`, `moe`, `parsers`, `speculative`, `pdDisagg`, `hicache`,
`hisparse`, `megamoe`) already cover the SGLang feature surface most cookbooks
need. Only add a new axis if a real cookbook needs it and the feature does not
fit any existing axis. Touches `_playground.jsx` only.

For the per-model config/cells/MDX reference see [authoring-reference.md](authoring-reference.md).

---

## 3.1 Decide

Before touching the engine, confirm:

- The feature is a STABLE part of the SGLang CLI surface (will appear in
  multiple cookbooks, not one-off).
- The feature cannot be expressed as a new option inside an existing axis
  (e.g. a new MoE backend belongs in `moe.backend.options`, not a new
  axis).
- The feature has a clean strip-prefix → emit-flag pattern.

If unsure, add it as data first (in one cookbook's config under an
existing axis) before promoting it to a built-in axis.

## 3.2 Pick the axis id and state shape

The axis id is the key in both `config.playgroundFeatures` and the
internal `deltas` object. Use camelCase, descriptive but short:
`mambaCache`, `attentionBackend`, `kvCacheDtype`.

The state shape is whatever `initState` returns. Common shapes:

- Single-select: a string sentinel (e.g. `"disabled"` / `"current"` / an
  option id).
- Multi-toggle: `{[itemId]: bool}`.
- Sub-knobs: `{[knobId]: value | null}`.
- Compound (axis with its own internal sub-state, like PD-Disagg's
  `{mode, ibDevice}`): a plain object.

Pick ONE "inherit base" sentinel and document it in the handler comment.

## 3.3 Implement the handler

Add one entry to `AXIS_HANDLERS` in `_playground.jsx`.
The handler owns everything: state init, apply (strip+insert), hidden-revert,
AND the JSX render. Engine main loop iterates `AXIS_HANDLERS` and calls each
method by name — adding a new axis is genuinely a one-place change.

Template:

```js
// ---- Axis: <Title> ----------------------------------------------------
// <one-paragraph description of what this axis controls and why it
//  exists. Mention the SGLang feature it wraps and the strip/insert
//  policy.>
<axisId>: {
  initState: (fc) => /* initial state value */,

  // Called when base cell changes. Return new value if the picked option
  // is now hidden by a constraint; otherwise return value unchanged.
  // Disabled picks are intentionally NOT auto-reverted (soft warning).
  revertHidden: (value, fc, base, h) => {
    // ... return value or a new value
    return value;
  },

  // Pure function. Receives the current (flags, env) and returns the next
  // (flags, env). Do NOT mutate inputs. The `value` argument is whatever
  // initState returned. The `fc` argument is config.playgroundFeatures[axisId].
  // The `sel` argument is the current base cell selection. The `h`
  // argument is the helpers bundle (strip/insert primitives + anchors).
  apply: ({ flags, env, value, fc, sel, h, derived }) => {
    if (/* value is the inherit-base sentinel */) return { flags, env };
    flags = h.stripFlagsByFirstToken(flags, [/* prefixes this axis owns */]);
    if (/* an option is picked */) {
      flags = h.insertAfter(flags, h.ANCHOR_NEAR_<X>, [/* new flags */]);
      // or: flags = h.insertBeforeTail(flags, [/* new flags */]);
      // if the axis mutates env:
      // env = h.stripEnvByPrefix(env, fc.stripEnv || []);
      // env = [...env, /* additional env vars */];
    }
    return { flags, env };
  },

  // Optional: read the base cell's flag array back into the same shape
  // initState/apply use. Render shows this as the default selection
  // (dropdown option or checked chip) when the state slot is the inherit
  // sentinel — so the user sees the cell's actual --tp / MoE backend /
  // spec preset instead of an opaque "Auto." When derive returns a real
  // value, the inherit-sentinel option is hidden from the control. Apply
  // also receives the derived
  // value (as `derived`) and may use it as a no-op shortcut when the
  // user's pick matches base. Skip when your axis owns flags that never
  // appear in base cells (PD-Disagg / HiCache / MegaMoE).
  // deriveFromBase: (cell, fc, h) => ({ ... }) | null,

  // Optional: hints for the renderer. Currently only pdDisagg uses this
  // to report its role banner. Omit if not needed.
  // getRenderHints: (value, fc) => ({ pdMode: ... }) | null,

  // Returns the axis card JSX. The outer div MUST have key={axisId} so
  // React can track it in the engine's map loop. Return null for
  // axis-level gating (e.g. MegaMoE on Hopper). Lay out as a single
  // compact horizontal row: title on the left, fields after.
  render: ({ axisId, value, setValue, fc, base, s, h, renderChip, renderSelect, derived }) => {
    if (/* axis-level gating fails */) return null;
    return (
      <div key={axisId} style={s.card}>
        <div style={s.compactRow}>
          <span style={s.axisTitle}>Axis Title</span>
          {/* For multi-option fields, use renderSelect(...) — the default.
              For on/off toggles or single-select chip groups, use
              renderChip instead (see "Control choice" in the conventions
              below). Read state from `value`; write via `setValue(next)`
              (replaces the whole axis slot). */}
          <span style={s.field}>
            <span style={s.fieldLabel}>Field</span>
            {renderSelect(value.slot, fc.entries, (v) =>
              setValue({ ...value, slot: v }), base)}
          </span>
        </div>
      </div>
    );
  },
},
```

**Important conventions**:

- Insert the entry in the position you want it rendered. `AXIS_HANDLERS`
  is iterated in insertion order for both render and apply.
- Use `h.ANCHOR_NEAR_*` constants for insertion. Add a new anchor to the
  helpers bundle if your axis needs to land somewhere new in the flag
  block.
- Use lowercase HTML JSX tags only. Capitalized tags get rebound by
  Mintlify.
- Inside `render`, read state via `value` (the slice for this axis).
  Write state via `setValue(next)` (replaces the whole slice). For
  compound axes, do `setValue({ ...value, [k]: nextK })`.
- Layout: one `s.compactRow` per axis card, `s.axisTitle` for the
  leading label, one `s.field` per (label + input) pair.
- Control choice — `renderSelect` vs `renderChip`:
  - `renderSelect(current, entries, onPick, base, labelFor?, opts?)` is
    the **default** compact control (a `<select>` dropdown). It filters
    hidden chips and disables greyed-out ones internally — no per-chip
    `evaluateChip` loop needed in the render body. Most axes use it
    (attention, moe, pdDisagg, hisparse, hicache, megamoe). Pass
    `{ hideValues: [<sentinel>] }` when your `deriveFromBase` resolved to
    a real value, so the inherit-sentinel ("Auto" / "Inherited" /
    "current") doesn't clutter the dropdown.
  - `renderChip(label, current, value, onPick, { disabled?, disabledReason? })`
    renders a **button** instead of a dropdown row. Use it for a chip
    group when you want the options laid out as buttons. It serves two
    shapes:
    - **Multi-toggle** (Parsers) — one independent on/off chip per item;
      `current` is that item's effective bool, `value` is `true`, so the
      chip is "checked" when the item is on.
    - **Single-select** (Speculative) — a radio-style group; pass the
      group's effective value as `current` and each option's id as
      `value`, so exactly one chip is checked (`current === value`).
    Chip groups own their visibility/disable filtering: loop
    `h.evaluateChip(opt, base)` in the render body, skip `c.hidden`,
    filter the inherit-sentinel yourself when `deriveFromBase` resolved
    to a real value, and forward `c.disabled` / `c.disableReason` into
    `renderChip`'s opts (this is what surfaces a disabled chip's tooltip,
    e.g. a "Coming soon" entry).
- Selected chips use the same terracotta (`#D45D44`) as the Deploy
  panel's selected button, so both widgets read as one
  visual system. Don't introduce a per-axis accent color.
- Default-from-base: if your axis can be read out of base cells'
  flags, implement `deriveFromBase` and have your render show the
  derived value when state is the sentinel (e.g.
  `const eff = value.tp !== null ? value.tp : (derived && derived.tp)`).
  This is what makes a fresh playground load show the user's actual
  recipe instead of "auto." Flag-parsing helpers on `h`:
  `parseIntFlag`, `hasFlag`, `findFlagArg`.
- **Avoid the `in` operator wrapped in unary** (`!(x in y)`). Mintlify's
  AST walker crashes on it (`TypeError: this[e] is not a function`). Use
  `obj.key === undefined` or `obj.id !== undefined` instead. Bare
  `if (key in obj)` (no surrounding `!`) is fine.

## 3.4 Document the per-cookbook schema

Edit the file header in `_playground.jsx` to add your new axis to the
"Recognised keys" list, with a one-line description of its schema.
Optionally add a paragraph below explaining its strip/insert policy.

Update the §2.3 axis table in [authoring-reference.md](authoring-reference.md) to list the new axis.

## 3.5 Migrate cookbooks that need it

For each cookbook that should expose this axis, add a
`playgroundFeatures.<axisId>` entry to its config. Verify the chip group
renders, options apply correctly, and the diff matches expectations.

---

## Pitfalls (engine work)

**Insertion anchor misses** — `insertAfter` falls back to right-after
`--model-path` if none of its anchor prefixes are present. If your axis
emits flags that should land somewhere specific, include the most likely
anchor prefixes in your call. Order doesn't matter (set semantics).

**Conditional strips** — Some axes strip ONLY when overridden
(`attention.tp`, `moe.backend`, `speculative`, `megamoe`). Others strip
UNCONDITIONALLY whenever declared (`parsers`, `pdDisagg`, `hicache`). The
header comment in `AXIS_HANDLERS` documents which policy each axis uses;
follow the same pattern when adding a new axis. If unsure, prefer
conditional strip — it preserves base behavior when the user does not
opt in.

**Closure of `AXIS_HANDLERS`** — Inside a handler method, you can
reference `AXIS_HANDLERS.<otherAxis>` for cross-handler calls (megamoe
does this for `_gateOpen`). This works because `AXIS_HANDLERS` is in
lexical scope. Do NOT use this for general logic — it tightly couples
handlers. Reserve it for one handler's helpers shared between its own
`render` and `revertHidden`.

---

## Review checklist for a new-axis PR

- [ ] `AXIS_HANDLERS` is the ONLY place that mentions the new axis id
      (apart from per-cookbook config). No `if (axisId === '<new>')`
      branches anywhere in the engine.
- [ ] `initState` is deterministic and idempotent (does not depend on
      the base cell).
- [ ] `apply` is pure — does not mutate inputs.
- [ ] `revertHidden` returns the same reference when nothing changed
      (avoids unnecessary re-renders).
- [ ] `render` returns `null` when axis-level gating fails (whole card
      hidden) — does not render an empty placeholder.
- [ ] `render` sets `key={axisId}` on its outer element.
- [ ] No `!(x in y)` patterns introduced (Mintlify AST walker crashes).
- [ ] File header lists the new axis in "Recognised keys".
- [ ] The §2.3 table in `authoring-reference.md` lists the new axis.
- [ ] One existing cookbook config is updated to consume the new axis,
      and visual verification shows the diff is correct.
