# Cookbook Authoring Guide

This guide tells future contributors (humans and coding agents) how to:

1. **Add a new model cookbook** — write a config file + an MDX page. No engine
   edits required. ~95% of tasks fall here.
2. **Add a new playground feature axis** — extend the engine with a new
   built-in axis. Rare; touches `_playground.jsx` only.

Before doing either, read the file headers of the two engine files (they are
the canonical contract specs):

- [_deployment.jsx](../../docs_new/src/snippets/_deployment.jsx) — the 5-dim matrix widget. Header
  lists every config field the engine reads.
- [_playground.jsx](../../docs_new/src/snippets/_playground.jsx) — the diff-based override widget.
  Header lists the recognised `playgroundFeatures` axes and the
  `AXIS_HANDLERS` interface.

A reference config to copy from:
[configs/deepseek-ai/deepseek-v4.jsx](../../docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx).

---

## 1. Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  cookbook/<category>/<Vendor>/<Model>.mdx                       │
│  ─────────────────────────────────────────────                  │
│  import { Deployment } from "/src/snippets/_deployment.jsx";    │
│  import { Playground } from "/src/snippets/_playground.jsx";    │
│  import { config }     from "/src/snippets/configs/.../X.jsx";  │
│  <Deployment config={config} />                                 │
│  <Playground config={config} />                                 │
└─────────────────────────────────────────────────────────────────┘
              │ (config passed as React prop)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/snippets/configs/<vendor>/<model>.jsx                      │
│  ──────────────────────────────────────────                     │
│  export const config = {                                        │
│    supportedHardware: [...],                                    │
│    variants: [...], quantizations: [...], ...                   │
│    cells: [                          // 5-dim matrix entries    │
│      { match: { hw, variant, quant, strategy, nodes },          │
│        env: [...], flags: [...] },                              │
│      ...                                                        │
│    ],                                                           │
│    playgroundFeatures: {             // opt-in per axis         │
│      attention: { knobs: [...] },                               │
│      parsers:   { items: [...] },                               │
│      ...                                                        │
│    },                                                           │
│  };                                                             │
└─────────────────────────────────────────────────────────────────┘
              │ (config consumed by both engines)
              ▼
┌──────────────────────────────────┬──────────────────────────────┐
│  _deployment.jsx                 │  _playground.jsx             │
│  Renders the verified matrix.    │  Renders override chips +    │
│  Picks one cell at a time and    │  diff against §3.1's cell.   │
│  shows its env/flags verbatim.   │  Strip+insert pipeline.      │
└──────────────────────────────────┴──────────────────────────────┘
```

The two widgets are coupled through:
- **URL hash** — `_deployment.jsx` mirrors its selection into the URL hash;
  `_playground.jsx` reads it on mount.
- **Custom event `sglang-deploy-sel`** — `_deployment.jsx` dispatches it on
  every selection change; `_playground.jsx` listens. (URL hash alone is not
  enough because `history.replaceState` does not fire `hashchange`.)
- **Shared localStorage key `sglang-deploy-env`** — both widgets share
  placeholder values (HOST/PORT/etc.) across visits and across cookbooks.

Engines contain NO model-specific code. Adding a cookbook = adding data only.

---

## 2. Adding a new model cookbook

### 2.1 Create the config file

**Path**: `src/snippets/configs/<vendor>/<model>.jsx`. The vendor folder is
the HuggingFace org (`deepseek-ai`, `Qwen`, `moonshotai`, ...); the file
name is a short hyphenated model id (`deepseek-v4`, `qwen3.5`, ...).

**Shape**: must be a single `export const config = { ... }` literal. Do not
use function calls, spreads, fragment refs, or IIFE — Mintlify re-evaluates
this export at hydration time with module-level identifiers out of scope,
and any non-literal value crashes with `ReferenceError`.

**Required fields** (engine reads these — see
[_deployment.jsx](../../docs_new/src/snippets/_deployment.jsx) header for the full contract):

| Field | Type | Purpose |
|---|---|---|
| `modelName` | string | Display label only. Not used for HF slug — see `modelNames`. |
| `supportedHardware` | `string[]` | Which hw ids appear in the catalog. Subset of the keys in `HARDWARE_CATALOG` in `_deployment.jsx`. Listing an id makes its button appear; if no cell uses it, the engine greys it out automatically. |
| `variants` | `{id, label, subtitle?}[]` | 2nd-dim option list. Use `default` / single-element if the model has no variant axis. |
| `quantizations` | `{id, label}[]` | 3rd-dim option list. |
| `strategies` | `{id, label}[]` | 4th-dim option list. Common ids: `low-latency`, `balanced`, `high-throughput`. |
| `nodesOptions` | `{id, label}[]` | 5th-dim option list. The `id` MUST be `single` or `multi-N` — the engine parses N from the id for `--nnodes`. |
| `cells` | `{match, verified?, env, flags}[]` | One per supported (hw × variant × quant × strategy × nodes) combination. See §2.2. |
| `modelNames` | `{[key]: string}` | HF slug lookup. Keys are either `hw\|variant\|quant` (most specific) or `variant\|quant` (fallback). |
| `placeholders` | `{[key]: {target, label, default?}}` | `{{KEY}}` interpolation map for command + curl. `target` is `'command'` or `'curl'`. Editable through the Env modal. |
| `curl` | string | cURL template. Uses `{{MODEL_NAME}}` + placeholder keys. |

**Optional fields**:

| Field | Type | Purpose |
|---|---|---|
| `multiNodeHints` | `{[hwId]: string[]}` | Lines prepended as `# ...` comments to multi-node commands (env-var hints). |
| `dockerImages` | `{[hwId]: string}` | Per-hw image name for `docker run` framing. Falls back to `lmsysorg/sglang:dev` if missing. |
| `playgroundFeatures` | `{[axisId]: {...}}` | Opts into the Playground widget. See §2.3. |
| `github` | `{owner?, repo?, issueTemplate?, cookbookModel?}` | Overrides for the "Submit verified cell" CTA in the playground. Defaults: `sgl-project/sglang` + `3-playground-verified-cell.yml` + `"deepseek-ai/deepseek-v4"`. Set `cookbookModel` to the value that matches the `model` dropdown in your issue template so it's pre-selected when the issue opens. |

### 2.2 Author the 5-dim matrix (`cells[]`)

Each cell describes one verified (or auto-estimated) launch recipe.

```js
{
  match: { hw: "b200", variant: "flash", quant: "fp4",
           strategy: "low-latency", nodes: "single" },
  verified: true,                  // green "Verified" badge; absence = yellow
  env: [
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024",
  ],
  flags: [
    "--trust-remote-code",
    "--model-path {{MODEL_NAME}}", // {{MODEL_NAME}} resolves from modelNames
    "--tp 4",
    "--moe-runner-backend flashinfer_mxfp4",
    "--host {{HOST_IP}}",
    "--port {{PORT}}",
  ],
},
```

**Rules**:

- `match` MUST contain exactly the 5 keys: `hw`, `variant`, `quant`,
  `strategy`, `nodes`. The engine looks up cells by tuple equality.
- `env` and `flags` are FLAT literals. The engine does NOT expand
  fragments, aliases, or templates — it consumes them verbatim
  (only `{{PLACEHOLDER}}` substitutions happen at render time).
- DO NOT include `--nnodes` / `--node-rank` / `--dist-init-addr` in
  `cell.flags` for multi-node cells. The renderer injects them
  automatically from `match.nodes` (`multi-N` → N nodes).
- DO NOT include `--host` / `--port` literally — use `{{HOST_IP}}` /
  `{{PORT}}` placeholders so users can override through the Env modal.
- Order flags as: `--model-path` first (after any `--trust-remote-code`),
  then parallelism (`--tp`, `--dp`, `--enable-dp-attention`), then MoE
  flags, then tuning knobs, with `--host` / `--port` last. The playground
  engine assumes this ordering when inserting overrides (its anchors target
  `--model-path` / `--tp` / etc., and inserts before the `--host` tail).

**Cells are denormalized on purpose** — common flags repeat across cells.
This makes each cell self-contained and easy to verify. When sweeping a
common change, edit every cell.

**Avoid premature cells**: only add a cell for a (hw × variant × quant ×
strategy × nodes) combination if you have a recipe that has been tested or
at least sanity-checked. The engine greys out un-listed combinations
automatically.

### 2.3 Configure `playgroundFeatures` (optional)

The Playground widget is opt-in per axis. Add only the axes that make sense
for this model. Recognised axis keys and their schemas (full reference in
[_playground.jsx](../../docs_new/src/snippets/_playground.jsx) header):

| Axis key | Widget | Use when |
|---|---|---|
| `attention` | TP / CP / DP-Attention sub-knobs (DP-Attention is a combined knob: its value is the DP degree AND toggles `--enable-dp-attention`) | Model exposes parallelism knobs in §3.1 cells and you want users to override them. |
| `moe` | Backend select + EP knob | Model is MoE and supports multiple `--moe-*-backend` choices. |
| `parsers` | Multi-toggle | Model has reasoning / tool-call parsers. |
| `speculative` | Single-select | Model has spec-decoding presets you want to expose. |
| `pdDisagg` | Mode + transfer backend (+ optional per-backend env via `envWhen` hw-gate) + IB device | Model supports prefill/decode disaggregation. |
| `hicache` | Enable + storage + write policy | Model is large enough that hierarchical KV cache matters. |
| `hisparse` | Enable + host-ratio select; whole card gated on the live PD-Disagg mode being `decode` | DSA-style model (DeepSeek-V3.2 / V4, GLM-5) that supports decode-side hierarchical sparse attention. |
| `megamoe` | Single-select with hw/strategy gating | Blackwell-only kernel fusion variant. |

**Per-chip constraints**: any chip entry in any axis can be wrapped with
`hide` / `disable` constraint objects:

```js
{ value: 16, disable: { nodes: ["single"] },
  disableReason: "TP=16 requires 16 ranks — switch §3.1's Nodes to multi-2 first." }
```

- `hide` — chip omitted entirely (use for hard impossibilities).
- `disable` — chip greyed out with tooltip (soft warning).
- Constraints are AND across keys, OR within each key's array.
- Bare `disabled: true` / `disable: true` is a static always-disabled form
  (used for "Coming soon" chips).

### 2.4 Create the MDX page

Path: `cookbook/<category>/<Vendor>/<Model>.mdx`. Import both widgets and
the per-model config, render them inside the relevant sections:

```mdx
## 3. Model Deployment

### 3.1 Basic Configuration

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/<vendor>/<model>.jsx";

<Deployment config={config} />

### 3.2 Configuration Tips

[model-specific tuning notes, caveats, links]

### 3.3 Playground

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />
```

The Playground reads §3.1's selection from the URL hash + custom event.
The heading slug `3-3-playground` is auto-generated by Mintlify from the
heading text — `_deployment.jsx` scrolls to that id when the user clicks
the "Open the Playground →" link.

### 2.5 Verify

Before committing:

1. The MDX page renders without errors in the local Mintlify preview
   (`mintlify dev` from `docs_new/`).
2. Every cell in `cells[]` produces a sensible command for its
   `(hw, variant, quant, strategy, nodes)` combo. Spot-check a few.
3. URL hash navigation works — clicking a cell updates the hash, and
   reloading the page restores the selection.
4. The Playground inherits the §3.1 selection live (no refresh needed).
5. Toggling each Playground axis produces the expected diff (green = added,
   red strikethrough = removed).
6. Docker mode (toggle in the command header) wraps in `docker run` with
   the right per-hw image.
7. Multi-node cells emit the multi-node header comment + `--nnodes N`.
8. cURL modal shows the model name resolved correctly.
9. Env modal placeholder values persist across reloads (localStorage).

---

## 3. Adding a new playground feature axis (engine work)

Rare. The current 7 built-in axes already cover the SGLang feature surface
most cookbooks need. Only add a new axis if a real cookbook needs it and
the feature does not fit any existing axis.

### 3.1 Decide

Before touching the engine, confirm:

- The feature is a STABLE part of the SGLang CLI surface (will appear in
  multiple cookbooks, not one-off).
- The feature cannot be expressed as a new option inside an existing axis
  (e.g. a new MoE backend belongs in `moe.backend.options`, not a new
  axis).
- The feature has a clean strip-prefix → emit-flag pattern.

If unsure, add it as data first (in one cookbook's config under an
existing axis) before promoting it to a built-in axis.

### 3.2 Pick the axis id and state shape

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

### 3.3 Implement the handler

Add one entry to `AXIS_HANDLERS` in [_playground.jsx](../../docs_new/src/snippets/_playground.jsx).
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
  // initState/apply use. Render shows this as the default dropdown
  // selection when the state slot is the inherit sentinel — so the user
  // sees the cell's actual --tp / MoE backend / spec preset instead of an
  // opaque "auto." When derive returns a real value, the inherit-sentinel
  // option is hidden from the dropdown. Apply also receives the derived
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
  render: ({ axisId, value, setValue, fc, base, s, renderChip, renderSelect, derived }) => {
    if (/* axis-level gating fails */) return null;
    return (
      <div key={axisId} style={s.card}>
        <div style={s.compactRow}>
          <span style={s.axisTitle}>Axis Title</span>
          {/* For multi-option fields, use renderSelect(...). For pure
              enable/disable fields, use renderChip(label, value, true,
              () => setValue(!value)). Read state from `value`; write
              via `setValue(next)` (replaces the whole axis slot). */}
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
  leading label, one `s.field` per (label + input) pair. Use
  `renderSelect(current, entries, onPick, base, labelFor?, opts?)` for
  any field with more than two options; reserve `renderChip` for pure
  on/off toggles. `renderSelect` filters hidden chips and disables
  greyed-out ones internally — no per-chip `evaluateChip` loop needed
  in the render body. Pass `{ hideValues: [<sentinel>] }` when your
  `deriveFromBase` resolved to a real value, so the inherit-sentinel
  ("auto" / "Inherited" / "current") doesn't clutter the dropdown.
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

### 3.4 Document the per-cookbook schema

Edit the file header in `_playground.jsx` to add your new axis to the
"Recognised keys" list, with a one-line description of its schema.
Optionally add a paragraph below explaining its strip/insert policy.

Update `cookbook-authoring.md` §2.3 table to list the new axis.

### 3.5 Migrate cookbooks that need it

For each cookbook that should expose this axis, add a
`playgroundFeatures.<axisId>` entry to its config. Verify the chip group
renders, options apply correctly, and the diff matches expectations.

---

## 4. Review checklist

Use this list when reviewing a new cookbook PR or a new axis PR.

**New cookbook PR**:

- [ ] Config is a single `export const config = { ... }` literal — no
      function calls, spreads, or IIFE.
- [ ] All cell `match` objects have exactly the 5 keys.
- [ ] No `--nnodes` / `--node-rank` / `--dist-init-addr` literal in any
      multi-node cell's flags (engine adds them).
- [ ] No `--host` / `--port` literal — uses `{{HOST_IP}}` / `{{PORT}}`.
- [ ] `modelNames` covers every cell (either by triple key or pair key).
- [ ] `supportedHardware` is a subset of the keys in
      `HARDWARE_CATALOG` in `_deployment.jsx`.
- [ ] `placeholders` declares every `{{KEY}}` that appears in `curl` or
      any cell.
- [ ] `dockerImages` covers the hw ids that have cells (otherwise users
      hit the `:dev` fallback).
- [ ] `multiNodeHints` covers the hw ids that have `multi-N` cells.
- [ ] The MDX page imports `Deployment` AND `Playground` from
      `/src/snippets/...` (absolute paths).
- [ ] The Playground section heading slug matches the `scrollIntoView`
      target id in `_deployment.jsx` (`3-3-playground`).

**New axis PR**:

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
- [ ] §2.3 table in this file lists the new axis.
- [ ] One existing cookbook config is updated to consume the new axis,
      and visual verification shows the diff is correct.

---

## 5. Common pitfalls

**Stale URL hash hydration** — If a user shares a link from an old cell
catalog and the hash names an impossible combination, `_deployment.jsx`'s
`validateSelection` snaps to the nearest real cell. The Playground reads
the hash too — make sure cookbook removals don't leave dangling shared
links pointing at hardware/quant combos that no longer exist.

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

**Mintlify constraints** — Module-level statements are stripped. Every
constant / helper / handler MUST live inside the wrapper function.
Capitalized JSX tags get rebound by `_provideComponents()`. Use only
lowercase HTML JSX tags (`<div>`, `<span>`, `<button>`, ...). If you need
React utilities like `Fragment`, write `<></>` shorthand (no key) or wrap
in a keyed `<div>` instead.

**Per-cell denormalization** — Cells repeat common flags on purpose. Do
not factor them into a shared `commonFlags` array — Mintlify will fail
to inline the reference. If you need to sweep a flag across cells, do it
with a global find-replace in the config file.
