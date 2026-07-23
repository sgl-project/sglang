# Cookbook config reference (fields · cells · playground · MDX)

Loaded on demand by the `cookbook-add-model` skill. This is the field-by-field
contract for when the clone needs more than a rename. The two engine files are
the canonical specs — read their headers first:

- [`_deployment.jsx`](../../../../docs/src/snippets/_deployment.jsx) — the 5-dim matrix widget; lists every config field.
- [`_playground.jsx`](../../../../docs/src/snippets/_playground.jsx) — the diff-based override widget; lists the `playgroundFeatures` axes + the `AXIS_HANDLERS` interface.

Engine extension (adding a new playground axis) lives in [engine-axis.md](engine-axis.md).

---

## 2.1 Create the config file

**Path**: `docs/src/snippets/configs/<vendor>/<model>.jsx`. The vendor folder is
the HuggingFace org (`deepseek-ai`, `Qwen`, `moonshotai`, ...); the file
name is a short hyphenated model id (`deepseek-v4`, `qwen3.5`, ...).

**Shape**: must be a single `export const config = { ... }` literal. Do not
use function calls, spreads, fragment refs, or IIFE — Mintlify re-evaluates
this export at hydration time with module-level identifiers out of scope,
and any non-literal value crashes with `ReferenceError`.

**Required fields** (engine reads these — see the `_deployment.jsx` header for
the full contract):

| Field | Type | Purpose |
|---|---|---|
| `modelName` | string | Display label only. Not used for HF slug — see `modelNames`. |
| `supportedHardware` | `string[]` | Which hw ids appear in the catalog. Subset of `HARDWARE_CATALOG` (in `_deployment.jsx`) ∪ `config.hardware`. Listing an id makes its button appear; if no cell uses it, the engine greys it out automatically. |
| `hardware` | `{id,label,vram,vendor}[]` | Optional. GPUs the shared `HARDWARE_CATALOG` doesn't carry (workstation / desktop / future chips, e.g. RTX PRO 6000). The engine merges these into the catalog, so a model-specific GPU is config data — **no engine-catalog edit**. Also add the id to `supportedHardware`. |
| `variants` | `{id, label, subtitle?}[]` | 2nd-dim option list. Use `default` / single-element if the model has no variant axis. |
| `quantizations` | `{id, label}[]` | 3rd-dim option list. |
| `strategies` | `{id, label}[]` | 4th-dim option list. Canonical ids: `low-latency` / `balanced` / `high-throughput` (never model-specific ids like `mtp`). **The count follows the page's operating points**: one recipe → a single `balanced`; two → `low-latency` + `high-throughput`; three → the full trio (the ideal). Tiers apply per (hw × variant × quant) combination — a single-recipe combination parks under its semantically honest tier (clear slant → that tier, e.g. DSv4's RTX 6000 → `low-latency`; no slant → `balanced`, e.g. Qwen3.5's Xeon); the page's list is the union and the engine greys unused chips per selection. Never invent a recipe just to fill chips. When two recipes differ by MTP / speculative decoding, the assignment is deterministic: spec ON → `low-latency`, spec OFF → `high-throughput` (at saturation the draft+verify overhead outweighs the speedup — same reason DSv4's high-throughput recipes disable MTP). The recurring markers in the other direction: dp-attention ON (MLA-attention models) and EP / DP+EP ON (MoE models) → `high-throughput`. |
| `nodesOptions` | `{id, label}[]` | 5th-dim option list. The `id` MUST be `single` or `multi-N` — the engine parses N from the id for `--nnodes`. |
| `cells` | `{match, verified?, env, flags}[]` | One per supported (hw × variant × quant × strategy × nodes) combination. See §2.2. |
| `modelNames` | `{[key]: string}` | HF slug lookup. Keys are either `hw\|variant\|quant` (most specific) or `variant\|quant` (fallback). |
| `placeholders` | `{[key]: {target, label, default?}}` | `{{KEY}}` interpolation map for command + curl. `target` is `'command'` or `'curl'`. Editable through the Env modal. |
| `curl` | string | cURL template. Uses `{{MODEL_NAME}}` + placeholder keys. |

**Optional fields**:

| Field | Type | Purpose |
|---|---|---|
| `multiNodeHints` | `{[hwId]: string[]}` | Lines prepended as `# ...` comments to multi-node commands (env-var hints). Per-hw, and only for hw whose **cluster fabric needs manual NIC config** (e.g. `gb200` NVL72/MNNVL → NVSHMEM/Gloo hints). NOT every multi-N hw needs an entry — standard-IB DeepEP (h200) auto-detects the HCA, and Marlin multi-node (h100) uses no DeepEP/NVSHMEM at all. |
| `dockerImages` | `{[key]: string}` | Image for `docker run` framing, keyed by `hw\|quant` (most specific) then `hw`. Use a `hw\|quant` key only when one quant on a shared GPU needs a different image (e.g. an NVFP4 dev build on b300/gb300 while FP8/BF16 stay on the release image); otherwise key by plain `hw`. **Ask the user which sglang build the recipes ran on; don't guess a supporting release.** Falls back to `lmsysorg/sglang:dev` if missing — also the sensible default when unsure. |
| `playgroundFeatures` | `{[axisId]: {...}}` | Opts into the Playground widget. See §2.3. |
| `benchmarkCommands` | `{speed: string, accuracy: {[accKey]: string \| {[variant]: string}}, numPromptsByConc?: {[c]: number}}` | Powers the benchmark card's **"⚡ Reproduce"** modal. `speed` is ONE `bench_serving` template; the engine fills `{{DATASET}}`/`{{ISL}}`/`{{OSL}}` from each cell's `speed[].workload`, the chip-picked `{{MAX_CONCURRENCY}}`, and `{{NUM_PROMPTS}}` (resolved `workload.num_prompts ?? numPromptsByConc[c] ?? max(c*2, 200)`). `accuracy` maps an accuracy field (e.g. `gsm8k_pct`) to a per-eval template — a string, OR a `{flash, pro, …}` object keyed by variant when the command differs per variant (e.g. GPQA/AIME `--max-tokens`). The modal renders a chip per eval (one command area, like Speed). Both also use `{{MODEL_NAME}}` + `{{CURL_HOST}}`/`{{CURL_PORT}}` like `curl`. `speed` should carry `--flush-cache` (bench_serving's `random` prompts are deterministic — warm reruns hit the radix cache and inflate throughput; measure cache-cold). Optional; the button only appears when this AND `benchmarks` are present. |
| `defaultAccuracy` | `{[variant]: {[accKey]: number}}` | Model-level accuracy applied to **every** cell of a variant (e.g. GPQA Diamond / AIME25 — hardware-independent). Merged UNDER each cell's measured `accuracy` (a per-cell value wins), so you set a variant's score once instead of copying it onto every benchmark entry. Keys must match `accuracyLabels` (below) + `benchmarkCommands.accuracy`. |
| `accuracyLabels` | `[key, label, unit][]` | The eval set rendered in the benchmark card and the "⚡ Reproduce" modal — **the engine ships no default**, every config declares its own (e.g. DSv4: GPQA/AIME25/GSM8K; Qwen3.5: GSM8K/MMMU). Required whenever the benchmarks carry accuracy data; without it the accuracy rows silently don't render. Every key used in `benchmarks[].accuracy`, `defaultAccuracy`, and `benchmarkCommands.accuracy` must appear here. |
| `latencyPercentile` | `"Mean" \| "P50"` | Optional, **temporary**; the percentile the benchmark TTFT/TPOT values are. **Default `"P50"`** — the card renders `TTFT (<pct>)` / `TPOT (<pct>)`. Set `"Mean"` only for legacy data recorded as Mean (being re-measured to P50). A benchmarks entry may carry its own `latencyPercentile` to override the page value per cell (entry → config → `"P50"`). `tokens_per_sec_per_gpu` is stored as **total (in+out)/GPU** = `output tok/s/GPU × (isl+osl)/osl`, shown by the card as-is. |
| `github` | `{owner?, repo?, issueTemplate?, cookbookModel?}` | Overrides for the "Submit verified cell" CTA in the playground. Defaults: `sgl-project/sglang` + `3-playground-verified-cell.yml` + `"deepseek-ai/deepseek-v4"`. Set `cookbookModel` to the model's HF id (`<hf-org>/<model-slug>`); it prefills the issue template's free-form `model` input when the issue opens. **Don't prune this block** — without it the engine falls back to `deepseek-ai/deepseek-v4` and submissions from your page get mislabeled. |

## 2.2 Author the 5-dim matrix (`cells[]`)

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
- Accuracy-degrading flags don't belong in cells by default: a cell's
  output quality should be exactly what its quantization chip declares.
  Runtime quant below the checkpoint's precision (e.g. MegaMoE **W4A4** —
  DSv4 gates it behind the Playground's `megamoeQuant` opt-in) and lossy
  KV-cache dtypes (`--kv-cache-dtype fp8_e4m3` over a higher-precision-KV
  checkpoint) default to Playground opt-ins or §2-tips material. If the
  model's recipe genuinely needs one in a cell, **flag it to the user and
  get explicit confirmation** — never ship it silently. (Migrations are the
  sanctioned exception: a flag baked into the legacy recipe's default
  command keeps verbatim — see the migrate skill.)

**Cells are denormalized on purpose** — common flags repeat across cells.
This makes each cell self-contained and easy to verify. When sweeping a
common change, edit every cell.

**Avoid premature cells**: only add a cell for a (hw × variant × quant ×
strategy × nodes) combination if you have a recipe that has been tested or
at least sanity-checked. The engine greys out un-listed combinations
automatically.

## 2.3 Configure `playgroundFeatures` (optional)

The Playground is **opt-out, not opt-in**: every cookbook ships the general
axes by default — `attention` (TP/CP/DP-Attn), `moe` (backend + EP, for MoE
models), `parsers`, `speculative`, `pdDisagg`, `hicache` — then adds
model-specific axes (e.g. MegaMoE for DeepSeek-V4) and deletes ONLY the axes
this model genuinely cannot use (e.g. `hisparse` on non-DSA models, `moe` on a
pure-dense model). Knobs that don't apply to a subset of variants/hw get
`disable` + `disableReason`, not removal. Recognised axis keys and their
schemas (full reference in the `_playground.jsx` header):

| Axis key | Widget | Use when |
|---|---|---|
| `attention` | TP / CP / DP-Attention sub-knobs (DP-Attention is a combined knob: its value is the DP degree AND toggles `--enable-dp-attention`) | Model exposes parallelism knobs in its cells (§2.2) and you want users to override them. |
| `moe` | Backend select (incl. MegaMoE) + EP knob; picking the MegaMoE backend reveals a Quantization sub-select (W4A8/W4A4) | Model is MoE and supports multiple `--moe-*-backend` choices. For Blackwell MoE kernel-fusion, give the `megamoe` backend option a `requiresHw` (and optional `excludesStrategy`) gate, then add a sibling `megamoeQuant` block (`{stripEnv, options}`): W4A8 = `NUM_MAX` only, W4A4 adds the FP4-activations env vars; both strip the DeepEP dispatch env. |
| `parsers` | Multi-toggle | Model has reasoning / tool-call parsers. |
| `speculative` | Single-select chip group | Model has spec-decoding presets you want to expose. |
| `pdDisagg` | Mode + transfer backend (+ optional per-backend env via `envWhen` hw-gate) + IB device + optional `router{port, command}` | Model supports prefill/decode disaggregation. When a PD role is active and `router` is set, the playground shows the router (SGLang Model Gateway) launch command as a separate companion block and retargets the cURL modal to `router.port` (clients hit the router, not the role servers). |
| `hicache` | Enable + storage + write policy | Model is large enough that hierarchical KV cache matters. |
| `hisparse` | Enable + host-ratio select; whole card gated on the live PD-Disagg mode being `decode` | DSA-style model (DeepSeek-V3.2 / V4, GLM-5) that supports decode-side hierarchical sparse attention. |
| `flagSelects` | A config-declared **list** of single-selects, each `{ id, title, stripPrefixes, options }` (option = `{ id, label, flags?, hide?, disable?, disableReason? }`); a flagless option is the "none"/accuracy-safe choice | A titled single-select that picks one value of a flag family the other axes don't model — e.g. KV-cache dtype (`--kv-cache-dtype`), mamba scheduler strategy (`--mamba-scheduler-strategy`). Generic: no engine change to add another. |

**Per-chip constraints**: any chip entry in any axis can be wrapped with
`hide` / `disable` constraint objects:

```js
{ value: 16, disable: { nodes: ["single"] },
  disableReason: "TP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." }
```

- `hide` — chip omitted entirely (use for hard impossibilities).
- `disable` — chip greyed out with tooltip (soft warning).
- Constraints are AND across keys, OR within each key's array.
- Bare `disabled: true` / `disable: true` is a static always-disabled form
  (used for "Coming soon" chips).
- `disable` may also be an ARRAY of `{when: constraint, reason}` items (OR
  across items, first match wins and supplies its own tooltip) — for
  conditions that need OR across keys or per-condition reasons, e.g. the
  GLM-5.2 CP knob (grayed on non-Hopper hw OR multi-node, different reasons).
- Constraint keys are the 5 cell dims (`hw`/`variant`/`quant`/`strategy`/
  `nodes`) plus cross-axis live facts: `dpAttnOn` (effective DP-Attention on),
  `cpOn` (effective prefill-CP on), `cpStrategy` (effective CP layout),
  `cpSizeTarget` (the only enable-able CP size — see below), `effTp`
  (effective TP degree — override else derived), and `pdMode` (live
  PD-Disagg role).
- On the `attention` axis, knob-level `hide`/`disable` (on the knob object,
  not a value entry) hides/grays the whole select; apply() skips
  knobs/values that are disabled under the live facts, so stale picks never
  emit a blocked combination. Interleave prefill-CP + DP-Attention is
  deliberately NOT grayed (combined support is planned upstream even though
  current releases assert `dp_size == 1` for interleave) — the engine shows
  a warning hint below the command box instead.
- The CP knob emits `--attn-cp-size N --enable-prefill-cp --cp-strategy S`,
  where S is: an optional `{ id: "cpStrategy", values: [null, "interleave",
  "zigzag"] }` knob's pick > the strategy already baked in the base cell
  (legacy mode flags map in: in-seq-split → zigzag, round-robin-split →
  interleave) > "interleave". Declare the cpStrategy knob only on models
  whose runtime accepts both layouts (DeepSeek-V4 rejects zigzag; DSA
  zigzag forces deepep + ep=tp + batch_size=1 — label it experimental).
- CP sizes auto-gate in the engine to the runtime derivation
  `attn_cp_size = tp/dp` (both in the grayed options and in apply), so
  configs list plain size values — no per-value `effTp` constraints needed.
  A model whose runtime honors arbitrary `--attn-cp-size` opts out with
  `freeSize: true` on the cp knob.
- Only expose a CP knob on models with model-side CP integration in SGLang
  (DeepSeek-family / Qwen-MoE / Mellum); on others the emitted flags do
  nothing or crash (that's why Hy3 and MiniMax-M3 have no CP knob).

## 2.4 Create the MDX page

Path: `docs/cookbook/<category>/<Vendor>/<Model>.mdx`. Import both widgets and
the per-model config, render them inside the relevant sections:

```mdx
## Deployment

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/<vendor>/<model>.jsx";
import { benchmarks } from "/src/snippets/configs/<vendor>/<model>-benchmarks.jsx";

{/* Install is a PREREQUISITE — keep it compact + collapsed at the top of the
    Deploy section (NOT a numbered section). Tabs mirror the widget's
    Python/Docker toggle. */}
<a id="install" />
<Accordion title="Install SGLang">
  <Tabs>
    <Tab title="Python (pip / uv)">…pip / uv install…</Tab>
    <Tab title="Docker">…docker pull + a `docker run … sglang serve` example…</Tab>
  </Tabs>
</Accordion>

<Deployment config={config} benchmarks={benchmarks} />

[model-specific tuning notes, caveats, links]

## Playground

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />
```

**Heading slugs matter** — the two widgets cross-link by scrolling to each
other's section id (Mintlify auto-slugs headings: lowercase, spaces →
hyphens, punctuation dropped). The engines look up:

- the **Deploy** panel by id `deployment` (falls back to `deploy`) — used
  by the Playground's "↑ Switch base" button and by deep-link scroll-on-
  load. Title the section `## Deployment` (or `## Deploy`).
- the **Playground** by id `playground` — used by `_deployment.jsx`'s
  "Open the Playground →" link. Title the section `## Playground`.

Avoid numbered headings like `## 3. Model Deployment` (slug
`3-model-deployment`) for these two sections — the cross-links would break.
The Playground reads the Deploy selection live via the URL hash + the
`sglang-deploy-sel` custom event, so the two can live in different parts
of the page.

The `benchmarks` prop is **optional**. It points at a sibling
`<model>-benchmarks.jsx` file (one entry per cell, keyed by the same
`match` tuple) that renders an accuracy + speed sub-card under the command
box; omit the import and the prop if the cookbook has no measured numbers
yet. See the `_deployment.jsx` header and `deepseek-v4-benchmarks.jsx` for
the full speed/accuracy schema.

Each entry's `sglang_version` must be a **reproducible anchor** — a release
tag/version (`v0.5.9`), a commit hash, or (for **Day-0 support**, before the
enabling PR merges or a release is cut) a specific PR (`PR #27944`) or commit
you can `gh pr checkout` / `git checkout`. Never a moving ref like `"main"` /
`"main (2026-06-11)"` (not reproducible). A spec-decoding model whose cell
carries `--speculative-algorithm` but no `--max-running-requests` auto-shows
an amber Deploy + Playground callout (SGLang otherwise caps it at 48) — it is
flag-driven, so no per-page prose is needed.

To let users *reproduce* those numbers, add a `benchmarkCommands` block to
the config (§2.1, next to `curl`). When present alongside `benchmarks`, the
benchmark card grows a **"⚡ Reproduce"** button that opens a modal listing
the runnable commands for the current cell — one `bench_serving` command for
Speed (with concurrency chips that rewrite `--max-concurrency`) plus an
Accuracy command with a chip per eval. No separate benchmark section needed.

---

## Pitfalls (authoring)

**Stale URL hash hydration** — If a user shares a link from an old cell
catalog and the hash names an impossible combination, `_deployment.jsx`'s
`validateSelection` snaps to the nearest real cell. The Playground reads
the hash too — make sure cookbook removals don't leave dangling shared
links pointing at hardware/quant combos that no longer exist.

**Mintlify constraints** — Module-level statements are stripped. The config
MUST be a single `export const config = { ... }` literal — no function calls,
spreads, fragment refs, or IIFE (Mintlify re-evaluates the export at hydration
with module-level identifiers out of scope; any non-literal crashes with
`ReferenceError`). In MDX, capitalized JSX tags get rebound — use the built-in
Mintlify components (`<Accordion>`, `<Tabs>`, `<Card>`, ...) as documented.
Avoid `!(x in y)` anywhere (Mintlify's AST walker crashes on it) — use
`obj.key === undefined`.

**Per-cell denormalization** — Cells repeat common flags on purpose. Do
not factor them into a shared `commonFlags` array — Mintlify will fail
to inline the reference. If you need to sweep a flag across cells, do it
with a global find-replace in the config file.
